"""CONUS-scale search for the recording location and start time.

Passes (each one logged with in/out counts and wall time):

  load        ADS-B rows for the time chunk (+ padding), clipped to the region bbox
  altitude    drop rows far too high to be any detected flyover (alt > max_d + terrain max)
  resample    1 s grid per track segment, ECEF position + velocity
  candidates  per event: track samples whose speed matches v and whose height above
              ground is <= d; each yields the 1-2 ground points at slant range d
              perpendicular to the track, plus an implied recording start T0
  region      candidate ground points outside the search region are dropped
  join        (>= 2 events) candidates of different events that agree in position and T0
  dedupe      one representative per (flight set, T0 bucket)
  silence     penalise points where some other low aircraft passed closer than any
              detected flyover during the recording yet was not heard
  prescore    full Doppler-curve misfit at the representative (quick T0 scan)
  refine      Nelder-Mead over (east, north, T0) on the best prescored candidates
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .audio import Event
from .geo import LocalFrame, ecef_to_geodetic
from .physics import received_ratio, speed_of_sound
from .regions import Region, haversine_m
from .timing import PassLog
from .tracks import Resampled, resample

MAX_TERRAIN_M = 4400.0
UNMATCHED_COST = 0.5  # misfit charged for a flyover no ADS-B flight explains
REF_HZ = 100.0  # misfit residuals are expressed relative to the tone: Hz at a 100 Hz tone
PRESCORE_GRID = [(de, dn) for de in (-500.0, 0.0, 500.0) for dn in (-500.0, 0.0, 500.0)]


@dataclass
class SearchConfig:
    temp_c: float = 15.0
    chunk_s: float = 3600.0
    dt: float = 2.0
    tol_v_frac: float = 0.10
    tol_v_min: float = 6.0
    tol_d_frac: float = 0.25
    tol_d_min: float = 150.0
    tol_t_min: float = 8.0  # real ADS-B timestamps carry a few seconds of error
    track_dt_sigma: float = 2.0  # prior on per-flight ADS-B time error (s) in refinement
    track_dt_weight: float = 0.1
    join_radius_m: float = 1000.0
    min_events: int = 2  # events that must be matched to an ADS-B flight (if >= 2 detected)
    silence_frac: float = 0.6  # silence radius = this * min detected d (0 disables)
    silence_weight: float = 3.0
    unheard_cost: float = 0.3  # per aircraft that should have been heard in a quiet stretch but wasn't
    audible_frac: float = 0.8  # audible radius is at least this x the heard flyovers' distance ...
    audible_max: float = 3.0  # ... and at most this x
    audible_margin_z: float = 2.0  # loudness safety margin (whitened z units) in the calibration
    detect_z: float = 3.0  # the detector's ridge threshold
    unheard_pad_s: float = 8.0  # the stretch around the predicted pass that must be quiet
    max_reps: int = 4000
    prescore_top: int = 3000
    keep_per_chunk: int = 300
    polish_top: int = 1500
    refine_top: int = 40
    chunk_budget: float = 0.6  # fraction of the timeout for the chunked passes; the rest refines
    explain_rms_hz: float = 1.0
    join_bonus: float = 0.3
    max_anchors: int = 3
    min_prescore: int = 1500
    workers: int = 4  # parallel time chunks (processes)  # minimum work per stage even past the deadline
    min_polish: int = 40
    min_refine: int = 8
    max_join_d: float = 5000.0
    timeout_s: float = 60.0
    isolate_margin: float = 0.5  # runner-up (elsewhere) must cost this much more
    isolate_km: float = 1.5
    good_rms_hz: float = 1.0


@dataclass
class Solution:
    lat: float
    lon: float
    T0: float
    cost: float
    rms_hz: list
    f0: list
    flights: list
    silence_hits: int = 0
    unheard: list = field(default_factory=list)  # aircraft that should have been heard but weren't
    stage: str = "refined"
    state: str | None = None

    def to_dict(self):
        from datetime import datetime, timezone

        d = dict(self.__dict__)
        d["start_iso"] = datetime.fromtimestamp(self.T0, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return d


@dataclass
class SearchResult:
    status: str  # "solved" | "ambiguous" | "none"
    best: Solution | None
    candidates: list
    log: PassLog
    timed_out: bool = False
    searched_s: float = 0.0
    window_s: float = 0.0
    notes: list = field(default_factory=list)

    def to_dict(self):
        return dict(status=self.status, best=self.best.to_dict() if self.best else None,
                    candidates=[c.to_dict() for c in self.candidates], timed_out=self.timed_out,
                    searched_fraction=self.searched_s / self.window_s if self.window_s else 1.0,
                    passes=self.log.as_dict(), notes=self.notes)


def _tols(ev: Event, cfg: SearchConfig):
    f = ev.fit
    se = f.stderr or {}
    tol_d = max(cfg.tol_d_frac * f.d, 3 * (se.get("d") or 0), cfg.tol_d_min)
    tol_v = max(cfg.tol_v_frac * f.v, 3 * (se.get("v") or 0), cfg.tol_v_min)
    tol_t = max(cfg.tol_t_min, 3 * (se.get("t_ca") or 0))
    return tol_d, tol_v, tol_t


def _up(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


@dataclass
class EventCands:
    seg: np.ndarray
    te: np.ndarray
    xyz: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    T0: np.ndarray
    cost: np.ndarray
    sigma: np.ndarray  # positional uncertainty (m)


def event_candidates(ev: Event, rs: Resampled, elev_under: np.ndarray, region: Region, T0_lo, T0_hi,
                     c: float, cfg: SearchConfig, log: PassLog | None = None, k: int = 0) -> EventCands:
    f = ev.fit
    tol_d, tol_v, _ = _tols(ev, cfg)
    log = log or PassLog()
    with log.timed(f"candidates[{k}]", n_in=rs.t.size) as h:
        speed = rs.speed if hasattr(rs, "speed") else np.linalg.norm(rs.vel, axis=1)
        T0 = rs.t + f.d / c - f.t_ca
        m = (np.abs(speed - f.v) < tol_v) & (rs.alt < f.d + tol_d + MAX_TERRAIN_M) & (T0 >= T0_lo) & (T0 < T0_hi)
        idx = np.nonzero(m)[0]
        hgt_i = rs.alt[idx] - (elev_under[idx] if not callable(elev_under) else elev_under(rs.lat[idx], rs.lon[idx]))
        ok_h = (hgt_i > -50) & (hgt_i < f.d + tol_d)
        idx, hgt_i = idx[ok_h], hgt_i[ok_h]
        up = _up(rs.lat[idx], rs.lon[idx])
        vel = rs.vel[idx]
        vh = vel - np.sum(vel * up, 1)[:, None] * up
        vh /= np.maximum(np.linalg.norm(vh, axis=1), 1e-6)[:, None]
        nrm = np.cross(up, vh)
        hh = np.clip(hgt_i, 0, None)
        s = np.sqrt(np.clip(f.d**2 - hh**2, 0, None))
        ground = rs.xyz[idx] - hh[:, None] * up
        cost = ((speed[idx] - f.v) / tol_v) ** 2 + np.where(hh > f.d, ((hh - f.d) / tol_d) ** 2, 0.0)
        sigma = np.maximum(150.0, tol_d * f.d / np.maximum(s, 0.35 * f.d))
        two = s > 20
        xyz = np.concatenate([ground + s[:, None] * nrm, (ground - s[:, None] * nrm)[two]])
        rep = lambda a: np.concatenate([a, a[two]])
        seg, te, T0c, cst, sg = rep(rs.seg[idx]), rep(rs.t[idx]), rep(T0[idx]), rep(cost), rep(sigma)
        h["out"] = len(seg)
    with log.timed(f"region[{k}]", n_in=len(seg)) as h:
        lat, lon, _ = ecef_to_geodetic(xyz) if len(xyz) else (np.zeros(0), np.zeros(0), None)
        ok = region.contains(lat, lon) if len(xyz) else np.zeros(0, bool)
        h["out"] = int(ok.sum())
    return EventCands(seg[ok], te[ok], xyz[ok], lat[ok], lon[ok], T0c[ok], cst[ok], sg[ok])


@dataclass
class Joint:
    xyz: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    T0: np.ndarray
    segs: np.ndarray  # (n, K) segment per event, -1 if unmatched
    cost: np.ndarray


def join_events(cands: list[EventCands], events: list[Event], cfg: SearchConfig, log: PassLog) -> Joint:
    """Candidates of different flyovers that agree in position and implied start time.

    Anchors are the most selective flyovers (fewest candidates, up to cfg.max_anchors);
    every other flyover is matched to each anchor point through a KD-tree in
    (ECEF position, scaled T0). Flyovers heard from far away (d > cfg.max_join_d) carry
    little positional information and huge candidate sets, so they are left out of the join
    (refinement can still explain them)."""
    K = len(cands)
    if K == 1:
        e = cands[0]
        return Joint(e.xyz, e.lat, e.lon, e.T0, e.seg[:, None], e.cost)
    tt = [_tols(ev, cfg)[2] for ev in events]
    usable = [k for k in range(K) if events[k].fit.d <= cfg.max_join_d and len(cands[k].T0)]
    if len(usable) < 2:
        usable = [k for k in range(K) if len(cands[k].T0)]
    order = sorted(usable, key=lambda i: len(cands[i].T0))
    need = min(cfg.min_events, len(usable))
    anchors = order[: max(1, min(cfg.max_anchors, len(order) - need + 1))]
    n_in = sum(len(c.T0) for c in cands)
    tol_ref = max(tt)
    scale = cfg.join_radius_m / (2 * tol_ref)
    trees = {}
    with log.timed("join", n_in=n_in) as h:
        parts = []
        for a in anchors:
            A = cands[a]
            n = len(A.T0)
            segs = np.full((n, K), -1)
            segs[:, a] = A.seg
            cost = A.cost.copy() + 4.0 * (K - len(usable))
            matched = np.ones(n, int)
            xyz_sum = A.xyz.copy()
            T0_sum = A.T0.copy()
            q = np.column_stack([A.xyz, A.T0 * scale])
            for b in usable:
                if b == a:
                    continue
                B = cands[b]
                if b not in trees:
                    trees[b] = cKDTree(np.column_stack([B.xyz, B.T0 * scale]))
                tol_t = np.hypot(tt[a], tt[b])
                dist, j = trees[b].query(q, k=4, distance_upper_bound=cfg.join_radius_m * 1.5)
                valid = np.isfinite(dist)
                jj = np.where(valid, j, 0)
                dpos = np.linalg.norm(B.xyz[jj] - A.xyz[:, None, :], axis=2)
                sig = np.hypot(B.sigma[jj], A.sigma[:, None])
                dT = (B.T0[jj] - A.T0[:, None]) / tol_t
                c2 = np.where(valid, B.cost[jj] + (dpos / sig) ** 2 + dT**2, np.inf)
                kbest = np.argmin(c2, axis=1)
                cb = c2[np.arange(n), kbest]
                hit = np.isfinite(cb)
                jb = jj[np.arange(n), kbest]
                segs[hit, b] = B.seg[jb][hit]
                cost += np.where(hit, cb, 4.0)
                matched += hit
                xyz_sum[hit] += B.xyz[jb][hit]
                T0_sum[hit] += B.T0[jb][hit]
            ok = matched >= need
            # position / start time: mean over the matched flyovers' candidates (rough intersection)
            xyz = xyz_sum[ok] / matched[ok, None]
            lat, lon, _ = ecef_to_geodetic(xyz) if ok.any() else (np.zeros(0), np.zeros(0), None)
            parts.append(Joint(xyz, lat, lon, T0_sum[ok] / matched[ok], segs[ok], cost[ok]))
        if parts:
            best = Joint(*(np.concatenate([getattr(p_, f) for p_ in parts]) for f in
                           ("xyz", "lat", "lon", "T0", "segs", "cost")))
        else:
            best = Joint(np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, K), int), np.zeros(0))
        h["out"] = len(best.T0)
    return best


def dedupe(J: Joint, cfg: SearchConfig, log: PassLog, t_bucket=20.0) -> Joint:
    with log.timed("dedupe", n_in=len(J.T0)) as h:
        if len(J.T0) == 0:
            h["out"] = 0
            return J
        o = np.argsort(J.cost)
        key = np.column_stack([J.segs[o], np.floor(J.T0[o] / t_bucket).astype(np.int64),
                               np.round(J.lat[o] * 50).astype(np.int64), np.round(J.lon[o] * 50).astype(np.int64)])
        _, first = np.unique(key, axis=0, return_index=True)
        sel = o[np.sort(first)][: cfg.max_reps]
        sel = sel[np.argsort(J.cost[sel])]
        h["out"] = len(sel)
        return Joint(J.xyz[sel], J.lat[sel], J.lon[sel], J.T0[sel], J.segs[sel], J.cost[sel])


def silence(J: Joint, rs: Resampled, events: list[Event], dur: float, c: float, cfg: SearchConfig,
            log: PassLog, elev_fn, low_rows=None) -> np.ndarray:
    hits = np.zeros(len(J.T0), int)
    if cfg.silence_frac <= 0 or len(J.T0) == 0 or rs.t.size == 0:
        return hits
    R = cfg.silence_frac * min(ev.fit.d for ev in events)
    with log.timed("silence", n_in=len(J.T0)) as h:
        if low_rows is None:
            low = np.nonzero(rs.alt - elev_fn(rs.lat, rs.lon) < R + 100)[0]
        else:
            low = low_rows[rs.alt[low_rows] - elev_fn(rs.lat[low_rows], rs.lon[low_rows]) < R + 100]
        if low.size:
            tree = cKDTree(rs.xyz[low])
            lists = tree.query_ball_point(J.xyz, R)
            for i, rows in enumerate(lists):
                if not rows:
                    continue
                rows = low[np.asarray(rows)]
                r = np.linalg.norm(rs.xyz[rows] - J.xyz[i], axis=1)
                trx = rs.t[rows] + r / c
                inwin = (trx >= J.T0[i]) & (trx <= J.T0[i] + dur)
                segs = set(rs.seg[rows[inwin]].tolist()) - set(J.segs[i].tolist())
                hits[i] = len(segs)
        h["out"] = int((hits == 0).sum())
        h["note"] = f"radius {R:.0f} m"
    return hits


# --------------------------------------------------------------------- curve misfit
def _local_tracks(rs: Resampled, segs, frame: LocalFrame):
    out = []
    for s in segs:
        if s < 0:
            out.append(None)
            continue
        sl = rs.rows(int(s))
        out.append((rs.t[sl], (rs.xyz[sl] - frame.origin) @ frame.R.T))
    return out


def curve_misfit(events, trs, obs, T0, c, huber=1.0, shifts=None):
    total, rms, f0s = 0.0, [], []
    used = 0
    for k, (ev, tr) in enumerate(zip(events, trs)):
        if tr is None:
            rms.append(float("nan"))
            f0s.append(float("nan"))
            continue
        t, p = tr
        t_abs = T0 + ev.t + (shifts[k] if shifts is not None else 0.0)
        m = (t >= t_abs[0] - 60) & (t <= t_abs[-1] + 5)
        if m.sum() < 5:
            rms.append(float("nan"))
            f0s.append(float("nan"))
            continue
        t_r, _, ratio = received_ratio(t[m], p[m], obs, c)
        g = np.interp(t_abs, t_r, ratio, left=np.nan, right=np.nan)
        ok = np.isfinite(g)
        if ok.mean() < 0.8:  # track does not cover this flyover: treat as unmatched
            rms.append(float("nan"))
            f0s.append(float("nan"))
            continue
        w, f, g = ev.w[ok], ev.f[ok], g[ok]
        f0 = np.sum(w * f * g) / np.sum(w * g * g)
        res = f - f0 * g
        a = np.abs(res) * (REF_HZ / max(f0, 1.0))  # scale-free: residual as if the tone were REF_HZ
        hub = np.where(a < huber, 0.5 * a * a, huber * (a - 0.5 * huber))
        total += min(np.sum(w * hub) / np.sum(w), UNMATCHED_COST)  # a bad match never beats no match
        rms.append(float(np.sqrt(np.average(res**2, weights=w))) * REF_HZ / max(f0, 1.0))
        f0s.append(float(f0))
        used += 1
    if used == 0:
        return np.inf, [], []
    return total + UNMATCHED_COST * (len(events) - used), rms, f0s


def _event_geoms(events, trs, obs, c):
    """Per event: (t_r, ratio) of the assigned track as heard at obs, or None."""
    out = []
    for ev, tr in zip(events, trs):
        if tr is None:
            out.append(None)
            continue
        t, p = tr
        t_r, _, ratio = received_ratio(t, p, obs, c)
        out.append((t_r, ratio))
    return out


def _misfit_geoms(events, geoms, T0, huber=1.0):
    """Fast misfit when only T0 changes (geometry cached)."""
    total, used = 0.0, 0
    for ev, gm in zip(events, geoms):
        if gm is None:
            continue
        g = np.interp(T0 + ev.t, gm[0], gm[1], left=np.nan, right=np.nan)
        ok = np.isfinite(g)
        if ok.mean() < 0.8:
            continue
        w, f, g = ev.w[ok], ev.f[ok], g[ok]
        f0 = np.sum(w * f * g) / np.sum(w * g * g)
        a = np.abs(f - f0 * g) * (REF_HZ / max(f0, 1.0))
        total += min(np.sum(w * np.where(a < huber, 0.5 * a * a, huber * (a - 0.5 * huber))) / np.sum(w),
                     UNMATCHED_COST)
        used += 1
    return total + UNMATCHED_COST * (len(events) - used) if used else np.inf


def _misfit_scan(events, geoms, T0s, huber=1.0):
    """Vectorised _misfit_geoms over many start times at once. Returns (len(T0s),) costs."""
    T0s = np.asarray(T0s, float)
    total = np.zeros(T0s.size)
    used = np.zeros(T0s.size)
    for ev, gm in zip(events, geoms):
        if gm is None:
            continue
        g = np.interp((T0s[:, None] + ev.t[None, :]).ravel(), gm[0], gm[1], left=np.nan,
                      right=np.nan).reshape(T0s.size, -1)
        ok = np.isfinite(g)
        cover = ok.mean(axis=1) >= 0.8
        g = np.where(ok, g, 0.0)
        w = ev.w[None, :] * ok
        f0 = np.sum(w * ev.f * g, 1) / np.maximum(np.sum(w * g * g, 1), 1e-12)
        a = np.abs(ev.f[None, :] - f0[:, None] * g) * (REF_HZ / np.maximum(f0, 1.0))[:, None]
        hub = np.where(a < huber, 0.5 * a * a, huber * (a - 0.5 * huber))
        cst = np.minimum(np.sum(w * hub, 1) / np.maximum(np.sum(w, 1), 1e-12), UNMATCHED_COST)
        total += np.where(cover, cst, 0.0)
        used += cover
    return np.where(used > 0, total + UNMATCHED_COST * (len(events) - used), np.inf)


def decimate_events(events, step_s=1.0):
    """Bin-averaged copies of the flyover curves (same shape, ~10x fewer points) for the
    cheap ranking passes; full-resolution events are kept for final refinement."""
    out = []
    for ev in events:
        b = np.floor((ev.t - ev.t[0]) / step_s).astype(int)
        n = b.max() + 1
        w = np.bincount(b, ev.w, n)
        ok = w > 0
        t = np.bincount(b, ev.w * ev.t, n)[ok] / w[ok]
        f = np.bincount(b, ev.w * ev.f, n)[ok] / w[ok]
        out.append(Event(t, f, w[ok] / np.bincount(b, None, n)[ok], ev.fit))
    return out


def _setup_local(lat, lon, elev_fn):
    h = float(elev_fn(np.array([lat]), np.array([lon]))[0])
    return LocalFrame(lat, lon, h), h


@dataclass
class LocalCand:
    """A candidate carried across chunks: everything refinement needs, in a local frame."""

    lat: float
    lon: float
    T0: float
    gh: float
    frame: LocalFrame
    trs: list  # per event: (t, enu) of the assigned flight, or None
    icaos: list  # per event
    hits: int
    pre_cost: float
    nearby: list = field(default_factory=list)  # (icao, t, enu) of other low flights near the point


def _nearby_tracks(rs: Resampled, tree, low_rows, xyz, frame, t_lo, t_hi, radius=12000.0, exclude=()):
    if tree is None:
        return []
    rows = tree.query_ball_point(xyz, radius)
    if not rows:
        return []
    segs = np.unique(rs.seg[low_rows[np.asarray(rows)]])
    out = []
    for s in segs:
        if s in exclude:
            continue
        sl = rs.rows(int(s))
        t = rs.t[sl]
        m = (t >= t_lo) & (t <= t_hi)
        if m.sum() >= 5:
            out.append((str(rs.icao[s]), t[m], (rs.xyz[sl][m] - frame.origin) @ frame.R.T))
    return out


def prescore(J: Joint, hits, events, rs, c, elev_fn, cfg: SearchConfig, log: PassLog, deadline,
             rec_dur=900.0, tree=None, low_rows=None, quiet=None):
    out = []
    with log.timed("prescore", n_in=len(J.T0)) as h:
        order = np.argsort(J.cost + cfg.silence_weight * hits)[: cfg.prescore_top]
        ghs = elev_fn(J.lat[order], J.lon[order]) if order.size else np.zeros(0)
        for n_done, i in enumerate(order):
            if time.monotonic() > deadline and n_done >= cfg.min_prescore:
                break
            gh = float(ghs[n_done])
            frame = LocalFrame(J.lat[i], J.lon[i], gh)
            trs = _local_tracks(rs, J.segs[i], frame)
            T0s = J.T0[i] + np.arange(-15, 15.01, 1.0)
            best = (np.inf, J.T0[i], 0.0, 0.0)
            for de, dn in PRESCORE_GRID:
                obs = frame.ground_enu(de, dn, gh)
                vals = _misfit_scan(events, _event_geoms(events, trs, obs, c), T0s)
                j = int(np.argmin(vals))
                if vals[j] < best[0]:
                    best = (float(vals[j]), float(T0s[j]), de, dn)
            if best[2] or best[3]:  # re-centre on the best grid point
                la_, lo_, _ = frame.from_enu(frame.ground_enu(best[2], best[3], gh))
                frame = LocalFrame(float(la_), float(lo_), gh)
                trs = _local_tracks(rs, J.segs[i], frame)
            if np.isfinite(best[0]):
                out.append(LocalCand(float(frame.lat0), float(frame.lon0), float(best[1]), gh, frame, trs,
                                     [str(rs.icao[s]) if s >= 0 else None for s in J.segs[i]], int(hits[i]),
                                     float(best[0] + 0.05 * cfg.silence_weight * hits[i])))
        out.sort(key=lambda lc: lc.pre_cost - cfg.join_bonus * sum(i is not None for i in lc.icaos))
        out = out[: cfg.keep_per_chunk]
        # attach nearby flights so that refinement can explain events the join left unmatched
        for lc in out:
            xyz = lc.frame.origin + lc.frame.ground_enu(0.0, 0.0, lc.gh) @ lc.frame.R
            lc.nearby = _nearby_tracks(rs, tree, low_rows, xyz, lc.frame, lc.T0 - 300, lc.T0 + rec_dur + 120,
                                       exclude=set(int(s) for s in J.segs[int(np.argmin(np.abs(J.T0 - lc.T0)))] if s >= 0))
        if quiet is not None:
            # silence evidence early, so coincidental matches in busy airspace do not crowd out the truth
            for lc in out:
                heard_rms = [0.0 if ic is not None else float("nan") for ic in lc.icaos]
                lc.pre_cost += cfg.unheard_cost * len(unheard_aircraft(lc, (0.0, 0.0, lc.T0), events, heard_rms,
                                                                       quiet, c, cfg))
            out.sort(key=lambda lc: lc.pre_cost - cfg.join_bonus * sum(i is not None for i in lc.icaos))
        h["out"] = len(out)
    return out


def _nm_refine(lc: LocalCand, events, c, cfg: SearchConfig, maxiter_per_dim=250):
    matched = [k for k, tr in enumerate(lc.trs) if tr is not None]
    sig, lam = cfg.track_dt_sigma, cfg.track_dt_weight
    use_sh = sig > 0
    K = len(events)

    def unpack(x):
        sh = np.zeros(K)
        if use_sh:
            sh[matched] = x[3:]
        return sh

    def obj(x):
        sh = unpack(x)
        pen = lam * float(np.sum((sh / sig) ** 2)) if use_sh else 0.0
        return curve_misfit(events, lc.trs, lc.frame.ground_enu(x[0], x[1], lc.gh), x[2], c, shifts=sh)[0] + pen

    nx = 3 + (len(matched) if use_sh else 0)
    x0 = np.zeros(nx)
    x0[2] = lc.T0
    steps = [200, 200, 1.5] + [1.0] * (nx - 3)
    simplex = np.vstack([x0] + [x0 + np.eye(nx)[i] * steps[i] for i in range(nx)])
    r = minimize(obj, x0, method="Nelder-Mead",
                 options=dict(initial_simplex=simplex, xatol=0.5, fatol=1e-5, maxiter=maxiter_per_dim * nx))
    sh = unpack(r.x)
    cost, rms, f0 = curve_misfit(events, lc.trs, lc.frame.ground_enu(r.x[0], r.x[1], lc.gh), r.x[2], c, shifts=sh)
    if use_sh:
        cost += lam * float(np.sum((sh / sig) ** 2))
    return r.x, cost, rms, f0


def _explain_more(lc: LocalCand, x, events, c, cfg: SearchConfig):
    """Try to attribute unmatched events to any nearby flight (full curve model, small time
    shift scan). Returns True if anything was added."""
    obs = lc.frame.ground_enu(x[0], x[1], lc.gh)
    added = False
    used = set(i for i in lc.icaos if i)
    for k, tr in enumerate(lc.trs):
        if tr is not None:
            continue
        best = None
        for icao, t, enu in lc.nearby:
            if icao in used:
                continue
            trs = [None] * len(events)
            trs[k] = (t, enu)
            for dT in np.arange(-4, 4.01, 1.0):
                cost, rms, _ = curve_misfit(events, trs, obs, x[2], c, shifts=np.full(len(events), dT))
                if rms and np.isfinite(rms[k]) and rms[k] < cfg.explain_rms_hz and (best is None or rms[k] < best[0]):
                    best = (rms[k], icao, t, enu)
        if best is not None:
            lc.trs[k] = (best[2], best[3])
            lc.icaos[k] = best[1]
            used.add(best[1])
            added = True
    return added


def audible_radius(heard: list, cfg: SearchConfig) -> float:
    """How far away a comparable aircraft would still have been detected, calibrated on the
    flyovers we did hear: tone power falls as 1/r^2, i.e. 2 ln(r/d) nats of log power, which is
    ~2 ln(r/d)/1.1 in the whitened z units the detector thresholds. Keep a safety margin (aircraft
    differ in loudness) and clamp to [audible_frac, audible_max] x the heard distance."""
    radii = []
    for e in heard:
        z = float(np.percentile(e.w, 90)) if len(e.w) else cfg.detect_z
        r = e.fit.d * np.exp(max(z - cfg.detect_z - cfg.audible_margin_z, 0.0) * 1.1 / 2)
        radii.append(float(np.clip(r, cfg.audible_frac * e.fit.d, cfg.audible_max * e.fit.d)))
    return float(np.median(radii))


def unheard_aircraft(lc: LocalCand, x, events, rms, quiet, c, cfg: SearchConfig):
    """Nearby aircraft that the matched flyovers say should have been clearly audible (closer than
    audible_frac x their median distance) and whose pass reaches the microphone during a quiet
    stretch of the recording: evidence against this location/time. Returns their ICAOs."""
    if quiet is None or not lc.nearby:
        return []
    good = [k for k, ic in enumerate(lc.icaos) if ic is not None and rms and k < len(rms)
            and np.isfinite(rms[k]) and rms[k] < 2 * cfg.good_rms_hz]
    R = audible_radius([events[k] for k in good] or [min(events, key=lambda e: e.fit.d)], cfg)
    obs = lc.frame.ground_enu(x[0], x[1], lc.gh)
    used = set(i for i in lc.icaos if i)
    out = []
    for icao, t, enu in lc.nearby:
        if icao in used:
            continue
        r = np.linalg.norm(enu - obs, axis=1)
        i = int(np.argmin(r))
        if r[i] > R or i == 0 or i == len(r) - 1:  # not closer than the heard ones, or CPA not seen
            continue
        t_rx = t[i] + r[i] / c - x[2]
        if quiet.is_quiet(t_rx - cfg.unheard_pad_s, t_rx + cfg.unheard_pad_s):
            out.append(icao)
    return out


def polish(pool: list, events, c, cfg: SearchConfig, log: PassLog, deadline, quiet=None):
    """Quick local optimisation of (east, north, T0) for many candidates, so that ones that
    started a few hundred metres / seconds off the truth can compete before full refinement."""
    with log.timed("polish", n_in=len(pool)) as h:
        done = []
        for n_done, lc in enumerate(pool[: cfg.polish_top]):
            if time.monotonic() > deadline and n_done >= cfg.min_polish:
                break

            def obj(x):
                return curve_misfit(events, lc.trs, lc.frame.ground_enu(x[0], x[1], lc.gh), x[2], c)[0]

            x0 = np.array([0.0, 0.0, lc.T0])
            simplex = np.array([x0, x0 + [300, 0, 0], x0 + [0, 300, 0], x0 + [0, 0, 4.0]])
            r = minimize(obj, x0, method="Nelder-Mead",
                         options=dict(initial_simplex=simplex, xatol=5.0, fatol=1e-3, maxiter=90))
            if np.isfinite(r.fun):
                lc.frame, lc.gh, lc.trs = _shift_frame(lc, r.x)
                lc.T0 = float(r.x[2])
                lc.pre_cost = float(r.fun + 0.05 * cfg.silence_weight * lc.hits)
                if quiet is not None:
                    rms = curve_misfit(events, lc.trs, lc.frame.ground_enu(0.0, 0.0, lc.gh), lc.T0, c)[1]
                    lc.pre_cost += cfg.unheard_cost * len(unheard_aircraft(lc, (0.0, 0.0, lc.T0), events, rms,
                                                                           quiet, c, cfg))
                lat, lon, _ = lc.frame.from_enu(lc.frame.ground_enu(0.0, 0.0, lc.gh))
                lc.lat, lc.lon = float(lat), float(lon)
            done.append(lc)
        done.sort(key=lambda lc: lc.pre_cost)
        h["out"] = len(done)
    return done + pool[len(done):]


def refine(pool: list, events, c, cfg: SearchConfig, log: PassLog, deadline, quiet=None):
    sols = []
    with log.timed("refine", n_in=len(pool)) as h:
        for n_done, lc in enumerate(pool[: cfg.refine_top]):
            if time.monotonic() > deadline and n_done >= cfg.min_refine:
                break
            x, cost, rms, f0 = _nm_refine(lc, events, c, cfg)
            if not np.isfinite(cost):
                continue
            if len(events) > 2 and lc.nearby and _explain_more(lc, x, events, c, cfg):
                lc.T0 = float(x[2])
                lc.frame, lc.gh, lc.trs = _shift_frame(lc, x)
                x2, cost2, rms2, f02 = _nm_refine(lc, events, c, cfg, maxiter_per_dim=150)
                if np.isfinite(cost2):
                    x, cost, rms, f0 = x2, cost2, rms2, f02
            lat, lon, _ = lc.frame.from_enu(lc.frame.ground_enu(x[0], x[1], lc.gh))
            # a flight whose curve does not fit is not an explanation: report it as unmatched
            fl = [ic if (ic is not None and k < len(rms) and np.isfinite(rms[k]) and rms[k] < 2 * cfg.good_rms_hz)
                  else None for k, ic in enumerate(lc.icaos)]
            rr = [r if fl[k] is not None else float("nan") for k, r in enumerate(rms)] if rms else rms
            unheard = unheard_aircraft(lc, x, events, rr, quiet, c, cfg)
            sols.append(Solution(float(lat), float(lon), float(x[2]),
                                 float(cost + 0.05 * cfg.silence_weight * lc.hits + cfg.unheard_cost * len(unheard)),
                                 rr, f0, fl, lc.hits, unheard))
        h["out"] = len(sols)
    return sols


def _shift_frame(lc: LocalCand, x):
    """Re-centre a candidate's local frame on the refined position (keeps NM steps sensible)."""
    lat, lon, _ = lc.frame.from_enu(lc.frame.ground_enu(x[0], x[1], lc.gh))
    nf = LocalFrame(float(lat), float(lon), lc.gh)
    conv = lambda enu: ((lc.frame.origin + enu @ lc.frame.R) - nf.origin) @ nf.R.T
    trs = [None if tr is None else (tr[0], conv(tr[1])) for tr in lc.trs]
    lc.nearby = [(i, t, conv(e)) for i, t, e in lc.nearby]
    return nf, lc.gh, trs


def _merge_solutions(sols, km=0.5):
    sols = sorted(sols, key=lambda s: s.cost)
    out = []
    for s in sols:
        if not any(haversine_m(s.lat, s.lon, u.lat, u.lon) < km * 1000 and abs(s.T0 - u.T0) < 10 for u in out):
            out.append(s)
    return out


def n_good(sol: Solution, cfg: SearchConfig) -> int:
    return int(sum(1 for r in sol.rms_hz if np.isfinite(r) and r < cfg.good_rms_hz))


def n_independent(sol: Solution, cfg: SearchConfig, events=None) -> int:
    """Well-explained flyovers that are independent constraints: several spectral lines of one
    aircraft's single pass (an engine/rotor comb) count once."""
    keys = set()
    for k, (r, ic) in enumerate(zip(sol.rms_hz, sol.flights)):
        if ic is None or not np.isfinite(r) or r >= cfg.good_rms_hz:
            continue
        tca = events[k].fit.t_ca if events is not None else k * 1000.0
        keys.add((ic, int(round(tca / 60.0))))
    return len(keys)


def decide(sols, cfg: SearchConfig, events=None):
    """solved: the best explains >= 2 flyovers well (rms < good_rms_hz) and every rival elsewhere
    (> isolate_km or > 30 s away) costs at least isolate_margin more (~one unexplained flyover)."""
    if not sols:
        return "none", None
    best = sols[0]
    need = min(2, len(best.rms_hz)) if best.rms_hz else 2
    rivals = [s for s in sols[1:] if s.cost < best.cost + cfg.isolate_margin and n_good(s, cfg) >= need and
              (haversine_m(s.lat, s.lon, best.lat, best.lon) > cfg.isolate_km * 1000 or abs(s.T0 - best.T0) > 30)]
    if n_good(best, cfg) >= 2 and n_independent(best, cfg, events) >= 2 and not rivals:
        return "solved", best
    return "ambiguous", best


_CTX = {}


def _chunk_worker(a, b, budget_end):
    log = PassLog()
    part, miss = _process_chunk(_CTX["ctx"], a, b, budget_end, log, None)
    return part, miss, log.passes


def _process_chunk(ctx, a, b, budget_end, log: PassLog, progress=None):
    """All per-chunk passes, from loading ADS-B to prescoring. Returns (LocalCands, missing days)."""
    from .tracks import TrackTable

    events, world, region, cfg, c = ctx["events"], ctx["world"], ctx["region"], ctx["cfg"], ctx["c"]
    rec_dur, max_d, max_v, bbox = ctx["rec_dur"], ctx["max_d"], ctx["max_v"], ctx["bbox"]
    missing = set()
    t_lo = a + min(e.fit.t_ca for e in events) - 120 - max_d / c
    t_hi = max(b + max(e.fit.t_end for e in events) + 120, b + rec_dur + 120)
    with log.timed("load") as h:
        tt = world.tracks(t_lo, t_hi, bbox, progress=progress, on_missing=lambda d: missing.add(d.isoformat()))
        h["out"] = tt.n_rows
    with log.timed("altitude", n_in=tt.n_rows) as h:
        # rows too high, or moving far too fast, to be any of the (join-usable) flyovers
        seg = np.repeat(np.arange(tt.n_segments), np.diff(tt.offsets))
        keep = tt.alt_m < max_d + MAX_TERRAIN_M
        if tt.n_rows > 2:
            dtr = np.diff(tt.t)
            dxy = np.hypot(np.diff(tt.lat) * 111_000, np.diff(tt.lon) * 111_000 * np.cos(np.radians(tt.lat[1:])))
            same = (seg[1:] == seg[:-1]) & (dtr > 0.5) & (dtr < 120)
            v_raw = np.where(same, dxy / np.where(same, dtr, 1), np.nan)
            v_row = np.fmin(np.r_[v_raw, np.nan], np.r_[np.nan, v_raw])
            keep &= ~(v_row > 1.4 * max_v + 20)
        tt = TrackTable.from_columns(tt.icao[seg[keep]], tt.t[keep], tt.lat[keep], tt.lon[keep], tt.alt_m[keep])
        h["out"] = tt.n_rows
    with log.timed("resample", n_in=tt.n_segments) as h:
        rs = resample(tt, t_lo, t_hi, dt=cfg.dt)
        h["out"] = rs.t.size
    rs.speed = np.linalg.norm(rs.vel, axis=1)
    cands = [event_candidates(ev, rs, world.elevation, region, a, b, c, cfg, log, k) for k, ev in enumerate(events)]
    J = join_events(cands, events, cfg, log)
    J = dedupe(J, cfg, log)
    with log.timed("terrain") as h:
        pre_low = np.nonzero(rs.alt < 3000 + MAX_TERRAIN_M)[0]
        if pre_low.size:
            el_low = world.elevation(rs.lat[pre_low], rs.lon[pre_low])
            low_rows = pre_low[rs.alt[pre_low] - el_low < 3000]
        else:
            low_rows = np.zeros(0, int)
        h["out"] = int(low_rows.size)
    tree = cKDTree(rs.xyz[low_rows]) if low_rows.size else None
    hits = silence(J, rs, events, rec_dur, c, cfg, log, world.elevation, low_rows=low_rows)
    part = prescore(J, hits, ctx["events_fast"], rs, c, world.elevation, cfg, log, budget_end, rec_dur, tree,
                    low_rows, ctx.get("quiet"))
    return part, missing


def search(events: list[Event], world, region: Region, window, rec_dur: float, cfg: SearchConfig | None = None,
           log: PassLog | None = None, ui=None, quiet=None) -> SearchResult:
    """window: TimeWindow for the recording start. ui: optional object with hooks (see cli.RichUI)."""
    cfg = cfg or SearchConfig()
    log = log or PassLog()
    c = speed_of_sound(cfg.temp_c)
    near = [e for e in events if e.fit.d <= cfg.max_join_d] or list(events)
    max_d = max(e.fit.d + _tols(e, cfg)[0] for e in near)
    max_v = max(e.fit.v + _tols(e, cfg)[1] for e in near)
    pad_m = max_d + 5000
    bbox = region.padded_bbox(pad_m)
    t_start = time.monotonic()
    deadline = t_start + cfg.timeout_s
    chunk_deadline = t_start + cfg.chunk_budget * cfg.timeout_s  # leave time for global refinement
    pool = []
    searched = 0.0
    timed_out = False
    notes = []
    chunks = list(window.chunks(cfg.chunk_s))
    if ui:
        ui.start_chunks(len(chunks))
    missing_days = set()
    ctx = dict(events=events, events_fast=decimate_events(events), quiet=quiet, world=world, region=region, cfg=cfg, c=c, rec_dur=rec_dur, max_d=max_d,
               max_v=max_v, bbox=bbox)
    workers = max(1, min(cfg.workers, len(chunks)))
    grace = min(2.0, 0.05 * cfg.timeout_s)
    if time.monotonic() >= chunk_deadline:
        timed_out = True
        chunks_to_run = []
    if timed_out:
        pass
    elif workers == 1:
        for ci, (a, b) in enumerate(chunks):
            if time.monotonic() > chunk_deadline:
                timed_out = True
                break
            remaining = max(len(chunks) - ci, 1)
            budget_end = min(chunk_deadline, time.monotonic() + max((chunk_deadline - time.monotonic()) / remaining, 0.5))
            part, miss = _process_chunk(ctx, a, b, budget_end, log, ui.download if ui else None)
            pool += part
            missing_days |= miss
            searched += b - a
            if ui:
                ui.chunk_done(ci, a, b, len(pool), log)
    else:
        import multiprocessing as mp
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        _CTX["ctx"] = ctx  # inherited by forked workers (no pickling of large worlds)
        ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"))
        try:
            n_waves = int(np.ceil(len(chunks) / workers))
            futs = {}
            for ci, (a, b) in enumerate(chunks):
                wave = ci // workers
                budget_end = t_start + (wave + 1) * (chunk_deadline - t_start) / n_waves
                futs[ex.submit(_chunk_worker, a, b, budget_end)] = (ci, a, b)
            pending = set(futs)
            while pending:
                left = chunk_deadline - time.monotonic()
                done, pending = wait(pending, timeout=max(left, 0.0) + grace, return_when=FIRST_COMPLETED)
                for f in done:
                    ci, a, b = futs[f]
                    part, miss, stats = f.result()
                    pool += part
                    missing_days |= miss
                    log.merge(stats)
                    searched += b - a
                    if ui:
                        ui.chunk_done(ci, a, b, len(pool), log)
                if time.monotonic() > chunk_deadline + grace and pending:
                    timed_out = True
                    for f in pending:
                        f.cancel()
                    break
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            _CTX.pop("ctx", None)
    # unpolished positions are a few hundred metres off, so prescores are noisy: favour
    # candidates whose join matched more flyovers consistently (position, T0, v, d)
    pool.sort(key=lambda lc: lc.pre_cost - cfg.join_bonus * sum(i is not None for i in lc.icaos))
    polish_deadline = time.monotonic() + 0.6 * max(deadline - time.monotonic(), 0)
    pool = polish(pool, ctx["events_fast"], c, cfg, log, polish_deadline, quiet)
    sols = refine(pool, events, c, cfg, log, deadline, quiet)
    sols = [s_ for s_ in _merge_solutions(sols) if n_good(s_, cfg) >= 1]  # must explain something
    status, best = decide(sols, cfg, events)
    if missing_days:
        notes.append(f"no ADS-B data available for: {', '.join(sorted(missing_days))}")
    if timed_out:
        notes.append(f"timeout after {cfg.timeout_s:g}s; searched {searched / max(window.span, 1):.0%} of the time window")
    from .regions import which_state

    for s in sols[:25]:
        s.state = which_state(s.lat, s.lon)
    return SearchResult(status, best, sols[:50], log, timed_out, searched, window.span, notes)
