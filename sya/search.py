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


@dataclass
class SearchConfig:
    temp_c: float = 15.0
    chunk_s: float = 3600.0
    dt: float = 1.0
    tol_v_frac: float = 0.10
    tol_v_min: float = 6.0
    tol_d_frac: float = 0.25
    tol_d_min: float = 150.0
    tol_t_min: float = 6.0  # real ADS-B timestamps carry a few seconds of error
    track_dt_sigma: float = 2.0  # prior on per-flight ADS-B time error (s) in refinement
    track_dt_weight: float = 0.02
    join_radius_m: float = 700.0
    min_events: int = 2  # events that must be matched to an ADS-B flight (if >= 2 detected)
    silence_frac: float = 0.6  # silence radius = this * min detected d (0 disables)
    silence_weight: float = 3.0
    max_reps: int = 4000
    prescore_top: int = 400
    refine_top: int = 60
    timeout_s: float = 60.0
    isolate_ratio: float = 2.5  # runner-up (elsewhere) must be this much worse
    isolate_km: float = 1.5
    good_rms_hz: float = 1.5


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
        speed = np.linalg.norm(rs.vel, axis=1)
        hgt = rs.alt - elev_under
        T0 = rs.t + f.d / c - f.t_ca
        m = (np.abs(speed - f.v) < tol_v) & (hgt > -50) & (hgt < f.d + tol_d) & (T0 >= T0_lo) & (T0 < T0_hi)
        idx = np.nonzero(m)[0]
        up = _up(rs.lat[idx], rs.lon[idx])
        vel = rs.vel[idx]
        vh = vel - np.sum(vel * up, 1)[:, None] * up
        vh /= np.maximum(np.linalg.norm(vh, axis=1), 1e-6)[:, None]
        nrm = np.cross(up, vh)
        hh = np.clip(hgt[idx], 0, None)
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
    K = len(cands)
    if K == 1:
        e = cands[0]
        return Joint(e.xyz, e.lat, e.lon, e.T0, e.seg[:, None], e.cost)
    tt = [_tols(ev, cfg)[2] for ev in events]
    order = sorted(range(K), key=lambda i: len(cands[i].T0))
    need = min(cfg.min_events, K)
    n_in = sum(len(c.T0) for c in cands)
    with log.timed("join", n_in=n_in) as h:
        best = None
        # try anchors in order of fewest candidates until enough anchors covered (an anchor
        # event may itself be the unmatched one when need < K)
        for ai, a in enumerate(order[: K - need + 1]):
            A = cands[a]
            if len(A.T0) == 0:
                continue
            n = len(A.T0)
            segs = np.full((n, K), -1)
            segs[:, a] = A.seg
            cost = A.cost.copy()
            matched = np.ones(n, int)
            for b in range(K):
                if b == a:
                    continue
                B = cands[b]
                if len(B.T0) == 0:
                    cost += 4.0
                    continue
                tol_t = np.hypot(tt[a], tt[b])
                scale = cfg.join_radius_m / (2 * tol_t)
                tree = cKDTree(np.column_stack([B.xyz, B.T0 * scale]))
                q = np.column_stack([A.xyz, A.T0 * scale])
                dist, j = tree.query(q, k=8, distance_upper_bound=cfg.join_radius_m * 1.5)
                valid = np.isfinite(dist)
                jj = np.where(valid, j, 0)
                dpos = np.linalg.norm(B.xyz[jj] - A.xyz[:, None, :], axis=2)
                sig = np.hypot(B.sigma[jj], A.sigma[:, None])
                dT = (B.T0[jj] - A.T0[:, None]) / tol_t
                c2 = np.where(valid, B.cost[jj] + (dpos / sig) ** 2 + dT**2, np.inf)
                kbest = np.argmin(c2, axis=1)
                cb = c2[np.arange(n), kbest]
                hit = np.isfinite(cb)
                segs[hit, b] = B.seg[jj[np.arange(n), kbest]][hit]
                cost += np.where(hit, cb, 4.0)
                matched += hit
            ok = matched >= need
            J = Joint(A.xyz[ok], A.lat[ok], A.lon[ok], A.T0[ok], segs[ok], cost[ok])
            if best is None:
                best = J
            else:
                best = Joint(*(np.concatenate([getattr(best, f), getattr(J, f)]) for f in
                               ("xyz", "lat", "lon", "T0", "segs", "cost")))
            if need == K:
                break
        if best is None:
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
            log: PassLog, elev_fn) -> np.ndarray:
    hits = np.zeros(len(J.T0), int)
    if cfg.silence_frac <= 0 or len(J.T0) == 0 or rs.t.size == 0:
        return hits
    R = cfg.silence_frac * min(ev.fit.d for ev in events)
    with log.timed("silence", n_in=len(J.T0)) as h:
        hgt = rs.alt - elev_fn(rs.lat, rs.lon)
        low = np.nonzero(hgt < R + 100)[0]
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
            return np.inf, [], []
        t_r, _, ratio = received_ratio(t[m], p[m], obs, c)
        g = np.interp(t_abs, t_r, ratio, left=np.nan, right=np.nan)
        ok = np.isfinite(g)
        if ok.mean() < 0.8:
            return np.inf, [], []
        w, f, g = ev.w[ok], ev.f[ok], g[ok]
        f0 = np.sum(w * f * g) / np.sum(w * g * g)
        res = f - f0 * g
        a = np.abs(res)
        hub = np.where(a < huber, 0.5 * a * a, huber * (a - 0.5 * huber))
        total += np.sum(w * hub) / np.sum(w)
        rms.append(float(np.sqrt(np.average(res**2, weights=w))))
        f0s.append(float(f0))
        used += 1
    if used == 0:
        return np.inf, [], []
    return total + 0.5 * (len(events) - used), rms, f0s


def _setup_local(lat, lon, elev_fn):
    h = float(elev_fn(np.array([lat]), np.array([lon]))[0])
    return LocalFrame(lat, lon, h), h


def prescore(J: Joint, hits, events, rs, c, elev_fn, cfg: SearchConfig, log: PassLog, deadline):
    out = []
    with log.timed("prescore", n_in=len(J.T0)) as h:
        order = np.argsort(J.cost + cfg.silence_weight * hits)[: cfg.prescore_top]
        for i in order:
            if time.monotonic() > deadline:
                break
            frame, gh = _setup_local(J.lat[i], J.lon[i], elev_fn)
            trs = _local_tracks(rs, J.segs[i], frame)
            obs = frame.ground_enu(0.0, 0.0, gh)
            best = (np.inf, J.T0[i])
            for dT in np.arange(-6, 6.01, 1.0):
                v = curve_misfit(events, trs, obs, J.T0[i] + dT, c)[0]
                if v < best[0]:
                    best = (v, J.T0[i] + dT)
            if np.isfinite(best[0]):
                out.append((best[0] + 0.05 * cfg.silence_weight * hits[i], int(i), best[1]))
        out.sort()
        h["out"] = len(out)
    return out


def refine(pre, J: Joint, hits, events, rs, c, elev_fn, cfg: SearchConfig, log: PassLog, deadline):
    sols = []
    with log.timed("refine", n_in=len(pre)) as h:
        for _, i, T0 in pre[: cfg.refine_top]:
            if time.monotonic() > deadline:
                break
            frame, gh = _setup_local(J.lat[i], J.lon[i], elev_fn)
            trs = _local_tracks(rs, J.segs[i], frame)

            K = len(events)
            sig, lam = cfg.track_dt_sigma, cfg.track_dt_weight

            def obj(x):
                sh = x[3:] if sig > 0 else None
                pen = lam * float(np.sum((x[3:] / sig) ** 2)) if sig > 0 else 0.0
                return curve_misfit(events, trs, frame.ground_enu(x[0], x[1], gh), x[2], c, shifts=sh)[0] + pen

            nx = 3 + (K if sig > 0 else 0)
            x0 = np.zeros(nx)
            x0[2] = T0
            steps = [200, 200, 1.5] + [1.0] * (nx - 3)
            simplex = np.vstack([x0] + [x0 + np.eye(nx)[i] * steps[i] for i in range(nx)])
            r = minimize(obj, x0, method="Nelder-Mead",
                         options=dict(initial_simplex=simplex, xatol=0.5, fatol=1e-5, maxiter=300 * nx))
            sh = r.x[3:] if sig > 0 else None
            cost, rms, f0 = curve_misfit(events, trs, frame.ground_enu(r.x[0], r.x[1], gh), r.x[2], c, shifts=sh)
            if sig > 0:
                cost += lam * float(np.sum((r.x[3:] / sig) ** 2))
            if not np.isfinite(cost):
                continue
            lat, lon, _ = frame.from_enu(frame.ground_enu(r.x[0], r.x[1], gh))
            fl = [str(rs.icao[s]) if s >= 0 else None for s in J.segs[i]]
            sols.append(Solution(float(lat), float(lon), float(r.x[2]), float(cost + 0.05 * cfg.silence_weight * hits[i]),
                                 rms, f0, fl, int(hits[i])))
        h["out"] = len(sols)
    return sols


def _merge_solutions(sols, km=0.5):
    sols = sorted(sols, key=lambda s: s.cost)
    out = []
    for s in sols:
        if not any(haversine_m(s.lat, s.lon, u.lat, u.lon) < km * 1000 and abs(s.T0 - u.T0) < 10 for u in out):
            out.append(s)
    return out


def decide(sols, cfg: SearchConfig):
    if not sols:
        return "none", None
    best = sols[0]
    good = np.nanmean([r for r in best.rms_hz if np.isfinite(r)] or [np.inf]) < cfg.good_rms_hz
    rivals = [s for s in sols[1:] if s.cost < cfg.isolate_ratio * best.cost and
              (haversine_m(s.lat, s.lon, best.lat, best.lon) > cfg.isolate_km * 1000 or abs(s.T0 - best.T0) > 30)]
    if good and not rivals:
        return "solved", best
    return "ambiguous", best


def search(events: list[Event], world, region: Region, window, rec_dur: float, cfg: SearchConfig | None = None,
           log: PassLog | None = None, ui=None) -> SearchResult:
    """window: TimeWindow for the recording start. ui: optional object with hooks (see cli.RichUI)."""
    cfg = cfg or SearchConfig()
    log = log or PassLog()
    c = speed_of_sound(cfg.temp_c)
    max_d = max(e.fit.d + _tols(e, cfg)[0] for e in events)
    pad_m = max_d + 5000
    bbox = region.padded_bbox(pad_m)
    t_start = time.monotonic()
    deadline = t_start + cfg.timeout_s
    sols = []
    searched = 0.0
    timed_out = False
    notes = []
    chunks = list(window.chunks(cfg.chunk_s))
    if ui:
        ui.start_chunks(len(chunks))
    missing_days = set()
    for ci, (a, b) in enumerate(chunks):
        if time.monotonic() > deadline:
            timed_out = True
            break
        t_lo = a + min(e.fit.t_ca for e in events) - 120 - max_d / c
        t_hi = b + max(e.fit.t_end for e in events) + rec_dur * 0 + 120
        t_hi = max(t_hi, b + rec_dur + 120)
        with log.timed("load") as h:
            tt = world.tracks(t_lo, t_hi, bbox, progress=ui.download if ui else None,
                              on_missing=lambda d: missing_days.add(d.isoformat()))
            h["out"] = tt.n_rows
        with log.timed("altitude", n_in=tt.n_rows) as h:
            keep = tt.alt_m < max_d + MAX_TERRAIN_M
            seg = np.repeat(np.arange(tt.n_segments), np.diff(tt.offsets))
            from .tracks import TrackTable

            tt = TrackTable.from_columns(tt.icao[seg[keep]], tt.t[keep], tt.lat[keep], tt.lon[keep], tt.alt_m[keep])
            h["out"] = tt.n_rows
        with log.timed("resample", n_in=tt.n_segments) as h:
            rs = resample(tt, t_lo, t_hi, dt=cfg.dt)
            h["out"] = rs.t.size
        with log.timed("terrain", n_in=rs.t.size) as h:
            elev = world.elevation(rs.lat, rs.lon) if rs.t.size else np.zeros(0)
            h["out"] = rs.t.size
        cands = [event_candidates(ev, rs, elev, region, a, b, c, cfg, log, k) for k, ev in enumerate(events)]
        J = join_events(cands, events, cfg, log)
        J = dedupe(J, cfg, log)
        hits = silence(J, rs, events, rec_dur, c, cfg, log, world.elevation)
        pre = prescore(J, hits, events, rs, c, world.elevation, cfg, log, deadline)
        sols += refine(pre, J, hits, events, rs, c, world.elevation, cfg, log, deadline)
        searched += b - a
        if ui:
            ui.chunk_done(ci, a, b, len(sols), log)
        if time.monotonic() > deadline and ci < len(chunks) - 1:
            timed_out = True
            break
    sols = _merge_solutions(sols)
    status, best = decide(sols, cfg)
    if missing_days:
        notes.append(f"no ADS-B data available for: {', '.join(sorted(missing_days))}")
    if timed_out:
        notes.append(f"timeout after {cfg.timeout_s:g}s; searched {searched / max(window.span, 1):.0%} of the time window")
    from .regions import which_state

    for s in sols[:25]:
        s.state = which_state(s.lat, s.lon)
    return SearchResult(status, best, sols[:50], log, timed_out, searched, window.span, notes)
