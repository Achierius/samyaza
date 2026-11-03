"""Find where (and when) a recording was made by matching flyover events to ADS-B tracks.

Unknowns: observer position (e, n) on the ground and absolute recording start time T0.

Stage A (coarse, vectorised): for every grid point x every flight, compute the closest
point of approach (slant range, 3-D speed, reception time of CPA sound). An event's
straight-pass fit (d, v, t_ca) must match those within tolerance; with >1 event, the
implied T0 = t_ca_abs - t_ca_rel must agree across events. That inter-event timing is
what pins down the location. Optional silence check: no other aircraft should have
passed closer than the detected ones while the recorder was running.

Stage B (fine): for the best coarse candidates, compute the full retarded-time Doppler
curve along the real (turning, climbing) ADS-B track and minimise the misfit to the
observed ridges over (e, n, T0), solving f0 per event in closed form.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from .adsb import Track
from .audio import Event
from .geo import LocalFrame
from .physics import received_ratio, speed_of_sound


@dataclass
class PTrack:
    icao: str
    callsign: str
    t: np.ndarray  # uniform emission times
    p: np.ndarray  # (N, 3) ENU
    v: np.ndarray  # (N, 3) ENU velocity


def prepare_tracks(tracks: list[Track], frame: LocalFrame, t_lo, t_hi, dt=1.0, smooth_s=2.0,
                   max_gap_s=120.0, max_range_m=None) -> list[PTrack]:
    out = []
    for tr in tracks:
        m = (tr.t >= t_lo - max_gap_s) & (tr.t <= t_hi + max_gap_s)
        if m.sum() < 2:
            continue
        t = tr.t[m]
        enu = frame.to_enu(tr.lat[m], tr.lon[m], tr.alt_m[m])
        # split on large gaps rather than interpolating across them
        cuts = np.where(np.diff(t) > max_gap_s)[0] + 1
        for s in np.split(np.arange(t.size), cuts):
            if s.size < 2:
                continue
            ts = t[s]
            a = max(ts[0], t_lo)
            b = min(ts[-1], t_hi)
            if b - a < 10 * dt:
                continue
            tg = np.arange(np.ceil(a / dt) * dt, b, dt)
            p = np.stack([np.interp(tg, ts, enu[s, k]) for k in range(3)], axis=1)
            if smooth_s > 0:
                p = gaussian_filter1d(p, smooth_s / dt, axis=0, mode="nearest")
            if max_range_m is not None and np.min(np.hypot(p[:, 0], p[:, 1])) > max_range_m:
                continue
            v = np.gradient(p, dt, axis=0)
            out.append(PTrack(tr.icao, tr.callsign, tg, p, v))
    return out


@dataclass
class Grid:
    e: np.ndarray
    n: np.ndarray
    p: np.ndarray  # (G, 3)


def make_grid(frame: LocalFrame, radius_m: float, spacing_m: float, ground_h: float) -> Grid:
    ax = np.arange(-radius_m, radius_m + 1e-6, spacing_m)
    E, N = np.meshgrid(ax, ax)
    E, N = E.ravel(), N.ravel()
    keep = np.hypot(E, N) <= radius_m
    E, N = E[keep], N[keep]
    return Grid(E, N, frame.ground_enu(E, N, ground_h))


def cpa_features(pt: PTrack, gp: np.ndarray, c: float, chunk: int = 2048):
    """Closest approach of track `pt` to each grid point. Returns rmin, t_ca_rx, speed (G,)."""
    G = gp.shape[0]
    rmin = np.full(G, np.inf)
    tca = np.full(G, np.nan)
    spd = np.full(G, np.nan)
    P = pt.p
    PP = np.einsum("ij,ij->i", P, P)
    dt = pt.t[1] - pt.t[0]
    speed = np.linalg.norm(pt.v, axis=1)
    for a in range(0, G, chunk):
        g = gp[a:a + chunk]
        r2 = PP[None, :] - 2 * g @ P.T + np.einsum("ij,ij->i", g, g)[:, None]
        i = np.argmin(r2, axis=1)
        interior = (i > 0) & (i < P.shape[0] - 1)
        ii = np.clip(i, 1, P.shape[0] - 2)
        rows = np.arange(g.shape[0])
        rm = np.sqrt(np.maximum(r2[rows, ii - 1], 0))
        r0 = np.sqrt(np.maximum(r2[rows, ii], 0))
        rp = np.sqrt(np.maximum(r2[rows, ii + 1], 0))
        den = rm - 2 * r0 + rp
        delta = np.where(den > 1e-9, 0.5 * (rm - rp) / den, 0.0).clip(-0.5, 0.5)
        r_ca = r0 - 0.25 * (rm - rp) * delta
        te = pt.t[ii] + delta * dt
        sl = slice(a, a + g.shape[0])
        rmin[sl] = np.where(interior, r_ca, np.inf)
        tca[sl] = te + r_ca / c
        spd[sl] = speed[ii]
    return rmin, tca, spd


@dataclass
class Candidate:
    g: int
    e: float
    n: float
    T0: float
    flights: list
    cost_a: float
    silence_hits: int = 0
    cost_b: float = np.inf
    rms_hz: list = field(default_factory=list)
    f0: list = field(default_factory=list)
    lat: float = np.nan
    lon: float = np.nan


def _tolerances(fit):
    se = fit.stderr or {}
    tol_d = max(0.25 * fit.d, 3 * (se.get("d") or 0), 150.0)
    tol_v = max(0.10 * fit.v, 3 * (se.get("v") or 0), 6.0)
    tol_t = max(4.0, 3 * (se.get("t_ca") or 0))
    return tol_d, tol_v, tol_t


def coarse_search(events: list[Event], ptracks: list[PTrack], grid: Grid, c: float,
                  T0_lo: float, T0_hi: float, rec_dur: float, silence_range_m: float | None = None,
                  silence_weight: float = 4.0, max_candidates: int = 400) -> list[Candidate]:
    F, G = len(ptracks), grid.p.shape[0]
    R = np.empty((F, G), np.float32)
    TC = np.empty((F, G), np.float64)
    SP = np.empty((F, G), np.float32)
    for k, pt in enumerate(ptracks):
        R[k], TC[k], SP[k] = cpa_features(pt, grid.p, c)

    per_event = []
    for ev in events:
        fit = ev.fit
        tol_d, tol_v, tol_t = _tolerances(fit)
        T0 = TC - fit.t_ca
        cost = ((R - fit.d) / tol_d) ** 2 + ((SP - fit.v) / tol_v) ** 2
        ok = (cost < 2.0) & (T0 >= T0_lo) & (T0 <= T0_hi) & np.isfinite(R)
        fi, gi = np.nonzero(ok)
        per_event.append(dict(f=fi, g=gi, T0=T0[fi, gi], cost=cost[fi, gi], tol_t=tol_t))

    # join on grid point with consistent T0 (anchor = event with fewest candidates)
    order = sorted(range(len(events)), key=lambda k: per_event[k]["g"].size)
    anc = per_event[order[0]]
    g_c, T0_c, cost_c = anc["g"], anc["T0"].copy(), anc["cost"].copy()
    flights = {order[0]: anc["f"]}
    alive = np.ones(g_c.size, bool)
    BIG = 1e7  # T0 offsets below this
    for k in order[1:]:
        ek = per_event[k]
        tol = np.hypot(ek["tol_t"], anc["tol_t"])
        key_k = ek["g"] * BIG + (ek["T0"] - T0_lo)
        o = np.argsort(key_k)
        key_k = key_k[o]
        key_a = g_c * BIG + (T0_c - T0_lo)
        lo = np.searchsorted(key_k, key_a - 3 * tol, "left")
        hi = np.searchsorted(key_k, key_a + 3 * tol, "right")
        best = np.full(g_c.size, np.inf)
        bidx = np.full(g_c.size, -1)
        width = int((hi - lo).max()) if hi.size else 0
        for off in range(width):
            j = lo + off
            valid = j < hi
            jj = np.where(valid, j, 0)
            src = o[jj]
            dT = (ek["T0"][src] - T0_c) / tol
            cst = np.where(valid, ek["cost"][src] + dT**2, np.inf)
            better = cst < best
            best = np.where(better, cst, best)
            bidx = np.where(better, src, bidx)
        alive &= np.isfinite(best)
        cost_c = cost_c + best
        flights[k] = np.where(bidx >= 0, ek["f"][np.maximum(bidx, 0)], -1)
    keep = np.nonzero(alive)[0]
    g_c, T0_c, cost_c = g_c[keep], T0_c[keep], cost_c[keep]
    fl = np.stack([flights[k][keep] for k in range(len(events))], axis=1) if keep.size else np.zeros((0, len(events)), int)

    hits = np.zeros(g_c.size, int)
    if silence_range_m and g_c.size:
        for a in range(0, g_c.size, 4096):
            sl = slice(a, a + 4096)
            r = R[:, g_c[sl]]
            dtc = TC[:, g_c[sl]] - T0_c[sl][None, :]
            heard = (r < silence_range_m) & (dtc >= 0) & (dtc <= rec_dur)
            for k in range(fl.shape[1]):
                heard[fl[sl, k], np.arange(heard.shape[1])] = False
            hits[sl] = heard.sum(axis=0)
        cost_c = cost_c + silence_weight * hits

    o = np.argsort(cost_c)
    cands, seen = [], set()
    for i in o:
        key = (int(g_c[i]), tuple(int(x) for x in fl[i]))
        if key in seen:
            continue
        seen.add(key)
        cands.append(Candidate(int(g_c[i]), float(grid.e[g_c[i]]), float(grid.n[g_c[i]]), float(T0_c[i]),
                               [int(x) for x in fl[i]], float(cost_c[i]), int(hits[i])))
        if len(cands) >= max_candidates:
            break
    return cands


def curve_misfit(events, tracks_for_events, obs, T0, c, huber=1.0):
    total, rms, f0s = 0.0, [], []
    for ev, pt in zip(events, tracks_for_events):
        t_abs = T0 + ev.t
        # only the part of the track whose sound arrives during this event
        m = (pt.t >= t_abs[0] - 60) & (pt.t <= t_abs[-1] + 5)
        if m.sum() < 5:
            return np.inf, [], []
        t_r, _, ratio = received_ratio(pt.t[m], pt.p[m], obs, c)
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
    return total, rms, f0s


def refine(cands: list[Candidate], events, ptracks, frame: LocalFrame, ground_h: float, c: float,
           spacing_m: float, top: int = 60) -> list[Candidate]:
    out = []
    for cd in cands[:top]:
        trs = [ptracks[k] for k in cd.flights]

        def obj(x):
            obs = frame.ground_enu(x[0], x[1], ground_h)
            return curve_misfit(events, trs, obs, x[2], c)[0]

        x0 = np.array([cd.e, cd.n, cd.T0])
        # coarse T0 scan first: curve shape is very sensitive to timing
        Ts = cd.T0 + np.arange(-8, 8.01, 0.5)
        vals = [obj([cd.e, cd.n, T]) for T in Ts]
        if np.all(~np.isfinite(vals)):
            continue
        x0[2] = Ts[int(np.nanargmin(vals))]
        simplex = np.array([x0, x0 + [spacing_m, 0, 0], x0 + [0, spacing_m, 0], x0 + [0, 0, 1.0]])
        r = minimize(obj, x0, method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, xatol=1.0, fatol=1e-5, maxiter=600))
        obs = frame.ground_enu(r.x[0], r.x[1], ground_h)
        cost, rms, f0 = curve_misfit(events, trs, obs, r.x[2], c)
        if not np.isfinite(cost):
            continue
        lat, lon, _ = frame.from_enu(obs)
        out.append(Candidate(cd.g, float(r.x[0]), float(r.x[1]), float(r.x[2]), cd.flights, cd.cost_a,
                             cd.silence_hits, float(cost + 0.05 * cd.silence_hits), rms, f0, float(lat), float(lon)))
    out.sort(key=lambda c_: c_.cost_b)
    # collapse candidates that converged to the same solution
    uniq = []
    for c_ in out:
        if not any(u.flights == c_.flights and np.hypot(u.e - c_.e, u.n - c_.n) < 2 * spacing_m
                   and abs(u.T0 - c_.T0) < 2 for u in uniq):
            uniq.append(c_)
    return uniq


def locate(events: list[Event], tracks: list[Track], center_lat: float, center_lon: float,
           radius_m: float, spacing_m: float, T0_lo: float, T0_hi: float, rec_dur: float,
           ground_h: float = 0.0, temp_c: float = 15.0, silence_range_m: float | None = None,
           refine_top: int = 60, verbose: bool = True):
    c = speed_of_sound(temp_c)
    frame = LocalFrame(center_lat, center_lon, ground_h)
    max_d = max(e.fit.d for e in events)
    ptracks = prepare_tracks(tracks, frame, T0_lo - 120, T0_hi + rec_dur + 120,
                             max_range_m=radius_m + 2 * max_d)
    grid = make_grid(frame, radius_m, spacing_m, ground_h)
    if verbose:
        print(f"[locate] {len(ptracks)} track segments near region, {grid.p.shape[0]} grid points, {len(events)} events")
    cands = coarse_search(events, ptracks, grid, c, T0_lo, T0_hi, rec_dur, silence_range_m)
    if verbose:
        print(f"[locate] {len(cands)} coarse candidates; refining top {min(refine_top, len(cands))}")
    res = refine(cands, events, ptracks, frame, ground_h, c, spacing_m, refine_top)
    for r in res:
        r.flights = [f"{ptracks[k].icao}/{ptracks[k].callsign}".rstrip("/") for k in r.flights]
    return res, grid, cands
