"""Repair trace timestamps in the Hugging Face adsb.lol mirror.

The mirror's `traces_XX.parquet` rows keep only the *fractional* second of each
position timestamp (the integer seconds were lost; every row reads 00:00:00.xx), but
rows remain in chronological order per aircraft. The mirror's heatmap files carry
absolute integer timestamps at a 10 s cadence. For each aircraft we:

  1. compute cumulative arc length along the (ordered) trace,
  2. match every heatmap fix to its nearest trace row (KD-tree), keep matches whose
     row index increases monotonically,
  3. interpolate an approximate absolute time for every trace row from the anchors
     (by arc length, falling back to row index when the aircraft is not moving),
  4. snap it to the nearest instant whose fractional second equals the surviving
     fractional part. Error after snapping is ~0 when step 3 is within ±0.5 s.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _xy(lat, lon, lat0, lon0):
    return np.column_stack([(lon - lon0) * np.cos(np.radians(lat0)) * 111_320.0, (lat - lat0) * 110_540.0])


def repair_aircraft(frac, lat, lon, h_t, h_lat, h_lon, gs=None, max_anchor_m=600.0):
    """Return absolute times (float s) for trace rows, NaN where not recoverable."""
    n = frac.size
    out = np.full(n, np.nan)
    if n < 2 or h_t.size < 2:
        return out
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    P = _xy(lat, lon, lat0, lon0)
    Q = _xy(h_lat, h_lon, lat0, lon0)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    # parameter: arc length, plus a tiny per-row increment so stationary rows stay ordered
    s = seg + 1e-3 * np.arange(n)
    d, idx = cKDTree(P).query(Q)
    o = np.argsort(h_t, kind="stable")
    d, idx, ht = d[o], idx[o], h_t[o]
    ok = d < max_anchor_m
    d, idx, ht = d[ok], idx[ok], ht[ok]
    if idx.size < 2:
        return out
    # keep a monotone subsequence: anchors agreeing with a running median of indices
    keep = np.ones(idx.size, bool)
    if idx.size >= 5:
        from scipy.ndimage import median_filter

        med = median_filter(idx.astype(float), size=5, mode="nearest")
        keep = np.abs(idx - med) <= np.maximum(20, 0.2 * med.clip(1))
    idx, ht = idx[keep], ht[keep]
    mono = np.r_[True, np.diff(idx) > 0] & np.r_[True, np.diff(ht) > 0]
    idx, ht = idx[mono], ht[mono]
    if idx.size < 2:
        return out
    q_xy = Q[o][ok][keep][mono]
    # heatmap stamps are the START of a 10 s slot: the fix was observed later (median ~6.5 s)
    t_est = np.interp(s, s[idx], ht + HEAT_LAG_MEDIAN, left=np.nan, right=np.nan)
    # allow a short extrapolation (<= 30 s) past the first/last anchor at the local rate
    for side in ("left", "right"):
        a, b = (0, 1) if side == "left" else (-2, -1)
        rate = (ht[b] - ht[a]) / max(s[idx[b]] - s[idx[a]], 1e-6)
        e = a if side == "left" else b
        rows = np.arange(0, idx[0]) if side == "left" else np.arange(idx[-1] + 1, n)
        if rows.size:
            ext = ht[e] + HEAT_LAG_MEDIAN + (s[rows] - s[idx[e]]) * rate
            okx = np.abs(ext - ht[e]) <= 30.0
            t_est[rows[okx]] = ext[okx]
    out = np.round(t_est - frac) + frac
    if gs is not None:
        out = _chain_with_speed(out, frac, P, gs, idx, ht, q_xy)
    # sanity: times must increase
    bad = np.r_[False, np.diff(out) <= 0]
    out[bad] = np.nan
    return out


# Empirical lag (true fix time - heatmap stamp), measured against ADSBExchange full traces
# on 2023-08-01: 5/10/50/90% = 0.5/1.0/6.5/20 s.
HEAT_LAG_MEDIAN = 6.5
HEAT_LAG_LOW = 1.0


def _chain_with_speed(out, frac, P, gs, a_idx, a_t, a_xy, max_dt=40.0, q=90, window_s=60.0):
    """Relative times between consecutive rows from distance / ground speed (snapped to the
    known fractional seconds). Each chain's absolute offset comes from the heatmap anchors,
    which are lower bounds on the true time: offset = q-th percentile of the bounds + the
    small lag that percentile corresponds to."""
    n = out.size
    dist = np.hypot(*np.diff(P, axis=0).T)
    g = 0.5 * (gs[1:] + gs[:-1])
    dt_est = dist / np.where(g > 15, g, np.nan)
    dfrac = np.diff(frac)
    dt = np.round(dt_est - dfrac) + dfrac
    dt = np.where(dt <= 0, dt + 1, dt)
    valid = np.isfinite(dt) & (dt_est < max_dt)
    brk = np.r_[0, np.nonzero(~valid)[0] + 1, n]
    res = out.copy()
    # direction of travel at each row (towards next row; last row uses previous)
    d = np.diff(P, axis=0)
    d = np.vstack([d, d[-1:]])
    u = d / np.maximum(np.linalg.norm(d, axis=1), 1e-6)[:, None]
    gg = np.where(np.isfinite(gs) & (gs > 15), gs, np.nan)
    for a, b in zip(brk[:-1], brk[1:]):
        if b - a < 3:
            continue
        tau = np.r_[0.0, np.cumsum(dt[a:b - 1])]
        m = (a_idx >= a) & (a_idx < b)
        if m.sum() == 0:
            continue
        i = a_idx[m]
        proj = np.sum((a_xy[m] - P[i]) * u[i], axis=1) / gg[i]
        tau_fix = tau[i - a] + proj
        lb = a_t[m] - tau_fix
        okk = np.isfinite(lb)
        lb, tf = lb[okk], tau_fix[okk]
        if lb.size == 0:
            continue
        if lb.size < 5:
            res[a:b] = np.median(lb) + HEAT_LAG_MEDIAN + tau
            continue
        # offset drifts slowly (ground-speed rounding accumulates): estimate it in windows
        order = np.argsort(tf)
        lb, tf = lb[order], tf[order]
        centers = np.arange(tf[0], tf[-1] + window_s / 2, window_s / 2)
        lo_i = np.searchsorted(tf, centers - window_s, "left")
        hi_i = np.searchsorted(tf, centers + window_s, "right")
        offs, cs = [], []
        for c, i0, i1 in zip(centers, lo_i, hi_i):
            if i1 - i0 >= 4:
                v = np.sort(lb[i0:i1])
                offs.append(v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] + HEAT_LAG_LOW)
                cs.append(c)
        if not offs:
            offs, cs = [np.percentile(lb, q) + HEAT_LAG_LOW], [tf.mean()]
        res[a:b] = tau + np.interp(tau, cs, offs)
    return res


def repair_shard(icao, frac, lat, lon, heat_index, gs=None):
    """icao: (N,) hex str rows grouped by aircraft (trace order). heat_index: icao -> (t, lat, lon)."""
    t = np.full(icao.size, np.nan)
    if icao.size == 0:
        return t
    brk = np.r_[0, np.nonzero(icao[1:] != icao[:-1])[0] + 1, icao.size]
    for a, b in zip(brk[:-1], brk[1:]):
        h = heat_index.get(icao[a])
        if h is None:
            continue
        t[a:b] = repair_aircraft(frac[a:b], lat[a:b], lon[a:b], *h, gs=None if gs is None else gs[a:b])
    return t
