"""Semi-real validation: REAL ADS-B traffic, SYNTHETIC audio.

Given a real world (cached/frozen ADS-B + terrain), an observer position and a start
time, render what a microphone there would have heard from every real aircraft. Each
flight's sound is rendered with a random time offset (default sigma 1.5 s) to mimic
the timing error of real ADS-B data relative to reality.
"""
from __future__ import annotations

import numpy as np

from .geo import LocalFrame
from .physics import speed_of_sound
from .synth import Flight, SynthWorld
from .tracks import resample


def hybrid_world(world, lat: float, lon: float, T0: float, dur: float, seed: int = 0, temp_c: float = 15.0,
                 max_range_m: float = 8000.0, time_jitter_s: float = 1.5, spec: str = "hybrid") -> SynthWorld:
    rng = np.random.default_rng(seed)
    gh = float(world.elevation(np.array([lat]), np.array([lon]))[0])
    frame = LocalFrame(lat, lon, gh)
    pad = 600
    dlat = 0.3
    dlon = 0.3 / np.cos(np.radians(lat))
    tt = world.tracks(T0 - pad, T0 + dur + pad, (lat - dlat, lon - dlon, lat + dlat, lon + dlon))
    rs = resample(tt, T0 - pad, T0 + dur + pad, dt=0.5, smooth_s=1.5)
    flights = []
    obs = frame.ground_enu(0.0, 0.0, gh)
    for s in range(len(rs.icao)):
        sl = rs.rows(s)
        p = (rs.xyz[sl] - frame.origin) @ frame.R.T
        if np.min(np.linalg.norm(p - obs, axis=1)) > max_range_m:
            continue
        spd = np.median(np.linalg.norm(rs.vel[sl], axis=1))
        loud = 0.8 if spd > 110 else 0.4
        jitter = rng.normal(0, time_jitter_s) if time_jitter_s > 0 else 0.0
        flights.append(Flight(str(rs.icao[s]), rs.t[sl] + jitter, p, rng.uniform(80, 150), loud,
                              (1.0, rng.uniform(0.3, 0.8), rng.uniform(0.1, 0.5))))
    return SynthWorld(spec, frame, gh, (0.0, 0.0), T0, dur, flights, speed_of_sound(temp_c))


def find_observers(world, lat, lon, t_lo, t_hi, n_try=400, half_m=15000, seed=0, max_slant=1800,
                   gap=(120, 700), min_speed=40, max_speed=180):
    """Ground points near (lat, lon) that hear >= 2 distinct low passes `gap` seconds apart."""
    rng = np.random.default_rng(seed)
    frame = LocalFrame(lat, lon, 0.0)
    dlat = (half_m + 5000) / 111_000
    dlon = dlat / np.cos(np.radians(lat))
    tt = world.tracks(t_lo, t_hi, (lat - dlat, lon - dlon, lat + dlat, lon + dlon))
    rs = resample(tt, t_lo, t_hi)
    enu = (rs.xyz - frame.origin) @ frame.R.T
    spd = np.linalg.norm(rs.vel, axis=1)
    found = []
    for e, n in rng.uniform(-half_m, half_m, (n_try, 2)):
        la, lo, _ = frame.from_enu(np.array([e, n, 0.0]))
        gh = float(world.elevation(np.array([la]), np.array([lo]))[0])
        obs = frame.ground_enu(e, n, gh)
        r = np.linalg.norm(enu - obs, axis=1)
        passes = []
        for s in range(len(rs.icao)):
            sl = rs.rows(s)
            i = int(np.argmin(r[sl]))
            if 0 < i < sl.stop - sl.start - 1 and r[sl][i] < max_slant and min_speed < spd[sl][i] < max_speed:
                passes.append((float(rs.t[sl][i]), float(r[sl][i]), float(spd[sl][i]), str(rs.icao[s])))
        passes.sort()
        for a in range(len(passes)):
            for b in range(a + 1, len(passes)):
                if passes[b][3] != passes[a][3] and gap[0] < passes[b][0] - passes[a][0] < gap[1]:
                    found.append(dict(lat=float(la), lon=float(lo), first=passes[a], second=passes[b]))
                    break
            else:
                continue
            break
    return found
