"""Synthetic world: aircraft tracks, the ADS-B a receiver network would log, and the
audio a hidden microphone would record. Used to validate the pipeline end to end."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .adsb import FT, Track
from .geo import LocalFrame
from .physics import speed_of_sound


@dataclass
class Flight:
    icao: str
    t: np.ndarray
    p: np.ndarray  # (N, 3) ENU truth
    f0: float
    loud: float  # tone amplitude at 1 km
    harm: tuple


def _fly(p_ref, t_ref, heading_deg, speed, turn_dps, climb, t0, t1, dt):
    """Integrate a constant-turn-rate trajectory through p_ref at t_ref, over [t0, t1]."""
    def leg(direction, T):
        n = int(np.ceil(abs(T) / dt)) + 1
        ts = np.arange(n) * dt * direction
        hdg = np.radians(heading_deg + turn_dps * ts)
        e = p_ref[0] + np.concatenate([[0], np.cumsum(speed * np.sin(hdg[:-1]) * dt * direction)])
        nn = p_ref[1] + np.concatenate([[0], np.cumsum(speed * np.cos(hdg[:-1]) * dt * direction)])
        u = p_ref[2] + climb * ts
        return t_ref + ts, np.stack([e, nn, u], 1)

    tb, pb = leg(-1, t_ref - t0)
    tf, pf = leg(+1, t1 - t_ref)
    t = np.concatenate([tb[::-1], tf[1:]])
    p = np.concatenate([pb[::-1], pf[1:]])
    return t, p


@dataclass
class World:
    frame: LocalFrame
    ground_h: float
    obs_en: tuple
    T0: float
    dur: float
    flights: list
    c: float

    @property
    def obs(self):
        return self.frame.ground_enu(*self.obs_en, self.ground_h)


def make_world(seed=0, center=(47.45, -122.10), ground_h=60.0, dur=900.0, n_background=60,
               targets=((200.0, 1100.0, 85.0, 0.25), (640.0, 1500.0, 160.0, 0.0)), temp_c=15.0):
    """targets: (t_cpa_rel, slant_range_m, speed_m_s, turn_deg_s) of guaranteed flyovers."""
    rng = np.random.default_rng(seed)
    frame = LocalFrame(center[0], center[1], 0.0)
    obs_en = tuple(rng.uniform(-4000, 4000, 2))
    T0 = 1_750_000_000.0 + float(rng.integers(0, 86400))
    c = speed_of_sound(temp_c)
    obs = frame.ground_enu(*obs_en, ground_h)
    t_lo, t_hi = T0 - 4000, T0 + dur + 4000
    flights = []
    dt = 0.05

    for k, (tc, slant, v, turn) in enumerate(targets):
        alt = rng.uniform(0.35, 0.8) * slant
        horiz = np.sqrt(slant**2 - alt**2)
        hdg = rng.uniform(0, 360)
        side = rng.choice([-1, 1])
        perp = np.radians(hdg + 90 * side)
        p_ref = obs + [horiz * np.sin(perp), horiz * np.cos(perp), alt]
        t_e = T0 + tc - slant / c
        t, p = _fly(p_ref, t_e, hdg, v, turn, 0.0, t_e - 600, t_e + 600, dt)
        flights.append(Flight(f"tgt{k:03d}", t, p, rng.uniform(85, 140), 0.4, (1.0, 0.6, 0.35)))

    for k in range(n_background):
        ang = rng.uniform(0, 2 * np.pi)
        start = np.array([45000 * np.cos(ang), 45000 * np.sin(ang)])
        hdg = np.degrees(np.arctan2(-start[0], -start[1])) + rng.uniform(-35, 35)
        jet = rng.random() < 0.7
        v = rng.uniform(140, 240) if jet else rng.uniform(55, 90)
        alt = rng.uniform(3000, 11500) if jet else rng.uniform(500, 3000)
        ts = rng.uniform(t_lo, t_hi - 600)
        p_ref = np.array([start[0], start[1], alt + ground_h])
        t, p = _fly(p_ref, ts, hdg, v, rng.uniform(-0.05, 0.05), 0.0, ts, ts + 90000 / v, 0.5)
        flights.append(Flight(f"bg{k:04x}", t, p, rng.uniform(70, 180), 0.8 if jet else 0.4,
                              (1.0, rng.uniform(0.3, 0.8), rng.uniform(0.1, 0.5))))
    return World(frame, ground_h, obs_en, T0, dur, flights, c)


def adsb_tracks(world: World, seed=1, pos_noise_m=5.0, min_dt=1.0, max_dt=8.0) -> list[Track]:
    rng = np.random.default_rng(seed)
    out = []
    for fl in world.flights:
        ts = [fl.t[0]]
        while ts[-1] < fl.t[-1]:
            ts.append(ts[-1] + rng.uniform(min_dt, max_dt))
        ts = np.array(ts[:-1])
        p = np.stack([np.interp(ts, fl.t, fl.p[:, k]) for k in range(3)], 1)
        p[:, :2] += rng.normal(0, pos_noise_m, (ts.size, 2))
        lat, lon, h = world.frame.from_enu(p)
        h = np.round(h / (25 * FT)) * 25 * FT  # 25 ft altitude quantisation
        out.append(Track(fl.icao, ts, lat, lon, h))
    return out


def render_audio(world: World, fs=4000, noise=1.0, seed=2, max_range=6000.0):
    rng = np.random.default_rng(seed)
    n = int(world.dur * fs)
    t_rx = world.T0 + np.arange(n) / fs
    x = rng.normal(0, noise, n)
    # low-frequency rumble (wind / traffic), 1/f-ish
    x += 0.05 * noise * lfilter([1.0], [1.0, -0.999], rng.normal(0, 1, n))
    obs = world.obs
    for fl in world.flights:
        d = fl.p - obs
        r = np.linalg.norm(d, axis=1)
        tr = fl.t + r / world.c
        m = (tr >= t_rx[0] - 5) & (tr <= t_rx[-1] + 5)
        if m.sum() < 4 or r[m].min() > max_range:
            continue
        te = np.interp(t_rx, tr[m], fl.t[m], left=np.nan, right=np.nan)
        rr = np.interp(t_rx, tr[m], r[m], left=np.nan, right=np.nan)
        ok = np.isfinite(te)
        amp = fl.loud * 1000.0 / rr[ok]
        sig = np.zeros(ok.sum())
        for h, a in enumerate(fl.harm, start=1):
            sig += a * np.sin(2 * np.pi * h * fl.f0 * (te[ok] - world.T0) + rng.uniform(0, 2 * np.pi))
        x[ok] += amp * sig
    return x / np.max(np.abs(x)) * 0.9, fs
