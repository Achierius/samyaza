"""Synthetic world: aircraft tracks, the ADS-B a receiver network would log, and the
audio a hidden microphone would record. Used to validate the pipeline end to end.

Synthetic worlds are addressed by a spec string so CLI/tests can reproduce them:
    synth:<preset>[:<seed>[:single]]   e.g. synth:bismarck:3, synth:phoenix, synth:fargo:2:single
("single" keeps only one guaranteed flyover, which leaves the location ambiguous.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .geo import LocalFrame
from .physics import speed_of_sound
from .terrain import FlatTerrain
from .tracks import TrackTable
from .world import FT, TableWorld

# observer neighbourhoods (lat, lon, ground ellipsoid height m) + a fixed UTC day
PRESETS = {
    "bismarck": (46.80, -100.76, 490.0, "2023-08-15T19:00:00Z"),
    "fargo": (46.88, -96.82, 245.0, "2023-06-07T17:00:00Z"),
    "phoenix": (33.45, -112.05, 310.0, "2023-09-28T18:00:00Z"),
    "flagstaff": (35.20, -111.63, 2080.0, "2023-08-08T19:00:00Z"),
    "seattle": (47.45, -122.10, 60.0, "2023-07-01T19:00:00Z"),
}


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
    return np.concatenate([tb[::-1], tf[1:]]), np.concatenate([pb[::-1], pf[1:]])


@dataclass
class SynthWorld:
    spec: str
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

    @property
    def obs_latlon(self):
        lat, lon, _ = self.frame.from_enu(self.obs)
        return float(lat), float(lon)


def parse_spec(spec: str):
    parts = spec.split(":")
    if parts[0] != "synth" or len(parts) < 2:
        raise ValueError("synthetic world spec must look like synth:<preset>[:<seed>]")
    preset = parts[1]
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    seed = int(parts[2]) if len(parts) > 2 else 0
    variant = parts[3] if len(parts) > 3 else ""
    if variant not in ("", "single"):
        raise ValueError(f"unknown variant {variant!r} (supported: single)")
    return preset, seed, variant


def make_world(spec: str = "synth:seattle:0", dur=900.0, n_background=150,
               targets=((200.0, 1100.0, 85.0, 0.25), (640.0, 1500.0, 160.0, 0.0)), temp_c=15.0,
               area_m=90000.0, pad_s=3 * 3600) -> SynthWorld:
    """targets: (t_cpa_rel, slant_range_m, speed_m_s, turn_deg_s) of guaranteed flyovers."""
    from .timespec import parse_timestamp

    preset, seed, variant = parse_spec(spec)
    if variant == "single":
        targets = targets[:1]
    lat, lon, ground_h, t_ref = PRESETS[preset]
    rng = np.random.default_rng(seed)
    frame = LocalFrame(lat, lon, 0.0)
    obs_en = tuple(rng.uniform(-4000, 4000, 2))
    T0 = parse_timestamp(t_ref) + float(rng.integers(-3 * 3600, 3 * 3600))
    c = speed_of_sound(temp_c)
    obs = frame.ground_enu(*obs_en, ground_h)
    t_lo, t_hi = T0 - pad_s, T0 + dur + pad_s
    flights = []

    for k, (tc, slant, v, turn) in enumerate(targets):
        alt = rng.uniform(0.35, 0.8) * slant
        horiz = np.sqrt(slant**2 - alt**2)
        hdg = rng.uniform(0, 360)
        perp = np.radians(hdg + 90 * rng.choice([-1, 1]))
        p_ref = obs + [horiz * np.sin(perp), horiz * np.cos(perp), alt]
        t_e = T0 + tc - slant / c
        t, p = _fly(p_ref, t_e, hdg, v, turn, 0.0, t_e - 900, t_e + 900, 0.05)
        flights.append(Flight(f"a{seed:02x}{k:03x}", t, p, rng.uniform(85, 140), 0.4, (1.0, 0.6, 0.35)))

    half = area_m / 2
    for k in range(n_background):
        ang = rng.uniform(0, 2 * np.pi)
        start = np.array([half * np.cos(ang), half * np.sin(ang)])
        hdg = np.degrees(np.arctan2(-start[0], -start[1])) + rng.uniform(-40, 40)
        kind = rng.random()
        if kind < 0.45:  # airliners at altitude
            v, alt, loud = rng.uniform(190, 250), rng.uniform(7000, 11500), 0.8
        elif kind < 0.65:  # arrivals/departures, mid altitude
            v, alt, loud = rng.uniform(110, 170), rng.uniform(1500, 5000), 0.8
        else:  # general aviation, low and slow
            v, alt, loud = rng.uniform(45, 85), rng.uniform(300, 2500), 0.4
        ts = rng.uniform(t_lo - 1800, t_hi)
        p_ref = np.array([start[0], start[1], alt + ground_h])
        t, p = _fly(p_ref, ts, hdg, v, rng.uniform(-0.08, 0.08), 0.0, ts, ts + 2 * area_m / v, 0.5)
        flights.append(Flight(f"b{seed:02x}{k:03x}", t, p, rng.uniform(70, 180), loud,
                              (1.0, rng.uniform(0.3, 0.8), rng.uniform(0.1, 0.5))))
    return SynthWorld(spec, frame, ground_h, obs_en, T0, dur, flights, c)


def adsb_table(world: SynthWorld, seed=1, pos_noise_m=5.0, min_dt=1.0, max_dt=8.0) -> TrackTable:
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("icao", "t", "lat", "lon", "alt_m")}
    for fl in world.flights:
        n = int((fl.t[-1] - fl.t[0]) / min_dt) + 2
        ts = fl.t[0] + np.cumsum(rng.uniform(min_dt, max_dt, n))
        ts = ts[ts < fl.t[-1]]
        p = np.stack([np.interp(ts, fl.t, fl.p[:, k]) for k in range(3)], 1)
        p[:, :2] += rng.normal(0, pos_noise_m, (ts.size, 2))
        lat, lon, h = world.frame.from_enu(p)
        h = np.round(h / (25 * FT)) * 25 * FT  # 25 ft altitude quantisation
        cols["icao"].append(np.full(ts.size, fl.icao, dtype=object))
        cols["t"].append(ts)
        cols["lat"].append(lat)
        cols["lon"].append(lon)
        cols["alt_m"].append(h)
    return TrackTable.from_columns(*(np.concatenate(cols[k]) for k in ("icao", "t", "lat", "lon", "alt_m")))


def synth_world_data(world: SynthWorld) -> TableWorld:
    return TableWorld(world.spec, adsb_table(world), FlatTerrain(world.ground_h))


def render_audio(world: SynthWorld, fs=4000, noise=1.0, seed=2, max_range=6000.0):
    rng = np.random.default_rng(seed)
    n = int(world.dur * fs)
    t_rx = world.T0 + np.arange(n) / fs
    x = rng.normal(0, noise, n)
    x += 0.05 * noise * lfilter([1.0], [1.0, -0.999], rng.normal(0, 1, n))  # low-frequency rumble
    obs = world.obs
    for fl in world.flights:
        r = np.linalg.norm(fl.p - obs, axis=1)
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
    return (x / np.max(np.abs(x)) * 0.9).astype(np.float32), fs
