"""Acoustic Doppler model for a moving source and a stationary observer.

Sound received at time t_r was emitted at t_e with t_r = t_e + r(t_e)/c, and its
frequency is f = f0 * c / (c + dr/dt_e). Using emission-time geometry (the
"retarded time") matters: a reception-time model skews the curve by ~1 Hz.
"""
from __future__ import annotations

import numpy as np


def speed_of_sound(temp_c: float = 15.0) -> float:
    return 331.3 * np.sqrt(1 + temp_c / 273.15)


def straight_pass(t_rx, f0, v, d, t_ca, c):
    """Received frequency for a constant-velocity straight flyby.

    t_rx : reception times
    f0   : emitted (rest-frame) tone frequency, Hz
    v    : source speed, m/s
    d    : slant distance at closest approach, m
    t_ca : *reception* time of the sound emitted at closest approach
    """
    t_rx = np.asarray(t_rx, float)
    te0 = t_ca - d / c
    tau = t_rx - te0
    a = c * c - v * v
    # Solve c^2 (tau - u)^2 = d^2 + v^2 u^2 for emission offset u = t_e - te0 (root with u < tau)
    disc = c * c * v * v * tau * tau + d * d * a
    u = (c * c * tau - np.sqrt(disc)) / a
    r = c * (tau - u)
    vr = v * v * u / r
    return f0 * c / (c + vr)


def received_ratio(t_e, pos, obs, c):
    """For a sampled track, return (t_r, r, ratio) where ratio = f_received / f0.

    t_e : (N,) emission times; pos : (N, 3) ENU positions; obs : (3,) observer ENU.
    """
    d = pos - obs
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    vr = np.gradient(r, t_e)
    t_r = t_e + r / c
    return t_r, r, c / (c + vr)
