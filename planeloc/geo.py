"""WGS84 <-> local East/North/Up frame conversions."""
from __future__ import annotations

import numpy as np

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
EARTH_R = 6371000.0


def geodetic_to_ecef(lat, lon, h):
    lat = np.radians(np.asarray(lat, float))
    lon = np.radians(np.asarray(lon, float))
    h = np.asarray(h, float)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - _E2) + h) * np.sin(lat)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(xyz):
    xyz = np.asarray(xyz, float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - _E2))
    for _ in range(6):
        n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - _E2 * n / (n + h)))
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


class LocalFrame:
    """Tangent-plane ENU frame anchored at (lat0, lon0, h0)."""

    def __init__(self, lat0: float, lon0: float, h0: float = 0.0):
        self.lat0, self.lon0, self.h0 = float(lat0), float(lon0), float(h0)
        self.origin = geodetic_to_ecef(lat0, lon0, h0)
        la, lo = np.radians(lat0), np.radians(lon0)
        self.R = np.array(
            [
                [-np.sin(lo), np.cos(lo), 0.0],
                [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
            ]
        )

    def to_enu(self, lat, lon, h):
        return (geodetic_to_ecef(lat, lon, h) - self.origin) @ self.R.T

    def from_enu(self, enu):
        return ecef_to_geodetic(self.origin + np.asarray(enu, float) @ self.R)

    def ground_enu(self, e, n, ground_h: float):
        """ENU of a ground point at horizontal offset (e, n) and ellipsoid height ground_h.

        Uses the spherical-earth curvature drop, accurate to < 1 m within ~50 km.
        """
        e = np.asarray(e, float)
        n = np.asarray(n, float)
        u = (ground_h - self.h0) - (e**2 + n**2) / (2 * EARTH_R)
        return np.stack(np.broadcast_arrays(e, n, u), axis=-1)
