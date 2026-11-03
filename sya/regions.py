"""Search-area definitions: CONUS (default), a single state, or a radius around a point."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
from matplotlib.path import Path

STATE_NAMES = {}  # filled lazily: name.lower() -> abbreviation
_RASTERS = {}


@lru_cache(maxsize=1)
def _states():
    with resources.files("sya.data").joinpath("states.json").open() as fh:
        d = json.load(fh)["states"]
    out = {}
    for ab, s in d.items():
        paths = [Path(np.asarray(r, float)) for r in s["rings"] if len(r) >= 3]
        rings = [np.asarray(r, float) for r in s["rings"] if len(r) >= 3]
        allpts = np.concatenate(rings)
        bbox = (allpts[:, 1].min(), allpts[:, 0].min(), allpts[:, 1].max(), allpts[:, 0].max())
        out[ab] = dict(name=s["name"], paths=paths, bbox=bbox)
        STATE_NAMES[s["name"].lower()] = ab
    return out


def state_code(s: str) -> str:
    st = _states()
    k = s.strip()
    if k.upper() in st:
        return k.upper()
    ab = STATE_NAMES.get(k.lower())
    if ab is None:
        raise ValueError(f"unknown CONUS state {s!r}; use a 2-letter code like AZ or ND")
    return ab


def _in_paths(paths, lat, lon):
    pts = np.column_stack([lon, lat])
    inside = np.zeros(len(pts), bool)
    for p in paths:
        inside ^= p.contains_points(pts)  # even-odd handles holes / multipolygons
    return inside


@dataclass
class Region:
    """Area the recording is assumed to be inside."""

    kind: str  # "conus" | "state" | "radius"
    label: str
    bbox: tuple  # (lat_min, lon_min, lat_max, lon_max)
    states: tuple = ()
    center: tuple | None = None
    radius_m: float | None = None

    def contains(self, lat, lon) -> np.ndarray:
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        b = self.bbox
        ok = (lat >= b[0]) & (lat <= b[2]) & (lon >= b[1]) & (lon <= b[3])
        if self.kind == "radius":
            ok &= haversine_m(lat, lon, *self.center) <= self.radius_m
            return ok
        res = np.zeros(lat.shape, bool)
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            return res
        if idx.size > 5000:
            return self._contains_raster(lat, lon, idx, res)
        st = _states()
        for ab in self.states:
            s = st[ab]
            sb = s["bbox"]
            sub = idx[(lat[idx] >= sb[0]) & (lat[idx] <= sb[2]) & (lon[idx] >= sb[1]) & (lon[idx] <= sb[3])]
            sub = sub[~res[sub]]
            if sub.size:
                res[sub] = _in_paths(s["paths"], lat[sub], lon[sub])
        return res

    def _contains_raster(self, lat, lon, idx, res, cell=0.05):
        """Fast path: classify by a cached raster; exact polygon test only in border cells."""
        key = (self.states, cell)
        if key not in _RASTERS:
            b = self.bbox
            la = np.arange(b[0], b[2] + cell, cell)
            lo = np.arange(b[1], b[3] + cell, cell)
            LA, LO = np.meshgrid(la + cell / 2, lo + cell / 2, indexing="ij")
            sub = Region(self.kind, self.label, self.bbox, self.states)
            inside = sub._contains_exact(LA.ravel(), LO.ravel()).reshape(LA.shape)
            pad = np.pad(inside, 1, mode="edge")
            border = np.zeros_like(inside)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    border |= pad[1 + dy:1 + dy + inside.shape[0], 1 + dx:1 + dx + inside.shape[1]] != inside
            _RASTERS[key] = (la[0], lo[0], inside, border)
        la0, lo0, inside, border = _RASTERS[key]
        i = np.clip(((lat[idx] - la0) / cell).astype(int), 0, inside.shape[0] - 1)
        j = np.clip(((lon[idx] - lo0) / cell).astype(int), 0, inside.shape[1] - 1)
        res[idx] = inside[i, j]
        bd = border[i, j]
        if bd.any():
            res[idx[bd]] = self._contains_exact(lat[idx[bd]], lon[idx[bd]])
        return res

    def _contains_exact(self, lat, lon):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        res = np.zeros(lat.shape, bool)
        st = _states()
        idx = np.arange(lat.size)
        for ab in self.states:
            s = st[ab]
            sb = s["bbox"]
            sub = idx[(lat >= sb[0]) & (lat <= sb[2]) & (lon >= sb[1]) & (lon <= sb[3])]
            sub = sub[~res[sub]]
            if sub.size:
                res[sub] = _in_paths(s["paths"], lat[sub], lon[sub])
        return res

    def padded_bbox(self, pad_m: float):
        dlat = pad_m / 111_000
        dlon = pad_m / (111_000 * np.cos(np.radians(max(abs(self.bbox[0]), abs(self.bbox[2])))))
        b = self.bbox
        return (b[0] - dlat, b[1] - dlon, b[2] + dlat, b[3] + dlon)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def conus() -> Region:
    st = _states()
    bb = np.array([s["bbox"] for s in st.values()])
    return Region("conus", "Continental US", (bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max()),
                  states=tuple(sorted(st)))


def state(s: str) -> Region:
    ab = state_code(s)
    st = _states()[ab]
    return Region("state", f"{st['name']} ({ab})", st["bbox"], states=(ab,))


def radius(lat: float, lon: float, radius_m: float) -> Region:
    dlat = radius_m / 111_000
    dlon = radius_m / (111_000 * max(np.cos(np.radians(lat)), 1e-3))
    reg = Region("radius", f"{radius_m / 1000:g} km around {lat:.4f},{lon:.4f}",
                 (lat - dlat, lon - dlon, lat + dlat, lon + dlon), center=(lat, lon), radius_m=radius_m)
    cz = conus()
    if not cz.contains(lat, lon)[0]:
        raise ValueError("center must be inside the continental US")
    return reg


def which_state(lat: float, lon: float) -> str | None:
    for ab, s in _states().items():
        b = s["bbox"]
        if b[0] <= lat <= b[2] and b[1] <= lon <= b[3] and _in_paths(s["paths"], [lat], [lon])[0]:
            return ab
    return None
