"""Ground elevation models.

`TileTerrain` reads AWS Open Data "Terrarium" elevation tiles
(https://registry.opendata.aws/terrain-tiles/), caching PNG tiles on disk. Heights are
metres above mean sea level; ADS-B geometric altitude is height above the WGS84
ellipsoid, so we add a coarse CONUS geoid undulation (~-28 m) to compare them.
"""
from __future__ import annotations

import io
import os
import threading
import urllib.request

import numpy as np

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
CONUS_GEOID_M = -28.0


class FlatTerrain:
    def __init__(self, h: float = 0.0):
        self.h = float(h)

    def elevation(self, lat, lon):
        return np.full(np.shape(lat), self.h, float)

    def describe(self):
        return f"flat {self.h:g} m"


class TileTerrain:
    def __init__(self, cache_dir: str, zoom: int = 10, offline: bool = False, geoid_m: float = CONUS_GEOID_M):
        self.dir = os.path.join(cache_dir, f"terrarium-z{zoom}")
        os.makedirs(self.dir, exist_ok=True)
        self.z = zoom
        self.offline = offline
        self.geoid = geoid_m
        self._mem = {}
        self._lock = threading.Lock()

    def describe(self):
        return f"terrarium tiles z{self.z}"

    def _tile(self, x, y):
        key = (x, y)
        if key in self._mem:
            return self._mem[key]
        path = os.path.join(self.dir, f"{x}_{y}.png")
        if not os.path.exists(path):
            if self.offline:
                arr = None
            else:
                url = TILE_URL.format(z=self.z, x=x, y=y)
                try:
                    data = urllib.request.urlopen(url, timeout=30).read()
                    tmp = path + ".tmp"
                    with open(tmp, "wb") as fh:
                        fh.write(data)
                    os.replace(tmp, path)
                except OSError:
                    data = None
                arr = self._decode(data) if data else None
        else:
            with open(path, "rb") as fh:
                arr = self._decode(fh.read())
        with self._lock:
            self._mem[key] = arr
        return arr

    @staticmethod
    def _decode(data):
        from PIL import Image

        im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)
        return im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768

    def elevation(self, lat, lon):
        """Ellipsoid height of the ground (m). Missing tiles -> 0 + geoid."""
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        shape = lat.shape
        lat, lon = lat.ravel(), lon.ravel()
        n = 2 ** self.z
        xf = (lon + 180) / 360 * n
        lr = np.radians(np.clip(lat, -85, 85))
        yf = (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / np.pi) / 2 * n
        tx, ty = np.floor(xf).astype(int), np.floor(yf).astype(int)
        px = np.clip(((xf - tx) * 256).astype(int), 0, 255)
        py = np.clip(((yf - ty) * 256).astype(int), 0, 255)
        out = np.zeros(lat.size)
        keys = tx.astype(np.int64) * 100000 + ty
        for k in np.unique(keys):
            m = keys == k
            arr = self._tile(int(k // 100000), int(k % 100000))
            if arr is not None:
                out[m] = arr[py[m], px[m]]
        return (np.maximum(out, -100) + self.geoid).reshape(shape)


def tiles_for_bbox(bbox, zoom):
    lat0, lon0, lat1, lon1 = bbox
    n = 2 ** zoom

    def xy(lat, lon):
        lr = np.radians(lat)
        return int((lon + 180) / 360 * n), int((1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / np.pi) / 2 * n)

    x0, y1 = xy(lat0, lon0)
    x1, y0 = xy(lat1, lon1)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def freeze_tiles(cache_dir: str, out_dir: str, bbox, zoom: int = 8) -> int:
    """Fetch (into the main cache) and copy the terrain tiles covering bbox into out_dir."""
    import shutil

    src = TileTerrain(cache_dir, zoom=zoom)
    dst = os.path.join(out_dir, "terrain", f"terrarium-z{zoom}")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for x, y in tiles_for_bbox(bbox, zoom):
        src._tile(x, y)
        f = os.path.join(src.dir, f"{x}_{y}.png")
        if os.path.exists(f):
            shutil.copy(f, dst)
            n += 1
    return n
