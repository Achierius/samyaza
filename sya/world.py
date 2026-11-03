"""World data: where aircraft were (ADS-B) and how high the ground is.

A `World` couples an ADS-B provider with a terrain model. Real worlds read historical
ADS-B through an on-disk range cache; synthetic worlds generate everything in memory,
so the search pipeline cannot tell them apart.

Cache layout (one parquet file per UTC hour, CONUS-filtered):
    <cache>/adsb/<source>/<YYYY-MM-DD>/<HH>.parquet
    <cache>/adsb/<source>/<YYYY-MM-DD>/_COMPLETE      (json; written after a full day fetch)
    <cache>/adsb/<source>/<YYYY-MM-DD>/_LAST_USED     (touched on every read; drives LRU pruning)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .terrain import FlatTerrain, TileTerrain
from .tracks import TrackTable

CONUS_FETCH_BBOX = (24.0, -125.5, 49.6, -66.5)  # generous: includes approaches from coast/borders
FT = 0.3048


def default_cache_dir() -> str:
    return os.environ.get("SYA_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sya")


def default_cache_limit(cache_dir: str) -> int:
    """Smart default: SYA_CACHE_LIMIT (e.g. '20G') or min(20 GB, 40% of free disk)."""
    env = os.environ.get("SYA_CACHE_LIMIT")
    if env:
        return parse_size(env)
    os.makedirs(cache_dir, exist_ok=True)
    free = shutil.disk_usage(cache_dir).free
    return int(min(20e9, 0.4 * (free + dir_size(cache_dir))))


def parse_size(s: str) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", s.upper())
    if not m:
        raise ValueError(f"bad size {s!r}")
    return int(float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}[m.group(2)])


def dir_size(path: str) -> int:
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def _days(t_lo: float, t_hi: float):
    d = datetime.fromtimestamp(t_lo, timezone.utc).date()
    end = datetime.fromtimestamp(t_hi, timezone.utc).date()
    while d <= end:
        yield d
        d += timedelta(days=1)


# --------------------------------------------------------------------------- sources
class HFAdsbLol:
    """adsb.lol globe_history daily dumps mirrored as parquet on Hugging Face
    (dataset alexisplacet/adsblol_globe_history, ODbL). One day = 256 shards by ICAO."""

    name = "adsblol"
    repo = "alexisplacet/adsblol_globe_history"
    api = "https://huggingface.co/api/datasets/{repo}/tree/main"
    file_url = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._dirs = None

    def _day_dirs(self):
        if self._dirs is not None:
            return self._dirs
        p = os.path.join(self.cache_dir, "adsb", self.name, "_dirs.json")
        if os.path.exists(p) and time.time() - os.path.getmtime(p) < 86400 * 7:
            self._dirs = json.load(open(p))
            return self._dirs
        url = self.api.format(repo=self.repo) + "?limit=1000"
        dirs = []
        while url:
            with urllib.request.urlopen(url, timeout=60) as r:
                dirs += [x["path"] for x in json.load(r) if x["type"] == "directory"]
                m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
                url = m.group(1) if m else None
        out = {}
        for d in dirs:
            m = re.match(r"v(\d{4})\.(\d{2})\.(\d{2})-", d)
            if m:
                out.setdefault(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", d)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(out, open(p, "w"))
        self._dirs = out
        return out

    def covers(self, day: date) -> bool:
        try:
            return day.isoformat() in self._day_dirs()
        except OSError:
            return False

    def fetch_day(self, day: date, bbox, progress=None, workers: int = 8):
        """Download all shards for `day`, keep airborne rows inside bbox. Returns dict of columns."""
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        d = self._day_dirs()[day.isoformat()]
        tmpdir = os.path.join(self.cache_dir, "adsb", self.name, "_tmp", day.isoformat())
        os.makedirs(tmpdir, exist_ok=True)
        shards = [f"{d}/traces_{i:02x}.parquet" for i in range(256)]

        def one(path):
            local = os.path.join(tmpdir, os.path.basename(path))
            if not os.path.exists(local):
                for attempt in range(4):
                    try:
                        with urllib.request.urlopen(self.file_url.format(repo=self.repo, path=path), timeout=120) as r:
                            data = r.read()
                        with open(local + ".part", "wb") as fh:
                            fh.write(data)
                        os.replace(local + ".part", local)
                        break
                    except OSError:
                        if attempt == 3:
                            raise
                        time.sleep(2 ** (attempt + 1))
            t = pq.read_table(local, columns=["icao", "timestamp", "lat", "lon", "altitude", "alt_geom", "on_ground"])
            m = pc.and_(pc.and_(pc.greater_equal(t["lat"], bbox[0]), pc.less_equal(t["lat"], bbox[2])),
                        pc.and_(pc.greater_equal(t["lon"], bbox[1]), pc.less_equal(t["lon"], bbox[3])))
            m = pc.and_(m, pc.invert(pc.fill_null(t["on_ground"], False)))
            t = t.filter(m)
            icao = np.array([b.hex() for b in t["icao"].to_pylist()], dtype=object)
            ts = t["timestamp"].cast("int64").to_numpy() / 1e6
            ag = t["alt_geom"].to_numpy(zero_copy_only=False).astype(float)
            ab = t["altitude"].to_numpy(zero_copy_only=False).astype(float)
            alt = np.where(np.isfinite(ag), ag, ab) * FT
            os.remove(local)
            return dict(icao=icao, t=ts, lat=t["lat"].to_numpy().astype(float), lon=t["lon"].to_numpy().astype(float),
                        alt_m=alt)

        parts = []
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(one, s) for s in shards]
            for f in as_completed(futs):
                parts.append(f.result())
                if progress:
                    progress(1, len(shards))
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ------------------------------------------------------------------------- cache
class AdsbCache:
    def __init__(self, cache_dir: str | None = None, sources=None, offline: bool = False,
                 limit_bytes: int | None = None, readonly: bool = False):
        self.dir = cache_dir or default_cache_dir()
        self.offline = offline
        self.readonly = readonly
        self.sources = sources if sources is not None else [HFAdsbLol(self.dir)]
        self._limit = limit_bytes

    @property
    def limit(self):
        if self._limit is None:
            self._limit = default_cache_limit(self.dir)
        return self._limit

    def day_dir(self, source: str, day: date):
        return os.path.join(self.dir, "adsb", source, day.isoformat())

    def cached_days(self):
        root = os.path.join(self.dir, "adsb")
        out = []
        if not os.path.isdir(root):
            return out
        for src in sorted(os.listdir(root)):
            sd = os.path.join(root, src)
            if not os.path.isdir(sd):
                continue
            for d in sorted(os.listdir(sd)):
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                    p = os.path.join(sd, d)
                    lu = os.path.join(p, "_LAST_USED")
                    out.append(dict(source=src, day=d, path=p, bytes=dir_size(p),
                                    complete=os.path.exists(os.path.join(p, "_COMPLETE")),
                                    hours=sorted(f[:2] for f in os.listdir(p) if f.endswith(".parquet")),
                                    last_used=os.path.getmtime(lu) if os.path.exists(lu) else os.path.getmtime(p)))
        return out

    def _find_day(self, day: date):
        root = os.path.join(self.dir, "adsb")
        if os.path.isdir(root):
            for src in sorted(os.listdir(root)):
                p = os.path.join(root, src, day.isoformat())
                if os.path.isdir(p) and any(f.endswith(".parquet") for f in os.listdir(p)):
                    return p
        return None

    def ensure_day(self, day: date, progress=None):
        p = self._find_day(day)
        if p is not None:
            return p
        if self.offline:
            return None
        for src in self.sources:
            if src.covers(day):
                cols = src.fetch_day(day, CONUS_FETCH_BBOX, progress=progress)
                p = self.day_dir(src.name, day)
                self._write_day(p, cols, day)
                self.prune(keep={p})
                return p
        return None

    def _write_day(self, p, cols, day):
        import pyarrow as pa
        import pyarrow.parquet as pq

        os.makedirs(p, exist_ok=True)
        t0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
        hour = np.floor((cols["t"] - t0) / 3600).astype(int)
        for h in range(24):
            m = hour == h
            tbl = pa.table(dict(icao=pa.array(cols["icao"][m].astype(str)), t=cols["t"][m],
                                lat=cols["lat"][m].astype(np.float32), lon=cols["lon"][m].astype(np.float32),
                                alt_m=cols["alt_m"][m].astype(np.float32)))
            pq.write_table(tbl, os.path.join(p, f"{h:02d}.parquet"), compression="zstd")
        json.dump(dict(rows=int(len(cols["t"])), fetched=time.time()), open(os.path.join(p, "_COMPLETE"), "w"))

    def prune(self, keep=frozenset(), limit=None):
        limit = self.limit if limit is None else limit
        days = sorted(self.cached_days(), key=lambda d: d["last_used"])
        total = sum(d["bytes"] for d in days)
        removed = []
        for d in days:
            if total <= limit:
                break
            if d["path"] in keep:
                continue
            shutil.rmtree(d["path"], ignore_errors=True)
            total -= d["bytes"]
            removed.append(d)
        return removed

    def load(self, t_lo: float, t_hi: float, bbox=None, progress=None, on_missing=None) -> TrackTable:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        tables = []
        for day in _days(t_lo, t_hi):
            p = self.ensure_day(day, progress=progress)
            if p is None:
                if on_missing:
                    on_missing(day)
                continue
            if not self.readonly:
                try:
                    with open(os.path.join(p, "_LAST_USED"), "w") as fh:
                        fh.write(str(time.time()))
                except OSError:
                    pass
            d0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
            for h in range(24):
                a, b = d0 + 3600 * h, d0 + 3600 * (h + 1)
                f = os.path.join(p, f"{h:02d}.parquet")
                if b < t_lo or a > t_hi or not os.path.exists(f):
                    continue
                t = pq.read_table(f)
                m = pc.and_(pc.greater_equal(t["t"], t_lo), pc.less_equal(t["t"], t_hi))
                if bbox is not None:
                    m = pc.and_(m, pc.and_(pc.and_(pc.greater_equal(t["lat"], bbox[0]), pc.less_equal(t["lat"], bbox[2])),
                                           pc.and_(pc.greater_equal(t["lon"], bbox[1]), pc.less_equal(t["lon"], bbox[3]))))
                tables.append(t.filter(m))
        if not tables:
            return TrackTable.empty()
        import pyarrow as pa

        t = pa.concat_tables(tables)
        return TrackTable.from_columns(t["icao"].to_numpy(zero_copy_only=False), t["t"].to_numpy(),
                                       t["lat"].to_numpy().astype(float), t["lon"].to_numpy().astype(float),
                                       t["alt_m"].to_numpy().astype(float))

    def freeze(self, out_dir: str, t_lo: float, t_hi: float, bbox) -> int:
        """Copy a small (time, bbox) subset of the cache into `out_dir` in the same layout."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        n = 0
        for day in _days(t_lo, t_hi):
            p = self._find_day(day)
            if p is None:
                continue
            src = os.path.basename(os.path.dirname(p))
            od = os.path.join(out_dir, "adsb", src, day.isoformat())
            os.makedirs(od, exist_ok=True)
            d0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
            for h in range(24):
                a, b = d0 + 3600 * h, d0 + 3600 * (h + 1)
                if b < t_lo or a > t_hi:
                    continue
                tt = self.load(max(a, t_lo), min(b, t_hi), bbox)
                seg = np.repeat(np.arange(tt.n_segments), np.diff(tt.offsets))
                tbl = pa.table(dict(icao=pa.array(tt.icao[seg].astype(str)), t=tt.t,
                                    lat=tt.lat.astype(np.float32), lon=tt.lon.astype(np.float32),
                                    alt_m=tt.alt_m.astype(np.float32)))
                f = os.path.join(od, f"{h:02d}.parquet")
                if os.path.exists(f):  # merge with an existing frozen subset
                    tbl = pa.concat_tables([pq.read_table(f), tbl])
                pq.write_table(tbl, f, compression="zstd")
                n += tbl.num_rows
        return n


# ------------------------------------------------------------------------- worlds
@dataclass
class World:
    """What the search needs: ADS-B tracks for a time/bbox, and ground elevation."""

    name: str
    terrain: object

    def tracks(self, t_lo, t_hi, bbox=None, progress=None, on_missing=None) -> TrackTable:  # pragma: no cover
        raise NotImplementedError

    def elevation(self, lat, lon):
        return self.terrain.elevation(lat, lon)


class CachedWorld(World):
    def __init__(self, cache: AdsbCache, terrain):
        super().__init__(f"adsb-cache:{cache.dir}", terrain)
        self.cache = cache

    def tracks(self, t_lo, t_hi, bbox=None, progress=None, on_missing=None):
        return self.cache.load(t_lo, t_hi, bbox, progress=progress, on_missing=on_missing)


class TableWorld(World):
    """Tracks held in memory (CSV/readsb files, or synthetic)."""

    def __init__(self, name: str, table: TrackTable, terrain):
        super().__init__(name, terrain)
        self.table = table

    def tracks(self, t_lo, t_hi, bbox=None, progress=None, on_missing=None):
        return self.table.select(t_lo, t_hi, bbox)


def make_real_world(cache_dir=None, offline=False, terrain="tiles", limit=None) -> CachedWorld:
    cache = AdsbCache(cache_dir, offline=offline, limit_bytes=limit)
    if isinstance(terrain, str) and terrain.startswith("flat:"):
        ter = FlatTerrain(float(terrain[5:]))
    else:
        ter = TileTerrain(os.path.join(cache.dir, "terrain"), offline=offline)
    return CachedWorld(cache, ter)
