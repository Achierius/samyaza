"""Track table, range cache (write/load/LRU prune/freeze) and world abstraction."""
import os
from datetime import date

import numpy as np

from sya.terrain import FlatTerrain, TileTerrain
from sya.tracks import TrackTable, resample
from sya.world import AdsbCache, CachedWorld, TableWorld, parse_size

T_DAY = 1691452800.0  # 2023-08-08T00:00Z


def _cols(n_ac=5, t0=T_DAY + 3000, dur=4000):
    rng = np.random.default_rng(0)
    ic, ts, la, lo, al = [], [], [], [], []
    for k in range(n_ac):
        t = t0 + np.arange(0, dur, 5.0)
        ic += [f"ab{k:04x}"] * t.size
        ts.append(t)
        la.append(46.8 + 0.001 * k + 0.0003 * np.arange(t.size))
        lo.append(np.full(t.size, -100.8 + 0.01 * k))
        al.append(np.full(t.size, 1000.0 + 100 * k))
    return dict(icao=np.array(ic, dtype=object), t=np.concatenate(ts), lat=np.concatenate(la),
                lon=np.concatenate(lo), alt_m=np.concatenate(al))


class FakeSource:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def covers(self, day):
        return day == date(2023, 8, 8)

    def fetch_day(self, day, bbox, progress=None):
        self.calls += 1
        return _cols()


def test_tracktable_splits_on_gaps_and_icao():
    t = np.r_[np.arange(0, 100, 5.0), np.arange(1000, 1100, 5.0)]
    tt = TrackTable.from_columns(["a"] * t.size + ["b"] * 10, np.r_[t, np.arange(10.0)],
                                 np.zeros(t.size + 10), np.zeros(t.size + 10), np.ones(t.size + 10))
    assert tt.n_segments == 3  # 'a' split by the 900 s gap, plus 'b'
    sub = tt.select(t_lo=1000)
    assert sub.n_segments == 1 and sub.icao[0] == "a"


def test_resample_uniform_grid_and_velocity():
    t = np.arange(0, 300, 7.0)
    lat = 46.8 + 0.0005 * t / 7  # ~55 m per 7 s -> ~7.9 m/s north
    tt = TrackTable.from_columns(["x"] * t.size, t, lat, np.full(t.size, -100.0), np.full(t.size, 1000.0))
    rs = resample(tt, 0, 300, dt=1.0)
    assert np.allclose(np.diff(rs.t), 1.0)
    spd = np.linalg.norm(rs.vel, axis=1)
    assert abs(np.median(spd) - 0.0005 * 111_000 / 7) < 0.3


def test_cache_fetch_once_then_reuse(tmp_path):
    src = FakeSource()
    cache = AdsbCache(str(tmp_path), sources=[src], limit_bytes=10**9)
    tt = cache.load(T_DAY + 3600, T_DAY + 5400)
    assert tt.n_segments == 5 and src.calls == 1
    tt2 = cache.load(T_DAY + 3600, T_DAY + 7200)
    assert src.calls == 1 and tt2.n_rows > tt.n_rows  # second range served from cache
    days = cache.cached_days()
    assert len(days) == 1 and days[0]["complete"]
    # a day no source covers is reported missing, not fetched
    missing = []
    cache.load(T_DAY + 86400, T_DAY + 86400 + 60, on_missing=missing.append)
    assert missing == [date(2023, 8, 9)]


def test_cache_offline_never_fetches(tmp_path):
    src = FakeSource()
    cache = AdsbCache(str(tmp_path), sources=[src], offline=True)
    assert cache.load(T_DAY, T_DAY + 3600).n_rows == 0 and src.calls == 0


def test_cache_prune_lru(tmp_path):
    src = FakeSource()
    cache = AdsbCache(str(tmp_path), sources=[src], limit_bytes=10**9)
    cache.load(T_DAY + 3600, T_DAY + 3700)
    # pretend a second, older day exists
    old = cache.day_dir("fake", date(2023, 8, 1))
    os.makedirs(old)
    with open(os.path.join(old, "00.parquet"), "wb") as fh:
        fh.write(b"x" * 1000)
    os.utime(old, (0, 0))
    removed = cache.prune(limit=1)
    assert any(r["day"] == "2023-08-01" for r in removed)
    assert parse_size("20G") == 20 * 10**9 and parse_size("512M") == 512 * 10**6


def test_freeze_subset_roundtrip(tmp_path):
    cache = AdsbCache(str(tmp_path / "c"), sources=[FakeSource()], limit_bytes=10**9)
    cache.load(T_DAY, T_DAY + 86000)
    n = cache.freeze(str(tmp_path / "frozen"), T_DAY + 3600, T_DAY + 4600, (46.79, -100.81, 46.9, -100.785))
    assert n > 0
    frozen = AdsbCache(str(tmp_path / "frozen"), offline=True)
    tt = frozen.load(T_DAY, T_DAY + 86400)
    assert 0 < tt.n_rows == n and set(tt.icao) <= {"ab0000", "ab0001"}


def test_worlds_share_interface(tmp_path):
    tt = TrackTable.from_columns(*(_cols()[k] for k in ("icao", "t", "lat", "lon", "alt_m")))
    w1 = TableWorld("mem", tt, FlatTerrain(500))
    cache = AdsbCache(str(tmp_path), sources=[FakeSource()], limit_bytes=10**9)
    w2 = CachedWorld(cache, FlatTerrain(500))
    a = w1.tracks(T_DAY + 3600, T_DAY + 4000)
    b = w2.tracks(T_DAY + 3600, T_DAY + 4000)
    assert a.n_rows == b.n_rows and a.n_segments == b.n_segments
    assert w1.elevation(np.array([1.0]), np.array([2.0]))[0] == 500


def test_tile_terrain_offline_missing_tile_is_sea_level(tmp_path):
    ter = TileTerrain(str(tmp_path), offline=True)
    assert abs(ter.elevation(np.array([46.8]), np.array([-100.8]))[0] - ter.geoid) < 1e-9


def test_tile_terrain_empty_input(tmp_path):
    ter = TileTerrain(str(tmp_path), offline=True)
    assert ter.elevation(np.zeros(0), np.zeros(0)).shape == (0,)
