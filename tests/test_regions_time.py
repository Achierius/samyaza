import numpy as np
import pytest

from sya import regions
from sya.timespec import (at_window, between_window, dates_window, parse_duration, parse_timestamp,
                          parse_when)


@pytest.mark.parametrize("lat,lon,st", [
    (46.8083, -100.7837, "ND"),   # Bismarck
    (46.8772, -96.7898, "ND"),    # Fargo
    (33.4484, -112.0740, "AZ"),   # Phoenix
    (35.1983, -111.6513, "AZ"),   # Flagstaff
    (39.7392, -104.9903, "CO"),
    (40.7128, -74.0060, "NY"),
])
def test_which_state(lat, lon, st):
    assert regions.which_state(lat, lon) == st


def test_conus_contains_and_excludes():
    c = regions.conus()
    inside = c.contains([46.8, 33.45, 39.74], [-100.78, -112.07, -104.99])
    assert inside.all()
    out = c.contains([61.2, 21.3, 19.4, 45.0], [-149.9, -157.8, -99.1, -40.0])  # AK, HI, Mexico City, Atlantic
    assert not out.any()


def test_state_region_by_code_and_name():
    a = regions.state("az")
    b = regions.state("Arizona")
    assert a.states == b.states == ("AZ",)
    assert a.contains(33.45, -112.07)[0] and not a.contains(46.8, -100.78)[0]
    with pytest.raises(ValueError):
        regions.state("Alaska")  # not CONUS


def test_radius_region():
    r = regions.radius(46.8, -100.78, 10_000)
    assert r.contains(46.8, -100.78)[0]
    assert r.contains(46.88, -100.78)[0]          # ~8.9 km north
    assert not r.contains(46.92, -100.78)[0]      # ~13 km north
    with pytest.raises(ValueError):
        regions.radius(61.2, -149.9, 1000)        # Anchorage: outside CONUS


def test_haversine():
    assert abs(regions.haversine_m(0, 0, 0, 1) - 111_195) < 50


def test_durations_and_timestamps():
    assert parse_duration("90") == 90 and parse_duration("15m") == 900 and parse_duration("2h") == 7200
    assert parse_timestamp("2023-08-08T00:00:00Z") == 1691452800
    assert parse_timestamp("2023-08-08T00:00", "America/Phoenix") == 1691452800 + 7 * 3600
    assert parse_timestamp("1691452800") == 1691452800


def test_windows():
    w = parse_when("2023-08-08")
    assert w.span == 86400 and w.t_lo == 1691452800
    w = dates_window("2023-08-07", "2023-08-09", "America/Chicago")
    assert w.span == 3 * 86400 and w.t_lo == 1691366400 + 5 * 3600
    w = parse_when("2023-08-07..2023-08-08")
    assert w.span == 2 * 86400
    w = parse_when("2023-08-08T10:00Z..2023-08-08T12:00Z")
    assert w.span == 7200
    w = between_window("2023-08-08T10:00", "2023-08-08T12:30", "UTC")
    assert w.span == 9000
    w = at_window("2023-08-08T10:00Z", "10m")
    assert w.span == 1200
    with pytest.raises(ValueError):
        between_window("2023-08-08T12:00Z", "2023-08-08T10:00Z")
    chunks = list(parse_when("2023-08-08").chunks(3600))
    assert len(chunks) == 24 and chunks[0][1] - chunks[0][0] == 3600
