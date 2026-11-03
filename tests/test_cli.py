import pytest

from sya.cli import area, build_parser, parse_radius, time_window


def _args(*argv):
    return build_parser().parse_args(["locate", "x.wav", *argv])


def test_time_flags_are_exclusive_and_required():
    with pytest.raises(SystemExit):
        time_window(_args())
    with pytest.raises(SystemExit):
        time_window(_args("--date", "2023-08-08", "--at", "2023-08-08T10:00Z"))
    assert time_window(_args("--date", "2023-08-08")).span == 86400
    assert time_window(_args("--dates", "2023-08-08", "2023-08-10")).span == 3 * 86400
    assert time_window(_args("--between", "2023-08-08T10:00Z", "2023-08-08T11:00Z")).span == 3600
    assert time_window(_args("--at", "2023-08-08T10:00Z", "--tolerance", "5m")).span == 600
    assert time_window(_args("--when", "2023-08-08..2023-08-09")).span == 2 * 86400


def test_area_flags():
    assert area(_args()).kind == "conus"
    assert area(_args("--state", "nd")).states == ("ND",)
    r = area(_args("--center", "33.45,-112.07", "--radius", "25km"))
    assert r.kind == "radius" and r.radius_m == 25000
    with pytest.raises(SystemExit):
        area(_args("--center", "33.45,-112.07"))  # radius missing
    with pytest.raises(SystemExit):
        area(_args("--radius", "5km"))  # center missing
    with pytest.raises(SystemExit):
        area(_args("--state", "AZ", "--center", "33.45,-112.07", "--radius", "5km"))
    with pytest.raises(SystemExit):
        area(_args("--state", "Hawaii"))


def test_parse_radius_units():
    assert parse_radius("10km") == 10000
    assert abs(parse_radius("1mi") - 1609.344) < 1e-6
    assert parse_radius("500") == 500
