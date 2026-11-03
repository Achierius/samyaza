"""Parse the user's prior on *when* the recording started.

Accepted forms (all produce a window [t_lo, t_hi] for the recording START, unix seconds):
  --date 2023-08-08                     whole local day (see --tz)
  --dates 2023-08-07 2023-08-09         inclusive range of whole days
  --between 2023-08-08T17:00 2023-08-08T20:30     arbitrary timestamp range
  --at 2023-08-08T20:10Z [--tolerance 15m]        a timestamp ± tolerance
  --when "2023-08-08"  /  --when "2023-08-07..2023-08-09"  /  --when "2023-08-08T17:00Z..2023-08-08T19:00Z"
Timestamps without an explicit offset are interpreted in --tz (default UTC); unix seconds also work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$")


def parse_duration(s: str | float) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    m = _DUR.match(s)
    if not m:
        raise ValueError(f"bad duration {s!r} (examples: 90, 90s, 15m, 2h, 1d)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _tz(name: str | None):
    if not name or name.upper() == "UTC":
        return timezone.utc
    return ZoneInfo(name)


def parse_timestamp(s: str, tz: str | None = None) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    s2 = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s2)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz(tz))
    return dt.timestamp()


def _is_date(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.strip()))


def day_window(d: str, tz: str | None = None) -> tuple[float, float]:
    dd = date.fromisoformat(d.strip())
    z = _tz(tz)
    a = datetime.combine(dd, time(0), z)
    b = a + timedelta(days=1)
    return a.timestamp(), b.timestamp()


@dataclass
class TimeWindow:
    t_lo: float
    t_hi: float
    label: str

    @property
    def span(self):
        return self.t_hi - self.t_lo

    def iso(self):
        f = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return f(self.t_lo), f(self.t_hi)

    def chunks(self, size_s: float):
        a = self.t_lo
        while a < self.t_hi:
            b = min(a + size_s, self.t_hi)
            yield a, b
            a = b


def parse_when(spec: str, tz: str | None = None) -> TimeWindow:
    spec = spec.strip()
    if ".." in spec:
        a, b = (x.strip() for x in spec.split("..", 1))
        if _is_date(a) and _is_date(b):
            return dates_window(a, b, tz)
        return between_window(a, b, tz)
    if _is_date(spec):
        lo, hi = day_window(spec, tz)
        return TimeWindow(lo, hi, f"date {spec} ({tz or 'UTC'})")
    t = parse_timestamp(spec, tz)
    return TimeWindow(t - 900, t + 900, f"{spec} ± 15m")


def dates_window(a: str, b: str, tz: str | None = None) -> TimeWindow:
    lo, _ = day_window(a, tz)
    _, hi = day_window(b, tz)
    if hi <= lo:
        raise ValueError("end date must not precede start date")
    return TimeWindow(lo, hi, f"dates {a}..{b} ({tz or 'UTC'})")


def between_window(a: str, b: str, tz: str | None = None) -> TimeWindow:
    lo, hi = parse_timestamp(a, tz), parse_timestamp(b, tz)
    if hi <= lo:
        raise ValueError("end of time range must be after its start")
    return TimeWindow(lo, hi, f"{a} .. {b}")


def at_window(ts: str, tol: str | float = "15m", tz: str | None = None) -> TimeWindow:
    t = parse_timestamp(ts, tz)
    d = parse_duration(tol)
    return TimeWindow(t - d, t + d, f"{ts} ± {tol}")
