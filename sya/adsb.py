"""Load historical ADS-B tracks.

Supported sources:
  * CSV with columns icao,time,lat,lon,alt_m[,callsign]  (time = unix seconds,
    alt_m = geometric/ellipsoid altitude preferred, metres)
  * readsb / tar1090 `trace_full_<icao>.json` files, as found in the adsb.lol
    globe_history daily archives (https://github.com/adsblol/globe_history_2025)
  * OpenSky historical database via `pyopensky` (needs research access)
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

FT = 0.3048


@dataclass
class Track:
    icao: str
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray
    callsign: str = ""


def split_gaps(tr: Track, max_gap_s: float = 300.0) -> list[Track]:
    o = np.argsort(tr.t)
    t, la, lo, al = tr.t[o], tr.lat[o], tr.lon[o], tr.alt_m[o]
    cuts = np.where(np.diff(t) > max_gap_s)[0] + 1
    out = []
    for s in np.split(np.arange(t.size), cuts):
        if s.size >= 2:
            out.append(Track(tr.icao, t[s], la[s], lo[s], al[s], tr.callsign))
    return out


def load_csv(path: str) -> list[Track]:
    rows = defaultdict(list)
    calls = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["icao"]].append((float(r["time"]), float(r["lat"]), float(r["lon"]), float(r["alt_m"])))
            if r.get("callsign"):
                calls[r["icao"]] = r["callsign"].strip()
    out = []
    for icao, pts in rows.items():
        a = np.array(pts)
        out += split_gaps(Track(icao, a[:, 0], a[:, 1], a[:, 2], a[:, 3], calls.get(icao, "")))
    return out


def save_csv(tracks: list[Track], path: str):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["icao", "time", "lat", "lon", "alt_m", "callsign"])
        for tr in tracks:
            for i in range(tr.t.size):
                w.writerow([tr.icao, f"{tr.t[i]:.2f}", f"{tr.lat[i]:.6f}", f"{tr.lon[i]:.6f}", f"{tr.alt_m[i]:.1f}", tr.callsign])


def _open_json(path):
    with open(path, "rb") as fh:
        head = fh.read(2)
    op = gzip.open if head == b"\x1f\x8b" else open
    with op(path, "rt") as fh:
        return json.load(fh)


def parse_readsb_trace(obj, t_lo=None, t_hi=None, bbox=None) -> list[Track]:
    """Parse a readsb trace_full JSON object.

    trace rows: [dt, lat, lon, alt_baro_ft|"ground", gs_kt, track, flags, vrate, detail, source, alt_geom_ft, ...]
    flags & 2 marks the start of a new leg.
    """
    base = float(obj["timestamp"])
    icao = obj.get("icao", "?")
    pts, legs = [], []
    call = ""
    for row in obj.get("trace", []):
        t = base + float(row[0])
        if (t_lo is not None and t < t_lo) or (t_hi is not None and t > t_hi):
            continue
        lat, lon, alt_b = row[1], row[2], row[3]
        if lat is None or lon is None or alt_b == "ground" or alt_b is None:
            continue
        if bbox and not (bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]):
            continue
        alt_g = row[10] if len(row) > 10 and isinstance(row[10], (int, float)) else None
        alt = (alt_g if alt_g is not None else alt_b) * FT
        flags = row[6] if len(row) > 6 and isinstance(row[6], int) else 0
        if flags & 2 and pts:
            legs.append(pts)
            pts = []
        if len(row) > 8 and isinstance(row[8], dict) and row[8].get("flight"):
            call = row[8]["flight"].strip()
        pts.append((t, lat, lon, alt))
    if pts:
        legs.append(pts)
    out = []
    for leg in legs:
        a = np.array(leg)
        if len(a) >= 2:
            out += split_gaps(Track(icao, a[:, 0], a[:, 1], a[:, 2], a[:, 3], call))
    return out


def load_readsb_dir(root: str, t_lo=None, t_hi=None, bbox=None) -> list[Track]:
    """Walk an extracted globe_history day (…/traces/xx/trace_full_<icao>.json)."""
    out = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.startswith("trace_full_") and fn.endswith(".json"):
                try:
                    out += parse_readsb_trace(_open_json(os.path.join(dp, fn)), t_lo, t_hi, bbox)
                except (OSError, ValueError, KeyError):
                    continue
    return out


def fetch_opensky(t_lo: float, t_hi: float, bbox) -> list[Track]:
    """bbox = (lat_min, lon_min, lat_max, lon_max). Requires `pip install pyopensky` + credentials."""
    from pyopensky.trino import Trino  # type: ignore

    df = Trino().history(
        start=int(t_lo), stop=int(t_hi), bounds=(bbox[1], bbox[0], bbox[3], bbox[2]),
        selected_columns=("time", "icao24", "callsign", "lat", "lon", "geoaltitude", "baroaltitude", "onground"),
    )
    if df is None or len(df) == 0:
        return []
    df = df[~df["onground"].astype(bool)].dropna(subset=["lat", "lon"])
    out = []
    for icao, g in df.groupby("icao24"):
        alt = g["geoaltitude"].fillna(g["baroaltitude"]).to_numpy(float)
        t = g["time"].astype("int64").to_numpy() / (1e9 if np.issubdtype(g["time"].dtype, np.datetime64) else 1)
        ok = np.isfinite(alt)
        tr = Track(icao, t[ok], g["lat"].to_numpy(float)[ok], g["lon"].to_numpy(float)[ok], alt[ok],
                   str(g["callsign"].dropna().iloc[0]).strip() if g["callsign"].notna().any() else "")
        out += split_gaps(tr)
    return out
