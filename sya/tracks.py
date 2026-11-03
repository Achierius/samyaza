"""Columnar container for many ADS-B tracks, plus resampling to a uniform time grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .geo import geodetic_to_ecef

COLUMNS = ("icao", "t", "lat", "lon", "alt_m")


@dataclass
class Track:
    icao: str
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray
    callsign: str = ""


@dataclass
class TrackTable:
    """Rows sorted by (segment, t). Segment s spans rows offsets[s]:offsets[s+1]."""

    icao: np.ndarray  # (S,) str
    offsets: np.ndarray  # (S+1,)
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray

    @property
    def n_segments(self):
        return len(self.icao)

    @property
    def n_rows(self):
        return len(self.t)

    @staticmethod
    def empty():
        z = np.zeros(0)
        return TrackTable(np.zeros(0, dtype=object), np.zeros(1, np.int64), z, z, z, z)

    @staticmethod
    def from_columns(icao, t, lat, lon, alt_m, max_gap_s: float = 300.0) -> "TrackTable":
        import pyarrow as pa

        t = np.asarray(t, float)
        if t.size == 0:
            return TrackTable.empty()
        # integer codes for the ICAO strings (hash-based, much faster than sorting strings)
        arr = pa.array(np.asarray(icao).astype(str))
        if isinstance(arr, pa.ChunkedArray):  # very large inputs (> 2 GB of strings) come back chunked
            arr = arr.combine_chunks()
        enc = arr.dictionary_encode()
        codes = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
        names = np.asarray(enc.dictionary.to_pylist(), dtype=object)
        order = np.lexsort((t, codes))
        codes, t = codes[order], t[order]
        lat = np.asarray(lat, float)[order]
        lon = np.asarray(lon, float)[order]
        alt = np.asarray(alt_m, float)[order]
        ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
        same = np.r_[False, codes[1:] == codes[:-1]]
        ok &= ~(same & np.r_[False, np.diff(t) <= 0.05])
        codes, t, lat, lon, alt = codes[ok], t[ok], lat[ok], lon[ok], alt[ok]
        if t.size == 0:
            return TrackTable.empty()
        brk = np.r_[True, (codes[1:] != codes[:-1]) | (np.diff(t) > max_gap_s)]
        starts = np.nonzero(brk)[0]
        offsets = np.r_[starts, t.size].astype(np.int64)
        seglen = np.diff(offsets)
        if (seglen < 2).any():
            seg_id = np.repeat(np.arange(seglen.size), seglen)
            rows = np.nonzero(seglen[seg_id] >= 2)[0]
            codes, t, lat, lon, alt = codes[rows], t[rows], lat[rows], lon[rows], alt[rows]
            if t.size == 0:
                return TrackTable.empty()
            brk = np.r_[True, (codes[1:] != codes[:-1]) | (np.diff(t) > max_gap_s)]
            starts = np.nonzero(brk)[0]
            offsets = np.r_[starts, t.size].astype(np.int64)
        return TrackTable(names[codes[starts]], offsets, t, lat, lon, alt)

    @staticmethod
    def from_tracks(tracks: list[Track]) -> "TrackTable":
        if not tracks:
            return TrackTable.empty()
        cat = lambda k: np.concatenate([getattr(tr, k) for tr in tracks])
        ic = np.concatenate([np.full(tr.t.size, tr.icao, dtype=object) for tr in tracks])
        return TrackTable.from_columns(ic, cat("t"), cat("lat"), cat("lon"), cat("alt_m"))

    def segment(self, s: int) -> Track:
        a, b = self.offsets[s], self.offsets[s + 1]
        return Track(str(self.icao[s]), self.t[a:b], self.lat[a:b], self.lon[a:b], self.alt_m[a:b])

    def select(self, t_lo=None, t_hi=None, bbox=None) -> "TrackTable":
        m = np.ones(self.n_rows, bool)
        if t_lo is not None:
            m &= self.t >= t_lo
        if t_hi is not None:
            m &= self.t <= t_hi
        if bbox is not None:
            m &= (self.lat >= bbox[0]) & (self.lat <= bbox[2]) & (self.lon >= bbox[1]) & (self.lon <= bbox[3])
        seg = np.repeat(np.arange(self.n_segments), np.diff(self.offsets))
        return TrackTable.from_columns(self.icao[seg[m]], self.t[m], self.lat[m], self.lon[m], self.alt_m[m])


@dataclass
class Resampled:
    """All segments on a uniform time grid, concatenated. Row arrays have length M."""

    seg: np.ndarray  # (M,) segment index
    t: np.ndarray
    xyz: np.ndarray  # (M, 3) ECEF
    vel: np.ndarray  # (M, 3) ECEF velocity
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    icao: np.ndarray  # (S,) per segment
    seg_offsets: np.ndarray  # (S+1,) offsets into rows

    def rows(self, s):
        return slice(self.seg_offsets[s], self.seg_offsets[s + 1])


def resample(tt: TrackTable, t_lo: float, t_hi: float, dt: float = 1.0, smooth_s: float = 2.0,
             max_interp_gap_s: float = 120.0) -> Resampled:
    segs, ts, lats, lons, alts, xyzs, vels, icaos = [], [], [], [], [], [], [], []
    offs = [0]
    k = 0
    for s in range(tt.n_segments):
        a, b = tt.offsets[s], tt.offsets[s + 1]
        t = tt.t[a:b]
        lo, hi = max(t[0], t_lo), min(t[-1], t_hi)
        if hi - lo < 4 * dt:
            continue
        # split on long gaps instead of interpolating across them
        gaps = np.nonzero(np.diff(t) > max_interp_gap_s)[0]
        bounds = np.r_[0, gaps + 1, t.size]
        for i in range(len(bounds) - 1):
            ta = t[bounds[i]:bounds[i + 1]]
            if ta.size < 2:
                continue
            lo2, hi2 = max(ta[0], t_lo), min(ta[-1], t_hi)
            if hi2 - lo2 < 4 * dt:
                continue
            g = np.arange(np.ceil(lo2 / dt) * dt, hi2, dt)
            sl = slice(a + bounds[i], a + bounds[i + 1])
            la = np.interp(g, ta, tt.lat[sl])
            lo_ = np.interp(g, ta, tt.lon[sl])
            al = np.interp(g, ta, tt.alt_m[sl])
            xyz = geodetic_to_ecef(la, lo_, al)
            if smooth_s > 0 and g.size > 3:
                xyz = gaussian_filter1d(xyz, smooth_s / dt, axis=0, mode="nearest")
            vel = np.gradient(xyz, dt, axis=0)
            segs.append(np.full(g.size, k))
            ts.append(g)
            lats.append(la)
            lons.append(lo_)
            alts.append(al)
            xyzs.append(xyz)
            vels.append(vel)
            icaos.append(tt.icao[s])
            offs.append(offs[-1] + g.size)
            k += 1
    if not ts:
        z = np.zeros(0)
        return Resampled(np.zeros(0, int), z, np.zeros((0, 3)), np.zeros((0, 3)), z, z, z,
                         np.zeros(0, dtype=object), np.zeros(1, np.int64))
    return Resampled(np.concatenate(segs), np.concatenate(ts), np.concatenate(xyzs), np.concatenate(vels),
                     np.concatenate(lats), np.concatenate(lons), np.concatenate(alts),
                     np.array(icaos, dtype=object), np.array(offs, np.int64))
