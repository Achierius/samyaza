"""Extract Doppler curves (frequency vs time) of aircraft tones from audio."""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares

from .physics import speed_of_sound, straight_pass


# --------------------------------------------------------------------------- I/O
def load_audio(path: str, fs: int | None = None) -> tuple[np.ndarray, int]:
    """Load mono float audio. Uses soundfile, falls back to ffmpeg (any container)."""
    try:
        import soundfile as sf

        x, sr = sf.read(path, always_2d=True, dtype="float64")
        x = x.mean(axis=1)
    except Exception:
        sr = fs or 16000
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
            check=True,
            capture_output=True,
        ).stdout
        x = np.frombuffer(raw, np.float32).astype(np.float64)
    if fs and fs != sr:
        x = _resample(x, sr, fs)
        sr = fs
    return x, int(sr)


def _resample(x, fs_in, fs_out):
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, fr.numerator, fr.denominator)


# ------------------------------------------------------------------- spectrogram
@dataclass
class Spectrogram:
    t: np.ndarray  # frame centre times (s)
    f: np.ndarray  # bin frequencies (Hz)
    z: np.ndarray  # (T, F) whitened log power (robust z-score)
    level: np.ndarray | None = None  # (T,) per-frame median log power before whitening


def spectrogram(x, fs, fmax, win_s=1.0, hop_s=0.1, oversample=2, block=1024) -> Spectrogram:
    """Blocked float32 STFT (Hann), log power up to fmax, robustly whitened."""
    from scipy import fft as sfft

    fs2 = max(200, int(2.5 * fmax))
    if fs2 < fs:
        fr = Fraction(fs2 / fs).limit_denominator(1000)
        x = signal.resample_poly(x, fr.numerator, fr.denominator)
        fs2 = fs * fr.numerator / fr.denominator
    else:
        fs2 = fs
    x = np.asarray(x, np.float32)
    nper = int(round(win_s * fs2))
    hop = max(1, int(round(hop_s * fs2)))
    nfft = 1 << int(np.ceil(np.log2(nper * oversample)))
    f = np.arange(nfft // 2 + 1) * fs2 / nfft
    nkeep = int(np.searchsorted(f, fmax, "right"))
    if len(x) < nper:
        x = np.pad(x, (0, nper - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, nper)[::hop]
    win = signal.windows.hann(nper, sym=False).astype(np.float32)
    L = np.empty((frames.shape[0], nkeep), np.float32)
    for a in range(0, frames.shape[0], block):
        F = sfft.rfft(frames[a:a + block] * win, n=nfft, axis=1, workers=-1)[:, :nkeep]
        L[a:a + block] = np.log(F.real**2 + F.imag**2 + 1e-20)
    t = (np.arange(frames.shape[0]) * hop + nper / 2) / fs2
    level = np.median(L[:, :: max(1, L.shape[1] // 400)], axis=1)
    return Spectrogram(t=t, f=f[:nkeep], z=_whiten(L), level=level)


def _whiten(L):
    """Remove stationary hum (per-bin median) and broadband bursts (per-frame median).

    Medians are estimated on subsampled rows/columns: same answer, ~10x faster."""
    st = max(1, L.shape[0] // 1500)
    L -= np.median(L[::st], axis=0, keepdims=True)
    sf_ = max(1, L.shape[1] // 400)
    L -= np.median(L[:, ::sf_], axis=1, keepdims=True)
    mad = np.median(np.abs(L[::st]), axis=0, keepdims=True) * 1.4826 + 1e-6
    L /= mad
    return L


def harmonic_sum(S: Spectrogram, fmin, fmax, harmonics=1) -> Spectrogram:
    """Score each candidate fundamental by summing whitened power at k*f, k=1..K."""
    df = S.f[1] - S.f[0]
    fund = S.f[(S.f >= fmin) & (S.f <= fmax)]
    out = np.zeros((S.z.shape[0], fund.size))
    used = 0
    for k in range(1, harmonics + 1):
        pos = k * fund / df
        if pos[-1] >= S.f.size - 1:
            # The spectrum is computed up to fmax*harmonics, so the top harmonic's last bin falls
            # off the end and that harmonic is skipped (all validation was tuned with this
            # behaviour: the default --harmonics 3 effectively sums 2). The fundamental itself
            # must never be skipped (that silently zeroed --harmonics 1).
            if used:
                break
            pos = np.minimum(pos, S.f.size - 1.000001)
        i0 = np.floor(pos).astype(int)
        a = pos - i0
        out += (1 - a) * S.z[:, i0] + a * S.z[:, i0 + 1]
        used += 1
    return Spectrogram(t=S.t, f=fund, z=out / np.sqrt(max(used, 1)))


# ------------------------------------------------------------------ ridge track
def viterbi_ridge(z, max_jump: int, penalty: float = 0.3):
    """Highest-scoring path through z (T, F).

    Transition cost is `penalty` per bin moved (exact 1-D distance transform via running
    maxima from the left and right, O(F) per frame). `max_jump` is kept for API
    compatibility; the linear penalty already discourages large jumps.
    """
    T, B = z.shape
    idx = np.arange(B, dtype=np.float64)
    ii = np.arange(B)
    big = B + 1
    score = z[0].astype(np.float64)
    back = np.zeros((T, B), np.int32)
    for t in range(1, T):
        # left[j] = max_{k<=j} score[k] - p (j - k);  right[j] = max_{k>=j} score[k] - p (k - j)
        A = score + penalty * idx
        la = np.maximum.accumulate(A)
        argl = np.maximum.accumulate(np.where(A >= la, ii, 0))
        left = la - penalty * idx
        Rv = (score - penalty * idx)[::-1]
        ra = np.maximum.accumulate(Rv)
        argr = np.minimum.accumulate(np.where(Rv >= ra, ii[::-1], big))[::-1]
        right = ra[::-1] + penalty * idx
        use_r = right > left
        best = np.where(use_r, right, left)
        arg = np.where(use_r, argr, argl)
        score = best + z[t]
        back[t] = arg
    path = np.empty(T, int)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


@dataclass
class Ridge:
    t: np.ndarray
    f: np.ndarray
    snr: np.ndarray  # ridge score per frame (z units)


def track_ridge(S: Spectrogram, max_slope_hz_s=12.0, penalty=0.3) -> Ridge:
    df = S.f[1] - S.f[0]
    hop = S.t[1] - S.t[0]
    max_jump = max(1, int(np.ceil(max_slope_hz_s * hop / df)))
    path = viterbi_ridge(S.z, max_jump, penalty)
    rows = np.arange(len(path))
    j = np.clip(path, 1, S.f.size - 2)
    ym, y0, yp = S.z[rows, j - 1], S.z[rows, j], S.z[rows, j + 1]
    den = ym - 2 * y0 + yp
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(np.abs(den) > 1e-9, 0.5 * (ym - yp) / den, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    return Ridge(t=S.t, f=S.f[j] + delta * df, snr=S.z[rows, path])


# -------------------------------------------------------------------- pass fit
@dataclass
class PassFit:
    f0: float
    v: float
    d: float
    t_ca: float
    rms_hz: float
    n: int
    stderr: dict = field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 0.0

    def to_dict(self):
        return asdict(self)


def fit_pass(t, f, c=None, w=None) -> PassFit:
    """Least-squares fit of the straight-flyby Doppler model to a ridge."""
    c = c or speed_of_sound()
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    w = np.ones_like(t) if w is None else np.asarray(w, float)
    sw = np.sqrt(w / w.mean())
    f_hi, f_lo = np.percentile(f, 95), np.percentile(f, 5)
    f0_i = 2 * f_hi * f_lo / (f_hi + f_lo)
    v_i = np.clip(c * (f_hi - f_lo) / (f_hi + f_lo), 15, 0.8 * c)
    fs = uniform_filter1d(f, max(3, len(f) // 30))
    tca_i = t[int(np.argmin(np.abs(fs - f0_i)))]

    def res(p, tt, ff, ss):
        return ss * (straight_pass(tt, p[0], p[1], p[2], p[3], c) - ff)

    lo = [1.0, 5.0, 20.0, t[0] - 120]
    hi = [1e5, 0.9 * c, 3e4, t[-1] + 120]
    best = None
    keep = np.ones_like(t, bool)
    for _ in range(2):  # fit, reject outliers, refit
        best = None
        for d_i in (200, 500, 1000, 2000, 4000, 8000):
            p0 = np.clip([f0_i, v_i, d_i, tca_i], np.add(lo, 1e-6), np.subtract(hi, 1e-6))
            r = least_squares(
                res, p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.5,
                args=(t[keep], f[keep], sw[keep]),
            )
            if best is None or r.cost < best.cost:
                best = r
        resid = straight_pass(t, *best.x, c) - f
        mad = np.median(np.abs(resid[keep])) * 1.4826 + 1e-6
        keep = np.abs(resid) < max(4 * mad, 0.5)
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (2 * best.cost / max(1, keep.sum() - 4))
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(4, np.nan)
    rms = float(np.sqrt(np.mean(resid[keep] ** 2)))
    return PassFit(
        f0=float(best.x[0]), v=float(best.x[1]), d=float(best.x[2]), t_ca=float(best.x[3]),
        rms_hz=rms, n=int(keep.sum()),
        stderr=dict(zip(["f0", "v", "d", "t_ca"], map(float, se))),
        t_start=float(t[0]), t_end=float(t[-1]),
    )


# -------------------------------------------------------------- event finding
@dataclass
class Event:
    t: np.ndarray  # seconds from recording start
    f: np.ndarray
    w: np.ndarray
    fit: PassFit

    def to_dict(self):
        return {"t": self.t.tolist(), "f": self.f.tolist(), "w": self.w.tolist(), "fit": self.fit.to_dict()}

    @staticmethod
    def from_dict(d):
        return Event(np.array(d["t"]), np.array(d["f"]), np.array(d["w"]), PassFit(**d["fit"]))


def extract_events(
    x, fs, fmin, fmax, harmonics=3, win_s=1.0, hop_s=0.1, snr_thresh=3.0, min_len_s=20.0,
    merge_gap_s=4.0, windows=None, c=None, max_slope_hz_s=12.0, log=None, max_tracks=8,
    max_rms_hz=1.5, min_window_s=30.0,
):
    """Find flyover tones and fit each one.

    Automatic mode extracts tones iteratively: track the strongest ridge, cut it into
    segments (SNR hysteresis, and at jumps faster than max_slope), fit each segment, then
    mask that ridge (and its harmonics / sub-harmonics) and repeat up to `max_tracks`
    times, so overlapping aircraft are separated. `windows` = [(t0, t1), ...] restricts the
    analysis to manual windows (one tone each).
    """
    from contextlib import nullcontext

    tm = (lambda name: log.timed(name)) if log is not None else (lambda name: nullcontext({}))
    with tm("spectrogram"):
        S = spectrogram(x, fs, fmax * harmonics, win_s, hop_s)
    with tm("harmonic-sum"):
        H = harmonic_sum(S, fmin, fmax, harmonics)
    low = 0.6 * snr_thresh  # hysteresis: seed at snr_thresh, extend down to `low`
    if windows is not None:
        with tm("fit"):
            events = _fit_segments(H, windows, low, hop_s, max_slope_hz_s, c, max_rms_hz)
        return events, H
    events = []
    Z = H.z.copy()
    df = H.f[1] - H.f[0]
    mask_bins = max(2, int(round(2.5 / df)))
    jump_hz = max_slope_hz_s * hop_s * 1.5 + 0.5
    for it in range(max_tracks):
        with tm("ridge"):
            ridge = track_ridge(Spectrogram(H.t, H.f, Z), max_slope_hz_s)
            sm = uniform_filter1d(ridge.snr, max(1, int(round(3.0 / hop_s))))
            on_lo = sm > low
            jumps = np.r_[False, np.abs(np.diff(ridge.f)) > jump_hz]
            on_lo &= ~jumps
            segs = _runs(H.t, sm > snr_thresh, on_lo, 0.0, min_len_s)
        if not segs:
            break
        with tm("fit"):
            snr1 = uniform_filter1d(ridge.snr, max(1, int(round(1.0 / hop_s))))
            for a, b in segs:
                events += _fit_scurves(ridge, snr1, a, b, low, hop_s, c, max_rms_hz)
        # mask this ridge (all frames where it was "on"), its harmonics and sub-harmonics
        rows = np.nonzero(on_lo)[0]
        for k in (1.0, 2.0, 3.0, 0.5, 1 / 3):
            fk = ridge.f[rows] * k
            ok = (fk >= H.f[0]) & (fk <= H.f[-1])
            jc = np.round((fk[ok] - H.f[0]) / df).astype(int)
            rr = rows[ok]
            for dj in range(-mask_bins, mask_bins + 1):
                jj = np.clip(jc + dj, 0, Z.shape[1] - 1)
                Z[rr, jj] = np.minimum(Z[rr, jj], 0.0)
    return _dedupe_events(events), H


def _plausible(fit: "PassFit", max_rms_hz: float) -> bool:
    return 15 < fit.v < 300 and 80 < fit.d < 10000 and fit.rms_hz <= max_rms_hz and _is_flyover(fit)


def _fit_scurves(ridge, snr1, a, b, low, hop_s, c, max_rms_hz, min_drop_hz_s=0.25, min_window_s=30.0):
    """Fit every Doppler S-curve inside segment [a, b] of a ridge.

    A long ridge can hold several flyovers (or a flyover plus an unrelated steady tone),
    so fit windows centred on each steep frequency descent, trying a few widths, and keep
    the best plausible fit per descent."""
    m = (ridge.t >= a) & (ridge.t <= b) & (snr1 > low)
    idx = np.nonzero(m)[0]
    if idx.size < 20:
        return []
    t, f = ridge.t[idx], ridge.f[idx]
    fs = uniform_filter1d(f, max(3, int(round(3.0 / hop_s))))
    slope = np.gradient(fs, t)
    order = np.argsort(slope)
    centres = []
    for i in order:
        if slope[i] > -min_drop_hz_s:
            break
        if all(abs(t[i] - t0) > 20 for t0 in centres):
            centres.append(t[i])
        if len(centres) >= 6:
            break
    if not centres:  # no clear descent: try the whole segment (e.g. partially observed pass)
        centres = [0.5 * (t[0] + t[-1])]
    out = []
    for t0 in centres:
        best = None
        for hw in ((5.0, 10.0) if min_window_s < 20 else ()) + (20.0, 35.0, 60.0, 100.0):
            w_ = (t >= t0 - hw) & (t <= t0 + hw)
            if w_.sum() < 20 or t[w_][-1] - t[w_][0] < min_window_s:
                continue
            ww = np.clip(ridge.snr[idx][w_], 0, 20)
            fit = fit_pass(t[w_], f[w_], c=c, w=ww)
            if not _plausible(fit, max_rms_hz):
                continue
            # prefer longer windows unless they fit clearly worse
            score = fit.rms_hz / np.sqrt(w_.sum())
            if best is None or score < best[0]:
                best = (score, Event(t[w_], f[w_], ww, fit))
        if best:
            out.append(best[1])
    return out


def _is_flyover(fit: "PassFit") -> bool:
    """The fitted closest approach must lie inside (or near) the observed segment, so the
    S-curve's turning point was actually seen."""
    span = fit.t_end - fit.t_start
    return fit.t_start - 0.15 * span <= fit.t_ca <= fit.t_end + 0.15 * span


def _dedupe_events(events):
    """Drop events that are the same tone seen twice (overlapping time, similar f0)."""
    events = sorted(events, key=lambda e: e.fit.rms_hz)
    out = []
    for e in events:
        dup = False
        for o in out:
            ov = min(e.fit.t_end, o.fit.t_end) - max(e.fit.t_start, o.fit.t_start)
            short = min(e.fit.t_end - e.fit.t_start, o.fit.t_end - o.fit.t_start)
            if ov > 0.5 * short and abs(e.fit.t_ca - o.fit.t_ca) < 10:
                ratio = e.fit.f0 / o.fit.f0
                if any(abs(ratio - k) < 0.04 * k for k in (1, 2, 3, 0.5, 1 / 3)):
                    dup = True
                    break
        if not dup:
            out.append(e)
    return sorted(out, key=lambda e: e.fit.t_ca)


def _fit_segments(H, segments, low, hop_s, max_slope_hz_s, c, max_rms_hz=3.0):
    events = []
    for a, b in segments:
        m = (H.t >= a) & (H.t <= b)
        if m.sum() < 20:
            continue
        sub = Spectrogram(H.t[m], H.f, H.z[m])
        r = track_ridge(sub, max_slope_hz_s)
        good = uniform_filter1d(r.snr, max(1, int(round(1.0 / hop_s)))) > low
        if good.sum() < 20:
            continue
        w = np.clip(r.snr[good], 0, 20)
        fit = fit_pass(r.t[good], r.f[good], c=c, w=w)
        if not (5 < fit.v < 330) or fit.rms_hz > max_rms_hz:
            continue
        events.append(Event(r.t[good], r.f[good], w, fit))
    return events


def _runs(t, hi, lo, merge_gap_s, min_len_s):
    """Segments of `lo` that contain at least one `hi` frame."""
    segs = []
    i = 0
    n = len(lo)
    while i < n:
        if lo[i]:
            j = i
            while j + 1 < n and lo[j + 1]:
                j += 1
            if hi[i:j + 1].any():
                segs.append([t[i], t[j]])
            i = j + 1
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged if b - a >= min_len_s]


# ----------------------------------------------------------------------- quiet map
@dataclass
class QuietMap:
    """Where in the recording an aircraft tone *would* have been detected had there been one.

    quiet[i] covers [t[i], t[i] + 1 s): no broadband masking (wind, traffic, speech bursts more
    than ~6 dB above the recording's typical level), no tonal energy above the detection
    threshold, and not inside a detected flyover."""

    t: np.ndarray
    quiet: np.ndarray
    duration: float

    def is_quiet(self, t0: float, t1: float) -> bool:
        if t0 < 0 or t1 > self.duration:
            return False
        i0, i1 = int(np.floor(t0)), int(np.ceil(t1))
        seg = self.quiet[max(i0, 0): min(i1, self.quiet.size)]
        return bool(seg.size) and bool(seg.all())

    @property
    def fraction(self):
        return float(self.quiet.mean()) if self.quiet.size else 0.0


def quiet_map(x, fs, fmin, fmax, harmonics, events, win_s=1.0, hop_s=0.1, snr_thresh=3.0,
              mask_db=6.0, event_pad_s=5.0) -> QuietMap:
    S = spectrogram(x, fs, fmax * harmonics, win_s, hop_s)
    H = harmonic_sum(S, fmin, fmax, harmonics)
    lvl = S.level
    masked = lvl > np.median(lvl) + mask_db / 10 * np.log(10)
    tonal = uniform_filter1d(H.z.max(axis=1), max(1, int(round(2.0 / hop_s)))) > snr_thresh
    busy = masked | tonal
    for e in events:
        busy |= (H.t >= e.t[0] - event_pad_s) & (H.t <= e.t[-1] + event_pad_s)
    dur = len(x) / fs
    nb = int(np.floor(dur))
    sec = np.floor(H.t).astype(int)
    bad = np.zeros(nb, bool)
    ok = (sec >= 0) & (sec < nb)
    np.logical_or.at(bad, sec[ok], busy[ok])
    covered = np.zeros(nb, bool)
    covered[sec[ok]] = True
    return QuietMap(np.arange(nb, dtype=float), covered & ~bad, dur)
