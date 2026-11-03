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


def spectrogram(x, fs, fmax, win_s=1.0, hop_s=0.1, oversample=4) -> Spectrogram:
    fs2 = max(200, int(2.5 * fmax))
    if fs2 < fs:
        fr = Fraction(fs2 / fs).limit_denominator(1000)
        x = signal.resample_poly(x, fr.numerator, fr.denominator)
        fs2 = fs * fr.numerator / fr.denominator
    else:
        fs2 = fs
    nper = int(round(win_s * fs2))
    hop = max(1, int(round(hop_s * fs2)))
    nfft = 1 << int(np.ceil(np.log2(nper * oversample)))
    f, t, Z = signal.stft(
        x, fs2, window="hann", nperseg=nper, noverlap=nper - hop, nfft=nfft, boundary=None, padded=False
    )
    keep = f <= fmax
    L = np.log(np.abs(Z[keep]) ** 2 + 1e-20).T  # (T, F)
    return Spectrogram(t=t, f=f[keep], z=_whiten(L))


def _whiten(L):
    """Remove stationary hum (per-bin median) and broadband bursts (per-frame median)."""
    L = L - np.median(L, axis=0, keepdims=True)
    L = L - np.median(L, axis=1, keepdims=True)
    mad = np.median(np.abs(L), axis=0, keepdims=True) * 1.4826 + 1e-9
    return L / mad


def harmonic_sum(S: Spectrogram, fmin, fmax, harmonics=1) -> Spectrogram:
    """Score each candidate fundamental by summing whitened power at k*f, k=1..K."""
    df = S.f[1] - S.f[0]
    fund = S.f[(S.f >= fmin) & (S.f <= fmax)]
    out = np.zeros((S.z.shape[0], fund.size))
    used = 0
    for k in range(1, harmonics + 1):
        pos = k * fund / df
        if pos[-1] >= S.f.size - 1:
            break
        i0 = np.floor(pos).astype(int)
        a = pos - i0
        out += (1 - a) * S.z[:, i0] + a * S.z[:, i0 + 1]
        used += 1
    return Spectrogram(t=S.t, f=fund, z=out / np.sqrt(max(used, 1)))


# ------------------------------------------------------------------ ridge track
def viterbi_ridge(z, max_jump: int, penalty: float = 0.3):
    """Highest-scoring path through z (T, F) with bounded per-frame bin jumps."""
    T, B = z.shape
    score = z[0].copy()
    back = np.zeros((T, B), np.int32)
    idx = np.arange(B)
    for t in range(1, T):
        best = np.full(B, -np.inf)
        arg = idx.copy()
        for s in range(-max_jump, max_jump + 1):
            cand = np.full(B, -np.inf)
            if s >= 0:
                cand[s:] = score[: B - s] if s else score
            else:
                cand[:s] = score[-s:]
            cand -= penalty * abs(s)
            better = cand > best
            best = np.where(better, cand, best)
            arg = np.where(better, idx - s, arg)
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
    merge_gap_s=4.0, windows=None, c=None, max_slope_hz_s=12.0,
):
    """Find flyover tones and fit each one. `windows` = [(t0, t1), ...] to restrict manually."""
    S = spectrogram(x, fs, fmax * harmonics, win_s, hop_s)
    H = harmonic_sum(S, fmin, fmax, harmonics)
    low = 0.6 * snr_thresh  # hysteresis: seed at snr_thresh, extend down to `low`
    segments = windows
    if segments is None:
        ridge = track_ridge(H, max_slope_hz_s)
        sm = uniform_filter1d(ridge.snr, max(1, int(round(3.0 / hop_s))))
        segments = _runs(H.t, sm > snr_thresh, sm > low, merge_gap_s, min_len_s)
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
        if not (5 < fit.v < 330) or fit.rms_hz > 3.0:
            continue
        events.append(Event(r.t[good], r.f[good], w, fit))
    return events, H


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
