"""Audio feature frontends for the SDK (host numpy, Pyodide-friendly).

Generic building blocks used by TTS voice cloning and any speech model:
  - stft(): torch-matching STFT (center pad, hann).
  - mel_spectrogram(): matcha/HiFiGAN log-mel (flow prompt features).
  - whisper_log_mel(): OpenAI-Whisper 128-mel log spectrogram (speech tokenizer input).
  - kaldi_fbank(): Kaldi-compatible fbank (speaker-embedding input).
Mel filter matrices are supplied by the caller (baked npz) so no librosa dependency.
"""
import numpy as np


def _hann(n, periodic=True):
    # torch.hann_window(n, periodic=True) default
    N = n if periodic else n - 1
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / N)).astype(np.float32)


def stft_mag(y, n_fft, hop, win_len, center=True, pad_reflect=None, eps=1e-9):
    """Magnitude STFT matching torch.stft(center=..., window=hann, onesided).
    Returns (F=n_fft//2+1, T)."""
    win = _hann(win_len)
    if win_len < n_fft:
        p = (n_fft - win_len) // 2; win = np.pad(win, (p, n_fft - win_len - p))
    if pad_reflect is not None:
        y = np.pad(y, pad_reflect, mode="reflect")
    elif center:
        y = np.pad(y, n_fft // 2, mode="reflect")
    T = 1 + (len(y) - n_fft) // hop
    frames = np.stack([y[t * hop:t * hop + n_fft] * win for t in range(T)], 0)   # (T,n_fft)
    S = np.fft.rfft(frames, axis=1).T                                            # (F,T)
    return np.sqrt(S.real ** 2 + S.imag ** 2 + eps).astype(np.float32)


def mel_spectrogram(y, mel_basis, n_fft=1920, hop=480, win=1920, center=False):
    """matcha/HiFiGAN log-mel: pad (n_fft-hop)/2 reflect, |STFT|, mel, log(clamp 1e-5).
    `mel_basis` (num_mels, n_fft//2+1) = librosa mel filters (baked)."""
    pad = int((n_fft - hop) / 2)
    mag = stft_mag(y, n_fft, hop, win, center=center, pad_reflect=(pad, pad))
    mel = mel_basis @ mag
    return np.log(np.clip(mel, 1e-5, None)).astype(np.float32)


def whisper_log_mel(y, mel_basis, n_fft=400, hop=160):
    """OpenAI Whisper log-mel (n_mels=mel_basis.shape[0], e.g. 128).
    STFT center=True, power spectrum, mel, log10, clamp to max-8, (x+4)/4.
    `mel_basis` (n_mels, 201) baked from whisper's mel_filters."""
    win = _hann(n_fft)
    yp = np.pad(y, n_fft // 2, mode="reflect")
    T = 1 + (len(yp) - n_fft) // hop
    frames = np.stack([yp[t * hop:t * hop + n_fft] * win for t in range(T)], 0)
    S = np.fft.rfft(frames, axis=1).T[:, :-1]            # whisper drops last frame
    power = (S.real ** 2 + S.imag ** 2).astype(np.float32)
    mel = mel_basis @ power
    logspec = np.log10(np.clip(mel, 1e-10, None))
    logspec = np.maximum(logspec, logspec.max() - 8.0)
    return ((logspec + 4.0) / 4.0).astype(np.float32)


def kaldi_fbank(y, mel_basis, num_mel_bins=80, frame_len=400, frame_shift=160,
                preemph=0.97, cmn=True):
    """Kaldi-compatible fbank (dither=0, snip_edges=True, povey window, energy floor).
    `mel_basis` (num_mel_bins, frame_len//2+1) baked from kaldi mel filters.
    Returns (T, num_mel_bins); with per-utterance mean subtraction if cmn."""
    y = np.asarray(y, np.float32) * 32768.0                    # kaldi works in int16 scale
    N = len(y); T = 1 + (N - frame_len) // frame_shift
    # povey window: (0.5 - 0.5 cos(2pi n/(L-1)))^0.85
    n = np.arange(frame_len)
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (frame_len - 1))) ** 0.85
    feats = np.empty((T, num_mel_bins), np.float32)
    nfft = 1
    while nfft < frame_len: nfft <<= 1                          # kaldi rounds up to pow2 (512)
    for t in range(T):
        f = y[t * frame_shift:t * frame_shift + frame_len].astype(np.float64).copy()
        f = f - f.mean()                                       # remove_dc_offset
        f[1:] = f[1:] - preemph * f[:-1]; f[0] = f[0] - preemph * f[0]   # preemphasis
        f = f * win
        S = np.fft.rfft(f, n=nfft)
        power = (S.real ** 2 + S.imag ** 2)
        mel = mel_basis @ power
        feats[t] = np.log(np.clip(mel, 1.19209290e-7, None))
    if cmn:
        feats = feats - feats.mean(0, keepdims=True)
    return feats
