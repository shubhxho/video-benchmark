"""Audio quality scoring — loudness, SNR, silence, wind noise detection."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_LIBROSA_AVAILABLE = False
try:
    import librosa
    import pyloudnorm
    import soundfile as sf

    _LIBROSA_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    return _LIBROSA_AVAILABLE


@dataclass
class AudioResult:
    loudness_lufs: float
    snr_db: float
    silence_pct: float
    has_wind_noise: bool
    overall_score: float  # 0-100


class AudioQualityMetric:
    """Audio quality analysis extracted from video files."""

    def __init__(self, target_sr: int = 16000) -> None:
        self.target_sr = target_sr

    def analyze(
        self, video_path: str, work_dir: Path, ffmpeg_path: str = "ffmpeg"
    ) -> AudioResult | None:
        """Extract and analyze audio from a video file.

        Returns AudioResult or None if librosa not installed or extraction fails.
        """
        if not _LIBROSA_AVAILABLE:
            logger.warning("librosa not installed — audio analysis unavailable")
            return None

        # Extract audio via FFmpeg
        wav_path = work_dir / "audio.wav"
        try:
            result = subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel", "error",
                    "-i", video_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(self.target_sr),
                    "-ac", "1",
                    "-y",
                    str(wav_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0 or not wav_path.exists():
                logger.debug(f"Audio extraction failed: {result.stderr}")
                return None
        except (subprocess.TimeoutExpired, OSError):
            logger.debug("Audio extraction timed out or failed")
            return None

        return self._analyze_wav(wav_path)

    def _analyze_wav(self, wav_path: Path) -> AudioResult | None:
        """Analyze a WAV file for quality metrics."""
        try:
            audio, sr = sf.read(str(wav_path))
            if len(audio) == 0:
                return None

            # Ensure 1D
            if audio.ndim > 1:
                audio = audio[:, 0]

            loudness = self._compute_loudness(audio, sr)
            snr = self._estimate_snr(audio, sr)
            silence_pct = self._compute_silence_pct(audio, sr)
            has_wind = self._detect_wind_noise(audio, sr)

            overall = self._compute_overall_score(
                loudness, snr, silence_pct, has_wind
            )

            return AudioResult(
                loudness_lufs=round(loudness, 1),
                snr_db=round(snr, 1),
                silence_pct=round(silence_pct, 1),
                has_wind_noise=has_wind,
                overall_score=round(overall, 1),
            )
        except Exception:
            logger.debug("Audio analysis failed", exc_info=True)
            return None

    def _compute_loudness(self, audio: np.ndarray, sr: int) -> float:
        """Compute integrated loudness in LUFS."""
        try:
            meter = pyloudnorm.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            # Clamp extreme values
            return max(-70.0, min(0.0, loudness))
        except Exception:
            return -70.0

    def _estimate_snr(self, audio: np.ndarray, sr: int) -> float:
        """Estimate signal-to-noise ratio in dB.

        Uses silent regions as noise reference.
        """
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]

        if len(rms) < 10:
            return 0.0

        # Separate signal and noise frames using Otsu-like threshold
        sorted_rms = np.sort(rms)
        threshold = sorted_rms[len(sorted_rms) // 4]  # Bottom 25% is noise

        noise_frames = rms[rms <= threshold]
        signal_frames = rms[rms > threshold]

        if len(noise_frames) == 0 or len(signal_frames) == 0:
            return 0.0

        noise_power = float(np.mean(noise_frames ** 2))
        signal_power = float(np.mean(signal_frames ** 2))

        if noise_power == 0:
            return 60.0  # Very clean

        snr = 10 * np.log10(signal_power / noise_power)
        return max(0.0, min(60.0, float(snr)))

    def _compute_silence_pct(self, audio: np.ndarray, sr: int) -> float:
        """Compute percentage of audio below -40dB threshold."""
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        # -40dB threshold in linear
        threshold = 10 ** (-40 / 20)
        silent_frames = np.count_nonzero(rms < threshold)
        return float(silent_frames / len(rms) * 100.0) if len(rms) > 0 else 100.0

    def _detect_wind_noise(self, audio: np.ndarray, sr: int) -> bool:
        """Detect wind noise via high zero-crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=2048, hop_length=512
        )[0]
        mean_zcr = float(np.mean(zcr))
        # ZCR > 0.15 often indicates wind or broadband noise
        return mean_zcr > 0.15

    def _compute_overall_score(
        self,
        loudness: float,
        snr: float,
        silence_pct: float,
        has_wind: bool,
    ) -> float:
        """Compute weighted overall audio quality score (0-100)."""
        # Loudness score: optimal range -24 to -14 LUFS
        if -24 <= loudness <= -14:
            loudness_score = 100.0
        elif loudness < -40:
            loudness_score = max(0.0, (loudness + 70) / 30 * 50)
        elif loudness > -14:
            loudness_score = max(0.0, 100 - (loudness + 14) * 10)
        else:
            loudness_score = max(0.0, 50 + (loudness + 24) / 10 * 50)

        # SNR score: 0-60 dB → 0-100
        snr_score = min(100.0, snr / 40 * 100)

        # Silence penalty: high silence = bad
        silence_score = max(0.0, 100.0 - silence_pct)

        # Wind penalty
        wind_penalty = 20.0 if has_wind else 0.0

        # Weighted combination
        score = (
            loudness_score * 0.30
            + snr_score * 0.30
            + silence_score * 0.25
            + (100.0 - wind_penalty) * 0.15
        )
        return max(0.0, min(100.0, score))
