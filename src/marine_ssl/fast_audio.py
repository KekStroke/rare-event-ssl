"""Bounded short-file decoding; None explicitly delegates to the FFmpeg path."""

from math import gcd
from pathlib import Path
import struct

BACKEND_ID = "soundfile-scipy-polyphase-pcm16-v1"


def _wav_complete(path, size):
    """Check RIFF chunk bounds, which some PCM decoders otherwise tolerate."""
    with path.open("rb") as stream:
        head = stream.read(12)
        if len(head) < 12 or head[:4] not in {b"RIFF", b"RIFX"}:
            return False  # RF64/streaming/other containers use the existing path.
        order = ">" if head[:4] == b"RIFX" else "<"
        declared = struct.unpack(order + "I", head[4:8])[0]
        if declared == 0xFFFFFFFF:
            return False
        if head[8:12] != b"WAVE" or declared + 8 > size:
            raise ValueError("Truncated or invalid WAV container")
        end = declared + 8
        while stream.tell() < end:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError("Truncated WAV chunk header")
            count = struct.unpack(order + "I", chunk[4:])[0]
            if stream.tell() + count > end:
                raise ValueError("Truncated WAV chunk")
            stream.seek(count + (count & 1), 1)
        return True


def decode_pcm16(path, target_rate=16000, max_seconds=3600,
                 max_source_bytes=64 * 1024**2, max_decoded_bytes=256 * 1024**2):
    """Return (metadata, full mono PCM16 bytes), or None for explicit fallback.

    A below-target native rate returns (skip metadata, None). Unknown codecs or
    missing optional dependencies delegate to FFmpeg. Recognized decode errors,
    nonfinite samples and truncated RIFF files raise ValueError. No five-second
    padding or segmentation occurs here. Downmix is the equal-weight mean;
    resampling uses scipy's Kaiser-5 polyphase filter and ceil(N*target/native)
    output frames. Conversion saturates and rounds to nearest PCM16 sample.
    """
    path = Path(path)
    if target_rate != 16000:
        raise ValueError("The fast decoder supports the project's 16 kHz target")
    suffix, source_bytes = path.suffix.lower(), path.stat().st_size
    if suffix not in {".wav", ".wave", ".flac", ".mp3"} or source_bytes > max_source_bytes:
        return None
    try:
        import numpy as np
        import soundfile as sf
        import scipy
        from scipy.signal import resample_poly
    except ImportError:
        return None
    if suffix in {".wav", ".wave"} and not _wav_complete(path, source_bytes):
        return None
    try:
        audio = sf.SoundFile(str(path))
    except (RuntimeError, OSError):
        return None
    with audio:
        if audio.format not in {"WAV", "WAVEX", "FLAC", "MPEG", "MP3"}:
            return None
        frames, native_rate, channels = len(audio), audio.samplerate, audio.channels
        if frames <= 0 or native_rate <= 0 or channels <= 0:
            raise ValueError("Empty or invalid native audio")
        native = {"sample_rate": str(native_rate), "channels": channels, "frames": frames,
                  "format": audio.format, "subtype": audio.subtype,
                  "duration_seconds": frames / native_rate}
        metadata = {"backend": BACKEND_ID,
                    "versions": {"soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
                                 "scipy": scipy.__version__, "numpy": np.__version__},
                    "native": native, "sample_rate_hz": target_rate, "channels": 1}
        if native_rate < target_rate:
            return {**metadata, "status": "native_rate_below_target", "full_decode_verified": False}, None
        if max_seconds is not None and frames / native_rate > max_seconds:
            return None
        expected = (frames * target_rate + native_rate - 1) // native_rate
        # Bound live native, mono, resampled and output arrays plus a small
        # conversion workspace. A mono view and equal-rate data are reused.
        required = frames * channels * 4 + expected * 4 + 1024 * 1024
        required += frames * 4 if channels > 1 else 0
        required += expected * 4 if native_rate != target_rate else 0
        if required > max_decoded_bytes:
            return None
        try:
            samples = audio.read(dtype="float32", always_2d=True)
        except RuntimeError as error:
            raise ValueError("Native full decode failed") from error
        if samples.shape != (frames, channels) or audio.tell() != frames:
            raise ValueError("Decoded native frame count differs from header")
    for start in range(0, frames, 65536):
        if not np.isfinite(samples[start:start + 65536]).all():
            raise ValueError("Nonfinite native audio samples")
    mono = samples[:, 0] if channels == 1 else samples.mean(axis=1, dtype=np.float32)
    if native_rate != target_rate:
        divisor = gcd(native_rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, native_rate // divisor,
                             window=("kaiser", 5.0), padtype="constant")
    if len(mono) != expected:
        raise ValueError("Resampled frame count differs from the transform contract")
    output = np.empty(expected, dtype="<i2")
    for start in range(0, expected, 65536):
        block = mono[start:start + 65536]
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite normalized audio")
        rounded = np.rint(np.clip(block, -1.0, 32767 / 32768) * 32768.0)
        output[start:start + len(block)] = rounded.astype("<i2")
    return {**metadata, "status": "decoded", "decoded_frames": expected,
            "native_decoded_frames": frames, "full_decode_verified": True}, output.tobytes()
