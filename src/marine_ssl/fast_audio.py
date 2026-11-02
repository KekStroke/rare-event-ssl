"""Bounded short-file decoding; None explicitly delegates to the FFmpeg path."""

from math import gcd
import io
from pathlib import Path
import struct

BACKEND_ID = "soundfile-scipy-polyphase-pcm16-v2"


def _wav_source(path, size, allow_repair):
    """Validate RIFF; explicitly recover only bounded legacy PCM16 containers."""
    with path.open("rb") as stream:
        head = stream.read(12)
        if len(head) < 12 or head[:4] not in {b"RIFF", b"RIFX"}:
            return None, None
        order = ">" if head[:4] == b"RIFX" else "<"
        declared = struct.unpack(order + "I", head[4:8])[0]
        if declared == 0xFFFFFFFF:
            return None, None
        if head[8:12] != b"WAVE":
            raise ValueError("Invalid WAV container")
        end = declared + 8
        problem = end > size
        fmt, data_offset, data_count = None, None, None
        while stream.tell() < min(end, size):
            chunk = stream.read(8)
            if len(chunk) != 8:
                problem = True
                break
            count = struct.unpack(order + "I", chunk[4:])[0]
            offset = stream.tell()
            if chunk[:4] == b"fmt " and count >= 16:
                fmt = stream.read(16)
                stream.seek(offset)
            if chunk[:4] == b"data":
                if data_offset is not None:
                    raise ValueError("Multiple WAV data chunks are unsupported")
                data_offset, data_count = offset, count
            if offset + count > min(end, size):
                problem = True
                break
            stream.seek(offset + count + (count & 1))
        if not problem:
            return path, None
        if not allow_repair:
            raise ValueError("Truncated or invalid WAV container/chunk")
        if head[:4] != b"RIFF" or fmt is None or len(fmt) != 16 or data_offset is None:
            raise ValueError("Legacy WAV repair requires known little-endian PCM")
        code, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", fmt)
        if (code != 1 or bits != 16 or not 1 <= channels <= 8 or rate <= 0
                or align != channels * 2 or byte_rate != rate * align
                or not data_count or data_count % align):
            raise ValueError("Legacy WAV repair requires aligned PCM16 metadata")
        physical = max(0, size - data_offset)
        available = min(data_count, physical)
        usable = available - available % align
        missing = data_count - available
        dropped = available - usable
        if not usable:
            raise ValueError("No complete PCM frames available")
        if missing or dropped:
            declared_seconds = data_count / byte_rate
            if (data_count - usable) / byte_rate > min(0.1, 0.001 * declared_seconds):
                raise ValueError("Legacy WAV truncation exceeds the bounded recovery policy")
            quality = "available_prefix"
        else:
            quality = "metadata_trailer_removed"
        stream.seek(data_offset)
        pcm = stream.read(usable)
        if len(pcm) != usable:
            raise ValueError("Legacy WAV payload changed during validation")
    header = b"RIFF" + struct.pack("<I", 36 + usable) + b"WAVEfmt " + struct.pack("<I", 16)
    header += fmt + b"data" + struct.pack("<I", usable)
    repair = {"quality": quality, "original_file_bytes": size,
              "original_riff_declared_bytes": end, "original_data_declared_bytes": data_count,
              "data_offset": data_offset, "physical_data_bytes_available": physical,
              "pcm_frames_retained": usable // align, "missing_declared_bytes": missing,
              "missing_declared_frames": (data_count - usable) // align,
              "dropped_partial_frame_bytes": dropped,
              "ignored_trailing_metadata_bytes": max(0, physical - data_count),
              "source_fully_present": not (missing or dropped),
              "policy": "legacy_pcm16_no_added_samples_max_missing_min_0.1s_or_0.1percent"}
    return io.BytesIO(header + pcm), repair


def decode_pcm16(path, target_rate=16000, max_seconds=3600,
                 max_source_bytes=64 * 1024**2, max_decoded_bytes=256 * 1024**2,
                 allow_header_repair=False):
    """Return (metadata, full mono PCM16 bytes), or None for explicit fallback.

    A below-target native rate returns (skip metadata, None). Unknown codecs or
    missing optional dependencies delegate to FFmpeg. Recognized decode errors,
    nonfinite samples and truncated RIFF files raise ValueError. No five-second
    padding or segmentation occurs here. Header repair is strict-off by default;
    callers may opt in only for approved legacy Orcasound sources. It preserves
    existing complete PCM frames and records every omitted byte/frame. Downmix is the equal-weight mean;
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
    audio_source, repair = path, None
    if suffix in {".wav", ".wave"}:
        audio_source, repair = _wav_source(path, source_bytes, allow_header_repair)
        if audio_source is None:
            return None
    try:
        audio = sf.SoundFile(str(audio_source) if isinstance(audio_source, Path) else audio_source)
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
        if repair:
            metadata.update(container_repair=repair, quality=repair["quality"])
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
