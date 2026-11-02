import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from marine_ssl.fast_audio import BACKEND_ID, decode_pcm16

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "soundfile"))


@unittest.skipUnless(AVAILABLE, "Run in the recorded audio acquisition environment")
class FastAudioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def wav(self, rate, values, channels=1):
        path = self.root / "audio.wav"
        pcm = struct.pack("<" + "h" * len(values), *values)
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(channels)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(pcm)
        return path, pcm

    def test_pcm16_identity_and_repeated_decode_are_stable(self):
        path, original = self.wav(16000, list(range(-1000, 1000)))
        metadata, pcm = decode_pcm16(path)
        self.assertEqual(original, pcm)
        self.assertEqual((metadata, pcm), decode_pcm16(path))
        self.assertEqual(BACKEND_ID, metadata["backend"])
        self.assertEqual(2000, metadata["decoded_frames"])
        self.assertTrue(metadata["full_decode_verified"])

    def test_resampling_stereo_count_and_short_tail_are_not_padded(self):
        frames, rate = 44101, 44100
        path, _ = self.wav(rate, [12000, -12000] * frames, channels=2)
        metadata, pcm = decode_pcm16(path)
        self.assertEqual((frames * 16000 + rate - 1) // rate, metadata["decoded_frames"])
        self.assertEqual(16001 * 2, len(pcm))
        self.assertEqual(b"\0" * len(pcm), pcm)
        self.assertEqual(frames, metadata["native_decoded_frames"])

    def test_alias_filter_keeps_passband_and_rejects_high_frequency(self):
        import numpy as np
        import soundfile as sf
        rate, count = 48000, 48000
        time = np.arange(count) / rate
        path = self.root / "tone.flac"
        sf.write(path, 0.25 * np.sin(2 * np.pi * 1000 * time), rate, subtype="PCM_16")
        _, low = decode_pcm16(path)
        sf.write(path, 0.25 * np.sin(2 * np.pi * 12000 * time), rate, subtype="PCM_16")
        _, high = decode_pcm16(path)
        low_rms = np.sqrt(np.mean(np.frombuffer(low, dtype="<i2")[100:-100].astype(float)**2))
        high_rms = np.sqrt(np.mean(np.frombuffer(high, dtype="<i2")[100:-100].astype(float)**2))
        self.assertGreater(low_rms, 4000)
        self.assertLess(high_rms, low_rms / 100)

    def test_memory_duration_bounds_fallback_and_low_rate_skip(self):
        path, _ = self.wav(16000, [100] * 16000)
        self.assertIsNone(decode_pcm16(path, max_decoded_bytes=1024))
        self.assertIsNone(decode_pcm16(path, max_source_bytes=10))
        self.assertIsNone(decode_pcm16(path, max_seconds=0.5))
        path, _ = self.wav(8000, [100] * 8000)
        metadata, pcm = decode_pcm16(path)
        self.assertEqual("native_rate_below_target", metadata["status"])
        self.assertIsNone(pcm)

    def test_truncated_wav_is_rejected(self):
        path, _ = self.wav(16000, [100] * 16000)
        path.write_bytes(path.read_bytes()[:-10])
        with self.assertRaises(ValueError):
            decode_pcm16(path)

    def test_opt_in_repairs_malformed_non_audio_trailer_without_changing_raw(self):
        path, _ = self.wav(16000, [12000, -12000] * 176000, channels=2)
        raw = bytearray(path.read_bytes())
        raw += b"INFO-\x00INFOIARTWHO|-\x00PortTownsend|legacy_recording.wav"
        raw[4:8] = struct.pack("<I", len(raw) - 8)
        original = bytes(raw)
        path.write_bytes(original)
        with self.assertRaises(ValueError):
            decode_pcm16(path)
        metadata, pcm = decode_pcm16(path, allow_header_repair=True)
        self.assertEqual("metadata_trailer_removed", metadata["quality"])
        self.assertEqual(176000 * 2, len(pcm))
        self.assertEqual(0, metadata["container_repair"]["missing_declared_frames"])
        self.assertGreater(metadata["container_repair"]["ignored_trailing_metadata_bytes"], 0)
        self.assertEqual(original, path.read_bytes())

    def test_opt_in_tiny_truncation_retains_only_complete_available_frames(self):
        path, _ = self.wav(16000, [1000, -1000] * 160160, channels=2)
        original = path.read_bytes()[:-7]
        path.write_bytes(original)
        metadata, pcm = decode_pcm16(path, allow_header_repair=True)
        repair = metadata["container_repair"]
        self.assertEqual("available_prefix", metadata["quality"])
        self.assertEqual(160158, metadata["decoded_frames"])
        self.assertEqual(160158 * 2, len(pcm))
        self.assertEqual(2, repair["missing_declared_frames"])
        self.assertEqual(7, repair["missing_declared_bytes"])
        self.assertEqual(1, repair["dropped_partial_frame_bytes"])
        self.assertFalse(repair["source_fully_present"])
        self.assertEqual(original, path.read_bytes())

    def test_opt_in_still_rejects_major_truncation(self):
        path, _ = self.wav(16000, [1000] * 16000)
        path.write_bytes(path.read_bytes()[:-1000])
        for flag in (False, True):
            with self.subTest(allow_header_repair=flag), self.assertRaises(ValueError):
                decode_pcm16(path, allow_header_repair=flag)


if __name__ == "__main__":
    unittest.main()
