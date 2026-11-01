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


if __name__ == "__main__":
    unittest.main()
