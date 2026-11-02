import hashlib
import importlib.util
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

from marine_ssl.acquire import acquire, confined, extract_zip, validate_audio


HAS_SOUNDFILE = importlib.util.find_spec("soundfile") is not None


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_confined_rejects_traversal_and_absolute_paths(self):
        for path in ("../outside.wav", "/outside.wav", "..\\outside.wav", "C:/outside.wav"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                confined(self.root / "raw", path)

    def test_archive_traversal_cannot_write_outside_dataset(self):
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../outside.txt", b"untrusted")
        with self.assertRaises(ValueError):
            extract_zip(archive, self.root / "extracted")
        self.assertFalse((self.root / "outside.txt").exists())

    def test_archive_symlinks_are_rejected(self):
        archive = self.root / "source.zip"
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr(member, "../outside")
        with self.assertRaises(ValueError):
            extract_zip(archive, self.root / "extracted")

    def test_confined_rejects_existing_symlink_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        raw = self.root / "raw"
        raw.mkdir()
        try:
            (raw / "linked").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("OS/account does not permit creating symlinks")
        with self.assertRaises(ValueError):
            confined(raw, "linked/audio.wav")

    def test_capacity_refusal_happens_before_any_network_transfer(self):
        plan = {"datasets": {"large": {"files": [{"dataset": "large", "size": 300,
                                                  "role": "archive"}]}}}
        with patch("marine_ssl.acquire.shutil.disk_usage") as usage, patch("marine_ssl.acquire.download") as transfer:
            usage.return_value.free = 999
            with self.assertRaises(RuntimeError):
                acquire(plan, self.root / "output", reserve_bytes=400)
            transfer.assert_not_called()

    def test_zip_expanded_bytes_preserve_capacity_reserve(self):
        archive = self.root / "compressed.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            stream.writestr("large.txt", b"0" * 10000)
        with patch("marine_ssl.acquire.shutil.disk_usage") as usage:
            usage.return_value.free = 5000
            with self.assertRaises(RuntimeError):
                extract_zip(archive, self.root / "extracted", reserve_bytes=100)
        self.assertFalse((self.root / "extracted" / "large.txt").exists())

    def test_unknown_sizes_are_refused_before_download(self):
        plan = {"datasets": {"unknown": {"files": [{"dataset": "unknown", "size": None,
                                                    "role": "audio"}]}}}
        with patch("marine_ssl.acquire.download") as transfer:
            with self.assertRaises(ValueError):
                acquire(plan, self.root / "output", reserve_bytes=0)
            transfer.assert_not_called()

    def test_mismatched_dataset_cannot_escape_output(self):
        plan = {"datasets": {"safe": {"files": [{"dataset": "../outside", "size": 1,
                                                 "role": "metadata"}]}}}
        with patch("marine_ssl.acquire.download") as transfer:
            with self.assertRaises(ValueError):
                acquire(plan, self.root / "output", reserve_bytes=0)
            transfer.assert_not_called()

    @unittest.skipUnless(HAS_SOUNDFILE, "soundfile is installed in the acquisition environment")
    def test_full_native_wav_decode_retains_exact_frame_contract(self):
        source = self.root / "original.wav"
        samples = struct.pack("<" + "h" * 160, *range(-80, 80))
        with wave.open(str(source), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(samples)
        result = validate_audio(source)
        self.assertEqual(160, result["frames"])
        self.assertEqual(16000, result["sample_rate_hz"])
        self.assertEqual(1, result["channels"])
        self.assertTrue(result["full_decode_verified"])
        self.assertEqual(result["pcm_sha256"], validate_audio(source)["pcm_sha256"])
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.write(source, "nested/original.wav")
        rows = extract_zip(archive, self.root / "extracted")
        self.assertEqual(1, len(rows))
        self.assertEqual(result["pcm_sha256"], rows[0]["pcm_sha256"])
        self.assertEqual(hashlib.sha256(source.read_bytes()).digest(),
                         hashlib.sha256(Path(rows[0]["path"]).read_bytes()).digest())

    @unittest.skipUnless(HAS_SOUNDFILE, "soundfile is installed in the acquisition environment")
    def test_corrupt_audio_cannot_be_reported_verified(self):
        source = self.root / "broken.wav"
        source.write_bytes(b"RIFFnot-a-wave")
        with self.assertRaises(Exception):
            validate_audio(source)


if __name__ == "__main__":
    unittest.main()
