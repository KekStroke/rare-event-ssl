import json
from pathlib import Path
import tempfile
import unittest

from marine_ssl.audit import audit_manifests


class ActualManifestAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def prepared(self, source):
        receipt = self.root / "receipt.json"
        receipt.write_text(json.dumps({"full_decode_verified": True, "decoded_frames": 16000 * 300,
                                      "sample_rate_hz": 16000, "source_sha256": "a" * 64}))
        return self.write("prepared.jsonl", [{"source": source, "receipt_path": str(receipt)}])

    def test_transport_ids_join_pinned_catalog_original_recording(self):
        name = "ICLISTENAF1000_20240101T000000.000Z"
        source = {"source_family": "onc", "source_id": name, "session_id": "ICLISTENAF1000:20240101"}
        prepared = self.prepared(source)
        self.write("ONC-Benchmark.jsonl", [{**source, "dataset": "ONC-Benchmark",
                                          "path_in_source": name + ".flac"}])
        benchmark = self.write("benchmark.jsonl", [{"dataset": "onc_benchmark", "source_family": "onc",
                       "source_id": "hf:DORI-SRKW/ONC-Benchmark@different-commit/" + name + ".flac",
                       "relative_path": name + ".flac", "role": "audio", "full_decode_verified": True,
                       "sha256": "b" * 64, "duration_seconds": 300}])
        result = audit_manifests(prepared, [benchmark], self.root)
        self.assertEqual("overlap", result["status"])
        self.assertIn("shared_source_id", result["overlap_reason_counts"])
        self.assertIn("shared_session_id", result["overlap_reason_counts"])
        self.assertFalse(result["metadata_clear"])

    def test_unknown_selected_orcasound_provenance_never_claims_clean(self):
        prepared = self.prepared({"source_family": "orcasound", "source_id": "original-session",
                                  "session_id": "orcasound:20240101"})
        benchmark = self.write("selected.jsonl", [{"dataset": "orcasound_hackathon2024",
                   "source_family": "orcasound", "source_id": "hf:repo@commit/opaque.flac",
                   "relative_path": "opaque.flac", "role": "audio", "full_decode_verified": True}])
        self.write("unselected.jsonl", [{"source_family": "onc", "source_id": "unused"}])
        result = audit_manifests(prepared, [benchmark], self.root)
        self.assertEqual("partial", result["status"])
        self.assertEqual(["orcasound"], result["unresolved_shared_families"])
        self.assertEqual(1, result["verified_benchmark_audio_checked"])
        self.assertFalse(result["metadata_clear"])
        missing = audit_manifests(prepared, [benchmark, self.root / "missing.jsonl"], self.root)
        self.assertIn(str(self.root / "missing.jsonl"), missing["missing_manifests"])


if __name__ == "__main__":
    unittest.main()
