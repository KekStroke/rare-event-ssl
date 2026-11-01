import unittest

from marine_ssl.leakage import ExclusionIndex, audit, canonical_source_id, filter_pretrain


def recording(**changes):
    row = {
        "source_family": "orcasound", "source_id": "s3://archive/original.wav",
        "session_id": "session-a", "hydrophone_id": "sensor-1",
        "recording_start_utc": "2024-01-01T00:00:00Z",
        "recording_end_utc": "2024-01-01T00:05:00Z",
    }
    return {**row, **changes}


class LeakageTests(unittest.TestCase):
    def test_shared_source_and_session_despite_changed_crop_name(self):
        protected = recording()
        crop = recording(source_id="s3://derived/crop-57.wav")
        self.assertIn("shared_session_id", ExclusionIndex([protected]).reasons(crop))

    def test_shifted_crop_uses_absolute_sensor_time(self):
        candidate = recording(source_id="different.wav", session_id="other-session",
                              recording_start_utc="2024-01-01T03:04:59+03:00",
                              recording_end_utc="2024-01-01T03:06:00+03:00")
        self.assertIn("overlapping_recording_time", ExclusionIndex([recording()]).reasons(candidate))

    def test_adjacent_intervals_do_not_overlap(self):
        candidate = recording(source_id="different.wav", session_id="other-session",
                              recording_start_utc="2024-01-01T00:05:00Z",
                              recording_end_utc="2024-01-01T00:06:00Z")
        self.assertEqual((), ExclusionIndex([recording()]).reasons(candidate))

    def test_exact_hash_crosses_source_namespaces(self):
        index = ExclusionIndex([recording(sha256="a" * 64, pcm_sha256="b" * 64)])
        candidate = {"source_family": "pacific_sound", "sha256": "a" * 64}
        self.assertEqual(("identical_sha256",), index.reasons(candidate))
        self.assertEqual(("identical_pcm_sha256",), index.reasons({**candidate, "sha256": "", "pcm_sha256": "b" * 64}))

    def test_identical_generic_basename_is_not_recording_identity(self):
        index = ExclusionIndex([recording(source_id="s3://one/clip.wav")])
        self.assertEqual((), index.reasons(recording(source_id="s3://two/clip.wav", session_id="b",
                                                    hydrophone_id="sensor-2")))
        self.assertEqual((), index.reasons(recording(source_family="pacific_sound")))

    def test_onc_original_name_matches_different_hf_publications(self):
        name = "ICLISTENAF2548_20210501T001103.000Z.wav"
        protected = {"source_family": "ONC-Benchmark", "source_id": f"https://huggingface.co/datasets/DORI-SRKW/ONC-Benchmark/resolve/abc/audio/{name}"}
        candidate = {"source_family": "DORI-ONC", "source_id": f"https://huggingface.co/datasets/DORI-SRKW/DORI-ONC/resolve/def/test/onc_{name}?download=true"}
        reasons = ExclusionIndex([protected]).reasons(candidate)
        self.assertIn("shared_onc_recording", reasons)
        self.assertIn("shared_session_id", reasons)
        self.assertNotIn("unresolved_pretrain_provenance", reasons)

    def test_onc_same_sensor_day_is_reserved(self):
        protected = {"source_family": "onc", "source_id": "ICLISTENAF2548_20210501T001103.000Z.wav"}
        candidate = {"source_family": "onc", "source_id": "ICLISTENAF2548_20210501T110000.000Z.wav"}
        self.assertIn("shared_session_id", ExclusionIndex([protected]).reasons(candidate))
        self.assertEqual((), ExclusionIndex([protected]).reasons({**candidate, "source_id": "ICLISTENAF2548_20210502T110000.000Z.wav"}))

    def test_unknown_provenance_fails_closed_only_in_shared_family(self):
        index = ExclusionIndex([recording()])
        self.assertIn("unresolved_pretrain_provenance", index.reasons({"source_family": "orcasound", "source_id": "crop.wav"}))
        self.assertEqual((), index.reasons({"source_family": "pacific_sound", "source_id": "crop.wav"}))
        self.assertIn("unknown_source_family", index.reasons({"source_id": "crop.wav"}))

    def test_unresolved_test_sources_quarantine_entire_family(self):
        index = ExclusionIndex([{"source_family": "DORI-Orcasound", "source_id": "crop.wav"}])
        self.assertIn("unresolved_protected_provenance", index.reasons(recording()))
        with self.assertRaises(ValueError):
            ExclusionIndex([{"source_id": "unknown-family.wav"}])

    def test_missing_required_protected_family_stays_quarantined(self):
        index = ExclusionIndex([], at_risk_families=["orcasound"])
        self.assertIn("unresolved_protected_provenance", index.reasons(recording()))

    def test_session_without_original_source_is_unresolved(self):
        index = ExclusionIndex([recording()])
        candidate = recording(source_id="", session_id="other", hydrophone_id="sensor-2")
        self.assertIn("unresolved_pretrain_provenance", index.reasons(candidate))

    def test_interval_index_preserves_gaps_after_merging(self):
        protected = [recording(), recording(source_id="b.wav", session_id="b",
                                            recording_start_utc="2024-01-01T00:03:00Z",
                                            recording_end_utc="2024-01-01T00:06:00Z"),
                     recording(source_id="c.wav", session_id="c",
                               recording_start_utc="2024-01-01T00:10:00Z",
                               recording_end_utc="2024-01-01T00:11:00Z")]
        index = ExclusionIndex(protected)
        gap = recording(source_id="gap.wav", session_id="gap",
                        recording_start_utc="2024-01-01T00:06:00Z",
                        recording_end_utc="2024-01-01T00:10:00Z")
        self.assertEqual((), index.reasons(gap))
        self.assertIn("overlapping_recording_time", index.reasons(
            {**gap, "recording_end_utc": "2024-01-01T00:10:01Z"}))

    def test_deployment_holdout_overrides_time_separation(self):
        index = ExclusionIndex([recording(deployment_id="deployment-1")])
        candidate = recording(source_id="different.wav", session_id="b", deployment_id="deployment-1",
                              recording_start_utc="2025-01-01T00:00:00Z", recording_end_utc="2025-01-01T01:00:00Z")
        self.assertIn("shared_deployment_id", index.reasons(candidate))

    def test_invalid_and_naive_times_are_not_cleared(self):
        index = ExclusionIndex([recording()])
        for changes in ({"recording_start_utc": "2024-01-01T00:00:00"},
                        {"recording_end_utc": ""},
                        {"recording_end_utc": "2023-01-01T00:00:00Z"},
                        {"sha256": "not-a-hash"}):
            self.assertIn("invalid_provenance", index.reasons(recording(**changes)))

    def test_partition_preserves_input_and_audit_matches(self):
        rows = [recording(), {"source_family": "pacific_sound", "source_id": "clean.wav"}]
        index = ExclusionIndex([recording()])
        accepted, rejected = filter_pretrain(rows, index)
        self.assertEqual(1, len(accepted))
        self.assertEqual(1, len(rejected))
        self.assertNotIn("exclusion_reasons", rows[0])
        self.assertEqual({"total": 2, "accepted": 1, "excluded": 1},
                         {key: audit(rows, index)[key] for key in ("total", "accepted", "excluded")})

    def test_hf_normalization_preserves_repository_and_drops_revision(self):
        source = "https://huggingface.co/datasets/owner/repo/resolve/deadbeef/train/a.wav?download=true"
        self.assertEqual("hf://owner/repo/train/a.wav", canonical_source_id(source))
        self.assertNotEqual(canonical_source_id(source), canonical_source_id(source.replace("owner/repo", "other/repo")))


if __name__ == "__main__":
    unittest.main()
