import unittest
from marine_ssl.collect import _normalize_orcasound
from marine_ssl.planning import legacy_calendar_identity, legacy_year_is_separated, selected_orcasound_test_years


class LegacyOrcasoundTests(unittest.TestCase):
    def test_explicit_legacy_years_are_preserved_without_utc(self):
        for year in (2009, 2012):
            row = {"source_family": "orcasound", "source_id": "original-id",
                   "path_in_source": f"wholistener/archive/Max_103_10_31_{year}_15_10_06.mp3",
                   "recording_start_utc": "2020-01-01T00:00:00Z"}
            result = _normalize_orcasound(row)
            self.assertEqual(result["source_id"], "original-id")
            self.assertEqual(result["legacy_calendar_year"], year)
            self.assertEqual(result["time_basis"], "unspecified_local_clock")
            self.assertNotIn("recording_start_utc", result)
            self.assertTrue(legacy_year_is_separated(year, {2017, 2019, 2021}))

    def test_adjacent_and_unknown_years_are_blocked(self):
        for year, protected in ((2016, {2017}), (2017, {2017}), (2009, None), (2009, set()), (None, {2017})):
            self.assertFalse(legacy_year_is_separated(year, protected))
        self.assertIsNone(selected_orcasound_test_years([{"source_family": "orcasound", "source_id": "unknown"}]))
        self.assertEqual(selected_orcasound_test_years([{"source_family": "orcasound", "source_id": "x", "session_id": "orcasound:20170927"}]), {2017})

    def test_invalid_or_unrequested_names_stay_unknown(self):
        for path in ("wholistener/call-examples.mp3", "wholistener/Max_103_02_31_2009_15_10_06.mp3", "other/Max_103_10_31_2009_15_10_06.mp3"):
            self.assertNotIn("legacy_calendar_year", legacy_calendar_identity({"path_in_source": path}))

    def test_timestamp_variants_share_one_candidate(self):
        import json
        from pathlib import Path
        import tempfile
        from marine_ssl.planning import build_plan
        names = ["PWRdB128_04_08_2009_00_48_00voice.wav",
                 "PWRdB128_04_08_2009_00_48_00voice2_L.wav",
                 "PWRdB128_04_08_2009_00_49_00voice.wav"]
        rows = [{"source_family": "orcasound", "source_id": name[:-4], "url": "https://example/" + name,
                 "path_in_source": "wholistener/archive/wholistener/lk/detections/" + name} for name in names]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "test.jsonl").write_text(json.dumps({"source_family": "orcasound", "source_id": "test",
                                                        "session_id": "orcasound:20170927"}) + "\n")
            (root / "legacy.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = {"protected_catalogs": ["test.jsonl"], "seed": 1, "sample_rate": 16000,
                      "segment_seconds": 5, "sources": {"orcasound": {"source_family": "orcasound",
                                                                       "catalog": "legacy.jsonl", "target_hours": 300}}}
            plan = build_plan(config, root)
        self.assertEqual(len(plan["candidates"]), 2)
        self.assertEqual(len(plan["rejected"]), 1)
        self.assertIn("duplicate_source_or_hash", plan["rejected"][0]["exclusion_reasons"])
        self.assertEqual(plan["candidates"][0]["source_id"] in {row["source_id"] for row in rows}, True)
        self.assertEqual(len({row["source_recording_id"] for row in plan["candidates"]}), 2)
