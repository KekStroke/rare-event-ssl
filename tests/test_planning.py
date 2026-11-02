import json
from pathlib import Path
import tempfile
import unittest

from marine_ssl.planning import build_plan


def onc(sensor, day, clock="000000", **extra):
    name = f"{sensor}_{day}T{clock}.000Z"
    return {"source_family": "onc", "source_id": name,
            "path_in_source": f"train/{name}.wav", "hydrophone_id": sensor,
            "deployment_id": sensor, "session_id": f"{sensor}:{day}",
            "recording_start_utc": f"{day[:4]}-{day[4:6]}-{day[6:]}T00:00:00+00:00",
            **extra}


class SelectedEvaluationPlanningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = {
            "seed": 7, "sample_rate": 16000, "segment_seconds": 5,
            "protected_catalogs": ["ONC-Benchmark.jsonl", "orcasound_hackathon2024.jsonl"],
            "sources": {
                "dori_onc": {"source_family": "onc", "catalog": "onc.jsonl", "target_hours": 900},
                "orcasound": {"source_family": "orcasound", "catalog": "orca.jsonl", "target_hours": 300},
            },
        }
        self.write("ONC-Benchmark.jsonl", [onc("ICLISTENAF1000", "20240101")])
        self.write("orcasound_hackathon2024.jsonl", [{"source_family": "orcasound",
                   "source_id": "selected-hackathon-recording", "session_id": "orcasound:20240202"}])
        self.write("orca.jsonl", [])

    def write(self, name, rows):
        (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_selected_sources_and_sessions_are_excluded_without_reserving_entire_sensor(self):
        same_recording = onc("ICLISTENAF1000", "20240101")
        same_session = onc("ICLISTENAF1000", "20240101", "003000")
        next_day = onc("ICLISTENAF1000", "20240102")
        unknown = {"source_family": "onc", "source_id": "opaque-source"}
        self.write("onc.jsonl", [same_recording, same_session, next_day, unknown])
        self.write("orca.jsonl", [{"source_family": "orcasound", "source_id": "different-crop",
                                  "session_id": "orcasound:20240202"}])
        plan = build_plan(self.config, self.root)
        self.assertEqual("metadata_candidates_only", plan["summary"]["status"])
        self.assertEqual([next_day["source_id"]], [row["source_id"] for row in plan["candidates"]])
        self.assertNotIn("deployment_id", plan["candidates"][0])
        excluded = {row["source_id"]: row["exclusion_reasons"] for row in plan["rejected"]}
        self.assertIn("shared_source_id", excluded[same_recording["source_id"]])
        self.assertIn("shared_session_id", excluded[same_session["source_id"]])
        self.assertIn("shared_session_id", excluded["different-crop"])
        self.assertIn("unresolved_pretrain_provenance", excluded["opaque-source"])

    def test_unselected_dclde_provider_does_not_block_but_missing_selected_catalog_does(self):
        self.write("dclde.jsonl", [{"source_family": "onc", "source_id": "unselected-unknown-source"}])
        self.write("onc.jsonl", [onc("ICLISTENAF2000", "20240102")])
        plan = build_plan(self.config, self.root)
        self.assertEqual("metadata_candidates_only", plan["summary"]["status"])
        self.assertEqual(1, len(plan["candidates"]))
        self.assertNotIn("dclde_error", plan["summary"])
        self.assertNotIn("protocol_blockers", plan["summary"])
        (self.root / "ONC-Benchmark.jsonl").unlink()
        blocked = build_plan(self.config, self.root)
        self.assertEqual("blocked", blocked["summary"]["status"])
        self.assertEqual([], blocked["candidates"])


if __name__ == "__main__":
    unittest.main()
