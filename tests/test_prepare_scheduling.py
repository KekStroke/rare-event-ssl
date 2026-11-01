"""Scheduling regression: a slow archive must not block ready sources."""

import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

prepare_module = importlib.import_module("marine_ssl.prepare")


class PrepareSchedulingTests(unittest.TestCase):
    def test_ready_source_runs_while_mbari_is_still_downloading(self):
        fast_processed = threading.Event()
        active, maximum, slow_observed = {}, {}, []
        lock = threading.Lock()
        sources = {"mbari": "pacific_sound", "noaa": "sanctsound"}
        config = {"sample_rate": 16000, "channels": 1, "sample_format": "PCM_16",
                  "segment_seconds": 5, "overlap_seconds": 0, "reserve_bytes": 0,
                  "sources": {key: {"source_family": family, "target_hours": 10 / 3600}
                              for key, family in sources.items()}}
        rows = [{"source_key": key, "source_family": family, "source_id": f"{key}-{number}",
                 "clearance_scope": "source_metadata_only", "bytes": 1}
                for key, family in sources.items() for number in range(2)]

        def transfer(row, *_):
            source = row["source_key"]
            with lock:
                active[source] = active.get(source, 0) + 1
                maximum[source] = max(maximum.get(source, 0), active[source])
            try:
                if row["source_id"] == "mbari-0":
                    slow_observed.append(fast_processed.wait(2))
                else:
                    time.sleep(0.01)
                return Path("unused-original"), None
            finally:
                with lock:
                    active[source] -= 1

        def convert(row, raw, scratch, *args):
            if row["source_key"] == "noaa":
                fast_processed.set()
            checksum = hashlib.sha256(row["source_id"].encode()).hexdigest()
            temporary = scratch / "test.tar.part"
            temporary.write_bytes(b"mock prepared content")
            return {"source": row, "selected_frames": 80000, "duration_seconds": 5,
                    "segment_count": 1, "normalized_pcm_sha256": checksum,
                    "segments": [{"pcm_sha256": checksum}]}, temporary

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(prepare_module, "_source", side_effect=transfer), \
                    patch.object(prepare_module, "_convert", side_effect=convert), \
                    patch.object(prepare_module.shutil, "which", return_value="mock-tool"):
                summary = prepare_module.prepare(config, rows, output, workers=2)
            receipts = [json.loads(path.read_text()) for path in (output / "receipts").glob("*/*.json")]
        self.assertEqual(slow_observed, [True], "ready NOAA conversion was blocked by slow MBARI")
        self.assertEqual(maximum, {"mbari": 1, "noaa": 1})
        self.assertTrue(all(source["quota_met"] for source in summary["sources"].values()))
        self.assertEqual(sorted(row["selection_order"] for row in receipts), list(range(4)))
