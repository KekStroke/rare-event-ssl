import json
import tempfile
import unittest
from pathlib import Path

from marine_ssl.experiments.datasets import (
    AVACHA_FOLDS, CALL_CLASSES, DatasetError, build_avacha_clips,
    build_avacha_field, build_dataset, build_dominica, build_hackathon,
    build_hiceas, build_manual, build_onc, build_voices, build_watkins,
    readiness_report, save_manifest, stable_hash,
)


def audio(path, duration=60, **extra):
    return {"path": path, "duration_seconds": duration, "full_decode_verified": True,
            "role": "audio", "pcm_sha256": stable_hash(path), **extra}


class ExperimentDatasetTests(unittest.TestCase):
    def test_avacha_definitions_grouping_quarantine_and_immutable_hash(self):
        records = [audio(f"/clips/{label}/x_Kam_{year}_August1_001_002.wav", 1,
                         class_from_archive=label)
                   for year in range(2000, 2010) for label in (*CALL_CLASSES, "noise")]
        records.append(audio("/clips/noise/unknown.wav", 1, class_from_archive="noise"))
        calls = build_avacha_clips(records)
        binary = build_avacha_clips(records, binary=True)
        self.assertTrue(calls["ready"], calls["reasons"])
        self.assertTrue(binary["ready"], binary["reasons"])
        self.assertEqual(set(calls["label_counts"]), set(CALL_CLASSES))
        self.assertEqual(set(binary["label_counts"]), {"0", "1"})
        self.assertEqual(binary["coverage"]["quarantined_unresolved_source"], 1)
        self.assertFalse(calls["protocol"]["official_split"])
        self.assertEqual(calls["fingerprint"], build_avacha_clips(reversed(records))["fingerprint"])
        self.assertFalse(any("unknown.wav" in row["path"] for row in binary["rows"]))
        # No file-level fallback when only two independent years exist.
        too_few = build_avacha_clips(records[:24])
        self.assertFalse(too_few["ready"])
        self.assertTrue(any("assignment" in reason for reason in too_few["reasons"]))

    def test_avacha_exact_duplicates_union_years_before_split(self):
        records = [audio(f"/clips/{label}/x_Kam_{year}_01081234_001_002.wav", 1,
                         class_from_archive=label)
                   for year in range(2000, 2010) for label in (*CALL_CLASSES, "noise")]
        duplicate = dict(records[0], path="/clips/K1/x_Kamch_2015_August1_001_002.wav")
        result = build_avacha_clips(records + [duplicate])
        pair = [r for r in result["rows"] if r["pcm_sha256"] == records[0]["pcm_sha256"]]
        self.assertEqual(len(pair), 2)
        self.assertEqual(len({r["group"] for r in pair}), 1)
        self.assertEqual(len({r["split"] for r in pair}), 1)

    def test_missing_validation_class_requires_explicit_policy_and_uses_val_support(self):
        records = [audio(f"/clips/{label}/x_Kam_{year}_001_002.wav", 1, class_from_archive=label)
                   for year in range(2000, 2003) for label in CALL_CLASSES
                   if not (year == 2001 and label == "K27")]
        assignments = {"avacha:year:2000": "train", "avacha:year:2001": "val", "avacha:year:2002": "test"}
        self.assertFalse(build_avacha_clips(records, group_splits=assignments)["ready"])
        allowed = build_avacha_clips(records, group_splits=assignments, allow_missing_validation_classes=True)
        self.assertTrue(allowed["ready"])
        self.assertNotIn("K27", allowed["protocol"]["selection_labels"])
        self.assertEqual(allowed["protocol"]["class_supports"]["val"]["K27"], 0)
        # The policy never permits a missing train/test class.
        records = [r for r in records if not ("2002" in r["path"] and "/K27/" in r["path"])]
        self.assertFalse(build_avacha_clips(records, group_splits=assignments,
                                          allow_missing_validation_classes=True)["ready"])

    def test_field_exact_paper_folds_and_annotation_bounds(self):
        records = [audio(f"/field/{stem}.wav") for fold in AVACHA_FOLDS for stem in fold]
        annotations = [{"stem": Path(r["path"]).stem,
                        "events": [{"start": 1, "end": 2, "class": "K1"}]} for r in records]
        result = build_avacha_field(records, annotations)
        self.assertTrue(result["ready"])
        self.assertEqual(len(result["rows"]), 51)
        for fold, expected in enumerate((6, 6, 5)):
            rows = [r for r in result["rows"] if r["fold"] == fold]
            self.assertEqual(sum(r["split"] == "test" for r in rows), expected)
            self.assertEqual(sum(r["split"] == "val" for r in rows), 2)
        self.assertFalse(build_avacha_field(records[:-1], annotations)["ready"])
        annotations[0]["events"][0]["end"] = 61
        with self.assertRaises(DatasetError):
            build_avacha_field(records, annotations)

    def test_watkins_matches_upstream_csv_order_and_detects_duplicate_leakage(self):
        from sklearn.model_selection import train_test_split
        annotations = [{"path": f"{label}/{i}.wav", "species": label}
                       for label in ("A", "B") for i in range(10)]
        annotations += [{"path": f"Weddell_Seal/{i}.wav", "species": "Weddell_Seal"} for i in range(2)]
        records = [audio("/watkins/" + a["path"], archive_member=a["path"]) for a in annotations]
        result = build_watkins(records, annotations)
        self.assertTrue(result["ready"])
        self.assertEqual(result["counts"], {"train": 12, "val": 4, "test": 4})
        train, rem = train_test_split(list(range(20)), test_size=.4, random_state=42,
                                     shuffle=True, stratify=[a["species"] for a in annotations[:20]])
        expected_train = {records[i]["path"] for i in train}
        self.assertEqual({r["path"] for r in result["rows"] if r["split"] == "train"}, expected_train)
        test_path = next(r["path"] for r in result["rows"] if r["split"] == "test")
        train_record = next(r for r in records if r["path"] in expected_train)
        next(r for r in records if r["path"] == test_path)["pcm_sha256"] = train_record["pcm_sha256"]
        leaked = build_watkins(records, annotations)
        self.assertFalse(leaked["ready"])
        self.assertEqual(leaked["duplicate_cross_split_groups"], 1)
        cleaned = build_watkins(records, annotations, duplicate_policy="holdout_priority")
        self.assertTrue(cleaned["ready"], cleaned["reasons"])
        self.assertEqual(cleaned["counts"], {"train": 11, "val": 4, "test": 4})
        self.assertEqual(cleaned["original_counts"], result["counts"])
        self.assertEqual(cleaned["exclusions"][0]["original_split"], "train")
        self.assertIn(test_path, {r["path"] for r in cleaned["rows"]})
        # Different-label content is quarantined everywhere, including test.
        next(r for r in records if r["path"] == test_path)["pcm_sha256"] = stable_hash("conflict")
        # Ensure the conflicting pair has different source labels.
        test_label = next(a["species"] for a in annotations if test_path.endswith(a["path"]))
        other_label = next(r for r in records if r["archive_member"].split("/")[0] != test_label
                           and not r["archive_member"].startswith("Weddell_Seal/"))
        other_label["pcm_sha256"] = stable_hash("conflict")
        conflicts = build_watkins(records, annotations, duplicate_policy="holdout_priority")
        self.assertEqual(sum(e["reason"] == "conflicting_label_content" for e in conflicts["exclusions"]), 2)
        self.assertFalse(any(r["pcm_sha256"] == stable_hash("conflict") for r in conflicts["rows"]))

    def test_hiceas_complete_ten_second_windows_keep_source_split(self):
        records = [audio("/hiceas/one.wav"), audio("/hiceas/two.wav", 9)]
        splits = {"test": [{"path": "data/hiceas/wav/one.wav", "length": 60,
                            "annotations": [{"st": 0, "ed": 5, "label": 1}]}],
                  "train": [{"path": "data/hiceas/wav/two.wav", "length": 9, "annotations": []}]}
        result = build_hiceas(records, splits)
        self.assertEqual(result["counts"], {"test": 11})
        self.assertEqual(sum(r["label"] for r in result["rows"]), 1)  # touching 5s boundary is not overlap
        self.assertEqual(result["rows"][-1]["end_seconds"], 60)
        splits["train"] = splits["test"]
        self.assertFalse(build_hiceas(records, splits)["ready"])

    def test_hackathon_preserves_metadata_and_flags_session_leakage(self):
        paths = ["train/positive/000/rpi-lab_2021_01_01_01_00_00.flac",
                 "validation/negative/000/rpi-lab_2021_01_02_01_00_00.flac",
                 "test/positive/000/rpi-lab_2021_01_03_01_00_00.flac"]
        records = [audio("/raw/" + path, relative_path=path) for path in paths]
        annotations = [{"file_name": p, "split": p.split("/")[0], "mammal_present": i != 1}
                       for i, p in enumerate(paths)]
        # CSV binary values must be explicit numbers, not Python booleans.
        for a in annotations:
            a["mammal_present"] = int(a["mammal_present"])
        self.assertTrue(build_hackathon(records, annotations)["ready"])
        records[2]["path"] = records[2]["path"].replace("2021_01_03", "2021_01_01")
        result = build_hackathon(records, annotations)
        self.assertFalse(result["ready"])
        self.assertTrue(result["group_overlap_examples"])
        annotations[0]["split"] = "test"
        with self.assertRaises(DatasetError):
            build_hackathon(records, annotations)

    def test_test_only_sets_do_not_invent_training_protocols(self):
        record = audio("/onc/ICLISTENHF1266_20230101T172500.flac", relative_path="test/a.flac")
        result = build_onc([record], [{"filename": "test/a.flac", "mammal_present": 1}])
        self.assertFalse(result["ready"])
        self.assertEqual(result["counts"], {"test": 1})
        external = build_onc([record], [{"filename": "test/a.flac", "mammal_present": 1}],
                             requires_head_from="avacha_presence")
        self.assertTrue(external["ready"])
        self.assertFalse(external["protocol"]["fit_or_select_on_onc"])
        self.assertFalse(build_voices([record])["ready"])
        self.assertFalse(build_dominica([dict(record, dataset="dominica")])["ready"])

    def test_manual_preserves_native_coverage_and_only_precise_rttm_rounding(self):
        records = [audio("/manual/1234abcd-75017CH1.wav", 10, dataset="manual_sound_noise"),
                   audio("/manual/no-label.wav", 10, dataset="manual_sound_noise")]
        labels = [{"audio": "1234abcd-75017CH1.wav"}]
        lists = {"train": ["1234abcd-75017CH1", "no-label"]}
        rttms = {"train": "SPEAKER 1234abcd-75017CH1 1 9.001 1.000 <NA> <NA> sound <NA> <NA>"}
        uems = {"train": "1234abcd-75017CH1 NA 0.000 10.000"}
        result = build_manual(records, labels, lists, rttms, uems)
        self.assertFalse(result["ready"])  # provenance remains unresolved
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["group"], "manual:75017")
        self.assertEqual(result["coverage"], {"verified_audio": 2, "matched_annotation_audio": 1,
                                             "evaluation_audio": 1, "excluded_audio": 1})
        self.assertEqual(result["exclusions"][0]["reason"], "no_matched_annotations")
        self.assertAlmostEqual(result["rows"][0]["events"][0]["end_seconds"], 10.001)
        rttms["train"] = rttms["train"].replace("9.001", "9.002")
        with self.assertRaises(DatasetError):
            build_manual(records, labels, lists, rttms, uems)
        rttms["train"] = rttms["train"].replace("9.002", "9.0007")
        with self.assertRaises(DatasetError):
            build_manual(records, labels, lists, rttms, uems)

    def test_uem_half_quantum_view_keeps_original_tokens_and_auxiliary_status(self):
        records = [audio("/manual/1234abcd-75017.wav", 9.9996, dataset="manual_sound_noise")]
        labels = [{"audio": "1234abcd-75017.wav"}]
        lists = {"train": ["1234abcd-75017"]}
        rttms = {"train": "SPEAKER 1234abcd-75017 1 1.000 1.000 <NA> <NA> sound <NA> <NA>"}
        uems = {"train": "1234abcd-75017 NA 0.000 10.000"}
        with self.assertRaises(DatasetError):
            build_manual(records, labels, lists, rttms, uems)
        result = build_manual(records, labels, lists, rttms, uems,
                              allow_rounded_uem_view=True, allow_partial_provenance=True)
        self.assertTrue(result["ready"])
        self.assertEqual(result["status"], "ready_auxiliary")
        self.assertEqual(result["rows"][0]["end_seconds"], 9.9996)
        self.assertEqual(result["protocol"]["uem_views"][0]["original_tokens"][-1], "10.000")
        for invalid in ("10.001", "10.0000"):
            with self.assertRaises(DatasetError):
                build_manual(records, labels, lists, rttms, {"train": uems["train"].replace("10.000", invalid)},
                             allow_rounded_uem_view=True, allow_partial_provenance=True)

    def test_voices_exact_ontology_lossless_selection_and_pcm_overlap(self):
        records = [audio("/voices/blueWhale/clip.wav", relative_path="baleen/blueWhale/clip.wav"),
                   audio("/voices/blueWhale/clip.mp3", relative_path="baleen/blueWhale/clip.mp3"),
                   audio("/voices/mysteryWhale/clip.wav", relative_path="baleen/mysteryWhale/clip.wav"),
                   audio("/voices/blueWhale/overlap.wav", relative_path="baleen/blueWhale/overlap.wav", pcm_sha256="overlap")]
        watkins = [audio("/watkins/Blue_Whale/a.wav", pcm_sha256="overlap")]
        result = build_voices(records, requires_head_from="watkins", watkins_records=watkins,
                              watkins_annotations=[{"species": "Blue_Whale"}])
        self.assertTrue(result["ready"])
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["label"], "Blue_Whale")
        self.assertTrue(result["rows"][0]["path"].endswith("clip.wav"))
        self.assertEqual({e["reason"] for e in result["exclusions"]},
                         {"codec_alternative", "no_exact_shared_species", "exact_pcm_overlap_watkins"})

    def test_dominica_new_protocol_is_opt_in_stratified_groups_with_point_clicks(self):
        records = [audio(f"/dominica/{folder}/{i}.wav", dataset="dominica")
                   for folder in ("Signal_parts", "Noise_parts") for i in range(10)]
        # Signal/noise basenames must identify different original recordings.
        for record in records:
            if "Signal_parts" in record["path"]:
                record["path"] = record["path"].replace("Signal_parts/", "Signal_parts/signal")
        annotations = [{"Recording name": f"signal{i}.wav", "Click time of arrivals (ToA) [sec]": "0.5"} for i in range(10)]
        self.assertFalse(build_dominica(records)["ready"])
        result = build_dominica(records, allow_new_group_split=True, annotations=annotations)
        self.assertTrue(result["ready"])
        self.assertEqual(result["counts"], {"train": 12, "val": 4, "test": 4})
        self.assertEqual(result["protocol"]["click_annotations"], 10)
        self.assertFalse(result["protocol"]["official_split"])
        self.assertEqual(next(r for r in result["rows"] if r["label"] == 1)["events"][0],
                         {"time_seconds": .5, "label": "click"})

    def test_publication_refuses_changed_freeze_and_missing_input_reports_not_ready(self):
        result = build_voices([])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "freeze.json"
            save_manifest(result, path)
            save_manifest(result, path)
            changed = dict(result, notes=["different protocol"])
            with self.assertRaises(DatasetError):
                save_manifest(changed, path)
            self.assertEqual(json.loads(path.read_text())["fingerprint"], result["fingerprint"])
            result = build_dataset("watkins", {"audio_manifest": str(Path(directory) / "missing.jsonl")})
            self.assertFalse(result["ready"])
            self.assertFalse(readiness_report({"watkins": result})["ready"])


if __name__ == "__main__":
    unittest.main()
