"""Freeze downstream cohorts from retained metadata, without reading audio.

Builders deliberately distinguish a usable audio cohort from a ready evaluation
protocol. They never manufacture negatives, independent training pools, or test
splits for datasets that do not supply them.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

SHARED = "/mnt/shared_ru.ml.SZ-2_000180/Iskhakov"
BENCHMARK_ROOT = f"{SHARED}/datasets/rare-event-ssl/2026-11-02/benchmarks"
LEGACY_ROOT = f"{SHARED}/datasets/rare-event-ssl/2026-10-30"
AUDIT_ROOT = f"{SHARED}/reports/rare-event-ssl/2026-11-02/final-audit-reviewed"
CALL_CLASSES = ("K1", "K4", "K5", "K7", "K10", "K12", "K13", "K14", "K17", "K21", "K27")
# Copied as data from common.py; importing that module would import torch/audio.
AVACHA_FOLDS = (
    ("010816-1409_K1K5K7K10K17", "280716-1838_K5K7K17", "070716-1731_K1K12",
     "280616-1355_K1K5K12K17", "060816-1435_K27", "220616-1109_K7K17"),
    ("230716-1546_distantK5K17", "070816-1632_K10K13K21", "010816-1816_K4K10K12K13",
     "040816-1836_K13K27", "010816-1633_K5K7K17", "080816-1600_K4K7"),
    ("010816-1611_K5K7K10K17", "010816-1341_K4", "070816-1640_K1K10",
     "040816-1906_K13K27", "010816-1322_k1k13k14"),
)
TASKS = ("avacha_calls", "avacha_presence", "avacha_field", "watkins", "hiceas",
         "hackathon", "onc", "manual", "dominica", "voices")


class DatasetError(ValueError):
    """Metadata contradicts the declared data contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(path: str | Path) -> str:
    # Large buffered metadata reads avoid one network round trip per small line.
    with Path(path).open(encoding="utf-8-sig", buffering=1024 * 1024) as stream:
        return stream.read()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _text(path).splitlines() if line.strip()]


def _csv(path: str | Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(_text(path))))


def _basename(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).name


def _stem(path: str) -> str:
    return PurePosixPath(_basename(path)).stem


def _duration(record: Mapping[str, Any]) -> float:
    duration = record.get("duration_seconds")
    if duration is None and record.get("frames") is not None:
        rate = record.get("sample_rate_hz", record.get("sample_rate"))
        if rate:
            duration = int(record["frames"]) / float(rate)
    if duration is None or not math.isfinite(float(duration)) or float(duration) <= 0:
        raise DatasetError(f"Missing/invalid verified duration: {record.get('path', record.get('source_id'))}")
    return float(duration)


def _audio(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in records if row.get("role", "audio") == "audio"
            and row.get("full_decode_verified", True) and row.get("size", row.get("bytes", 1)) != 0]


def _row(record: Mapping[str, Any], *, task: str, label: str | int | None,
         group: str, split: str, start: float = 0, end: float | None = None,
         events: list[dict[str, Any]] | None = None, fold: int | None = None) -> dict[str, Any]:
    path = str(record["path"])
    recording = str(record.get("source_id", record.get("relative_path", path)))
    end = _duration(record) if end is None else float(end)
    if not 0 <= float(start) < end <= _duration(record) + 1e-8:
        raise DatasetError(f"Invalid interval for {path}: {start}, {end}")
    result = {"path": path, "recording_id": recording, "group": str(group), "split": split,
              "label": label, "start_seconds": float(start), "end_seconds": end,
              "events": events or [], "fold": fold}
    for key in ("sha256", "pcm_sha256", "sample_rate_hz", "sample_rate", "channels"):
        if record.get(key) is not None:
            result[key] = record[key]
    result["row_id"] = stable_hash({"task": task, **result})
    return result


def _event(start: Any, end: Any, label: Any, duration: float, *, rounding: float = 0) -> dict[str, Any]:
    start, end = float(start), float(end)
    if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end <= start or end > duration + rounding + 1e-9:
        raise DatasetError(f"Annotation outside recording: {start}, {end}, duration={duration}")
    # Keep original annotation endpoints, including explicitly tolerated RTTM rounding.
    return {"start_seconds": start, "end_seconds": end, "label": label}


def _finish(task: str, rows: list[dict[str, Any]], protocol: dict[str, Any],
            reasons: list[str] | None = None, notes: list[str] | None = None,
            inputs: list[str] | None = None) -> dict[str, Any]:
    reasons = list(reasons or [])
    rows.sort(key=lambda r: (r["fold"] if r["fold"] is not None else -1,
                             r["split"], r["recording_id"], r["start_seconds"]))
    seen, assignments, content_assignments = set(), defaultdict(set), defaultdict(set)
    for row in rows:
        if row["row_id"] in seen:
            raise DatasetError("Duplicate immutable row ID")
        seen.add(row["row_id"])
        if row["split"] not in ("train", "val", "test", "unassigned"):
            raise DatasetError(f"Unknown split {row['split']}")
        assignments[(row["fold"], row["group"])].add(row["split"])
        content = row.get("pcm_sha256", row.get("sha256"))
        if content:
            content_assignments[(row["fold"], content)].add(row["split"])
    overlaps = [str(key) for key, splits in assignments.items() if len(splits - {"unassigned"}) > 1]
    if overlaps:
        reasons.append(f"source groups cross splits ({len(overlaps)} groups)")
    duplicate_overlaps = sum(len(splits - {"unassigned"}) > 1 for splits in content_assignments.values())
    if duplicate_overlaps:
        reasons.append(f"exact audio fingerprints cross splits ({duplicate_overlaps} groups)")
    if not rows:
        reasons.append("no eligible verified audio rows")
    result = {"schema_version": 1, "task": task, "status": "ready" if not reasons else "not_ready",
              "ready": not reasons, "reasons": sorted(set(reasons)), "notes": notes or [],
              "protocol": protocol, "rows": rows, "inputs": inputs or [],
              "counts": dict(sorted(Counter(r["split"] for r in rows).items())),
              "label_counts": dict(sorted(Counter(str(r["label"]) for r in rows).items())),
              "group_overlap_examples": overlaps[:10], "duplicate_cross_split_groups": duplicate_overlaps}
    result["fingerprint"] = stable_hash({k: result[k] for k in ("schema_version", "task", "protocol", "rows")})
    return result


def _binary(value: Any) -> int:
    if str(value).strip() not in ("0", "1", "0.0", "1.0"):
        raise DatasetError(f"Not an explicit binary label: {value!r}")
    return int(float(value))


def _split(value: str) -> str:
    return {"validation": "val", "valid": "val", "development": "val", "dev": "val"}.get(value, value)


def _group_assignments(groups: Iterable[str], seed: int = 42) -> dict[str, str]:
    ordered = sorted(set(groups), key=lambda group: stable_hash([seed, group]))
    if len(ordered) < 3:
        return {group: "unassigned" for group in ordered}
    n_test = max(1, round(len(ordered) * .2))
    n_val = max(1, round(len(ordered) * .2))
    return {group: "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for i, group in enumerate(ordered)}


def avacha_source_group(path: str) -> str:
    """Conservative year grouping merges Kam/Kamch and date-string aliases."""
    match = re.search(r"(?:^|_)((?:19|20)\d{2})(?:_|$)", _stem(path))
    return f"avacha:year:{match.group(1)}" if match else "avacha:unresolved"


def build_avacha_clips(records: Iterable[Mapping[str, Any]], *, binary: bool = False,
                       group_splits: Mapping[str, str] | None = None,
                       source_groups: Mapping[str, str] | None = None, seed: int = 42,
                       allow_missing_validation_classes: bool = False) -> dict[str, Any]:
    task = "avacha_presence" if binary else "avacha_calls"
    chosen = []
    audio = _audio(records)
    for record in audio:
        label = record.get("class_from_archive") or PurePosixPath(str(record["path"]).replace("\\", "/")).parent.name
        if label not in CALL_CLASSES and label != "noise":
            raise DatasetError(f"Unexpected Avacha class {label}")
        if binary or label != "noise":
            chosen.append((record, 0 if label == "noise" else 1 if binary else label))
    groups = {str(r["path"]): str((source_groups or {}).get(str(r["path"]), avacha_source_group(str(r["path"])))) for r, _ in chosen}
    quarantined = [path for path, group in groups.items() if group == "avacha:unresolved"]
    quarantined_set = set(quarantined)
    chosen = [(r, label) for r, label in chosen if str(r["path"]) not in quarantined_set]
    groups = {path: group for path, group in groups.items() if path not in quarantined_set}
    # Duplicate content joins otherwise different source groups before assignment.
    parents = {g: g for g in groups.values()}
    def find(g: str) -> str:
        while parents[g] != g:
            parents[g] = parents[parents[g]]
            g = parents[g]
        return g
    hashes: dict[str, str] = {}
    for record, _ in chosen:
        g = groups[str(record["path"])]
        h = record.get("pcm_sha256", record.get("sha256"))
        if h in hashes:
            a, b = find(g), find(hashes[h])
            parents[max(a, b)] = min(a, b)
        elif h:
            hashes[str(h)] = g
    groups = {path: find(g) for path, g in groups.items()}
    assignments = dict(group_splits) if group_splits is not None else _group_assignments(groups.values(), seed)
    rows = [_row(r, task=task, label=label, group=groups[str(r["path"])],
                 split=_split(assignments.get(groups[str(r["path"])], "unassigned"))) for r, label in chosen]
    reasons = []
    if any(row["split"] == "unassigned" for row in rows):
        reasons.append("source groups require explicit train/val/test assignment")
    labels = {0, 1} if binary else set(CALL_CLASSES)
    missing = {split: sorted(map(str, labels - {r["label"] for r in rows if r["split"] == split})) for split in ("train", "val", "test")}
    if missing["train"] or missing["test"] or (missing["val"] and not allow_missing_validation_classes):
        reasons.append(f"class support missing in split: {missing}")
    supports = {split: {str(label): sum(r["label"] == label and r["split"] == split for r in rows)
                        for label in sorted(labels, key=str)} for split in ("train", "val", "test")}
    coverage = {"verified_audio": len(audio), "eligible_audio": len(chosen),
                "quarantined_unresolved_source": len(quarantined),
                "quarantined_paths_sha256": stable_hash(sorted(quarantined)),
                "noise_excluded": sum(r.get("class_from_archive") == "noise" for r in audio) if not binary else 0}
    result = _finish(task, rows, {"definition": "call_vs_noise" if binary else "11_call_types_noise_excluded",
        "split": "explicit_group_manifest" if group_splits is not None else "conservative_year_groups_60_20_20",
        "seed": seed, "official_split": False, "grouping": "explicit" if source_groups else "year",
        "classes": sorted(map(str, labels)), "missing_classes": missing, "coverage": coverage,
        "class_supports": supports, "allow_missing_validation_classes": allow_missing_validation_classes,
        "selection_labels": sorted(str(label) for label in labels if supports["val"][str(label)] > 0)}, reasons,
        ["Year grouping is conservative; this is a newly frozen protocol, not a recovered official split."] +
        ([f"Validation has no examples for {missing['val']}; selection uses supported validation labels only."]
         if missing["val"] and allow_missing_validation_classes else []))
    result["coverage"] = coverage
    result["quarantined_paths"] = sorted(quarantined)
    if not result["ready"]:
        result["status"] = "needs_protocol_input"
    return result


def build_avacha_field(records: Iterable[Mapping[str, Any]], annotations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {_stem(r["path"]): r for r in _audio(records)}
    annotation_map = {str(a["stem"]): a["events"] for a in annotations}
    expected = {stem for fold in AVACHA_FOLDS for stem in fold}
    reasons = []
    if set(indexed) != expected or set(annotation_map) != expected:
        reasons.append("17-field audio/annotation cohort does not match frozen paper folds")
    rows = []
    for fold, test in enumerate(AVACHA_FOLDS):
        others = [f for i, f in enumerate(AVACHA_FOLDS) if i != fold]
        val = {others[0][0], others[1][0]}
        for stem in sorted(expected & set(indexed) & set(annotation_map)):
            record = indexed[stem]
            events = [_event(e["start"], e["end"], e["class"], _duration(record)) for e in annotation_map[stem]]
            if any(e["label"] not in CALL_CLASSES for e in events):
                raise DatasetError("Unknown field call class")
            rows.append(_row(record, task="avacha_field", label=None, group=f"avacha:field:{stem}",
                             split="test" if stem in test else "val" if stem in val else "train", events=events, fold=fold))
    return _finish("avacha_field", rows, {"split": "paper_recording_held_out_3fold", "folds": AVACHA_FOLDS,
        "validation": "first recording of each non-test fold", "annotation_time_scale": "original_seconds",
        "classes": CALL_CLASSES}, reasons,
        ["Kclass source/tape relatives of field test/dev must also be excluded from any combined training protocol."])


def build_watkins(records: Iterable[Mapping[str, Any]], annotations: Iterable[Mapping[str, Any]], *, seed: int = 42,
                  duplicate_policy: str | None = None) -> dict[str, Any]:
    if duplicate_policy not in (None, "holdout_priority"):
        raise DatasetError("Unknown Watkins duplicate policy")
    # Exact upstream sklearn algorithm and annotation CSV order, not filename sort.
    from sklearn.model_selection import train_test_split
    audio = {str(r.get("archive_member", r.get("source_id"))): r for r in _audio(records)}
    eligible = [dict(a) for a in annotations if a["species"] != "Weddell_Seal"]
    paths = [str(a["path"]) for a in eligible]
    if len(set(paths)) != len(paths):
        raise DatasetError("Duplicate Watkins annotation path")
    missing = [path for path in paths if path not in audio]
    if missing:
        return _finish("watkins", [], {"split": "BEANS_60_20_20_seed42", "seed": seed}, [f"missing verified audio: {len(missing)}"])
    indices = list(range(len(eligible)))
    labels = [a["species"] for a in eligible]
    train, remaining = train_test_split(indices, test_size=.4, random_state=seed, shuffle=True, stratify=labels)
    val, test = train_test_split(remaining, test_size=.5, random_state=seed, shuffle=True,
                               stratify=[labels[i] for i in remaining])
    assignment = {i: split for split, values in (("train", train), ("val", val), ("test", test)) for i in values}
    rows, exclusions = [], []
    for i, a in enumerate(eligible):
        record = audio[a["path"]]
        # Exact PCM duplicates share a group and cross-split duplicates block readiness.
        group = "watkins:" + str(record.get("pcm_sha256", a["path"]))
        rows.append(_row(record, task="watkins", label=a["species"], group=group, split=assignment[i]))
    original_counts = dict(Counter(r["split"] for r in rows))
    duplicate_groups = defaultdict(list)
    for row in rows:
        duplicate_groups[row["group"]].append(row)
    conflicts = [group for group, members in duplicate_groups.items() if len({r["label"] for r in members}) > 1]
    if duplicate_policy == "holdout_priority":
        priority = {"test": 0, "val": 1, "train": 2}
        retained = []
        for group, members in duplicate_groups.items():
            if group in conflicts:
                winner = None
            else:
                winner = min(members, key=lambda r: (priority[r["split"]], r["recording_id"]))
                retained.append(winner)
            for row in members:
                if row is not winner:
                    exclusions.append({"path": row["path"], "label": row["label"],
                        "original_split": row["split"], "content_group": group,
                        "reason": "conflicting_label_content" if winner is None else "exact_duplicate_holdout_priority",
                        "retained_path": winner["path"] if winner else None})
        rows = retained
    exclusions.sort(key=lambda r: (r["original_split"], r["path"]))
    classes = set(labels)
    supports = {split: {label: sum(r["split"] == split and r["label"] == label for r in rows)
                        for label in sorted(classes)} for split in ("train", "val", "test")}
    reasons = []
    if any(n == 0 for split in ("train", "test") for n in supports[split].values()):
        reasons.append("clean derived cohort lacks training or test support for some target classes")
    result = _finish("watkins", rows, {"split": "BEANS_60_20_20_seed42" if duplicate_policy is None else "BEANS_derived_holdout_priority",
        "seed": seed, "official_assignment_preserved": True, "duplicate_policy": duplicate_policy,
        "original_counts": original_counts, "class_supports": supports,
        "selection_labels": [label for label, count in supports["val"].items() if count],
        "conflicting_content_groups": len(conflicts), "exclusion_manifest_sha256": stable_hash(exclusions),
        "excluded_species": ["Weddell_Seal"], "annotation_order_preserved": True,
        "source": "https://github.com/earthspecies/beans/blob/main/scripts/watkins.py"}, reasons,
        notes=["Records never move between upstream splits. Conflicting-label content is quarantined with the explicit clean policy.",
               "Tape/session independence is not established from numeric Watkins filenames alone."])
    result["original_counts"] = original_counts
    result["exclusions"] = exclusions
    if not result["ready"]:
        result["status"] = "needs_protocol_input"
    return result


def build_hiceas(records: Iterable[Mapping[str, Any]], splits: Mapping[str, Iterable[Mapping[str, Any]]],
                  *, window_seconds: float = 10, stride_seconds: float = 5) -> dict[str, Any]:
    if window_seconds <= 0 or stride_seconds <= 0:
        raise DatasetError("Window/stride must be positive")
    audio = {_basename(r["path"]): r for r in _audio(records)}
    rows, reasons = [], []
    for name, annotations in splits.items():
        split = _split(name)
        for annotation in annotations:
            record = audio.get(_basename(annotation["path"]))
            if record is None:
                reasons.append(f"missing verified audio: {annotation['path']}")
                continue
            duration = _duration(record)
            if abs(duration - float(annotation["length"])) > 1 / float(record.get("sample_rate_hz", 22050)) + 1e-8:
                raise DatasetError("HICEAS annotation length differs from verified audio")
            events = [_event(a["st"], a["ed"], _binary(a["label"]), duration) for a in annotation["annotations"]]
            # Complete windows only, including [50,60] for a 60-second recording.
            n_windows = max(0, math.floor((duration - window_seconds + 1e-8) / stride_seconds) + 1)
            for i in range(n_windows):
                start, end = i * stride_seconds, i * stride_seconds + window_seconds
                overlaps = [e for e in events if min(end, e["end_seconds"]) > max(start, e["start_seconds"])]
                rows.append(_row(record, task="hiceas", label=int(any(e["label"] == 1 for e in overlaps)),
                    group="hiceas:" + str(record.get("pcm_sha256", record["path"])), split=split,
                    start=start, end=end, events=overlaps))
    return _finish("hiceas", rows, {"split": "supplied_train_valid_test_jsonl", "window_seconds": window_seconds,
        "stride_seconds": stride_seconds, "label_rule": "positive_annotation_any_strict_overlap",
        "tail": "drop_incomplete", "source": "https://github.com/earthspecies/beans/blob/main/datasets.yml"}, reasons)


def _session(record: Mapping[str, Any], family: str) -> str:
    name = _stem(str(record["path"]))
    date = re.search(r"(\d{4})[_-](\d{2})[_-](\d{2})", name)
    if date:
        # deployment-day is deliberately coarser than individual timestamped clips.
        return f"{family}:{name[:date.start()].rstrip('_-')}:{'-'.join(date.groups())}"
    onc = re.match(r"([^_]+)_(\d{8})T", name)
    if onc:
        return f"{family}:{onc.group(1)}:{onc.group(2)}"
    return family + ":" + str(record.get("pcm_sha256", name))


def build_hackathon(records: Iterable[Mapping[str, Any]], annotations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    labels = {}
    for annotation in annotations:
        key = str(annotation["file_name"])
        value = (_split(str(annotation["split"])), _binary(annotation["mammal_present"]))
        if key in labels and labels[key] != value:
            raise DatasetError(f"Conflicting Hackathon metadata for {key}")
        labels[key] = value
    rows, reasons = [], []
    for record in _audio(records):
        path = str(record["relative_path"])
        if path not in labels:
            reasons.append(f"verified audio missing label: {path}")
            continue
        split, label = labels[path]
        if _split(path.split("/")[0]) != split:
            raise DatasetError("Hackathon path and metadata splits disagree")
        rows.append(_row(record, task="hackathon", label=label, group=_session(record, "orcasound"), split=split))
    return _finish("hackathon", rows, {"split": "published_paths_and_metadata", "target": "mammal_present",
        "expected_available_counts": {"train": 3208, "val": 551, "test": 536}}, reasons,
        ["Upstream zero-byte object is excluded by verified audio manifest.", "Orca labels do not establish SRKW ecotype."])


def build_onc(records: Iterable[Mapping[str, Any]], annotations: Iterable[Mapping[str, Any]],
              training_rows: Iterable[Mapping[str, Any]] | None = None,
              *, requires_head_from: str | None = None) -> dict[str, Any]:
    if requires_head_from not in (None, "avacha_presence"):
        raise DatasetError("ONC external protocol requires the avacha_presence head")
    if requires_head_from and training_rows is not None:
        raise DatasetError("ONC external-head protocol cannot fit an ONC training pool")
    labels = {str(a["filename"]): _binary(a["mammal_present"]) for a in annotations}
    rows, reasons = [], []
    for record in _audio(records):
        key = str(record["relative_path"])
        if key not in labels:
            reasons.append(f"verified audio missing expert label: {key}")
            continue
        rows.append(_row(record, task="onc", label=labels[key], group=_session(record, "onc"), split="test"))
    if training_rows is None and not requires_head_from:
        reasons.append("independent labeled train/val pool manifest not supplied")
    elif training_rows is not None:
        training = [dict(r) for r in training_rows]
        if {r.get("split") for r in training} != {"train", "val"}:
            raise DatasetError("ONC training pool must contain train and val only")
        for record in training:
            if record.get("independent_of_test") is not True:
                reasons.append("ONC training pool has no explicit independence audit")
            rows.append(_row(record, task="onc", label=_binary(record["label"]),
                             group=str(record["group"]), split=str(record["split"])))
    return _finish("onc", rows, {"split": "published_expert_external_test" if requires_head_from else "published_expert_test_plus_explicit_independent_pool",
        "requires_head_from": requires_head_from, "fit_or_select_on_onc": False,
        "target": "file_mammal_present", "window_labels": "not_inferred"}, reasons,
        ["386 metadata rows; 385 published usable audio objects. Missing upstream file is not invented."])


def build_manual(records: Iterable[Mapping[str, Any]], labels: Iterable[Mapping[str, Any]],
                 split_lists: Mapping[str, Iterable[str]], rttms: Mapping[str, str], uems: Mapping[str, str],
                 *, provenance_resolved: bool = False, allow_rounded_uem_view: bool = False,
                 allow_partial_provenance: bool = False) -> dict[str, Any]:
    audio = {_stem(r["path"]): r for r in _audio(records) if r.get("dataset") == "manual_sound_noise"}
    covered = {_stem(str(a["audio"])) for a in labels}
    rows, reasons, rounded_overruns, uem_views = [], [], [], []
    for name, stems in split_lists.items():
        split, intervals, events = _split(name), defaultdict(list), defaultdict(list)
        for line in uems.get(name, "").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 4:
                raise DatasetError("Malformed UEM")
            if fields[0] not in audio:
                raise DatasetError("UEM refers to absent verified audio")
            duration = _duration(audio[fields[0]])
            endpoint = float(fields[3])
            if endpoint > duration:
                if not (allow_rounded_uem_view and re.fullmatch(r"\d+\.\d{3}", fields[3])
                        and endpoint - duration <= .0005 + 1e-9):
                    raise DatasetError(f"UEM endpoint outside recording: {fields[0]} {fields[3]} > {duration}")
                uem_views.append({"recording": fields[0], "original_tokens": fields,
                    "original_end_seconds": endpoint, "evaluation_end_seconds": duration,
                    "overrun_seconds": endpoint - duration})
                endpoint = duration
            coverage = _event(fields[2], endpoint, "coverage", duration)
            intervals[fields[0]].append((coverage["start_seconds"], coverage["end_seconds"]))
        for line in rttms.get(name, "").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 10 or fields[0] != "SPEAKER" or fields[7] != "sound":
                raise DatasetError("Unsupported RTTM row")
            if fields[1] not in audio:
                raise DatasetError("RTTM refers to absent verified audio")
            tolerance = .001 if all(re.fullmatch(r"\d+\.\d{3}", fields[i]) for i in (3, 4)) else 0
            event = _event(fields[3], float(fields[3]) + float(fields[4]), "sound",
                           _duration(audio[fields[1]]), rounding=tolerance)
            overrun = event["end_seconds"] - _duration(audio[fields[1]])
            if overrun > 0:
                rounded_overruns.append(overrun)
            events[fields[1]].append(event)
        for stem in stems:
            stem = _stem(stem.strip())
            if not stem or stem not in covered:
                continue  # no matched annotations never implies negative
            if stem not in audio or not intervals[stem]:
                reasons.append(f"missing verified audio/UEM: {stem}")
                continue
            record = audio[stem]
            # Upload prefix removed for recording aliases; exact PCM duplicates share group.
            source_name = re.sub(r"^[0-9a-f]{8}-", "", stem, flags=re.I)
            source_name = re.sub(r"(?:ch|hi|lo)\d+$", "", source_name, flags=re.I).lower()
            group = "manual:" + source_name
            for start, end in intervals[stem]:
                selected = [e for e in events[stem] if min(end, e["end_seconds"]) > max(start, e["start_seconds"])]
                rows.append(_row(record, task="manual", label=None, group=group, split=split,
                                 start=start, end=end, events=selected))
    if not provenance_resolved and not allow_partial_provenance:
        reasons.append("original source-family provenance unresolved")
    selected_stems = {_stem(row["path"]) for row in rows}
    exclusions = [{"path": record["path"], "reason": "no_matched_annotations" if stem not in covered
                   else "not_in_preserved_native_split_with_UEM"}
                  for stem, record in audio.items() if stem not in selected_stems]
    exclusions.sort(key=lambda r: r["path"])
    coverage = {"verified_audio": len(audio), "matched_annotation_audio": len(set(audio) & covered),
                "evaluation_audio": len(selected_stems), "excluded_audio": len(exclusions)}
    result = _finish("manual", rows, {"split": "preserved_pyannote_lists_rttm_uem", "coverage": "matched_annotation_files_and_UEM_only",
        "coverage_counts": coverage, "exclusion_manifest_sha256": stable_hash(exclusions),
        "target": "sound", "unmatched_files_are_negatives": False, "rttm_rounding_tolerance_seconds": .001,
        "allow_rounded_uem_view": allow_rounded_uem_view, "uem_half_quantum_seconds": .0005,
        "uem_views": uem_views, "auxiliary": not provenance_resolved,
        "tolerated_rounding_count": len(rounded_overruns),
        "max_tolerated_overrun_seconds": max(rounded_overruns, default=0)}, reasons,
        ["Nine unmatched files are excluded; original annotation files are retained."] +
        (["Auxiliary only: original source-family provenance remains unresolved."] if not provenance_resolved else []))
    result["auxiliary"] = not provenance_resolved
    result["exclusions"] = exclusions
    result["coverage"] = coverage
    if result["ready"] and result["auxiliary"]:
        result["status"] = "ready_auxiliary"
    return result


def build_dominica(records: Iterable[Mapping[str, Any]], group_manifest: Iterable[Mapping[str, Any]] | None = None,
                    *, allow_new_group_split: bool = False, annotations: Iterable[Mapping[str, Any]] | None = None,
                    seed: int = 42) -> dict[str, Any]:
    selected = _audio(r for r in records if r.get("dataset") == "dominica")
    if allow_new_group_split:
        if group_manifest is not None or annotations is None:
            raise DatasetError("New Dominica protocol requires CSV clicks and no competing group manifest")
        from sklearn.model_selection import train_test_split
        by_name = {_basename(r["path"]): r for r in selected}
        if len(by_name) != len(selected):
            raise DatasetError("Dominica CSV basename mapping is ambiguous across original recordings")
        clicks = defaultdict(list)
        for annotation in annotations:
            name = str(annotation["Recording name"])
            if name not in by_name:
                raise DatasetError(f"Click annotation has no verified audio: {name}")
            time = float(annotation["Click time of arrivals (ToA) [sec]"])
            if not math.isfinite(time) or not 0 <= time <= _duration(by_name[name]):
                raise DatasetError(f"Click time outside audio: {name} {time}")
            clicks[name].append({"time_seconds": time, "label": "click"})
        groups = defaultdict(list)
        group_labels = defaultdict(set)
        for record in selected:
            parts = str(record["path"]).replace("\\", "/").split("/")
            if "Signal_parts" in parts:
                label = 1
                if not clicks[_basename(record["path"])]:
                    raise DatasetError("Signal recording has no CSV click annotations")
            elif "Noise_parts" in parts:
                label = 0
                if clicks[_basename(record["path"])]:
                    raise DatasetError("Noise recording has positive click annotations")
            else:
                raise DatasetError("Unknown Dominica signal/noise source folder")
            group = "dominica:" + str(record.get("pcm_sha256", record.get("sha256", record["path"])))
            groups[group].append((record, label))
            group_labels[group].add(label)
        conflicts = {g for g, labels in group_labels.items() if len(labels) != 1}
        eligible_groups = sorted(set(groups) - conflicts)
        labels = [next(iter(group_labels[group])) for group in eligible_groups]
        train, remaining = train_test_split(eligible_groups, test_size=.4, random_state=seed,
                                           shuffle=True, stratify=labels)
        val, test = train_test_split(remaining, test_size=.5, random_state=seed, shuffle=True,
                                     stratify=[next(iter(group_labels[group])) for group in remaining])
        assignment = {g: split for split, values in (("train", train), ("val", val), ("test", test)) for g in values}
        rows, exclusions = [], []
        for group, members in groups.items():
            for record, label in members:
                if group in conflicts:
                    exclusions.append({"path": record["path"], "reason": "conflicting_signal_noise_content", "group": group})
                    continue
                rows.append(_row(record, task="dominica", label=label, group=group, split=assignment[group],
                    events=sorted(clicks[_basename(record["path"])], key=lambda e: e["time_seconds"])))
        result = _finish("dominica", rows, {"split": "new_recording_groups_stratified_60_20_20", "official_split": False,
            "seed": seed, "grouping": "whole_recordings_union_exact_content", "annotation_kind": "point",
            "click_annotations": sum(len(r["events"]) for r in rows), "target": "signal_noise_and_clicks",
            "exclusion_manifest_sha256": stable_hash(exclusions)}, notes=[
                "New protocol: source session independence beyond full recordings is not established.",
                "Signal/noise channel counts differ; a common channel/preprocessing policy is required."])
        result["exclusions"] = exclusions
        return result
    if group_manifest is None:
        return _finish("dominica", [], {"split": "explicit_source_group_manifest_required"},
                       ["official split not recovered; explicit reviewed group/label/interval manifest required"],
                       [f"{len(selected)} verified original audio files retained; no random split invented."])
    indexed = {str(r["path"]): r for r in selected}
    rows = []
    for entry in group_manifest:
        record = indexed[str(entry["path"])]
        if not entry.get("group"):
            raise DatasetError("Dominica explicit manifest requires source group")
        events = [_event(e["start_seconds"], e["end_seconds"], e["label"], _duration(record)) for e in entry.get("events", [])]
        rows.append(_row(record, task="dominica", label=entry.get("label"), group=str(entry["group"]),
            split=_split(str(entry["split"])), start=float(entry.get("start_seconds", 0)),
            end=entry.get("end_seconds"), events=events))
    reasons = [] if {r["split"] for r in rows} == {"train", "val", "test"} else ["explicit manifest must provide train/val/test"]
    return _finish("dominica", rows, {"split": "explicit_reviewed_group_manifest", "official_split": False}, reasons)


def normalize_species(value: str) -> str:
    """Case/separator normalization only; no taxonomic synonym guessing."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def build_voices(records: Iterable[Mapping[str, Any]], mapping_manifest: Iterable[Mapping[str, Any]] | None = None,
                 *, requires_head_from: str | None = None,
                 watkins_records: Iterable[Mapping[str, Any]] | None = None,
                 watkins_annotations: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if requires_head_from is not None:
        if requires_head_from != "watkins" or watkins_records is None or watkins_annotations is None:
            raise DatasetError("Voices external protocol requires Watkins head, verified PCM hashes and ontology")
        if mapping_manifest is not None:
            raise DatasetError("Exact normalized mapping cannot be combined with an override mapping")
        ontology = {}
        for annotation in watkins_annotations:
            label = str(annotation["species"])
            if label == "Weddell_Seal":
                continue
            normalized = normalize_species(label)
            if normalized in ontology and ontology[normalized] != label:
                raise DatasetError("Ambiguous normalized Watkins ontology")
            ontology[normalized] = label
        training_audio = _audio(watkins_records)
        if any(not r.get("pcm_sha256") for r in training_audio):
            raise DatasetError("Watkins raw PCM fingerprint audit is incomplete")
        watkins_hashes = {r["pcm_sha256"] for r in training_audio}
        audio = _audio(records)
        codec_groups = defaultdict(list)
        for record in audio:
            relative = PurePosixPath(str(record.get("relative_path", record["path"])))
            codec_groups[str(relative.with_suffix(""))].append(record)
        lossless = {".wav": 0, ".flac": 1, ".aiff": 2, ".aif": 3, ".au": 4}
        rows, exclusions, unmatched, mapping, seen_pcm = [], [], set(), {}, set()
        for group, variants in sorted(codec_groups.items()):
            candidates = [r for r in variants if PurePosixPath(r["path"]).suffix.lower() in lossless
                          and not str(r.get("subtype", "")).startswith("MPEG")]
            chosen = min(candidates, key=lambda r: (lossless[PurePosixPath(r["path"]).suffix.lower()], r["path"])) if candidates else None
            for record in variants:
                if record is not chosen:
                    exclusions.append({"path": record["path"], "reason": "codec_alternative" if chosen else "no_lossless_variant", "group": group})
            if chosen is None:
                continue
            if not chosen.get("pcm_sha256"):
                raise DatasetError("Voices raw PCM fingerprint audit is incomplete")
            species = PurePosixPath(group).parent.name
            label = ontology.get(normalize_species(species))
            if label is None:
                unmatched.add(species)
                exclusions.append({"path": chosen["path"], "reason": "no_exact_shared_species", "species": species, "group": group})
                continue
            mapping[species] = label
            h = chosen["pcm_sha256"]
            if h in watkins_hashes or h in seen_pcm:
                exclusions.append({"path": chosen["path"], "reason": "exact_pcm_overlap_watkins" if h in watkins_hashes else "exact_pcm_duplicate_voices", "group": group})
                continue
            seen_pcm.add(h)
            rows.append(_row(chosen, task="voices", label=label, group="voices:" + group, split="test"))
        exclusions.sort(key=lambda r: r["path"])
        result = _finish("voices", rows, {"role": "external_test", "requires_head_from": "watkins",
            "fit_or_select_on_voices": False, "mapping": "exact_case_separator_normalized_shared_species",
            "ontology_mapping": mapping, "unmatched_species": sorted(unmatched), "codec_policy": "one_lossless_per_parent_stem",
            "raw_pcm_against_watkins": "all_verified_original_watkins_audio", "auxiliary": True,
            "coverage": {"original_audio": len(audio), "codec_groups": len(codec_groups), "eligible_examples": len(rows)},
            "exclusion_manifest_sha256": stable_hash(exclusions)}, notes=[
                "Auxiliary external examples; unmatched species are N/A, not scored as guessed labels.",
                "Exact native PCM audit does not prove absence of cropped/resampled/cross-codec relatives."])
        result["exclusions"] = exclusions
        result["auxiliary"] = True
        if result["ready"]:
            result["status"] = "ready_auxiliary"
        return result
    if mapping_manifest is None:
        return _finish("voices", [], {"role": "external_test"},
                       ["reviewed ontology mapping and train-pool dedup audit not supplied"])
    audio = {str(r["path"]): r for r in _audio(records)}
    rows, reasons = [], []
    for entry in mapping_manifest:
        if entry.get("dedup_cleared") is not True or not entry.get("training_ontology"):
            reasons.append("ontology/dedup audit incomplete")
        record = audio[str(entry["path"])]
        rows.append(_row(record, task="voices", label=entry["label"], group=str(entry["group"]), split="test"))
    return _finish("voices", rows, {"role": "external_test", "split": "explicit_mapping_manifest"}, reasons,
                   ["No fitting, model selection, or threshold tuning on this external example set."])


def build_dataset(task: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read known metadata paths and return a self-contained immutable cohort.

    ``config`` accepts metadata path overrides, not audio-directory globs.
    Missing inputs and contradictory metadata become explicit not_ready results.
    The pure builders raise DatasetError for contradictions; this batch-facing
    adapter isolates them so one blocked task does not erase other task reports.
    """
    cfg = dict(config or {})
    if task not in TASKS:
        raise DatasetError(f"Unknown task: {task}")
    paths: list[str] = []
    metadata_hashes: dict[str, str] = {}
    def read(path: str) -> str:
        paths.append(path)
        content = Path(path).read_bytes()
        metadata_hashes[path] = hashlib.sha256(content).hexdigest()
        return content.decode("utf-8-sig")
    def jl(key: str, default: str) -> list[dict[str, Any]]:
        path = str(cfg.get(key, default))
        return [json.loads(line) for line in read(path).splitlines() if line.strip()]
    def js(key: str) -> Any:
        return json.loads(read(str(cfg[key])))
    def table(key: str, default: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(read(str(cfg.get(key, default))))))
    try:
        if task in ("avacha_calls", "avacha_presence"):
            records = jl("audio_manifest", f"{LEGACY_ROOT}/animal_spot_trainingfiles/manifests/audio_files.jsonl")
            result = build_avacha_clips(records, binary=task == "avacha_presence",
                group_splits=js("group_splits") if cfg.get("group_splits") else None,
                source_groups=js("source_groups") if cfg.get("source_groups") else None, seed=int(cfg.get("seed", 42)),
                allow_missing_validation_classes=cfg.get("allow_missing_validation_classes") is True)
        elif task == "avacha_field":
            records = jl("audio_manifest", f"{LEGACY_ROOT}/avacha_field/manifests/audio_files.jsonl")
            path = str(cfg.get("annotations", f"{LEGACY_ROOT}/avacha_field/manifests/annotations.json"))
            result = build_avacha_field(records, json.loads(read(path)))
        elif task == "watkins":
            result = build_watkins(jl("audio_manifest", f"{AUDIT_ROOT}/beans_watkins_verified.jsonl"),
                table("annotations", f"{BENCHMARK_ROOT}/beans_watkins/extracted/annotations.csv"), seed=int(cfg.get("seed", 42)),
                duplicate_policy=cfg.get("duplicate_policy"))
        elif task == "hiceas":
            records = jl("audio_manifest", f"{AUDIT_ROOT}/beans_hiceas_verified.jsonl")
            splits = {name: jl(name, f"{BENCHMARK_ROOT}/beans_hiceas/extracted/{filename}.jsonl")
                      for name, filename in (("train", "train"), ("val", "valid"), ("test", "test"))}
            result = build_hiceas(records, splits)
        elif task in ("hackathon", "onc"):
            dataset = "orcasound_hackathon2024" if task == "hackathon" else "onc_benchmark"
            records = jl("audio_manifest", f"{BENCHMARK_ROOT}/{dataset}/manifest.jsonl")
            annotations = table("annotations", f"{BENCHMARK_ROOT}/{dataset}/raw/metadata.csv")
            result = build_hackathon(records, annotations) if task == "hackathon" else build_onc(records, annotations,
                jl("training_manifest", "") if cfg.get("training_manifest") else None, requires_head_from=cfg.get("requires_head_from"))
        elif task == "manual":
            records = jl("audio_manifest", f"{AUDIT_ROOT}/local_verified.jsonl")
            labels = table("annotations", f"{BENCHMARK_ROOT}/manual_sound_noise/raw/manifest.csv")
            lists, rttms, uems = {}, {}, {}
            for name, lst in (("train", "train"), ("val", "development"), ("test", "test")):
                rttm = "dev" if name == "val" else name
                for target, key, default in ((lists, "list", f"lists/{lst}.lst"), (rttms, "rttm", f"rttms/{rttm}.rttm"), (uems, "uem", f"uems/{rttm}.uem")):
                    path = str(cfg.get(f"{name}_{key}", f"{BENCHMARK_ROOT}/manual_sound_noise/pyannote/{default}"))
                    value = read(path)
                    target[name] = value.splitlines() if key == "list" else value
            result = build_manual(records, labels, lists, rttms, uems,
                provenance_resolved=cfg.get("provenance_resolved") is True,
                allow_rounded_uem_view=cfg.get("allow_rounded_uem_view") is True,
                allow_partial_provenance=cfg.get("allow_partial_provenance") is True)
        elif task == "dominica":
            result = build_dominica(jl("audio_manifest", f"{AUDIT_ROOT}/local_verified.jsonl"),
                jl("group_manifest", "") if cfg.get("group_manifest") else None,
                allow_new_group_split=cfg.get("allow_new_group_split") is True,
                annotations=table("annotations", f"{BENCHMARK_ROOT}/dominica/raw/Annotations_Dominica.csv")
                            if cfg.get("allow_new_group_split") is True else None, seed=int(cfg.get("seed", 42)))
        else:
            result = build_voices(jl("audio_manifest", f"{BENCHMARK_ROOT}/voices_in_the_sea/manifest.jsonl"),
                jl("mapping_manifest", "") if cfg.get("mapping_manifest") else None,
                requires_head_from=cfg.get("requires_head_from"),
                watkins_records=jl("watkins_audio_manifest", f"{AUDIT_ROOT}/beans_watkins_verified.jsonl")
                    if cfg.get("requires_head_from") else None,
                watkins_annotations=table("watkins_annotations", f"{BENCHMARK_ROOT}/beans_watkins/extracted/annotations.csv")
                    if cfg.get("requires_head_from") else None)
    except (FileNotFoundError, ImportError) as error:
        result = _finish(task, [], {}, [f"required input/dependency unavailable: {error}"])
    except ValueError as error:
        result = _finish(task, [], {}, [f"metadata contract not satisfied: {error}"])
        result["status"] = "needs_protocol_input"
    result["inputs"] = paths
    result["input_metadata_sha256"] = metadata_hashes
    identity = {k: result[k] for k in ("schema_version", "task", "protocol", "rows")}
    # Metadata locations may be Windows Y: overrides or the equivalent Linux mount.
    identity["input_metadata_sha256"] = sorted(metadata_hashes.values())
    result["fingerprint"] = stable_hash(identity)
    return result


def prepare_datasets(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = config.get("datasets", config)
    return {task: build_dataset(task, cfg or {}) for task, cfg in datasets.items()}


def readiness_report(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {"schema_version": 1, "ready": all(r["ready"] for r in results.values()) and bool(results),
            "datasets": {task: {k: result[k] for k in ("status", "ready", "reasons", "counts", "fingerprint", "notes")}
                         for task, result in results.items()}}


def save_manifest(result: Mapping[str, Any], path: str | Path) -> Path:
    """Exclusive publication; an existing different freeze is never overwritten."""
    path = Path(path)
    data = canonical_json(result) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
    except FileExistsError:
        if _text(path) != data:
            raise DatasetError(f"Immutable manifest already exists with different content: {path}") from None
    return path
