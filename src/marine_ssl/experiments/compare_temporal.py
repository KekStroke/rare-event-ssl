"""Paired source-group confidence intervals for immutable temporal predictions.

Ranked AP is evaluated from exact per-recording maximum-matching gains. Cluster
replication then weights these independent graphs and score ties; it never lets
copies of a recording match each other's references. This is equivalent to
materializing replicas with unique recording IDs, with bounded bootstrap memory.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .metrics import _augment, _edges, _events, _score_groups, match_events


SECONDARY_AP_OPERATION_BUDGET = 200_000_000


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _artifact_paths(path):
    path = Path(path)
    directory = path if path.is_dir() or path.suffix != ".jsonl" else path.parent
    predictions = directory / "temporal_predictions.jsonl" if directory == path else path
    return predictions, directory / "metrics.json"


def _load(path):
    predictions, metadata = _artifact_paths(path)
    missing = [str(item) for item in (predictions, metadata) if not item.is_file()]
    if missing:
        return None, missing
    raw, meta_raw = predictions.read_bytes(), metadata.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    meta = json.loads(meta_raw.decode("utf-8-sig"))
    by_id = {}
    for row in rows:
        if row["row_id"] in by_id:
            raise ValueError("Duplicate temporal row IDs cannot be silently deduplicated")
        by_id[row["row_id"]] = row
        if not math.isfinite(float(row["duration_seconds"])) or row["duration_seconds"] <= 0:
            raise ValueError("Positive finite source exposure is required")
        if any(event.get("recording_id") != row["recording_id"] for event in
               row["references"] + row["predictions"] + row.get("call_predictions", [])):
            raise ValueError("Event recording ID differs from its immutable source row")
    if meta.get("status") not in ("completed", "complete", "complete_auxiliary"):
        raise ValueError("Temporal metric artifact is not a completed evaluation")
    if not meta.get("dataset_fingerprint"):
        raise ValueError("Missing immutable dataset fingerprint")
    if not meta.get("folds") or any(fold.get("selection", {}).get("data") != "validation_only" for fold in meta["folds"]):
        raise ValueError("Temporal heads lack validation-only selection provenance")
    return {"rows": by_id, "metadata": meta, "artifacts": {
        "predictions": str(predictions), "predictions_sha256": hashlib.sha256(raw).hexdigest(),
        "metrics": str(metadata), "metrics_sha256": hashlib.sha256(meta_raw).hexdigest()}}, []


def _strict_pair(a, b):
    ma, mb = a["metadata"], b["metadata"]
    for key in ("task", "dataset_fingerprint", "labels", "config"):
        if key not in ma or key not in mb or _canonical(ma[key]) != _canonical(mb[key]):
            raise ValueError(f"Paired temporal {key} mismatch")
    if ma["task"] not in ("manual", "dominica", "avacha_field"):
        raise ValueError("Unsupported temporal comparison task")
    expected = {"manual": ["sound"], "dominica": ["click"]}.get(ma["task"])
    if expected is not None and ma["labels"] != expected:
        raise ValueError("Primary AP ontology is not the declared sound/click task")
    if a["rows"].keys() != b["rows"].keys():
        raise ValueError("Paired temporal comparisons require identical immutable test row IDs")
    for key in a["rows"]:
        left, right = a["rows"][key], b["rows"][key]
        for field in ("recording_id", "group", "fold", "duration_seconds"):
            if left.get(field) != right.get(field):
                raise ValueError(f"Temporal source/group/exposure mismatch: {field}")
        truth_a = sorted((_canonical(event) for event in left["references"]))
        truth_b = sorted((_canonical(event) for event in right["references"]))
        if truth_a != truth_b:
            raise ValueError("Temporal ground truth differs between models")
    return ma["task"], ma["config"], sorted({row["group"] for row in a["rows"].values()})


def _recordings(artifact):
    result = {}
    for row in artifact["rows"].values():
        key = row["recording_id"]
        target = result.setdefault(key, {"group": row["group"], "fold": row.get("fold"),
            "references": [], "predictions": [], "duration_seconds": 0., "rows": []})
        if target["group"] != row["group"] or target["fold"] != row.get("fold"):
            raise ValueError("One test recording crosses source groups or held-out folds")
        target["references"].extend(row["references"])
        target["predictions"].extend(row["predictions"])
        target["duration_seconds"] += row["duration_seconds"]
        target["rows"].append(row)
    return result


def _ap_histogram(artifact, groups, *, tolerance=None, min_iou=None):
    index = {group: i for i, group in enumerate(groups)}
    reference_counts = np.zeros(len(groups), dtype=np.int64)
    entries = []
    for record in _recordings(artifact).values():
        group = index[record["group"]]
        refs, preds = _events(record["references"]), _events(record["predictions"], scored=True)
        reference_counts[group] += len(refs)
        edges = _edges(refs, preds, tolerance, min_iou)
        matched_pred, matched_ref = {}, {}
        for score_group in _score_groups([prediction["score"] for prediction in preds]):
            before = len(matched_pred)
            for i in score_group:
                _augment(i, edges, matched_pred, matched_ref)
            entries.append((preds[score_group[0]]["score"], group, len(score_group), len(matched_pred) - before))
    scores = sorted({entry[0] for entry in entries}, reverse=True)
    position = {score: i for i, score in enumerate(scores)}
    return {"reference_counts": reference_counts, "positions": np.asarray([position[e[0]] for e in entries], dtype=np.int64),
            "groups": np.asarray([e[1] for e in entries], dtype=np.int64),
            "predictions": np.asarray([e[2] for e in entries], dtype=np.int64),
            "gains": np.asarray([e[3] for e in entries], dtype=np.int64), "score_count": len(scores)}


def _weighted_ap(histogram, weights):
    references = int(np.dot(histogram["reference_counts"], weights))
    if not references:
        return None
    if not histogram["score_count"]:
        return 0.
    multiplicities = weights[histogram["groups"]]
    predictions = np.bincount(histogram["positions"], weights=multiplicities * histogram["predictions"],
                               minlength=histogram["score_count"])
    gains = np.bincount(histogram["positions"], weights=multiplicities * histogram["gains"],
                        minlength=histogram["score_count"])
    cumulative_predictions = np.cumsum(predictions)
    precision = np.divide(np.cumsum(gains), cumulative_predictions, out=np.zeros(len(predictions)),
                          where=cumulative_predictions > 0)
    return float(np.dot(gains, precision) / references)


def _budget_counts(artifact, groups, budget, cfg):
    index = {group: i for i, group in enumerate(groups)}
    counts = np.zeros((len(groups), 4), dtype=np.float64)  # TP, FP, references, seconds
    folds = {fold["fold"]: fold for fold in artifact["metadata"]["folds"]}
    for record in _recordings(artifact).values():
        refs = [{**event, "label": "call"} for event in record["references"]]
        preds = []
        fold = folds[record["fold"]]
        if fold.get("selection", {}).get("data") != "validation_only":
            raise ValueError("Field operating points lack validation-only selection provenance")
        expected = fold["recall_at_dev_far"][str(budget)]["validation"]["threshold"]
        for row in record["rows"]:
            threshold = row.get("budget_thresholds", {}).get(str(budget))
            if threshold is None or float(threshold) != float(expected):
                raise ValueError("Field row threshold differs from saved validation operating point")
            preds.extend(event for event in row["call_predictions"] if event["score"] >= threshold)
        tp = len(match_events(refs, preds, onset_tolerance_seconds=cfg["field_call_onset_tolerance_seconds"],
                              min_iou=cfg["field_call_min_iou"])["matches"])
        counts[index[record["group"]]] += [tp, len(preds) - tp, len(refs), record["duration_seconds"]]
    return counts


def _weighted_recall(counts, weights):
    total = weights @ counts
    return float(total[0] / total[2]) if total[2] else None


def _ci(reference, candidate, samples, *, seed):
    ones = np.ones(samples.shape[1], dtype=np.int64)
    observed_a, observed_b = reference(ones), candidate(ones)
    differences = []
    for weights in samples:
        a, b = reference(weights), candidate(weights)
        if a is not None and b is not None:
            differences.append(b - a)
    defined = len(differences)
    return {"reference": observed_a, "candidate": observed_b,
        "difference_candidate_minus_reference": observed_b - observed_a if observed_a is not None and observed_b is not None else None,
        "paired_source_bootstrap_95_percent": np.quantile(differences, [.025, .975]).tolist() if defined else None,
        "resamples": len(samples), "defined_resamples": defined, "undefined_resamples": len(samples) - defined,
        "seed": seed, "status": "undefined_metric" if observed_a is None or observed_b is None else "ok" if defined == len(samples) else "undefined_resamples"}


def paired_temporal_comparison(pathA, pathB, *, n_resamples=2000, seed=42):
    """Compare task directories or their temporal_predictions.jsonl files.

    Missing artifacts return an explicit status. Conflicting cohorts, truth,
    groups, criteria or validation-threshold provenance raise ValueError.
    """
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples <= 0:
        raise ValueError("Positive integer n_resamples required")
    a, missing_a = _load(pathA)
    b, missing_b = _load(pathB)
    if missing_a or missing_b:
        return {"status": "missing_artifact", "reference_missing": missing_a, "candidate_missing": missing_b,
                "metrics": {}, "paired_source_bootstrap_95_percent": None}
    task, cfg, groups = _strict_pair(a, b)
    if not groups:
        return {"status": "undefined_metric", "reason": "no test source groups", "task": task,
                "metrics": {}, "paired_source_bootstrap_95_percent": None}
    generator = np.random.default_rng(seed)
    samples = generator.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=n_resamples)
    result = {"schema_version": 1, "task": task, "status": "ok", "metrics": {},
        "dataset_fingerprint": a["metadata"]["dataset_fingerprint"], "bootstrap_groups": len(groups),
        "test_rows": len(a["rows"]), "grouping": "saved_source_group",
        "reference_artifacts": a["artifacts"], "candidate_artifacts": b["artifacts"],
        "replication": "weighted_independent_recording_graphs_equivalent_to_unique_replica_recording_ids",
        "limitation": "very few source groups; independence beyond saved grouping is not established" if len(groups) < 10 else
                      "independence beyond saved source grouping is not established"}
    if task == "avacha_field":
        primary = []
        for budget in cfg["field_far_budgets"]:
            ca, cb = _budget_counts(a, groups, budget, cfg), _budget_counts(b, groups, budget, cfg)
            name = f"recall_at_dev_fa_{budget:g}_per_hour"
            metric = _ci(lambda w: _weighted_recall(ca, w), lambda w: _weighted_recall(cb, w), samples, seed=seed)
            ones = np.ones(len(groups), dtype=np.int64)
            for label, counts in (("reference", ca), ("candidate", cb)):
                total = ones @ counts
                metric[label + "_test_false_alarms_per_hour"] = float(total[1] * 3600 / total[3]) if total[3] else None
                metric[label + "_operating_counts"] = {"true_positive": int(total[0]), "false_positive": int(total[1]),
                                                       "reference_count": int(total[2]), "exposure_seconds": float(total[3])}
            result["metrics"][name] = metric
            primary.append(name)
        result["primary_metrics"] = primary
    else:
        criteria = [("ap_iou_0.5", None, .5)] if task == "manual" else [
            ("ap_onset_10ms", .01, None), ("ap_onset_1ms", .001, None), ("ap_onset_5ms", .005, None)]
        result["primary_metrics"] = [criteria[0][0]]
        for number, (name, tolerance, iou) in enumerate(criteria):
            ha, hb = _ap_histogram(a, groups, tolerance=tolerance, min_iou=iou), _ap_histogram(b, groups, tolerance=tolerance, min_iou=iou)
            operations = (ha["score_count"] + hb["score_count"]) * n_resamples
            if number and operations > SECONDARY_AP_OPERATION_BUDGET:
                ones = np.ones(len(groups), dtype=np.int64)
                va, vb = _weighted_ap(ha, ones), _weighted_ap(hb, ones)
                metric = {"reference": va, "candidate": vb, "difference_candidate_minus_reference": vb - va if va is not None and vb is not None else None,
                    "paired_source_bootstrap_95_percent": None, "status": "secondary_ci_not_computed_budget_limit",
                    "estimated_score_operations": operations, "operation_budget": SECONDARY_AP_OPERATION_BUDGET}
            else:
                metric = _ci(lambda w: _weighted_ap(ha, w), lambda w: _weighted_ap(hb, w), samples, seed=seed)
            metric.update({"onset_tolerance_seconds": tolerance, "min_iou": iou})
            result["metrics"][name] = metric
    primary_statuses = [result["metrics"][name]["status"] for name in result["primary_metrics"]]
    if any(status != "ok" for status in primary_statuses):
        result["status"] = "undefined_primary_or_resamples"
    return result
