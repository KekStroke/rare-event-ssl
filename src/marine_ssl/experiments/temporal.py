"""Frozen-encoder temporal probes with recording-held-out, validation-only selection.

This module reads cached physical feature grids, never waveforms. Clicks are an
explicit coarse frame-center baseline: no interpolation or millisecond accuracy
is claimed. All decoder/head choices are recorded in the resolved configuration.
"""
from __future__ import annotations

import csv
from bisect import bisect_left, insort
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import warnings

import numpy as np

from .metrics import event_metrics, match_events


DEFAULT_CONFIG = {
    "layers": [6, 9, 12], "embedding_dim": 768, "seed": 42,
    "max_train_frames": 50000, "min_positive_frames_per_class": 64,
    "C": 1.0, "max_iter": 500, "tol": 1e-4, "class_weight": "balanced",
    "solver": "liblinear", "standardize": True,
    "edge_threshold": .3, "interval_min_seconds": .02,
    "interval_merge_gap_seconds": 0.0,
    "threshold_grid": [.3, .4, .5, .6, .7, .8, .9, 1.0000001],
    "point_candidate_threshold": .05, "point_min_separation_seconds": .02,
    "selection_iou": .5, "report_ious": [.5, .8],
    "selection_tolerance_seconds": .01, "point_tolerances_seconds": [.001, .005, .01],
    "field_far_budgets": [50, 100, 300],
    "field_call_onset_tolerance_seconds": None, "field_call_min_iou": .5,
    "artifact_policy": "background", "artifact_annotations": None,
}


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical(value):
    return json.dumps(_json(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolved(config):
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown temporal configuration keys: {sorted(unknown)}")
    cfg = {**DEFAULT_CONFIG, **config}
    if not cfg["layers"] or len(set(cfg["layers"])) != len(cfg["layers"]):
        raise ValueError("Unique nonempty one-based layers required")
    if any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 1 for layer in cfg["layers"]):
        raise ValueError("One-based integer layers required")
    if cfg["max_train_frames"] < 1 or cfg["embedding_dim"] < 1 or cfg["C"] <= 0 or cfg["max_iter"] < 1:
        raise ValueError("Positive frame budget, feature dimension and head parameters required")
    if cfg["solver"] != "liblinear" or cfg["artifact_policy"] not in ("background", "mask"):
        raise ValueError("Supported fixed head: liblinear; artifact policy: background or mask")
    if not 0 < cfg["edge_threshold"] < 1 or not 0 <= cfg["point_candidate_threshold"] < 1:
        raise ValueError("Invalid candidate thresholds")
    if not cfg["threshold_grid"] or any(not math.isfinite(float(t)) or t < 0 for t in cfg["threshold_grid"]):
        raise ValueError("Finite nonnegative validation threshold grid required")
    if any(t < cfg["edge_threshold"] for t in cfg["threshold_grid"]):
        raise ValueError("Operating thresholds must not be below interval candidate edge threshold")
    if any(t <= 0 for t in cfg["point_tolerances_seconds"]):
        raise ValueError("Positive click tolerances required")
    return cfg


def _validate_rows(rows):
    assignments, intervals = {}, {}
    for row in rows:
        fold = row.get("fold")
        for kind, identity in (("group", row["group"]), ("recording", row["recording_id"])):
            key = (fold, kind, identity)
            if key in assignments and assignments[key] != row["split"]:
                raise ValueError("Source group/recording crosses train/val/test within a fold")
            assignments[key] = row["split"]
        key = (fold, row["recording_id"])
        intervals.setdefault(key, []).append((float(row["start_seconds"]), float(row["end_seconds"])))
    for values in intervals.values():
        values.sort()
        if any(left[1] > right[0] + 1e-9 for left, right in zip(values, values[1:])):
            raise ValueError("Overlapping evaluation intervals would duplicate exposure/references")


def _artifact_lookup(task_result, rows, cfg):
    if task_result["task"] != "manual":
        return {}, {"status": "not_applicable"}
    pinned = task_result.get("input_metadata_sha256", {})
    candidates = [path for path in pinned if Path(path).name == "manifest.csv"]
    path = cfg["artifact_annotations"] or (candidates[0] if len(candidates) == 1 else None)
    if path is None or path not in pinned:
        if cfg["artifact_policy"] == "mask":
            raise ValueError("Artifact masking requires the exact hash-pinned annotation CSV")
        return {}, {"status": "not_available", "reason": "hash-pinned artifact CSV not supplied"}
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != pinned[path]:
        raise ValueError("Pinned artifact annotation CSV checksum mismatch")
    annotations = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    known, result = set(), {}
    for annotation in annotations:
        name = Path(annotation["audio"].replace("\\", "/")).name
        known.add(name)
        if annotation["label"].casefold() == "artifact":
            start, end = float(annotation["start_s"]), float(annotation["end_s"])
            if not math.isfinite(start + end) or start < 0 or end <= start:
                raise ValueError("Invalid artifact interval")
            result.setdefault(name, []).append((start, end))
    names = [Path(row["path"]).name for row in rows]
    missing = sorted(set(names) - known)
    if missing:
        if cfg["artifact_policy"] == "mask":
            raise ValueError("Artifact CSV incompletely matches evaluation files")
        return {}, {"status": "not_available", "reason": "incomplete exact-basename annotation coverage", "missing": missing}
    return result, {"status": "ok", "source": path, "sha256": pinned[path], "matched_audio_files": len(set(names))}


def _features(loader, row, layer, cfg):
    item = loader(row, [layer])[layer]
    x, times = np.asarray(item["features"]), np.asarray(item["times"], dtype=np.float64)
    if x.ndim != 2 or x.shape != (len(times), cfg["embedding_dim"]) or not len(times):
        raise ValueError("Feature grid must be nonempty [T, embedding_dim] with T physical centers")
    if not np.isfinite(x).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Nonfinite or nonmonotonic feature grid")
    start, end = float(row["start_seconds"]), float(row["end_seconds"])
    if np.any(times < start - 1e-9) or np.any(times > end + 1e-9):
        raise ValueError("Feature centers must be ABSOLUTE recording seconds inside the evaluation interval")
    if abs(float(item["duration_seconds"]) - (end - start)) > 1e-6:
        raise ValueError("Feature duration does not match the exact row interval")
    hop = float(item["frame_hop_seconds"])
    if not math.isfinite(hop) or hop <= 0:
        raise ValueError("Physical frame hop is required")
    if len(times) > 1 and np.max(np.diff(times)) > 2.5 * hop + 1e-6:
        raise ValueError("Feature grid has missing context chunks")
    return x, times, hop, float(item.get("context_seconds", 0))


def _references(row, labels, point):
    result = []
    start, end = float(row["start_seconds"]), float(row["end_seconds"])
    for event in row.get("events", []):
        if event["label"] == "artifact":
            continue
        if event["label"] not in labels:
            raise ValueError(f"Unexpected temporal label {event['label']}")
        if point:
            time = float(event["time_seconds"])
            if start <= time <= end:
                result.append({"start": time, "end": time, "label": event["label"], "recording_id": row["recording_id"]})
        else:
            a, b = max(start, float(event["start_seconds"])), min(end, float(event["end_seconds"]))
            if b > a:
                result.append({"start": a, "end": b, "label": event["label"], "recording_id": row["recording_id"]})
    return result


def _targets(row, times, labels, point, artifacts, cfg):
    y = np.zeros((len(times), len(labels)), dtype=np.uint8)
    refs = _references(row, labels, point)
    distances, point_indices = [], []
    for event in refs:
        label = labels.index(event["label"])
        if point:
            right = min(int(np.searchsorted(times, event["start"])), len(times) - 1)
            left = max(0, right - 1)
            index = min((left, right), key=lambda i: (abs(times[i] - event["start"]), i))
            y[index, label] = 1
            distances.append(abs(float(times[index]) - event["start"]))
            point_indices.append(index)
        else:
            y[(times >= event["start"]) & (times < event["end"]), label] = 1
    artifact = np.zeros(len(times), dtype=bool)
    for start, end in artifacts.get(Path(row["path"]).name, []):
        artifact |= (times >= start) & (times < end)
    artifact &= ~y.any(axis=1)  # overlapping sound remains a positive sound target
    valid = ~artifact if cfg["artifact_policy"] == "mask" else np.ones(len(times), dtype=bool)
    return y, valid, artifact, distances, len(point_indices) - len(set(point_indices))


def _sample_indices(y, valid, quota, row, cfg):
    available = np.flatnonzero(valid)
    if len(available) <= quota:
        return available
    reserve_each = min(cfg["min_positive_frames_per_class"], quota // (2 * y.shape[1]))
    reserved = set()
    for column in range(y.shape[1]):
        positives = np.flatnonzero(valid & (y[:, column] == 1))
        n = min(len(positives), reserve_each)
        if n:
            reserved.update(positives[np.linspace(0, len(positives) - 1, n, dtype=int)].tolist())
    remaining = np.setdiff1d(available, np.fromiter(reserved, dtype=int), assume_unique=True)
    seed = int(hashlib.sha256(f"{cfg['seed']}:{row['row_id']}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    picked = rng.choice(remaining, size=quota - len(reserved), replace=False)
    return np.array(sorted(reserved | set(picked.tolist())), dtype=int)


def _fit(rows, layer, loader, labels, point, artifacts, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if cfg["max_train_frames"] < len(rows):
        raise ValueError("Training frame budget must cover every training recording")
    quota = cfg["max_train_frames"] // len(rows)
    xs, ys = [], []
    for row in rows:
        x, times, _, _ = _features(loader, row, layer, cfg)
        y, valid, _, _, _ = _targets(row, times, labels, point, artifacts, cfg)
        indices = _sample_indices(y, valid, quota, row, cfg)
        if len(indices):
            xs.append(np.asarray(x[indices], dtype=np.float32))
            ys.append(y[indices])
    if not xs:
        raise ValueError("No eligible training frames")
    x, y = np.concatenate(xs), np.concatenate(ys)
    scaler = StandardScaler() if cfg["standardize"] else None
    if scaler:
        x = scaler.fit_transform(x)
    models, fit_warnings = [], []
    for column, label in enumerate(labels):
        if len(np.unique(y[:, column])) < 2:
            models.append(float(y[0, column]))
            fit_warnings.append(f"{label}: constant training target; constant predictor, no synthetic labels")
            continue
        model = LogisticRegression(C=cfg["C"], solver=cfg["solver"], class_weight=cfg["class_weight"],
            max_iter=cfg["max_iter"], tol=cfg["tol"], random_state=cfg["seed"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x, y[:, column])
        fit_warnings.extend(f"{label}: {warning.message}" for warning in caught)
        models.append(model)
    return {"models": models, "scaler": scaler, "training_frames": len(y),
            "positive_frames": {label: int(y[:, i].sum()) for i, label in enumerate(labels)},
            "warnings": fit_warnings, "training_recordings": sorted({r["recording_id"] for r in rows})}


def _probabilities(head, x):
    # Bound scaler/predict temporary allocations even for multi-hour recordings.
    output = np.empty((len(x), len(head["models"])), dtype=np.float32)
    for start in range(0, len(x), 8192):
        end = min(len(x), start + 8192)
        batch = np.asarray(x[start:end], dtype=np.float32)
        if head["scaler"] is not None:
            batch = head["scaler"].transform(batch)
        for column, model in enumerate(head["models"]):
            output[start:end, column] = model if isinstance(model, float) else model.predict_proba(batch)[:, 1]
    return output


def _intervals(scores, times, hop, row, label, cfg):
    indices = np.flatnonzero(scores >= cfg["edge_threshold"])
    runs = []
    for index in indices:
        if runs and index == runs[-1][-1] + 1 and times[index] - times[runs[-1][-1]] <= 1.5 * hop + 1e-8:
            runs[-1].append(int(index))
        else:
            runs.append([int(index)])
    events = []
    for run in runs:
        start = max(float(row["start_seconds"]), float(times[run[0]]) - hop / 2)
        end = min(float(row["end_seconds"]), float(times[run[-1]]) + hop / 2)
        event = {"start": start, "end": end, "score": float(np.max(scores[run])),
                 "label": label, "recording_id": row["recording_id"]}
        if events and start - events[-1]["end"] <= cfg["interval_merge_gap_seconds"] + 1e-9:
            events[-1]["end"] = end
            events[-1]["score"] = max(events[-1]["score"], event["score"])
        else:
            events.append(event)
    return [event for event in events if event["end"] - event["start"] >= cfg["interval_min_seconds"] - 1e-9]


def _points(scores, times, row, label, cfg):
    left, right = np.r_[-np.inf, scores[:-1]], np.r_[scores[1:], -np.inf]
    candidates = np.flatnonzero((scores >= left) & (scores > right) & (scores >= cfg["point_candidate_threshold"]))
    chosen, selected_times = [], []
    for index in sorted(candidates, key=lambda i: (-float(scores[i]), int(i))):
        time = float(times[index])
        insertion = bisect_left(selected_times, time)
        neighbors = selected_times[max(0, insertion - 1):insertion + 1]
        if all(abs(time - other) >= cfg["point_min_separation_seconds"] - 1e-9 for other in neighbors):
            chosen.append(int(index))
            insort(selected_times, time)
    return [{"start": float(times[i]), "end": float(times[i]), "score": float(scores[i]),
             "label": label, "recording_id": row["recording_id"]} for i in sorted(chosen)]


def _predict(rows, head, layer, loader, labels, point, artifacts, cfg):
    result = []
    for row in rows:
        x, times, hop, context = _features(loader, row, layer, cfg)
        probs = _probabilities(head, x)
        _, _, artifact, distances, collisions = _targets(row, times, labels, point, artifacts, cfg)
        decode = _points if point else lambda s, t, r, label, c: _intervals(s, t, hop, r, label, c)
        predictions = [event for i, label in enumerate(labels) for event in decode(probs[:, i], times, row, label, cfg)]
        refs = _references(row, labels, point)
        calls = [] if point else _intervals(probs.max(axis=1), times, hop, row, "call", cfg)
        result.append({"row_id": row["row_id"], "recording_id": row["recording_id"], "group": row["group"],
            "fold": row.get("fold"), "references": refs, "predictions": predictions, "call_predictions": calls,
            "duration_seconds": float(row["end_seconds"] - row["start_seconds"]),
            "frame_hop_seconds": hop, "context_seconds": context, "feature_frames": len(times),
            "artifact_scores": probs[artifact, 0].copy(), "point_grid_distances": distances,
            "point_target_collisions": collisions})
    return result


def _flatten(records, key):
    return [event for record in records for event in record[key]]


def _matching_config(point, cfg):
    return {"onset_tolerance_seconds": cfg["selection_tolerance_seconds"]} if point else {"min_iou": cfg["selection_iou"]}


def _selection_ap(records, labels, point, cfg):
    refs, preds = _flatten(records, "references"), _flatten(records, "predictions")
    exposure = sum(r["duration_seconds"] for r in records)
    values = {}
    for label in labels:
        r = [event for event in refs if event["label"] == label]
        if not r:
            values[label] = None
            continue
        values[label] = event_metrics(r, [event for event in preds if event["label"] == label],
                                     duration_seconds=exposure, **_matching_config(point, cfg))["average_precision"]
    defined = [value for value in values.values() if value is not None]
    return sum(defined) / len(defined) if defined else None, values


def _counts(refs, preds, exposure, threshold, matching):
    selected = [event for event in preds if event["score"] >= threshold]
    tp = len(match_events(refs, selected, **matching)["matches"])
    fp, fn = len(selected) - tp, len(refs) - tp
    return {"threshold": float(threshold), "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "recall": tp / len(refs) if refs else None, "precision": tp / len(selected) if selected else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "false_alarms_per_hour": fp * 3600 / exposure}


def _choose_threshold(records, point, cfg):
    refs, preds = _flatten(records, "references"), _flatten(records, "predictions")
    exposure = sum(r["duration_seconds"] for r in records)
    # Counts only: AP is not recomputed at every operating threshold.
    grid = [_counts(refs, preds, exposure, t, _matching_config(point, cfg)) for t in cfg["threshold_grid"]]
    best = max(grid, key=lambda row: (row["f1"] if row["f1"] is not None else -1,
                                     -row["false_alarms_per_hour"], row["threshold"]))
    return best, grid


def _budget_thresholds(records, cfg):
    refs = [{**event, "label": "call"} for event in _flatten(records, "references")]
    preds = _flatten(records, "call_predictions")
    exposure = sum(r["duration_seconds"] for r in records)
    matching = {"onset_tolerance_seconds": cfg["field_call_onset_tolerance_seconds"], "min_iou": cfg["field_call_min_iou"]}
    grid = [_counts(refs, preds, exposure, t, matching) for t in cfg["threshold_grid"]]
    result = {}
    for budget in cfg["field_far_budgets"]:
        eligible = [row for row in grid if row["false_alarms_per_hour"] <= budget]
        if not eligible:
            raise ValueError("Threshold grid has no feasible dev false-alarm point; include a reject-all threshold")
        result[str(budget)] = max(eligible, key=lambda row: (row["recall"] if row["recall"] is not None else -1,
                                                          -row["false_alarms_per_hour"], row["threshold"]))
    return result


def _reports(records, labels, point, threshold, cfg):
    refs, preds = _flatten(records, "references"), _flatten(records, "predictions")
    exposure = sum(r["duration_seconds"] for r in records)
    criteria = [(f"onset_{t * 1000:g}ms", {"onset_tolerance_seconds": t}) for t in cfg["point_tolerances_seconds"]] if point else [
        (f"iou_{value:g}", {"min_iou": value}) for value in cfg["report_ious"]]
    result = {}
    for name, matching in criteria:
        pooled = event_metrics(refs, preds, duration_seconds=exposure, threshold=threshold, **matching)
        per_class = {labels[0]: pooled} if len(labels) == 1 else {label: event_metrics(
            [e for e in refs if e["label"] == label], [e for e in preds if e["label"] == label],
            duration_seconds=exposure, threshold=threshold, **matching) for label in labels}
        defined = [value["average_precision"] for value in per_class.values() if value["average_precision"] is not None]
        result[name] = {"pooled": pooled, "per_class": per_class,
                       "macro_average_precision": sum(defined) / len(defined) if defined else None,
                       "evaluated_seconds": exposure}
    return result


def _save_head(path, head, labels, cfg):
    models = head["models"]
    coefficients = np.zeros((len(labels), cfg["embedding_dim"]), dtype=np.float64)
    intercepts, constants = np.zeros(len(labels)), np.full(len(labels), -1.0)
    for i, model in enumerate(models):
        if isinstance(model, float):
            constants[i] = model
        else:
            coefficients[i], intercepts[i] = model.coef_[0], model.intercept_[0]
    scaler = head["scaler"]
    np.savez_compressed(path, coefficients=coefficients, intercepts=intercepts, constant_probabilities=constants,
        labels=np.asarray(labels), scaler_mean=scaler.mean_ if scaler is not None else np.zeros(cfg["embedding_dim"]),
        scaler_scale=scaler.scale_ if scaler is not None else np.ones(cfg["embedding_dim"]))


def _pool_operating(parts, exposure, candidate_metric=None):
    tp = sum(part["true_positive"] for part in parts)
    fp = sum(part["false_positive"] for part in parts)
    fn = sum(part["false_negative"] for part in parts)
    result = dict(candidate_metric or {})
    for key in ("matches", "unmatched_prediction_indices", "unmatched_reference_indices"):
        result.pop(key, None)
    result.update({"threshold": None, "threshold_policy": "separate_validation_selected_threshold_per_fold",
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "reference_count": tp + fn, "selected_prediction_count": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_alarms_per_hour": fp * 3600 / exposure})
    for name in ("onset_mae_seconds", "onset_mean_signed_error_seconds"):
        values = [(part[name], part["true_positive"]) for part in parts if part.get(name) is not None]
        result[name] = sum(value * count for value, count in values) / sum(count for _, count in values) if values else None
    undefined = dict(result.get("undefined_metrics", {}))
    if result["onset_mae_seconds"] is not None:
        undefined.pop("onset_mae_seconds", None)
    result["undefined_metrics"] = undefined
    return result


def evaluate_temporal(task_result, feature_loader, output_dir, config):
    """Fit on train, select layer/threshold on val, evaluate test once per fold.

    Loader contract: ``loader(row, [one_based_layer]) -> {layer: {features,
    times, duration_seconds, frame_hop_seconds, context_seconds}}``. ``times``
    are absolute physical recording centers; no frame or sample is fabricated.
    CPU memory is bounded by one loaded recording plus the train-frame budget.
    """
    # Generic clip-head settings (including a C grid) belong to the caller's
    # classifier protocol. Temporal choices have their own explicit namespace.
    cfg = _resolved(config.get("temporal", {}))
    started = time.monotonic()
    task, rows = task_result["task"], task_result.get("rows", [])
    if task not in ("avacha_field", "manual", "dominica"):
        raise ValueError("Temporal evaluator supports avacha_field, manual and dominica")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"dataset_fingerprint": task_result.get("fingerprint"), "task": task, "config": cfg}
    frozen = output / "temporal_config.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != _json(identity):
        raise ValueError("Output directory belongs to a different immutable temporal run")
    _write_json(frozen, identity)
    if not task_result.get("ready"):
        summary = {"status": "not_ready", "task": task, "reasons": task_result.get("reasons", [])}
        _write_json(output / "temporal_summary.json", summary)
        return summary
    _validate_rows(rows)
    artifacts, artifact_status = _artifact_lookup(task_result, rows, cfg)
    point = task == "dominica"
    labels = ["click"] if point else ["sound"] if task == "manual" else list(task_result["protocol"]["classes"])
    folds = sorted({row.get("fold") for row in rows}, key=lambda value: -1 if value is None else value)
    summaries, all_test, prediction_records = [], [], []
    print(_canonical({"stage": "temporal_start", "task": task, "layers": cfg["layers"],
                      "folds": folds, "max_train_frames": cfg["max_train_frames"]}), flush=True)
    for fold in folds:
        split = {name: [row for row in rows if row.get("fold") == fold and row["split"] == name]
                 for name in ("train", "val", "test")}
        if any(not values for values in split.values()):
            raise ValueError("Each fold requires nonempty train/val/test cohorts")
        candidates = []
        for layer in cfg["layers"]:
            head = _fit(split["train"], layer, feature_loader, labels, point, artifacts, cfg)
            validation = _predict(split["val"], head, layer, feature_loader, labels, point, artifacts, cfg)
            score, per_class_ap = _selection_ap(validation, labels, point, cfg)
            candidates.append({"layer": layer, "head": head, "validation": validation,
                               "score": score, "validation_per_class_ap": per_class_ap})
            print(_canonical({"stage": "temporal_validation", "task": task, "fold": fold, "layer": layer,
                              "training_frames": head["training_frames"], "validation_macro_ap": score,
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
        defined = [candidate for candidate in candidates if candidate["score"] is not None]
        if not defined:
            raise ValueError("No validation references; test labels cannot select a layer")
        chosen = max(defined, key=lambda candidate: (candidate["score"], -candidate["layer"]))
        operating, grid = _choose_threshold(chosen["validation"], point, cfg)
        threshold, layer, head = operating["threshold"], chosen["layer"], chosen["head"]
        test = _predict(split["test"], head, layer, feature_loader, labels, point, artifacts, cfg)
        budget_points = _budget_thresholds(chosen["validation"], cfg) if task == "avacha_field" else {}
        budget_test = {}
        for budget, dev in budget_points.items():
            refs = [{**e, "label": "call"} for e in _flatten(test, "references")]
            preds = _flatten(test, "call_predictions")
            budget_test[budget] = {"validation": dev, "test": _counts(refs, preds,
                sum(r["duration_seconds"] for r in test), dev["threshold"],
                {"onset_tolerance_seconds": cfg["field_call_onset_tolerance_seconds"], "min_iou": cfg["field_call_min_iou"]})}
        artifact_scores = np.concatenate([r["artifact_scores"] for r in test])
        artifact_metrics = {**artifact_status, "artifact_only_frames": len(artifact_scores),
            "sound_false_positive_frames": int(np.sum(artifact_scores >= threshold)),
            "sound_false_positive_rate": float(np.mean(artifact_scores >= threshold)) if len(artifact_scores) and artifact_status["status"] == "ok" else None,
            "auxiliary": True}
        distances = [distance for record in test for distance in record["point_grid_distances"]]
        resolution = {"mode": "coarse_physical_frame_centers_no_refinement" if point else "physical_frame_grid",
            "observed_hops_seconds": sorted({r["frame_hop_seconds"] for r in test}),
            "context_seconds": sorted({r["context_seconds"] for r in test})}
        if point:
            resolution.update({"subframe_localization": False, "native_source_sample_rates": sorted({r.get("sample_rate_hz", r.get("sample_rate")) for r in split["test"] if r.get("sample_rate_hz", r.get("sample_rate"))}),
                "nearest_grid_reference_count": len(distances), "point_target_collisions": sum(r["point_target_collisions"] for r in test),
                "nearest_grid_recall_upper_bound": {f"{t * 1000:g}ms": sum(distance <= t + 1e-12 for distance in distances) / len(distances) if distances else None for t in cfg["point_tolerances_seconds"]},
                "limitation": "20ms features do not provide 1ms localization; tolerance reports measure this coarse baseline, without interpolation or waveform refinement."})
        fold_summary = {"fold": fold, "selected_layer": layer, "threshold": threshold,
            "selection": {"data": "validation_only", "metric": "macro_event_average_precision", "candidates": [
                {"layer": c["layer"], "score": c["score"], "per_class_ap": c["validation_per_class_ap"]} for c in candidates],
                "threshold_metric": "event_f1", "validation_operating_point": operating, "threshold_grid_counts": grid},
            "head": {k: head[k] for k in ("training_frames", "positive_frames", "warnings", "training_recordings")},
            "validation_recordings": sorted({r["recording_id"] for r in split["val"]}),
            "test_recordings": sorted({r["recording_id"] for r in split["test"]}),
            "test": _reports(test, labels, point, threshold, cfg), "recall_at_dev_far": budget_test,
            "artifact_metrics": artifact_metrics, "resolution": resolution}
        summaries.append(fold_summary)
        _save_head(output / f"head_fold_{fold if fold is not None else 0}_layer_{layer}.npz", head, labels, cfg)
        for record in test:
            persisted = {k: v for k, v in record.items() if k not in ("artifact_scores", "point_grid_distances")}
            persisted["selected_layer"], persisted["threshold"] = layer, threshold
            persisted["operating_predictions"] = [event for event in persisted["predictions"] if event["score"] >= threshold]
            persisted["budget_thresholds"] = {budget: dev["threshold"] for budget, dev in budget_points.items()}
            prediction_records.append(persisted)
        all_test.extend(test)
    summary = {"schema_version": 1, "task": task, "status": "completed", "auxiliary": task_result.get("auxiliary", False),
        "dataset_fingerprint": task_result.get("fingerprint"), "config": cfg, "labels": labels, "folds": summaries,
        "test_recording_count": len({r["recording_id"] for r in all_test}),
        "test_exposure_seconds": sum(r["duration_seconds"] for r in all_test),
        "prediction_artifact": str(output / "temporal_predictions.jsonl"),
        "limitations": ["Frozen-encoder frame probes; context, sampling budget and decoder are fixed in config.",
            "References are intersected with evaluation UEM; original annotations remain in the input manifest.",
            "Decoder does not bridge missing context-boundary frames; sustained events may fragment at five-second context boundaries.",
            "Confidence intervals must resample paired source groups, not individual feature frames."]}
    if len(summaries) == 1:
        summary["test"] = summaries[0]["test"]
        summary["recall_at_dev_far"] = summaries[0]["recall_at_dev_far"]
    else:
        # AP ranks all OOF candidates. Operating counts pool the independently
        # dev-fixed fold thresholds; there is no fabricated global test threshold.
        pooled = _reports(all_test, labels, point, 1.0000001, cfg)
        for criterion, report in pooled.items():
            report["pooled"] = _pool_operating([fold["test"][criterion]["pooled"] for fold in summaries],
                                               summary["test_exposure_seconds"], report["pooled"])
            report["per_class"] = {label: _pool_operating([fold["test"][criterion]["per_class"][label] for fold in summaries],
                summary["test_exposure_seconds"], report["per_class"][label]) for label in labels}
        summary["test"] = pooled
        summary["recall_at_dev_far"] = {str(budget): _pool_operating(
            [fold["recall_at_dev_far"][str(budget)]["test"] for fold in summaries], summary["test_exposure_seconds"])
            for budget in cfg["field_far_budgets"]} if task == "avacha_field" else {}
        summary["limitations"].append("Pooled OOF AP ranks scores from independently fitted fold heads; fold calibration may differ.")
    temporary = output / "temporal_predictions.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in prediction_records:
            stream.write(_canonical(record) + "\n")
    os.replace(temporary, output / "temporal_predictions.jsonl")
    _write_json(output / "temporal_summary.json", summary)
    _write_json(output / "metrics.json", summary)
    print(_canonical({"stage": "temporal_completed", "task": task, "test_recordings": summary["test_recording_count"],
                      "elapsed_seconds": time.monotonic() - started}), flush=True)
    return summary
