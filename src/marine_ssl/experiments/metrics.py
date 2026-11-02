"""Evaluation primitives. Thresholds and event tolerances must come from dev/config.

Undefined quantities are JSON ``null`` with an accompanying status; they are
never replaced with zero. Event times are seconds on each recording's clock.
Inputs/predictions are not modified, so callers can retain the original schema.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from bisect import bisect_left, bisect_right
import math
import random
from typing import Any


def _finite(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float | None]) -> float | None:
    defined = [x for x in values if x is not None]
    return sum(defined) / len(defined) if defined else None


def multiclass_metrics(y_true: Sequence[Any], y_pred: Sequence[Any], *,
                       labels: Sequence[Any] | None = None) -> dict:
    """Macro F1 over defined classes; balanced accuracy over supported classes.

    Explicit labels absent from both truth and predictions have undefined F1
    and are reported in ``undefined_f1_labels``, rather than counted as zero.
    A supported but never predicted class has recall/F1 zero as usual.
    """
    truth, prediction = list(y_true), list(y_pred)
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction lengths differ")
    classes = list(dict.fromkeys(labels if labels is not None else truth + prediction))
    if any(x not in classes for x in truth + prediction):
        raise ValueError("labels omit an observed class")
    per_class = []
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        support = sum(t == label for t in truth)
        predicted = sum(p == label for p in prediction)
        per_class.append({"label": label, "support": support, "predicted": predicted,
                          "true_positive": tp, "precision": _ratio(tp, predicted),
                          "recall": _ratio(tp, support),
                          "f1": _ratio(2 * tp, support + predicted)})
    undefined = [r["label"] for r in per_class if r["f1"] is None]
    return {"n": len(truth), "accuracy": _ratio(sum(t == p for t, p in zip(truth, prediction)), len(truth)),
            "macro_f1": _mean([r["f1"] for r in per_class]),
            "balanced_accuracy": _mean([r["recall"] for r in per_class]),
            "per_class": per_class, "undefined_f1_labels": undefined,
            "status": "empty" if not truth else ("absent_classes" if undefined else "ok")}


def _binary_inputs(y_true: Sequence[int], scores: Sequence[float]):
    truth, values = list(y_true), [_finite(s, "score") for s in scores]
    if len(truth) != len(values):
        raise ValueError("truth and score lengths differ")
    if any(y not in (0, 1) for y in truth):
        raise ValueError("binary truth must contain only 0/1")
    return truth, values


def _score_groups(scores: Sequence[float]):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    groups = []
    for i in order:
        if not groups or scores[i] != scores[groups[-1][0]]:
            groups.append([])
        groups[-1].append(i)
    return groups


def _binary_ap(truth, scores):
    positives = sum(truth)
    if not positives:
        return None
    tp = seen = 0
    area = 0.0
    for group in _score_groups(scores):
        gained = sum(truth[i] for i in group)
        tp += gained
        seen += len(group)
        area += gained / positives * tp / seen
    return area


def _roc_auc(truth, scores):
    positives, negatives = sum(truth), len(truth) - sum(truth)
    if not positives or not negatives:
        return None
    negatives_below = 0
    wins = 0.0
    for group in reversed(_score_groups(scores)):
        p = sum(truth[i] for i in group)
        n = len(group) - p
        wins += p * (negatives_below + n / 2)
        negatives_below += n
    return wins / (positives * negatives)


def _counts(tp, fp, fn):
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def binary_metrics(y_true: Sequence[int], scores: Sequence[float], *, threshold: float) -> dict:
    """Tie-aware AP/ROC-AUC and metrics at the supplied, dev-fixed threshold."""
    truth, values = _binary_inputs(y_true, scores)
    threshold = _finite(threshold, "threshold")
    prediction = [s >= threshold for s in values]
    tp = sum(y == 1 and p for y, p in zip(truth, prediction))
    fp = sum(y == 0 and p for y, p in zip(truth, prediction))
    fn = sum(y == 1 and not p for y, p in zip(truth, prediction))
    statuses = {}
    if not sum(truth):
        statuses["average_precision"] = "no_positive_references"
    if not truth or len(set(truth)) < 2:
        statuses["roc_auc"] = "requires_both_classes"
    if not truth:
        statuses["accuracy"] = "empty"
    return {"n": len(truth), "positive_support": sum(truth), "threshold": threshold,
            **_counts(tp, fp, fn), "average_precision": _binary_ap(truth, values),
            "roc_auc": _roc_auc(truth, values),
            "accuracy": _ratio(sum(y == p for y, p in zip(truth, prediction)), len(truth)),
            "undefined_metrics": statuses}


def _events(events, *, scored=False):
    result = []
    for event in events:
        start = _finite(event["start"], "event start")
        end = _finite(event.get("end", start), "event end")
        if end < start:
            raise ValueError("event end precedes start")
        item = {"start": start, "end": end, "recording_id": event.get("recording_id", ""),
                "label": event.get("label")}
        if scored:
            item["score"] = _finite(event["score"], "event score")
        result.append(item)
    return result


def _edges(references, predictions, tolerance, min_iou):
    if tolerance is None and min_iou is None:
        raise ValueError("explicit onset tolerance or minimum IoU is required")
    if tolerance is not None and _finite(tolerance, "onset tolerance") < 0:
        raise ValueError("onset tolerance must be nonnegative")
    if min_iou is not None and not 0 < _finite(min_iou, "minimum IoU") <= 1:
        raise ValueError("minimum IoU must be in (0, 1]")
    # Matching is independent between recording/label groups. Point tasks may
    # contain tens of thousands of references; only onsets inside the explicit
    # tolerance can form an edge, so avoid a corpus-wide Cartesian scan.
    grouped = {}
    for j, reference in enumerate(references):
        grouped.setdefault((reference["recording_id"], reference["label"]), []).append((reference["start"], j))
    grouped = {key: sorted(values) for key, values in grouped.items()}
    starts = {key: [value[0] for value in values] for key, values in grouped.items()}
    edges = []
    for prediction in predictions:
        eligible = []
        key = (prediction["recording_id"], prediction["label"])
        candidates = grouped.get(key, [])
        if tolerance is not None:
            lo = bisect_left(starts.get(key, []), math.nextafter(prediction["start"] - tolerance - 1e-12, -math.inf))
            hi = bisect_right(starts.get(key, []), math.nextafter(prediction["start"] + tolerance + 1e-12, math.inf))
            candidates = candidates[lo:hi]
        for _, j in candidates:
            reference = references[j]
            delta = abs(prediction["start"] - reference["start"])
            if tolerance is not None and delta > tolerance + 1e-12:
                continue
            if min_iou is not None:
                intersection = max(0.0, min(prediction["end"], reference["end"]) - max(prediction["start"], reference["start"]))
                union = max(prediction["end"], reference["end"]) - min(prediction["start"], reference["start"])
                if not union or intersection / union < min_iou:
                    continue
            eligible.append(j)
        edges.append(sorted(eligible, key=lambda j: (abs(prediction["start"] - references[j]["start"]), j)))
    return edges


def _augment(root, edges, matched_pred, matched_ref):
    """Increase maximum matching by one, rerouting earlier matches if needed."""
    queue = deque([root])
    visited_pred, parent_ref = {root}, {}
    while queue:
        p = queue.popleft()
        for r in edges[p]:
            if r in parent_ref:
                continue
            parent_ref[r] = p
            if r not in matched_ref:
                while r is not None:
                    p = parent_ref[r]
                    previous = matched_pred.get(p)
                    matched_pred[p], matched_ref[r] = r, p
                    r = previous
                return True
            next_pred = matched_ref[r]
            if next_pred not in visited_pred:
                visited_pred.add(next_pred)
                queue.append(next_pred)
    return False


def match_events(references: Sequence[Mapping], predictions: Sequence[Mapping], *,
                 onset_tolerance_seconds: float | None = None, min_iou: float | None = None) -> dict:
    """Maximum-cardinality one-to-one matching, independently per recording.

    Events must have the same recording_id and optional label. Both timing
    constraints apply when supplied. Equal alternatives are deterministic
    (nearest onset then reference index); localization errors describe this
    matching, rather than a separately optimized localization assignment.
    """
    refs, preds = _events(references), _events(predictions)
    edges = _edges(refs, preds, onset_tolerance_seconds, min_iou)
    matched_pred, matched_ref = {}, {}
    for i in range(len(preds)):
        _augment(i, edges, matched_pred, matched_ref)
    matches = [{"prediction_index": p, "reference_index": r,
                "onset_error_seconds": preds[p]["start"] - refs[r]["start"]}
               for p, r in sorted(matched_pred.items())]
    return {"matches": matches,
            "unmatched_prediction_indices": [i for i in range(len(preds)) if i not in matched_pred],
            "unmatched_reference_indices": [i for i in range(len(refs)) if i not in matched_ref]}


def event_metrics(references: Sequence[Mapping], predictions: Sequence[Mapping], *,
                  onset_tolerance_seconds: float | None = None, min_iou: float | None = None,
                  duration_seconds: float | None = None, threshold: float | None = None) -> dict:
    """Ranked AP and fixed-threshold event recall/F1/FAR/localization.

    Score ties enter together. Duplicate detections remain false positives.
    AP uses maximum one-to-one matching at every score group; no greedy loss of
    feasible matches. Omitted threshold means all supplied events are retained.
    FAR exposure is the sum of evaluated recording durations, in seconds.
    """
    refs, preds = _events(references), _events(predictions, scored=True)
    if threshold is not None:
        threshold = _finite(threshold, "threshold")
    exposure = None if duration_seconds is None else _finite(duration_seconds, "duration_seconds")
    if exposure is not None and exposure <= 0:
        raise ValueError("duration_seconds must be positive")
    edges = _edges(refs, preds, onset_tolerance_seconds, min_iou)
    matched_pred, matched_ref = {}, {}
    area, seen = 0.0, 0
    for group in _score_groups([p["score"] for p in preds]):
        before = len(matched_pred)
        for i in group:
            _augment(i, edges, matched_pred, matched_ref)
        seen += len(group)
        if refs:
            area += (len(matched_pred) - before) / len(refs) * len(matched_pred) / seen
    selected = [i for i, p in enumerate(preds) if threshold is None or p["score"] >= threshold]
    matching = match_events(refs, [preds[i] for i in selected],
                            onset_tolerance_seconds=onset_tolerance_seconds, min_iou=min_iou)
    # Restore caller's prediction indices after thresholding.
    for match in matching["matches"]:
        match["prediction_index"] = selected[match["prediction_index"]]
    matching["unmatched_prediction_indices"] = [selected[i] for i in matching["unmatched_prediction_indices"]]
    tp = len(matching["matches"])
    fp, fn = len(selected) - tp, len(refs) - tp
    errors = [m["onset_error_seconds"] for m in matching["matches"]]
    undefined = {}
    if not refs:
        undefined.update(average_precision="no_positive_references", recall="no_positive_references")
    if exposure is None:
        undefined["false_alarms_per_hour"] = "missing_exposure"
    if not errors:
        undefined["onset_mae_seconds"] = "no_matches"
    return {"reference_count": len(refs), "prediction_count": len(preds), "selected_prediction_count": len(selected),
            "threshold": threshold, "onset_tolerance_seconds": onset_tolerance_seconds, "min_iou": min_iou,
            **_counts(tp, fp, fn), "average_precision": area if refs else None,
            "false_alarms_per_hour": fp * 3600 / exposure if exposure else None,
            "onset_mae_seconds": _mean([abs(e) for e in errors]),
            "onset_mean_signed_error_seconds": _mean(errors),
            "undefined_metrics": undefined, **matching}


def _percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def paired_recording_bootstrap_ci(recordings: Mapping[str, Any],
                                  statistic: Callable[[list[Any]], float | None], *,
                                  n_resamples: int = 2000, confidence: float = 0.95,
                                  seed: int = 0) -> dict:
    """Percentile CI resampling whole paired recording records with replacement.

    Each mapping value contains both models' predictions/metrics. ``statistic``
    receives sampled values and returns the paired difference (or any aggregate)
    using that same sample for both models. Undefined samples are explicitly
    counted and excluded from percentiles; no zero substitution is performed.
    """
    if not isinstance(recordings, Mapping):
        raise TypeError("recordings must map unique recording IDs to paired records")
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("positive n_resamples and confidence in (0,1) required")
    items = list(recordings.values())
    if not items:
        return {"estimate": None, "lower": None, "upper": None, "status": "empty", "seed": seed,
                "recording_count": 0, "n_resamples": n_resamples, "valid_resamples": 0,
                "undefined_resamples": n_resamples, "confidence": confidence}
    observed = statistic(items)
    if observed is not None:
        observed = _finite(observed, "observed statistic")
    generator, values = random.Random(seed), []
    for _ in range(n_resamples):
        value = statistic([items[generator.randrange(len(items))] for _ in items])
        if value is not None:
            values.append(_finite(value, "bootstrap statistic"))
    alpha = (1 - confidence) / 2
    return {"estimate": observed, "lower": _percentile(values, alpha) if values else None,
            "upper": _percentile(values, 1 - alpha) if values else None,
            "confidence": confidence, "seed": seed, "recording_count": len(items),
            "n_resamples": n_resamples, "valid_resamples": len(values),
            "undefined_resamples": n_resamples - len(values),
            "status": "undefined_statistic" if observed is None else ("ok" if len(values) == n_resamples else "undefined_resamples")}
