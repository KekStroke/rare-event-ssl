"""Frozen-encoder transfer heads with validation-only selection.

Published encoders, new DAPT encoders and the historical reference all use this
same implementation. External ONC/Voices never fit or select a head.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time
import warnings

import numpy as np

from .metrics import binary_metrics, multiclass_metrics


BINARY_TASKS = {"avacha_presence", "hiceas", "hackathon", "onc"}
TEMPORAL_TASKS = {"avacha_field", "manual", "dominica"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validation_threshold(labels, scores):
    """Best validation F1; ties prefer the higher threshold. No test input."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        raise ValueError("No validation score thresholds")
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros_like(precision[:-1]), where=precision[:-1] + recall[:-1] != 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


def _matrix(rows, store, layer):
    return np.stack([store.get(row)[f"pooled_{layer}"] for row in rows])


def _predictions(rows, scores, classes, *, binary, threshold, task=None):
    output = []
    for row, probabilities in zip(rows, scores):
        if binary:
            score = float(probabilities[list(classes).index(1)])
            predicted = int(score >= threshold)
        else:
            score = None
            predicted = classes[int(np.argmax(probabilities))]
            if isinstance(predicted, np.generic):
                predicted = predicted.item()
        date = re.search(r"(?<!\d)(20\d{6})(?!\d)", str(row.get("path", row["recording_id"]))) if task == "hiceas" else None
        output.append({"row_id": row["row_id"], "recording_id": row["recording_id"], "group": row["group"],
                       "bootstrap_group": "hiceas:day:" + date.group(1) if date else row["group"],
                       "bootstrap_group_basis": "source_day" if date else "cohort_source_group",
                       "truth": row["label"], "prediction": predicted, "score": score,
                       "probabilities": [float(p) for p in probabilities]})
    return output


def score_predictions(predictions, classes, *, binary, threshold):
    if binary:
        return binary_metrics([int(p["truth"]) for p in predictions], [p["score"] for p in predictions], threshold=threshold)
    return multiclass_metrics([p["truth"] for p in predictions], [p["prediction"] for p in predictions], labels=list(classes))


def selection_metric(metrics, cohort, *, binary):
    if binary:
        return metrics["average_precision"]
    labels = cohort["protocol"].get("selection_labels")
    if labels is None:
        return metrics["macro_f1"]
    allowed = set(map(str, labels))
    values = [r["f1"] for r in metrics["per_class"] if str(r["label"]) in allowed and r["f1"] is not None]
    return float(np.mean(values)) if values else None


def fit_task(cohort, store, output_dir, config):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    task = cohort["task"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {split: [row for row in cohort["rows"] if row["split"] == split] for split in ("train", "val", "test")}
    if not all(splits.values()):
        raise ValueError("Head training requires explicit nonempty train/validation/test")
    binary = task in BINARY_TASKS
    y = {name: np.asarray([row["label"] for row in rows]) for name, rows in splits.items()}
    candidates, winner, winner_score = [], None, -float("inf")
    for layer in config["layers"]:
        train, validation = _matrix(splits["train"], store, layer), _matrix(splits["val"], store, layer)
        for c in config["C"]:
            tick = time.monotonic()
            head = make_pipeline(StandardScaler(), LogisticRegression(C=float(c), class_weight="balanced",
                solver="lbfgs", max_iter=int(config.get("max_iter", 1000)), random_state=int(config["seed"]), n_jobs=1))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                head.fit(train, y["train"])
            converged = not any(isinstance(w.message, ConvergenceWarning) for w in caught)
            if not converged:
                raise RuntimeError(f"Unconverged downstream head: task={task} layer={layer} C={c}; no test selected")
            scores = head.predict_proba(validation)
            classes = head.classes_.tolist()
            threshold = validation_threshold(y["val"], scores[:, classes.index(1)]) if binary else None
            predictions = _predictions(splits["val"], scores, classes, binary=binary, threshold=threshold, task=task)
            metrics = score_predictions(predictions, classes, binary=binary, threshold=threshold)
            objective = selection_metric(metrics, cohort, binary=binary)
            candidate = {"layer": layer, "C": c, "validation": metrics, "converged": converged,
                         "selection_objective": objective, "selection_labels": cohort["protocol"].get("selection_labels"),
                         "fit_seconds": time.monotonic() - tick}
            candidates.append(candidate)
            print(json.dumps({"stage": "head_candidate", "task": task, **candidate}), flush=True)
            # Stable order is the predeclared tie-break: lower layer, then smaller C.
            if objective is not None and objective > winner_score:
                winner_score = objective
                winner = {"head": head, "layer": layer, "C": c, "threshold": threshold, "classes": classes}
    if winner is None:
        raise ValueError("No defined validation selection metric")
    head = winner.pop("head")
    joblib.dump({**winner, "head": head, "cohort_fingerprint": cohort["fingerprint"],
                 "task": task, "feature_identity": store.identity}, output_dir / "head.joblib", compress=3)
    test_scores = head.predict_proba(_matrix(splits["test"], store, winner["layer"]))
    predictions = _predictions(splits["test"], test_scores, winner["classes"], binary=binary, threshold=winner["threshold"], task=task)
    result = {"task": task, "status": "complete", "cohort_fingerprint": cohort["fingerprint"],
              "selection": winner, "candidates": candidates, "test": score_predictions(predictions,
                  winner["classes"], binary=binary, threshold=winner["threshold"]),
              "test_recordings": len({row["recording_id"] for row in splits["test"]}),
              "test_source_groups": len({row["group"] for row in splits["test"]}),
              "notes": cohort["notes"], "protocol": cohort["protocol"]}
    write_json(output_dir / "metrics.json", result)
    write_json(output_dir / "test_predictions.json", {"classes": winner["classes"], "rows": predictions})
    return result


def evaluate_external(cohort, store, output_dir, source_head_path):
    import joblib
    fitted = joblib.load(source_head_path)
    if fitted["task"] != cohort["protocol"]["requires_head_from"] or fitted["feature_identity"] != store.identity:
        raise ValueError("External evaluation head/model provenance mismatch")
    rows = cohort["rows"]
    if any(row["split"] != "test" for row in rows):
        raise ValueError("External cohort must contain only test rows")
    layer, head = fitted["layer"], fitted["head"]
    binary = cohort["task"] in BINARY_TASKS
    scores = []
    for row in rows:
        features = store.get(row)
        if binary:
            # Predeclared OR-presence aggregation, no target-domain tuning.
            chunk_probabilities = head.predict_proba(features[f"chunks_{layer}"])
            positive = float(chunk_probabilities[:, fitted["classes"].index(1)].max())
            scores.append([positive if c == 1 else 1 - positive for c in fitted["classes"]])
        else:
            scores.append(head.predict_proba(features[f"pooled_{layer}"][None])[0])
    predictions = _predictions(rows, scores, fitted["classes"], binary=binary, threshold=fitted["threshold"], task=cohort["task"])
    result = {"task": cohort["task"], "status": "complete_auxiliary" if cohort.get("auxiliary") else "complete",
              "source_head_task": fitted["task"], "source_head_cohort": fitted["cohort_fingerprint"],
              "cohort_fingerprint": cohort["fingerprint"], "selection": {k: fitted[k] for k in ("layer", "C", "threshold")},
              "protocol": cohort["protocol"], "notes": cohort["notes"],
              "test": score_predictions(predictions, fitted["classes"], binary=binary, threshold=fitted["threshold"])}
    if cohort["task"] == "voices":
        shared = set(cohort["protocol"].get("ontology_mapping", {}).values())
        per_shared = [r for r in result["test"]["per_class"] if r["label"] in shared and r["f1"] is not None]
        result["test"]["shared_ontology_macro_f1"] = float(np.mean([r["f1"] for r in per_shared])) if per_shared else None
        result["test"]["shared_ontology_classes"] = sorted(shared)
        result["test"]["out_of_shared_ontology_predictions"] = sum(p["prediction"] not in shared for p in predictions)
        result["test"]["primary"] = "shared_ontology_macro_f1"
    write_json(Path(output_dir) / "metrics.json", result)
    write_json(Path(output_dir) / "test_predictions.json", {"classes": fitted["classes"], "rows": predictions})
    return result


def evaluate_cohorts(cohorts, store, output_dir, config, *, include_temporal=True):
    output_dir = Path(output_dir)
    summaries = {}
    tasks = [task for task in cohorts if task not in {"onc", "voices"}] + [task for task in ("onc", "voices") if task in cohorts]
    for task in tasks:
        cohort = cohorts[task]
        if not cohort["ready"]:
            summaries[task] = {"status": "not_ready", "reasons": cohort["reasons"]}
        elif task in TEMPORAL_TASKS:
            if include_temporal:
                from .temporal import evaluate_temporal
                summaries[task] = evaluate_temporal(cohort, store.temporal_loader, output_dir / task, config)
            else:
                summaries[task] = {"status": "pending_temporal_stage"}
        elif cohort["protocol"].get("requires_head_from"):
            source = cohort["protocol"]["requires_head_from"]
            summaries[task] = evaluate_external(cohort, store, output_dir / task, output_dir / source / "head.joblib")
        else:
            summaries[task] = fit_task(cohort, store, output_dir / task, config)
        write_json(output_dir / "summary.json", summaries)
    return summaries
