"""Paired, source-group bootstrap for saved downstream predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import BINARY_TASKS, write_json
from .metrics import binary_metrics


def _primary(rows, classes, binary, weights=None, shared=None):
    if weights is None:
        weights = np.ones(len(rows), dtype=int)
    if binary:
        truth, scores = [], []
        for row, count in zip(rows, weights):
            truth.extend([int(row["truth"])] * int(count))
            scores.extend([row["score"]] * int(count))
        return binary_metrics(truth, scores, threshold=.5)["average_precision"]
    index = {label: i for i, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for row, count in zip(rows, weights):
        matrix[index[row["truth"]], index[row["prediction"]]] += count
    denom = matrix.sum(0) + matrix.sum(1)
    eligible = denom > 0
    if shared is not None:
        eligible &= np.asarray([label in shared for label in classes])
    f1 = np.divide(2 * matrix.diagonal(), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(f1[eligible].mean()) if eligible.any() else None


def paired_comparison(reference, candidate, *, classes, binary, n_resamples=2000, seed=42, shared=None):
    left = {row["row_id"]: row for row in reference}
    right = {row["row_id"]: row for row in candidate}
    if left.keys() != right.keys():
        raise ValueError("Paired comparisons require identical immutable test row IDs")
    ids = sorted(left)
    a, b = [left[key] for key in ids], [right[key] for key in ids]
    if any(x["truth"] != y["truth"] or x["group"] != y["group"]
           or x.get("bootstrap_group", x["group"]) != y.get("bootstrap_group", y["group"]) for x, y in zip(a, b)):
        raise ValueError("Truth/source-group mismatch between model predictions")
    groups = sorted({row.get("bootstrap_group", row["group"]) for row in a})
    gi = {group: i for i, group in enumerate(groups)}
    positions = np.asarray([gi[row.get("bootstrap_group", row["group"])] for row in a])
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_resamples):
        counts = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        weights = counts[positions]
        va = _primary(a, classes, binary, weights, shared)
        vb = _primary(b, classes, binary, weights, shared)
        if va is not None and vb is not None:
            differences.append(vb - va)
    va, vb = _primary(a, classes, binary, shared=shared), _primary(b, classes, binary, shared=shared)
    return {"reference": va, "candidate": vb,
            "difference_candidate_minus_reference": vb - va if va is not None and vb is not None else None,
            "paired_source_bootstrap_95_percent": np.quantile(differences, [.025, .975]).tolist() if differences else None,
            "resamples": n_resamples, "defined_resamples": len(differences), "seed": seed,
            "bootstrap_groups": len(groups), "test_rows": len(ids),
            "group_basis": sorted({row.get("bootstrap_group_basis", "cohort_source_group") for row in a}),
            "independence": "Source/session independence is not implied by recording identifiers; see cohort provenance",
            "limitation": "very few independent groups" if len(groups) < 10 else None}


def compare_run(report_root, *, reference="hubert_published", n_resamples=2000, seed=42):
    root = Path(report_root)
    base = root / "models" / reference
    output = {}
    for candidate in sorted((root / "models").iterdir()):
        if candidate.name == reference or not candidate.is_dir():
            continue
        results = {}
        for task in sorted(base.iterdir()):
            prediction = task / "test_predictions.json"
            other = candidate / task.name / "test_predictions.json"
            if not prediction.is_file() or not other.is_file():
                temporal_a, temporal_b = task / "temporal_predictions.jsonl", candidate / task.name / "temporal_predictions.jsonl"
                if temporal_a.is_file() and temporal_b.is_file():
                    from .compare_temporal import paired_temporal_comparison
                    results[task.name] = paired_temporal_comparison(temporal_a, temporal_b,
                        n_resamples=n_resamples, seed=seed)
                else:
                    results[task.name] = {"status": "not_available", "reason": "paired prediction artifacts not complete"}
                continue
            a, b = json.loads(prediction.read_text()), json.loads(other.read_text())
            if a["classes"] != b["classes"]:
                raise ValueError("Model head ontologies differ")
            shared = {r["truth"] for r in a["rows"]} if task.name == "voices" else None
            results[task.name] = paired_comparison(a["rows"], b["rows"], classes=a["classes"],
                binary=task.name in BINARY_TASKS, n_resamples=n_resamples, seed=seed, shared=shared)
        output[candidate.name] = results
    write_json(root / "paired_comparisons.json", {"reference_model": reference, "models": output})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("report_root")
    args = parser.parse_args()
    compare_run(args.report_root)
