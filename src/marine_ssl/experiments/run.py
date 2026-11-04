"""MLS entry point: freeze cohorts, extract features, then fit CPU heads."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .datasets import prepare_datasets, readiness_report, save_manifest, stable_hash
from .encoders import EncoderSpec, load_encoder
from .evaluate import evaluate_cohorts, write_json
from .features import FeatureStore


def published_models(config):
    shared = Path(config["shared_root"])
    retained = json.loads((shared / "models/legacy_selected/20261026/MANIFEST.json").read_text())
    aves = next(row for row in retained["files"] if row["name"] == "aves_base_bio__published_encoder.pt")
    inventory = json.loads(Path(config["inventory_index"]).read_text())
    aves_config = next(row for row in inventory["files"] if row["archive_member"].endswith("/aves-base-bio.torchaudio.model_config.json"))
    hubert = json.loads((shared / "models/rare-event-ssl/baselines/hubert-base-ls960/selected.json").read_text())
    hf = {row["name"]: row for row in hubert["files"]}
    models = {
        "aves_published": EncoderSpec("aves_torchaudio", aves["destination"], config=aves_config["path"],
            expected_sha256=aves["sha256"], config_sha256=aves_config["sha256"]),
        "hubert_published": EncoderSpec("hubert_transformers", hubert["path"],
            expected_sha256=hf["pytorch_model.bin"]["sha256"], config_sha256=hf["config.json"]["sha256"],
            preprocessor_sha256=hf["preprocessor_config.json"]["sha256"], revision=hubert["revision"], model_id=hubert["model_id"])}
    for name, spec in config.get("additional_models", {}).items():
        if name in models:
            raise ValueError("Registered model IDs are immutable")
        models[name] = EncoderSpec(**spec)
    return models


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("freeze", "features", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run_root = Path(config["report_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    config_sha = stable_hash(config)
    freeze = run_root / "cohorts"
    if args.stage == "freeze":
        save_manifest({"datasets": config["datasets"]}, freeze / "dataset_config.json")
        cohorts = prepare_datasets(config)
        for task, result in cohorts.items():
            save_manifest(result, freeze / (task + ".json"))
        report = readiness_report(cohorts)
        write_json(run_root / "readiness.json", report)
        print(json.dumps(report), flush=True)
        if not report["ready"]:
            raise SystemExit("A requested ready-audio task failed metadata preflight")
        return
    if json.loads((freeze / "dataset_config.json").read_text()) != {"datasets": config["datasets"]}:
        raise ValueError("Dataset configuration differs from immutable cohort freeze")
    cohorts = {task: json.loads((freeze / (task + ".json")).read_text()) for task in config["datasets"]}
    for cohort in cohorts.values():
        identity = {key: cohort[key] for key in ("schema_version", "task", "protocol", "rows")}
        identity["input_metadata_sha256"] = sorted(cohort["input_metadata_sha256"].values())
        if stable_hash(identity) != cohort["fingerprint"]:
            raise ValueError("Saved cohort content no longer matches its immutable fingerprint")
    models = published_models(config)
    selected = args.models or config["models"]
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchaudio", "transformers", "numpy", "scipy", "scikit-learn", "soundfile")}
    summary = {"stage": args.stage, "config_sha256": config_sha, "versions": versions,
               "started_utc": datetime.now(timezone.utc).isoformat(), "models": {},
               "code_revision": config["code_revision"], "status": "running"}
    # Independent model jobs share cohorts/caches, but never a status writer.
    report = run_root / (args.stage + "_status_" + stable_hash(selected)[:12] + ".json")
    for name in selected:
        spec = models[name]
        encoder = load_encoder(spec, device="cuda:0") if args.stage == "features" else None
        cache = Path(config["cache_root"]) / (name + ".sqlite")
        store = FeatureStore(cache, spec, encoder=encoder, layers=config["head"]["layers"],
                             max_bytes=config["cache_budget_bytes_per_model"], readonly=args.stage == "evaluate")
        print(json.dumps({"stage": args.stage, "model": name, "spec": asdict(spec)}), flush=True)
        try:
            if args.stage == "features":
                store.prepare(cohorts)
                result = {"status": "complete", "computed": store.computed, "cache_hits": store.hits,
                          "cache": str(cache), "cache_bytes": cache.stat().st_size}
            else:
                result = evaluate_cohorts(cohorts, store, run_root / "models" / name, config["head"])
            summary["models"][name] = result
            write_json(report, summary)
        finally:
            store.close()
            del store, encoder
            gc.collect()
            if args.stage == "features":
                import torch
                torch.cuda.empty_cache()
    summary.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(report, summary)
    print(json.dumps({"stage": args.stage, "status": "complete", "report": str(report)}), flush=True)


if __name__ == "__main__":
    main()
