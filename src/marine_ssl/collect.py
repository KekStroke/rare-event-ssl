"""Collect only configured pretrain sources, using cached selected-test catalogs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import threading

from .io import client, read_jsonl, write_json, write_jsonl
from .planning import build_plan, object_catalog, legacy_calendar_identity
from .prepare import prepare


def _normalize_orcasound(row):
    """Use explicit filename/prefix dates; ambiguous originals stay unresolved."""
    legacy = legacy_calendar_identity(row)
    if legacy.get("time_basis") == "unspecified_local_clock":
        return legacy
    if row.get("recording_start_utc"):
        return row
    row = {key: value for key, value in row.items() if key != "session_id"}
    name = str(row.get("path_in_source", ""))
    os_date = re.search(r"OS_(\d{1,2})_(\d{1,2})_(\d{4})_", name, re.I)
    if os_date:
        month, day, year = map(int, os_date.groups())
        try:
            date = datetime(year, month, day, tzinfo=timezone.utc)
            return {**row, "session_id": f"orcasound:{date:%Y%m%d}", "recording_start_utc": date.isoformat(),
                    "identity_evidence": "original_OS_month_day_year_filename"}
        except ValueError:
            return row
    match = re.search(r"(20\d{2})[-_]?([01]?\d)[-_]?([0-3]?\d)(?:\D|$)", name)
    if match:
        try:
            date = datetime(*map(int, match.groups()), tzinfo=timezone.utc)
            return {**row, "session_id": f"orcasound:{date:%Y%m%d}",
                    "recording_start_utc": date.isoformat(), "identity_evidence": "dated_source_object_path"}
        except ValueError:
            pass
    return row


def collect(config, catalog_dir, output, *, workers=8, plan_only=False):
    """Discover bounded prefixes, screen selected sources, then meet actual quotas."""
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required in the job image")
    original, output = Path(catalog_dir), Path(output)
    cached = output / "catalogs"
    cached.mkdir(parents=True, exist_ok=True)
    errors, metadata_counts = {}, {}
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(60):
            print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "stage": "metadata",
                              "catalog_counts": dict(metadata_counts), "errors": list(errors)}), flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    print(json.dumps({"stage": "collect_start", "plan_only": plan_only, "workers": workers,
                      "targets": {key: spec["target_hours"] for key, spec in config["sources"].items()}}), flush=True)
    try:
        for filename in config["protected_catalogs"]:
            path = original / filename
            if not path.is_file():
                errors[filename] = "Selected downstream catalog is missing"
            else:
                write_jsonl(cached / filename, read_jsonl(path))
        for source, spec in config["sources"].items():
            filename = spec.get("hf_catalog", spec["catalog"])
            path = original / filename
            if spec.get("repo"):
                if not path.is_file():
                    errors[source] = "Cached pretrain Hugging Face catalog is missing"
                    write_jsonl(cached / spec["catalog"], [])
                    continue
                revision = spec.get("hf_revision") or spec.get("revision")
                rows = list(read_jsonl(path))
                if any(row.get("revision") != revision for row in rows):
                    raise ValueError(f"{source}: cached HF catalog differs from pinned revision")
                write_jsonl(cached / spec["catalog"], rows)

        def discover(source, spec):
            if not spec.get("prefixes") or any(not prefix for prefix in spec["prefixes"]):
                raise ValueError("Remote discovery requires explicit nonempty prefixes")
            rows = list(object_catalog(spec, session=client()))
            if spec["source_family"] == "orcasound":
                rows = [_normalize_orcasound(row) for row in rows]
            return rows

        jobs = {source: spec for source, spec in config["sources"].items() if spec.get("kind") in {"s3", "gcs"}}
        with ThreadPoolExecutor(max_workers=min(workers, 3)) as pool:
            futures = {pool.submit(discover, source, spec): source for source, spec in jobs.items()}
            for future in as_completed(futures):
                source = futures[future]
                spec = jobs[source]
                target = cached / spec["catalog"]
                try:
                    remote = future.result()
                    base = list(read_jsonl(target)) if spec.get("hf_catalog") and target.exists() else []
                    write_jsonl(target, base + remote)
                    metadata_counts[source] = {"cached_hf": len(base), "remote_objects": len(remote)}
                except Exception as exc:
                    errors[source] = f"{type(exc).__name__}: {exc}"
                    if not spec.get("hf_catalog"):
                        write_jsonl(target, [])
        plan = build_plan(config, cached)
        write_jsonl(output / "pretrain_candidates.jsonl", plan["candidates"])
        write_jsonl(output / "pretrain_excluded.jsonl", plan["rejected"])
        write_json(output / "pretrain_plan.json", plan["summary"])
    finally:
        stop.set(); thread.join(timeout=1)
    summary = {"metadata_errors": errors, "catalog_counts": metadata_counts,
               "plan": plan["summary"], "collection_complete": False, "training_ready": False}
    if not plan_only and plan["candidates"]:
        try:
            result = prepare(config, plan["candidates"], output / "prepared", workers=workers)
            summary["preparation"] = result
            summary["collection_complete"] = all(source["quota_met"] for source in result["sources"].values())
        except Exception as exc:
            summary["preparation_error"] = f"{type(exc).__name__}: {exc}"
    write_json(output / "collection.json", summary)
    print(json.dumps({"stage": "collect_finished", "collection_complete": summary["collection_complete"],
                      "metadata_errors": errors, "preparation_error": summary.get("preparation_error"),
                      "sources": summary.get("preparation", {}).get("sources", plan["summary"]["sources"])}), flush=True)
    return summary


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "catalog-dir", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    summary = collect(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")),
                      args.catalog_dir, args.output, workers=args.workers, plan_only=args.plan_only)
    return int(bool(summary["metadata_errors"]) or (not args.plan_only and not summary["collection_complete"]))


if __name__ == "__main__":
    raise SystemExit(main())
