"""MLS metadata-only inventory. No audio downloads or changes to existing data."""

import argparse
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .io import write_json
from .sources import AUDIO_SUFFIXES, hf_catalog, hf_audio_rows


def run(shared_root, output, config=None):
    root, output = Path(shared_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print(json.dumps({"stage": "inventory", "started_utc": started, "root": str(root)}), flush=True)
    usage = shutil.disk_usage(root)
    existing = {}
    next_progress = time.monotonic() + 60
    for directory, dirs, files in os.walk(root / "datasets" / "rare-event-ssl", followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        relative = Path(directory).relative_to(root / "datasets")
        group = "/".join(relative.parts[:3])
        entry = existing.setdefault(group, {"files": 0, "bytes": 0, "audio_files": 0})
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            entry["files"] += 1
            entry["bytes"] += path.stat().st_size
            entry["audio_files"] += path.suffix.lower() in AUDIO_SUFFIXES
        if time.monotonic() >= next_progress:
            print(json.dumps({"stage": "inventory", "groups": len(existing)}), flush=True)
            next_progress = time.monotonic() + 60
    summary = {"started_utc": started, "disk_total_bytes": usage.total,
               "disk_free_bytes": usage.free, "existing": existing, "sources": {}, "errors": {}}
    for repo, family in [("DORI-SRKW/ONC-Benchmark", "onc"),
                         ("DORI-SRKW/orcasound_hackathon2024", "orcasound"),
                         ("DORI-SRKW/DORI-Orcasound", "orcasound"),
                         ("DORI-SRKW/DORI-ONC", "onc")]:
        print(json.dumps({"stage": "hf_metadata", "repo": repo}), flush=True)
        try:
            catalog = hf_catalog(repo, output / "metadata")
            rows = list(hf_audio_rows(catalog, family))
            from .io import write_jsonl
            write_jsonl(output / "catalogs" / (repo.split("/")[-1] + ".jsonl"), rows)
            summary["sources"][repo] = {"revision": catalog["revision"], "audio_files": len(rows),
                "bytes": sum(row.get("bytes") or 0 for row in rows),
                "tables": {name: len(table) for name, table in catalog["tables"].items()},
                "directories": dict(Counter(row["path_in_source"].split("/")[0] for row in rows))}
        except Exception as exc:
            summary["errors"][repo] = f"{type(exc).__name__}: {exc}"
        write_json(output / "inventory.json", summary)
    if config:
        import yaml
        from .benchmarks import plan_benchmarks
        with Path(config).open(encoding="utf-8") as stream:
            benchmark_config = yaml.safe_load(stream)
        plan = plan_benchmarks(benchmark_config)
        write_json(output / "benchmark_plan.json", plan)
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "inventory.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark-config")
    args = parser.parse_args()
    result = run(args.shared_root, args.output, args.benchmark_config)
    raise SystemExit(bool(result["errors"]))
