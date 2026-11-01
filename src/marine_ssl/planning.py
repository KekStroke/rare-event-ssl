"""Build auditable candidate queues from cached catalogs, without audio transfers."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .io import read_jsonl, write_json, write_jsonl
from .leakage import ExclusionIndex
from .sources import AUDIO_SUFFIXES, gcs_catalog, s3_catalog, recording_identity


def legacy_calendar_identity(row):
    """Preserve explicit legacy calendar time without claiming a UTC clock."""
    name = str(row.get("path_in_source") or "")
    if not name.startswith(("wholistener/", "orcasounds/")):
        return row
    match = re.search(r"(?:^|_)(\d{1,2})_(\d{1,2})_(20\d{2})_(\d{2})_(\d{2})_(\d{2})", Path(name).name)
    if not match:
        return row
    month, day, year, hour, minute, second = map(int, match.groups())
    try:
        local = datetime(year, month, day, hour, minute, second)
    except ValueError:
        return row
    result = {key: value for key, value in row.items()
              if key not in {"recording_start_utc", "recording_end_utc"}}
    station = re.search(r"(?:^|/)wholistener/(lk|os|bp|pt)(?:/|$)", name, re.I)
    station = station[1].lower() if station else "all-stations"
    result.update(source_recording_id=f"orcasound-legacy:{station}:{local.isoformat()}",
                  time_basis="unspecified_local_clock", recording_local_time=local.isoformat(),
                  recording_local_date=local.date().isoformat(), legacy_calendar_year=year,
                  session_id=f"orcasound-legacy:{local:%Y%m%d}",
                  identity_evidence="explicit_legacy_month_day_year_filename")
    return result


def legacy_year_is_separated(year, selected_test_years):
    """Unknown test coverage and same/adjacent years cannot clear a local clock."""
    return (isinstance(year, int) and not isinstance(year, bool)
            and bool(selected_test_years)
            and all(isinstance(other, int) and abs(year - other) >= 2 for other in selected_test_years))


def selected_orcasound_test_years(protected):
    years = set()
    for row in protected:
        if row.get("source_family") != "orcasound":
            continue
        session = str(_provenance(row).get("session_id") or "")
        match = re.fullmatch(r"orcasound:(20\d{2})\d{4}", session)
        if not match:
            return None  # Any unresolved selected-test year blocks this shortcut.
        years.add(int(match[1]))
    return years or None


def object_catalog(spec, *, session=None):
    """Explicit MLS-only metadata enumeration for configured S3/GCS prefixes."""
    bucket, kind = spec["bucket"], spec["kind"]
    if kind not in {"s3", "gcs"}:
        raise ValueError("object catalog requires s3 or gcs kind")
    fetch = gcs_catalog if kind == "gcs" else s3_catalog
    seen = set()
    for prefix in spec.get("prefixes", [""]):
        options = {"session": session}
        if kind == "s3" and "max_objects_per_prefix" in spec:
            options["max_objects"] = spec["max_objects_per_prefix"]
        for item in fetch(bucket, prefix, **options):
            name = item["name"]
            if name in seen or Path(name).suffix.lower() not in AUDIO_SUFFIXES:
                continue
            seen.add(name)
            host = "storage.googleapis.com/" + bucket if kind == "gcs" else bucket + ".s3.amazonaws.com"
            match = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", Path(name).name)
            day = "".join(match.groups()) if match else None
            yield {"source_family": spec["source_family"], "source_id": f"{kind}://{bucket}/{name}",
                   "path_in_source": name, "url": "https://" + host + "/" + quote(name),
                   "bytes": int(item["size"]), "session_id": day,
                   "source_md5_base64": item.get("md5Hash"), "source_etag": item.get("etag"),
                   "source_generation": item.get("generation"), "license": spec.get("license"),
                   **(recording_identity("orcasound", name) if spec["source_family"] == "orcasound" else {})}


def _provenance(row):
    """Do not invent an interval from a duration estimate."""
    result = dict(row)
    if not result.get("recording_end_utc"):
        result.pop("recording_start_utc", None)
    # Catalog helpers identify a sensor, not a complete acquisition deployment.
    # Reserve the original recording/session unless an explicit deployment is
    # supplied by the selected evaluation manifest.
    if result.get("deployment_id") == result.get("hydrophone_id"):
        result.pop("deployment_id", None)
    if result.get("source_family") == "orcasound":
        name = Path(result.get("source_id") or result.get("path_in_source", "")).stem
        date = str(row.get("recording_start_utc") or "")[:10]
        match = re.search(r"OS_(\d{1,2})_(\d{1,2})_(\d{4})_", name)
        if match:
            month, day, year = map(int, match.groups())
            date = datetime(year, month, day).strftime("%Y-%m-%d")
        if date:
            result["session_id"] = "orcasound:" + date.replace("-", "")
    return result


def _priority(row):
    labels = " ".join(str(annotation.get(key) or "")
                      for annotation in row.get("annotations", [])
                      for key in ("species_label_clean", "ecotype_label_clean")).lower()
    return 0 if "srkw" in labels else 1 if "orca" in labels else 2 if labels.strip() else 3


def _diverse(rows, seed):
    """Prioritize labels, then round-robin acquisition days deterministically."""
    groups = defaultdict(list)
    key = lambda row: hashlib.sha256(f"{seed}:{row.get('source_id')}".encode()).hexdigest()
    for row in rows:
        day = row.get("session_id") or (row.get("recording_start_utc") or "")[:10]
        groups[(_priority(row), day)].append(row)
    for priority in range(4):
        days = [sorted(group, key=key) for (rank, _), group in sorted(groups.items()) if rank == priority]
        for position in range(max(map(len, days), default=0)):
            for group in days:
                if position < len(group):
                    yield group[position]


def build_plan(config, catalog_dir, *, protected_manifest=None):
    """Return candidate/rejection rows and quota estimates; never claim audio clean."""
    directory = Path(catalog_dir)
    protected = []
    missing = []
    for filename in config["protected_catalogs"]:
        path = directory / filename
        if path.is_file():
            protected.extend(_provenance(row) for row in read_jsonl(path))
        else:
            missing.append(filename)
    if protected_manifest:
        protected.extend(_provenance(row) for row in read_jsonl(protected_manifest))
    if missing:
        for family in ("onc", "orcasound"):
            protected.append({"source_family": family, "source_id": "missing-required-protected-catalog"})
    selected_test_years = selected_orcasound_test_years(protected)
    index = ExclusionIndex(protected, at_risk_families=("onc", "orcasound"))
    candidates, rejected, summaries = [], [], {}
    seen_sources, seen_hashes = set(), set()
    for source, spec in config["sources"].items():
        path = directory / spec["catalog"]
        accepted, counts = [], Counter()
        if path.is_file():
            for row in read_jsonl(path):
                row = {**row, "source_key": source, "target_hours": spec["target_hours"]}
                if row.get("source_family") == "orcasound":
                    row = legacy_calendar_identity(row)
                provenance = _provenance(row)
                reasons = set(index.reasons(provenance))
                if (row.get("time_basis") == "unspecified_local_clock"
                        and not legacy_year_is_separated(row.get("legacy_calendar_year"), selected_test_years)):
                    reasons.update({"unresolved_pretrain_provenance", "legacy_year_not_separated_from_selected_tests"})
                original = str(row.get("path_in_source") or "")
                if row.get("source_family") != spec["source_family"]:
                    reasons.add("unexpected_source_family")
                if spec.get("revision") and row.get("revision") != spec["revision"]:
                    reasons.add("unexpected_source_revision")
                if original.startswith("test/"):
                    reasons.add("upstream_test_split")
                if any(prefix.lower() in original.lower() for prefix in spec.get("excluded_prefixes", [])):
                    reasons.add("reserved_source_prefix")
                identity = (spec["source_family"], row.get("source_recording_id") or row.get("source_id"))
                sha = row.get("sha256")
                if identity in seen_sources or (sha and sha in seen_hashes):
                    reasons.add("duplicate_source_or_hash")
                if reasons:
                    rejected.append({**row, "exclusion_reasons": sorted(reasons)})
                    counts.update(reasons)
                    continue
                seen_sources.add(identity)
                if sha:
                    seen_hashes.add(sha)
                row.update({key: value for key, value in provenance.items()
                            if key in {"session_id", "deployment_id", "hydrophone_id"}})
                if "deployment_id" not in provenance:
                    row.pop("deployment_id", None)
                if not row.get("duration_seconds") and spec.get("duration_estimate_seconds"):
                    row["duration_estimate_seconds"] = spec["duration_estimate_seconds"]
                row["clearance_scope"] = "source_metadata_only"
                accepted.append(row)
        candidates.extend(_diverse(accepted, config["seed"]))
        estimates = [row.get("duration_seconds") or row.get("duration_estimate_seconds") for row in accepted]
        summaries[source] = {"target_hours": spec["target_hours"], "candidate_count": len(accepted),
                             "catalog_missing": not path.is_file(),
                             "candidate_bytes": sum(row.get("bytes") or 0 for row in accepted),
                             "candidate_estimated_hours": sum(value or 0 for value in estimates) / 3600,
                             "unknown_duration_count": sum(value is None for value in estimates),
                             "exclusion_reasons": dict(counts),
                             "quota_met": False}
    target = sum(spec["target_hours"] for spec in config["sources"].values())
    active_families = {spec["source_family"] for spec in config["sources"].values()}
    unresolved = sorted(index.unresolved_families & active_families)
    return {"candidates": candidates, "rejected": rejected, "summary": {
        "status": "blocked" if unresolved else "metadata_candidates_only",
        "created_utc": datetime.now(timezone.utc).isoformat(), "sources": summaries,
        "unresolved_families": unresolved,
        "missing_protected_catalogs": missing,
        "protected_catalogs": list(config["protected_catalogs"]),
        "extra_protected_manifest": str(protected_manifest) if protected_manifest else None,
        "cross_deployment_holdout_status": "not_defined",
        "training_ready": False,
        "prepared_pcm16_bytes_at_quota": int(target * 3600 * config["sample_rate"] * 2),
        "five_second_windows_at_quota": int(target * 3600 / config["segment_seconds"]),
        "collector_rule": "Read candidate queue until verified usable duration reaches each quota; estimates never establish readiness.",
        "verification_required": ["decode/count/duration/sample rate/channels", "source and decoded PCM hashes", "cross-encoding/cropped overlap where provenance is incomplete"]}}


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "catalog-dir", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--protected-manifest", help="Original-source rows for additional explicitly selected evaluation audio")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    plan = build_plan(config, args.catalog_dir, protected_manifest=args.protected_manifest)
    output = Path(args.output)
    write_jsonl(output / "pretrain_candidates.jsonl", plan["candidates"])
    write_jsonl(output / "pretrain_excluded.jsonl", plan["rejected"])
    write_json(output / "pretrain_plan.json", plan["summary"])
    return int(plan["summary"]["status"] == "blocked")


if __name__ == "__main__":
    raise SystemExit(main())
