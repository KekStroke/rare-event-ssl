"""Audit only explicit, actually prepared/verified audio manifests.

Run on an MLS CPU worker. This reads referenced receipts and selected benchmark
catalogs, never the entire pretrain candidate queue or an audio directory tree.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import re

from .io import read_jsonl, write_json
from .leakage import ExclusionIndex, canonical_family
from .planning import _provenance
from .sources import recording_identity


def _slug(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _original(row, duration=None):
    result = dict(row)
    family = canonical_family(result.get("source_family"))
    result["source_family"] = family
    if family in {"onc", "orcasound"} and not result.get("session_id"):
        name = result.get("relative_path") or result.get("path_in_source") or result.get("source_id", "")
        identity = recording_identity(family, str(name))
        if identity.get("session_id"):
            result.update(identity)
    if result.get("recording_start_utc") and duration:
        start = datetime.fromisoformat(result["recording_start_utc"].replace("Z", "+00:00"))
        if start.tzinfo is not None:
            result["recording_end_utc"] = (start + timedelta(seconds=duration)).isoformat()
    # Benchmark int32/native PCM hashes and prepared int16/resampled hashes use
    # different byte contracts. Compare original-file SHA-256 only here.
    result.pop("pcm_sha256", None)
    return _provenance(result)


def audit_manifests(prepared_manifest, benchmark_manifests, catalog_dir):
    """Return bounded evidence; metadata clearance is not an acoustic proof."""
    benchmark_manifests = list(benchmark_manifests)
    missing, problems, protected, prepared = [], [], [], []
    catalogs = {}
    catalog_dir = Path(catalog_dir)

    def enrich(row):
        dataset = str(row.get("dataset") or "")
        slug = _slug(dataset)
        if slug not in catalogs:
            known = {"oncbenchmark": "ONC-Benchmark.jsonl",
                     "orcasoundhackathon2024": "orcasound_hackathon2024.jsonl"}
            filename = known.get(slug)
            if filename is None and re.fullmatch(r"[A-Za-z0-9_.-]+", dataset):
                filename = dataset + ".jsonl"
            path = catalog_dir / filename if filename else None
            exact, basenames = {}, defaultdict(list)
            if path and path.is_file():
                for item in read_jsonl(path):
                    source_path = item.get("path_in_source")
                    if source_path:
                        exact[source_path] = item
                        basenames[Path(source_path).name].append(item)
            catalogs[slug] = (exact, basenames)
        exact, basenames = catalogs[slug]
        relative = str(row.get("relative_path") or row.get("path_in_source") or "")
        match = exact.get(relative)
        if match is None:
            matches = basenames.get(Path(relative).name, [])
            if len(matches) == 1:
                match = matches[0]
        if match is not None:
            # Keep hashes/duration from the verified local file; retain original
            # source identity and acquisition metadata from its pinned catalog.
            row = {**row, **{key: value for key, value in match.items()
                            if key in {"source_id", "source_family", "session_id", "deployment_id",
                                       "hydrophone_id", "recording_start_utc", "recording_end_utc"}}}
        return _original(row, row.get("duration_seconds"))

    for filename in benchmark_manifests:
        path = Path(filename)
        if not path.is_file():
            missing.append(str(path))
            continue
        for row in read_jsonl(path):
            if row.get("role") != "audio" and not row.get("full_decode_verified"):
                continue
            if not row.get("full_decode_verified"):
                problems.append("benchmark_audio_not_decode_verified")
                continue
            row = enrich(row)
            if not row.get("source_family"):
                problems.append("benchmark_source_family_unresolved")
            else:
                protected.append(row)
    path = Path(prepared_manifest)
    if not path.is_file():
        missing.append(str(path))
    else:
        for row in read_jsonl(path):
            source = dict(row.get("source") or row)
            receipt_path = row.get("receipt_path")
            if not receipt_path or not Path(receipt_path).is_file():
                problems.append("prepared_receipt_missing")
                continue
            receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
            if not receipt.get("full_decode_verified"):
                problems.append("prepared_audio_not_decode_verified")
                continue
            if receipt.get("source") and receipt["source"].get("source_id") != source.get("source_id"):
                problems.append("prepared_receipt_source_mismatch")
                continue
            if receipt.get("source_sha256"):
                source["sha256"] = receipt["source_sha256"]
            frames, rate = receipt.get("decoded_frames"), receipt.get("sample_rate_hz")
            prepared.append(_original(source, frames / rate if frames and rate else None))
    index = ExclusionIndex(protected)
    counts, examples = Counter(), []
    for row in prepared:
        reasons = index.reasons(row)
        if reasons:
            counts.update(reasons)
            if len(examples) < 25:
                examples.append({"source_family": row.get("source_family"),
                                 "source_id": row.get("source_id"), "reasons": list(reasons)})
    active = {row.get("source_family") for row in prepared}
    unresolved = sorted(index.unresolved_families & active)
    uncertain = any(reason.startswith("unresolved_") or reason in {"invalid_provenance", "unknown_source_family"} for reason in counts)
    incomplete = bool(missing or problems or unresolved or uncertain or not protected or not prepared)
    return {"status": "partial" if incomplete else "overlap" if counts else "metadata_clear",
            "metadata_clear": not incomplete and not counts,
            "prepared_recordings_checked": len(prepared), "verified_benchmark_audio_checked": len(protected),
            "benchmark_manifests": [str(path) for path in benchmark_manifests],
            "missing_manifests": missing, "problem_counts": dict(Counter(problems)),
            "unresolved_shared_families": unresolved, "overlap_reason_counts": dict(counts),
            "overlap_examples": examples,
            "scope": "original-file SHA256, source/session and known absolute-time metadata only",
            "limitations": "No full audio reads, artifact rehash, or cropped/cross-codec acoustic fingerprint proof."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--benchmark-manifests", nargs="+", required=True)
    parser.add_argument("--catalog-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_manifests(args.prepared_manifest, args.benchmark_manifests, args.catalog_dir)
    write_json(args.output, result)
    return int(not result["metadata_clear"])


if __name__ == "__main__":
    raise SystemExit(main())
