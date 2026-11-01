"""Conservative source/session exclusions for SSL manifests.

Manifests must describe original recordings, not only derived five-second clips.
Time intervals are absolute, timezone-aware, and half-open [start, end).
Hashes detect identical files/decoded PCM only; they do not identify arbitrary
cropped, resampled, or compressed versions of the same recording.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import PurePosixPath
import re
from typing import Iterable, Mapping
from urllib.parse import unquote, urlsplit

Row = Mapping[str, object]
_ALIASES = {
    "dori_onc": "onc", "onc_benchmark": "onc", "dori_onc_benchmark": "onc",
    "dori_orcasound": "orcasound", "orcasound_hackathon2024": "orcasound",
    "orcasound_hackathon_2024": "orcasound", "pod_cast": "orcasound",
    "mbari_pacific_sound": "pacific_sound", "pacific_sound_16khz": "pacific_sound",
    "noaa_sanctsound": "sanctsound",
}
_ONC_NAME = re.compile(
    r"(?:onc_)?(iclisten[a-z0-9-]+)_(\d{8})t\d{6}(?:\.\d+)?z$", re.I
)
_HASH = re.compile(r"[0-9a-f]{64}$")


def canonical_family(value: object) -> str:
    """Map known publications of the same archive to a shared namespace."""
    name = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(name, name)


def canonical_source_id(value: object) -> str:
    """Remove transport-only URL components while retaining the source namespace."""
    raw = str(value or "").strip().replace("\\", "/")
    parsed = urlsplit(raw)
    if parsed.scheme in {"http", "https"}:
        path = unquote(parsed.path)
        # A revision change is not a new source recording. Preserve the repo,
        # rather than matching arbitrary basenames from different HF datasets.
        if (parsed.hostname or "").lower() == "huggingface.co":
            parts = path.strip("/").split("/")
            if len(parts) >= 6 and parts[0] == "datasets" and parts[3] in {"resolve", "blob"}:
                return "hf://" + "/".join(parts[1:3] + parts[5:])
        return "https://" + parsed.netloc.lower() + path
    if parsed.scheme == "s3":
        return "s3://" + parsed.netloc.lower() + unquote(parsed.path)
    return raw


def _text(row: Row, field: str) -> str:
    return str(row.get(field) or "").strip()


def _onc_identity(source: str) -> tuple[str, str] | None:
    stem = PurePosixPath(source).stem
    match = _ONC_NAME.fullmatch(stem)
    if not match:
        return None
    # Only recognizable original ONC names cross publication boundaries.
    recording = re.sub(r"^onc_", "", stem, flags=re.I).lower()
    sensor, date = match.groups()
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return None
    return recording, f"onc_day:{sensor.lower()}:{date}"


def _interval(row: Row) -> tuple[float, float] | None:
    start, end = (_text(row, field) for field in ("recording_start_utc", "recording_end_utc"))
    if not start and not end:
        return None
    if not start or not end:
        raise ValueError("both recording time endpoints are required")
    times = []
    for value in (start, end):
        time = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if time.tzinfo is None or time.utcoffset() is None:
            raise ValueError("recording times must have a UTC offset")
        times.append(time.timestamp())
    if times[0] >= times[1]:
        raise ValueError("recording start must precede end")
    return times[0], times[1]


def _hashes(row: Row) -> set[tuple[str, str]]:
    values = set()
    for field in ("sha256", "pcm_sha256"):
        value = _text(row, field).lower()
        if value:
            if not _HASH.fullmatch(value):
                raise ValueError(f"invalid {field}")
            values.add((field, value))
    return values


def _identities(row: Row, family: str) -> tuple[set[tuple[str, str]], bool]:
    keys = set()
    source = canonical_source_id(row.get("source_id"))
    if source:
        keys.add(("source_id", source))
    for field in ("session_id", "deployment_id", "hydrophone_id"):
        if _text(row, field):
            keys.add((field, _text(row, field)))
    onc = _onc_identity(source) if family == "onc" else None
    if onc:
        keys.update({("onc_recording", onc[0]), ("session_id", onc[1])})
        keys.add(("hydrophone_id", onc[1].split(":")[1]))
    interval = _interval(row)
    sufficient = any(kind in {"session_id", "deployment_id"} for kind, _ in keys)
    sufficient |= interval is not None and any(kind == "hydrophone_id" for kind, _ in keys)
    sufficient = bool(source) and sufficient
    return keys, sufficient


class ExclusionIndex:
    """Index all protected downstream sources before selecting SSL recordings.

    A protected deployment excludes that entire deployment; omit deployment_id
    only if deployment-level holdout is not part of the intended protocol.
    Shared families without session/deployment or sensor/time provenance are
    quarantined. ONC timestamps support a conservative sensor/day session key.
    Identifiers supplied by the caller must already refer to original sources.
    """

    def __init__(self, protected_rows: Iterable[Row], *, at_risk_families: Iterable[str] = ()):
        self.at_risk_families = {canonical_family(f) for f in at_risk_families}
        self.unresolved_families: set[str] = set()
        self._exact: set[tuple[str, str, str]] = set()
        self._hashes: set[tuple[str, str]] = set()
        self._intervals: dict[tuple[str, str, str], tuple[list[float], list[float]]] = {}
        intervals = defaultdict(list)
        present_families = set()
        self.protected_count = 0
        for row in protected_rows:
            family = canonical_family(row.get("source_family"))
            if not family:
                raise ValueError("protected rows require source_family")
            self.protected_count += 1
            present_families.add(family)
            self.at_risk_families.add(family)
            self._hashes.update(_hashes(row))
            keys, sufficient = _identities(row, family)
            if not sufficient:
                self.unresolved_families.add(family)
            self._exact.update((family, kind, value) for kind, value in keys if kind != "hydrophone_id")
            interval = _interval(row)
            if interval:
                for kind, value in keys:
                    if kind in {"session_id", "deployment_id", "hydrophone_id"}:
                        intervals[(family, kind, value)].append(interval)
        # A missing shared-family test manifest is not evidence of no overlap.
        self.unresolved_families.update(self.at_risk_families - present_families)
        # Merge once, then query each identity in O(log n), not against every
        # protected segment. Adjacent intervals merge without changing overlap.
        for key, spans in intervals.items():
            merged: list[list[float]] = []
            for start, end in sorted(spans):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(end, merged[-1][1])
                else:
                    merged.append([start, end])
            self._intervals[key] = ([p[0] for p in merged], [p[1] for p in merged])

    def reasons(self, row: Row) -> tuple[str, ...]:
        """Return stable exclusion codes; an empty tuple means metadata cleared."""
        family = canonical_family(row.get("source_family"))
        reasons = set()
        if not family:
            reasons.add("unknown_source_family")
        try:
            hashes = _hashes(row)
            keys, sufficient = _identities(row, family)
            interval = _interval(row)
        except (ValueError, OverflowError):
            return tuple(sorted(reasons | {"invalid_provenance"}))
        for kind, value in hashes & self._hashes:
            reasons.add(f"identical_{kind}")
        for kind, value in keys:
            if (family, kind, value) in self._exact:
                reasons.add(f"shared_{kind}")
            if interval and (family, kind, value) in self._intervals:
                starts, ends = self._intervals[(family, kind, value)]
                previous = bisect_left(starts, interval[1]) - 1
                if previous >= 0 and ends[previous] > interval[0]:
                    reasons.add("overlapping_recording_time")
        if family in self.at_risk_families:
            if family in self.unresolved_families:
                reasons.add("unresolved_protected_provenance")
            if not sufficient:
                reasons.add("unresolved_pretrain_provenance")
        return tuple(sorted(reasons))


def filter_pretrain(
    rows: Iterable[Row], protected: Iterable[Row] | ExclusionIndex, *, at_risk_families: Iterable[str] = ()
) -> tuple[list[dict], list[dict]]:
    """Partition source rows; keep rejected rows and auditable reason codes."""
    index = protected if isinstance(protected, ExclusionIndex) else ExclusionIndex(
        protected, at_risk_families=at_risk_families
    )
    accepted, rejected = [], []
    for row in rows:
        reasons = index.reasons(row)
        if reasons:
            rejected.append({**row, "exclusion_reasons": list(reasons)})
        else:
            accepted.append(dict(row))
    return accepted, rejected


def audit(
    rows: Iterable[Row], protected: Iterable[Row] | ExclusionIndex, *, at_risk_families: Iterable[str] = ()
) -> dict:
    """Summarize metadata exclusions; this is not an acoustic near-duplicate audit."""
    index = protected if isinstance(protected, ExclusionIndex) else ExclusionIndex(
        protected, at_risk_families=at_risk_families
    )
    counts = Counter()
    total = rejected = 0
    for row in rows:
        total += 1
        reasons = index.reasons(row)
        if reasons:
            rejected += 1
            counts.update(reasons)
    return {
        "total": total, "accepted": total - rejected, "excluded": rejected,
        "protected_rows": index.protected_count, "reasons": dict(sorted(counts.items())),
        "at_risk_families": sorted(index.at_risk_families),
        "unresolved_protected_families": sorted(index.unresolved_families),
        "scope": "original-source/session/deployment/time metadata and exact hashes",
    }
