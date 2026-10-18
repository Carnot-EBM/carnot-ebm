"""Build a pinned, source-bearing EnokiQA calibration corpus.

The published labels are machine annotations. They are useful for a bounded
study, but they are not exact truth. Predictor records therefore contain only
the text available to a future scorer. Annotation fields stay in a separate
evaluator view.

Spec refs: REQ-AUTO-7410 and SCENARIO-AUTO-7410-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.request import urlopen

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
Fetcher = Callable[[str, float, int], bytes]
Decoder = Callable[[Path], Iterable[Mapping[str, Any]]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.650"
EXPERIMENT_ID = "exp7410-source-corpus"
SCHEMA = "carnot.exp7410.v650.source_corpus.v1"
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
RESULT_PATH = Path("results/experiment_7410_v650_source_corpus.json")
RAW_DIR = Path("results/raw/experiment_7410_v650_source_corpus")
MODULE_PATH = Path("python/carnot/experiment_7410_v650_source_corpus.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7410_v650_source_corpus.py")
TEST_PATH = Path("tests/python/test_experiment_7410_v650_source_corpus.py")
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "CARNOT_EXP7410_CACHE",
        str(Path.home() / ".cache/carnot/experiment_7410_v650_source_corpus"),
    )
)

ENOKIQA_REPO = "s-nlp/EnokiQA"
ENOKIQA_REVISION = "06638fd6fa5c599f3249e27d1cb489b9bd584411"
DEV_PARQUET = "data/dev-00000-of-00001.parquet"
TEST_PARQUET = "data/test-00000-of-00001.parquet"
ASSET_PATHS = ("README.md", DEV_PARQUET, TEST_PARQUET)
EXPECTED_ASSETS: dict[str, JsonDict] = {
    "README.md": {
        "size_bytes": 11_122,
        "sha256": "sha256:e1c239fbb1cded7b2c6565cb21acf8a2ff7f5eae28b91216afe4a83542005db6",
    },
    DEV_PARQUET: {
        "size_bytes": 104_366_673,
        "sha256": "sha256:8fabcee42c36872be84fe8c7dd9ea056d89bdfc371ff3689d7e03f602edb2a39",
    },
    TEST_PARQUET: {
        "size_bytes": 102_499_321,
        "sha256": "sha256:8b0ebcf26596103dd6e4dd85d2d2a8371a57ba48cdd5a88fef28122cff076831",
    },
}
MAX_TRANSFER_BYTES = 300 * 1024 * 1024
FETCH_TIMEOUT_S = 600.0
MAX_PUBLISHED_ROWS = 3_990
PARTITION_SALT = "carnot-v650-source-1"
FINAL_TEST_TOKEN = "exp7410-trusted-evaluator"
LICENSE = "cc-by-sa-4.0"
ATTRIBUTION = (
    "EnokiQA by Elisei Rykov, Timur Ionov, Nikolay Ivanov, Maksim Savkin, "
    "Maksim Makarenko, Alexander Panchenko, Vasily Konovalov, and Julia Belikova; "
    "Wikipedia-derived content and annotations under CC BY-SA 4.0."
)
TEXT_CAPS = {"question": 1_024, "context": 4_096, "answer": 2_048, "sentence": 1_024}
INFERENCE_SUBSTRATE = "no_model_load"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7382_v648_decision_protocol.py"),
    Path("python/carnot/experiment_7396_v649_decision_diagnosis.py"),
    Path("results/experiment_7396_v649_decision_diagnosis.json"),
    Path("data/fover_corpus_v4.json"),
    SPEC_PATH,
    Path("openspec/capabilities/constraint-verification/spec.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "source_corpus_ready_score",
    "corpus_manifest_path",
    "split_manifest",
    "label_authority",
    "source_disposition_rows",
)


class SourceBlocked(RuntimeError):
    """Name an external source check that prevents corpus construction."""


class CorpusInvalid(ValueError):
    """Name a corpus shard or boundary that failed deterministic replay."""


def utc_now() -> str:
    """Return an aware UTC boundary for the current process."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print one flushed phase or long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7410] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed SHA-256 identity for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def asset_url(relative: str) -> str:
    """Build an immutable URL for one allowed dataset asset."""

    if relative not in ASSET_PATHS:
        raise ValueError(f"asset_not_allowed:{relative}")
    return f"https://huggingface.co/datasets/{ENOKIQA_REPO}/resolve/{ENOKIQA_REVISION}/{relative}"


def _http_fetch(url: str, timeout_s: float, byte_limit: int) -> bytes:  # pragma: no cover
    """Download one asset while enforcing byte and silence bounds."""

    started = time.monotonic()
    chunks: list[bytes] = []
    completed = 0
    last_report = started
    with urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - URL is a fixed HTTPS origin.
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > byte_limit:
            raise SourceBlocked(f"transfer_limit_exceeded:{url}:{declared}>{byte_limit}")
        while True:
            chunk = response.read(min(1024 * 1024, byte_limit - completed + 1))
            if not chunk:
                break
            chunks.append(chunk)
            completed += len(chunk)
            now = time.monotonic()
            if completed > byte_limit:
                raise SourceBlocked(f"transfer_limit_exceeded:{url}:{completed}>{byte_limit}")
            if now - last_report >= 60.0:
                progress(started, "fetch", "heartbeat", completed_bytes=completed, url=url)
                last_report = now
            if now - started > timeout_s:
                raise SourceBlocked(f"fetch_timeout:{url}:{timeout_s}")
    return b"".join(chunks)


def _asset_manifest_path(cache_root: Path) -> Path:
    return cache_root / ENOKIQA_REVISION / "asset_manifest.json"


def authenticate_assets(cache_root: Path, receipt: Mapping[str, Any] | None = None) -> JsonDict:
    """Rehash the three allowed files and reject revision or content drift."""

    manifest_path = _asset_manifest_path(cache_root)
    if receipt is None:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceBlocked(f"cache_manifest_unavailable:{type(exc).__name__}:{exc}") from exc
        if not isinstance(loaded, dict):
            raise SourceBlocked("cache_manifest_invalid:not_object")
        receipt = loaded
    if receipt.get("revision") != ENOKIQA_REVISION:
        raise SourceBlocked(
            f"cache_revision_mismatch:{receipt.get('revision')}!={ENOKIQA_REVISION}"
        )
    files = receipt.get("files")
    if not isinstance(files, list) or [row.get("path") for row in files] != list(ASSET_PATHS):
        raise SourceBlocked("cache_asset_paths_mismatch")
    total = 0
    checked: list[JsonDict] = []
    for row in files:
        relative = str(row["path"])
        path = cache_root / ENOKIQA_REVISION / relative
        if not path.is_file():
            raise SourceBlocked(f"cache_asset_missing:{relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        expected = EXPECTED_ASSETS[relative]
        if size != expected["size_bytes"] or digest != expected["sha256"]:
            raise SourceBlocked(
                f"cache_asset_identity_mismatch:{relative}:"
                f"{size}:{digest}!={expected['size_bytes']}:{expected['sha256']}"
            )
        total += size
        checked.append(
            {
                "path": relative,
                "cache_path": str(path),
                "url": asset_url(relative),
                "size_bytes": size,
                "sha256": digest,
            }
        )
    if total > MAX_TRANSFER_BYTES:
        raise SourceBlocked(f"transfer_limit_exceeded:total:{total}>{MAX_TRANSFER_BYTES}")
    return {
        "revision": ENOKIQA_REVISION,
        "repository": ENOKIQA_REPO,
        "files": checked,
        "total_bytes": total,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
    }


def fetch_assets(
    cache_root: Path,
    *,
    fetcher: Fetcher = _http_fetch,
    timeout_s: float = FETCH_TIMEOUT_S,
    byte_limit: int = MAX_TRANSFER_BYTES,
) -> JsonDict:
    """Fetch only the pinned card and two Parquet files into a private cache."""

    manifest_path = _asset_manifest_path(cache_root)
    if manifest_path.is_file():
        return authenticate_assets(cache_root)
    target_root = manifest_path.parent
    target_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    total = 0
    files: list[JsonDict] = []
    for completed, relative in enumerate(ASSET_PATHS, start=1):
        remaining_time = timeout_s - (time.monotonic() - started)
        remaining_bytes = byte_limit - total
        if remaining_time <= 0:
            raise SourceBlocked(f"fetch_timeout:total:{timeout_s}")
        try:
            payload = fetcher(asset_url(relative), remaining_time, remaining_bytes)
        except SourceBlocked:
            raise
        except Exception as exc:
            raise SourceBlocked(f"network_failure:{relative}:{type(exc).__name__}:{exc}") from exc
        total += len(payload)
        if total > byte_limit:
            raise SourceBlocked(f"transfer_limit_exceeded:total:{total}>{byte_limit}")
        expected = EXPECTED_ASSETS[relative]
        digest = sha256_bytes(payload)
        if len(payload) != expected["size_bytes"] or digest != expected["sha256"]:
            raise SourceBlocked(f"source_identity_mismatch:{relative}:{len(payload)}:{digest}")
        path = target_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append(
            {
                "path": relative,
                "cache_path": str(path),
                "url": asset_url(relative),
                "size_bytes": len(payload),
                "sha256": digest,
            }
        )
        progress(started, "fetch", "asset_complete", completed=completed, total=len(ASSET_PATHS))
    receipt = {
        "revision": ENOKIQA_REVISION,
        "repository": ENOKIQA_REPO,
        "files": files,
        "total_bytes": total,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
    }
    atomic_json(manifest_path, receipt)
    return authenticate_assets(cache_root, receipt)


def _parquet_rows(path: Path) -> Iterable[Mapping[str, Any]]:  # pragma: no cover
    """Yield small decoded batches so the nested annotation table stays bounded."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = ("id", "title", "question", "answer", "context", "sentences")
    for batch in parquet.iter_batches(batch_size=16, columns=list(columns)):
        yield from batch.to_pylist()


def decode_rows(
    asset_receipt: Mapping[str, Any], *, decoder: Decoder = _parquet_rows
) -> list[JsonDict]:
    """Decode exactly the two published splits without loading dataset code."""

    paths = {str(row["path"]): Path(str(row["cache_path"])) for row in asset_receipt["files"]}
    rows: list[JsonDict] = []
    try:
        for split, relative in (("dev", DEV_PARQUET), ("test", TEST_PARQUET)):
            for index, value in enumerate(decoder(paths[relative])):
                row = dict(value)
                row["_official_split"] = split
                row["_source_index"] = index
                rows.append(row)
                if len(rows) > MAX_PUBLISHED_ROWS:
                    raise SourceBlocked(f"published_row_limit:{len(rows)}>{MAX_PUBLISHED_ROWS}")
                if len(rows) % 250 == 0:
                    print(
                        f"[exp7410] phase=decode event=progress completed_rows={len(rows)}",
                        flush=True,
                    )
    except SourceBlocked:
        raise
    except Exception as exc:
        raise SourceBlocked(f"decoder_unavailable:{type(exc).__name__}:{exc}") from exc
    return rows


def normalize_text(value: Any) -> str:
    """Normalize representation without using annotation values."""

    return " ".join(str(value or "").casefold().split())


def _stable_hash(*parts: Any) -> str:
    return canonical_hash(list(parts))


def build_connected_groups(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Connect examples through source text identities before consulting labels."""

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    observed: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        context = str(row.get("context") or "")
        keys = (
            ("title", normalize_text(row.get("title"))),
            ("context", sha256_bytes(context.encode("utf-8")) if context else ""),
            ("question", normalize_text(row.get("question"))),
            ("answer", normalize_text(row.get("answer"))),
        )
        for key in keys:
            if not key[1]:
                continue
            if key in observed:
                union(index, observed[key])
            else:
                observed[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    group_ids = {
        member: _stable_hash(
            "source-group",
            [
                [
                    rows[index].get("_official_split"),
                    rows[index].get("_source_index"),
                    rows[index].get("id"),
                ]
                for index in members
            ],
        )
        for members in components.values()
        for member in members
    }
    return [{"row_index": index, "group_id": group_ids[index]} for index in range(len(rows))]


def assign_source_dispositions(
    rows: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Keep test rows final and exclude every connected development overlap."""

    group_by_index = {int(row["row_index"]): str(row["group_id"]) for row in groups}
    test_groups = {
        group_by_index[index]
        for index, row in enumerate(rows)
        if row.get("_official_split") == "test"
    }
    dispositions: list[JsonDict] = []
    for index, row in enumerate(rows):
        group_id = group_by_index[index]
        split = str(row.get("_official_split"))
        disposition = (
            "official_test"
            if split == "test"
            else "excluded_test_overlap"
            if group_id in test_groups
            else "eligible_dev"
        )
        dispositions.append(
            {
                "row_index": index,
                "source_id": str(row.get("id") or ""),
                "official_split": split,
                "group_id": group_id,
                "disposition": disposition,
            }
        )
    return dispositions


def _development_partition(group_id: str) -> str:
    bucket = int(_stable_hash(PARTITION_SALT, group_id).split(":", 1)[1][:16], 16) % 100
    if bucket < 40:
        return "train"
    if bucket < 55:
        return "probability_calibration"
    if bucket < 70:
        return "policy_calibration"
    return "online_stream"


def assign_partitions(dispositions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Assign whole development groups with one frozen salted hash."""

    group_dispositions: dict[str, set[str]] = defaultdict(set)
    for row in dispositions:
        group_dispositions[str(row["group_id"])].add(str(row["disposition"]))
    allowed = (
        {"eligible_dev"},
        {"official_test"},
        {"official_test", "excluded_test_overlap"},
    )
    for group_id, values in group_dispositions.items():
        if values not in allowed:
            raise CorpusInvalid(f"group_partition_conflict:{group_id}:{sorted(values)}")

    def role(row: Mapping[str, Any]) -> str:
        disposition = row["disposition"]
        if disposition == "eligible_dev":
            return _development_partition(str(row["group_id"]))
        if disposition == "official_test":
            return "final_test"
        return "excluded_test_overlap"

    return [
        {
            "row_index": int(row["row_index"]),
            "group_id": str(row["group_id"]),
            "partition": role(row),
        }
        for row in dispositions
    ]


def select_sentence(row: Mapping[str, Any]) -> JsonDict | None:
    """Select one annotation by answer identity and sentence index only."""

    sentences = row.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        return None
    answer_id = str(row.get("id") or "")
    candidates = [dict(value) for value in sentences if isinstance(value, Mapping)]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda value: _stable_hash(answer_id, value.get("sentence_index")),
    )


def _annotation_reason(row: Mapping[str, Any], sentence: Mapping[str, Any] | None) -> str | None:
    if sentence is None:
        return "empty_annotation"
    text = str(sentence.get("sentence") or "")
    answer = str(row.get("answer") or "")
    if not text:
        return "empty_annotation"
    if normalize_text(text) not in normalize_text(answer):
        return "sentence_unmatched"
    probability = sentence.get("hall_prob")
    if (
        not isinstance(probability, (int, float))
        or isinstance(probability, bool)
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        return "invalid_machine_label"
    triples = sentence.get("triples")
    if not isinstance(triples, list) or not triples:
        return "empty_annotation"
    for triple in triples:
        if not isinstance(triple, Mapping):
            return "malformed_span"
        span = triple.get("span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in span)
            or span[0] < 0
            or span[1] <= span[0]
            or span[1] > len(answer)
        ):
            return "malformed_span"
    return None


def _cap(value: Any, field: str) -> str:
    return str(value or "")[: TEXT_CAPS[field]]


def project_row(row: Mapping[str, Any], partition: str, group_id: str) -> JsonDict:
    """Separate predictor text from machine annotation authority."""

    selected = select_sentence(row)
    reason = _annotation_reason(row, selected)
    sentence_text = None if reason else _cap(selected["sentence"], "sentence")
    probability = None if reason else float(selected["hall_prob"])
    row_key = _stable_hash(
        "source-row",
        row.get("_official_split"),
        row.get("_source_index"),
        row.get("id"),
    )
    return {
        "row_key": row_key,
        "group_id": group_id,
        "partition": partition,
        "predictor_view": {
            "question": _cap(row.get("question"), "question"),
            "context": _cap(row.get("context"), "context"),
            "answer": _cap(row.get("answer"), "answer"),
            "sentence": sentence_text,
        },
        "evaluator_view": {
            "row_key": row_key,
            "label": None if probability is None else int(probability > 0.5),
            "label_authority": "machine_annotation",
        },
        "selected_sentence_index": None if selected is None else selected.get("sentence_index"),
        "disposition_reason": reason or "scored_machine_annotation",
    }


def _write_shard(path: Path, kind: str, records: Sequence[Mapping[str, Any]]) -> JsonDict:
    payload = {
        "schema": f"carnot.exp7410.{kind}_shard.v1",
        "label_authority": "machine_annotation",
        "attribution": ATTRIBUTION,
        "records": [deepcopy(dict(row)) for row in records],
    }
    atomic_json(path, payload)
    return {
        "path": path.name,
        "kind": kind,
        "row_count": len(records),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "label_authority": "machine_annotation",
        "attribution": ATTRIBUTION,
    }


def _manifest_hash(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("_runtime_raw_dir", None)
    payload["manifest_hash"] = ""
    return canonical_hash(payload)


def write_corpus(
    raw_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    *,
    shard_size: int = 250,
) -> JsonDict:
    """Write capped predictor and evaluator shards plus a compact manifest."""

    if shard_size <= 0:
        raise ValueError("shard_size_must_be_positive")
    raw_dir.mkdir(parents=True, exist_ok=True)
    group_by_index = {int(row["row_index"]): str(row["group_id"]) for row in groups}
    role_by_index = {int(row["row_index"]): str(row["partition"]) for row in memberships}
    projected = [
        project_row(row, role_by_index[index], group_by_index[index])
        for index, row in enumerate(rows)
    ]
    shards: list[JsonDict] = []
    for kind in ("predictor", "evaluator"):
        records = [
            {
                "row_key": row["row_key"],
                "group_id": row["group_id"],
                "partition": row["partition"],
                **(row["predictor_view"] if kind == "predictor" else row["evaluator_view"]),
            }
            for row in projected
        ]
        for start in range(0, len(records), shard_size):
            ordinal = start // shard_size
            path = raw_dir / f"{kind}-{ordinal:03d}.json"
            shards.append(_write_shard(path, kind, records[start : start + shard_size]))
    disposition_rows = [
        {
            "row_key": row["row_key"],
            "group_id": row["group_id"],
            "partition": row["partition"],
            "selected_sentence_index": row["selected_sentence_index"],
            "annotation_disposition": row["disposition_reason"],
        }
        for row in projected
    ]
    manifest: JsonDict = {
        "schema": "carnot.exp7410.source_corpus_manifest.v1",
        "source_revision": ENOKIQA_REVISION,
        "partition_salt": PARTITION_SALT,
        "label_authority": "machine_annotation",
        "attribution": ATTRIBUTION,
        "text_caps": dict(TEXT_CAPS),
        "source_row_count": len(rows),
        "shards": shards,
        "split_membership": [deepcopy(dict(row)) for row in memberships],
        "source_disposition_rows": disposition_rows,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    atomic_json(raw_dir / "corpus_manifest.json", manifest)
    manifest["_runtime_raw_dir"] = str(raw_dir)
    return manifest


def reload_corpus(raw_dir: Path, manifest: Mapping[str, Any] | None = None) -> JsonDict:
    """Rehash every shard and reconstruct the two views in a fresh-reader shape."""

    if manifest is None:
        try:
            value = json.loads((raw_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusInvalid(f"manifest_unreadable:{type(exc).__name__}:{exc}") from exc
        if not isinstance(value, dict):
            raise CorpusInvalid("manifest_not_object")
        manifest = value
    if manifest.get("manifest_hash") != _manifest_hash(manifest):
        raise CorpusInvalid("manifest_hash_mismatch")
    views: dict[str, list[JsonDict]] = {"predictor": [], "evaluator": []}
    for shard in manifest.get("shards", []):
        path = raw_dir / str(shard["path"])
        if not path.is_file() or sha256_file(path) != shard.get("sha256"):
            raise CorpusInvalid(f"shard_hash_mismatch:{shard.get('path')}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CorpusInvalid(f"shard_unreadable:{path.name}:{exc}") from exc
        kind = str(shard.get("kind"))
        if (
            kind not in views
            or payload.get("label_authority") != "machine_annotation"
            or payload.get("attribution") != ATTRIBUTION
        ):
            raise CorpusInvalid(f"shard_metadata_invalid:{path.name}")
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != shard.get("row_count"):
            raise CorpusInvalid(f"shard_row_count_mismatch:{path.name}")
        views[kind].extend(dict(row) for row in records)
    predictor_keys = [row.get("row_key") for row in views["predictor"]]
    evaluator_keys = [row.get("row_key") for row in views["evaluator"]]
    if (
        predictor_keys != evaluator_keys
        or len(set(predictor_keys)) != len(predictor_keys)
        or len(predictor_keys) != manifest.get("source_row_count")
    ):
        raise CorpusInvalid("view_identity_mismatch")
    group_roles: dict[str, set[str]] = defaultdict(set)
    for row in views["predictor"]:
        group_roles[str(row["group_id"])].add(str(row["partition"]))
        if "label" in row:
            raise CorpusInvalid("predictor_teacher_field")
    if any(
        len(roles) != 1 and roles != {"final_test", "excluded_test_overlap"}
        for roles in group_roles.values()
    ):
        raise CorpusInvalid("group_partition_overlap")
    return {
        "manifest_hash": manifest["manifest_hash"],
        "predictor_row_count": len(views["predictor"]),
        "evaluator_row_count": len(views["evaluator"]),
        "predictor_rows": views["predictor"],
        "evaluator_rows": views["evaluator"],
    }


class CorpusReaders:
    """Expose predictor text while keeping final labels behind one explicit token."""

    def __init__(self, reloaded: Mapping[str, Any]) -> None:
        self._predictors = [dict(row) for row in reloaded["predictor_rows"]]
        self._labels = [dict(row) for row in reloaded["evaluator_rows"]]

    def read_predictors(self, partition: str) -> list[JsonDict]:
        return [deepcopy(row) for row in self._predictors if row["partition"] == partition]

    def read_labels(self, partition: str, *, token: str | None = None) -> list[JsonDict]:
        if partition == "final_test" and token != FINAL_TEST_TOKEN:
            raise PermissionError("sealed_final_test_labels")
        return [deepcopy(row) for row in self._labels if row["partition"] == partition]


def _partition_summary(reloaded: Mapping[str, Any]) -> list[JsonDict]:
    predictors = reloaded["predictor_rows"]
    labels = {row["row_key"]: row["label"] for row in reloaded["evaluator_rows"]}
    partitions = (
        "train",
        "probability_calibration",
        "policy_calibration",
        "online_stream",
        "final_test",
        "excluded_test_overlap",
    )
    summary = []
    for partition in partitions:
        selected = [row for row in predictors if row["partition"] == partition]
        scored = [labels[row["row_key"]] for row in selected if labels[row["row_key"]] is not None]
        summary.append(
            {
                "partition": partition,
                "group_count": len({row["group_id"] for row in selected}),
                "row_count": len(selected),
                "scored_count": len(scored),
                "label_counts": {
                    "0": sum(label == 0 for label in scored),
                    "1": sum(label == 1 for label in scored),
                    "unscored": len(selected) - len(scored),
                },
            }
        )
    return summary


def _source_hashes(root: Path, assets: Mapping[str, Any] | None = None) -> JsonDict:
    hashes: JsonDict = {}
    for relative in (*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH):
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": None,
            }
    for row in (assets or {}).get("files", []):
        hashes[f"enokiqa:{row['path']}"] = {
            "path": row["cache_path"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "revision": ENOKIQA_REVISION,
            "url": row["url"],
            "original_flagged_adversarial": None,
        }
    return hashes


def _field_principles() -> dict[str, str]:
    defaults = {
        key: "Keep this ordinary field machine-readable and independently replayable."
        for key in REQUIRED_ARTIFACT_FIELDS
    }
    defaults.update(
        {
            "schema": "Version ordinary top-level fields with experiment identity and terminal status.",
            "run_date": "Use 20260919 and retain actual UTC start and end boundaries.",
            "preconditions_checked": "Record exact paths, hashes, and resource checks before dependent work.",
            "MODEL_SPECS": "Use an empty list because this experiment makes no current LLM call.",
            "model_invoked": "Set true only for an actual current attempted LLM operation.",
            "invocation_counts": "Count current owned attempts and terminal states from events.",
            "inference_substrate": "Use a string and keep device details in a separate field.",
            "inference_substrate_class": "Declare no_model_load without padding work.",
            "execution_venue": "Use the closed host value and keep device details separate.",
            "duration_s": "Measure current monotonic work apart from historical and validation time.",
            "phase_spans": "Retain real phase times, checkpoints, and completed units.",
            "random_seed": "Use null because salted hashes replace random sampling.",
            "reproducibility_checksum": "Bind code, protocol, source bytes, and emitted raw rows.",
            "source_artifact_hashes": "Hash exact source paths and preserve original flags.",
            "rows": "Retain every partition condition, including empty and unsupported conditions.",
            "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted rows.",
            "acceptance_gate_results": "Keep category, operator, operands, outcome, and principle separate.",
            "gate_check_summary": "Name each blocked upstream, path, check, field, expected, and observed value.",
            "verifier_is_oracle": "Machine annotations do not make the deployed verifier an oracle.",
            "honest_verdict": "Use complete_ for finished findings and blocked_ for unavailable prerequisites.",
            "verdict_class": "Use the closed terminal class without treating corpus readiness as benefit.",
            "flagged_adversarial": "Preserve every critical finding and deny readiness when flagged.",
            "validation_receipts": "Retain exact argv, environment, exits, durations, and hashed logs.",
            "field_principles": "Explain fields separately while gate scalars remain ordinary values.",
            "promotion_score": "Remain zero because this result changes no production or publication state.",
            "source_corpus_ready_score": "One means authenticated data and sealed reader boundaries, not truth.",
            "corpus_manifest_path": "Name the exact compact manifest consumed by downstream readers.",
            "split_manifest": "Freeze group IDs and roles while keeping labels separate.",
            "label_authority": "Declare machine_annotation and make no exact-truth claim.",
            "source_disposition_rows": "Account for every input, grouping decision, and unsupported annotation.",
        }
    )
    return defaults


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    required = [row for row in gates if row["category"] != "scientific_support"]
    failures = [dict(row) for row in required if row.get("passed") is not True]
    support = [dict(row) for row in gates if row["category"] == "scientific_support"]
    return {
        "required_checks_passed": not failures,
        "failed_required_checks": failures,
        "support_limited": any(row.get("passed") is not True for row in support),
        "blocked_revision": None,
        "blocked_path": None,
        "blocked_check": None,
        "blocked_field": None,
        "blocked_expected": None,
        "blocked_observed": None,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def _source_disposition_rows(
    dispositions: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[JsonDict]:
    annotation = {row["row_key"]: row for row in manifest.get("source_disposition_rows", [])}
    memberships = {int(row["row_index"]): row for row in manifest["split_membership"]}
    result = []
    for row in dispositions:
        member = memberships[int(row["row_index"])]
        key = manifest["source_disposition_rows"][int(row["row_index"])]["row_key"]
        result.append(
            {
                **deepcopy(dict(row)),
                "partition": member["partition"],
                "row_key": key,
                "selected_sentence_index": annotation[key]["selected_sentence_index"],
                "annotation_disposition": annotation[key]["annotation_disposition"],
            }
        )
    return result


def build_artifact(
    *,
    asset_receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    corpus_manifest: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_ns: int,
    ended_ns: int,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one schema-complete result from raw corpus and validation evidence."""

    raw_dir = Path(str(corpus_manifest.get("_runtime_raw_dir") or REPO_ROOT / RAW_DIR))
    reloaded = reload_corpus(raw_dir, corpus_manifest) if rows else None
    if reloaded is None:
        raise CorpusInvalid("ready_artifact_requires_rows")
    dispositions = assign_source_dispositions(rows, groups)
    source_rows = _source_disposition_rows(dispositions, corpus_manifest)
    partition_rows = _partition_summary(reloaded)
    by_partition = {row["partition"]: row for row in partition_rows}
    required_names = set(validation_scope.REQUIRED_CHECK_NAMES)
    received = {
        row.get("name")
        for row in validation_receipts
        if row.get("passed") is True and row.get("exit_code") == 0
    }
    validation_passed = required_names.issubset(received)
    gates = [
        _gate(
            "asset_revision",
            "validity",
            "==",
            ENOKIQA_REVISION,
            asset_receipt.get("revision"),
            asset_receipt.get("revision") == ENOKIQA_REVISION,
            "Only pinned source bytes are eligible.",
        ),
        _gate(
            "published_row_bound",
            "validity",
            "<=",
            MAX_PUBLISHED_ROWS,
            len(rows),
            len(rows) <= MAX_PUBLISHED_ROWS,
            "The new corpus cannot exceed the published release.",
        ),
        _gate(
            "deterministic_reload",
            "safety",
            "==",
            corpus_manifest["manifest_hash"],
            reloaded["manifest_hash"],
            corpus_manifest["manifest_hash"] == reloaded["manifest_hash"],
            "Fresh reload must reproduce exact bytes and membership.",
        ),
        _gate(
            "affected_validation",
            "validation",
            "contains",
            sorted(required_names),
            sorted(received),
            validation_passed,
            "All frozen affected checks must pass.",
        ),
        _gate(
            "probability_calibration_groups",
            "scientific_support",
            ">=",
            20,
            by_partition["probability_calibration"]["group_count"],
            by_partition["probability_calibration"]["group_count"] >= 20,
            "Small calibration support limits later inference but not corpus validity.",
        ),
        _gate(
            "policy_calibration_groups",
            "scientific_support",
            ">=",
            20,
            by_partition["policy_calibration"]["group_count"],
            by_partition["policy_calibration"]["group_count"] >= 20,
            "Small calibration support limits later inference but not corpus validity.",
        ),
        _gate(
            "online_stream_groups",
            "scientific_support",
            ">=",
            80,
            by_partition["online_stream"]["group_count"],
            by_partition["online_stream"]["group_count"] >= 80,
            "Small online support limits later inference but not corpus validity.",
        ),
    ]
    summary = _gate_summary(gates)
    ready = int(summary["required_checks_passed"] and not flagged_adversarial)
    invocation = build_current_work_receipt(
        run_id="exp7410-current",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "work": "host CPU Parquet decode, hash grouping, and JSON serialization",
            "enoki_installed": False,
            "neural_model_loaded": False,
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=phase_spans,
        small_ebm_training={"performed": False, "receipts": []},
    )
    group_roles = sorted({(str(row["group_id"]), str(row["partition"])) for row in memberships})
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete" if ready else "disqualified",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **invocation,
        "random_seed": None,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": partition_rows,
        "sample_size_budget": {
            "planned": MAX_PUBLISHED_ROWS,
            "attempted": len(rows),
            "completed": len(source_rows),
            "failed": 0,
            "censored": sum(
                row["annotation_disposition"] != "scored_machine_annotation" for row in source_rows
            ),
            "unstarted": MAX_PUBLISHED_ROWS - len(rows),
            "independent_groups": len({group_id for group_id, _role in group_roles}),
            "stop_rule": "read at most the published 3990 rows; never resplit for class balance",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "honest_verdict": "complete_null_source_corpus_ready_machine_annotations_not_truth"
        if ready
        else "complete_disqualified_source_corpus_contract",
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "source_corpus_ready_score": ready,
        "corpus_manifest_path": (RAW_DIR / "corpus_manifest.json").as_posix(),
        "corpus_manifest_hash": corpus_manifest["manifest_hash"],
        "split_manifest": {
            "salt": PARTITION_SALT,
            "group_memberships": [
                {"group_id": group_id, "partition": role} for group_id, role in group_roles
            ],
            "labels_in_manifest": False,
        },
        "label_authority": "machine_annotation",
        "source_disposition_rows": source_rows,
        "asset_receipt": deepcopy(dict(asset_receipt)),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(
    *,
    asset_receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    corpus_manifest: Mapping[str, Any],
) -> JsonDict:
    """Build a deterministic ready fixture without launching child commands."""

    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in validation_scope.REQUIRED_CHECK_NAMES
    ]
    return build_artifact(
        asset_receipt=asset_receipt,
        rows=rows,
        groups=groups,
        memberships=memberships,
        corpus_manifest=corpus_manifest,
        validation_receipts=receipts,
        preconditions=[{"check": "fixture", "passed": True}],
        source_hashes={},
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:00+00:00",
        started_ns=0,
        ended_ns=0,
    )


def build_blocked_artifact(
    *, check: str, upstream: str, field: str, expected: Any, observed: Any
) -> JsonDict:
    """Preserve an unavailable external source as a schema-complete block."""

    blocked = {
        "upstream": upstream,
        "path": None,
        "check": check,
        "field": field,
        "expected": expected,
        "observed": observed,
    }
    started = time.monotonic_ns()
    invocation = build_current_work_receipt(
        run_id="exp7410-blocked",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={"work": "precondition checks only"},
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started,
        ended_monotonic_ns=started,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": utc_now(),
        "completed_at_utc": utc_now(),
        "preconditions_checked": [{**blocked, "passed": False}],
        **invocation,
        "random_seed": None,
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": MAX_PUBLISHED_ROWS,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": MAX_PUBLISHED_ROWS,
            "independent_groups": 0,
            "stop_rule": "blocked before source read",
        },
        "acceptance_gate_results": [
            _gate(
                check,
                "precondition",
                "==",
                expected,
                observed,
                False,
                "Unavailable pinned source bytes cannot promote science.",
            )
        ],
        "gate_check_summary": {
            "required_checks_passed": False,
            "failed_required_checks": [{**blocked, "passed": False}],
            "support_limited": False,
            "blocked_revision": blocked,
            "blocked_path": blocked,
            "blocked_check": blocked,
            "blocked_field": blocked,
            "blocked_expected": blocked,
            "blocked_observed": blocked,
        },
        "verifier_is_oracle": False,
        "honest_verdict": f"blocked_{check}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "source_corpus_ready_score": 0,
        "corpus_manifest_path": None,
        "split_manifest": {
            "salt": PARTITION_SALT,
            "group_memberships": [],
            "labels_in_manifest": False,
        },
        "label_authority": "machine_annotation",
        "source_disposition_rows": [],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check declarations, counters, dispositions, gates, and checksum."""

    errors = [f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "label_authority": "machine_annotation",
        "promotion_score": 0,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"declaration_mismatch:{field}")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("verdict_class") == "blocked":
        if value.get("source_corpus_ready_score") != 0 or not str(
            value.get("honest_verdict", "")
        ).startswith("blocked_"):
            errors.append("blocked_disposition_invalid")
    else:
        expected_ready = int(
            value.get("gate_check_summary", {}).get("required_checks_passed") is True
            and value.get("flagged_adversarial") is False
        )
        if value.get("source_corpus_ready_score") != expected_ready:
            errors.append("source_corpus_ready_score_mismatch")
        budget = value.get("sample_size_budget") or {}
        if len(value.get("source_disposition_rows") or []) != budget.get("completed"):
            errors.append("source_disposition_count_mismatch")
        roles: dict[str, set[str]] = defaultdict(set)
        for row in value.get("split_manifest", {}).get("group_memberships", []):
            roles[str(row.get("group_id"))].add(str(row.get("partition")))
        if any(
            len(partitions) != 1 and partitions != {"final_test", "excluded_test_overlap"}
            for partitions in roles.values()
        ):
            errors.append("split_group_overlap")
    if set(value.get("field_principles") or {}) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    errors.extend(validate_current_work_receipt(value, root=REPO_ROOT))
    return list(dict.fromkeys(errors))


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Hash every declared local authority before dependent source work."""

    checks: list[JsonDict] = []
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "expected": "readable_nonempty_bytes",
                "observed": observed,
                "passed": observed == "readable_nonempty_bytes",
            }
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "path": SPEC_PATH.as_posix(),
            "field": "REQ-*",
            "expected": "REQ-AUTO-7410",
            "observed": "REQ-AUTO-7410" if "REQ-AUTO-7410" in spec_text else None,
            "passed": "REQ-AUTO-7410" in spec_text,
        }
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    quarantined = "experiment_id: 7410" in exclusion
    checks.append(
        {
            "check": "current_task_not_quarantined",
            "upstream": "ops/exclusion_manifest.yaml",
            "path": "ops/exclusion_manifest.yaml",
            "field": EXPERIMENT_ID,
            "expected": False,
            "observed": quarantined,
            "passed": not quarantined,
        }
    )
    return checks, _source_hashes(root)


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    python = ".venv/bin/python"
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--cold-replay",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def cold_replay(path: Path) -> list[str]:  # pragma: no cover
    """Reload one candidate and independently rehash its corpus shards."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"artifact_unreadable:{type(exc).__name__}:{exc}"]
    errors = validate_artifact(value)
    manifest_path = REPO_ROOT / str(value.get("corpus_manifest_path") or "")
    try:
        reloaded = reload_corpus(manifest_path.parent)
    except CorpusInvalid as exc:
        errors.append(str(exc))
    else:
        if reloaded["manifest_hash"] != value.get("corpus_manifest_hash"):
            errors.append("corpus_manifest_hash_mismatch")
        if reloaded["predictor_row_count"] != len(value.get("source_disposition_rows") or []):
            errors.append("cold_row_count_mismatch")
    return list(dict.fromkeys(errors))


def run_experiment(  # pragma: no cover - exercised by the declared entrypoint.
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> JsonDict:
    """Fetch, seal, validate, replay, and atomically publish the corpus."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "start")
    preconditions, source_hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row["passed"] is not True), None)
    if failed:
        artifact = build_blocked_artifact(
            check=str(failed["check"]),
            upstream=str(failed["upstream"]),
            field=str(failed["field"]),
            expected=failed["expected"],
            observed=failed["observed"],
        )
        atomic_json(root / output_path, artifact)
        return artifact

    phase_started = time.monotonic()
    progress(run_started, "acquisition", "before_fetch", revision=ENOKIQA_REVISION)
    try:
        assets = fetch_assets(cache_root)
        progress(run_started, "acquisition", "after_fetch", bytes=assets["total_bytes"])
        progress(run_started, "decode", "before_parquet_decode")
        rows = decode_rows(assets)
        progress(run_started, "decode", "after_parquet_decode", rows=len(rows))
    except SourceBlocked as exc:
        progress(run_started, "acquisition", "blocked", observed=str(exc))
        artifact = build_blocked_artifact(
            check="enoki_source_available",
            upstream=ENOKIQA_REPO,
            field="revision_assets_decoder",
            expected={"revision": ENOKIQA_REVISION, "rows_at_most": MAX_PUBLISHED_ROWS},
            observed=str(exc),
        )
        atomic_json(root / output_path, artifact)
        return artifact
    spans.append(_span("acquisition_and_decode", phase_started, run_started, len(rows)))
    source_hashes = _source_hashes(root, assets)

    phase_started = time.monotonic()
    progress(run_started, "seal", "start", rows=len(rows))
    groups = build_connected_groups(rows)
    dispositions = assign_source_dispositions(rows, groups)
    memberships = assign_partitions(dispositions)
    raw_dir = root / RAW_DIR
    manifest = write_corpus(raw_dir, rows, groups, memberships)
    replay = reload_corpus(raw_dir, manifest)
    spans.append(_span("seal", phase_started, run_started, replay["predictor_row_count"]))
    progress(run_started, "seal", "end", groups=len({row["group_id"] for row in groups}))

    phase_started = time.monotonic()
    private_root = Path(tempfile.mkdtemp(prefix="exp7410-validation-", dir="/tmp"))
    affected = AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(TEST_PATH.as_posix(),),
        changed_modules=(MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
    )
    commands = build_command_plan(root, affected, private_root)
    plan_errors = validate_command_plan(root, affected, commands)
    progress(
        run_started, "validation", "before_affected_subprocesses", plan_errors=len(plan_errors)
    )
    validation_receipts: list[JsonDict] = []
    if not plan_errors:
        validation_receipts = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    reduction = reduce_affected_receipts(root, affected, validation_receipts)
    spans.append(_span("affected_validation", phase_started, run_started, len(validation_receipts)))
    progress(run_started, "validation", "after_affected_subprocesses", passed=reduction["passed"])

    candidate = build_artifact(
        asset_receipt=assets,
        rows=rows,
        groups=groups,
        memberships=memberships,
        corpus_manifest=manifest,
        validation_receipts=validation_receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        flagged_adversarial=not reduction["passed"],
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(run_started, "terminal_validation", "before_subprocesses")
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    terminal_passed = all(row["passed"] for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    final = build_artifact(
        asset_receipt=assets,
        rows=rows,
        groups=groups,
        memberships=memberships,
        corpus_manifest=manifest,
        validation_receipts=[*validation_receipts, *terminal],
        preconditions=preconditions,
        source_hashes=source_hashes,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        flagged_adversarial=not reduction["passed"] or not terminal_passed or critical,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date and cold-replay mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cold-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source corpus or independently replay one candidate."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output, cache_root=args.cache_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
