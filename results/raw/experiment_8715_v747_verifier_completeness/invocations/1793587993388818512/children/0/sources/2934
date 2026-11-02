"""Seal a human-annotated RAGTruth source-support protocol.

RAGTruth labels identify response spans that annotators judged unsupported by
the supplied source. They do not establish that a statement is false in the
world. Predictor and evaluator shards therefore keep the labels behind an
explicit reader boundary.

Spec refs: REQ-AUTO-7423 and SCENARIO-AUTO-7423-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import unicodedata
from urllib.request import urlopen

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7412_v650_source_features import (
    ACCEPT_THRESHOLDS,
    MAX_STEPS,
    REJECT_THRESHOLDS,
    SOURCE_FEATURE_NAMES,
    bounded_word_tokens,
    extract_feature_row,
    feature_definitions,
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
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
EXPERIMENT_ID = "exp7423-v651-annotated-protocol"
SCHEMA = "carnot.exp7423.v651.annotated_protocol.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
RESULT_PATH = Path("results/experiment_7423_v651_annotated_protocol.json")
RAW_DIR = Path("results/raw/experiment_7423_v651_annotated_protocol")
MODULE_PATH = Path("python/carnot/experiment_7423_v651_annotated_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7423_v651_annotated_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7423_v651_annotated_protocol.py")

RAGTRUTH_REPO = "https://github.com/ParticleMedia/RAGTruth"
RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
ASSET_PATHS = ("dataset/source_info.jsonl", "dataset/response.jsonl")
EXPECTED_ASSETS: dict[str, JsonDict] = {
    "dataset/source_info.jsonl": {
        "size_bytes": 15_117_971,
        "sha256": "sha256:0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    },
    "dataset/response.jsonl": {
        "size_bytes": 21_458_735,
        "sha256": "sha256:e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    },
}
LICENSE = "MIT"
ATTRIBUTION = "RAGTruth, Particle Media; human annotations released by the authors."
MAX_TRANSFER_BYTES = 64 * 1024 * 1024
FETCH_TIMEOUT_S = 600.0
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "CARNOT_EXP7423_CACHE",
        str(Path.home() / ".cache/carnot/experiment_7423_v651_annotated_protocol"),
    )
)
LEGACY_CACHE_ROOT = Path.home() / ".cache/carnot-rewrite-20260731/work/data/ragtruth"

PARTITION_SALT = "carnot-v651-human-1"
CAPS = {
    "fit": 2_000,
    "probability_calibration": 500,
    "policy_calibration": 500,
    "prospective_stream": 1_000,
    "final_test": 1_000,
}
DEVELOPMENT_PARTITIONS = (
    "fit",
    "probability_calibration",
    "policy_calibration",
    "prospective_stream",
)
ELIGIBLE_PARTITIONS = (*DEVELOPMENT_PARTITIONS, "final_test")
TRAINING_SEEDS = (65_101, 65_102, 65_103, 65_104, 65_105)
ARMS = (
    "training_prevalence",
    "l2_logistic_six_input",
    "response_only_2_4_1_gibbs",
    "source_aware_6_4_1_gibbs",
)
ONLINE_ORDERS = ("hash_order", "reversed_block_order")
FEEDBACK_REGIMES = ("deterministic_75_percent", "frozen_baseline_selected")
FEEDBACK_DELAYS = (1, 32)
SOURCE_TOKEN_LIMIT = 4_096
RESPONSE_TOKEN_LIMIT = 1_024
EVALUATOR_TOKEN = "exp7423-human-label-evaluator"
INFERENCE_SUBSTRATE = "no_model_load"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
LABEL_AUTHORITY = "human_annotation_source_support"
MAX_SHARD_BYTES = 8 * 1024 * 1024

PREDICTOR_FIELDS = (
    "row_key",
    "group_id",
    "partition",
    "task_type",
    "source_text",
    "response_text",
    "source_token_count",
    "response_token_count",
    "source_truncated",
    "response_truncated",
    "certificate_selected",
    "features",
)
EVALUATOR_FIELDS = (
    "row_key",
    "group_id",
    "partition",
    "task_type",
    "source_id",
    "response_id",
    "source_name",
    "model",
    "temperature",
    "official_split",
    "quality",
    "annotations",
    "primary_label",
    "implicit_true_excluded_label",
    "certificate_selected",
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
    "annotated_protocol_ready_score",
    "corpus_manifest",
    "support_counts",
    "label_authority",
    "feature_contract",
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7410_v650_source_corpus.py"),
    Path("python/carnot/experiment_7412_v650_source_features.py"),
    Path("results/experiment_7413_v650_source_calibration.json"),
    Path("results/experiment_7414_v650_selected_feedback.json"),
    SPEC_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


class SourceBlocked(RuntimeError):
    """Name an external source failure that prevents a scientific audit."""


class CorpusInvalid(ValueError):
    """Name malformed release data or drift in a sealed local shard."""


def utc_now() -> str:  # pragma: no cover - real run boundary.
    """Return one aware UTC boundary for the current process."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print one flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7423] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _stable_hash(value: str) -> str:
    """Create an opaque stable identity from label-blind text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def asset_url(relative: str) -> str:
    """Build an immutable raw URL for one allowed release file."""

    if relative not in ASSET_PATHS:
        raise ValueError(f"asset_not_allowed:{relative}")
    return f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_COMMIT}/{relative}"


def _download(url: str, timeout_s: float, byte_limit: int) -> bytes:  # pragma: no cover
    """Read one pinned file with byte, wall-time, and progress bounds."""

    started = time.monotonic()
    chunks: list[bytes] = []
    completed = 0
    last_report = started
    with urlopen(url, timeout=min(timeout_s, 60.0)) as response:  # noqa: S310 - fixed HTTPS URL.
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
                progress(started, "acquisition", "download_heartbeat", completed_units=completed)
                last_report = now
            if now - started > timeout_s:
                raise SourceBlocked(f"fetch_timeout:{url}:{timeout_s}")
    return b"".join(chunks)


def _asset_path(cache_root: Path, relative: str) -> Path:
    """Resolve one release file below the commit-named cache directory."""

    return cache_root / RAGTRUTH_COMMIT / relative


def _receipt_for_paths(cache_root: Path, paths: Mapping[str, Path]) -> JsonDict:
    """Authenticate release bytes before any label is decoded."""

    files: list[JsonDict] = []
    total = 0
    for relative in ASSET_PATHS:
        path = paths[relative]
        if not path.is_file():
            raise SourceBlocked(f"cache_asset_missing:{relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        expected = EXPECTED_ASSETS[relative]
        if size != expected["size_bytes"]:
            raise SourceBlocked(f"cache_size_mismatch:{relative}:{size}")
        if digest != expected["sha256"]:
            raise SourceBlocked(f"cache_hash_mismatch:{relative}:{digest}")
        files.append(
            {
                "path": relative,
                "cache_path": str(path.resolve()),
                "url": asset_url(relative),
                "size_bytes": size,
                "sha256": digest,
            }
        )
        total += size
    return {
        "repository": RAGTRUTH_REPO,
        "commit": RAGTRUTH_COMMIT,
        "commit_url": f"{RAGTRUTH_REPO}/commit/{RAGTRUTH_COMMIT}",
        "files": files,
        "total_bytes": total,
        "cache_root": str(cache_root.resolve()),
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "authenticated": True,
    }


def authenticate_assets(cache_root: Path) -> JsonDict:  # pragma: no cover - live cache path.
    """Reuse exact cached release bytes, including the authenticated legacy cache."""

    standard = {relative: _asset_path(cache_root, relative) for relative in ASSET_PATHS}
    if all(path.is_file() for path in standard.values()):
        return _receipt_for_paths(cache_root, standard)
    legacy = {
        "dataset/source_info.jsonl": LEGACY_CACHE_ROOT / "source_info.jsonl",
        "dataset/response.jsonl": LEGACY_CACHE_ROOT / "response.jsonl",
    }
    if all(path.is_file() for path in legacy.values()):
        return _receipt_for_paths(LEGACY_CACHE_ROOT, legacy)
    raise SourceBlocked("cache_assets_unavailable")


def fetch_assets(cache_root: Path) -> JsonDict:  # pragma: no cover - bounded network path.
    """Reuse exact bytes or fetch only the two pinned public JSONL files."""

    try:
        return authenticate_assets(cache_root)
    except SourceBlocked:
        pass
    transferred = 0
    for relative in ASSET_PATHS:
        remaining = MAX_TRANSFER_BYTES - transferred
        if remaining <= 0:
            raise SourceBlocked("transfer_limit_exceeded:release")
        try:
            payload = _download(asset_url(relative), FETCH_TIMEOUT_S, remaining)
        except (OSError, TimeoutError) as exc:
            raise SourceBlocked(f"network_failure:{relative}:{type(exc).__name__}:{exc}") from exc
        expected = EXPECTED_ASSETS[relative]
        if len(payload) != expected["size_bytes"] or _sha256_bytes(payload) != expected["sha256"]:
            raise SourceBlocked(f"download_identity_mismatch:{relative}")
        path = _asset_path(cache_root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        transferred += len(payload)
    return authenticate_assets(cache_root)


def _jsonl_rows(path: Path, *, started: float | None = None) -> list[JsonDict]:  # pragma: no cover
    """Decode one authenticated JSONL file with bounded progress reports."""

    rows: list[JsonDict] = []
    last_report = time.monotonic()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusInvalid(f"jsonl_invalid:{path.name}:{line_number}") from exc
            if not isinstance(value, dict):
                raise CorpusInvalid(f"jsonl_row_not_object:{path.name}:{line_number}")
            rows.append(value)
            now = time.monotonic()
            if started is not None and now - last_report >= 60.0:
                progress(started, "decode", "jsonl_heartbeat", completed_units=len(rows))
                last_report = now
    return rows


def load_release(
    receipt: Mapping[str, Any], *, started: float | None = None
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Decode labels only after both release files pass identity checks."""

    files = {str(row["path"]): Path(str(row["cache_path"])) for row in receipt["files"]}
    return (
        _jsonl_rows(files["dataset/source_info.jsonl"], started=started),
        _jsonl_rows(files["dataset/response.jsonl"], started=started),
    )


def _normalized_text(value: str) -> str:
    """Normalize source representation without using annotations or labels."""

    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def serialize_source(task_type: str, source_info: Any) -> str:
    """Serialize each official task type without model or annotation fields."""

    if task_type == "Summary":
        if not isinstance(source_info, str):
            raise CorpusInvalid("source_info_invalid:Summary")
        return source_info
    if task_type in {"QA", "Data2txt"}:
        if not isinstance(source_info, Mapping):
            raise CorpusInvalid(f"source_info_invalid:{task_type}")
        return json.dumps(source_info, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    raise CorpusInvalid(f"task_type_invalid:{task_type}")


def _validated_annotations(response: Mapping[str, Any]) -> list[JsonDict]:
    """Copy exact human spans only after checking their response offsets."""

    text = response.get("response")
    labels = response.get("labels")
    if not isinstance(text, str) or not isinstance(labels, list):
        raise CorpusInvalid(f"response_shape_invalid:{response.get('id')}")
    allowed = {
        "start",
        "end",
        "text",
        "label_type",
        "due_to_null",
        "implicit_true",
        "meta",
    }
    copied: list[JsonDict] = []
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping) or not {"start", "end", "text", "label_type"}.issubset(
            label
        ):
            raise CorpusInvalid(f"annotation_shape_invalid:{response.get('id')}:{index}")
        start = label.get("start")
        end = label.get("end")
        span_text = label.get("text")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end < start
            or end > len(text)
            or not isinstance(span_text, str)
            or text[start:end] != span_text
        ):
            raise CorpusInvalid(f"annotation_span_invalid:{response.get('id')}:{index}")
        copied.append({key: deepcopy(label.get(key)) for key in allowed})
    return copied


def join_release(
    source_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Join by source ID and apply the frozen quality exclusion rule."""

    sources: dict[str, Mapping[str, Any]] = {}
    for source in source_rows:
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in sources:
            raise CorpusInvalid(f"source_identity_invalid:{source_id}")
        task_type = str(source.get("task_type") or "")
        serialize_source(task_type, source.get("source_info"))
        sources[source_id] = source

    joined: list[JsonDict] = []
    dispositions: list[JsonDict] = []
    seen_responses: set[str] = set()
    for response in response_rows:
        response_id = response.get("id")
        source_id = response.get("source_id")
        if not isinstance(response_id, str) or not response_id or response_id in seen_responses:
            raise CorpusInvalid(f"response_identity_invalid:{response_id}")
        seen_responses.add(response_id)
        if source_id not in sources:
            raise CorpusInvalid(f"source_id_missing:{source_id}")
        split = response.get("split")
        quality = response.get("quality")
        if split not in {"train", "test"}:
            raise CorpusInvalid(f"split_invalid:{response_id}:{split}")
        if quality not in {"good", "incorrect_refusal", "truncated"}:
            raise CorpusInvalid(f"quality_invalid:{response_id}:{quality}")
        if quality != "good":
            dispositions.append(
                {
                    "response_id": response_id,
                    "source_id": source_id,
                    "reason": f"excluded_{quality}",
                    "official_split": split,
                }
            )
            continue
        annotations = _validated_annotations(response)
        source = sources[str(source_id)]
        task_type = str(source["task_type"])
        source_text = serialize_source(task_type, source.get("source_info"))
        primary_label = int(not annotations)
        sensitivity_label = int(not any(not bool(row.get("implicit_true")) for row in annotations))
        joined.append(
            {
                "response_id": response_id,
                "source_id": source_id,
                "task_type": task_type,
                "source_name": source.get("source"),
                "source_text": source_text,
                "source_normalized": _normalized_text(source_text),
                "response_text": response["response"],
                "model": response.get("model"),
                "temperature": response.get("temperature"),
                "official_split": split,
                "quality": quality,
                "annotations": annotations,
                "primary_label": primary_label,
                "implicit_true_excluded_label": sensitivity_label,
            }
        )
    return joined, dispositions


def _development_partition(group_id: str) -> str:
    """Map one group to fixed ranges without reading its labels."""

    rank = int(_stable_hash(f"{PARTITION_SALT}:partition:{group_id}")[:16], 16) / 2**64
    if rank < 0.40:
        return "fit"
    if rank < 0.55:
        return "probability_calibration"
    if rank < 0.70:
        return "policy_calibration"
    return "prospective_stream"


def assign_groups_and_partitions(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep duplicate source bytes and all sibling responses in one group."""

    by_source: dict[str, set[str]] = defaultdict(set)
    by_normalized: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_id = str(row["source_id"])
        normalized = str(row["source_normalized"])
        by_source[source_id].add(source_id)
        by_normalized[normalized].add(source_id)

    parent = {source_id: source_id for source_id in by_source}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for source_ids in by_normalized.values():
        ordered = sorted(source_ids)
        for source_id in ordered[1:]:
            union(ordered[0], source_id)

    component_members: dict[str, list[str]] = defaultdict(list)
    for source_id in sorted(parent):
        component_members[find(source_id)].append(source_id)
    group_for_source: dict[str, str] = {}
    for members in component_members.values():
        group_id = "group-" + _stable_hash("\n".join(members))[:24]
        for source_id in members:
            group_for_source[source_id] = group_id

    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_id = group_for_source[str(row["source_id"])]
        splits_by_group[group_id].add(str(row["official_split"]))

    assigned: list[JsonDict] = []
    for source in rows:
        row = deepcopy(dict(source))
        group_id = group_for_source[str(row["source_id"])]
        row["group_id"] = group_id
        splits = splits_by_group[group_id]
        if row["official_split"] == "test":
            row["partition"] = "final_test"
        elif "test" in splits:
            row["partition"] = "excluded_test_overlap"
        else:
            row["partition"] = _development_partition(group_id)
        assigned.append(row)
    return assigned


def apply_group_caps(
    rows: Sequence[Mapping[str, Any]], *, caps: Mapping[str, int] = CAPS
) -> list[JsonDict]:
    """Apply label-blind hash caps while retaining every sibling response."""

    selected_groups: dict[str, set[str]] = {}
    for partition, cap in caps.items():
        groups = {str(row["group_id"]) for row in rows if row.get("partition") == partition}
        ranked = sorted(
            groups,
            key=lambda group_id: _stable_hash(f"{PARTITION_SALT}:cap:{partition}:{group_id}"),
        )
        selected_groups[partition] = set(ranked[:cap])
    return [
        deepcopy(dict(row))
        for row in rows
        if row.get("partition") not in caps
        or str(row.get("group_id")) in selected_groups[str(row["partition"])]
    ]


def label_blind_representatives(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Select one response per group from opaque response identity only."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    selected: list[JsonDict] = []
    for group_id in sorted(grouped):
        winner = min(
            grouped[group_id],
            key=lambda row: _stable_hash(
                f"{PARTITION_SALT}:response:{group_id}:{row['response_id']}"
            ),
        )
        selected.append(deepcopy(dict(winner)))
    return selected


def _bounded(value: str, limit: int) -> tuple[str, int, bool]:
    """Bound lexical work and record whether source text was shortened."""

    all_tokens = bounded_word_tokens(value, token_limit=max(limit + 1, 1))
    truncated = len(all_tokens) > limit
    retained = all_tokens[:limit]
    return " ".join(retained), len(retained), truncated


def project_predictor(row: Mapping[str, Any]) -> JsonDict:
    """Create one label-free predictor row with the six shipped proxies."""

    source_text, source_count, source_truncated = _bounded(
        str(row["source_text"]), SOURCE_TOKEN_LIMIT
    )
    response_text, response_count, response_truncated = _bounded(
        str(row["response_text"]), RESPONSE_TOKEN_LIMIT
    )
    row_key = "row-" + _stable_hash(f"{PARTITION_SALT}:{row['group_id']}:{row['response_id']}")[:24]
    feature_input = {
        "row_key": row_key,
        "group_id": row["group_id"],
        "partition": row["partition"],
        "question": "",
        "context": source_text,
        "answer": response_text,
        "sentence": response_text,
    }
    return {
        "row_key": row_key,
        "group_id": row["group_id"],
        "partition": row["partition"],
        "task_type": row["task_type"],
        "source_text": source_text,
        "response_text": response_text,
        "source_token_count": source_count,
        "response_token_count": response_count,
        "source_truncated": source_truncated,
        "response_truncated": response_truncated,
        "certificate_selected": bool(row.get("certificate_selected")),
        "features": extract_feature_row(feature_input)["source_features"],
    }


def project_evaluator(row: Mapping[str, Any], row_key: str) -> JsonDict:
    """Keep source-support authority in the evaluator-only view."""

    return {
        "row_key": row_key,
        "group_id": row["group_id"],
        "partition": row["partition"],
        "task_type": row["task_type"],
        "source_id": row["source_id"],
        "response_id": row["response_id"],
        "source_name": row["source_name"],
        "model": row["model"],
        "temperature": row["temperature"],
        "official_split": row["official_split"],
        "quality": row["quality"],
        "annotations": deepcopy(row["annotations"]),
        "primary_label": row["primary_label"],
        "implicit_true_excluded_label": row["implicit_true_excluded_label"],
        "certificate_selected": bool(row.get("certificate_selected")),
    }


def protocol_plan() -> JsonDict:
    """Return the fixed future fitting and online evaluation design."""

    return {
        "training_seeds": list(TRAINING_SEEDS),
        "training_budget_steps": MAX_STEPS,
        "fit_role": "fit",
        "probability_calibration_role": "probability_calibration",
        "policy_calibration_role": "policy_calibration",
        "stream_role": "prospective_stream",
        "final_static_role": "final_test",
        "accept_thresholds": list(ACCEPT_THRESHOLDS),
        "reject_thresholds": list(REJECT_THRESHOLDS),
        "arms": list(ARMS),
        "online_orders": list(ONLINE_ORDERS),
        "feedback_regimes": list(FEEDBACK_REGIMES),
        "feedback_delays": list(FEEDBACK_DELAYS),
        "partition_salt": PARTITION_SALT,
        "group_caps": deepcopy(CAPS),
        "primary_label": "1 means no human-annotated source-unsupported span",
        "sensitivity_label": "ignore only spans marked implicit_true; never tune on this view",
    }


def planned_rows() -> list[JsonDict]:
    """Enumerate every deferred arm, seed, and online condition."""

    return [
        {
            "unit": "future_human_label_measurement",
            "arm": arm,
            "seed": seed,
            "condition": {
                "order": order,
                "feedback_regime": regime,
                "delay": delay,
            },
            "status": "unstarted",
            "attempted": False,
            "completed": False,
            "censored": False,
            "failure": None,
        }
        for arm in ARMS
        for seed in TRAINING_SEEDS
        for order in ONLINE_ORDERS
        for regime in FEEDBACK_REGIMES
        for delay in FEEDBACK_DELAYS
    ]


def _manifest_hash(value: Mapping[str, Any]) -> str:
    """Bind manifest content without its self-referential hash field."""

    stable = {key: deepcopy(item) for key, item in value.items() if key != "manifest_hash"}
    return canonical_hash(stable)


def _write_jsonl_shards(
    raw_dir: Path,
    prefix: str,
    kind: str,
    records: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Write deterministic shards that remain below the repository size cap."""

    shards: list[JsonDict] = []
    chunk: list[bytes] = []
    chunk_size = 0

    def flush() -> None:
        nonlocal chunk, chunk_size
        if not chunk:
            return
        index = len(shards)
        path = raw_dir / f"{prefix}-{index:03d}.jsonl"
        payload = b"".join(chunk)
        path.write_bytes(payload)
        shards.append(
            {
                "path": path.name,
                "kind": kind,
                "rows": len(chunk),
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
        chunk = []
        chunk_size = 0

    for record in records:
        encoded = (
            json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if len(encoded) > MAX_SHARD_BYTES:
            raise CorpusInvalid(f"single_record_exceeds_shard_limit:{prefix}")
        if chunk and chunk_size + len(encoded) > MAX_SHARD_BYTES:
            flush()
        chunk.append(encoded)
        chunk_size += len(encoded)
    flush()
    return shards


def _disposition_rows(
    assigned: Sequence[Mapping[str, Any]],
    capped: Sequence[Mapping[str, Any]],
    quality_dispositions: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Count each quality, overlap, and label-blind cap exclusion explicitly."""

    kept = {str(row["response_id"]) for row in capped if row["partition"] in ELIGIBLE_PARTITIONS}
    rows = [deepcopy(dict(row)) for row in quality_dispositions]
    for row in assigned:
        response_id = str(row["response_id"])
        if row["partition"] == "excluded_test_overlap":
            reason = "excluded_cross_split_duplicate_component"
        elif response_id not in kept:
            reason = "excluded_partition_group_cap"
        else:
            continue
        rows.append(
            {
                "response_id": response_id,
                "source_id": row["source_id"],
                "reason": reason,
                "official_split": row["official_split"],
                "group_id": row["group_id"],
            }
        )
    return sorted(rows, key=lambda row: (str(row["reason"]), str(row["response_id"])))


def seal_corpus(
    raw_dir: Path,
    source_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    *,
    asset_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Write separate predictor and evaluator shards plus one frozen manifest."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    joined, quality_dispositions = join_release(source_rows, response_rows)
    assigned = assign_groups_and_partitions(joined)
    capped = apply_group_caps(assigned)
    eligible = [row for row in capped if row["partition"] in ELIGIBLE_PARTITIONS]
    representatives = {str(row["response_id"]) for row in label_blind_representatives(eligible)}
    predictors: list[JsonDict] = []
    evaluators: list[JsonDict] = []
    for source in sorted(eligible, key=lambda row: str(row["response_id"])):
        row = deepcopy(dict(source))
        row["certificate_selected"] = str(row["response_id"]) in representatives
        predictor = project_predictor(row)
        predictors.append(predictor)
        evaluators.append(project_evaluator(row, str(predictor["row_key"])))
    dispositions = _disposition_rows(assigned, eligible, quality_dispositions)

    shards: list[JsonDict] = []
    for partition in ELIGIBLE_PARTITIONS:
        partition_predictors = [row for row in predictors if row["partition"] == partition]
        partition_evaluators = [row for row in evaluators if row["partition"] == partition]
        shards.extend(
            _write_jsonl_shards(
                raw_dir,
                f"predictor-{partition}",
                "predictor",
                partition_predictors,
            )
        )
        shards.extend(
            _write_jsonl_shards(
                raw_dir,
                f"evaluator-{partition}",
                "evaluator",
                partition_evaluators,
            )
        )
    shards.extend(_write_jsonl_shards(raw_dir, "dispositions", "disposition", dispositions))
    manifest: JsonDict = {
        "schema": "carnot.exp7423.corpus_manifest.v1",
        "repository": RAGTRUTH_REPO,
        "commit": RAGTRUTH_COMMIT,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "label_authority": LABEL_AUTHORITY,
        "label_authority_limit": "fallible human source-support judgment; not formal truth",
        "asset_receipt": deepcopy(dict(asset_receipt or {})),
        "protocol": protocol_plan(),
        "feature_contract": feature_contract(),
        "predictor_fields": list(PREDICTOR_FIELDS),
        "evaluator_fields": list(EVALUATOR_FIELDS),
        "predictor_row_count": len(predictors),
        "evaluator_row_count": len(evaluators),
        "independent_group_count": len({row["group_id"] for row in predictors}),
        "partition_group_counts": {
            partition: len({row["group_id"] for row in predictors if row["partition"] == partition})
            for partition in ELIGIBLE_PARTITIONS
        },
        "task_type_counts": dict(sorted(Counter(row["task_type"] for row in predictors).items())),
        "exclusion_counts": dict(
            sorted(Counter(str(row["reason"]) for row in dispositions).items())
        ),
        "source_truncation_count": sum(bool(row["source_truncated"]) for row in predictors),
        "response_truncation_count": sum(bool(row["response_truncated"]) for row in predictors),
        "shards": shards,
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    atomic_json(raw_dir / "corpus_manifest.json", manifest)
    return manifest


def _load_jsonl(path: Path) -> list[JsonDict]:
    """Load one already authenticated local shard."""

    rows: list[JsonDict] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CorpusInvalid(f"shard_row_not_object:{path.name}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusInvalid(f"shard_unreadable:{path.name}:{type(exc).__name__}") from exc
    return rows


def reload_corpus(raw_dir: Path) -> JsonDict:
    """Rehash every sealed byte and reconstruct both access-controlled views."""

    manifest_path = raw_dir / "corpus_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusInvalid(f"manifest_unreadable:{type(exc).__name__}") from exc
    if not isinstance(manifest, dict) or manifest.get("manifest_hash") != _manifest_hash(manifest):
        raise CorpusInvalid("manifest_hash_mismatch")
    by_kind: dict[str, list[JsonDict]] = defaultdict(list)
    for shard in manifest.get("shards") or []:
        path = raw_dir / str(shard.get("path"))
        if not path.is_file() or sha256_file(path) != shard.get("sha256"):
            raise CorpusInvalid(f"shard_hash_mismatch:{shard.get('path')}")
        rows = _load_jsonl(path)
        if len(rows) != shard.get("rows"):
            raise CorpusInvalid(f"shard_row_count_mismatch:{shard.get('path')}")
        by_kind[str(shard.get("kind"))].extend(rows)
    predictors = by_kind["predictor"]
    evaluators = by_kind["evaluator"]
    if len(predictors) != manifest.get("predictor_row_count"):
        raise CorpusInvalid("predictor_row_count_mismatch")
    if len(evaluators) != manifest.get("evaluator_row_count"):
        raise CorpusInvalid("evaluator_row_count_mismatch")
    if any(set(row) != set(PREDICTOR_FIELDS) for row in predictors):
        raise CorpusInvalid("predictor_fields_mismatch")
    if any(set(row) != set(EVALUATOR_FIELDS) for row in evaluators):
        raise CorpusInvalid("evaluator_fields_mismatch")
    predictor_keys = {row["row_key"] for row in predictors}
    evaluator_keys = {row["row_key"] for row in evaluators}
    if predictor_keys != evaluator_keys or len(predictor_keys) != len(predictors):
        raise CorpusInvalid("view_row_keys_mismatch")
    partitions_by_group: dict[str, set[str]] = defaultdict(set)
    for row in predictors:
        partitions_by_group[str(row["group_id"])].add(str(row["partition"]))
    if any(len(partitions) != 1 for partitions in partitions_by_group.values()):
        raise CorpusInvalid("group_partition_leak")
    return {
        "raw_dir": str(raw_dir.resolve()),
        "manifest": manifest,
        "manifest_hash": manifest["manifest_hash"],
        "predictors": predictors,
        "evaluators": evaluators,
        "dispositions": by_kind["disposition"],
    }


class ProtocolReaders:
    """Expose predictor text while requiring a token for human annotations."""

    def __init__(self, reloaded: Mapping[str, Any]) -> None:
        self._predictors = deepcopy(list(reloaded["predictors"]))
        self._evaluators = deepcopy(list(reloaded["evaluators"]))

    def read_predictors(self, partition: str) -> list[JsonDict]:
        """Return only the declared predictor fields for one sealed role."""

        if partition not in ELIGIBLE_PARTITIONS:
            raise ValueError(f"partition_invalid:{partition}")
        return [deepcopy(row) for row in self._predictors if row["partition"] == partition]

    def read_evaluators(self, partition: str, token: str) -> list[JsonDict]:
        """Require explicit evaluator authority before exposing human labels."""

        if token != EVALUATOR_TOKEN:
            raise PermissionError("evaluator access denied")
        if partition not in ELIGIBLE_PARTITIONS:
            raise ValueError(f"partition_invalid:{partition}")
        return [deepcopy(row) for row in self._evaluators if row["partition"] == partition]


def reduce_support(evaluator_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count label-blind representatives by partition, label, and task domain."""

    selected = [row for row in evaluator_rows if row.get("certificate_selected") is True]
    result: JsonDict = {}
    for partition in ELIGIBLE_PARTITIONS:
        rows = [row for row in selected if row.get("partition") == partition]
        labels = Counter(str(row.get("primary_label")) for row in rows)
        by_domain: JsonDict = {}
        for task_type in ("QA", "Summary", "Data2txt"):
            domain_rows = [row for row in rows if row.get("task_type") == task_type]
            domain_labels = Counter(str(row.get("primary_label")) for row in domain_rows)
            by_domain[task_type] = {
                "independent_groups": len({row.get("group_id") for row in domain_rows}),
                "label_counts": {"0": domain_labels["0"], "1": domain_labels["1"]},
            }
        groups = len({row.get("group_id") for row in rows})
        if partition in {"probability_calibration", "policy_calibration"}:
            minimum_groups, minimum_class = 100, 20
        elif partition == "prospective_stream":
            minimum_groups, minimum_class = 400, 40
        else:
            minimum_groups, minimum_class = 0, 0
        result[partition] = {
            "independent_groups": groups,
            "label_counts": {"0": labels["0"], "1": labels["1"]},
            "minimum_independent_groups": minimum_groups,
            "minimum_each_class": minimum_class,
            "support_ready": groups >= minimum_groups
            and labels["0"] >= minimum_class
            and labels["1"] >= minimum_class,
            "by_domain": by_domain,
        }
    return result


def classify_support(support: Mapping[str, Any], *, contract_valid: bool) -> tuple[str, str, int]:
    """Keep protocol readiness independent from later inferential value."""

    if not contract_valid:
        return "complete_disqualified_protocol_contract", "disqualified", 0
    required = ("probability_calibration", "policy_calibration", "prospective_stream")
    if not all(bool(support.get(name, {}).get("support_ready")) for name in required):
        return "complete_null_insufficient_human_label_support", "null", 1
    return "complete_null_protocol_only_no_benefit_test", "null", 1


def feature_contract() -> JsonDict:
    """Describe the exact six proxy inputs and evaluator-field denial."""

    return {
        "source_feature_names": list(SOURCE_FEATURE_NAMES),
        "definitions": feature_definitions(),
        "source_token_limit": SOURCE_TOKEN_LIMIT,
        "response_token_limit": RESPONSE_TOKEN_LIMIT,
        "predictor_fields": list(PREDICTOR_FIELDS),
        "denied_predictor_fields": [
            "annotations",
            "primary_label",
            "implicit_true_excluded_label",
            "model",
            "quality",
            "source_id",
            "response_id",
            "official_split",
            "temperature",
        ],
        "semantics": "lexical and PCIB proxy inputs; not entailment or proof",
    }


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358 plan with private test and coverage parents."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject scope expansion or missing private validation directories."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep each machine-readable gate scalar separate from its explanation."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def _validation_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing receipt for each affected and terminal check."""

    expected = {
        *validation_scope.REQUIRED_CHECK_NAMES,
        "declared_entrypoint_cold_replay",
        "independent_cold_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is not True
        for name in expected
    )


def _field_principles() -> dict[str, str]:
    """Explain field intent without wrapping the ordinary field values."""

    explicit = {
        "schema": "Use a versioned plain top-level schema with terminal identity.",
        "run_date": "Use the fixed execution date with actual UTC and monotonic timing.",
        "preconditions_checked": "Record exact paths, identities, and observed values first.",
        "MODEL_SPECS": "List current LLMs; use an empty list when no LLM runs.",
        "model_invoked": "Count only actual current attempted model use.",
        "invocation_counts": "Record actual owned current call lifecycle counts.",
        "inference_substrate": "State the current inference substrate as a plain string.",
        "inference_substrate_class": "State the actual current compute class.",
        "execution_venue": "Use the closed host venue value.",
        "duration_s": "Measure current work and separate validation and cold-start time.",
        "phase_spans": "Record real phase times, progress boundaries, and checkpoints.",
        "random_seed": "Freeze partition, fitting, sampling, and future resampling seeds.",
        "reproducibility_checksum": "Bind code, protocol, inputs, raw rows, and validation scope.",
        "source_artifact_hashes": "Identify exact source and shard bytes.",
        "rows": "List every deferred comparative arm, seed, and condition.",
        "sample_size_budget": "Separate planned, attempted, completed, failed, and unstarted work.",
        "acceptance_gate_results": "Keep contract validity separate from inferential support.",
        "gate_check_summary": "Name exact failed or blocked fields without hiding zero values.",
        "verifier_is_oracle": "Human source-support labels are fallible, not an oracle.",
        "honest_verdict": "Use complete or blocked terminal prefixes.",
        "verdict_class": "Use the closed terminal disposition enum.",
        "flagged_adversarial": "Preserve critical findings and deny readiness when flagged.",
        "validation_receipts": "Retain exact scoped commands, exits, durations, and log hashes.",
        "field_principles": "Explain field intent separately from scalar values.",
        "promotion_score": "Remain zero; this protocol cannot trigger deployment.",
        "annotated_protocol_ready_score": "Measure valid authority, masks, and sealed partitions.",
        "corpus_manifest": "Record revision, attribution, hashes, assignments, and exclusions.",
        "support_counts": "Count independent groups and labels by partition and domain.",
        "label_authority": "Name fallible human source-support annotation authority.",
        "feature_contract": "Freeze six proxies and deny evaluator fields.",
    }
    return {
        field: explicit.get(field, f"Record the measured {field.replace('_', ' ')} value.")
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable terminal evidence without the self-referential checksum."""

    excluded = {"reproducibility_checksum"}
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _source_hashes(
    raw_dir: Path, manifest: Mapping[str, Any], root: Path | None = None
) -> JsonDict:
    """Bind local authorities, source assets, code, and emitted shards."""

    base = (root or REPO_ROOT).resolve()
    result: JsonDict = {
        "ragtruth": deepcopy(dict(manifest.get("asset_receipt") or {})),
        "corpus_manifest": {
            "path": str((raw_dir / "corpus_manifest.json").resolve()),
            "sha256": sha256_file(raw_dir / "corpus_manifest.json"),
            "manifest_hash": manifest["manifest_hash"],
        },
        "raw_shards": deepcopy(list(manifest["shards"])),
    }
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        path = base / relative
        if path.is_file():
            result[relative.as_posix()] = sha256_file(path)
    return result


def _base_artifact(
    *,
    manifest: Mapping[str, Any],
    reloaded: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    raw_dir: Path,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_ns: int,
    ended_ns: int,
    flagged_adversarial: bool,
    fixture_artifact: bool,
) -> JsonDict:
    """Build one schema-complete terminal record from independently reloadable rows."""

    support = reduce_support(reloaded["evaluators"])
    validations_pass = _validation_complete(validation_receipts)
    contract_valid = (
        validations_pass
        and not flagged_adversarial
        and manifest.get("manifest_hash") == reloaded.get("manifest_hash")
        and manifest.get("label_authority") == LABEL_AUTHORITY
    )
    verdict, verdict_class, readiness = classify_support(support, contract_valid=contract_valid)
    current = build_current_work_receipt(
        run_id="exp7423-fixture" if fixture_artifact else EXPERIMENT_ID,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "device": "host_cpu",
            "work": "RAGTruth reduction and lexical proxy extraction",
            "current_llm": None,
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=phase_spans,
        small_ebm_training={
            "performed": False,
            "receipts": [],
            "future_training_budget_steps": MAX_STEPS,
            "future_training_seeds": list(TRAINING_SEEDS),
        },
    )
    gates = [
        _gate(
            "human_label_authority",
            "validity",
            "==",
            LABEL_AUTHORITY,
            manifest.get("label_authority"),
            manifest.get("label_authority") == LABEL_AUTHORITY,
            "Only author human source-support annotations may label this protocol.",
        ),
        _gate(
            "sealed_view_reload",
            "validity",
            "==",
            manifest.get("manifest_hash"),
            reloaded.get("manifest_hash"),
            manifest.get("manifest_hash") == reloaded.get("manifest_hash"),
            "Freshly hashed shards must reproduce the frozen manifest.",
        ),
        _gate(
            "predictor_field_mask",
            "leakage",
            "==",
            list(PREDICTOR_FIELDS),
            manifest.get("predictor_fields"),
            manifest.get("predictor_fields") == list(PREDICTOR_FIELDS),
            "Predictors cannot observe evaluator authority.",
        ),
        _gate(
            "required_validation",
            "completion",
            "==",
            True,
            validations_pass,
            validations_pass,
            "All affected and terminal readers must pass once.",
        ),
        _gate(
            "inferential_support",
            "scientific_support",
            "==",
            True,
            all(
                support[name]["support_ready"]
                for name in (
                    "probability_calibration",
                    "policy_calibration",
                    "prospective_stream",
                )
            ),
            True,
            "Support limits future benefit claims but does not invalidate this audit.",
        ),
    ]
    failed_required = [gate for gate in gates[:-1] if gate["passed"] is not True]
    planned = planned_rows()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": verdict,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **current,
        "validation_duration_s": sum(
            float(row.get("duration_s") or 0.0) for row in validation_receipts
        ),
        "model_duration_s": 0.0,
        "cold_start_duration_s": sum(
            float(row.get("duration_s") or 0.0)
            for row in validation_receipts
            if row.get("name") in {"declared_entrypoint_cold_replay", "independent_cold_reducer"}
        ),
        "random_seed": {
            "partition_salt": PARTITION_SALT,
            "training_seeds": list(TRAINING_SEEDS),
            "future_resampling_seed": 65_123_07,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": planned,
        "sample_size_budget": {
            "planned": len(planned),
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(planned),
            "independent_groups": manifest["independent_group_count"],
            "eligible_responses": manifest["predictor_row_count"],
            "stop_rule": "protocol audit only; no label fitting or benefit test in Exp7423",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "required_checks_passed": not failed_required,
            "failed_required_checks": failed_required,
            "support_limited": not gates[-1]["observed"],
        },
        "verifier_is_oracle": False,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "annotated_protocol_ready_score": readiness,
        "corpus_manifest": deepcopy(dict(manifest)),
        "corpus_manifest_path": str((raw_dir / "corpus_manifest.json").resolve()),
        "support_counts": support,
        "label_authority": LABEL_AUTHORITY,
        "label_authority_limit": "fallible human source-support judgment; not formal truth",
        "feature_contract": feature_contract(),
        "protocol_plan": protocol_plan(),
        "implicit_true_semantics": "unsupported by provided source; not false in the world",
        "sensitivity_tuned": False,
        "benefit_claim": False,
        "historical_graph_extraction_repeated": False,
        "fixture_artifact": fixture_artifact,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(
    raw_dir: Path,
    source_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a deterministic fixture artifact without child processes or network access."""

    receipt = {
        "repository": RAGTRUTH_REPO,
        "commit": RAGTRUTH_COMMIT,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "authenticated": True,
        "files": [],
    }
    manifest = seal_corpus(raw_dir, source_rows, response_rows, asset_receipt=receipt)
    reloaded = reload_corpus(raw_dir)
    return _base_artifact(
        manifest=manifest,
        reloaded=reloaded,
        validation_receipts=validation_receipts,
        preconditions=[{"check": "fixture", "expected": True, "observed": True, "passed": True}],
        source_hashes=_source_hashes(raw_dir, manifest),
        raw_dir=raw_dir,
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:00+00:00",
        started_ns=0,
        ended_ns=0,
        flagged_adversarial=False,
        fixture_artifact=True,
    )


def build_blocked_artifact(
    *,
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Publish external absence as blocked, never as unfinished owned work."""

    current = build_current_work_receipt(
        run_id="exp7423-blocked",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={"work": "precondition and source availability checks"},
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        small_ebm_training={"performed": False, "receipts": []},
    )
    blocked = {
        "upstream": upstream,
        "path": path,
        "check": check,
        "field": field,
        "expected": expected,
        "observed": observed,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": "2026-09-19T00:00:00+00:00",
        "completed_at_utc": "2026-09-19T00:00:00+00:00",
        "preconditions_checked": [{**blocked, "passed": False}],
        **current,
        "validation_duration_s": 0.0,
        "model_duration_s": 0.0,
        "cold_start_duration_s": 0.0,
        "random_seed": {"partition_salt": PARTITION_SALT, "training_seeds": list(TRAINING_SEEDS)},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": len(planned_rows()),
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(planned_rows()),
            "independent_groups": 0,
            "stop_rule": "external source unavailable before dependent science",
        },
        "acceptance_gate_results": [
            _gate(
                check,
                "precondition",
                "==",
                expected,
                observed,
                False,
                "Unavailable human-label authority cannot be replaced by synthetic evidence.",
            )
        ],
        "gate_check_summary": {
            "required_checks_passed": False,
            "failed_required_checks": [{**blocked, "passed": False}],
            "support_limited": False,
            "blocked_upstream": upstream,
            "blocked_path": path,
            "blocked_check": check,
            "blocked_field": field,
            "blocked_expected": expected,
            "blocked_observed": observed,
        },
        "verifier_is_oracle": False,
        "honest_verdict": f"blocked_{check}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "annotated_protocol_ready_score": 0,
        "corpus_manifest": {},
        "corpus_manifest_path": None,
        "support_counts": {},
        "label_authority": LABEL_AUTHORITY,
        "label_authority_limit": "fallible human source-support judgment; not formal truth",
        "feature_contract": feature_contract(),
        "protocol_plan": protocol_plan(),
        "implicit_true_semantics": "unsupported by provided source; not false in the world",
        "sensitivity_tuned": False,
        "benefit_claim": False,
        "historical_graph_extraction_repeated": False,
        "fixture_artifact": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _manifest_root(value: Mapping[str, Any]) -> Path:
    """Resolve the declared manifest without trusting the process working directory."""

    path = Path(str(value.get("corpus_manifest_path") or ""))
    return path.parent


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check declarations, current calls, masks, support, and checksum."""

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
        "promotion_score": 0,
        "label_authority": LABEL_AUTHORITY,
        "verifier_is_oracle": False,
        "feature_contract": feature_contract(),
        "benefit_claim": False,
        "historical_graph_extraction_repeated": False,
        "sensitivity_tuned": False,
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
    if set(value.get("field_principles") or {}) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    errors.extend(validate_current_work_receipt(value, root=REPO_ROOT))
    if value.get("verdict_class") == "blocked":
        if value.get("annotated_protocol_ready_score") != 0:
            errors.append("blocked_readiness_invalid")
        if not str(value.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_invalid")
        return list(dict.fromkeys(errors))
    if not str(value.get("honest_verdict") or "").startswith(("complete_", "complete:")):
        errors.append("terminal_verdict_prefix_invalid")
    if value.get("rows") != planned_rows():
        errors.append("planned_rows_mismatch")
    manifest = value.get("corpus_manifest")
    if not isinstance(manifest, Mapping) or manifest.get("manifest_hash") != _manifest_hash(
        manifest
    ):
        errors.append("corpus_manifest_invalid")
    else:
        try:
            reloaded = reload_corpus(_manifest_root(value))
        except CorpusInvalid as exc:
            errors.append(str(exc))
        else:
            if reloaded["manifest_hash"] != manifest.get("manifest_hash"):
                errors.append("corpus_manifest_reload_mismatch")
            expected_support = reduce_support(reloaded["evaluators"])
            if value.get("support_counts") != expected_support:
                errors.append("support_counts_mismatch")
    validations_pass = _validation_complete(value.get("validation_receipts") or [])
    expected_ready = int(validations_pass and value.get("flagged_adversarial") is False)
    if value.get("annotated_protocol_ready_score") != expected_ready:
        errors.append("annotated_protocol_ready_score_mismatch")
    return list(dict.fromkeys(errors))


def independent_reduce_artifact(value: Mapping[str, Any]) -> list[str]:
    """Recompute raw support and terminal disposition without trusting summaries."""

    errors = validate_artifact(value)
    if value.get("verdict_class") == "blocked":
        return errors
    try:
        reloaded = reload_corpus(_manifest_root(value))
    except CorpusInvalid as exc:
        errors.append(str(exc))
        return list(dict.fromkeys(errors))
    support = reduce_support(reloaded["evaluators"])
    if value.get("support_counts") != support:
        errors.append("support_counts_mismatch")
    contract_valid = _validation_complete(value.get("validation_receipts") or []) and not bool(
        value.get("flagged_adversarial")
    )
    verdict, verdict_class, readiness = classify_support(support, contract_valid=contract_valid)
    if value.get("honest_verdict") != verdict or value.get("status") != verdict:
        errors.append("terminal_verdict_mismatch")
    if value.get("verdict_class") != verdict_class:
        errors.append("terminal_class_mismatch")
    if value.get("annotated_protocol_ready_score") != readiness:
        errors.append("terminal_readiness_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path) -> list[str]:
    """Reload one candidate from fresh bytes and reduce it independently."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"artifact_unreadable:{type(exc).__name__}:{exc}"]
    if not isinstance(value, dict):
        return ["artifact_not_object"]
    return independent_reduce_artifact(value)


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate branch-local authorities before reading external labels."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "operator": "==",
                "expected": "readable_nonempty_bytes",
                "observed": observed,
                "passed": observed == "readable_nonempty_bytes",
            }
        )
        if observed is not None:
            hashes[relative.as_posix()] = sha256_file(path)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "path": SPEC_PATH.as_posix(),
            "field": "REQ-*",
            "operator": "==",
            "expected": "REQ-AUTO-7423",
            "observed": "REQ-AUTO-7423" if "REQ-AUTO-7423" in spec_text else None,
            "passed": "REQ-AUTO-7423" in spec_text,
        }
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7423" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        {
            "check": "current_task_not_quarantined",
            "upstream": "ops/exclusion_manifest.yaml",
            "path": "ops/exclusion_manifest.yaml",
            "field": EXPERIMENT_ID,
            "operator": "==",
            "expected": False,
            "observed": excluded,
            "passed": not excluded,
        }
    )
    return checks, hashes


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover
    """Record one disjoint phase with a resumable completed-unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def _terminal_commands(
    root: Path, candidate: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover
    """Build fresh entrypoint, reducer, adversarial, and strict-reader checks."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7423_v651_annotated_protocol import independent_reduce_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=independent_reduce_artifact(v);print(json.dumps({'errors':e}),flush=True);"
        "raise SystemExit(bool(e))"
    )
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
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
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
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
    ]


def run_experiment(  # pragma: no cover - exercised by the declared entrypoint.
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> JsonDict:
    """Authenticate, seal, validate, replay, and atomically publish the protocol."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "start", completed_units=0)
    preconditions, local_hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(
        run_started,
        "preconditions",
        "end",
        completed_units=len(preconditions),
        passed=all(row["passed"] for row in preconditions),
    )
    failed = next((row for row in preconditions if row["passed"] is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(
            check=str(failed["check"]),
            upstream=str(failed["upstream"]),
            path=str(failed["path"]),
            field=str(failed["field"]),
            expected=failed["expected"],
            observed=failed["observed"],
        )
        progress(run_started, "write", "before_atomic_terminal", completed_units=0)
        atomic_json(root / output_path, blocked)
        progress(run_started, "write", "after_atomic_terminal", completed_units=1)
        return blocked

    phase_started = time.monotonic()
    progress(run_started, "acquisition", "before_fetch", completed_units=0)
    try:
        assets = fetch_assets(cache_root)
        progress(
            run_started,
            "acquisition",
            "after_fetch",
            completed_units=len(assets["files"]),
            bytes=assets["total_bytes"],
        )
        progress(run_started, "decode", "before_release_decode", completed_units=0)
        source_rows, response_rows = load_release(assets, started=run_started)
        progress(
            run_started,
            "decode",
            "after_release_decode",
            completed_units=len(source_rows) + len(response_rows),
        )
    except (SourceBlocked, CorpusInvalid) as exc:
        progress(run_started, "acquisition", "blocked", completed_units=0, observed=str(exc))
        blocked = build_blocked_artifact(
            check="ragtruth_assets_available",
            upstream=RAGTRUTH_REPO,
            path="dataset/source_info.jsonl,dataset/response.jsonl",
            field="commit_and_sha256",
            expected={"commit": RAGTRUTH_COMMIT, "assets": EXPECTED_ASSETS},
            observed=str(exc),
        )
        atomic_json(root / output_path, blocked)
        return blocked
    spans.append(
        _span(
            "acquisition_and_decode",
            phase_started,
            run_started,
            len(source_rows) + len(response_rows),
        )
    )

    raw_dir = root / RAW_DIR
    phase_started = time.monotonic()
    progress(run_started, "reduction", "before_corpus_reduction", completed_units=0)
    manifest = seal_corpus(raw_dir, source_rows, response_rows, asset_receipt=assets)
    reloaded = reload_corpus(raw_dir)
    spans.append(
        _span("corpus_reduction", phase_started, run_started, manifest["predictor_row_count"])
    )
    progress(
        run_started,
        "reduction",
        "after_corpus_reduction",
        completed_units=manifest["predictor_row_count"],
    )

    phase_started = time.monotonic()
    private_root = Path(tempfile.mkdtemp(prefix="exp7423-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    progress(
        run_started,
        "validation",
        "before_affected_subprocesses",
        completed_units=0,
        planned=len(commands),
        plan_errors=len(plan_errors),
    )
    affected_receipts: list[JsonDict] = []
    if not plan_errors:
        affected_receipts = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
            heartbeat_s=60.0,
        )
    affected = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected_receipts)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected_receipts)))
    progress(
        run_started,
        "validation",
        "after_affected_subprocesses",
        completed_units=len(affected_receipts),
        passed=affected["passed"] and not plan_errors,
    )

    source_hashes = {**local_hashes, **_source_hashes(raw_dir, manifest, root)}
    candidate = _base_artifact(
        manifest=manifest,
        reloaded=reloaded,
        validation_receipts=affected_receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        raw_dir=raw_dir,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        flagged_adversarial=not affected["passed"] or bool(plan_errors),
        fixture_artifact=False,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    terminal_specs = _terminal_commands(root, candidate_path)
    progress(
        run_started,
        "terminal_validation",
        "before_subprocesses",
        completed_units=0,
        planned=len(terminal_specs),
    )
    terminal_receipts = run_categorized_commands(
        root,
        [PlannedCommand(spec, "terminal_validation", True) for spec in terminal_specs],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal_receipts)))
    terminal_passed = all(row.get("passed") is True for row in terminal_receipts)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_receipts)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=terminal_passed,
        critical=critical,
    )

    final = _base_artifact(
        manifest=manifest,
        reloaded=reloaded,
        validation_receipts=[*affected_receipts, *terminal_receipts],
        preconditions=preconditions,
        source_hashes=source_hashes,
        raw_dir=raw_dir,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        flagged_adversarial=(
            not affected["passed"] or bool(plan_errors) or not terminal_passed or critical
        ),
        fixture_artifact=False,
    )
    errors = independent_reduce_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", completed_units=0, path=output_path)
    atomic_json(root / output_path, final)
    progress(
        run_started, "write", "after_atomic_terminal", completed_units=1, status=final["status"]
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date and fresh-process replay mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cold-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI branch.
    """Run the annotated protocol or independently replay one candidate."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(
        REPO_ROOT,
        args.date,
        output_path=args.output,
        cache_root=args.cache_root,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
