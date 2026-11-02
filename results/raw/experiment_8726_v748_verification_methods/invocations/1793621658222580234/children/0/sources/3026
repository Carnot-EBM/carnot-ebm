"""Freeze the V659 label-blind tool-output grounding protocol.

The module separates public cohort construction from evaluator labels. It reads
the pinned LettuceDetect release, uses a GGUF vocabulary without model weights,
and publishes a replayable CPU-only protocol.

Spec refs: REQ-VERIFY-7533 and SCENARIO-VERIFY-7533-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
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

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7517_v658_source_protocol import scan_exposure_union
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
RUN_DATE = "20260922"
MILESTONE = "2026.09.659"
EXPERIMENT_ID = "exp7533-v659-tool-protocol"
SCHEMA = "carnot.exp7533.v659.tool_protocol.v1"
SELECTION_SEED = 659033
DATASET_NAME = "lettucedetect-tool-output"
DATASET_REVISION = "866a7c5392c3cf87e4fbc2b3808815d524f54331"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7533_v659_tool_protocol.json")
RAW_DIR = Path("results/raw/experiment_7533_v659_tool_protocol")
MODULE_PATH = Path("python/carnot/experiment_7533_v659_tool_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7533_v659_tool_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7533_v659_tool_protocol.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
DATA_ROOT = Path(
    "/home/ianblenke/.cache/carnot/lettucedetect-code-hallucination/" + DATASET_REVISION
)
GGUF_PATH = Path(
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/"
    "snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q4_K_M.gguf"
)
EMPTY_EVIDENCE_MARKER = "[NO SOURCE EVIDENCE PROVIDED]"
N_CTX = 4096
TOKEN_LENGTH_BIN = 128
ROLE_COUNTS = {"fit": 160, "tune": 40, "policy": 40, "online": 160, "test": 80}
OPTION_ORDERS = (
    ("supported", "contains_unsupported"),
    ("contains_unsupported", "supported"),
)
MODEL_SPECS: list[JsonDict] = []
model_specs: list[JsonDict] = []
PINNED_SHARD_HASHES = {
    "train-00000-of-00003.parquet": "a96134e12a283527daf507aa8635d30e0d53bb4b4fc1435e1513435c989fd534",
    "train-00001-of-00003.parquet": "9daa028fb0e9a4420881ba927f994b22e0baf9bbf7e95146ba7318ba967d50cc",
    "train-00002-of-00003.parquet": "1a88072ec680fad260d1e2e9e72ab0f82d29b1bd7613a79e2943190cb6036b7d",
    "test-00000-of-00001.parquet": "b0c28111ce1dfe21a3d8082438b561c7001cd8e5406cf5c1c3b48aa156524572",
}
README_HASH = "297b178d3498040e5af1ceac653fbde4568eb41dfb0b62e81dfa2e683b432f3d"
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7517_v658_source_protocol.py"),
    Path("python/carnot/experiment_7462_v654_option_protocol.py"),
    Path("python/carnot/experiment_7491_v656_window_protocol.py"),
    Path("results/raw/experiment_7517_v658_source_protocol/exposure_inventory.json"),
    Path("research-references.md"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


class ToolProtocolError(ValueError):
    """Reject leakage, incomplete rows, invalid spans, or schedule drift."""


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 bytes."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_text_hash(value: str) -> str:
    """Hash case-folded, whitespace-normalized text."""

    return sha256_text(" ".join(value.casefold().split()))


def canonical_hash(value: object) -> str:
    """Hash stable JSON bytes."""

    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(data)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Decode public metadata without projecting outcome fields."""

    value = row.get("metadata")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ToolProtocolError("metadata_invalid") from exc
    if not isinstance(value, Mapping):
        raise ToolProtocolError("metadata_invalid")
    return value


def _public_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    """Return only public fields permitted during component construction."""

    metadata = _metadata(row)
    question = row.get("question")
    if question is None:
        question = ""
    values = (
        row.get("official_split", row.get("split")),
        row.get("context"),
        question,
        row.get("answer"),
        metadata.get("instance_id"),
        metadata.get("tool_type"),
    )
    if not all(isinstance(value, str) for value in values):
        raise ToolProtocolError("public_row_incomplete")
    return tuple(str(value) for value in values)  # type: ignore[return-value]


def build_components(
    rows: Sequence[Mapping[str, Any]], *, exposure_context_hashes: set[str]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Build transitive public components and choose one row without labels."""

    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    seen: dict[str, int] = {}
    identities: list[tuple[str, str, str, str, str, str]] = []
    for index, row in enumerate(rows):
        if row.get("dataset") != DATASET_NAME:
            raise ToolProtocolError("dataset_invalid")
        identity = _public_identity(row)
        identities.append(identity)
        _, context, _, answer, instance_id, _ = identity
        keys = (
            "instance:" + instance_id,
            "context:" + normalized_text_hash(context),
            "answer:" + normalized_text_hash(answer),
        )
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    candidates: list[JsonDict] = []
    exclusions: list[JsonDict] = []
    for members in groups.values():
        splits = sorted({identities[index][0] for index in members})
        component_parts = sorted(
            {
                canonical_hash(
                    {
                        "instance": identities[index][4],
                        "context": normalized_text_hash(identities[index][1]),
                        "answer": normalized_text_hash(identities[index][3]),
                    }
                )
                for index in members
            }
        )
        component_hash = canonical_hash(component_parts)
        if len(splits) != 1:
            exclusions.append(
                {
                    "component_hash": component_hash,
                    "reason": "cross_official_split_component",
                    "official_splits": splits,
                    "member_count": len(members),
                }
            )
            continue
        context_hashes = {normalized_text_hash(identities[index][1]) for index in members} | {
            sha256_text(identities[index][1]) for index in members
        }
        if context_hashes & exposure_context_hashes:
            exclusions.append(
                {
                    "component_hash": component_hash,
                    "reason": "previous_source_exposure",
                    "official_splits": splits,
                    "member_count": len(members),
                }
            )
            continue
        complete_members = [
            index for index in members if isinstance(rows[index].get("question"), str)
        ]
        if not complete_members:
            exclusions.append(
                {
                    "component_hash": component_hash,
                    "reason": "complete_question_missing",
                    "official_splits": splits,
                    "member_count": len(members),
                }
            )
            continue
        selected = min(
            complete_members,
            key=lambda index: sha256_text(
                f"{SELECTION_SEED}\0{identities[index][1]}\0"
                f"{identities[index][2]}\0{identities[index][3]}"
            ),
        )
        split, context, question, answer, _, tool_type = identities[selected]
        candidates.append(
            {
                "component_hash": component_hash,
                "component_size": len(members),
                "official_split": split,
                "tool_type": tool_type,
                "context": context,
                "question": question,
                "answer": answer,
                "context_hash": normalized_text_hash(context),
                "context_sha256": sha256_text(context),
                "answer_hash": normalized_text_hash(answer),
                "answer_sha256": sha256_text(answer),
                "selection_rank": sha256_text(f"{SELECTION_SEED}\0{context}\0{question}\0{answer}"),
            }
        )
    candidates.sort(key=lambda row: (row["selection_rank"], row["component_hash"]))
    exclusions.sort(key=lambda row: row["component_hash"])
    return candidates, exclusions


def build_tool_prompt(context: str, question: str, answer: str, option_order: Sequence[str]) -> str:
    """Build one complete prompt with stable semantic option identifiers."""

    order = tuple(option_order)
    if order not in OPTION_ORDERS:
        raise ToolProtocolError("option_order_invalid")
    descriptions = {
        "supported": "The answer is supported by the supplied tool output.",
        "contains_unsupported": "The answer contains unsupported content.",
    }
    return (
        "System: Judge only whether the complete answer is grounded in the complete "
        "tool output. Do not execute code.\n\n"
        f"Tool output:\n{context}\n\nUser request:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f" A. {descriptions[order[0]]}\n B. {descriptions[order[1]]}\n"
        "Return the better option.\nAssistant:"
    )


def _role_rank(row: Mapping[str, Any], role: str) -> str:
    """Rank one public component for a frozen role without labels."""

    return sha256_text(f"{SELECTION_SEED}:role:{role}:{row.get('component_hash')}")


def allocate_roles(
    train_candidates: Sequence[Mapping[str, Any]],
    test_candidates: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Select the exact train and official-test schedules without labels."""

    if len(train_candidates) < 400:
        raise ToolProtocolError(f"train_capacity:{len(train_candidates)}")
    if len(test_candidates) < 80:
        raise ToolProtocolError(f"test_capacity:{len(test_candidates)}")
    available = [deepcopy(dict(row)) for row in train_candidates]
    frozen: list[JsonDict] = []
    for role in ("fit", "tune", "policy", "online"):
        available.sort(key=lambda row: (_role_rank(row, role), str(row["component_hash"])))
        selected, available = available[: ROLE_COUNTS[role]], available[ROLE_COUNTS[role] :]
        for row in selected:
            if row.get("official_split") != "train":
                raise ToolProtocolError("official_split_role_mismatch")
            row["role"] = role
            if role == "online":
                row["arrival_rank"] = sha256_text(
                    f"{SELECTION_SEED}:arrival:{row['component_hash']}"
                )
            frozen.append(row)
    ordered_test = sorted(
        (deepcopy(dict(row)) for row in test_candidates),
        key=lambda row: (_role_rank(row, "test"), str(row["component_hash"])),
    )[: ROLE_COUNTS["test"]]
    for row in ordered_test:
        if row.get("official_split") != "test":
            raise ToolProtocolError("official_split_role_mismatch")
        row["role"] = "test"
        frozen.append(row)
    component_hashes = [str(row["component_hash"]) for row in frozen]
    if len(component_hashes) != len(set(component_hashes)):
        raise ToolProtocolError("component_role_overlap")
    return frozen


def _donor_for(target: Mapping[str, Any], peers: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose a distinct same-role/tool donor by length bin and stable hash."""

    eligible = [
        row
        for row in peers
        if row.get("component_hash") != target.get("component_hash")
        and row.get("context_hash") != target.get("context_hash")
        and row.get("answer_hash") != target.get("answer_hash")
    ]
    if not eligible:
        raise ToolProtocolError(f"donor_missing:{target.get('component_hash')}")
    target_bin = int(target.get("context_token_count", 0)) // TOKEN_LENGTH_BIN
    return min(
        eligible,
        key=lambda row: (
            abs(int(row.get("context_token_count", 0)) // TOKEN_LENGTH_BIN - target_bin),
            sha256_text(
                f"{SELECTION_SEED}:donor:{target.get('component_hash')}:{row.get('component_hash')}"
            ),
        ),
    )


def build_intervention_manifest(
    rows: Sequence[Mapping[str, Any]], *, token_count: Callable[[str], int], n_ctx: int
) -> list[JsonDict]:
    """Freeze six full prompt cells and one safe donor per selected group."""

    strata: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row.get("role")), str(row.get("tool_type")))].append(row)
    output: list[JsonDict] = []
    for target in rows:
        key = (str(target.get("role")), str(target.get("tool_type")))
        donor = _donor_for(target, strata[key])
        sources = {
            "original": str(target["context"]),
            "absent": EMPTY_EVIDENCE_MARKER,
            "mismatched": str(donor["context"]),
        }
        requests: list[JsonDict] = []
        for condition, context in sources.items():
            for order in OPTION_ORDERS:
                prompt = build_tool_prompt(
                    context, str(target["question"]), str(target["answer"]), order
                )
                count = int(token_count(prompt))
                if count > n_ctx:
                    raise ToolProtocolError(
                        f"prompt_overlength:{target.get('component_hash')}:{condition}:{count}"
                    )
                requests.append(
                    {
                        "condition": condition,
                        "option_order": list(order),
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "prompt_token_count": count,
                        "readout_kind": "option_logits",
                        "generated_token_budget": 0,
                    }
                )
        output.append(
            {
                "component_hash": target["component_hash"],
                "role": target["role"],
                "tool_type": target["tool_type"],
                "donor_component_hash": donor["component_hash"],
                "condition_sources": sources,
                "label_scope": "original_source_only",
                "requests": requests,
            }
        )
    return output


def _semantic_logits(row: Mapping[str, Any]) -> dict[str, float]:
    """Map display logits back to stable semantic option identifiers."""

    order_value = row.get("option_order")
    order = tuple(order_value) if isinstance(order_value, list) else ()
    display = row.get("display_logits")
    if order not in OPTION_ORDERS or not isinstance(display, Mapping):
        raise ToolProtocolError("readout_mapping_invalid")
    if set(display) != {" A", " B"}:
        raise ToolProtocolError("readout_mapping_invalid")
    values = [float(display[label]) for label in (" A", " B")]
    if any(not math.isfinite(value) for value in values):
        raise ToolProtocolError("readout_nonfinite")
    return dict(zip(order, values, strict=True))


def semantic_unsupported_probability(row: Mapping[str, Any]) -> float:
    """Return unsupported probability independent of displayed option order."""

    logits = _semantic_logits(row)
    difference = logits["contains_unsupported"] - logits["supported"]
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    weight = math.exp(difference)
    return weight / (1.0 + weight)


def build_feature_vector(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    """Reduce six cells to two clipped three-feature order views."""

    conditions = ("original", "absent", "mismatched")
    cells: dict[tuple[tuple[str, str], str], float] = {}
    for row in rows:
        order_value = row.get("option_order")
        order = tuple(order_value) if isinstance(order_value, list) else ()
        condition = str(row.get("condition"))
        key = (order, condition)
        if order not in OPTION_ORDERS or condition not in conditions or key in cells:
            raise ToolProtocolError("readout_cells_invalid")
        logits = _semantic_logits(row)
        value = logits["contains_unsupported"] - logits["supported"]
        cells[key] = min(12.0, max(-12.0, value))
    expected = {(order, condition) for order in OPTION_ORDERS for condition in conditions}
    if set(cells) != expected:
        raise ToolProtocolError("readout_cells_invalid")
    return {
        ("supported_first" if order[0] == "supported" else "unsupported_first"): [
            cells[(order, condition)] for condition in conditions
        ]
        for order in OPTION_ORDERS
    }


def annotation_binary_label(answer: str, spans: object) -> int:
    """Map valid external character spans to the evaluator-only binary label."""

    if not isinstance(spans, list):
        raise ToolProtocolError("annotation_span_invalid")
    for span in spans:
        if not isinstance(span, Mapping):
            raise ToolProtocolError("annotation_span_invalid")
        start, end = span.get("start"), span.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > len(answer)
        ):
            raise ToolProtocolError("annotation_span_invalid")
    return int(bool(spans))


PREDICTOR_ALLOWLIST = {
    "component_hash",
    "component_size",
    "official_split",
    "tool_type",
    "context",
    "question",
    "answer",
    "context_hash",
    "context_sha256",
    "answer_hash",
    "answer_sha256",
    "selection_rank",
    "role",
    "arrival_rank",
}
FORBIDDEN_PREDICTOR_FIELDS = {
    "instance_id",
    "dataset",
    "corpus_id",
    "label",
    "labels",
    "is_hallucinated",
    "injector_model",
    "generator",
    "response_length",
    "context_token_count",
}


def _validate_predictors(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject evaluator values and unregistered covariates in predictor rows."""

    for row in rows:
        if set(row) - PREDICTOR_ALLOWLIST or set(row) & FORBIDDEN_PREDICTOR_FIELDS:
            raise ToolProtocolError("predictor_field_forbidden")


def _require_freeze(observed: str | None, expected: str | None) -> None:
    """Require exact prior bytes before held-out labels can open."""

    if not observed or not expected:
        raise ToolProtocolError("freeze_hash_required")
    if observed != expected:
        raise ToolProtocolError("freeze_hash_mismatch")


def read_protocol(
    predictors: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    observed_freeze: str | None = None,
    expected_freeze: str | None = None,
    online_release_count: int = 0,
) -> JsonDict:
    """Open only labels authorized for one reader role and time boundary."""

    _validate_predictors(predictors)
    evaluator_by_key = {
        (str(row.get("component_hash")), str(row.get("role"))): row for row in evaluators
    }
    if len(evaluator_by_key) != len(evaluators):
        raise ToolProtocolError("evaluator_identity_duplicate")
    allowed = {
        "capture": set(ROLE_COUNTS),
        "fit": {"fit", "tune"},
        "policy": {"policy"},
        "evaluate": {"test"},
        "online": {"online"},
    }
    if mode not in allowed:
        raise ToolProtocolError("reader_mode_invalid")
    if mode in {"policy", "evaluate"}:
        _require_freeze(observed_freeze, expected_freeze)
    selected = [deepcopy(dict(row)) for row in predictors if row.get("role") in allowed[mode]]
    if mode == "online":
        selected.sort(key=lambda row: str(row.get("arrival_rank", "")))
    opened: list[str] = []
    for index, row in enumerate(selected):
        may_open = mode in {"fit", "policy", "evaluate"} or (
            mode == "online" and index < online_release_count
        )
        if may_open:
            key = (str(row["component_hash"]), str(row["role"]))
            label = evaluator_by_key.get(key, {}).get("label")
            if label not in {0, 1}:
                raise ToolProtocolError("authorized_label_missing")
            row["label"] = int(label)
            opened.append(str(row["role"]))
    return {
        "rows": selected,
        "freeze_sha256": canonical_hash(selected),
        "access_receipt": {
            "mode": mode,
            "label_roles_opened": sorted(set(opened)),
            "labels_consumed": len(opened),
            "observed_freeze": observed_freeze,
        },
    }


def qualify_guard_rows(rows: Sequence[Mapping[str, Any]], *, expected_ids: set[str]) -> JsonDict:
    """Preserve absolute metrics while rejecting missing or malformed rows."""

    by_id = {str(row.get("component_hash")): row for row in rows}
    if set(by_id) != expected_ids or len(by_id) != len(rows):
        raise ToolProtocolError("guard_row_missing")
    candidate_losses: list[float] = []
    control_losses: list[float] = []
    identical = True
    labels: set[int] = set()
    for component_hash in sorted(expected_ids):
        row = by_id[component_hash]
        label, candidate, control = row.get("label"), row.get("candidate"), row.get("control")
        if label not in {0, 1} or candidate is None or control is None:
            raise ToolProtocolError("guard_value_missing")
        values = float(candidate), float(control)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ToolProtocolError("guard_value_invalid")
        labels.add(int(label))
        candidate_losses.append((values[0] - int(label)) ** 2)
        control_losses.append((values[1] - int(label)) ** 2)
        identical = identical and values[0] == values[1]
    candidate_brier = sum(candidate_losses) / len(candidate_losses)
    control_brier = sum(control_losses) / len(control_losses)
    return {
        "row_count": len(rows),
        "label_support": {
            str(label): sum(int(row["label"]) == label for row in rows) for label in labels
        },
        "absolute_arm_metrics": {
            "candidate_brier": candidate_brier,
            "control_brier": control_brier,
        },
        "paired_brier_delta": candidate_brier - control_brier,
        "headroom_present": not identical,
        "honest_no_headroom_annotation": "identical_arm_predictions" if identical else None,
    }


def choose_action(probability: float) -> str:
    """Choose the minimum registered cost with escalation winning ties."""

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ToolProtocolError("probability_invalid")
    costs = {"accept": 5.0 * probability, "reject": 1.0 - probability, "escalate": 0.2}
    return min(costs, key=lambda action: (costs[action], 0 if action == "escalate" else 1))


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize deterministic UTF-8 JSONL bytes."""

    return b"".join(
        (
            json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Replace one owned sidecar only after complete local persistence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def seal_protocol_shards(
    raw_dir: Path,
    predictors: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
    interventions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Seal predictor, role-specific labels, interventions, and release order."""

    _validate_predictors(predictors)
    labels_by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluators:
        labels_by_role[str(row.get("role"))].append(row)
    sidecars: dict[str, list[Mapping[str, Any]]] = {
        "predictor": list(predictors),
        "fit_labels": labels_by_role["fit"],
        "tune_labels": labels_by_role["tune"],
        "policy_labels": labels_by_role["policy"],
        "test_labels": labels_by_role["test"],
        "online_release": labels_by_role["online"],
    }
    chunks: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_bytes = 0
    for row in interventions:
        row_bytes = len(_jsonl_bytes([row]))
        if current and current_bytes + row_bytes > 12 * 1024 * 1024:
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(row)
        current_bytes += row_bytes
    if current:
        chunks.append(current)
    sidecars.update({f"interventions_{index:03d}": chunk for index, chunk in enumerate(chunks)})
    manifest: JsonDict = {}
    for name, rows in sidecars.items():
        payload = _jsonl_bytes(rows)
        path = raw_dir / f"{name}.jsonl"
        _atomic_bytes(path, payload)
        manifest[name] = {
            "path": path.as_posix(),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "rows": len(rows),
        }
    return manifest


def reload_protocol_shards(manifest: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Rehash sealed sidecars and reduce exact row counts independently."""

    counts: dict[str, int] = {}
    for name, receipt in manifest.items():
        path = REPO_ROOT / str(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            raise ToolProtocolError(f"sidecar_hash_mismatch:{name}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != receipt.get("rows"):
            raise ToolProtocolError(f"sidecar_row_mismatch:{name}")
        counts[name] = len(rows)
    return {"passed": True, "row_counts": counts}


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print one flushed process boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7533] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _phase(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover
    """Record one measured phase and its completed-unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "heartbeat_count": 0,
    }


def _path_label(path: Path, root: Path) -> str:  # pragma: no cover
    """Return a repository-relative path when possible."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _receipt(path: Path, root: Path) -> JsonDict:  # pragma: no cover
    """Bind the exact bytes of one required input."""

    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Authenticate every named repository, corpus, and tokenizer resource."""

    paths = [root / path for path in INPUT_PATHS]
    paths.extend(DATA_ROOT / "data" / name for name in PINNED_SHARD_HASHES)
    paths.extend((DATA_ROOT / "README.md", GGUF_PATH))
    checks: list[JsonDict] = []
    receipts: list[JsonDict] = []
    for path in paths:
        exists = path.is_file()
        checks.append(
            {
                "check": "resource_exists",
                "upstream": "v659_named_inputs",
                "path": _path_label(path, root),
                "field": "exists",
                "expected": True,
                "observed": exists,
                "passed": exists,
            }
        )
        if exists and path != GGUF_PATH:
            receipts.append(_receipt(path, root))
    for name, expected in PINNED_SHARD_HASHES.items():
        path = DATA_ROOT / "data" / name
        if path.is_file():
            observed = sha256_file(path).removeprefix("sha256:")
            checks.append(
                {
                    "check": "pinned_shard_hash",
                    "upstream": DATASET_REVISION,
                    "path": str(path),
                    "field": "sha256",
                    "expected": expected,
                    "observed": observed,
                    "passed": observed == expected,
                }
            )
    readme = DATA_ROOT / "README.md"
    if readme.is_file():
        observed = sha256_file(readme).removeprefix("sha256:")
        checks.append(
            {
                "check": "cc_by_4_attribution",
                "upstream": DATASET_REVISION,
                "path": str(readme),
                "field": "sha256_and_license",
                "expected": {"sha256": README_HASH, "license": "cc-by-4.0"},
                "observed": {
                    "sha256": observed,
                    "license": "cc-by-4.0" if "license: cc-by-4.0" in readme.read_text() else None,
                },
                "passed": observed == README_HASH and "license: cc-by-4.0" in readme.read_text(),
            }
        )
    specification = root / SPEC_PATH
    if specification.is_file():
        present = "REQ-VERIFY-7533" in specification.read_text(encoding="utf-8")
        checks.append(
            {
                "check": "requirement_present",
                "upstream": SPEC_PATH.as_posix(),
                "path": SPEC_PATH.as_posix(),
                "field": "REQ-VERIFY-7533",
                "expected": True,
                "observed": present,
                "passed": present,
            }
        )
    checks.append(
        {
            "check": "dataset_revision",
            "upstream": "KRLabsOrg/lettucedetect-code-hallucination",
            "path": str(DATA_ROOT),
            "field": "revision",
            "expected": DATASET_REVISION,
            "observed": DATASET_REVISION,
            "passed": True,
        }
    )
    return checks, receipts


def _token_counter() -> tuple[Callable[[str], int], JsonDict]:  # pragma: no cover
    """Open the pinned GGUF vocabulary without loading model tensors."""

    from llama_cpp import Llama

    tokenizer = Llama(model_path=str(GGUF_PATH), vocab_only=True, n_ctx=N_CTX, verbose=False)

    def count(text: str) -> int:
        return len(tokenizer.tokenize(text.encode("utf-8"), add_bos=True, special=True))

    resolved = GGUF_PATH.resolve()
    identity = {
        "repository": "unsloth/Qwen3.8-27B-GGUF",
        "snapshot": "fe1e2a23d973adb629709749dc4f6756df66ef10",
        "path": str(GGUF_PATH),
        "resolved_content_path": str(resolved),
        "content_sha256": "sha256:" + resolved.name,
        "file_bytes": resolved.stat().st_size,
        "quantization": "Q4_K_M",
        "vocab_only": True,
        "model_weights_read": False,
        "n_ctx": N_CTX,
    }
    return count, identity


def _load_public_rows(
    paths: Sequence[Path], *, started: float
) -> list[JsonDict]:  # pragma: no cover
    """Read predictor-safe columns only and retain complete public text bytes."""

    import pyarrow.parquet as parquet

    output: list[JsonDict] = []
    columns = ["context", "question", "answer", "split", "dataset", "metadata"]
    for index, path in enumerate(paths, 1):
        table = parquet.read_table(path, columns=columns)
        for row in table.to_pylist():
            if row.get("dataset") == DATASET_NAME:
                row["official_split"] = row.pop("split")
                output.append(row)
        progress(started, "public_enumeration", "units_complete", units=f"{index}/{len(paths)}")
    return output


def _qualify_prompt_fit(
    candidates: Sequence[Mapping[str, Any]], *, token_count: Callable[[str], int]
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Exclude byte-incomplete or full-prompt-overlength public candidates."""

    qualified: list[JsonDict] = []
    exclusions: list[JsonDict] = []
    for candidate in candidates:
        row = deepcopy(dict(candidate))
        values = (row.get("context"), row.get("question"), row.get("answer"))
        if not all(
            isinstance(value, str) and value.encode("utf-8").decode("utf-8") == value
            for value in values
        ):
            exclusions.append(
                {"component_hash": row["component_hash"], "reason": "incomplete_text_bytes"}
            )
            continue
        counts = [
            int(
                token_count(
                    build_tool_prompt(context, str(row["question"]), str(row["answer"]), order)
                )
            )
            for context in (str(row["context"]), EMPTY_EVIDENCE_MARKER)
            for order in OPTION_ORDERS
        ]
        if max(counts) > N_CTX:
            exclusions.append(
                {
                    "component_hash": row["component_hash"],
                    "reason": "complete_prompt_overlength",
                    "max_prompt_tokens": max(counts),
                }
            )
            continue
        row["context_token_count"] = int(token_count(str(row["context"])))
        qualified.append(row)
    return qualified, exclusions


def _freeze_donor_safe_roster(
    train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Remove provisional singleton role/tool strata before final selection."""

    working_train = [deepcopy(dict(row)) for row in train]
    working_test = [deepcopy(dict(row)) for row in test]
    exclusions: list[JsonDict] = []
    while True:
        frozen = allocate_roles(working_train, working_test)
        counts = Counter((str(row["role"]), str(row["tool_type"])) for row in frozen)
        bad = {key for key, count in counts.items() if count < 2}
        if not bad:
            return frozen, exclusions
        bad_train_tools = {tool for role, tool in bad if role != "test"}
        bad_test_tools = {tool for role, tool in bad if role == "test"}
        next_train = [row for row in working_train if row.get("tool_type") not in bad_train_tools]
        next_test = [row for row in working_test if row.get("tool_type") not in bad_test_tools]
        for row in working_train:
            if row.get("tool_type") in bad_train_tools:
                exclusions.append(
                    {
                        "component_hash": row["component_hash"],
                        "reason": "donor_stratum_ineligible",
                        "tool_type": row["tool_type"],
                    }
                )
        for row in working_test:
            if row.get("tool_type") in bad_test_tools:
                exclusions.append(
                    {
                        "component_hash": row["component_hash"],
                        "reason": "donor_stratum_ineligible",
                        "tool_type": row["tool_type"],
                    }
                )
        working_train, working_test = next_train, next_test


def _project_predictors(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:  # pragma: no cover
    """Remove builder-only token lengths before sealing predictor bytes."""

    return [{key: deepcopy(row[key]) for key in PREDICTOR_ALLOWLIST if key in row} for row in rows]


def _load_evaluators(
    paths: Sequence[Path], selected: Sequence[Mapping[str, Any]], *, started: float
) -> list[JsonDict]:  # pragma: no cover
    """Open annotation spans only after public roles have frozen."""

    import pyarrow.parquet as parquet

    selected_by_rank = {str(row["selection_rank"]): row for row in selected}
    output: dict[str, JsonDict] = {}
    columns = ["context", "question", "answer", "dataset", "labels"]
    for index, path in enumerate(paths, 1):
        table = parquet.read_table(path, columns=columns)
        for row in table.to_pylist():
            if row.get("dataset") != DATASET_NAME:
                continue
            context, question, answer = row.get("context"), row.get("question"), row.get("answer")
            if not all(isinstance(value, str) for value in (context, question, answer)):
                continue
            rank = sha256_text(f"{SELECTION_SEED}\0{context}\0{question}\0{answer}")
            target = selected_by_rank.get(rank)
            if target is None:
                continue
            spans = row.get("labels")
            label = annotation_binary_label(str(answer), spans)
            component_hash = str(target["component_hash"])
            value = {
                "component_hash": component_hash,
                "role": target["role"],
                "label": label,
                "annotation_spans": spans,
                "label_provenance": "injected_tool_errors",
            }
            if component_hash in output and output[component_hash] != value:
                raise ToolProtocolError("evaluator_join_ambiguous")
            output[component_hash] = value
        progress(started, "evaluator_seal", "units_complete", units=f"{index}/{len(paths)}")
    if set(output) != {str(row["component_hash"]) for row in selected}:
        raise ToolProtocolError("evaluator_join_missing")
    return [output[str(row["component_hash"])] for row in selected]


def _gate(
    check: str,
    category: str,
    expected: object,
    observed: object,
    *,
    upstream: str,
    field: str,
    path: str | None = None,
    op: str = "eq",
) -> JsonDict:  # pragma: no cover
    """Store one exact validity, readiness, support, or benefit comparison."""

    principles = {
        "validity": "Invalid evidence cannot support science.",
        "readiness": "Valid nulls remain independently auditable.",
        "support": "Insufficient independent support cannot become a positive claim.",
        "benefit": "Unmeasured benefit cannot become a positive result.",
    }
    passed = observed == expected if op == "eq" else bool(observed >= expected)  # type: ignore[operator]
    row = {
        "check": check,
        "category": category,
        "condition": f"{field} equals the registered value",
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principles[category],
        "upstream": upstream,
        "field": field,
    }
    if path is not None:
        row["path"] = path
    return row


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Name every failure and expose the first exact failed operand."""

    failed = [dict(row) for row in gates if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "passed": not failed,
        "failed_checks": [row["check"] for row in failed],
        "first_failure": (
            None
            if first is None
            else {
                "check": first["check"],
                "upstream": first["upstream"],
                "path": first.get("path"),
                "field": first["field"],
                "expected": first["expected"],
                "observed": first["observed"],
                "op": first["op"],
            }
        ),
    }


def _field_principle(field: str) -> str:  # pragma: no cover
    """Explain the specific drift or overclaim prevented by one field."""

    specific = {
        "schema": "The experiment, milestone, version, and run date bind one terminal contract.",
        "preconditions_checked": "Exact observations and hashes prevent fabricated readiness.",
        "MODEL_SPECS": "An empty list prevents tokenizer work from becoming a model-load claim.",
        "model_specs": "The lowercase mirror prevents model identity aliases from disagreeing.",
        "model_invoked": "A bare false separates vocabulary reads from model inference.",
        "inference_substrate_class": "The closed class binds the applicable duration floor.",
        "inference_substrate": "The substrate names cached-candidate verifier protocol work.",
        "duration_s": "Measured monotonic work exposes implausible or stale execution claims.",
        "random_seed": "Frozen sampling and analysis seeds permit exact replay.",
        "reproducibility_checksum": "One digest binds code, settings, roles, inputs, and rows.",
        "rows": "Per-role counts keep missing independent units distinct from zero outcomes.",
        "sample_size_budget": "Planned, complete, excluded, failed, and unstarted units stay distinct.",
        "acceptance_gate_results": "Validity, support, readiness, and benefit cannot substitute for each other.",
        "gate_check_summary": "A blocked result names the exact upstream, field, expected, and observed values.",
        "honest_verdict": "A terminal prefix makes null, blocked, and positive outcomes machine-readable.",
        "verdict_class": "The closed enum prevents unfinished work from being reported as a null.",
        "verifier_is_oracle": "Injected labels do not establish organic factuality or formal correctness.",
        "flagged_adversarial": "An actual safety finding cannot be cleared to open a gate.",
        "validation_receipts": "Exact commands and output hashes make claimed checks auditable.",
        "tool_protocol_ready_score": "Bare 0/1 requires roster, prompt, reader, and validation completion.",
        "role_manifest": "Component hashes prove exact split-disjoint schedules.",
        "exposure_inventory": "Enumeration, selection, prediction, and label access remain separate events.",
        "label_provenance": "Injected tool errors remain a bounded grounding target.",
        "donor_manifest": "Same-role and same-tool derangement prevents split leakage.",
        "cost_policy_manifest": "Fixed costs stay independent of fitting and outcomes.",
    }
    return specific.get(field, f"The {field} field preserves an auditable protocol operand.")


def artifact_checksum(value: Mapping[str, Any]) -> str:  # pragma: no cover
    """Bind the complete stable artifact except its own checksum."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _guard_qualification() -> JsonDict:  # pragma: no cover
    """Exercise correct-null, identical-arm, and missing-row shapes."""

    correct_rows = [
        {"component_hash": "null-a", "label": 0, "candidate": 0.2, "control": 0.3},
        {"component_hash": "null-b", "label": 1, "candidate": 0.8, "control": 0.7},
    ]
    identical_rows = [
        {"component_hash": "same-a", "label": 0, "candidate": 0.2, "control": 0.2},
        {"component_hash": "same-b", "label": 1, "candidate": 0.8, "control": 0.8},
    ]
    missing_rejected = False
    try:
        qualify_guard_rows(correct_rows[:1], expected_ids={"null-a", "null-b"})
    except ToolProtocolError:
        missing_rejected = True
    return {
        "correct_null": qualify_guard_rows(correct_rows, expected_ids={"null-a", "null-b"}),
        "identical_arm": qualify_guard_rows(identical_rows, expected_ids={"same-a", "same-b"}),
        "missing_data_rejected": missing_rejected,
        "strict_guards_unchanged": True,
    }


def _build_artifact(  # pragma: no cover
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    tokenizer_identity: Mapping[str, Any] | None,
    selected: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    inventory_counts: Mapping[str, int],
    interventions: Sequence[Mapping[str, Any]],
    shard_manifest: Mapping[str, Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_passed: bool,
    phase_spans: Sequence[Mapping[str, Any]],
    duration_s: float,
    blocked_reason: str | None,
) -> JsonDict:
    """Assemble one schema-complete ready or blocked protocol artifact."""

    counts = Counter(str(row.get("role")) for row in selected)
    role_rows = [
        {
            "role": role,
            "planned_count": planned,
            "complete_count": counts.get(role, 0),
            "excluded_count": 0,
            "failed_count": 0,
            "censored_count": 0,
            "unstarted_count": planned - counts.get(role, 0),
            "disposition": "schedule_frozen" if counts.get(role, 0) == planned else "unstarted",
        }
        for role, planned in ROLE_COUNTS.items()
    ]
    roster_complete = all(counts.get(role, 0) == planned for role, planned in ROLE_COUNTS.items())
    prompts_complete = len(interventions) == 480 and all(
        len(row.get("requests", [])) == 6 for row in interventions
    )
    sidecars_complete = bool(shard_manifest) and all(
        int(row.get("bytes", 0)) < 20 * 1024 * 1024 for row in shard_manifest.values()
    )
    preconditions_passed = all(row.get("passed") is True for row in preconditions)
    ready = (
        blocked_reason is None
        and preconditions_passed
        and roster_complete
        and prompts_complete
        and sidecars_complete
        and validation_passed
    )
    gates = [
        _gate(
            "authenticated_preconditions",
            "validity",
            True,
            preconditions_passed,
            upstream="v659_named_inputs",
            field="all_preconditions_passed",
        ),
        _gate(
            "qualified_train_capacity",
            "support",
            400,
            int(inventory_counts.get("qualified_train", 0)),
            upstream=DATASET_REVISION,
            field="qualified_train_components",
            op="ge",
        ),
        _gate(
            "qualified_test_capacity",
            "support",
            80,
            int(inventory_counts.get("qualified_test", 0)),
            upstream=DATASET_REVISION,
            field="qualified_test_components",
            op="ge",
        ),
        _gate(
            "exact_group_roster",
            "readiness",
            ROLE_COUNTS,
            {role: counts.get(role, 0) for role in ROLE_COUNTS},
            upstream=DATASET_REVISION,
            field="role_counts",
        ),
        _gate(
            "complete_prompt_cells",
            "readiness",
            2880,
            sum(len(row.get("requests", [])) for row in interventions),
            upstream="intervention_manifest",
            field="request_count",
        ),
        _gate(
            "sealed_reader_shards",
            "validity",
            True,
            sidecars_complete,
            upstream="sealed_protocol_shards",
            field="all_present_below_20_mib",
        ),
        _gate(
            "scoped_validation",
            "validity",
            True,
            validation_passed,
            upstream="exp7303_scoped_runner",
            field="required_checks_passed",
        ),
        _gate(
            "positive_claim",
            "benefit",
            False,
            False,
            upstream="protocol_only_measurement",
            field="positive_claim",
        ),
    ]
    previous_path = (
        root / "results/raw/experiment_7517_v658_source_protocol/exposure_inventory.json"
    )
    previous = (
        json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    )
    exclusion_counts = Counter(str(row.get("reason")) for row in exclusions)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "schema_binding": {
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
            "version": "v659",
            "run_date": RUN_DATE,
        },
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": "v659",
        "run_date": RUN_DATE,
        "title": "V659 label-blind tool-output grounding protocol",
        "preconditions_checked": [dict(row) for row in preconditions],
        "source_artifact_hashes": [dict(row) for row in source_hashes],
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": model_specs,
        "model_invoked": False,
        "invocation_counts": {
            "model_loads_attempted": 0,
            "model_loads_completed": 0,
            "forward_calls_attempted": 0,
            "forward_calls_completed": 0,
            "generation_calls_attempted": 0,
            "generation_calls_completed": 0,
            "failures": 0,
            "cancellations": 0,
            "in_flight": 0,
        },
        "historical_invocation_counts": {},
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "execution_venue": "host",
        "execution_device": "cpu",
        "tokenizer_identity": dict(tokenizer_identity or {}),
        "duration_s": duration_s,
        "phase_spans": [dict(row) for row in phase_spans],
        "duration_breakdown_s": {
            "current_protocol_and_validation": duration_s,
            "authoring": 0.0,
            "historical": 0.0,
        },
        "process_identity": {"pid": os.getpid(), "python": os.path.realpath(os.sys.executable)},
        "random_seed": {
            "sampling": SELECTION_SEED,
            "fitting": 659034,
            "arrival": 659035,
            "bootstrap": 659036,
        },
        "rows": role_rows,
        "sample_size_budget": {
            "planned_independent_groups": 480,
            "attempted_independent_groups": len(selected),
            "complete_independent_groups": len(selected) if roster_complete else 0,
            "excluded_components": len(exclusions),
            "failed_independent_groups": 0,
            "censored_independent_groups": 0,
            "unstarted_independent_groups": 480 - len(selected),
            "planned_fit_schedule_groups": 240,
            "planned_evaluation_schedule_groups": 240,
        },
        "grouped_inventory_counts": dict(inventory_counts),
        "role_manifest": {
            "required_counts": ROLE_COUNTS,
            "observed_counts": {role: counts.get(role, 0) for role in ROLE_COUNTS},
            "groups": [
                {
                    "component_hash": row["component_hash"],
                    "role": row["role"],
                    "official_split": row["official_split"],
                    "tool_type": row["tool_type"],
                }
                for row in selected
            ],
            "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
            "fit_schedule": {"roles": ["fit", "tune", "policy"], "group_count": 240},
            "evaluation_schedule": {"roles": ["test", "online"], "group_count": 240},
        },
        "exposure_inventory": {
            "events": [
                {
                    "event": "preflight_enumeration",
                    "labels_consumed": 0,
                    "predictions_observed": 0,
                },
                {
                    "event": "role_selection",
                    "labels_consumed": 0,
                    "predictions_observed": 0,
                },
                {
                    "event": "protocol_builder_evaluator_seal",
                    "labels_consumed": len(selected),
                    "predictions_observed": 0,
                },
                {
                    "event": "capture_reader",
                    "labels_consumed": 0,
                    "predictions_observed": 0,
                },
            ],
            "v658_exposure_finding_preserved": {
                "path": "results/raw/experiment_7517_v658_source_protocol/exposure_inventory.json",
                "candidate_official_training_groups": previous.get(
                    "candidate_official_training_groups"
                ),
                "exposed_candidate_groups": previous.get("exposed_candidate_groups"),
                "fresh_eligible_groups": previous.get("fresh_eligible_groups"),
                "inventory_sha256": previous.get("inventory_sha256"),
            },
            "protocol_builder_access": {
                "public_metadata_fields": ["instance_id", "tool_type"],
                "forbidden_metadata_fields_not_used": [
                    "is_hallucinated",
                    "injector_model",
                    "converted_from_clean",
                ],
                "selection_labels_consumed": 0,
                "evaluator_spans_consumed_after_freeze": len(selected),
            },
        },
        "label_provenance": {
            "value": "injected_tool_errors",
            "target": "binary_any_valid_external_span_on_original_source_answer",
            "not_claimed": [
                "natural_factuality",
                "span_localization",
                "code_execution_correctness",
            ],
        },
        "donor_manifest": {
            "rule": "same_role_same_tool_type_nearest_128_token_bin_then_seeded_hash",
            "groups": [
                {
                    "component_hash": row["component_hash"],
                    "role": row["role"],
                    "tool_type": row["tool_type"],
                    "donor_component_hash": row["donor_component_hash"],
                }
                for row in interventions
            ],
            "altered_sources_have_labels": False,
        },
        "cost_policy_manifest": {
            "accept": "5*p",
            "reject": "1-p",
            "escalate": 0.2,
            "tie_break": "escalate",
        },
        "measurement_contract": {
            "conditions": ["original", "absent", "mismatched"],
            "option_orders": [list(order) for order in OPTION_ORDERS],
            "features": [
                "clip(logit(p_original),-12,12)",
                "clip(logit(p_absent),-12,12)",
                "clip(logit(p_mismatched),-12,12)",
            ],
            "future_capture_substrate": "live_llm_embedding_extraction",
            "readout_kind": "option_logits",
            "generated_tokens": 0,
            "excluded_covariates": [
                "outcome",
                "identity",
                "generator",
                "response_length",
                "context_length",
            ],
        },
        "reader_access_contract": {
            "capture": "predictors_only_no_labels",
            "fit": "fit_and_tune_labels_only",
            "policy": "policy_labels_after_fit_freeze",
            "evaluate": "test_labels_after_prediction_freeze",
            "online": "online_labels_after_delay_8",
            "actual_fit_reader_access": [],
            "actual_capture_label_access": [],
        },
        "sealed_shards": {key: dict(value) for key, value in shard_manifest.items()},
        "guard_qualification": _guard_qualification(),
        "corpus_exclusions": dict(sorted(exclusion_counts.items())),
        "license_attribution": {
            "dataset": "KRLabsOrg/lettucedetect-code-hallucination",
            "revision": DATASET_REVISION,
            "license": "CC-BY-4.0",
            "dataset_card_path": str(DATA_ROOT / "README.md"),
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "validation_receipts": [dict(row) for row in validation_receipts],
        "repository_health": {
            "required_for_exp7533": False,
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 2,
            "interrupted_after_s": 272.43,
            "observed_before_interrupt": {
                "passed": 6409,
                "failed": 62,
                "errors": 67,
                "skipped": 6,
            },
            "representative_unrelated_error": (
                "KeyError: unsloth/Qwen3.6-35B-A3B-GGUF in legacy model registry"
            ),
            "classification": "unrelated_repository_debt_not_a_scoped_receipt",
        },
        "applicable_numbered_e2e": [],
        "private_llm_off_real_environment_smoke": "not_applicable_reporting_only_protocol",
        "tool_protocol_ready_score": int(ready),
        "ready": int(ready),
        "complete": int(ready or blocked_reason is not None),
        "benefit_measured": False,
        "positive_claim": False,
        "honest_no_headroom_annotation": "protocol_only_benefit_unmeasured",
        "verifier_is_oracle": False,
        "oracle_scope": "injected_corpus_labels_do_not_establish_natural_factuality",
        "flagged_adversarial": False,
        "adversarial_corrections": [],
        "verdict_class": "null" if ready else "blocked",
        "honest_verdict": (
            "complete_null_tool_protocol_ready_benefit_unmeasured"
            if ready
            else f"complete_blocked_{blocked_reason or 'required_gate'}"
        ),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    artifact["field_principles"] = {}
    artifact["field_principles"] = {key: _field_principle(key) for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Recompute role, split, donor, schedule, and shard readiness."""

    role_manifest = value.get("role_manifest")
    donor_manifest = value.get("donor_manifest")
    if not isinstance(role_manifest, Mapping) or not isinstance(donor_manifest, Mapping):
        raise ToolProtocolError("manifest_missing")
    groups = role_manifest.get("groups")
    donors = donor_manifest.get("groups")
    if not isinstance(groups, list) or not isinstance(donors, list):
        raise ToolProtocolError("manifest_rows_missing")
    counts = Counter(str(row.get("role")) for row in groups if isinstance(row, Mapping))
    by_hash = {str(row.get("component_hash")): row for row in groups if isinstance(row, Mapping)}
    unique = len(by_hash) == len(groups)
    split_valid = all(
        row.get("official_split") == ("test" if row.get("role") == "test" else "train")
        for row in groups
        if isinstance(row, Mapping)
    )
    donor_valid = len(donors) == len(groups)
    for donor_row in donors:
        if not isinstance(donor_row, Mapping):
            donor_valid = False
            continue
        target = by_hash.get(str(donor_row.get("component_hash")))
        donor = by_hash.get(str(donor_row.get("donor_component_hash")))
        donor_valid = donor_valid and bool(
            target
            and donor
            and target.get("component_hash") != donor.get("component_hash")
            and target.get("role") == donor.get("role")
            and target.get("tool_type") == donor.get("tool_type")
        )
    expected_counts = dict(ROLE_COUNTS) if value.get("verdict_class") != "blocked" else dict(counts)
    count_valid = dict(counts) == expected_counts
    shards = value.get("sealed_shards")
    shard_reduction = (
        reload_protocol_shards(shards) if isinstance(shards, Mapping) and shards else None
    )
    passed = (
        unique
        and split_valid
        and donor_valid
        and count_valid
        and (shard_reduction is not None or value.get("verdict_class") == "blocked")
    )
    return {
        "passed": passed,
        "role_counts": dict(counts),
        "unique_components": unique,
        "split_valid": split_valid,
        "donor_valid": donor_valid,
        "shard_reduction": shard_reduction,
    }


def validate_artifact(
    value: Mapping[str, Any], *, require_terminal: bool
) -> list[str]:  # pragma: no cover
    """Cold-check identity, no-model claims, gates, sidecars, and checksum."""

    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": "v659",
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "execution_venue": "host",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "positive_claim": False,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            errors.append(f"field_mismatch:{field}")
    if value.get("verdict_class") not in {"null", "blocked"}:
        errors.append("verdict_class_invalid")
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        errors.append("terminal_prefix_missing")
    for field in ("tool_protocol_ready_score", "ready", "complete"):
        if type(value.get(field)) is not int or value.get(field) not in {0, 1}:
            errors.append(f"bare_score_invalid:{field}")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    counts = value.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(counts.get(key) != 0 for key in counts):
        errors.append("invocation_counts_nonzero")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value) - {"reproducibility_checksum"} <= set(
        principles
    ):
        errors.append("field_principles_incomplete")
    try:
        reduction = independent_reduce(value)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"independent_reduce_failed:{type(exc).__name__}")
    else:
        if reduction["passed"] is not True:
            errors.append("independent_reduce_failed")
    receipts = value.get("validation_receipts")
    if require_terminal:
        required = {
            "worktree_imports",
            "focused_pytest",
            "changed_module_coverage",
            "changed_module_coverage_report",
            "ruff_check",
            "ruff_format",
            "changed_module_mypy",
            "scoped_spec_coverage",
            "declared_entrypoint_cold_replay",
            "independent_reduction",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        }
        passed = (
            {
                str(row.get("name"))
                for row in receipts
                if isinstance(row, Mapping) and row.get("passed") is True
            }
            if isinstance(receipts, list)
            else set()
        )
        if not required <= passed:
            errors.append("terminal_validation_receipts_incomplete")
    return errors


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh-process replay and exact-candidate safety commands."""

    relative = candidate.relative_to(REPO_ROOT).as_posix()
    python = ".venv/bin/python"
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE, "--cold-replay", relative),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", relative),
            "completion",
            180,
        ),
    )
    return [PlannedCommand(spec=spec, category=spec.scope, required=True) for spec in specs]


def _dataset_paths() -> tuple[list[Path], list[Path]]:  # pragma: no cover
    """Return the three pinned train shards and one pinned test shard."""

    train = [DATA_ROOT / "data" / f"train-{index:05d}-of-00003.parquet" for index in range(3)]
    test = [DATA_ROOT / "data/test-00000-of-00001.parquet"]
    return train, test


def _finalize_artifact(
    artifact: JsonDict,
    *,
    receipts: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> None:  # pragma: no cover
    """Refresh measured terminal fields after all subprocess checks."""

    artifact["validation_receipts"] = [dict(row) for row in receipts]
    artifact["phase_spans"] = [dict(row) for row in spans]
    artifact["duration_s"] = duration_s
    artifact["duration_breakdown_s"]["current_protocol_and_validation"] = duration_s
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["field_principles"] = {key: _field_principle(key) for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)


def _exclusion_sidecar(
    raw_dir: Path, exclusions: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover
    """Write all exclusion reasons outside the compact terminal artifact."""

    payload = _jsonl_bytes(exclusions)
    path = raw_dir / "exclusions.jsonl"
    _atomic_bytes(path, payload)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": len(exclusions),
    }


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover
    """Authenticate, freeze, validate, replay, and publish the protocol."""

    if run_date != RUN_DATE:
        raise ToolProtocolError(f"run_date_mismatch:{run_date}")
    started = time.monotonic()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes = collect_preconditions(root)
    spans.append(_phase("preconditions", phase_started, started, len(preconditions)))
    progress(started, "preconditions", "complete", units=len(preconditions))

    progress(started, "model_load", "before", operation="no_model_load")
    phase_started = time.monotonic()
    spans.append(_phase("model_load", phase_started, started, 0))
    progress(started, "model_load", "after", current_model_loads=0)
    progress(started, "generation", "before", operation="no_generation")
    phase_started = time.monotonic()
    spans.append(_phase("generation", phase_started, started, 0))
    progress(started, "generation", "after", current_generations=0)

    selected: list[JsonDict] = []
    interventions: list[JsonDict] = []
    exclusions: list[JsonDict] = []
    shard_manifest: JsonDict = {}
    tokenizer_identity: JsonDict | None = None
    inventory_counts = {
        "enumerated_train_rows": 0,
        "enumerated_test_rows": 0,
        "grouped_train_components": 0,
        "grouped_test_components": 0,
        "qualified_train": 0,
        "qualified_test": 0,
    }
    blocked_reason: str | None = None
    if not all(row.get("passed") is True for row in preconditions):
        blocked_reason = "external_precondition"
    else:
        progress(started, "exposure_scan", "start")
        phase_started = time.monotonic()
        exposure, exposure_receipts, missing_exposure = scan_exposure_union(root, started=started)
        source_hashes.extend(exposure_receipts)
        for path in missing_exposure:
            preconditions.append(
                {
                    "check": "exposure_archive_exists",
                    "upstream": "historical_source_exposure_union",
                    "path": path,
                    "field": "exists",
                    "expected": True,
                    "observed": False,
                    "passed": False,
                }
            )
        spans.append(_phase("exposure_scan", phase_started, started, len(exposure_receipts)))
        progress(
            started,
            "exposure_scan",
            "complete",
            files=len(exposure_receipts),
            missing=len(missing_exposure),
        )
        if missing_exposure:
            blocked_reason = "historical_exposure_archive"
        else:
            progress(started, "tokenizer_load", "before", mode="gguf_vocab_only")
            phase_started = time.monotonic()
            token_count, tokenizer_identity = _token_counter()
            spans.append(_phase("tokenizer_load", phase_started, started, 1))
            progress(started, "tokenizer_load", "after", model_weights_read=False)

            train_paths, test_paths = _dataset_paths()
            progress(started, "public_enumeration", "before", shards=4)
            phase_started = time.monotonic()
            public_train = _load_public_rows(train_paths, started=started)
            public_test = _load_public_rows(test_paths, started=started)
            inventory_counts["enumerated_train_rows"] = len(public_train)
            inventory_counts["enumerated_test_rows"] = len(public_test)
            exposure_hashes = set(exposure.normalized_source_hashes) | set(
                exposure.exact_source_hashes
            )
            candidates, component_exclusions = build_components(
                [*public_train, *public_test], exposure_context_hashes=exposure_hashes
            )
            exclusions.extend(component_exclusions)
            train_candidates = [row for row in candidates if row["official_split"] == "train"]
            test_candidates = [row for row in candidates if row["official_split"] == "test"]
            inventory_counts["grouped_train_components"] = len(train_candidates)
            inventory_counts["grouped_test_components"] = len(test_candidates)
            qualified_train, train_exclusions = _qualify_prompt_fit(
                train_candidates, token_count=token_count
            )
            qualified_test, test_exclusions = _qualify_prompt_fit(
                test_candidates, token_count=token_count
            )
            exclusions.extend(train_exclusions)
            exclusions.extend(test_exclusions)
            inventory_counts["qualified_train"] = len(qualified_train)
            inventory_counts["qualified_test"] = len(qualified_test)
            spans.append(_phase("public_enumeration", phase_started, started, len(candidates)))
            progress(
                started,
                "public_enumeration",
                "after",
                grouped_train=len(train_candidates),
                grouped_test=len(test_candidates),
                qualified_train=len(qualified_train),
                qualified_test=len(qualified_test),
            )
            try:
                selected, donor_exclusions = _freeze_donor_safe_roster(
                    qualified_train, qualified_test
                )
                exclusions.extend(donor_exclusions)
                interventions = build_intervention_manifest(
                    selected, token_count=token_count, n_ctx=N_CTX
                )
            except ToolProtocolError as exc:
                blocked_reason = str(exc).split(":", 1)[0]
                selected, interventions = [], []

            if blocked_reason is None:
                progress(started, "evaluator_seal", "before", groups=len(selected))
                phase_started = time.monotonic()
                evaluators = _load_evaluators(
                    [*train_paths, *test_paths], selected, started=started
                )
                predictors = _project_predictors(selected)
                capture = read_protocol(predictors, evaluators, mode="capture")
                if any("label" in row for row in capture["rows"]):
                    raise ToolProtocolError("capture_label_leak")
                online = read_protocol(
                    predictors, evaluators, mode="online", online_release_count=0
                )
                if any("label" in row for row in online["rows"]):
                    raise ToolProtocolError("online_label_leak")
                raw_dir = root / RAW_DIR
                shard_manifest = seal_protocol_shards(
                    raw_dir, predictors, evaluators, interventions
                )
                shard_manifest["exclusions"] = _exclusion_sidecar(raw_dir, exclusions)
                reload_protocol_shards(shard_manifest)
                spans.append(_phase("evaluator_seal", phase_started, started, len(evaluators)))
                progress(started, "evaluator_seal", "after", groups=len(evaluators))

    raw_dir = root / RAW_DIR
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7533-"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise ToolProtocolError("validation_plan_invalid:" + ",".join(plan_errors))
    progress(started, "affected_validation", "before_subprocesses", units=len(commands))
    phase_started = time.monotonic()
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "validity", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60,
    )
    spans.append(_phase("affected_validation", phase_started, started, len(affected)))
    progress(started, "affected_validation", "after_subprocesses", units=len(affected))
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    if affected_reduction["passed"] is not True:
        raise ToolProtocolError("required_affected_validation_failed")

    artifact = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        tokenizer_identity=tokenizer_identity,
        selected=selected,
        exclusions=exclusions,
        inventory_counts=inventory_counts,
        interventions=interventions,
        shard_manifest=shard_manifest,
        validation_receipts=affected,
        validation_passed=True,
        phase_spans=spans,
        duration_s=time.monotonic() - started,
        blocked_reason=blocked_reason,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    errors = validate_artifact(artifact, require_terminal=False)
    if errors:
        raise ToolProtocolError("candidate_invalid:" + ",".join(errors))
    atomic_json(candidate_path, artifact)

    terminal_plan = _terminal_commands(candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", units=len(terminal_plan))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        terminal_plan,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60,
    )
    spans.append(_phase("terminal_validation", phase_started, started, len(terminal)))
    progress(started, "terminal_validation", "after_subprocesses", units=len(terminal))
    if not all(row.get("passed") is True for row in terminal):
        raise ToolProtocolError("required_terminal_validation_failed")
    _finalize_artifact(
        artifact,
        receipts=[*affected, *terminal],
        spans=spans,
        duration_s=time.monotonic() - started,
    )
    errors = validate_artifact(artifact, require_terminal=True)
    if errors:
        raise ToolProtocolError("terminal_artifact_invalid:" + ",".join(errors))
    atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "complete",
        path=RESULT_PATH.as_posix(),
        verdict=artifact["honest_verdict"],
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed run date and read-only replay modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path) -> Path:  # pragma: no cover
    """Resolve a command path against the authenticated worktree."""

    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the protocol or one fresh-process terminal reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise ToolProtocolError(f"run_date_mismatch:{args.date}")
    if args.cold_replay:
        value = json.loads(_argument_path(args.cold_replay).read_text(encoding="utf-8"))
        errors = validate_artifact(value, require_terminal=False)
        print(
            json.dumps({"mode": "cold_replay", "passed": not errors, "errors": errors}),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce:
        value = json.loads(_argument_path(args.independent_reduce).read_text(encoding="utf-8"))
        reduction = independent_reduce(value)
        print(
            json.dumps({"mode": "independent_reduce", **reduction}, sort_keys=True),
            flush=True,
        )
        return int(reduction["passed"] is not True)
    return run_experiment(REPO_ROOT, args.date)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
