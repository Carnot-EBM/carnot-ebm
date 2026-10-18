"""Freeze the V658 source-removal and mismatch measurement protocol.

This CPU-only module audits whether 480 untouched RAGTruth training source
groups still exist.  It also defines the reusable role, intervention, readout,
reader, and strict-reduction boundaries for a later native-logit capture.

Spec refs: REQ-VERIFY-7517 and SCENARIO-VERIFY-7517-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
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
from carnot.experiment_7462_v654_option_protocol import OPTION_IDS, build_option_prompt
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
RUN_DATE = "20260922"
MILESTONE = "2026.09.658"
EXPERIMENT_ID = "exp7517-v658-source-protocol"
SCHEMA = "carnot.exp7517.v658.source_protocol.v1"
SELECTION_SEED = 658017
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7517_v658_source_protocol.json")
RAW_DIR = Path("results/raw/experiment_7517_v658_source_protocol")
MODULE_PATH = Path("python/carnot/experiment_7517_v658_source_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7517_v658_source_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7517_v658_source_protocol.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RAGTRUTH_ROOT = Path("/home/ianblenke/.cache/carnot-rewrite-20260731/work/data/ragtruth")
RAGTRUTH_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
PINNED_EXTERNAL_HASHES = {
    "source_info.jsonl": "sha256:0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    "response.jsonl": "sha256:e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "tokenizer.json": "sha256:0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
}
TOKENIZER_PATH = Path(
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B/"
    "snapshots/3ea932cee0a432ae86e9c7826cbe8aef52323a28/tokenizer.json"
)
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
    Path("python/carnot/experiment_7491_v656_window_protocol.py"),
    Path("python/carnot/experiment_7462_v654_option_protocol.py"),
    Path("python/carnot/experiment_7504_v657_evidence_interface.py"),
    Path("results/experiment_7507_v657_static_evaluation.json"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
EXPOSURE_RAW_DIRS = (
    "experiment_7423_v651_annotated_protocol",
    "experiment_7449_v653_source_protocol",
    "experiment_7462_v654_option_protocol",
    "experiment_7479_v655_source_fit_capture",
    "experiment_7480_v655_source_eval_capture",
    "experiment_7481_v655_typed_calibration",
    "experiment_7483_v655_continuous_learning",
    "experiment_7491_v656_window_protocol",
    "experiment_7493_v656_window_fit_capture",
    "experiment_7494_v656_window_eval_capture",
    "experiment_7504_v657_evidence_interface",
    "experiment_7505_v657_energy_fit",
    "experiment_7506_v657_causal_prototype",
    "experiment_7507_v657_static_evaluation",
    "experiment_7508_v657_static_audit",
    "experiment_7509_v657_causal_online",
    "experiment_7510_v657_causal_audit",
    "experiment_7514_v657_service_trace",
)
EXPOSURE_RESULT_NAMES = (*EXPOSURE_RAW_DIRS, "experiment_7515_v657_capstone")
TOKEN_CEILING = 2_048
SOURCE_LENGTH_BIN = 64
EMPTY_EVIDENCE_MARKER = "[NO SOURCE EVIDENCE PROVIDED]"
ROLE_COUNTS = {
    "training": 160,
    "calibration_tuning": 40,
    "calibration_policy": 40,
    "test": 120,
    "online": 120,
}
OPTION_ORDERS = (tuple(OPTION_IDS), tuple(reversed(OPTION_IDS)))
PREDICTOR_ALLOWLIST = (
    "group_id",
    "source_family",
    "source_text",
    "response_text",
    "source_hash",
    "source_sha256",
    "response_hash",
    "source_token_count",
    "selection_rank",
    "role",
    "arrival_order",
)
FORBIDDEN_PREDICTOR_FIELDS = {
    "source_id",
    "response_id",
    "corpus",
    "official_split",
    "label",
    "labels",
    "annotations",
    "annotation_labels",
    "response_generator_identity",
    "model",
}
MODEL_SPECS: list[JsonDict] = []
model_specs: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


class SourceProtocolError(ValueError):
    """Reject evidence that can leak labels, identities, or incomplete rows."""


@dataclass(frozen=True)
class ExposureUnion:
    """Public identities consumed by any earlier development activity."""

    source_ids: frozenset[str]
    normalized_source_hashes: frozenset[str]
    exact_source_hashes: frozenset[str]
    response_hashes: frozenset[str]


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 bytes without silent normalization."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_text_hash(value: str) -> str:
    """Hash case-folded whitespace-normalized source text for deduplication."""

    return sha256_text(" ".join(value.casefold().split()))


def canonical_hash(value: object) -> str:
    """Hash stable JSON bytes so one changed contract operand changes identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(encoded)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large historical captures remain bounded."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _natural_source_text(row: Mapping[str, Any]) -> str | None:
    """Serialize only the natural evidence field used by the qualified prompt."""

    value = row.get("source_info")
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return None


def _bytes_complete(row: Mapping[str, Any], field: str) -> bool:
    """Apply registered quality metadata and an exact UTF-8 round trip."""

    value = row.get(field)
    if not isinstance(value, str):
        return False
    if row.get("byte_complete") is False or row.get("bytes_complete") is False:
        return False
    return value.encode("utf-8").decode("utf-8") == value


def _public_candidate(
    source: Mapping[str, Any], response: Mapping[str, Any], token_count: Callable[[str], int]
) -> JsonDict | None:
    """Project one source/response pair without copying evaluator-only fields."""

    source_text = _natural_source_text(source)
    response_text = response.get("response")
    if source_text is None or not _bytes_complete(response, "response"):
        return None
    source_hash = normalized_text_hash(source_text)
    response_hash = sha256_text(str(response_text))
    prompts = [
        build_option_prompt(evidence, str(response_text), order)
        for evidence in (source_text, EMPTY_EVIDENCE_MARKER)
        for order in OPTION_ORDERS
    ]
    if any(int(token_count(prompt)) > TOKEN_CEILING for prompt in prompts):
        return None
    result: JsonDict = {
        "group_id": "group-" + source_hash.removeprefix("sha256:")[:24],
        "source_family": str(source.get("task_type") or "unknown"),
        "source_text": source_text,
        "response_text": str(response_text),
        "source_hash": source_hash,
        "source_sha256": sha256_text(source_text),
        "response_hash": response_hash,
        "source_token_count": int(token_count(source_text)),
        "selection_rank": sha256_text(f"{SELECTION_SEED}:{source_hash}"),
    }
    if not set(result) <= set(PREDICTOR_ALLOWLIST):  # pragma: no cover - fixed projection.
        raise SourceProtocolError("predictor_field_forbidden")
    return result


def select_fresh_groups(
    source_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    exposure: ExposureUnion,
    *,
    requested: int | None = 480,
    token_count: Callable[[str], int],
) -> list[JsonDict]:
    """Select one official-training response per unexposed source before labels."""

    sources = {
        str(row["source_id"]): row
        for row in source_rows
        if row.get("source_id") is not None and _natural_source_text(row) is not None
    }
    choices: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in response_rows:
        source_id = str(row.get("source_id"))
        if (
            source_id in sources
            and row.get("split") == "train"
            and row.get("quality") == "good"
            and _bytes_complete(row, "response")
        ):
            choices[source_id].append(row)
    candidates: list[JsonDict] = []
    for source_id, responses in choices.items():
        source = sources[source_id]
        source_text = _natural_source_text(source)
        assert source_text is not None
        normalized_hash = normalized_text_hash(source_text)
        exact_hash = sha256_text(source_text)
        if (
            source_id in exposure.source_ids
            or normalized_hash in exposure.normalized_source_hashes
            or exact_hash in exposure.exact_source_hashes
        ):
            continue
        public_responses = [
            row
            for row in responses
            if sha256_text(str(row["response"])) not in exposure.response_hashes
        ]
        if not public_responses:
            continue
        selected = min(
            public_responses,
            key=lambda row: sha256_text(
                f"{SELECTION_SEED}:{normalized_hash}:{row.get('id')}:{sha256_text(str(row['response']))}"
            ),
        )
        candidate = _public_candidate(source, selected, token_count)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda row: (str(row["selection_rank"]), str(row["group_id"])))
    if requested is None:
        return candidates
    selected = candidates[:requested]
    return selected if len(selected) == requested else []


def role_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return every frozen role, including explicit zero counts."""

    counts = Counter(str(row.get("role")) for row in rows)
    return {role: counts.get(role, 0) for role in ROLE_COUNTS}


def freeze_roles(
    candidates: Sequence[Mapping[str, Any]],
    *,
    role_targets: Mapping[str, int] = ROLE_COUNTS,
) -> list[JsonDict]:
    """Allocate exact roles proportionally within family without label access."""

    total = sum(int(value) for value in role_targets.values())
    if len(candidates) != total:
        raise SourceProtocolError(f"fresh_inventory_count:{len(candidates)}")
    source_hashes = [str(row.get("source_hash")) for row in candidates]
    if len(source_hashes) != len(set(source_hashes)):
        raise SourceProtocolError("normalized_source_duplicate")
    remaining: dict[str, list[JsonDict]] = defaultdict(list)
    for value in candidates:
        row = {key: deepcopy(value[key]) for key in PREDICTOR_ALLOWLIST if key in value}
        family = str(row.get("source_family"))
        remaining[family].append(row)
    for family, rows in remaining.items():
        rows.sort(
            key=lambda row: sha256_text(f"{SELECTION_SEED}:role:{family}:{row.get('source_hash')}")
        )
    frozen: list[JsonDict] = []
    remaining_total = total
    for role_index, (role, target) in enumerate(role_targets.items()):
        target = int(target)
        if role_index == len(role_targets) - 1:
            quotas = {family: len(rows) for family, rows in remaining.items()}
        else:
            raw = {
                family: len(rows) * target / remaining_total for family, rows in remaining.items()
            }
            quotas = {family: math.floor(value) for family, value in raw.items()}
            deficit = target - sum(quotas.values())
            order = sorted(
                remaining,
                key=lambda family: (
                    -(raw[family] - quotas[family]),
                    sha256_text(f"{SELECTION_SEED}:{role}:{family}"),
                ),
            )
            for family in order[:deficit]:
                quotas[family] += 1
        for family in sorted(remaining):
            selected, remaining[family] = (
                remaining[family][: quotas[family]],
                remaining[family][quotas[family] :],
            )
            for row in selected:
                row["role"] = role
                frozen.append(row)
        remaining_total -= target
    frozen.sort(key=lambda row: (list(role_targets).index(str(row["role"])), str(row["group_id"])))
    if role_counts(frozen) != dict(role_targets):  # pragma: no cover - allocation invariant.
        raise SourceProtocolError("role_count_mismatch")
    return frozen


def _donor_for(target: Mapping[str, Any], peers: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose the nearest source-length bin, then the frozen hash tie break."""

    target_hash = str(target.get("source_hash"))
    target_exact = str(target.get("source_sha256"))
    target_bin = int(target.get("source_token_count", 0)) // SOURCE_LENGTH_BIN
    eligible = [
        row
        for row in peers
        if row.get("group_id") != target.get("group_id")
        and row.get("source_hash") != target_hash
        and row.get("source_sha256") != target_exact
    ]
    if not eligible:
        raise SourceProtocolError(f"donor_missing:{target.get('group_id')}")
    return min(
        eligible,
        key=lambda row: (
            abs(int(row.get("source_token_count", 0)) // SOURCE_LENGTH_BIN - target_bin),
            sha256_text(f"{SELECTION_SEED}:donor:{target_hash}:{row.get('source_hash')}"),
        ),
    )


def _request_cells(
    target: Mapping[str, Any], donor: Mapping[str, Any], token_count: Callable[[str], int]
) -> list[JsonDict]:
    """Build all three full-response conditions in both stable option orders."""

    conditions = {
        "original": str(target["source_text"]),
        "absent": EMPTY_EVIDENCE_MARKER,
        "mismatched": str(donor["source_text"]),
    }
    requests: list[JsonDict] = []
    for condition, source_text in conditions.items():
        for order in OPTION_ORDERS:
            prompt = build_option_prompt(source_text, str(target["response_text"]), order)
            prompt_tokens = int(token_count(prompt))
            if prompt_tokens > TOKEN_CEILING:
                raise SourceProtocolError(
                    f"prompt_overlength:{target.get('group_id')}:{condition}:{prompt_tokens}"
                )
            requests.append(
                {
                    "condition": condition,
                    "option_order": list(order),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_token_count": prompt_tokens,
                    "generated_token_budget": 0,
                    "readout_kind": "option_logits",
                }
            )
    return requests


def build_intervention_manifest(
    rows: Sequence[Mapping[str, Any]], *, token_count: Callable[[str], int]
) -> list[JsonDict]:
    """Freeze one within-role, within-family donor and six requests per group."""

    by_stratum: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[(str(row.get("role")), str(row.get("source_family")))].append(row)
    output: list[JsonDict] = []
    for target in rows:
        stratum = (str(target.get("role")), str(target.get("source_family")))
        donor = _donor_for(target, by_stratum[stratum])
        output.append(
            {
                "group_id": target["group_id"],
                "role": target["role"],
                "source_family": target["source_family"],
                "source_hash": target["source_hash"],
                "source_sha256": target["source_sha256"],
                "donor_group_id": donor["group_id"],
                "donor_source_hash": donor["source_hash"],
                "condition_sources": {
                    "original": target["source_text"],
                    "absent": EMPTY_EVIDENCE_MARKER,
                    "mismatched": donor["source_text"],
                },
                "human_label_scope": "original_source_only",
                "mismatched_is_sensitivity_probe": True,
                "requests": _request_cells(target, donor, token_count),
            }
        )
    return output


def _semantic_logits(row: Mapping[str, Any]) -> dict[str, float]:
    """Map display-label logits back to the two stable semantic option IDs."""

    order_value = row.get("option_order")
    order = tuple(order_value) if isinstance(order_value, list) else ()
    display = row.get("display_logits")
    if order not in OPTION_ORDERS or not isinstance(display, Mapping):
        raise SourceProtocolError("readout_mapping_invalid")
    if set(display) != {" A", " B"}:
        raise SourceProtocolError("readout_mapping_invalid")
    values = [float(display[label]) for label in (" A", " B")]
    if any(not math.isfinite(value) for value in values):
        raise SourceProtocolError("readout_nonfinite")
    return dict(zip(order, values, strict=True))


def semantic_unsupported_probability(row: Mapping[str, Any]) -> float:
    """Return the semantic unsupported probability independent of option order."""

    logits = _semantic_logits(row)
    difference = logits["contains_unsupported"] - logits["supported"]
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    weight = math.exp(difference)
    return weight / (1.0 + weight)


def build_three_vector_views(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    """Reduce six forwards to two clipped condition-logit views."""

    conditions = ("original", "absent", "mismatched")
    cells: dict[tuple[tuple[str, str], str], float] = {}
    for row in rows:
        order_value = row.get("option_order")
        order = tuple(order_value) if isinstance(order_value, list) else ()
        condition = str(row.get("condition"))
        key = (order, condition)
        if order not in OPTION_ORDERS or condition not in conditions or key in cells:
            raise SourceProtocolError("readout_cells_invalid")
        logits = _semantic_logits(row)
        semantic_logit = logits["contains_unsupported"] - logits["supported"]
        cells[key] = min(12.0, max(-12.0, semantic_logit))
    expected = {(order, condition) for order in OPTION_ORDERS for condition in conditions}
    if set(cells) != expected:
        raise SourceProtocolError("readout_cells_invalid")
    return {
        ("supported_first" if order[0] == "supported" else "unsupported_first"): [
            cells[(order, condition)] for condition in conditions
        ]
        for order in OPTION_ORDERS
    }


def _validate_predictors(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject evaluator identities even if a caller does not use their values."""

    for row in rows:
        extras = set(row) - set(PREDICTOR_ALLOWLIST)
        if extras or set(row) & FORBIDDEN_PREDICTOR_FIELDS:
            raise SourceProtocolError("predictor_field_forbidden")


def _freeze_required(observed: str | None, expected: str | None) -> None:
    """Require exact prior bytes before policy or evaluation labels can open."""

    if not observed or not expected:
        raise SourceProtocolError("freeze_hash_required")
    if observed != expected:
        raise SourceProtocolError("freeze_hash_mismatch")


def read_protocol(
    predictors: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    observed_freeze: str | None = None,
    expected_freeze: str | None = None,
    online_release_count: int = 0,
) -> JsonDict:
    """Expose labels only to the registered role and temporal reader boundary."""

    _validate_predictors(predictors)
    evaluator_by_key = {(str(row.get("group_id")), str(row.get("role"))): row for row in evaluators}
    if len(evaluator_by_key) != len(evaluators):
        raise SourceProtocolError("evaluator_identity_duplicate")
    allowed = {
        "fit": {"training", "calibration_tuning"},
        "policy": {"calibration_policy"},
        "predict": set(ROLE_COUNTS),
        "evaluate": {"test"},
        "online": {"online"},
    }
    if mode not in allowed:
        raise SourceProtocolError("reader_mode_invalid")
    if mode in {"policy", "evaluate"}:
        _freeze_required(observed_freeze, expected_freeze)
    selected = [deepcopy(dict(row)) for row in predictors if row.get("role") in allowed[mode]]
    if mode == "online":
        selected.sort(key=lambda row: int(row.get("arrival_order", 0)))
    labels_opened: list[str] = []
    for index, row in enumerate(selected):
        may_open = mode in {"fit", "policy", "evaluate"} or (
            mode == "online" and index < online_release_count
        )
        if may_open:
            key = (str(row.get("group_id")), str(row.get("role")))
            label = evaluator_by_key.get(key, {}).get("label")
            if label not in {0, 1}:
                raise SourceProtocolError("authorized_label_missing")
            row["label"] = int(label)
            labels_opened.append(str(row["role"]))
    return {
        "rows": selected,
        "freeze_sha256": canonical_hash(selected),
        "access_receipt": {
            "mode": mode,
            "label_roles_opened": sorted(set(labels_opened)),
            "held_out_labels_opened": bool(set(labels_opened) & {"test", "online"}),
            "observed_freeze": observed_freeze,
        },
    }


def qualify_guard_rows(rows: Sequence[Mapping[str, Any]], *, expected_ids: set[str]) -> JsonDict:
    """Preserve absolute arm metrics while rejecting null or missing operands."""

    by_id = {str(row.get("group_id")): row for row in rows}
    if set(by_id) != expected_ids or len(by_id) != len(rows):
        raise SourceProtocolError("guard_row_missing")
    candidate_losses: list[float] = []
    control_losses: list[float] = []
    predictions_equal = True
    labels: set[int] = set()
    for group_id in sorted(expected_ids):
        row = by_id[group_id]
        label = row.get("label")
        candidate = row.get("candidate")
        control = row.get("control")
        if label not in {0, 1} or candidate is None or control is None:
            raise SourceProtocolError("guard_value_missing")
        values = (float(candidate), float(control))
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise SourceProtocolError("guard_value_invalid")
        labels.add(int(label))
        candidate_losses.append((values[0] - int(label)) ** 2)
        control_losses.append((values[1] - int(label)) ** 2)
        predictions_equal = predictions_equal and values[0] == values[1]
    candidate_brier = sum(candidate_losses) / len(candidate_losses)
    control_brier = sum(control_losses) / len(control_losses)
    return {
        "row_count": len(rows),
        "label_support": {
            str(label): sum(row["label"] == label for row in rows) for label in labels
        },
        "absolute_arm_metrics": {
            "candidate_brier": candidate_brier,
            "control_brier": control_brier,
        },
        "paired_brier_delta": candidate_brier - control_brier,
        "headroom_present": not predictions_equal,
        "honest_no_headroom_annotation": (
            "identical_arm_predictions" if predictions_equal else None
        ),
    }


def write_exposure_sidecar(
    path: Path,
    *,
    files: Sequence[Mapping[str, Any]],
    candidate_count: int,
    exposed_count: int,
    fresh_count: int,
) -> None:
    """Atomically seal inventory counts and every searched historical file."""

    value = {
        "schema": "carnot.exp7517.exposure.v1",
        "candidate_official_training_groups": int(candidate_count),
        "exposed_candidate_groups": int(exposed_count),
        "fresh_eligible_groups": int(fresh_count),
        "files": [dict(row) for row in files],
    }
    value["inventory_sha256"] = canonical_hash(value)
    atomic_json(path, value)


def reduce_exposure_sidecar(path: Path) -> JsonDict:
    """Independently check sealed inventory arithmetic and its self-identity."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceProtocolError("exposure_object_invalid")
    checksum = value.pop("inventory_sha256", None)
    candidate = value.get("candidate_official_training_groups")
    exposed = value.get("exposed_candidate_groups")
    fresh = value.get("fresh_eligible_groups")
    if not all(
        isinstance(item, int) and not isinstance(item, bool) for item in (candidate, exposed, fresh)
    ):
        raise SourceProtocolError("exposure_count_invalid")
    if int(exposed) + int(fresh) != int(candidate):
        raise SourceProtocolError("exposure_accounting_mismatch")
    if checksum != canonical_hash(value):
        raise SourceProtocolError("exposure_checksum_mismatch")
    value["inventory_sha256"] = checksum
    value["accounting_passed"] = True
    return value


def _gate(
    check: str,
    category: str,
    expected: object,
    observed: object,
    *,
    path: str | None = None,
    upstream: str | None = None,
    field: str | None = None,
) -> JsonDict:
    """Keep validity, readiness, and benefit checks independently auditable."""

    principle = {
        "validity": "Favorable science cannot excuse invalid evidence.",
        "readiness": "A valid null remains eligible for auditing.",
        "benefit": "A hypothesis or insufficient support is not a positive result.",
    }[category]
    row = {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": observed == expected,
        "principle": principle,
    }
    if path is not None:
        row["path"] = path
    if upstream is not None:
        row["upstream"] = upstream
    if field is not None:
        row["field"] = field
    return row


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed check and retain the first exact failed operand."""

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
                "upstream": first.get("upstream", "pinned_ragtruth_official_training_inventory"),
                "field": first.get("field", "fresh_eligible_groups"),
                "path": first.get("path"),
                "expected": first["expected"],
                "observed": first["observed"],
                "op": first["op"],
            }
        ),
    }


def frozen_measurement_contract() -> JsonDict:
    """Freeze formulas, equal-information controls, arrival, and policy costs."""

    return {
        "conditions": ["original", "absent", "mismatched"],
        "option_orders": [list(order) for order in OPTION_ORDERS],
        "forwards_per_group": 6,
        "views": {
            "supported_first": [
                "clip(logit(p_original),-12,12)",
                "clip(logit(p_absent),-12,12)",
                "clip(logit(p_mismatched),-12,12)",
            ],
            "unsupported_first": [
                "clip(logit(p_original),-12,12)",
                "clip(logit(p_absent),-12,12)",
                "clip(logit(p_mismatched),-12,12)",
            ],
        },
        "predictor_field_allowlist": list(PREDICTOR_ALLOWLIST),
        "excluded_covariates": [
            "window_maxima",
            "response_length",
            "corpus_id",
            "response_generator_identity",
            "annotations",
        ],
        "equal_information_controls": ["original_only", "identical_arm"],
        "online_arrival": {
            "order": f"sha256:{SELECTION_SEED}:online:source_hash",
            "feedback_delay": 8,
            "block_release": 8,
        },
        "cost_matrix": [
            {"false_reject": 1, "false_accept": false_accept, "escalation": escalation}
            for false_accept in (1, 5, 20)
            for escalation in (0.1, 0.5, 1.0)
        ],
        "guard_qualification": {
            "fixtures": ["all_null", "identical_arm", "mixed_label", "missing_row"],
            "strict_detection_unchanged": True,
            "absolute_arm_metrics_required": True,
            "honest_no_headroom_annotation_required": True,
        },
    }


def reader_access_contract() -> JsonDict:
    """Describe label custody without implying that labels were opened."""

    return {
        "fit": {"roles": ["training", "calibration_tuning"], "labels": "allowed"},
        "policy": {
            "roles": ["calibration_policy"],
            "labels": "after_fit_freeze_hash",
        },
        "predict": {"roles": list(ROLE_COUNTS), "labels": "forbidden"},
        "evaluate": {"roles": ["test"], "labels": "after_prediction_freeze_hash"},
        "online": {"roles": ["online"], "labels": "arrival_order_feedback_schedule"},
        "actual_label_access": [],
        "labels_available_on_disk": True,
    }


def _field_principle(field: str) -> str:
    """Explain the concrete drift or overclaim prevented by one artifact field."""

    specific = {
        "schema": "Version, experiment_id and milestone bind the terminal reader contract.",
        "run_date": "The fixed date and measured clocks prevent stale-run substitution.",
        "preconditions_checked": "Exact resource observations and hashes prevent invented readiness.",
        "MODEL_SPECS": "An empty list prevents historical Qwen work becoming a current load claim.",
        "model_specs": "The lowercase mirror prevents model identity aliases from disagreeing.",
        "model_invoked": "A bare false separates current CPU protocol work from archived inference.",
        "invocation_counts": "Attempted and terminal counters expose unfinished model operations.",
        "source_protocol_ready_score": "Bare 0/1 requires fresh disjoint roles and lossless prompts.",
        "exposure_audit": "Consumed source hashes are excluded and unverifiable freshness closes work.",
        "reader_access_contract": "Only registered fitting roles can expose labels to fitting code.",
        "flagged_adversarial": "Actual verifier findings cannot be cleared to open a gate.",
    }
    return specific.get(field, f"The {field} field preserves an auditable protocol operand.")


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every stable code, role, inventory, and model-identity operand."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _path_label(path: Path, root: Path) -> str:
    """Use a stable repository-relative label when the file is below root."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _build_blocked_artifact(
    inventory_path: Path,
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_passed: bool,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    exposure_complete: bool = True,
) -> JsonDict:
    """Build the complete terminal block without selecting an exposed group."""

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidate_count = int(inventory["candidate_official_training_groups"])
    exposed_count = int(inventory["exposed_candidate_groups"])
    fresh_count = int(inventory["fresh_eligible_groups"])
    inventory_label = _path_label(inventory_path, root)
    precondition_gates = [
        _gate(
            f"precondition:{row.get('check')}",
            "validity",
            row.get("expected"),
            row.get("observed"),
            path=str(row.get("path")),
            upstream=str(row.get("path")),
            field=str(row.get("field")),
        )
        for row in preconditions
    ]
    gates = [
        *precondition_gates,
        _gate("required_validation", "validity", True, validation_passed),
        _gate("exposure_search_complete", "readiness", True, exposure_complete),
        _gate(
            "fresh_source_inventory",
            "readiness",
            480,
            fresh_count,
            path=f"{inventory_label}#/fresh_eligible_groups",
        ),
        _gate("predictive_benefit_measured", "benefit", False, False),
    ]
    precondition_failure = next(
        (row for row in preconditions if row.get("passed") is not True), None
    )
    if precondition_failure is not None:
        verdict = "complete_blocked_prerequisite_missing_or_mismatched"
    elif not exposure_complete:
        verdict = "complete_blocked_exposure_audit_incomplete"
    else:
        verdict = "complete_blocked_fresh_source_inventory"
    now = datetime.now(UTC).isoformat()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7517,
        "milestone": MILESTONE,
        "title": "V658 source removal and mismatch protocol",
        "run_date": RUN_DATE,
        "terminal_status": "complete",
        "started_at_utc": now,
        "completed_at_utc": now,
        "process_identity": {
            "pid": os.getpid(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "source_revision": _source_revision(root),
        },
        "preconditions_checked": [dict(row) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_details": {
            "current_work": [
                "historical_exposure_union",
                "official_training_inventory",
                "tokenizer_only_prompt_accounting",
                "protocol_sealing",
            ],
            "future_native_option_forwards": {
                "substrate": "live_llm_embedding_extraction",
                "readout_kind": "option_logits",
                "generated_tokens": 0,
                "forwards_per_group": 6,
            },
        },
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "duration_breakdown_s": {
            "current_protocol_and_validation": float(duration_s),
            "authoring": 0.0,
            "historical_capture": 0.0,
        },
        "phase_spans": [dict(row) for row in phase_spans],
        "random_seed": {
            "selection": SELECTION_SEED,
            "fitting": 658018,
            "arrival": 658019,
            "bootstrap": 658020,
        },
        "source_artifact_hashes": [dict(row) for row in source_hashes],
        "rows": [],
        "sample_size_budget": {
            "planned": 480,
            "attempted": 0,
            "completed": 0,
            "excluded": exposed_count,
            "failed": 0,
            "censored": 0,
            "unstarted": 480,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "honest_verdict": verdict,
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [dict(row) for row in validation_receipts],
        "source_protocol_ready_score": 0,
        "role_manifest": {
            "planned_role_counts": dict(ROLE_COUNTS),
            "selected_role_counts": {role: 0 for role in ROLE_COUNTS},
            "selected_group_count": 0,
            "assignment_sha256": None,
            "selection_before_label_access": True,
        },
        "intervention_manifest": {
            "status": "blocked_before_selection",
            "planned_conditions": 1_440,
            "planned_forwards": 2_880,
            "registered_conditions": [],
            "donor_map_sha256": None,
            "empty_evidence_marker": EMPTY_EVIDENCE_MARKER,
        },
        "exposure_audit": {
            "inventory_path": inventory_label,
            "inventory_sha256": sha256_file(inventory_path),
            "candidate_official_training_groups": candidate_count,
            "exposed_candidate_groups": exposed_count,
            "fresh_eligible_groups": fresh_count,
            "official_test_rows_opened": 0,
            "freshness_complete": exposure_complete,
        },
        "reader_access_contract": reader_access_contract(),
        "measurement_contract": frozen_measurement_contract(),
        "historical_model_provenance": {
            "model_id": "unsloth/Qwen3.8-27B-GGUF",
            "counted_as_current_invocation": False,
            "source": "V655-V657 archived option-logit captures",
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay_required": True,
            "numbered_runtime_e2e": "not_applicable_reporting_only",
        },
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "research_conductor_modified": False,
        "external_publication_performed": False,
        "push_performed": False,
        "reproducibility_checksum": "",
        "field_principles": {},
    }
    artifact["field_principles"] = {key: _field_principle(key) for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _source_revision(root: Path) -> str:  # pragma: no cover - live repository identity.
    """Read the worktree revision without starting an untracked subprocess."""

    head = root / ".git/HEAD"
    if not head.is_file():
        return "unavailable"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref_path = root / ".git" / value.removeprefix("ref: ")
        return ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else value
    return value


def build_blocked_artifact_for_test(inventory_path: Path) -> JsonDict:
    """Build a tiny authentic blocked record without repository or model work."""

    receipt = {
        "path": inventory_path.name,
        "sha256": sha256_file(inventory_path),
        "bytes": inventory_path.stat().st_size,
    }
    return _build_blocked_artifact(
        inventory_path,
        root=inventory_path.parent,
        preconditions=[
            {
                "check": "fixture_inventory_exists",
                "path": inventory_path.name,
                "field": "exists",
                "expected": True,
                "observed": True,
                "passed": True,
            }
        ],
        source_hashes=[receipt],
        validation_receipts=[],
        validation_passed=True,
        duration_s=0.001,
        phase_spans=[
            {
                "phase": "inventory",
                "start_s": 0.0,
                "end_s": 0.001,
                "duration_s": 0.001,
                "completed_units": 1,
                "heartbeat_count": 0,
            }
        ],
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path, require_terminal: bool = True
) -> list[str]:
    """Cold-check terminal identity, inventory bytes, gates, and current work."""

    errors: list[str] = []
    required = {
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "preconditions_checked",
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "inference_substrate_class",
        "inference_substrate",
        "duration_s",
        "phase_spans",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "honest_verdict",
        "verdict_class",
        "verifier_is_oracle",
        "flagged_adversarial",
        "validation_receipts",
        "field_principles",
        "source_protocol_ready_score",
        "role_manifest",
        "intervention_manifest",
        "exposure_audit",
        "reader_access_contract",
    }
    if not required <= set(value):
        errors.append("required_fields_missing")
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != RUN_DATE
    ):
        errors.append("terminal_identity_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_nonempty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_accounting_mismatch")
    if (
        value.get("inference_substrate_class") != "no_model_load"
        or value.get("inference_substrate") != "aggregation_from_upstream_artifacts"
    ):
        errors.append("inference_substrate_mismatch")
    if value.get("verdict_class") == "blocked":
        if value.get("source_protocol_ready_score") != 0:
            errors.append("blocked_readiness_mismatch")
        if not str(value.get("honest_verdict", "")).startswith("complete_blocked_"):
            errors.append("blocked_verdict_mismatch")
        if value.get("rows") != []:
            errors.append("blocked_rows_present")
    audit = value.get("exposure_audit")
    if isinstance(audit, Mapping):
        label = audit.get("inventory_path")
        path = Path(str(label))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != audit.get("inventory_sha256"):
            errors.append("exposure_inventory_hash_mismatch")
        if audit.get("official_test_rows_opened") != 0:
            errors.append("official_test_touched")
    else:
        errors.append("exposure_audit_invalid")
    summary = value.get("gate_check_summary")
    first_failure = summary.get("first_failure") if isinstance(summary, Mapping) else None
    expected_failure = (
        "fresh_source_inventory"
        if value.get("honest_verdict") == "complete_blocked_fresh_source_inventory"
        else None
    )
    if not isinstance(first_failure, Mapping) or (
        expected_failure is not None and first_failure.get("check") != expected_failure
    ):
        errors.append("gate_summary_mismatch")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(value):
        errors.append("field_principles_incomplete")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if require_terminal:
        passed = {
            row.get("name")
            for row in value.get("validation_receipts", [])
            if isinstance(row, Mapping) and row.get("passed") is True
        }
        required_names = set(validation_scope.REQUIRED_CHECK_NAMES) | {
            "declared_entrypoint_cold_replay",
            "independent_exposure_reduction",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        }
        if not required_names <= passed:
            errors.append("terminal_validation_incomplete")
    return list(dict.fromkeys(errors))


def _load_jsonl(path: Path) -> list[JsonDict]:  # pragma: no cover - live pinned inputs.
    """Load object rows from one authenticated JSONL resource."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SourceProtocolError(f"jsonl_object_required:{path}:{number}")
            rows.append(value)
    return rows


def _load_public_training_responses(path: Path) -> list[JsonDict]:  # pragma: no cover
    """Project official-training public fields without reading evaluator values."""

    rows: list[JsonDict] = []
    for value in _load_jsonl(path):
        if value.get("split") != "train":
            continue
        rows.append(
            {
                key: value.get(key)
                for key in (
                    "id",
                    "source_id",
                    "response",
                    "split",
                    "quality",
                    "byte_complete",
                    "bytes_complete",
                )
                if key in value
            }
        )
    return rows


def _scan_exposure_value(
    value: object,
    source_ids: set[str],
    normalized_hashes: set[str],
    exact_hashes: set[str],
    response_hashes: set[str],
) -> None:  # pragma: no cover - live historical bytes.
    """Collect public identities recursively from one historical JSON value."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"source_id", "source_alias"} and isinstance(item, (str, int)):
                source_ids.add(str(item))
            elif key in {"source_hash", "normalized_source_hash"} and isinstance(item, str):
                normalized_hashes.add(item)
            elif key in {"source_sha256", "exact_source_hash"} and isinstance(item, str):
                exact_hashes.add(item)
            elif key in {"response_hash", "response_sha256"} and isinstance(item, str):
                response_hashes.add(item)
            elif key in {"source_text", "original_source", "mismatched_source"} and isinstance(
                item, str
            ):
                normalized_hashes.add(normalized_text_hash(item))
                exact_hashes.add(sha256_text(item))
            _scan_exposure_value(item, source_ids, normalized_hashes, exact_hashes, response_hashes)
    elif isinstance(value, list):
        for item in value:
            _scan_exposure_value(item, source_ids, normalized_hashes, exact_hashes, response_hashes)


def _exposure_files(root: Path) -> tuple[list[Path], list[str]]:  # pragma: no cover
    """Resolve every registered lineage file and name missing directories."""

    paths: set[Path] = set()
    missing: list[str] = []
    for name in EXPOSURE_RAW_DIRS:
        directory = root / "results/raw" / name
        if not directory.is_dir():
            missing.append(directory.relative_to(root).as_posix())
            continue
        paths.update(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        )
    for name in EXPOSURE_RESULT_NAMES:
        path = root / "results" / f"{name}.json"
        if not path.is_file():
            missing.append(path.relative_to(root).as_posix())
        else:
            paths.add(path)
    return sorted(paths), missing


def scan_exposure_union(
    root: Path, *, started: float
) -> tuple[ExposureUnion, list[JsonDict], list[str]]:  # pragma: no cover - capability E2E.
    """Hash and scan all registered source lineage bytes with unit progress."""

    paths, missing = _exposure_files(root)
    source_ids: set[str] = set()
    normalized_hashes: set[str] = set()
    exact_hashes: set[str] = set()
    response_hashes: set[str] = set()
    receipts: list[JsonDict] = []
    for index, path in enumerate(paths, 1):
        before = tuple(
            len(values) for values in (source_ids, normalized_hashes, exact_hashes, response_hashes)
        )
        if path.suffix == ".jsonl":
            for row in _load_jsonl(path):
                _scan_exposure_value(
                    row, source_ids, normalized_hashes, exact_hashes, response_hashes
                )
        else:
            _scan_exposure_value(
                json.loads(path.read_text(encoding="utf-8")),
                source_ids,
                normalized_hashes,
                exact_hashes,
                response_hashes,
            )
        after = tuple(
            len(values) for values in (source_ids, normalized_hashes, exact_hashes, response_hashes)
        )
        receipts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "new_source_id_count": after[0] - before[0],
                "new_normalized_source_hash_count": after[1] - before[1],
                "new_exact_source_hash_count": after[2] - before[2],
                "new_response_hash_count": after[3] - before[3],
            }
        )
        if index % 25 == 0 or index == len(paths):
            progress(started, "exposure_scan", "units_complete", units=f"{index}/{len(paths)}")
    return (
        ExposureUnion(
            source_ids=frozenset(source_ids),
            normalized_source_hashes=frozenset(normalized_hashes),
            exact_source_hashes=frozenset(exact_hashes),
            response_hashes=frozenset(response_hashes),
        ),
        receipts,
        missing,
    )


def _token_counter() -> Callable[[str], int]:  # pragma: no cover - tokenizer-only work.
    """Load tokenizer JSON only; no language-model weights are opened."""

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    return lambda text: len(tokenizer.encode(text).ids)


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit one flushed phase, model, benchmark, or subprocess boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7517] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _phase(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover
    """Record one real monotonic span and completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "heartbeat_count": 0,
    }


def _source_receipt(path: Path, root: Path) -> JsonDict:  # pragma: no cover
    """Bind exact input bytes while retaining stable local labels where possible."""

    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover - live resources.
    """Check every named repository input and exact external corpus byte hash."""

    paths = [root / path for path in INPUT_PATHS]
    paths.extend(
        [RAGTRUTH_ROOT / "source_info.jsonl", RAGTRUTH_ROOT / "response.jsonl", TOKENIZER_PATH]
    )
    checks: list[JsonDict] = []
    receipts: list[JsonDict] = []
    for path in paths:
        exists = path.is_file()
        checks.append(
            {
                "check": "resource_exists",
                "path": _path_label(path, root),
                "field": "exists",
                "expected": True,
                "observed": exists,
                "passed": exists,
            }
        )
        if exists:
            receipts.append(_source_receipt(path, root))
    for path in paths[-3:]:
        if not path.is_file():
            continue
        observed = sha256_file(path)
        expected = PINNED_EXTERNAL_HASHES[path.name]
        checks.append(
            {
                "check": "pinned_external_hash",
                "path": str(path),
                "field": "sha256",
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    v657_path = root / "results/experiment_7507_v657_static_evaluation.json"
    if v657_path.is_file():
        v657 = json.loads(v657_path.read_text(encoding="utf-8"))
        expected_fields = {
            "schema": "carnot.exp7507.v657.static_evaluation.v1",
            "honest_verdict": "complete_null_static_evaluation_exploratory_prior_exposure",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "static_evaluation_complete_score": 1,
            "model_invoked": False,
        }
        for field, expected in expected_fields.items():
            observed = v657.get(field)
            checks.append(
                {
                    "check": "v657_terminal_field",
                    "path": v657_path.relative_to(root).as_posix(),
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                    "passed": observed == expected,
                }
            )
    checks.append(
        {
            "check": "ragtruth_release_declared",
            "path": str(RAGTRUTH_ROOT),
            "field": "release_revision",
            "expected": RAGTRUTH_REVISION,
            "observed": RAGTRUTH_REVISION,
            "passed": True,
        }
    )
    return checks, receipts


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build the fresh-process replay and exact-candidate safety readers."""

    relative = candidate.relative_to(REPO_ROOT).as_posix()
    python = ".venv/bin/python"
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--cold-replay",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "independent_exposure_reduction",
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


def _finalize_artifact(
    artifact: JsonDict,
    *,
    receipts: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> None:  # pragma: no cover
    """Refresh measured terminal fields after subprocess validation."""

    artifact["validation_receipts"] = [dict(row) for row in receipts]
    artifact["phase_spans"] = [dict(row) for row in spans]
    artifact["duration_s"] = duration_s
    artifact["duration_breakdown_s"]["current_protocol_and_validation"] = duration_s
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["field_principles"] = {key: _field_principle(key) for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover - capability E2E.
    """Audit fresh inventory, validate the protocol, and publish atomically."""

    if run_date != RUN_DATE:
        raise SourceProtocolError(f"run_date_mismatch:{run_date}")
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

    progress(started, "exposure_scan", "start")
    phase_started = time.monotonic()
    exposure, exposure_files, missing_exposure = scan_exposure_union(root, started=started)
    for path in missing_exposure:
        preconditions.append(
            {
                "check": "exposure_archive_exists",
                "path": path,
                "field": "exists",
                "expected": True,
                "observed": False,
                "passed": False,
            }
        )
    spans.append(_phase("exposure_scan", phase_started, started, len(exposure_files)))
    progress(
        started,
        "exposure_scan",
        "complete",
        files=len(exposure_files),
        missing=len(missing_exposure),
    )

    progress(started, "tokenizer_inventory", "before_tokenizer_load")
    phase_started = time.monotonic()
    token_count = _token_counter()
    progress(started, "tokenizer_inventory", "after_tokenizer_load")
    source_rows = _load_jsonl(RAGTRUTH_ROOT / "source_info.jsonl")
    response_rows = _load_public_training_responses(RAGTRUTH_ROOT / "response.jsonl")
    empty_exposure = ExposureUnion(frozenset(), frozenset(), frozenset(), frozenset())
    official_candidates = select_fresh_groups(
        source_rows,
        response_rows,
        empty_exposure,
        requested=None,
        token_count=token_count,
    )
    fresh_candidates = select_fresh_groups(
        source_rows,
        response_rows,
        exposure,
        requested=None,
        token_count=token_count,
    )
    if len(fresh_candidates) >= 480:
        selected = fresh_candidates[:480]
        frozen = freeze_roles(selected)
        build_intervention_manifest(frozen, token_count=token_count)
        raise SourceProtocolError("unexpected_fresh_inventory_requires_ready_artifact")
    candidate_count = len(official_candidates)
    fresh_count = len(fresh_candidates)
    exposed_count = candidate_count - fresh_count
    spans.append(_phase("tokenizer_inventory", phase_started, started, candidate_count))
    progress(
        started,
        "tokenizer_inventory",
        "complete",
        candidates=candidate_count,
        exposed=exposed_count,
        fresh=fresh_count,
    )

    raw_dir = root / RAW_DIR
    inventory_path = raw_dir / "exposure_inventory.json"
    write_exposure_sidecar(
        inventory_path,
        files=exposure_files,
        candidate_count=candidate_count,
        exposed_count=exposed_count,
        fresh_count=fresh_count,
    )
    inventory_reduction = reduce_exposure_sidecar(inventory_path)
    if inventory_reduction["fresh_eligible_groups"] != fresh_count:
        raise SourceProtocolError("independent_inventory_mismatch")
    source_hashes.append(_source_receipt(inventory_path, root))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7517-"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise SourceProtocolError("validation_plan_invalid:" + ",".join(plan_errors))
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
        raise SourceProtocolError("required_affected_validation_failed")

    artifact = _build_blocked_artifact(
        inventory_path,
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=affected,
        validation_passed=True,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        exposure_complete=not missing_exposure,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    candidate_errors = validate_artifact(artifact, root=root, require_terminal=False)
    if candidate_errors:
        raise SourceProtocolError("candidate_invalid:" + ",".join(candidate_errors))
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
        raise SourceProtocolError("required_terminal_validation_failed")
    _finalize_artifact(
        artifact,
        receipts=[*affected, *terminal],
        spans=spans,
        duration_s=time.monotonic() - started,
    )
    errors = validate_artifact(artifact, root=root, require_terminal=True)
    if errors:
        raise SourceProtocolError("terminal_artifact_invalid:" + ",".join(errors))
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "publish", "complete", path=RESULT_PATH.as_posix())
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed run date and read-only fresh-process modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path) -> Path:  # pragma: no cover
    """Resolve a CLI path relative to the authenticated worktree."""

    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI.
    """Run the protocol or one fresh-process terminal reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SourceProtocolError(f"run_date_mismatch:{args.date}")
    if args.cold_replay:
        path = _argument_path(args.cold_replay)
        value = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_artifact(value, root=REPO_ROOT, require_terminal=False)
        print(
            json.dumps({"mode": "cold_replay", "passed": not errors, "errors": errors}),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce:
        path = _argument_path(args.independent_reduce)
        value = json.loads(path.read_text(encoding="utf-8"))
        audit = value["exposure_audit"]
        inventory = _argument_path(Path(str(audit["inventory_path"])))
        reduction = reduce_exposure_sidecar(inventory)
        fields = (
            "candidate_official_training_groups",
            "exposed_candidate_groups",
            "fresh_eligible_groups",
        )
        passed = all(reduction[field] == audit[field] for field in fields)
        print(
            json.dumps(
                {"mode": "independent_reduce", "passed": passed, "reduction": reduction},
                sort_keys=True,
            ),
            flush=True,
        )
        return int(not passed)
    return run_experiment(REPO_ROOT, args.date)


if __name__ == "__main__":  # pragma: no cover - module execution.
    raise SystemExit(main())
