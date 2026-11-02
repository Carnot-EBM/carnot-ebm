#!/usr/bin/env python3
"""Exp5354: arithmetic carry token-energy diagnostic.

Spec refs: REQ-VERIFY-5354, SCENARIO-VERIFY-5354.

This module runs a deliberately tiny diagnostic after Exp5353 has proved that
the local GGUF backend exposes real token-probability rows. It asks yes/no
addition questions with known correct and perturbed targets, then converts only
backend top-logprob rows into negative-logprob token energies. The generated
text is treated as receipt material only: it is not scored, reranked, trained
against, or promoted into a broad hallucination or reasoning-verification claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import contextlib
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


JsonDict = dict[str, Any]
PreconditionsProvider = Callable[[], JsonDict]
TokenProbabilityProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5354_arithmetic_carry_token_energy_v488"
EXPERIMENT_NUMBER = 5354
MILESTONE = "2026.07.488"
RUN_DATE = "20260707"
RESULT_RELATIVE_PATH = Path("results/experiment_5354_arithmetic_carry_token_energy_v488.json")
SCHEMA = "carnot.experiment_5354.arithmetic_carry_token_energy.v488"
INFERENCE_SUBSTRATE_LIVE = "live_llm_inference"
INFERENCE_SUBSTRATE_AGGREGATION = "aggregation_from_upstream_artifacts"
ALLOWED_INFERENCE_SUBSTRATES = (INFERENCE_SUBSTRATE_LIVE, INFERENCE_SUBSTRATE_AGGREGATION)
SPEC_REFS = ("REQ-VERIFY-5354", "SCENARIO-VERIFY-5354")
TERMINAL_PREFIXES = ("complete:", "blocked_")
RANDOM_SEED = 5354
MIN_LIVE_DURATION_S = 60.0
N_PROBS = 32
N_PREDICT = 1
MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quantization": "Q4_K_M",
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "quantization": "Q4_K_M",
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "quantization": "Q4_K_M",
    },
)
EXPECTED_ROLES = tuple(str(spec["role"]) for spec in MANDATED_MODEL_SPECS)
EXPECTED_HF_BY_ROLE = {str(spec["role"]): str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
EXPECTED_MODEL_IDS = tuple(str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS)
EXP5353_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5353_tokenprob_feature_audit_corrigendum_v488.json"
)


class _Exp5353Paths:
    RESULT_RELATIVE_PATH = EXP5353_RESULT_RELATIVE_PATH


exp5353 = _Exp5353Paths()

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": "Prevents feature availability from crossing runs without gating.",
    "status": "Lets capstone distinguish clean diagnostic from blocked features.",
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous signal claims."
    ),
    "inference_substrate": (
        "Expected value is live_llm_inference only if local GGUF calls actually ran."
    ),
    "MODEL_SPECS": "Confirms mandated local SOTA GGUF models are included.",
    "preconditions_checked": "Records Exp5353 and backend readiness checks.",
    "selected_model_spec": "Identifies the probed mandated model.",
    "tests_run": "Lists deterministic fixture and schema checks.",
    "diagnostic_case_count": "Bare integer fixes the bounded probe size.",
    "carry_case_count": "Bare integer proves carry rows were actually present.",
    "feature_complete_rate": "Bare numeric prevents missing rows from being hidden.",
    "correct_vs_perturbed_margin": "Bare numeric is the only local signal under test.",
    "unsafe_false_accepts": "Bare integer guards against promoting wrong arithmetic.",
    "external_text_scorer_reopened": "Bare boolean must be false under the manifest.",
    "no_broad_hallucination_claim": "Bare boolean prevents scope inflation.",
    "carry_token_energy_signal_ready": (
        "Bare boolean summarizes whether this tiny signal merits future work."
    ),
    "carry_token_energy_feature_rows": (
        "Records transparent per-case token energies and missing-feature fallbacks."
    ),
}

REQUIRED_WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "selected_model_spec",
    "tests_run",
)
WRAPPED_FIELDS = REQUIRED_WRAPPED_FIELDS + ("carry_token_energy_feature_rows",)
REQUIRED_ARTIFACT_FIELDS = WRAPPED_FIELDS + (
    "diagnostic_case_count",
    "carry_case_count",
    "feature_complete_rate",
    "correct_vs_perturbed_margin",
    "unsafe_false_accepts",
    "external_text_scorer_reopened",
    "no_broad_hallucination_claim",
    "carry_token_energy_signal_ready",
)


@dataclass(frozen=True)
class AdditionCase:
    """One controlled addition equation for token-probability measurement."""

    case_id: str
    category: str
    left: int
    right: int
    displayed_answer: int
    true_answer: int
    correct_aliases: tuple[str, ...]
    perturbed_aliases: tuple[str, ...]
    carry_positions: tuple[int, ...]
    is_perturbed_answer_control: bool

    @property
    def carry_kind(self) -> str:
        if not self.carry_positions:
            return "no_carry"
        if len(self.carry_positions) == 1:
            return "single_carry"
        return "multi_carry"

    @property
    def prompt(self) -> str:
        return (
            "Answer with one lowercase token, yes or no: "
            f"{self.left} + {self.right} = {self.displayed_answer}. Answer:"
        )


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def sha16(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):  # pragma: no cover - defensive I/O
        return {}


def _raw_or_wrapped_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _numeric(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _round_float(value: float | None) -> float | None:
    return None if value is None else round(float(value), 9)


def _normalise_token(value: Any) -> str:
    text = re.sub(r"^[^\w+-]+", "", str(value or "")).strip().lower()
    return text.strip(".,:;!?\"'()[]{}")


def _normalise_aliases(values: Sequence[str]) -> set[str]:
    return {_normalise_token(value) for value in values}


def _carry_positions(left: int, right: int) -> tuple[int, ...]:
    positions: list[int] = []
    carry = 0
    max_digits = max(len(str(abs(left))), len(str(abs(right))))
    for position in range(max_digits):
        left_digit = (abs(left) // (10**position)) % 10
        right_digit = (abs(right) // (10**position)) % 10
        total = left_digit + right_digit + carry
        if total >= 10:
            positions.append(position)
            carry = 1
        else:
            carry = 0
    return tuple(positions)


def _make_case(
    case_id: str,
    category: str,
    left: int,
    right: int,
    displayed_answer: int,
) -> AdditionCase:
    true_answer = left + right
    equation_true = displayed_answer == true_answer
    return AdditionCase(
        case_id=case_id,
        category=category,
        left=left,
        right=right,
        displayed_answer=displayed_answer,
        true_answer=true_answer,
        correct_aliases=("yes", "true") if equation_true else ("no", "false"),
        perturbed_aliases=("no", "false") if equation_true else ("yes", "true"),
        carry_positions=_carry_positions(left, right),
        is_perturbed_answer_control=not equation_true,
    )


ADDITION_CASES: tuple[AdditionCase, ...] = (
    _make_case("no_carry_12_23", "no_carry", 12, 23, 35),
    _make_case("no_carry_104_205", "no_carry", 104, 205, 309),
    _make_case("no_carry_321_456", "no_carry", 321, 456, 777),
    _make_case("no_carry_2304_1052", "no_carry", 2304, 1052, 3356),
    _make_case("single_carry_17_25", "single_carry", 17, 25, 42),
    _make_case("single_carry_46_37", "single_carry", 46, 37, 83),
    _make_case("single_carry_108_207", "single_carry", 108, 207, 315),
    _make_case("single_carry_509_304", "single_carry", 509, 304, 813),
    _make_case("multi_carry_58_67", "multi_carry", 58, 67, 125),
    _make_case("multi_carry_789_654", "multi_carry", 789, 654, 1443),
    _make_case("multi_carry_999_1", "multi_carry", 999, 1, 1000),
    _make_case("multi_carry_476_589", "multi_carry", 476, 589, 1065),
    _make_case("perturbed_single_17_25", "perturbed_control", 17, 25, 43),
    _make_case("perturbed_multi_58_67", "perturbed_control", 58, 67, 124),
    _make_case("perturbed_no_carry_321_456", "perturbed_control", 321, 456, 778),
    _make_case("perturbed_multi_999_1", "perturbed_control", 999, 1, 999),
)


def _model_specs_from_exp5353(exp5353_artifact: Mapping[str, Any]) -> JsonDict:
    prior = _raw_or_wrapped_value(exp5353_artifact, "MODEL_SPECS")
    prior = prior if isinstance(prior, Mapping) else {}
    specs: JsonDict = {}
    for spec in MANDATED_MODEL_SPECS:
        role = str(spec["role"])
        row = prior.get(role) if isinstance(prior.get(role), Mapping) else {}
        specs[role] = {
            "role": role,
            "hf_id": str(spec["hf_id"]),
            "quantization": str(row.get("quantization") or spec.get("quantization", "Q4_K_M")),
            "model_path": row.get("model_path"),
            "status": str(row.get("status") or "missing_local_gguf"),
            "autotokenizer_used": False,
            "file_receipts": row.get("file_receipts"),
            "metadata": row.get("metadata"),
        }
    return specs


def _selected_model_from_exp5353(
    exp5353_artifact: Mapping[str, Any], model_specs: Mapping[str, Any]
) -> JsonDict | None:
    selected = _raw_or_wrapped_value(exp5353_artifact, "selected_model_spec")
    if isinstance(selected, Mapping) and selected.get("hf_id") in EXPECTED_MODEL_IDS:
        out = dict(selected)
        out["autotokenizer_used"] = False
        return out
    fallback = model_specs.get("flagship_dense")
    return dict(fallback) if isinstance(fallback, Mapping) else None


def _server_path(preconditions: Mapping[str, Any], exp5353_artifact: Mapping[str, Any]) -> str | None:
    binary_paths = preconditions.get("binary_paths")
    if isinstance(binary_paths, Mapping) and binary_paths.get("llama-server"):
        return str(binary_paths["llama-server"])
    prior = _raw_or_wrapped_value(exp5353_artifact, "preconditions_checked")
    if isinstance(prior, Mapping) and prior.get("selected_backend_kind") == "llama-server":
        path = prior.get("selected_backend_path")
        return str(path) if path else None
    return None


def _exp5353_ready(exp5353_artifact: Mapping[str, Any]) -> bool:
    return bool(
        _raw_or_wrapped_value(exp5353_artifact, "tokenprob_feature_rows_ready") is True
        and _raw_or_wrapped_value(exp5353_artifact, "per_token_logprob_available") is True
        and _raw_or_wrapped_value(exp5353_artifact, "topk_alternatives_available") is True
        and _raw_or_wrapped_value(exp5353_artifact, "external_text_scorer_reopened") is False
        and _raw_or_wrapped_value(exp5353_artifact, "no_quality_claim") is True
    )


def _retired_scope_check(root: Path, exp5353_artifact: Mapping[str, Any]) -> JsonDict:
    manifest_path = root / "ops" / "exclusion_manifest.yaml"
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        manifest_text = ""
    prior = _raw_or_wrapped_value(exp5353_artifact, "preconditions_checked")
    prior_scope = {}
    if isinstance(prior, Mapping) and isinstance(prior.get("retired_scope_check"), Mapping):
        prior_scope = dict(prior["retired_scope_check"])
    marker_present = bool(
        "phase_d_external_text_scorer_retired" in manifest_text
        or prior_scope.get("phase_d_external_text_scorer_retired_marker_present") is True
    )
    return {
        "manifest_path": str(manifest_path),
        "phase_d_external_text_scorer_retired_marker_present": marker_present,
        "external_text_scorer_reopened": False,
        "retired_scope_reopened": False,
    }


def _precondition_blockers(
    *,
    root: Path,
    exp5353_artifact: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    selected_model_spec: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if not _exp5353_ready(exp5353_artifact):
        blockers.append("exp5353_tokenprob_feature_rows_not_ready")
    if selected_model_spec is None:
        blockers.append("selected_model_spec_unavailable")
    elif selected_model_spec.get("hf_id") not in EXPECTED_MODEL_IDS:
        blockers.append("selected_model_not_mandated")
    else:
        model_path = selected_model_spec.get("model_path")
        if not model_path or not Path(str(model_path)).is_file():
            blockers.append("selected_model_file_missing")
        if selected_model_spec.get("autotokenizer_used") is True:
            blockers.append("autotokenizer_used_for_gguf")
    if preconditions.get("gpu_visible") is not True:
        blockers.append("gpu_not_visible")
    server = _server_path(preconditions, exp5353_artifact)
    if not server or not Path(server).is_file():
        blockers.append("llama_server_binary_missing")
    if _retired_scope_check(root, exp5353_artifact)[
        "phase_d_external_text_scorer_retired_marker_present"
    ] is not True:
        blockers.append("retired_external_text_scorer_manifest_missing")
    return list(dict.fromkeys(blockers))


def _completion_probability_rows(receipt: Mapping[str, Any]) -> list[JsonDict]:
    response = receipt.get("response_json") if isinstance(receipt.get("response_json"), Mapping) else {}
    rows = response.get("completion_probabilities") or receipt.get("completion_probabilities") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _top_logprob_rows(receipt: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for completion_row in _completion_probability_rows(receipt):
        top_rows = completion_row.get("top_logprobs")
        if isinstance(top_rows, Sequence) and not isinstance(top_rows, (str, bytes)):
            rows.extend(dict(row) for row in top_rows if isinstance(row, Mapping))
        elif _numeric(completion_row.get("logprob")) is not None:
            rows.append(completion_row)
    return rows


def _target_logprob(rows: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> float | None:
    target_aliases = _normalise_aliases(aliases)
    for row in rows:
        logprob = _numeric(row.get("logprob"))
        if logprob is not None and _normalise_token(row.get("token")) in target_aliases:
            return logprob
    return None


def _case_receipts_by_id(case_receipts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(receipt.get("case_id")): receipt
        for receipt in case_receipts
        if isinstance(receipt, Mapping) and receipt.get("case_id")
    }


def build_carry_token_energy_rows(
    case_receipts: Sequence[Mapping[str, Any]],
    diagnostic_cases: Sequence[AdditionCase] = ADDITION_CASES,
) -> list[JsonDict]:
    """Compute transparent token-energy rows and explicit fallbacks."""

    receipts_by_id = _case_receipts_by_id(case_receipts)
    interim_rows: list[JsonDict] = []
    for case in diagnostic_cases:
        receipt = receipts_by_id.get(case.case_id, {})
        top_rows = _top_logprob_rows(receipt)
        correct_logprob = _target_logprob(top_rows, case.correct_aliases)
        perturbed_logprob = _target_logprob(top_rows, case.perturbed_aliases)
        correct_energy = -correct_logprob if correct_logprob is not None else None
        perturbed_energy = -perturbed_logprob if perturbed_logprob is not None else None
        interim_rows.append(
            {
                "case": case,
                "top_logprob_row_count": len(top_rows),
                "correct_logprob": correct_logprob,
                "perturbed_logprob": perturbed_logprob,
                "correct_energy": correct_energy,
                "perturbed_energy": perturbed_energy,
            }
        )

    no_carry_energies = [
        float(row["correct_energy"])
        for row in interim_rows
        if row["case"].carry_kind == "no_carry" and row["correct_energy"] is not None
    ]
    no_carry_baseline = (
        sum(no_carry_energies) / len(no_carry_energies) if no_carry_energies else None
    )

    rows: list[JsonDict] = []
    for row in interim_rows:
        case = row["case"]
        correct_energy = row["correct_energy"]
        perturbed_energy = row["perturbed_energy"]
        margin = (
            float(perturbed_energy) - float(correct_energy)
            if perturbed_energy is not None and correct_energy is not None
            else None
        )
        if correct_energy is not None and no_carry_baseline is not None:
            carry_contrast = 0.0 if case.carry_kind == "no_carry" else correct_energy - no_carry_baseline
        else:
            carry_contrast = None
        missing_features: list[str] = []
        if row["top_logprob_row_count"] == 0:
            missing_features.append("completion_probabilities")
        if row["correct_logprob"] is None:
            missing_features.append("correct_target_logprob")
        if row["perturbed_logprob"] is None:
            missing_features.append("perturbed_target_logprob")
        if carry_contrast is None:
            missing_features.append("carry_position_contrast")
        feature_complete = not missing_features
        unsafe_false_accept = bool(
            case.is_perturbed_answer_control
            and feature_complete
            and margin is not None
            and margin <= 0
        )
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "carry_kind": case.carry_kind,
                "left": case.left,
                "right": case.right,
                "displayed_answer": case.displayed_answer,
                "true_answer": case.true_answer,
                "is_perturbed_answer_control": case.is_perturbed_answer_control,
                "prompt_checksum": sha16(case.prompt),
                "carry_positions": list(case.carry_positions),
                "carry_count": len(case.carry_positions),
                "top_logprob_row_count": int(row["top_logprob_row_count"]),
                "correct_target_aliases": list(case.correct_aliases),
                "perturbed_target_aliases": list(case.perturbed_aliases),
                "correct_target_logprob": _round_float(row["correct_logprob"]),
                "perturbed_target_logprob": _round_float(row["perturbed_logprob"]),
                "answer_token_negative_logprob": _round_float(correct_energy),
                "perturbed_answer_token_negative_logprob": _round_float(perturbed_energy),
                "carry_position_contrast": _round_float(carry_contrast),
                "correct_vs_perturbed_margin": _round_float(margin),
                "feature_complete": feature_complete,
                "missing_features": missing_features,
                "unsafe_false_accept": unsafe_false_accept,
                "feature_source": "backend_top_logprobs",
                "quality_interpretation": None,
            }
        )
    return rows


def _feature_missing_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id"))
        for feature in row.get("missing_features") or []:
            missing.append(f"{case_id}:{feature}")
    return missing


def _feature_complete_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    complete = sum(1 for row in rows if row.get("feature_complete") is True)
    return round(complete / len(rows), 6)


def _weakest_margin(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows or any(row.get("feature_complete") is not True for row in rows):
        return 0.0
    margins = [_numeric(row.get("correct_vs_perturbed_margin")) for row in rows]
    numeric_margins = [float(margin) for margin in margins if margin is not None]
    return round(min(numeric_margins), 9) if numeric_margins else 0.0


def _unsafe_false_accepts(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("unsafe_false_accept") is True)


def _carry_case_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("feature_complete") is True and int(row.get("carry_count") or 0) > 0
    )


def _honest_verdict(
    ready: bool,
    live_probe_attempted: bool,
    missing_feature_names: Sequence[str],
    unsafe_false_accepts: int,
) -> str:
    if ready:
        return "complete: carry_token_energy_signal_ready"
    if not live_probe_attempted:
        return "blocked_preconditions:" + ",".join(
            missing_feature_names or ["unknown_precondition"]
        )
    if unsafe_false_accepts:
        return "blocked_unsafe_false_accepts"
    return "blocked_carry_token_features_incomplete:" + ",".join(
        missing_feature_names or ["unknown_feature"]
    )


def _build_preconditions_record(
    *,
    root: Path,
    exp5353_artifact_path: Path,
    exp5353_artifact: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    selected_model_spec: Mapping[str, Any] | None,
    blockers: Sequence[str],
    live_probe_attempted: bool,
) -> JsonDict:
    return {
        "exp5353_artifact_path": str(exp5353_artifact_path),
        "exp5353_tokenprob_feature_rows_ready": _raw_or_wrapped_value(
            exp5353_artifact, "tokenprob_feature_rows_ready"
        ),
        "exp5353_honest_verdict": _raw_or_wrapped_value(exp5353_artifact, "honest_verdict"),
        "gpu_visible": preconditions.get("gpu_visible"),
        "nvidia_smi": preconditions.get("nvidia_smi"),
        "free_vram_mb": preconditions.get("free_vram_mb"),
        "selected_backend_kind": "llama-server",
        "selected_backend_path": _server_path(preconditions, exp5353_artifact),
        "selected_model_hf_id": (selected_model_spec or {}).get("hf_id"),
        "selected_model_path": (selected_model_spec or {}).get("model_path"),
        "selected_model_file_present": bool(
            selected_model_spec
            and selected_model_spec.get("model_path")
            and Path(str(selected_model_spec["model_path"])).is_file()
        ),
        "retired_scope_check": _retired_scope_check(root, exp5353_artifact),
        "external_text_scorer_reopened": False,
        "blocked_preconditions": list(blockers),
        "live_probe_attempted": live_probe_attempted,
    }


def build_artifact(
    *,
    root: Path,
    exp5353_artifact: Mapping[str, Any],
    exp5353_artifact_path: Path,
    preconditions: Mapping[str, Any],
    token_probability_probe: TokenProbabilityProbe,
    tests_run: Sequence[Any],
) -> JsonDict:
    """Build the terminal artifact, blocking before probing if Step 0 fails."""

    started = time.perf_counter()
    model_specs = _model_specs_from_exp5353(exp5353_artifact)
    selected_model_spec = _selected_model_from_exp5353(exp5353_artifact, model_specs)
    blockers = _precondition_blockers(
        root=root,
        exp5353_artifact=exp5353_artifact,
        preconditions=preconditions,
        selected_model_spec=selected_model_spec,
    )
    live_probe_attempted = not blockers
    raw_probe: JsonDict = {}
    if live_probe_attempted:
        raw_probe = dict(
            token_probability_probe(
                selected_model_spec=selected_model_spec,
                preconditions=preconditions,
                diagnostic_cases=ADDITION_CASES,
                n_probs=N_PROBS,
                n_predict=N_PREDICT,
                minimum_duration_s=MIN_LIVE_DURATION_S,
            )
        )
    case_receipts = raw_probe.get("case_receipts") if isinstance(raw_probe.get("case_receipts"), list) else []
    feature_rows = build_carry_token_energy_rows(case_receipts) if live_probe_attempted else []
    methodology_duration_s = (
        round(float(raw_probe.get("wall_clock_s") or 0.0), 6) if live_probe_attempted else 0.0
    )
    feature_complete_rate = _feature_complete_rate(feature_rows)
    correct_vs_perturbed_margin = _weakest_margin(feature_rows)
    unsafe_false_accepts = _unsafe_false_accepts(feature_rows)
    missing_feature_names = list(blockers)
    if live_probe_attempted:
        if raw_probe.get("status") != "completed":
            missing_feature_names.append("probe_status_not_completed")
        if methodology_duration_s < MIN_LIVE_DURATION_S:
            missing_feature_names.append("methodology_duration_below_60s")
        missing_feature_names.extend(_feature_missing_names(feature_rows))
        if correct_vs_perturbed_margin <= 0:
            missing_feature_names.append("correct_vs_perturbed_margin_nonpositive")
        if unsafe_false_accepts:
            missing_feature_names.append("unsafe_false_accepts")
    if not tests_run:
        missing_feature_names.append("tests_run_unrecorded")
    missing_feature_names = list(dict.fromkeys(missing_feature_names))
    ready = bool(
        live_probe_attempted
        and len(feature_rows) == len(ADDITION_CASES)
        and feature_complete_rate == 1.0
        and correct_vs_perturbed_margin > 0
        and unsafe_false_accepts == 0
        and not missing_feature_names
        and tests_run
    )
    substrate = (
        INFERENCE_SUBSTRATE_LIVE if live_probe_attempted else INFERENCE_SUBSTRATE_AGGREGATION
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NUMBER,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", "complete" if ready else "blocked"),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(
                ready, live_probe_attempted, missing_feature_names, unsafe_false_accepts
            ),
        ),
        "inference_substrate": _wrap("inference_substrate", substrate),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "preconditions_checked": _wrap(
            "preconditions_checked",
            _build_preconditions_record(
                root=root,
                exp5353_artifact_path=exp5353_artifact_path,
                exp5353_artifact=exp5353_artifact,
                preconditions=preconditions,
                selected_model_spec=selected_model_spec,
                blockers=blockers,
                live_probe_attempted=live_probe_attempted,
            ),
        ),
        "selected_model_spec": _wrap("selected_model_spec", selected_model_spec),
        "tests_run": _wrap("tests_run", list(tests_run)),
        "diagnostic_case_count": len(feature_rows) if live_probe_attempted else 0,
        "carry_case_count": _carry_case_count(feature_rows),
        "feature_complete_rate": feature_complete_rate,
        "correct_vs_perturbed_margin": correct_vs_perturbed_margin,
        "unsafe_false_accepts": unsafe_false_accepts,
        "external_text_scorer_reopened": False,
        "no_broad_hallucination_claim": True,
        "carry_token_energy_signal_ready": ready,
        "carry_token_energy_feature_rows": _wrap("carry_token_energy_feature_rows", feature_rows),
        "missing_feature_names": missing_feature_names,
        "methodology_duration_s": methodology_duration_s,
        "separability_summary": {
            "weakest_margin": correct_vs_perturbed_margin,
            "complete_row_count": sum(
                1 for row in feature_rows if row.get("feature_complete") is True
            ),
            "total_row_count": len(feature_rows),
            "unsafe_false_accepts": unsafe_false_accepts,
            "interpretation": "bounded_addition_token_energy_diagnostic_only",
        },
        "token_probability_receipt": {
            "live_probe_attempted": live_probe_attempted,
            "backend_kind": raw_probe.get("backend_kind"),
            "endpoint": raw_probe.get("endpoint"),
            "status": raw_probe.get("status"),
            "round_count": raw_probe.get("round_count", 0),
            "case_receipt_count": len(case_receipts),
            "wall_clock_s": raw_probe.get("wall_clock_s", 0.0),
            "quality_interpretation": None,
        },
        "diagnostic_cases": [
            {
                "case_id": case.case_id,
                "category": case.category,
                "carry_kind": case.carry_kind,
                "prompt_checksum": sha16(case.prompt),
                "is_perturbed_answer_control": case.is_perturbed_answer_control,
            }
            for case in ADDITION_CASES
        ],
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
    }
    artifact["duration_s"] = (
        methodology_duration_s if live_probe_attempted else round(time.perf_counter() - started, 6)
    )
    artifact["reproducibility_checksum"] = sha16(
        _stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "selected_model_spec": selected_model_spec,
                "feature_rows": feature_rows,
                "missing_feature_names": missing_feature_names,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    exp5353_artifact_path: Path | None = None,
    preconditions_provider: PreconditionsProvider | None = None,
    token_probability_probe: TokenProbabilityProbe | None = None,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5354 and write the requested result artifact."""

    result_path = result_path or root / RESULT_RELATIVE_PATH
    exp5353_artifact_path = exp5353_artifact_path or root / exp5353.RESULT_RELATIVE_PATH
    preconditions_provider = preconditions_provider or (lambda: _collect_preconditions(root))
    token_probability_probe = token_probability_probe or default_token_probability_probe
    artifact = build_artifact(
        root=root,
        exp5353_artifact=_read_json(exp5353_artifact_path),
        exp5353_artifact_path=exp5353_artifact_path,
        preconditions=dict(preconditions_provider()),
        token_probability_probe=token_probability_probe,
        tests_run=list(tests_run or []),
    )
    if write:
        _write_json(result_path, artifact)
    return artifact


def _collect_preconditions(root: Path) -> JsonDict:  # pragma: no cover - live dependency import
    from carnot import experiment_5323_native_gguf_backend_flag_bisect_v486 as exp5323

    return dict(exp5323.collect_preconditions(root))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fields that downstream gates rely on."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        value = artifact.get(field)
        if (
            not isinstance(value, Mapping)
            or value.get("principle") != FIELD_PRINCIPLES[field]
            or "value" not in value
        ):
            errors.append(f"{field} must be principle wrapped")

    if (artifact.get("experiment_id") or {}).get("value") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if (artifact.get("milestone") or {}).get("value") != MILESTONE:
        errors.append("milestone mismatch")
    status = (artifact.get("status") or {}).get("value")
    if status not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    honest = (artifact.get("honest_verdict") or {}).get("value")
    if not isinstance(honest, str) or not honest.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    substrate = (artifact.get("inference_substrate") or {}).get("value")
    if substrate not in ALLOWED_INFERENCE_SUBSTRATES:
        errors.append("inference_substrate must be live_llm_inference or aggregation")

    for field in (
        "external_text_scorer_reopened",
        "no_broad_hallucination_claim",
        "carry_token_energy_signal_ready",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare boolean")
    if artifact.get("external_text_scorer_reopened") is not False:
        errors.append("external_text_scorer_reopened must be bare false")
    if artifact.get("no_broad_hallucination_claim") is not True:
        errors.append("no_broad_hallucination_claim must be bare true")
    for field in ("diagnostic_case_count", "carry_case_count", "unsafe_false_accepts"):
        if not isinstance(artifact.get(field), int) or isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare integer")
    for field in ("feature_complete_rate", "correct_vs_perturbed_margin"):
        if not isinstance(artifact.get(field), int | float) or isinstance(
            artifact.get(field), bool
        ):
            errors.append(f"{field} must be a bare numeric")

    model_specs = (artifact.get("MODEL_SPECS") or {}).get("value")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        if set(model_specs) != set(EXPECTED_ROLES):
            errors.append("MODEL_SPECS roles mismatch")
        for role in set(model_specs) & set(EXPECTED_ROLES):
            row = model_specs[role]
            if not isinstance(row, Mapping) or row.get("hf_id") != EXPECTED_HF_BY_ROLE[role]:
                errors.append("MODEL_SPECS hf_id mismatch")
            if isinstance(row, Mapping) and row.get("autotokenizer_used") is not False:
                errors.append("autotokenizer_used must stay false")

    selected = (artifact.get("selected_model_spec") or {}).get("value")
    if selected is not None:
        if not isinstance(selected, Mapping):
            errors.append("selected_model_spec must be null or object")
        elif selected.get("hf_id") not in EXPECTED_MODEL_IDS:
            errors.append("selected_model_spec must name a mandated model")
        elif selected.get("autotokenizer_used") is True:
            errors.append("selected_model_spec autotokenizer_used must stay false")
    tests_run = (artifact.get("tests_run") or {}).get("value")
    if not isinstance(tests_run, list):
        errors.append("tests_run must be a list")
    rows = (artifact.get("carry_token_energy_feature_rows") or {}).get("value")
    if not isinstance(rows, list):
        errors.append("carry_token_energy_feature_rows must be a principle-wrapped list")
        rows = []
    missing = artifact.get("missing_feature_names", [])
    if not isinstance(missing, list):
        errors.append("missing_feature_names must be a list")
        missing = []

    ready = artifact.get("carry_token_energy_signal_ready")
    if ready is True:
        if status != "complete":
            errors.append("ready artifact must have complete status")
        if substrate != INFERENCE_SUBSTRATE_LIVE:
            errors.append("ready artifact must use live_llm_inference")
        if artifact.get("diagnostic_case_count") != len(ADDITION_CASES):
            errors.append("ready artifact requires all diagnostic cases")
        if artifact.get("carry_case_count", 0) <= 0:
            errors.append("ready artifact requires carry_case_count > 0")
        if artifact.get("feature_complete_rate") != 1.0:
            errors.append("ready artifact requires feature_complete_rate 1.0")
        if _numeric(artifact.get("correct_vs_perturbed_margin")) is None or float(
            artifact.get("correct_vs_perturbed_margin")
        ) <= 0:
            errors.append("ready artifact requires positive correct_vs_perturbed_margin")
        if artifact.get("unsafe_false_accepts") != 0:
            errors.append("ready artifact requires zero unsafe_false_accepts")
        if missing:
            errors.append("ready artifact must not have missing_feature_names")
        if not tests_run:
            errors.append("ready artifact requires tests_run")
        if not rows or not all(
            isinstance(row, Mapping) and row.get("feature_complete") is True for row in rows
        ):
            errors.append("ready artifact requires complete feature rows")
        if _numeric(artifact.get("methodology_duration_s")) is None or float(
            artifact.get("methodology_duration_s")
        ) < MIN_LIVE_DURATION_S:
            errors.append("ready artifact requires methodology_duration_s >= 60")
    elif ready is False:
        if status != "blocked":
            errors.append("blocked artifact must have blocked status")
    else:
        errors.append("carry_token_energy_signal_ready must be a bare boolean")

    if errors:
        raise ValueError("; ".join(errors))


def default_token_probability_probe(
    *,
    selected_model_spec: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    diagnostic_cases: Sequence[AdditionCase] = ADDITION_CASES,
    n_probs: int = N_PROBS,
    n_predict: int = N_PREDICT,
    minimum_duration_s: float = MIN_LIVE_DURATION_S,
) -> JsonDict:  # pragma: no cover - live llama-server integration
    """Run llama-server and keep only first-round token-probability receipts."""

    server = _server_path(preconditions, {})
    if not server:
        return {"status": "blocked_llama_server_missing", "wall_clock_s": 0.0, "case_receipts": []}
    port = _free_port()
    command = [
        str(server),
        "-m",
        str(selected_model_spec["model_path"]),
        "-c",
        "512",
        "-b",
        "512",
        "-ub",
        "128",
        "-ngl",
        "all",
        "-sm",
        "layer",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-webui",
        "-np",
        "1",
        "--metrics",
        "--props",
        "--slots",
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    case_receipts: list[JsonDict] = []
    all_request_count = 0
    runtime_error: str | None = None
    round_count = 0
    try:
        if not _wait_for_health(port, 180.0):
            runtime_error = "llama-server health endpoint did not become ready"
        else:
            while time.perf_counter() - started < minimum_duration_s or round_count == 0:
                round_count += 1
                for case in diagnostic_cases:
                    response = _post_completion(
                        port, case.prompt, n_probs=n_probs, n_predict=n_predict, timeout_s=90.0
                    )
                    all_request_count += 1
                    if round_count == 1:
                        case_receipts.append(
                            {
                                "case_id": case.case_id,
                                "prompt": case.prompt,
                                "response_json": response,
                            }
                        )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    return {
        "status": "completed" if runtime_error is None and case_receipts else "blocked_probe_failed",
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "command": command,
        "wall_clock_s": round(time.perf_counter() - started, 6),
        "round_count": round_count,
        "all_request_count": all_request_count,
        "case_receipts": case_receipts,
        "runtime_error": runtime_error,
    }


def _free_port() -> int:  # pragma: no cover - live integration helper
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, timeout_s: float) -> bool:  # pragma: no cover
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib_request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib_error.URLError, TimeoutError, OSError):
            time.sleep(1.0)
    return False


def _post_completion(
    port: int,
    prompt: str,
    *,
    n_probs: int,
    n_predict: int,
    timeout_s: float,
) -> JsonDict:  # pragma: no cover
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
        "n_probs": n_probs,
    }
    req = urllib_request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", "replace")
    data = json.loads(body)
    return {
        "content": data.get("content", ""),
        "tokens_predicted": data.get("tokens_predicted"),
        "tokens_evaluated": data.get("tokens_evaluated"),
        "timings": data.get("timings"),
        "completion_probabilities": data.get("completion_probabilities") or [],
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--exp5353", type=Path, default=REPO_ROOT / exp5353.RESULT_RELATIVE_PATH)
    parser.add_argument("--tests-run-json", default="[]")
    args = parser.parse_args(argv)
    artifact = run(
        result_path=args.out,
        exp5353_artifact_path=args.exp5353,
        tests_run=json.loads(args.tests_run_json),
    )
    print(
        f"[exp5354] status={artifact['status']['value']} "
        f"ready={artifact['carry_token_energy_signal_ready']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
