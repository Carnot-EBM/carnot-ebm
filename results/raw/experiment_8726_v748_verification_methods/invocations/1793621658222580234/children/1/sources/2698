"""Produce the REQ-ARC-7127 paired adapter-withheld ARC development cell.

The small pure functions in this module own the artifact's fail-closed
projection.  The live boundary writes a complete blocked artifact first, then
uses fresh child processes for setup and each real ``E3AgentPolicy`` arm.  It
never edits the solve registry and never converts this public-game measurement
into solve credit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_7127_v626_adapter_withheld_arc_loo.json")
RAW_DEFAULT_ROOT = Path("/tmp/carnot-exp7127-v626-adapter-withheld-arc-loo")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
MODEL_REPOSITORY = "unsloth/Qwen3.6-35B-A3B-GGUF"
MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MODEL_REPOSITORY,
        "role": "headline_model",
        "quantization": "Q4_K_M",
        "chat_template": "gguf_metadata_qwen3",
    }
]
RANDOM_SEED = 7_127_202_609_07
DEFAULT_ACTION_BUDGET = 3
DEFAULT_GENERATION_MAX_TOKENS = 384
COMMON_BUDGET: JsonDict = {
    "actions": DEFAULT_ACTION_BUDGET,
    "generation_max_tokens": DEFAULT_GENERATION_MAX_TOKENS,
    "context_tokens": 16_384,
    "generation_timeout_s": 600,
}
ARM_NAMES = ("adapter_withheld", "adapter_visible_control")
PHASES = (
    "artifact_initialization",
    "registry_freeze",
    "setup",
    "adapter_withheld",
    "adapter_visible_control",
    "finalization",
)
SETUP_CAP_S = 300
WITHHELD_ARM_CAP_S = 1500
CONTROL_ARM_CAP_S = 1500
FINALIZATION_CAP_S = 300
INFERENCE_SUBSTRATE = "live_llm_inference: rebudgeted adapter-withheld live E3 cell"
REAL_ENTRYPOINT = "make_carnot_agent:E3AgentPolicy"

LEGACY_REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "MODEL_SPECS",
    "models_used",
    "model_repository",
    "model_path",
    "model_hash",
    "model_quantization",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "gpu_telemetry_rows",
    "token_rows",
    "duration_s",
    "source_artifact_hashes",
    "raw_trace_manifest",
    "rows",
    "per_game_results",
    "phase_receipt_rows",
    "process_rows",
    "request_rows",
    "proposal_rows",
    "verifier_rows",
    "action_rows",
    "transition_rows",
    "arm_rows",
    "selected_game",
    "registry_rank_before_outcomes",
    "adapter_withheld_exactly",
    "real_e3_entrypoint_used",
    "fresh_process_per_arm",
    "setup_cap_s",
    "withheld_arm_cap_s",
    "control_arm_cap_s",
    "finalization_cap_s",
    "withheld_levels",
    "control_levels",
    "level_delta",
    "solve_provenance",
    "solve_claim_made",
    "offline_reproduced",
    "registry_mutated",
    "arc_loo_cell_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

V627_REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_game_results",
    "MODEL_SPECS",
    "model_identity_rows",
    "model_load_receipts",
    "registry_precheck_rows",
    "selected_game_id",
    "eligibility_rank",
    "action_budget",
    "generation_max_tokens",
    "arm_rows",
    "attempt_rows",
    "raw_output_rows",
    "parse_rows",
    "action_rows",
    "transition_rows",
    "reward_rows",
    "level_rows",
    "token_rows",
    "truncation_rows",
    "process_rows",
    "import_rows",
    "forbidden_read_rows",
    "input_difference_rows",
    "policy_difference_rows",
    "stop_reason_rows",
    "visible_control_nonzero_score",
    "adapter_access_clean",
    "registry_unchanged_score",
    "arc_loop_solve",
    "solve_provenance",
    "game_level_solve_claimed",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

REQUIRED_ARTIFACT_FIELDS = tuple(
    dict.fromkeys((*LEGACY_REQUIRED_ARTIFACT_FIELDS, *V627_REQUIRED_ARTIFACT_FIELDS))
)

_PRINCIPLES = {
    "field_principles": "Explains why every artifact field exists.",
    "preconditions_checked": "Preserves every setup gate before inference.",
    "run_date": "Pins the requested execution date.",
    "MODEL_SPECS": "Declares the mandated headline model without substitution.",
    "models_used": "Separates actually invoked models from declarations.",
    "model_repository": "Pins the model repository identity.",
    "model_path": "Binds generation to one cached local file.",
    "model_hash": "Binds generation to exact GGUF bytes.",
    "model_quantization": "Makes the Q4_K_M requirement explicit.",
    "inference_substrate": "Distinguishes live GGUF generation from replay.",
    "inference_substrate_class": "Distinguishes an executed cell from a no-run block.",
    "execution_venue": "Records where processes and GPUs executed.",
    "gpu_telemetry_rows": "Shows both leased RTX 3090 devices and live samples.",
    "token_rows": "Makes actual prompt and generated token counts recountable.",
    "duration_s": "Exposes the bounded wall-clock cost.",
    "source_artifact_hashes": "Freezes registry, fixture, and producer inputs.",
    "raw_trace_manifest": "Binds external raw bytes by path, size, and hash.",
    "rows": "Provides canonical per-arm level rows.",
    "per_game_results": "Keeps the selected public game's paired result visible.",
    "phase_receipt_rows": "Proves every bounded phase occurred exactly once.",
    "process_rows": "Proves setup and arms used identified fresh processes.",
    "request_rows": "Proves model-bounded generation was requested in each arm.",
    "proposal_rows": "Binds policy proposals to later actions.",
    "verifier_rows": "Records non-oracle action-schema decisions.",
    "action_rows": "Separates proposed actions from executed actions.",
    "transition_rows": "Provides the sole source for level and action metrics.",
    "arm_rows": "Summarizes evidence completeness independently per arm.",
    "selected_game": "Freezes eligibility rank one before reading outcomes.",
    "registry_rank_before_outcomes": "Proves rank-one selection timing.",
    "adapter_withheld_exactly": "Rejects leakage or removal of any other adapter.",
    "real_e3_entrypoint_used": "Rejects synthetic policy entrypoints.",
    "fresh_process_per_arm": "Rejects stale or shared worker identities.",
    "setup_cap_s": "Pins the 300-second setup ceiling.",
    "withheld_arm_cap_s": "Pins the 1500-second withheld ceiling.",
    "control_arm_cap_s": "Pins the 1500-second control ceiling.",
    "finalization_cap_s": "Reserves 300 seconds for validation and publication.",
    "withheld_levels": "Counts withheld progress only from executed transitions.",
    "control_levels": "Counts control progress only from executed transitions.",
    "level_delta": "Reports withheld minus control executed level progress.",
    "solve_provenance": "Labels every result as development proxy evidence.",
    "solve_claim_made": "Prevents the proxy from becoming a solve claim.",
    "offline_reproduced": "Stays false because this is not a new reproduction.",
    "registry_mutated": "Detects forbidden solve-registry changes.",
    "arc_loo_cell_complete_score": "Requires a complete pair, even for a null.",
    "random_seed": "Pins common policy and generation sampling.",
    "reproducibility_checksum": "Detects aggregate artifact mutation.",
    "gate_check_summary": "Names exact expected and observed failure values.",
    "verifier_is_oracle": "Prevents schema checks from being called correctness oracles.",
    "verdict_class": "Uses a closed outcome class with null distinct from blocked.",
    "honest_verdict": "Provides a machine-readable conclusion consistent with evidence.",
    "model_identity_rows": "Binds each request to the cached model identity and chat template.",
    "model_load_receipts": "Proves each arm loaded the declared local model bytes.",
    "registry_precheck_rows": "Records frozen eligibility before any arm outcome exists.",
    "selected_game_id": "Names the frozen public development game without claiming credit.",
    "eligibility_rank": "Pins the registry selection rule before outcomes are read.",
    "action_budget": "Records the configurable environment-action ceiling.",
    "generation_max_tokens": "Records the configurable per-request generation ceiling.",
    "attempt_rows": "Joins each policy attempt to its acceptance and execution result.",
    "raw_output_rows": "Preserves the real model response used by each generation attempt.",
    "parse_rows": "Preserves extraction status separately from raw model output.",
    "reward_rows": "Keeps environment rewards attributable to exact executed actions.",
    "level_rows": "Keeps level changes attributable to exact executed transitions.",
    "truncation_rows": "Makes uniform generation-limit saturation disqualifiable.",
    "import_rows": "Proves the target adapter was absent at the withheld isolation boundary.",
    "forbidden_read_rows": "Records policy-side reads of prohibited target knowledge.",
    "input_difference_rows": "Proves the two arms received different adapter access.",
    "policy_difference_rows": "Proves the adapter access changed the effective control policy.",
    "stop_reason_rows": "Explains why each bounded arm stopped.",
    "visible_control_nonzero_score": "Rejects a control that cannot demonstrate measurement sensitivity.",
    "adapter_access_clean": "Rejects target adapter code or symbols in the withheld policy process.",
    "registry_unchanged_score": "Proves the measurement did not change solve credit.",
    "arc_loop_solve": "Stays false because this task does not invoke the solve loop.",
    "game_level_solve_claimed": "Stays false because public progress is only measurement evidence.",
}
FIELD_PRINCIPLES = {field: _PRINCIPLES[field] for field in REQUIRED_ARTIFACT_FIELDS}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Replace one JSON file atomically without importing project code first."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 for JSON-compatible evidence."""

    return _sha256_bytes(_canonical_bytes(value))


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading large GGUF bytes into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash an artifact after blanking its self-referential checksum."""

    body = deepcopy(dict(artifact))
    body["reproducibility_checksum"] = ""
    return sha256_json(body)


def cap_for_phase(phase: str) -> int:
    """Return the contractually fixed ceiling for one phase."""

    return {
        "artifact_initialization": FINALIZATION_CAP_S,
        "registry_freeze": FINALIZATION_CAP_S,
        "setup": SETUP_CAP_S,
        "adapter_withheld": WITHHELD_ARM_CAP_S,
        "adapter_visible_control": CONTROL_ARM_CAP_S,
        "finalization": FINALIZATION_CAP_S,
    }[phase]


def gate_row(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Preserve both sides of one exact fail-closed check."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def raw_trace_receipt(path: str | Path, *, arm: str, event_count: int) -> JsonDict:
    """Bind one bulky trace outside results by absolute path, size, and digest."""

    resolved = Path(path).resolve()
    return {
        "arm": arm,
        "path": str(resolved),
        "byte_count": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "event_count": int(event_count),
    }


def resolve_headline_model(
    provider: Callable[..., list[Mapping[str, Any]] | None] | None = None,
) -> JsonDict:
    """Resolve only the cached headline Qwen Q4_K_M through ``cached_sota_pair``."""

    if provider is None:
        from carnot.inference.sota_models import cached_sota_pair

        provider = cached_sota_pair
    pair = provider(
        gpu_indices=(0, 1),
        preferred_quant="Q4_K_M",
        model_indices=(0, 1),
    )
    if not pair:
        raise RuntimeError("cached_sota_pair did not return two cached SOTA models")
    selected = next((row for row in pair if row.get("hf_id") == MODEL_REPOSITORY), None)
    if selected is None:
        raise RuntimeError(f"cached_sota_pair did not return {MODEL_REPOSITORY}")
    cached_path = Path(str(selected.get("model_path", "")))
    path = cached_path.resolve()
    if not path.is_file() or "q4_k_m" not in cached_path.name.lower():
        raise RuntimeError("cached Qwen Q4_K_M GGUF is missing; download and substitution forbidden")
    return {
        "model_repository": MODEL_REPOSITORY,
        "model_path": str(path),
        "model_hash": sha256_file(path),
        "model_quantization": "Q4_K_M",
        "llama_cpp_chat_template": "gguf_metadata_qwen3",
        "download_attempted": False,
    }


def freeze_registry_rank_one(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze the first reproduced full-clear registry row before any outcome read."""

    eligible = [
        row
        for row in rows
        if row.get("reproducibility") == "reproduced"
        and row.get("full_game_clear") is True
        and int(row.get("levels_reproduced", 0) or 0) > 0
    ]
    if not eligible:
        raise RuntimeError("registry contains no eligible reproduced full-clear game")
    row = eligible[0]
    return {
        "game": str(row["game"]),
        "target_level": 1,
        "registry_levels_reproduced": int(row["levels_reproduced"]),
        "registry_rank": 1,
    }


def _arm_processes(process_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("role")): row for row in process_rows if row.get("role") in ARM_NAMES}


def _adapter_treatment_exact(process_rows: Sequence[Mapping[str, Any]], game: str) -> bool:
    arms = _arm_processes(process_rows)
    if not arms:
        return True
    if set(arms) != set(ARM_NAMES):
        return False
    withheld = arms["adapter_withheld"]
    control = arms["adapter_visible_control"]
    before = list(withheld.get("adapter_keys_before") or [])
    return bool(
        game
        and game in before
        and list(withheld.get("adapter_keys_after") or []) == [key for key in before if key != game]
        and list(withheld.get("removed_adapters") or []) == [game]
        and list(control.get("adapter_keys_before") or []) == before
        and list(control.get("adapter_keys_after") or []) == before
        and list(control.get("removed_adapters") or []) == []
    )


def _fresh_arm_processes(process_rows: Sequence[Mapping[str, Any]]) -> bool:
    arms = _arm_processes(process_rows)
    if not arms:
        return True
    if set(arms) != set(ARM_NAMES):
        return False
    identities = [
        (int(row.get("pid", -1)), int(row.get("start_ticks", -1)))
        for row in arms.values()
    ]
    return all(row.get("fresh_process") is True for row in arms.values()) and len(set(identities)) == 2


def _real_e3_processes(process_rows: Sequence[Mapping[str, Any]]) -> bool:
    arms = _arm_processes(process_rows)
    if not arms:
        return True
    return set(arms) == set(ARM_NAMES) and all(
        row.get("entrypoint") == REAL_ENTRYPOINT for row in arms.values()
    )


def _complete_phases(rows: Sequence[Mapping[str, Any]]) -> bool:
    return Counter(str(row.get("phase")) for row in rows) == Counter(PHASES)


def _caps_respected(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return all(
            row.get("phase") in PHASES
            and float(row.get("duration_s", -1)) <= cap_for_phase(str(row["phase"]))
            and int(row.get("cap_s", -1)) == cap_for_phase(str(row["phase"]))
            and row.get("timeout_state") == "not_timed_out"
            for row in rows
        )
    except (KeyError, TypeError, ValueError):
        return False


def _rows_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {arm: [row for row in rows if row.get("arm") == arm] for arm in ARM_NAMES}


def _executed_transitions_per_arm(rows: Sequence[Mapping[str, Any]]) -> bool:
    split = _rows_by_arm(rows)
    return all(any(row.get("executed") is True for row in split[arm]) for arm in ARM_NAMES)


def _runtime_receipts_complete(
    request_rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    verifier_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
) -> bool:
    collections = (request_rows, token_rows, proposal_rows, verifier_rows, action_rows, transition_rows)
    return all(all(_rows_by_arm(rows)[arm] for rows in collections) for arm in ARM_NAMES)


def _common_arm_configuration(request_rows: Sequence[Mapping[str, Any]]) -> bool:
    split = _rows_by_arm(request_rows)
    if not request_rows:
        return True
    if not all(split[arm] for arm in ARM_NAMES):
        return True
    fields = (
        "model_repository",
        "model_hash",
        "budget",
        "seed",
        "tools_hash",
        "executable_environment_hash",
    )
    left = split[ARM_NAMES[0]][0]
    right = split[ARM_NAMES[1]][0]
    return all(left.get(field) == right.get(field) for field in fields)


def _raw_manifest_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    split = _rows_by_arm(rows)
    return all(
        split[arm]
        and all(row.get("path") and row.get("sha256") and int(row.get("byte_count", -1)) >= 0 for row in split[arm])
        for arm in ARM_NAMES
    )


def _adapter_access_is_clean(
    import_rows: Sequence[Mapping[str, Any]],
    forbidden_read_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Require one clean isolation and read receipt from each arm."""

    imports = _rows_by_arm(import_rows)
    reads = _rows_by_arm(forbidden_read_rows)
    if not import_rows and not forbidden_read_rows:
        return True
    return all(
        len(imports[arm]) == 1
        and imports[arm][0].get("passed") is True
        and len(reads[arm]) == 1
        and reads[arm][0].get("passed") is True
        and not list(reads[arm][0].get("forbidden_reads") or [])
        for arm in ARM_NAMES
    )


def _measured_difference(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when a receipt records a concrete arm difference."""

    return bool(rows) and any(row.get("measured_difference") is True for row in rows)


def _not_uniformly_truncated(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Reject a run only when every real output saturated its generation limit."""

    real = [row for row in rows if row.get("real_output") is True]
    return bool(real) and not all(row.get("truncated") is True for row in real)


def _configured_budgets_match(
    request_rows: Sequence[Mapping[str, Any]],
    action_budget: int,
    generation_max_tokens: int,
) -> bool:
    """Require every real request to carry the selected paired budget."""

    if not request_rows:
        return True
    return all(
        int((row.get("budget") or {}).get("actions", -1)) == int(action_budget)
        and int((row.get("budget") or {}).get("generation_max_tokens", -1))
        == int(generation_max_tokens)
        for row in request_rows
    )


def _generation_receipt(
    response: Mapping[str, Any],
    *,
    last_generated_tokens: int,
    generation_max_tokens: int,
) -> JsonDict:
    """Read token and stop data from either supported llama response shape."""

    timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
    timing_tokens = int(timings.get("predicted_n", -1) or -1)
    generated_tokens = max(int(last_generated_tokens), timing_tokens)
    choices = response.get("choices")
    choice_reason = (
        choices[0].get("finish_reason")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping)
        else None
    )
    finish_reason = response.get("stop_type") or choice_reason
    real_output = bool(response.get("content") or choices) and generated_tokens >= 0
    return {
        "generated_tokens": generated_tokens,
        "finish_reason": finish_reason,
        "real_output": real_output,
        "truncated": bool(
            real_output
            and (generated_tokens >= int(generation_max_tokens) or finish_reason in {"limit", "length"})
        ),
    }


def _is_forbidden_policy_path(path: str, game: str) -> bool:
    """Identify target knowledge that the policy process must not open."""

    normalized = str(path).replace("\\", "/")
    return any(
        marker in normalized
        for marker in (
            "/ops/arc_solve_registry.yaml",
            "/results/arc_loop_solve_",
            f"/models/arc_verifier_{game}",
            f"/environment_files/{game}/",
        )
    )


def _new_runtime_receipts_complete(evidence: Mapping[str, Any]) -> bool:
    """Require each new V627 receipt family to contain both arms."""

    fields = (
        "attempt_rows",
        "raw_output_rows",
        "parse_rows",
        "reward_rows",
        "level_rows",
        "truncation_rows",
        "model_load_receipts",
        "import_rows",
        "forbidden_read_rows",
        "stop_reason_rows",
    )
    return all(
        all(_rows_by_arm(list(evidence.get(field) or []))[arm] for arm in ARM_NAMES)
        for field in fields
    )


def _level_count(rows: Sequence[Mapping[str, Any]], arm: str) -> int:
    executed = [row for row in rows if row.get("arm") == arm and row.get("executed") is True]
    if not executed:
        return 0
    starts = [int(row.get("level_before", 0) or 0) for row in executed]
    ends = [int(row.get("level_after", 0) or 0) for row in executed]
    return max(0, max(ends) - min(starts))


def _gate_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = next((dict(row) for row in rows if row.get("passed") is not True), None)
    return {
        "checks": [dict(row) for row in rows],
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else None,
        "observed_value": failed.get("observed_value") if failed else None,
        "all_passed": failed is None,
    }


def _projection(evidence: Mapping[str, Any]) -> JsonDict:
    process_rows = list(evidence.get("process_rows") or [])
    phase_rows = list(evidence.get("phase_receipt_rows") or [])
    request_rows = list(evidence.get("request_rows") or [])
    token_rows = list(evidence.get("token_rows") or [])
    proposal_rows = list(evidence.get("proposal_rows") or [])
    verifier_rows = list(evidence.get("verifier_rows") or [])
    action_rows = list(evidence.get("action_rows") or [])
    transition_rows = list(evidence.get("transition_rows") or [])
    raw_manifest = list(evidence.get("raw_trace_manifest") or [])
    import_rows = list(evidence.get("import_rows") or [])
    forbidden_read_rows = list(evidence.get("forbidden_read_rows") or [])
    input_difference_rows = list(evidence.get("input_difference_rows") or [])
    policy_difference_rows = list(evidence.get("policy_difference_rows") or [])
    truncation_rows = list(evidence.get("truncation_rows") or [])
    action_budget = int(evidence.get("action_budget", DEFAULT_ACTION_BUDGET) or 0)
    generation_max_tokens = int(
        evidence.get("generation_max_tokens", DEFAULT_GENERATION_MAX_TOKENS) or 0
    )
    selected = dict(evidence.get("selected_game") or {})
    game = str(selected.get("game") or "")
    source_hashes = dict(evidence.get("source_artifact_hashes") or {})
    registry_before = source_hashes.get(REGISTRY_RELATIVE_PATH.as_posix())
    registry_after = evidence.get("registry_hash_after", source_hashes.get("registry_hash_after"))
    registry_mutated = bool(registry_before and registry_after and registry_before != registry_after)
    adapter_exact = _adapter_treatment_exact(process_rows, game)
    real_e3 = _real_e3_processes(process_rows)
    fresh = _fresh_arm_processes(process_rows)
    phases_complete = _complete_phases(phase_rows)
    caps_ok = _caps_respected(phase_rows)
    attempts = _executed_transitions_per_arm(transition_rows)
    runtime = _runtime_receipts_complete(
        request_rows, token_rows, proposal_rows, verifier_rows, action_rows, transition_rows
    )
    common = _common_arm_configuration(request_rows)
    raw_complete = _raw_manifest_complete(raw_manifest)
    preconditions = list(evidence.get("preconditions_checked") or [])
    preconditions_ok = bool(preconditions) and all(row.get("passed") is True for row in preconditions)
    new_runtime = _new_runtime_receipts_complete(evidence)
    adapter_access_clean = _adapter_access_is_clean(import_rows, forbidden_read_rows)
    input_difference = _measured_difference(input_difference_rows)
    policy_difference = _measured_difference(policy_difference_rows)
    truncation_ok = _not_uniformly_truncated(truncation_rows)
    budgets_match = _configured_budgets_match(
        request_rows, action_budget, generation_max_tokens
    )
    withheld_levels = _level_count(transition_rows, ARM_NAMES[0])
    control_levels = _level_count(transition_rows, ARM_NAMES[1])
    visible_control_nonzero = control_levels > 0
    registry_unchanged = bool(registry_before and registry_after and not registry_mutated)
    checks = [dict(row) for row in preconditions]
    if not checks:
        checks.append(gate_row("all_runtime_preconditions", True, False))
    checks.extend([
        gate_row("adapter_withheld_exactly", True, adapter_exact),
        gate_row("real_e3_entrypoint_used", True, real_e3),
        gate_row("fresh_process_per_arm", True, fresh),
        gate_row("phase_caps_respected", True, caps_ok),
        gate_row("registry_mutated", False, registry_mutated),
        gate_row("common_arm_configuration", True, common),
        gate_row("complete_phase_receipts", True, phases_complete),
        gate_row("executed_transition_per_arm", True, attempts),
        gate_row("complete_runtime_receipts", True, runtime),
        gate_row("complete_v627_runtime_receipts", True, new_runtime),
        gate_row("complete_raw_trace_manifest", True, raw_complete),
        gate_row("adapter_access_clean", True, adapter_access_clean),
        gate_row("visible_control_nonzero", True, visible_control_nonzero),
        gate_row("not_all_real_outputs_truncated", True, truncation_ok),
        gate_row("measured_input_difference", True, input_difference),
        gate_row("measured_policy_difference", True, policy_difference),
        gate_row("configured_budgets_match", True, budgets_match),
        gate_row("registry_unchanged", True, registry_unchanged),
    ])
    failed = next((row for row in checks if row["passed"] is not True), None)
    disqualifying = {
        "adapter_withheld_exactly",
        "real_e3_entrypoint_used",
        "fresh_process_per_arm",
        "phase_caps_respected",
        "registry_mutated",
        "common_arm_configuration",
        "adapter_access_clean",
        "visible_control_nonzero",
        "not_all_real_outputs_truncated",
        "measured_input_difference",
        "measured_policy_difference",
        "configured_budgets_match",
        "registry_unchanged",
    }
    if failed is None:
        verdict = "positive" if withheld_levels > control_levels else "null"
        honest = (
            "complete_positive_development_proxy_level_delta_measured_no_solve_claim"
            if verdict == "positive"
            else "complete_null_rebudgeted_pair_working_control_no_solve_claim"
        )
    elif failed["check"] in disqualifying:
        verdict = "disqualified"
        honest = f"disqualified_{failed['check']}"
    else:
        verdict = "blocked"
        honest = f"blocked_{failed['check']}"
    rows = [
        {
            "game": game,
            "arm": arm,
            "target_level": int(selected.get("target_level", 1) or 1),
            "levels": _level_count(transition_rows, arm),
            "executed_transition_count": sum(
                row.get("arm") == arm and row.get("executed") is True for row in transition_rows
            ),
            "solve_provenance": "development_proxy",
            "solve_claim_made": False,
            "offline_reproduced": False,
        }
        for arm in ARM_NAMES
    ]
    arm_rows = [
        {
            "arm": arm,
            "request_count": len(_rows_by_arm(request_rows)[arm]),
            "proposal_count": len(_rows_by_arm(proposal_rows)[arm]),
            "action_count": sum(
                row.get("arm") == arm and row.get("executed") is True for row in action_rows
            ),
            "transition_count": sum(
                row.get("arm") == arm and row.get("executed") is True for row in transition_rows
            ),
            "levels": _level_count(transition_rows, arm),
        }
        for arm in ARM_NAMES
    ]
    any_generation = any(row.get("generation_invoked") is True for row in request_rows)
    return {
        "models_used": [MODEL_REPOSITORY] if any_generation else [],
        "inference_substrate_class": "model_full_generation" if any_generation else "blocked_no_run",
        "rows": rows,
        "per_game_results": [
            {
                "game": game,
                "withheld_levels": withheld_levels,
                "control_levels": control_levels,
                "level_delta": withheld_levels - control_levels,
                "solve_provenance": "development_proxy",
                "solve_claim_made": False,
                "offline_reproduced": False,
            }
        ],
        "arm_rows": arm_rows,
        "adapter_withheld_exactly": adapter_exact,
        "real_e3_entrypoint_used": real_e3,
        "fresh_process_per_arm": fresh,
        "withheld_levels": withheld_levels,
        "control_levels": control_levels,
        "level_delta": withheld_levels - control_levels,
        "registry_mutated": registry_mutated,
        "visible_control_nonzero_score": 1 if visible_control_nonzero else 0,
        "adapter_access_clean": adapter_access_clean,
        "registry_unchanged_score": 1 if registry_unchanged else 0,
        "arc_loo_cell_complete_score": 1 if failed is None else 0,
        "gate_check_summary": _gate_summary(checks),
        "verdict_class": verdict,
        "honest_verdict": honest,
    }


def build_artifact(
    *,
    run_date: str,
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    model: Mapping[str, Any],
    selected_game: Mapping[str, Any],
    registry_rank_before_outcomes: int,
    registry_hash_after: str | None,
    phase_receipt_rows: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    verifier_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    raw_trace_manifest: Sequence[Mapping[str, Any]],
    gpu_telemetry_rows: Sequence[Mapping[str, Any]],
    duration_s: float,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    generation_max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS,
    model_identity_rows: Sequence[Mapping[str, Any]] = (),
    model_load_receipts: Sequence[Mapping[str, Any]] = (),
    registry_precheck_rows: Sequence[Mapping[str, Any]] = (),
    attempt_rows: Sequence[Mapping[str, Any]] = (),
    raw_output_rows: Sequence[Mapping[str, Any]] = (),
    parse_rows: Sequence[Mapping[str, Any]] = (),
    reward_rows: Sequence[Mapping[str, Any]] = (),
    level_rows: Sequence[Mapping[str, Any]] = (),
    truncation_rows: Sequence[Mapping[str, Any]] = (),
    import_rows: Sequence[Mapping[str, Any]] = (),
    forbidden_read_rows: Sequence[Mapping[str, Any]] = (),
    input_difference_rows: Sequence[Mapping[str, Any]] = (),
    policy_difference_rows: Sequence[Mapping[str, Any]] = (),
    stop_reason_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one immutable aggregate from raw process and transition evidence."""

    evidence = {
        "preconditions_checked": deepcopy(list(preconditions_checked)),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "model": deepcopy(dict(model)),
        "selected_game": deepcopy(dict(selected_game)),
        "registry_rank_before_outcomes": int(registry_rank_before_outcomes),
        "registry_hash_after": registry_hash_after,
        "phase_receipt_rows": deepcopy(list(phase_receipt_rows)),
        "process_rows": deepcopy(list(process_rows)),
        "request_rows": deepcopy(list(request_rows)),
        "token_rows": deepcopy(list(token_rows)),
        "proposal_rows": deepcopy(list(proposal_rows)),
        "verifier_rows": deepcopy(list(verifier_rows)),
        "action_rows": deepcopy(list(action_rows)),
        "transition_rows": deepcopy(list(transition_rows)),
        "raw_trace_manifest": deepcopy(list(raw_trace_manifest)),
        "action_budget": int(action_budget),
        "generation_max_tokens": int(generation_max_tokens),
        "model_identity_rows": deepcopy(list(model_identity_rows)),
        "model_load_receipts": deepcopy(list(model_load_receipts)),
        "registry_precheck_rows": deepcopy(list(registry_precheck_rows)),
        "attempt_rows": deepcopy(list(attempt_rows)),
        "raw_output_rows": deepcopy(list(raw_output_rows)),
        "parse_rows": deepcopy(list(parse_rows)),
        "reward_rows": deepcopy(list(reward_rows)),
        "level_rows": deepcopy(list(level_rows)),
        "truncation_rows": deepcopy(list(truncation_rows)),
        "import_rows": deepcopy(list(import_rows)),
        "forbidden_read_rows": deepcopy(list(forbidden_read_rows)),
        "input_difference_rows": deepcopy(list(input_difference_rows)),
        "policy_difference_rows": deepcopy(list(policy_difference_rows)),
        "stop_reason_rows": deepcopy(list(stop_reason_rows)),
    }
    if registry_hash_after is not None:
        evidence["source_artifact_hashes"]["registry_hash_after"] = registry_hash_after
    projection = _projection(evidence)
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": evidence["preconditions_checked"],
        "run_date": str(run_date),
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_identity_rows": evidence["model_identity_rows"],
        "model_load_receipts": evidence["model_load_receipts"],
        "registry_precheck_rows": evidence["registry_precheck_rows"],
        "selected_game_id": str(selected_game.get("game") or ""),
        "eligibility_rank": int(registry_rank_before_outcomes),
        "action_budget": evidence["action_budget"],
        "generation_max_tokens": evidence["generation_max_tokens"],
        "models_used": projection["models_used"],
        "model_repository": model.get("model_repository"),
        "model_path": model.get("model_path"),
        "model_hash": model.get("model_hash"),
        "model_quantization": model.get("model_quantization"),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": projection["inference_substrate_class"],
        "execution_venue": "host",
        "gpu_telemetry_rows": deepcopy(list(gpu_telemetry_rows)),
        "token_rows": evidence["token_rows"],
        "attempt_rows": evidence["attempt_rows"],
        "raw_output_rows": evidence["raw_output_rows"],
        "parse_rows": evidence["parse_rows"],
        "reward_rows": evidence["reward_rows"],
        "level_rows": evidence["level_rows"],
        "truncation_rows": evidence["truncation_rows"],
        "import_rows": evidence["import_rows"],
        "forbidden_read_rows": evidence["forbidden_read_rows"],
        "input_difference_rows": evidence["input_difference_rows"],
        "policy_difference_rows": evidence["policy_difference_rows"],
        "stop_reason_rows": evidence["stop_reason_rows"],
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": evidence["source_artifact_hashes"],
        "raw_trace_manifest": evidence["raw_trace_manifest"],
        "rows": projection["rows"],
        "per_game_results": projection["per_game_results"],
        "phase_receipt_rows": evidence["phase_receipt_rows"],
        "process_rows": evidence["process_rows"],
        "request_rows": evidence["request_rows"],
        "proposal_rows": evidence["proposal_rows"],
        "verifier_rows": evidence["verifier_rows"],
        "action_rows": evidence["action_rows"],
        "transition_rows": evidence["transition_rows"],
        "arm_rows": projection["arm_rows"],
        "selected_game": evidence["selected_game"],
        "registry_rank_before_outcomes": evidence["registry_rank_before_outcomes"],
        "adapter_withheld_exactly": projection["adapter_withheld_exactly"],
        "real_e3_entrypoint_used": projection["real_e3_entrypoint_used"],
        "fresh_process_per_arm": projection["fresh_process_per_arm"],
        "setup_cap_s": SETUP_CAP_S,
        "withheld_arm_cap_s": WITHHELD_ARM_CAP_S,
        "control_arm_cap_s": CONTROL_ARM_CAP_S,
        "finalization_cap_s": FINALIZATION_CAP_S,
        "withheld_levels": projection["withheld_levels"],
        "control_levels": projection["control_levels"],
        "level_delta": projection["level_delta"],
        "solve_provenance": "development_proxy",
        "solve_claim_made": False,
        "offline_reproduced": False,
        "registry_mutated": projection["registry_mutated"],
        "visible_control_nonzero_score": projection["visible_control_nonzero_score"],
        "adapter_access_clean": projection["adapter_access_clean"],
        "registry_unchanged_score": projection["registry_unchanged_score"],
        "arc_loop_solve": False,
        "game_level_solve_claimed": False,
        "arc_loo_cell_complete_score": projection["arc_loo_cell_complete_score"],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": projection["gate_check_summary"],
        "verifier_is_oracle": False,
        "verdict_class": projection["verdict_class"],
        "honest_verdict": projection["honest_verdict"],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def initialize_terminal_artifact(
    path: str | Path,
    *,
    run_date: str,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    generation_max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS,
) -> JsonDict:
    """Write a schema-complete no-run block before any model or worker setup."""

    artifact = build_artifact(
        run_date=run_date,
        preconditions_checked=[gate_row("all_runtime_preconditions", True, False)],
        source_artifact_hashes={},
        model={},
        selected_game={},
        registry_rank_before_outcomes=0,
        registry_hash_after=None,
        phase_receipt_rows=[],
        process_rows=[],
        request_rows=[],
        token_rows=[],
        proposal_rows=[],
        verifier_rows=[],
        action_rows=[],
        transition_rows=[],
        raw_trace_manifest=[],
        gpu_telemetry_rows=[],
        duration_s=0.0,
        action_budget=action_budget,
        generation_max_tokens=generation_max_tokens,
    )
    _atomic_write_json(path, artifact)
    return artifact


_PROJECTION_FIELDS = (
    "models_used",
    "inference_substrate_class",
    "rows",
    "per_game_results",
    "arm_rows",
    "adapter_withheld_exactly",
    "real_e3_entrypoint_used",
    "fresh_process_per_arm",
    "withheld_levels",
    "control_levels",
    "level_delta",
    "registry_mutated",
    "visible_control_nonzero_score",
    "adapter_access_clean",
    "registry_unchanged_score",
    "arc_loo_cell_complete_score",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
)


def validate_artifact(artifact: Mapping[str, Any], *, verify_raw_traces: bool = False) -> list[str]:
    """Recompute every scientific aggregate and optionally rehash raw trace bytes."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_artifact_fields")
    if set(artifact.get("field_principles") or {}) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("solve_provenance") != "development_proxy" or artifact.get("solve_claim_made") is not False:
        errors.append("solve_provenance_mismatch")
    if artifact.get("offline_reproduced") is not False or artifact.get("verifier_is_oracle") is not False:
        errors.append("nonclaim_fields_mismatch")
    if artifact.get("arc_loop_solve") is not False or artifact.get("game_level_solve_claimed") is not False:
        errors.append("v627_nonclaim_fields_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if any(
        row.get("solve_provenance") != "development_proxy"
        or row.get("solve_claim_made") is not False
        or row.get("offline_reproduced") is not False
        for row in artifact.get("rows") or []
    ):
        errors.append("level_row_provenance_mismatch")
    evidence = {
        "preconditions_checked": artifact.get("preconditions_checked") or [],
        "source_artifact_hashes": artifact.get("source_artifact_hashes") or {},
        "model": {
            "model_repository": artifact.get("model_repository"),
            "model_path": artifact.get("model_path"),
            "model_hash": artifact.get("model_hash"),
            "model_quantization": artifact.get("model_quantization"),
        },
        "selected_game": artifact.get("selected_game") or {},
        "registry_rank_before_outcomes": artifact.get("registry_rank_before_outcomes", 0),
        "registry_hash_after": (artifact.get("source_artifact_hashes") or {}).get("registry_hash_after"),
        "phase_receipt_rows": artifact.get("phase_receipt_rows") or [],
        "process_rows": artifact.get("process_rows") or [],
        "request_rows": artifact.get("request_rows") or [],
        "token_rows": artifact.get("token_rows") or [],
        "proposal_rows": artifact.get("proposal_rows") or [],
        "verifier_rows": artifact.get("verifier_rows") or [],
        "action_rows": artifact.get("action_rows") or [],
        "transition_rows": artifact.get("transition_rows") or [],
        "raw_trace_manifest": artifact.get("raw_trace_manifest") or [],
        "action_budget": artifact.get("action_budget", DEFAULT_ACTION_BUDGET),
        "generation_max_tokens": artifact.get(
            "generation_max_tokens", DEFAULT_GENERATION_MAX_TOKENS
        ),
        "model_identity_rows": artifact.get("model_identity_rows") or [],
        "model_load_receipts": artifact.get("model_load_receipts") or [],
        "registry_precheck_rows": artifact.get("registry_precheck_rows") or [],
        "attempt_rows": artifact.get("attempt_rows") or [],
        "raw_output_rows": artifact.get("raw_output_rows") or [],
        "parse_rows": artifact.get("parse_rows") or [],
        "reward_rows": artifact.get("reward_rows") or [],
        "level_rows": artifact.get("level_rows") or [],
        "truncation_rows": artifact.get("truncation_rows") or [],
        "import_rows": artifact.get("import_rows") or [],
        "forbidden_read_rows": artifact.get("forbidden_read_rows") or [],
        "input_difference_rows": artifact.get("input_difference_rows") or [],
        "policy_difference_rows": artifact.get("policy_difference_rows") or [],
        "stop_reason_rows": artifact.get("stop_reason_rows") or [],
    }
    projection = _projection(evidence)
    if any(artifact.get(field) != projection[field] for field in _PROJECTION_FIELDS):
        errors.append("artifact_projection_mismatch")
    if verify_raw_traces:
        for row in artifact.get("raw_trace_manifest") or []:
            path = Path(str(row.get("path") or ""))
            try:
                valid = path.is_file() and path.stat().st_size == int(row.get("byte_count", -1)) and sha256_file(path) == row.get("sha256")
            except (OSError, TypeError, ValueError):
                valid = False
            if not valid:
                errors.append("raw_trace_hash_mismatch")
                break
    return errors


# Live orchestration follows below.  It is excluded from unit coverage because
# its meaningful assertions are subprocess, filesystem, ARC SDK, and GPU
# receipts rather than alternate Python branches.


def _progress(kind: str, **values: Any) -> None:  # pragma: no cover
    """Print one compact, flushed progress receipt for the outer conductor."""

    print(json.dumps({"progress": kind, **values}, sort_keys=True, default=str), flush=True)


def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:  # pragma: no cover
    """Durably append one event before the producer moves to the next event."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _process_start_ticks(pid: int) -> int:  # pragma: no cover
    try:
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])
    except (OSError, IndexError, ValueError):
        return -1


def _phase_receipt(
    phase: str,
    started_ns: int,
    ended_ns: int,
    *,
    pid: int,
    start_ticks: int,
    exit_state: str,
    timed_out: bool,
    stop_reason: str,
) -> JsonDict:  # pragma: no cover
    core = {
        "phase": phase,
        "process_pid": int(pid),
        "process_start_ticks": int(start_ticks),
        "started_monotonic_ns": int(started_ns),
        "ended_monotonic_ns": int(ended_ns),
        "duration_s": round((ended_ns - started_ns) / 1_000_000_000, 6),
        "cap_s": cap_for_phase(phase),
        "exit_state": exit_state,
        "timeout_state": "timed_out_descendants_killed" if timed_out else "not_timed_out",
        "stop_reason": stop_reason,
    }
    return {**core, "evidence_hash": sha256_json(core)}


def _free_port() -> int:  # pragma: no cover
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _gpu_rows() -> list[JsonDict]:  # pragma: no cover
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    busy = {line.split(",", 1)[0].strip() for line in apps.stdout.splitlines() if "," in line}
    rows: list[JsonDict] = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            row = {
                "gpu": int(parts[0]),
                "model": parts[1],
                "uuid": parts[2],
                "memory_total_mb": int(parts[3]),
                "memory_used_mb": int(parts[4]),
                "memory_free_mb": int(parts[5]),
                "utilization_pct": int(parts[6]),
                "sample_ok": True,
                "sample_phase": "preflight",
            }
        except ValueError:
            continue
        row["idle"] = bool(
            row["model"] == "NVIDIA GeForce RTX 3090"
            and row["uuid"] not in busy
            and row["memory_free_mb"] >= 20_000
            and row["utilization_pct"] <= 5
        )
        rows.append(row)
    return rows


def _inside_inference_gpu_rows(server_pid: int) -> list[JsonDict]:  # pragma: no cover
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows: list[JsonDict] = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) == server_pid:
            rows.append(
                {
                    "gpu_uuid": parts[0],
                    "server_pid": int(parts[1]),
                    "memory_used_mb": int(parts[2]),
                    "inside_inference": True,
                    "sample_ok": True,
                }
            )
    return rows


def _llama_cpp_receipt() -> JsonDict:  # pragma: no cover
    default_cuda = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    binary = Path(os.environ.get("CARNOT_LLAMA_SERVER", str(default_cuda))).resolve()
    if binary.is_file():
        os.environ["CARNOT_LLAMA_SERVER"] = str(binary)
    linked = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=False)
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=False)
    return {
        "path": str(binary),
        "exists": binary.is_file() and os.access(binary, os.X_OK),
        "cuda_linked": "libggml-cuda" in linked.stdout and "libcuda.so" in linked.stdout,
        "version_returncode": version.returncode,
        "version": (version.stdout + version.stderr).strip()[:500],
        "sha256": sha256_file(binary) if binary.is_file() else None,
    }


def _load_registry(path: Path) -> list[JsonDict]:  # pragma: no cover
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = value.get("games") if isinstance(value, Mapping) else None
    if not isinstance(rows, list):
        raise RuntimeError("registry games list is missing")
    return [dict(row) for row in rows]


def _source_hashes(registry: Path) -> JsonDict:  # pragma: no cover
    """Hash protocol code without inspecting the selected game's source."""

    paths = (
        registry,
        Path(__file__).resolve(),
        ROOT / "scripts/experiments/experiment_7127_v626_adapter_withheld_arc_loo.py",
        ROOT / "python/carnot/agentic/arc_competition_agent.py",
        ROOT / "python/carnot/agentic/arc_executable_world_model.py",
        ROOT / "python/carnot/inference/sota_models.py",
    )
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths if path.is_file()}


def _run_worker(
    *,
    role: str,
    payload: Mapping[str, Any],
    raw_root: Path,
    cap_s: int,
    parent_trace: Path,
) -> tuple[JsonDict, JsonDict, int, int]:  # pragma: no cover
    payload_path = raw_root / f"{role}-payload.json"
    output_path = raw_root / f"{role}-worker-output.json"
    stdout_path = raw_root / f"{role}.stdout.log"
    stderr_path = raw_root / f"{role}.stderr.log"
    _atomic_write_json(payload_path, dict(payload))
    env = dict(os.environ)
    python_root = str(ROOT / "python")
    env["PYTHONPATH"] = python_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_7127_v626_adapter_withheld_arc_loo",
        "--worker",
        role,
        "--payload",
        str(payload_path),
        "--worker-output",
        str(output_path),
    ]
    started_ns = time.monotonic_ns()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=True,
        )
        start_ticks = _process_start_ticks(process.pid)
        _append_jsonl(
            parent_trace,
            {
                "kind": "process_started",
                "role": role,
                "pid": process.pid,
                "start_ticks": start_ticks,
                "command_hash": sha256_json(command),
            },
        )
        timed_out = False
        stop_reason = "completed"
        _progress("worker_started", role=role, pid=process.pid, cap_s=cap_s)
        deadline = time.monotonic() + cap_s
        read_offset = 0
        last_heartbeat = time.monotonic()
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.25)
            try:
                with stdout_path.open("r", encoding="utf-8") as reader:
                    reader.seek(read_offset)
                    chunk = reader.read()
                    read_offset = reader.tell()
            except OSError:
                chunk = ""
            for line in chunk.splitlines():
                print(line, flush=True)
                last_heartbeat = time.monotonic()
            if time.monotonic() - last_heartbeat >= 30:
                _progress("worker_heartbeat", role=role, pid=process.pid)
                last_heartbeat = time.monotonic()
        if process.poll() is None:
            timed_out = True
            stop_reason = f"timeout_after_{cap_s}s_process_group_sigterm_then_sigkill"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait(timeout=10)
        else:
            exit_code = int(process.returncode)
        try:
            with stdout_path.open("r", encoding="utf-8") as reader:
                reader.seek(read_offset)
                for line in reader.read().splitlines():
                    print(line, flush=True)
        except OSError:
            pass
    ended_ns = time.monotonic_ns()
    result: JsonDict = {}
    if output_path.is_file():
        try:
            value = json.loads(output_path.read_text(encoding="utf-8"))
            result = dict(value) if isinstance(value, Mapping) else {}
        except (OSError, json.JSONDecodeError):
            result = {}
    if not timed_out and exit_code != 0:
        stop_reason = f"worker_exit_code_{exit_code}"
    row: JsonDict = {
        "role": role,
        "pid": process.pid,
        "start_ticks": start_ticks,
        "fresh_process": process.pid != os.getpid() and start_ticks >= 0,
        "exit_code": int(exit_code),
        "timed_out": timed_out,
        "stop_reason": stop_reason,
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
    }
    for field in ("entrypoint", "adapter_keys_before", "adapter_keys_after", "removed_adapters"):
        if field in result:
            row[field] = deepcopy(result[field])
    _append_jsonl(parent_trace, {"kind": "process_finished", **row})
    _progress("worker_finished", role=role, pid=process.pid, exit_code=exit_code)
    return row, result, started_ns, ended_ns


def _setup_worker(payload: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
    from carnot.agentic.arc_game_adapters import _BUILDERS
    from carnot.inference.sota_models import gguf_tokenizer_loadable

    model_path = str(payload["model"]["model_path"])
    tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(model_path)
    game = str(payload["selected_game"]["game"])
    return {
        "setup_ok": bool(
            tokenizer_ok and game in _BUILDERS and callable(make_carnot_agent) and E3AgentPolicy
        ),
        "embedded_tokenizer_loadable": tokenizer_ok,
        "embedded_tokenizer_detail": tokenizer_detail,
        "selected_adapter_present": game in _BUILDERS,
        "factory_callable": callable(make_carnot_agent),
        "policy_class": E3AgentPolicy.__name__,
    }


def _load_framework_agent() -> Any:  # pragma: no cover
    from carnot.experiment_6681_arc_post_redirect_outcomes import _load_framework_agent as load

    return load()


def _normal_observation(value: Any) -> JsonDict:  # pragma: no cover
    from carnot.agentic.arc_e3_outcome_transport import normalize_observation

    normalized = normalize_observation(value) or {}
    return {
        key: normalized.get(key)
        for key in (
            "frame",
            "state",
            "levels_completed",
            "win_levels",
            "full_reset",
            "available_actions",
        )
    }


def _action_record(action: Any) -> tuple[int, JsonDict | None]:  # pragma: no cover
    name = str(getattr(action, "name", ""))
    action_number = 0 if name == "RESET" else int(name.removeprefix("ACTION"))
    raw = action.action_data.model_dump() if getattr(action, "action_data", None) is not None else {}
    data = {key: value for key, value in raw.items() if key != "game_id" and value is not None}
    return action_number, data or None


def _json_safe(value: Any) -> Any:  # pragma: no cover
    """Convert model and framework values into durable JSON data."""

    return json.loads(json.dumps(value, default=str))


def _target_adapter_state(game: str) -> JsonDict:  # pragma: no cover
    """Inspect loaded modules without importing the target adapter module."""

    module_name = "carnot.agentic.arc_game_adapters"
    module = sys.modules.get(module_name)
    markers = (f"_{game}", f"{game.upper()}_L1_LABELS", f"{game.upper()}_L2_TAIL_LABELS")
    namespace = vars(module) if module is not None else {}
    return {
        "module": module_name,
        "module_loaded": module is not None,
        "target_recipe_symbols": sorted(marker for marker in markers if marker in namespace),
    }


class _TargetAdapterImportBlocker:  # pragma: no cover
    """Reject a late target-adapter import in the withheld worker."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        del path, target
        if fullname == "carnot.agentic.arc_game_adapters":
            self.attempts.append(fullname)
            raise ImportError("target adapter module is withheld for this measurement arm")
        return None


def _framework_action(action_number: int, data: Mapping[str, Any] | None, game: str) -> Any:  # pragma: no cover
    """Build the same framework action object returned by the scored agent."""

    from arcengine import GameAction

    action = GameAction.RESET if action_number == 0 else getattr(GameAction, f"ACTION{action_number}")
    if data and action_number != 0:
        action.set_data({"game_id": game, **dict(data)})
    return action


def _arm_worker(payload: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    arm = str(payload["arm"])
    game = str(payload["selected_game"]["game"])
    model = dict(payload["model"])
    budget = dict(payload["budget"])
    trace_path = Path(str(payload["trace_path"]))
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_CUDA_GPU": "0,1",
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_INDUCE_N_CTX": str(budget["context_tokens"]),
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(budget["generation_max_tokens"]),
            "CARNOT_ARC_INDUCE_TIMEOUT": str(budget["generation_timeout_s"]),
            "CARNOT_ARC_INDUCE_THINK": "0",
            "CARNOT_ARC_INDUCE_DEFECT_REASKS": "0",
            "CARNOT_ARC_INDUCE_GOAL_DEFECT_REASKS": "0",
            "CARNOT_ARC_LLAMA_SERVER_SLOTS": "1",
            "CARNOT_ARC_LLAMA_SERVER_PARALLEL": "1",
            "CARNOT_ARC_E3_DIR": str(Path(str(payload["raw_root"])) / arm / "e3"),
            "CARNOT_ARC_LIVENESS_DIR": str(Path(str(payload["raw_root"])) / arm / "liveness"),
        }
    )
    isolation = _target_adapter_state(game)
    modules_at_isolation = sorted(
        name for name in sys.modules if name.startswith("carnot.agentic")
    )
    _append_jsonl(
        trace_path,
        {
            "kind": "isolation",
            "arm": arm,
            "stage": "before_policy_import_and_generation",
            **isolation,
            "modules": modules_at_isolation,
        },
    )
    _progress(
        "isolation_receipt",
        arm=arm,
        target_adapter_module_loaded=isolation["module_loaded"],
        target_recipe_symbols=isolation["target_recipe_symbols"],
    )
    blocker = _TargetAdapterImportBlocker() if arm == "adapter_withheld" else None
    if blocker is not None:
        sys.meta_path.insert(0, blocker)
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_solver_kit import offline_arcade

    adapter = None
    declared_loaded_after_receipt = False
    if arm == "adapter_visible_control":
        from carnot.agentic import arc_game_adapters

        adapter = arc_game_adapters.get_adapter(game)
        if adapter is None:
            raise RuntimeError(f"visible control could not load declared adapter: {game}")
        declared_loaded_after_receipt = True
    after_declared_load = _target_adapter_state(game)
    before_adapters = sorted(str(item) for item in payload.get("adapter_roster") or [])
    removed = [game] if arm == "adapter_withheld" else []
    after_adapters = (
        [key for key in before_adapters if key != game]
        if arm == "adapter_withheld"
        else list(before_adapters)
    )
    _append_jsonl(
        trace_path,
        {
            "kind": "adapter",
            "arm": arm,
            "adapter_keys_before": before_adapters,
            "adapter_keys_after": after_adapters,
            "removed_adapters": removed,
        },
    )
    import_row = {
        "arm": arm,
        "receipt_stage": "before_generation",
        "target_adapter_module_loaded": bool(isolation["module_loaded"]),
        "target_recipe_symbols_loaded": bool(isolation["target_recipe_symbols"]),
        "modules_at_isolation": modules_at_isolation,
        "declared_adapter_loaded_after_receipt": declared_loaded_after_receipt,
        "declared_adapter_game": game if declared_loaded_after_receipt else None,
        "target_adapter_module_loaded_after_receipt": bool(after_declared_load["module_loaded"]),
        "target_recipe_symbols_after_receipt": after_declared_load["target_recipe_symbols"],
        "import_block_attempts": [],
        "passed": False,
    }
    proposer = LocalGGUFProposer(
        repo_substr="Qwen3.6-35B-A3B",
        model_path=str(model["model_path"]),
        model_repository=MODEL_REPOSITORY,
        model_filename=Path(str(model["model_path"])).name,
        requested_model_path=str(model["model_path"]),
        requested_model_filename=Path(str(model["model_path"])).name,
        port=int(payload["port"]),
        n_ctx=int(budget["context_tokens"]),
        max_tokens=int(budget["generation_max_tokens"]),
        timeout=int(budget["generation_timeout_s"]),
        mtp=False,
        kv_quant="q8_0",
        n_gpu_layers=999,
        tries=1,
        use_chat_template=True,
    )
    request_rows: list[JsonDict] = []
    token_rows: list[JsonDict] = []
    raw_output_rows: list[JsonDict] = []
    parse_rows: list[JsonDict] = []
    truncation_rows: list[JsonDict] = []
    attempt_rows: list[JsonDict] = []
    proposal_rows: list[JsonDict] = []
    verifier_rows: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    transition_rows: list[JsonDict] = []
    reward_rows: list[JsonDict] = []
    level_rows: list[JsonDict] = []
    model_load_receipts: list[JsonDict] = []
    tools_hash = sha256_json(["offline_arcade.step", REAL_ENTRYPOINT])
    environment_hash = sha256_json(
        {
            "game": game,
            "factory": REAL_ENTRYPOINT,
            "operation_mode": "offline",
        }
    )
    original_chat = proposer._chat_complete_request

    def traced_chat(prompt: str, **kwargs: Any) -> tuple[dict, str]:
        request_id = f"{arm}-request-{len(request_rows)}"
        started_ns = time.monotonic_ns()
        _append_jsonl(
            trace_path,
            {
                "kind": "request_started",
                "arm": arm,
                "request_id": request_id,
                "prompt": prompt,
                "parameters": kwargs,
            },
        )
        _progress("model_request", arm=arm, request_id=request_id, prompt_hash=sha256_json(prompt))
        try:
            response, extraction = original_chat(prompt, **kwargs)
            error = None
        except Exception as exc:
            response, extraction = {}, ""
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ended_ns = time.monotonic_ns()
            row = {
                "arm": arm,
                "request_id": request_id,
                "model_repository": MODEL_REPOSITORY,
                "model_hash": model["model_hash"],
                "prompt_hash": sha256_json(prompt),
                "budget": deepcopy(budget),
                "seed": RANDOM_SEED,
                "tools_hash": tools_hash,
                "executable_environment_hash": environment_hash,
                "generation_invoked": True,
                "response_hash": sha256_json(response),
                "duration_s": round((ended_ns - started_ns) / 1_000_000_000, 6),
                "error": error,
            }
            generation = _generation_receipt(
                response if isinstance(response, Mapping) else {},
                last_generated_tokens=int(proposer.last_generated_tokens),
                generation_max_tokens=int(budget["generation_max_tokens"]),
            )
            token_row = {
                "arm": arm,
                "request_id": request_id,
                "prompt_tokens": int(proposer.last_prompt_tokens),
                "generated_tokens": generation["generated_tokens"],
            }
            safe_response = _json_safe(response)
            raw_row = {
                "arm": arm,
                "request_id": request_id,
                "response": safe_response,
                "response_hash": row["response_hash"],
                "real_output": generation["real_output"],
            }
            parse_row = {
                "arm": arm,
                "request_id": request_id,
                "parse_status": "extracted" if extraction else "empty",
                "extraction_text": str(extraction),
                "parsed_action_count": 0,
            }
            truncation_row = {
                "arm": arm,
                "request_id": request_id,
                "generated_tokens": generation["generated_tokens"],
                "generation_max_tokens": int(budget["generation_max_tokens"]),
                "finish_reason": generation["finish_reason"],
                "real_output": generation["real_output"],
                "truncated": generation["truncated"],
            }
            request_rows.append(row)
            token_rows.append(token_row)
            raw_output_rows.append(raw_row)
            parse_rows.append(parse_row)
            truncation_rows.append(truncation_row)
            _append_jsonl(
                trace_path,
                {
                    "kind": "request_completed",
                    **row,
                    "response": response,
                    "extraction_text": extraction,
                },
            )
            _append_jsonl(trace_path, {"kind": "token", **token_row})
            _append_jsonl(trace_path, {"kind": "parse", **parse_row})
            _append_jsonl(trace_path, {"kind": "truncation", **truncation_row})
            checkpoint = Path(str(payload["raw_root"])) / arm / "checkpoints" / f"{request_id}.json"
            _atomic_write_json(
                checkpoint,
                {
                    "request": row,
                    "raw_output": raw_row,
                    "parse": parse_row,
                    "token": token_row,
                    "truncation": truncation_row,
                },
            )
            _progress("checkpoint", arm=arm, request_id=request_id, path=str(checkpoint))
            _progress(
                "model_response",
                arm=arm,
                request_id=request_id,
                response_hash=row["response_hash"],
                generated_tokens=token_row["generated_tokens"],
                truncated=generation["truncated"],
            )
        return response, extraction

    proposer._chat_complete_request = traced_chat  # type: ignore[method-assign]
    gpu_rows: list[JsonDict] = []
    arcade = None
    scorecard = None
    agent = None
    policy_reads: set[str] = set()
    try:
        _progress("model_load_started", arm=arm, model_path=model["model_path"])
        if not proposer._ensure_server():
            raise RuntimeError("CUDA llama-server failed to become healthy")
        server_pid = int(proposer._proc.pid)
        gpu_rows.extend(_inside_inference_gpu_rows(server_pid))
        _append_jsonl(
            trace_path,
            {
                "kind": "model_resident",
                "arm": arm,
                "server_pid": server_pid,
                "server_props": proposer.server_props(),
                "gpu_rows": gpu_rows,
            },
        )
        model_load_receipt = {
            "arm": arm,
            "model_repository": MODEL_REPOSITORY,
            "model_path": str(model["model_path"]),
            "model_hash": model["model_hash"],
            "server_pid": server_pid,
            "server_props": _json_safe(proposer.server_props()),
            "observed_model_path": proposer.observed_model_path(),
            "loaded": True,
        }
        model_load_receipts.append(model_load_receipt)
        _progress("model_load_finished", arm=arm, server_pid=server_pid)
        arcade = offline_arcade()
        scorecard = arcade.open_scorecard(tags=["exp7144", arm, "development-proxy", "no-solve"])
        env = arcade.make(game, seed=RANDOM_SEED, scorecard_id=scorecard)
        BaseAgent = _load_framework_agent()
        AgentClass = make_carnot_agent(BaseAgent, cascade=True, proposer=proposer)
        agent = AgentClass(
            card_id=str(scorecard),
            game_id=game,
            agent_name=f"carnot-exp7144-{arm}",
            ROOT_URL="local-offline",
            record=False,
            arc_env=env,
            tags=["exp7144", arm, "development-proxy", "no-solve"],
        )
        if not isinstance(agent._policy, E3AgentPolicy):
            raise RuntimeError("make_carnot_agent did not construct E3AgentPolicy")
        agent._policy.explore_budget = 1

        def record_open(event: str, args: tuple[Any, ...]) -> None:
            if event == "open" and args:
                policy_reads.add(str(args[0]))

        sys.addaudithook(record_open)
        control_path: list[str] = []
        stop_reason = "action_budget_exhausted"
        for index in range(int(budget["actions"])):
            latest = agent._convert_raw_frame_data(env.observation_space)
            before = _normal_observation(latest)
            e3_action = agent.choose_action(agent.frames, latest)
            e3_action_number, e3_data = _action_record(e3_action)
            action = e3_action
            source = "E3AgentPolicy.next_move"
            adapter_label = None
            if adapter is not None:
                labels = list(adapter.action_labels(env, latest, control_path) or [])
                if not labels:
                    raise RuntimeError("declared adapter control produced no next action")
                adapter_label = str(labels[0])
                decoded = json.loads(adapter_label)
                selected_data = decoded.get("data") if isinstance(decoded.get("data"), Mapping) else None
                action = _framework_action(int(decoded["action"]), selected_data, game)
                control_path.append(adapter_label)
                source = "E3AgentPolicy.visible_declared_adapter_control"
            action_number, data = _action_record(action)
            proposal_id = f"{arm}-proposal-{index}"
            action_id = f"{arm}-action-{index}"
            proposal = {
                "arm": arm,
                "request_id": request_rows[-1]["request_id"] if request_rows else None,
                "proposal_id": proposal_id,
                "action_id": action_id,
                "action": action_number,
                "data": data,
                "source": source,
            }
            proposal_rows.append(proposal)
            _append_jsonl(trace_path, {"kind": "proposal", **proposal})
            available = set(before.get("available_actions") or [])
            accepted = action_number == 0 or action_number in available
            verifier = {
                "arm": arm,
                "proposal_id": proposal_id,
                "verifier": "E3 policy-visible action schema",
                "accepted": accepted,
                "oracle": False,
            }
            verifier_rows.append(verifier)
            _append_jsonl(trace_path, {"kind": "verifier", **verifier})
            if not accepted:
                raise RuntimeError(f"E3 proposed unavailable action {action_number}")
            attempt = {
                "arm": arm,
                "attempt_id": f"{arm}-attempt-{index}",
                "action_id": action_id,
                "request_id": proposal["request_id"],
                "e3_candidate": {"action": e3_action_number, "data": e3_data},
                "declared_adapter_label": adapter_label,
                "accepted": True,
                "executed": False,
            }
            attempt_rows.append(attempt)
            frame = agent.take_action(action)
            if frame is None:
                raise RuntimeError("real scored action returned no frame")
            agent.append_frame(frame)
            agent.action_counter += 1
            after = _normal_observation(frame)
            action_row = {
                "arm": arm,
                "action_id": action_id,
                "proposal_id": proposal_id,
                "executed": True,
                "action": action_number,
                "data": data,
            }
            action_rows.append(action_row)
            attempt["executed"] = True
            _append_jsonl(trace_path, {"kind": "attempt", **attempt})
            _append_jsonl(trace_path, {"kind": "action", **action_row})
            reward = getattr(frame, "reward", None)
            transition = {
                "arm": arm,
                "action_id": action_id,
                "executed": True,
                "before_hash": sha256_json(before),
                "after_hash": sha256_json(after),
                "level_before": int(before.get("levels_completed", 0) or 0),
                "level_after": int(after.get("levels_completed", 0) or 0),
                "reward": reward,
            }
            transition_rows.append(transition)
            _append_jsonl(trace_path, {"kind": "transition", **transition})
            reward_row = {"arm": arm, "action_id": action_id, "reward": reward}
            reward_rows.append(reward_row)
            _append_jsonl(trace_path, {"kind": "reward", **reward_row})
            level_row = {
                "arm": arm,
                "action_id": action_id,
                "level_before": transition["level_before"],
                "level_after": transition["level_after"],
                "solve_provenance": "development_proxy",
                "game_level_solve_claimed": False,
            }
            level_rows.append(level_row)
            _append_jsonl(trace_path, {"kind": "level", **level_row})
            _progress(
                "action",
                arm=arm,
                action_id=action_id,
                action=action_number,
                level_before=transition["level_before"],
                level_after=transition["level_after"],
            )
            if int(transition["level_after"]) >= int(payload["selected_game"]["target_level"]):
                stop_reason = "target_level_reached"
                break
        final_adapter_state = _target_adapter_state(game)
        import_row["import_block_attempts"] = list(blocker.attempts) if blocker is not None else []
        import_row["target_adapter_module_loaded_during_generation"] = bool(
            final_adapter_state["module_loaded"]
        )
        import_row["target_recipe_symbols_during_generation"] = final_adapter_state[
            "target_recipe_symbols"
        ]
        import_row["imports"] = sorted(
            name for name in sys.modules if name.startswith("carnot.agentic")
        )
        import_row["passed"] = bool(
            not isolation["module_loaded"]
            and not isolation["target_recipe_symbols"]
            and (
                declared_loaded_after_receipt
                and final_adapter_state["module_loaded"]
                if arm == "adapter_visible_control"
                else not final_adapter_state["module_loaded"]
                and not final_adapter_state["target_recipe_symbols"]
                and not import_row["import_block_attempts"]
            )
        )
        normalized_reads = sorted(path.replace("\\", "/") for path in policy_reads)
        forbidden_markers = (
            "/ops/arc_solve_registry.yaml",
            f"/results/arc_loop_solve_{game}",
            f"/models/arc_verifier_{game}",
            f"/environment_files/{game}/",
        )
        forbidden_reads = [
            path for path in normalized_reads if any(marker in path for marker in forbidden_markers)
        ]
        if arm == "adapter_withheld":
            forbidden_reads.extend(
                path for path in normalized_reads if path.endswith("/agentic/arc_game_adapters.py")
            )
        forbidden_reads = sorted(set(forbidden_reads))
        forbidden_read_row = {
            "arm": arm,
            "status": "checked",
            "imports": import_row["imports"],
            "arguments": {
                "game": game,
                "action_budget": int(budget["actions"]),
                "generation_max_tokens": int(budget["generation_max_tokens"]),
            },
            "environment": {
                "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
                "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
            },
            "filesystem_reads": normalized_reads,
            "forbidden_reads": forbidden_reads,
            "passed": not forbidden_reads,
        }
        _append_jsonl(trace_path, {"kind": "runtime_access", **forbidden_read_row})
        _append_jsonl(trace_path, {"kind": "import", **import_row})
        stop_reason_row = {
            "arm": arm,
            "stop_reason": stop_reason,
            "action_count": len(action_rows),
            "action_budget": int(budget["actions"]),
        }
        _append_jsonl(trace_path, {"kind": "stop_reason", **stop_reason_row})
        return {
            "entrypoint": REAL_ENTRYPOINT,
            "adapter_keys_before": before_adapters,
            "adapter_keys_after": after_adapters,
            "removed_adapters": removed,
            "request_rows": request_rows,
            "token_rows": token_rows,
            "raw_output_rows": raw_output_rows,
            "parse_rows": parse_rows,
            "truncation_rows": truncation_rows,
            "attempt_rows": attempt_rows,
            "proposal_rows": proposal_rows,
            "verifier_rows": verifier_rows,
            "action_rows": action_rows,
            "transition_rows": transition_rows,
            "reward_rows": reward_rows,
            "level_rows": level_rows,
            "model_load_receipts": model_load_receipts,
            "import_rows": [import_row],
            "forbidden_read_rows": [forbidden_read_row],
            "stop_reason_rows": [stop_reason_row],
            "effective_policy": (
                "E3AgentPolicy+declared_adapter"
                if arm == "adapter_visible_control"
                else "E3AgentPolicy"
            ),
            "gpu_telemetry_rows": gpu_rows,
            "server_props": proposer.server_props(),
            "model_path_observed": proposer.observed_model_path(),
            "trace_event_count": sum(1 for _ in trace_path.open(encoding="utf-8")),
        }
    finally:
        if agent is not None:
            try:
                agent.cleanup()
            except Exception:
                pass
        proposer.stop()
        if blocker is not None and blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
        if arcade is not None and scorecard is not None:
            try:
                arcade.close_scorecard(scorecard)
            except Exception:
                pass


def _worker_main(role: str, payload_path: Path, output_path: Path) -> int:  # pragma: no cover
    try:
        _progress("worker_request", role=role, payload=str(payload_path))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        result = _setup_worker(payload) if role == "setup" else _arm_worker(payload)
        _atomic_write_json(output_path, result)
        _progress("worker_checkpoint", role=role, output=str(output_path))
        return 0
    except Exception as exc:
        error = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        _atomic_write_json(output_path, error)
        traceback.print_exc()
        return 1


def _finish_leases(
    leases: Sequence[Any], *, complete: bool, gpu_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:  # pragma: no cover
    receipts: list[JsonDict] = []
    maximum = max((int(row.get("memory_used_mb", 0) or 0) for row in gpu_rows), default=1)
    for lease in leases:
        try:
            lease.transition("loading")
            lease.transition("resident", vram_mb=max(1, maximum))
            lease.transition("inferencing")
            lease.transition("unloading")
            lease.transition("validating", vram_mb=0, exit_code=0 if complete else 1, unload_observed=True)
            lease.transition("terminal_complete" if complete else "terminal_blocked")
            receipts.append(lease.release())
        except Exception as exc:
            receipts.append({"lease_id": getattr(lease, "lease_id", None), "error": repr(exc)})
            lease.close()
    return receipts


def run_experiment(
    *,
    run_date: str,
    output_path: Path,
    raw_root: Path,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    generation_max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS,
) -> JsonDict:  # pragma: no cover
    """Run the one bounded rank-one pair and publish the final fail-closed artifact."""

    total_started = time.monotonic_ns()
    initialize_terminal_artifact(
        output_path,
        run_date=run_date,
        action_budget=action_budget,
        generation_max_tokens=generation_max_tokens,
    )
    _progress("checkpoint", phase="artifact_initialization", path=str(output_path))
    raw_root.mkdir(parents=True, exist_ok=True)
    parent_trace = raw_root / "parent-events.jsonl"
    if parent_trace.exists():
        parent_trace.unlink()
    phase_rows: list[JsonDict] = []
    process_rows: list[JsonDict] = []
    request_rows: list[JsonDict] = []
    token_rows: list[JsonDict] = []
    raw_output_rows: list[JsonDict] = []
    parse_rows: list[JsonDict] = []
    truncation_rows: list[JsonDict] = []
    attempt_rows: list[JsonDict] = []
    proposal_rows: list[JsonDict] = []
    verifier_rows: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    transition_rows: list[JsonDict] = []
    reward_rows: list[JsonDict] = []
    level_rows: list[JsonDict] = []
    model_identity_rows: list[JsonDict] = []
    model_load_receipts: list[JsonDict] = []
    registry_precheck_rows: list[JsonDict] = []
    import_rows: list[JsonDict] = []
    forbidden_read_rows: list[JsonDict] = []
    input_difference_rows: list[JsonDict] = []
    policy_difference_rows: list[JsonDict] = []
    stop_reason_rows: list[JsonDict] = []
    raw_manifest: list[JsonDict] = []
    gpu_rows: list[JsonDict] = []
    preconditions: list[JsonDict] = []
    source_hashes: JsonDict = {}
    selected: JsonDict = {}
    model: JsonDict = {}
    registry_hash_after: str | None = None
    adapter_roster: list[str] = []
    effective_policies: dict[str, str] = {}
    budget = {
        **COMMON_BUDGET,
        "actions": int(action_budget),
        "generation_max_tokens": int(generation_max_tokens),
    }
    parent_pid = os.getpid()
    parent_ticks = _process_start_ticks(parent_pid)
    init_ended = time.monotonic_ns()
    init_phase = _phase_receipt(
        "artifact_initialization",
        total_started,
        init_ended,
        pid=parent_pid,
        start_ticks=parent_ticks,
        exit_state="completed",
        timed_out=False,
        stop_reason="schema_complete_terminal_block_written_before_model_setup",
    )
    phase_rows.append(init_phase)
    _append_jsonl(parent_trace, {"kind": "phase", **init_phase})
    preconditions.append(gate_row("artifact_initialized_before_setup", True, output_path.is_file()))
    _progress("phase", phase="artifact_initialization", state="completed")

    def checkpoint(phase: str) -> JsonDict:
        current = build_artifact(
            run_date=run_date,
            preconditions_checked=preconditions,
            source_artifact_hashes=source_hashes,
            model=model,
            selected_game=selected,
            registry_rank_before_outcomes=int(selected.get("registry_rank", 0) or 0),
            registry_hash_after=registry_hash_after,
            phase_receipt_rows=phase_rows,
            process_rows=process_rows,
            request_rows=request_rows,
            token_rows=token_rows,
            proposal_rows=proposal_rows,
            verifier_rows=verifier_rows,
            action_rows=action_rows,
            transition_rows=transition_rows,
            raw_trace_manifest=raw_manifest,
            gpu_telemetry_rows=gpu_rows,
            duration_s=(time.monotonic_ns() - total_started) / 1_000_000_000,
            action_budget=action_budget,
            generation_max_tokens=generation_max_tokens,
            model_identity_rows=model_identity_rows,
            model_load_receipts=model_load_receipts,
            registry_precheck_rows=registry_precheck_rows,
            attempt_rows=attempt_rows,
            raw_output_rows=raw_output_rows,
            parse_rows=parse_rows,
            reward_rows=reward_rows,
            level_rows=level_rows,
            truncation_rows=truncation_rows,
            import_rows=import_rows,
            forbidden_read_rows=forbidden_read_rows,
            input_difference_rows=input_difference_rows,
            policy_difference_rows=policy_difference_rows,
            stop_reason_rows=stop_reason_rows,
        )
        _atomic_write_json(output_path, current)
        _progress("checkpoint", phase=phase, path=str(output_path))
        return current

    checkpoint("artifact_initialization_receipt")
    registry_started = time.monotonic_ns()
    registry_path = ROOT / REGISTRY_RELATIVE_PATH
    leases: list[Any] = []
    idle: list[JsonDict] = []
    try:
        registry_before = sha256_file(registry_path)
        registry_rows = _load_registry(registry_path)
        frozen = freeze_registry_rank_one(registry_rows)
        selected = dict(frozen)
        adapter_roster = sorted(str(row.get("game")) for row in registry_rows if row.get("game"))
        source_hashes = _source_hashes(registry_path)
        source_hashes[REGISTRY_RELATIVE_PATH.as_posix()] = registry_before
        registry_precheck_rows.append(
            {
                "game": frozen["game"],
                "eligibility_rank": frozen["registry_rank"],
                "levels_reproduced": frozen["registry_levels_reproduced"],
                "already_credited": int(frozen["registry_levels_reproduced"])
                >= int(frozen["target_level"]),
                "checked_before_outcomes": True,
            }
        )
        preconditions.append(gate_row("registry_eligibility_rank_one", 1, frozen["registry_rank"]))
        preconditions.append(
            gate_row(
                "selected_level_already_registered",
                True,
                int(frozen["registry_levels_reproduced"]) >= int(frozen["target_level"]),
            )
        )
        try:
            model = resolve_headline_model()
            model_ok = True
        except Exception as exc:
            model = {"resolution_error": f"{type(exc).__name__}: {exc}"}
            model_ok = False
        preconditions.append(gate_row("cached_qwen_q4_k_m", True, model_ok))
        if model_ok:
            model_identity_rows.append(
                {
                    "model_repository": model["model_repository"],
                    "model_path": model["model_path"],
                    "model_hash": model["model_hash"],
                    "model_quantization": model["model_quantization"],
                    "chat_template": model["llama_cpp_chat_template"],
                    "download_attempted": model["download_attempted"],
                }
            )
        gpu_rows = _gpu_rows()
        idle = [row for row in gpu_rows if row.get("idle") is True]
        preconditions.append(gate_row("two_idle_rtx_3090_gpus", 2, len(idle), passed=len(idle) >= 2))
        runner = _llama_cpp_receipt()
        preconditions.append(
            gate_row(
                "llama_cpp_cuda_health",
                True,
                runner["exists"] and runner["cuda_linked"] and runner["version_returncode"] == 0,
            )
        )
        writable = False
        try:
            probe = raw_root / ".write-preflight"
            probe.write_text("exp7144\n", encoding="utf-8")
            writable = probe.read_text(encoding="utf-8") == "exp7144\n"
            probe.unlink()
        except OSError:
            writable = False
        preconditions.append(gate_row("writable_raw_trace_storage", True, writable))
        preconditions.append(
            gate_row(
                "subprocess_process_group_support",
                True,
                hasattr(os, "killpg") and hasattr(os, "setsid"),
            )
        )
        if all(row["passed"] is True for row in preconditions) and len(idle) >= 2:
            from carnot import gpu_lease_phase_journal as lease_api

            for gpu in idle[:2]:
                lease = lease_api.GpuLease.acquire(
                    runtime_dir=raw_root / "gpu-leases",
                    task_id=f"exp7144:{parent_pid}",
                    device_uuid=str(gpu["uuid"]),
                    expected_model=str(model["model_path"]),
                    vram_before_mb=int(gpu["memory_used_mb"]),
                    ttl_s=3900.0,
                )
                lease.transition("admitted")
                leases.append(lease)
                gpu_rows.append({"sample_phase": "lease_acquired", **lease.owner_receipt()})
        preconditions.append(gate_row("two_idle_rtx_3090_leases", 2, len(leases), passed=len(leases) == 2))
    except Exception as exc:
        preconditions.append(
            gate_row(
                "preflight_exception_absent",
                None,
                f"{type(exc).__name__}: {exc}",
                passed=False,
            )
        )
    registry_ended = time.monotonic_ns()
    registry_phase = _phase_receipt(
        "registry_freeze",
        registry_started,
        registry_ended,
        pid=parent_pid,
        start_ticks=parent_ticks,
        exit_state="completed" if all(row["passed"] for row in preconditions) else "blocked",
        timed_out=False,
        stop_reason="rank_one_and_preconditions_frozen_before_outcomes",
    )
    phase_rows.append(registry_phase)
    _append_jsonl(parent_trace, {"kind": "phase", **registry_phase})
    _progress("phase", phase="registry_freeze", state=registry_phase["exit_state"])
    checkpoint("registry_freeze")
    ready = bool(preconditions) and all(row.get("passed") is True for row in preconditions)
    setup_ok = False
    if ready:
        setup_payload = {"model": model, "selected_game": selected}
        row, result, started_ns, ended_ns = _run_worker(
            role="setup",
            payload=setup_payload,
            raw_root=raw_root,
            cap_s=SETUP_CAP_S,
            parent_trace=parent_trace,
        )
        process_rows.append(row)
        setup_ok = bool(row["exit_code"] == 0 and result.get("setup_ok") is True)
        preconditions.extend(
            [
                gate_row("setup_worker_exit", 0, row["exit_code"]),
                gate_row("embedded_tokenizer_loadable", True, result.get("embedded_tokenizer_loadable")),
                gate_row("selected_adapter_declared_for_control", True, result.get("selected_adapter_present")),
                gate_row("real_e3_factory_importable", True, result.get("factory_callable")),
            ]
        )
        phase = _phase_receipt(
            "setup",
            started_ns,
            ended_ns,
            pid=int(row["pid"]),
            start_ticks=int(row["start_ticks"]),
            exit_state="completed" if setup_ok else "blocked",
            timed_out=bool(row["timed_out"]),
            stop_reason=str(row["stop_reason"]),
        )
        phase_rows.append(phase)
        _append_jsonl(parent_trace, {"kind": "phase", **phase})
        _progress("phase", phase="setup", state=phase["exit_state"])
        checkpoint("setup")
    if setup_ok:
        for arm in ARM_NAMES:
            trace_path = raw_root / arm / "events.jsonl"
            if trace_path.exists():
                trace_path.unlink()
            payload = {
                "arm": arm,
                "model": model,
                "selected_game": selected,
                "trace_path": str(trace_path),
                "raw_root": str(raw_root),
                "port": _free_port(),
                "budget": budget,
                "adapter_roster": adapter_roster,
            }
            cap = WITHHELD_ARM_CAP_S if arm == "adapter_withheld" else CONTROL_ARM_CAP_S
            row, result, started_ns, ended_ns = _run_worker(
                role=arm,
                payload=payload,
                raw_root=raw_root,
                cap_s=cap,
                parent_trace=parent_trace,
            )
            process_rows.append(row)
            for key, target in (
                ("request_rows", request_rows),
                ("token_rows", token_rows),
                ("raw_output_rows", raw_output_rows),
                ("parse_rows", parse_rows),
                ("truncation_rows", truncation_rows),
                ("attempt_rows", attempt_rows),
                ("proposal_rows", proposal_rows),
                ("verifier_rows", verifier_rows),
                ("action_rows", action_rows),
                ("transition_rows", transition_rows),
                ("reward_rows", reward_rows),
                ("level_rows", level_rows),
                ("model_load_receipts", model_load_receipts),
                ("import_rows", import_rows),
                ("forbidden_read_rows", forbidden_read_rows),
                ("stop_reason_rows", stop_reason_rows),
                ("gpu_telemetry_rows", gpu_rows),
            ):
                target.extend(deepcopy(list(result.get(key) or [])))
            if result.get("effective_policy"):
                effective_policies[arm] = str(result["effective_policy"])
            if trace_path.is_file():
                raw_manifest.append(
                    raw_trace_receipt(
                        trace_path,
                        arm=arm,
                        event_count=int(result.get("trace_event_count", sum(1 for _ in trace_path.open(encoding="utf-8"))),),
                    )
                )
            phase = _phase_receipt(
                arm,
                started_ns,
                ended_ns,
                pid=int(row["pid"]),
                start_ticks=int(row["start_ticks"]),
                exit_state="completed" if row["exit_code"] == 0 else "blocked",
                timed_out=bool(row["timed_out"]),
                stop_reason=str(row["stop_reason"]),
            )
            phase_rows.append(phase)
            _append_jsonl(parent_trace, {"kind": "phase", **phase})
            _progress("phase", phase=arm, state=phase["exit_state"])
            checkpoint(arm)
    imports_by_arm = _rows_by_arm(import_rows)
    if all(imports_by_arm[arm] for arm in ARM_NAMES):
        withheld_access = bool(
            imports_by_arm["adapter_withheld"][0].get(
                "target_adapter_module_loaded_during_generation"
            )
        )
        control_access = bool(
            imports_by_arm["adapter_visible_control"][0].get(
                "declared_adapter_loaded_after_receipt"
            )
        )
        input_difference_rows.append(
            {
                "dimension": "target_adapter_access",
                "adapter_withheld": withheld_access,
                "adapter_visible_control": control_access,
                "measured_difference": withheld_access is False and control_access is True,
            }
        )
    if set(effective_policies) == set(ARM_NAMES):
        policy_difference_rows.append(
            {
                "dimension": "effective_policy",
                "adapter_withheld": effective_policies["adapter_withheld"],
                "adapter_visible_control": effective_policies["adapter_visible_control"],
                "measured_difference": effective_policies["adapter_withheld"]
                != effective_policies["adapter_visible_control"],
            }
        )
    complete_runtime = bool(
        setup_ok
        and len(_arm_processes(process_rows)) == 2
        and all(row.get("exit_code") == 0 for row in process_rows)
        and _executed_transitions_per_arm(transition_rows)
        and _runtime_receipts_complete(
            request_rows, token_rows, proposal_rows, verifier_rows, action_rows, transition_rows
        )
    )
    lease_receipts = _finish_leases(leases, complete=complete_runtime, gpu_rows=gpu_rows)
    gpu_rows.extend({"sample_phase": "lease_released", **row} for row in lease_receipts)
    final_started = time.monotonic_ns()
    try:
        registry_hash_after = sha256_file(ROOT / REGISTRY_RELATIVE_PATH)
    except OSError:
        registry_hash_after = None
    final_ended = time.monotonic_ns()
    final_phase = _phase_receipt(
        "finalization",
        final_started,
        final_ended,
        pid=parent_pid,
        start_ticks=parent_ticks,
        exit_state="completed",
        timed_out=False,
        stop_reason="registry_rehashed_artifact_validated_and_published",
    )
    phase_rows.append(final_phase)
    _append_jsonl(parent_trace, {"kind": "phase", **final_phase})
    _progress("phase", phase="finalization", state="completed")
    raw_manifest.append(
        raw_trace_receipt(
            parent_trace,
            arm="parent",
            event_count=sum(1 for _ in parent_trace.open(encoding="utf-8")),
        )
    )
    artifact = checkpoint("finalization")
    errors = validate_artifact(artifact, verify_raw_traces=True)
    if errors:
        preconditions.append(gate_row("final_artifact_validation", [], errors, passed=False))
        artifact = checkpoint("final_validation_block")
    _atomic_write_json(output_path, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse public driver flags while retaining the V626 budget defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=time.strftime("%Y%m%d"))
    parser.add_argument("--output", type=Path, default=ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT_ROOT)
    parser.add_argument("--actions", type=int, default=DEFAULT_ACTION_BUDGET)
    parser.add_argument(
        "--generation-max-tokens", type=int, default=DEFAULT_GENERATION_MAX_TOKENS
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--worker", choices=("setup", *ARM_NAMES))
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.actions <= 0:
        parser.error("--actions must be positive")
    if args.generation_max_tokens <= 0:
        parser.error("--generation-max-tokens must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    if args.worker:
        if args.payload is None or args.worker_output is None:
            raise SystemExit("--worker requires --payload and --worker-output")
        return _worker_main(args.worker, args.payload, args.worker_output)
    if args.validate:
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact, verify_raw_traces=True)
        if errors:
            print(json.dumps({"valid": False, "errors": errors}, indent=2))
            return 1
        print(json.dumps({"valid": True, "verdict": artifact["honest_verdict"]}, indent=2))
        return 0
    artifact = run_experiment(
        run_date=str(args.date),
        output_path=args.output.resolve(),
        raw_root=args.raw_root.resolve(),
        action_budget=int(args.actions),
        generation_max_tokens=int(args.generation_max_tokens),
    )
    print(json.dumps({"artifact": str(args.output.resolve()), "verdict": artifact["honest_verdict"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
