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

from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.sota_models import cached_sota_pair


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
COMMON_BUDGET: JsonDict = {
    "actions": 3,
    "generation_max_tokens": 384,
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
INFERENCE_SUBSTRATE = "model_bounded_generation: live local GGUF ARC cell"
REAL_ENTRYPOINT = "make_carnot_agent:E3AgentPolicy"

REQUIRED_ARTIFACT_FIELDS = (
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
}
FIELD_PRINCIPLES = {field: _PRINCIPLES[field] for field in REQUIRED_ARTIFACT_FIELDS}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


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
    provider: Callable[..., list[Mapping[str, Any]] | None] = cached_sota_pair,
) -> JsonDict:
    """Resolve only the cached headline Qwen Q4_K_M through ``cached_sota_pair``."""

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
        "prompt_hash",
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
    checks = [
        gate_row("adapter_withheld_exactly", True, adapter_exact),
        gate_row("real_e3_entrypoint_used", True, real_e3),
        gate_row("fresh_process_per_arm", True, fresh),
        gate_row("phase_caps_respected", True, caps_ok),
        gate_row("registry_mutated", False, registry_mutated),
        gate_row("common_arm_configuration", True, common),
        gate_row("all_runtime_preconditions", True, preconditions_ok),
        gate_row("complete_phase_receipts", True, phases_complete),
        gate_row("executed_transition_per_arm", True, attempts),
        gate_row("complete_runtime_receipts", True, runtime),
        gate_row("complete_raw_trace_manifest", True, raw_complete),
    ]
    failed = next((row for row in checks if row["passed"] is not True), None)
    disqualifying = {
        "adapter_withheld_exactly",
        "real_e3_entrypoint_used",
        "fresh_process_per_arm",
        "phase_caps_respected",
        "registry_mutated",
        "common_arm_configuration",
    }
    if failed is None:
        verdict = "positive" if _level_count(transition_rows, ARM_NAMES[0]) > 0 else "null"
        honest = (
            "complete_positive_development_proxy_level_delta_measured_no_solve_claim"
            if verdict == "positive"
            else "complete_null_executed_pair_zero_withheld_levels_no_solve_claim"
        )
    elif failed["check"] in disqualifying:
        verdict = "disqualified"
        honest = f"disqualified_{failed['check']}"
    else:
        verdict = "blocked"
        honest = f"blocked_{failed['check']}"
    withheld_levels = _level_count(transition_rows, ARM_NAMES[0])
    control_levels = _level_count(transition_rows, ARM_NAMES[1])
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
        "inference_substrate_class": "model_bounded_generation" if any_generation else "blocked_no_run",
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
    }
    if registry_hash_after is not None:
        evidence["source_artifact_hashes"]["registry_hash_after"] = registry_hash_after
    projection = _projection(evidence)
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": evidence["preconditions_checked"],
        "run_date": str(run_date),
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
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


def initialize_terminal_artifact(path: str | Path, *, run_date: str) -> JsonDict:
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
    )
    atomic_write_json(Path(path), artifact, allow_override=False)
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


def _fixture_for_game(game: str) -> Path:  # pragma: no cover
    matches = sorted((ROOT / "environment_files" / game).glob(f"*/{game}.py"))
    if len(matches) != 1:
        raise RuntimeError(f"selected public fixture count for {game}: {len(matches)}")
    return matches[0]


def _source_hashes(registry: Path, fixture: Path) -> JsonDict:  # pragma: no cover
    paths = (
        registry,
        fixture,
        Path(__file__).resolve(),
        ROOT / "scripts/experiments/experiment_7127_v626_adapter_withheld_arc_loo.py",
        ROOT / "python/carnot/agentic/arc_competition_agent.py",
        ROOT / "python/carnot/agentic/arc_executable_world_model.py",
        ROOT / "python/carnot/agentic/arc_game_adapters.py",
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
    atomic_write_json(payload_path, dict(payload), allow_override=False)
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
        try:
            exit_code = process.wait(timeout=cap_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_reason = f"timeout_after_{cap_s}s_process_group_sigterm_then_sigkill"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait(timeout=10)
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


def _arm_worker(payload: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    arm = str(payload["arm"])
    game = str(payload["selected_game"]["game"])
    model = dict(payload["model"])
    trace_path = Path(str(payload["trace_path"]))
    # E3_DIR is resolved at arc_executable_world_model import time.  Apply the
    # raw-root override before importing either it or arc_competition_agent so
    # prompt/transition staging cannot land under tracked results/arc_e3.
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_CUDA_GPU": "0,1",
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_INDUCE_N_CTX": str(COMMON_BUDGET["context_tokens"]),
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(COMMON_BUDGET["generation_max_tokens"]),
            "CARNOT_ARC_INDUCE_TIMEOUT": str(COMMON_BUDGET["generation_timeout_s"]),
            "CARNOT_ARC_INDUCE_THINK": "0",
            "CARNOT_ARC_INDUCE_DEFECT_REASKS": "0",
            "CARNOT_ARC_INDUCE_GOAL_DEFECT_REASKS": "0",
            "CARNOT_ARC_LLAMA_SERVER_SLOTS": "1",
            "CARNOT_ARC_LLAMA_SERVER_PARALLEL": "1",
            "CARNOT_ARC_E3_DIR": str(Path(str(payload["raw_root"])) / arm / "e3"),
            "CARNOT_ARC_LIVENESS_DIR": str(Path(str(payload["raw_root"])) / arm / "liveness"),
        }
    )
    from carnot.agentic import arc_game_adapters
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_solver_kit import offline_arcade

    before_adapters = sorted(arc_game_adapters._BUILDERS)
    removed: list[str] = []
    if arm == "adapter_withheld":
        if game not in arc_game_adapters._BUILDERS:
            raise RuntimeError(f"selected adapter absent before withholding: {game}")
        arc_game_adapters._BUILDERS.pop(game)
        removed = [game]
    after_adapters = sorted(arc_game_adapters._BUILDERS)
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
    proposer = LocalGGUFProposer(
        repo_substr="Qwen3.6-35B-A3B",
        model_path=str(model["model_path"]),
        model_repository=MODEL_REPOSITORY,
        model_filename=Path(str(model["model_path"])).name,
        requested_model_path=str(model["model_path"]),
        requested_model_filename=Path(str(model["model_path"])).name,
        port=int(payload["port"]),
        n_ctx=int(COMMON_BUDGET["context_tokens"]),
        max_tokens=int(COMMON_BUDGET["generation_max_tokens"]),
        timeout=int(COMMON_BUDGET["generation_timeout_s"]),
        mtp=False,
        kv_quant="q8_0",
        n_gpu_layers=999,
        tries=1,
        use_chat_template=True,
    )
    request_rows: list[JsonDict] = []
    token_rows: list[JsonDict] = []
    proposal_rows: list[JsonDict] = []
    verifier_rows: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    transition_rows: list[JsonDict] = []
    tools_hash = sha256_json(["offline_arcade.step", REAL_ENTRYPOINT])
    environment_hash = sha256_json(
        {
            "game": game,
            "fixture_hash": payload["selected_game"]["fixture_hash"],
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
                "budget": deepcopy(COMMON_BUDGET),
                "seed": RANDOM_SEED,
                "tools_hash": tools_hash,
                "executable_environment_hash": environment_hash,
                "generation_invoked": True,
                "response_hash": sha256_json(response),
                "duration_s": round((ended_ns - started_ns) / 1_000_000_000, 6),
                "error": error,
            }
            token_row = {
                "arm": arm,
                "request_id": request_id,
                "prompt_tokens": int(proposer.last_prompt_tokens),
                "generated_tokens": int(proposer.last_generated_tokens),
            }
            request_rows.append(row)
            token_rows.append(token_row)
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
        return response, extraction

    proposer._chat_complete_request = traced_chat  # type: ignore[method-assign]
    gpu_rows: list[JsonDict] = []
    arcade = None
    scorecard = None
    agent = None
    try:
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
        arcade = offline_arcade()
        scorecard = arcade.open_scorecard(tags=["exp7127", arm, "development-proxy", "no-solve"])
        env = arcade.make(game, seed=RANDOM_SEED, scorecard_id=scorecard)
        BaseAgent = _load_framework_agent()
        AgentClass = make_carnot_agent(BaseAgent, cascade=True, proposer=proposer)
        agent = AgentClass(
            card_id=str(scorecard),
            game_id=game,
            agent_name=f"carnot-exp7127-{arm}",
            ROOT_URL="local-offline",
            record=False,
            arc_env=env,
            tags=["exp7127", arm, "development-proxy", "no-solve"],
        )
        if not isinstance(agent._policy, E3AgentPolicy):
            raise RuntimeError("make_carnot_agent did not construct E3AgentPolicy")
        agent._policy.explore_budget = 1
        for index in range(int(COMMON_BUDGET["actions"])):
            latest = agent._convert_raw_frame_data(env.observation_space)
            before = _normal_observation(latest)
            action = agent.choose_action(agent.frames, latest)
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
                "source": "E3AgentPolicy.next_move",
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
            _append_jsonl(
                trace_path,
                {"kind": "reward", "arm": arm, "action_id": action_id, "reward": reward},
            )
            _append_jsonl(
                trace_path,
                {
                    "kind": "level",
                    "arm": arm,
                    "action_id": action_id,
                    "level_before": transition["level_before"],
                    "level_after": transition["level_after"],
                    "solve_provenance": "development_proxy",
                    "solve_claim_made": False,
                    "offline_reproduced": False,
                },
            )
        return {
            "entrypoint": REAL_ENTRYPOINT,
            "adapter_keys_before": before_adapters,
            "adapter_keys_after": after_adapters,
            "removed_adapters": removed,
            "request_rows": request_rows,
            "token_rows": token_rows,
            "proposal_rows": proposal_rows,
            "verifier_rows": verifier_rows,
            "action_rows": action_rows,
            "transition_rows": transition_rows,
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
        if arcade is not None and scorecard is not None:
            try:
                arcade.close_scorecard(scorecard)
            except Exception:
                pass


def _worker_main(role: str, payload_path: Path, output_path: Path) -> int:  # pragma: no cover
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        result = _setup_worker(payload) if role == "setup" else _arm_worker(payload)
        atomic_write_json(output_path, result, allow_override=False)
        return 0
    except Exception as exc:
        error = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        atomic_write_json(output_path, error, allow_override=False)
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
    *, run_date: str, output_path: Path, raw_root: Path
) -> JsonDict:  # pragma: no cover
    """Run the one bounded rank-one pair and publish the final fail-closed artifact."""

    from carnot import gpu_lease_phase_journal as lease_api
    from carnot.agentic.arc_game_adapters import _BUILDERS

    total_started = time.monotonic_ns()
    raw_root.mkdir(parents=True, exist_ok=True)
    parent_trace = raw_root / "parent-events.jsonl"
    if parent_trace.exists():
        parent_trace.unlink()
    phase_rows: list[JsonDict] = []
    process_rows: list[JsonDict] = []
    request_rows: list[JsonDict] = []
    token_rows: list[JsonDict] = []
    proposal_rows: list[JsonDict] = []
    verifier_rows: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    transition_rows: list[JsonDict] = []
    raw_manifest: list[JsonDict] = []
    gpu_rows: list[JsonDict] = []
    preconditions: list[JsonDict] = []
    source_hashes: JsonDict = {}
    selected: JsonDict = {}
    model: JsonDict = {}
    registry_hash_after: str | None = None
    parent_pid = os.getpid()
    parent_ticks = _process_start_ticks(parent_pid)
    init_started = time.monotonic_ns()
    initialize_terminal_artifact(output_path, run_date=run_date)
    init_ended = time.monotonic_ns()
    init_phase = _phase_receipt(
        "artifact_initialization",
        init_started,
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
    registry_started = time.monotonic_ns()
    registry_path = ROOT / REGISTRY_RELATIVE_PATH
    leases: list[Any] = []
    try:
        registry_before = sha256_file(registry_path)
        registry_rows = _load_registry(registry_path)
        frozen = freeze_registry_rank_one(registry_rows)
        fixture = _fixture_for_game(str(frozen["game"]))
        fixture_hash = sha256_file(fixture)
        selected = {
            **frozen,
            "fixture_path": str(fixture.relative_to(ROOT)),
            "fixture_hash": fixture_hash,
        }
        source_hashes = _source_hashes(registry_path, fixture)
        preconditions.append(gate_row("registry_eligibility_rank_one", 1, frozen["registry_rank"]))
        preconditions.append(gate_row("selected_public_fixture_hashed", True, bool(fixture_hash)))
        preconditions.append(gate_row("selected_level_already_registered", True, int(frozen["registry_levels_reproduced"]) >= int(frozen["target_level"])))
        preconditions.append(gate_row("selected_adapter_available_for_exact_withholding", True, str(frozen["game"]) in _BUILDERS))
        try:
            model = resolve_headline_model()
            model_ok = True
        except Exception as exc:
            model = {"resolution_error": f"{type(exc).__name__}: {exc}"}
            model_ok = False
        preconditions.append(gate_row("cached_qwen_q4_k_m", True, model_ok))
        gpu_rows = _gpu_rows()
        idle = [row for row in gpu_rows if row.get("idle") is True]
        preconditions.append(gate_row("two_idle_rtx_3090_gpus", 2, len(idle), passed=len(idle) >= 2))
        runner = _llama_cpp_receipt()
        preconditions.append(gate_row("llama_cpp_cuda_health", True, runner["exists"] and runner["cuda_linked"] and runner["version_returncode"] == 0))
        writable = False
        try:
            probe = raw_root / ".write-preflight"
            probe.write_text("exp7127\n", encoding="utf-8")
            writable = probe.read_text(encoding="utf-8") == "exp7127\n"
            probe.unlink()
        except OSError:
            writable = False
        preconditions.append(gate_row("writable_raw_trace_storage", True, writable))
        preconditions.append(gate_row("subprocess_process_group_support", True, hasattr(os, "killpg") and hasattr(os, "setsid")))
        try:
            from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent

            route_ok = callable(make_carnot_agent) and E3AgentPolicy.__name__ == "E3AgentPolicy"
        except Exception:
            route_ok = False
        preconditions.append(gate_row("real_e3_factory_importable", True, route_ok))
        if all(row["passed"] is True for row in preconditions) and len(idle) >= 2:
            for gpu in idle[:2]:
                lease = lease_api.GpuLease.acquire(
                    runtime_dir=raw_root / "gpu-leases",
                    task_id=f"exp7127:{parent_pid}",
                    device_uuid=str(gpu["uuid"]),
                    expected_model=str(model["model_path"]),
                    vram_before_mb=int(gpu["memory_used_mb"]),
                    ttl_s=3900.0,
                )
                lease.transition("admitted")
                leases.append(lease)
                gpu_rows.append({"sample_phase": "lease_acquired", **lease.owner_receipt()})
        preconditions.append(gate_row("two_idle_rtx_3090_leases", 2, len(leases), passed=len(leases) == 2))
        source_hashes[REGISTRY_RELATIVE_PATH.as_posix()] = registry_before
    except Exception as exc:
        preconditions.append(gate_row("preflight_exception_absent", None, f"{type(exc).__name__}: {exc}", passed=False))
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
                ("proposal_rows", proposal_rows),
                ("verifier_rows", verifier_rows),
                ("action_rows", action_rows),
                ("transition_rows", transition_rows),
                ("gpu_telemetry_rows", gpu_rows),
            ):
                target.extend(deepcopy(list(result.get(key) or [])))
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
    registry_hash_after = sha256_file(ROOT / REGISTRY_RELATIVE_PATH)
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
    raw_manifest.append(
        raw_trace_receipt(
            parent_trace,
            arm="parent",
            event_count=sum(1 for _ in parent_trace.open(encoding="utf-8")),
        )
    )
    artifact = build_artifact(
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
    )
    errors = validate_artifact(artifact, verify_raw_traces=True)
    if errors:
        raise RuntimeError(f"final artifact validation failed: {errors}")
    atomic_write_json(output_path, artifact, allow_override=False)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=time.strftime("%Y%m%d"))
    parser.add_argument("--output", type=Path, default=ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT_ROOT)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--worker", choices=("setup", *ARM_NAMES))
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        if args.payload is None or args.worker_output is None:
            parser.error("--worker requires --payload and --worker-output")
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
        run_date=str(args.date), output_path=args.output.resolve(), raw_root=args.raw_root.resolve()
    )
    print(json.dumps({"artifact": str(args.output.resolve()), "verdict": artifact["honest_verdict"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
