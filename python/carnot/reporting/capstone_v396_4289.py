"""Build the Exp 4289 v396 capstone aggregation artifact.

Spec refs: REQ-CAPSTONE-4289, SCENARIO-CAPSTONE-4289.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]
PublicationGateRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import adversarial_verify as av  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_4289_capstone_v396.json")
EXPERIMENT_ID = 4289
RANDOM_SEED = 4289
SCHEMA = "carnot.capstone_v396_4289.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4289", "SCENARIO-CAPSTONE-4289"]
BLOCKED_CHECKSUM = hashlib.sha256(b"blocked_v396_artifacts_missing").hexdigest()

THESIS_STATES = {
    "external_verifier_improves_generation",
    "ties_model_self_guidance",
    "execution_grounded_only",
    "partial_state_blocked",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4281_diffusiongemma": Upstream(
        4281, Path("results/experiment_4281_diffusiongemma_energy_guided_full_run.json")
    ),
    "4282_arcgen": Upstream(
        4282, Path("results/experiment_4282_arcgen_cross_family_stress.json")
    ),
    "4283_self_learning": Upstream(
        4283, Path("results/experiment_4283_self_learning_repowered_arcgen.json")
    ),
    "4284_efficiency": Upstream(
        4284, Path("results/experiment_4284_verifier_efficiency_vs_llm_judge.json")
    ),
    "4285_arc_progress": Upstream(
        4285, Path("results/experiment_4285_arc_incremental_progress_new_game.json")
    ),
    "4287_registry": Upstream(
        4287, Path("results/experiment_4287_verifier_registry_gaps_hygiene.json")
    ),
    "4288_hardware": Upstream(
        4288, Path("results/experiment_4288_hardware_continuity.json")
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "guidance_moat_holds",
    "cross_family_hardens_on_arcgen",
    "verifier_efficiency_parity",
    "diffusiongemma_thesis_state",
    "flagged_artifacts_excluded",
    "paper_ready",
    "verifier_is_oracle_honored",
    "reproducibility_checksum",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .396 close-state -- whether the external "
        "verifier improved GENERATION (the §5 thesis verdict) + the "
        "cross-family/efficiency/ARC reads."
    ),
    "headline_outcome": (
        "One honest string aggregating the DiffusionGemma-guidance + ARC-GEN "
        "+ efficiency + self-learning + ARC reads; the single line the .397 "
        "planner frames from."
    ),
    "guidance_moat_holds": (
        "BARE bool: did the LEARNED (oracle-distinct) verifier-guided "
        "DiffusionGemma beat RFG model-self-guidance with CI95-excl-0 -- the "
        "§5 in-generation moat verdict (NOT a circular execution-grounded win)."
    ),
    "cross_family_hardens_on_arcgen": (
        "BARE bool: did the cross-family selection win replicate on the "
        "independent ARC-GEN substrate -- closing the single-partition critique."
    ),
    "verifier_efficiency_parity": (
        "BARE bool: did the energy verifier match the LLM-judge at <=0.1x "
        "cost -- the owed §5 efficiency win condition."
    ),
    "diffusiongemma_thesis_state": (
        "One honest string for the §5 thesis state "
        "(external_verifier_improves_generation / ties_model_self_guidance / "
        "execution_grounded_only / partial_state_blocked) -- the framing the "
        ".397 planner inherits."
    ),
    "flagged_artifacts_excluded": (
        "List of .396 artifacts excluded for flagged_adversarial -- the "
        "fabrication gate (their numbers are NOT aggregated)."
    ),
    "paper_ready": (
        "From publication_gate.py --json -- the G1-G4 status (FoVer headline "
        "stays the publication target; a verified in-generation moat would be "
        "a new headline-grade supporting result)."
    ),
    "verifier_is_oracle_honored": (
        "BARE bool=true -- confirms every cited moat/headline result carried "
        "verifier_is_oracle=false (no circular/execution-grounded result "
        "headlines a moat)."
    ),
    "reproducibility_checksum": (
        "Hash of the aggregated upstream sha256 set; lets a third party "
        "re-derive the capstone."
    ),
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream; skipped "
        "upstreams import no numbers."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4281_diffusiongemma": [
        "diffusiongemma_guidance_moat",
        "carnot_minus_rfg_delta",
        "guidance_moat_ci95",
        "verifier_is_oracle",
        "headline_arm",
        "execution_grounded_arm",
        "guidance_changes_selection",
    ],
    "4282_arcgen": [
        "arcgen_cross_family_holds",
        "cross_family_delta",
        "cross_family_ci95",
        "per_substrate_delta",
        "verifier_is_oracle",
    ],
    "4283_self_learning": [
        "online_adaptation_helps",
        "static_cross_family_delta",
        "online_cross_family_delta",
        "online_minus_static_ci95",
        "verifier_is_oracle",
    ],
    "4284_efficiency": [
        "efficiency_parity_at_lower_cost",
        "cost_ratio",
        "accuracy_energy_verifier",
        "accuracy_llm_judge",
        "accuracy_delta",
        "accuracy_delta_ci95",
        "verifier_is_oracle",
    ],
    "4285_arc_progress": [
        "total_levels",
        "total_levels_solved",
        "levels_completed",
        "new_levels_solved_this_task",
        "game_advanced",
    ],
    "4287_registry": ["registry_reconciled", "regression_guard_passed", "gaps_logged"],
    "4288_hardware": [
        "kv260_terminal_confirmed",
        "kv260_step_taken",
        "polarfire_step_taken",
        "gatemate_step_taken",
    ],
}


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, bool) else None


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def list_metric(payload: Mapping[str, Any] | None, field: str) -> list[Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return list(value) if isinstance(value, list) else []


def dict_metric(payload: Mapping[str, Any] | None, field: str) -> dict[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return dict(value) if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def sha_from_payload_checksum(payload: Mapping[str, Any]) -> str:
    value = payload.get("reproducibility_checksum")
    if not isinstance(value, str):
        return ""
    normalized = value.removeprefix("sha256:")
    return normalized if is_sha256(normalized) else ""


def live_has_critical(flags: list[dict[str, Any]]) -> bool:
    return any(str(flag.get("severity", "")).lower() == "critical" for flag in flags)


def run_live_flags(path: Path) -> list[dict[str, Any]]:  # pragma: no cover
    return list(av.verify_artifact(path).get("flags", []))


def run_summarize_artifact(path: Path, root: Path) -> int:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, "scripts/summarize_artifact.py", str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(proc.returncode)


def run_publication_gate(root: Path) -> JsonDict:
    proc = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError("publication_gate.py --json returned a non-object payload")
    return payload


def read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("non-object")
    return payload


def clean_payload(payload: JsonDict | None, skipped: bool) -> JsonDict | None:
    return None if skipped or payload is None else payload


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    provenance: list[JsonDict] = []
    exclusions: list[JsonDict] = []
    missing: list[JsonDict] = []
    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        if not path.exists():
            missing.append(
                {"artifact_key": key, "experiment_id": upstream.experiment_id, "reason": "missing"}
            )
            continue
        sha = sha256_file(path)
        try:
            payload = read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError):
            missing.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "reason": "unparsable_or_non_object",
                }
            )
            continue
        summarize_exit_code = summarize_runner(path, root)
        live_flags = live_flag_runner(path)
        stamped = payload.get("flagged_adversarial") is True
        critical = live_has_critical(live_flags)
        skipped = stamped or critical
        payloads[key] = payload
        provenance.append(
            {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(upstream.path),
                "sha256": sha,
                "payload_reproducibility_checksum": sha_from_payload_checksum(payload),
                "summarize_exit_code": summarize_exit_code,
                "live_adversarial_flags": live_flags,
                "stamped_flagged_adversarial": stamped,
                "live_critical": critical,
                "skipped": skipped,
                "fields_imported": _fields_for_payload(key, skipped),
            }
        )
        if skipped:
            exclusions.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "stamped_flagged_adversarial": stamped,
                    "live_critical": critical,
                    "live_critical_flags": [
                        flag
                        for flag in live_flags
                        if str(flag.get("severity", "")).lower() == "critical"
                    ],
                    "reason": "flagged_adversarial_or_live_critical",
                }
            )
    return payloads, provenance, exclusions, missing


def checksum_from_provenance(provenance: list[Mapping[str, Any]]) -> str:
    if not provenance:
        return BLOCKED_CHECKSUM
    shas = sorted(str(row["sha256"]) for row in provenance)
    return hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()


def _ci_excludes_zero(payload: Mapping[str, Any] | None, field: str) -> bool:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    low, high = value
    if not isinstance(low, (int, float)) or isinstance(low, bool):
        return False
    if not isinstance(high, (int, float)) or isinstance(high, bool):
        return False
    return float(low) > 0.0 or float(high) < 0.0


def _headline_arm_verifier_is_oracle(payload: Mapping[str, Any] | None) -> bool | None:
    arm = dict_metric(payload, "headline_arm")
    arm_value = bool_metric(arm, "verifier_is_oracle")
    if arm_value is not None:
        return arm_value
    per_arm = dict_metric(payload, "per_arm_verifier_is_oracle")
    per_arm_value = bool_metric(per_arm, "headline_learned")
    if per_arm_value is not None:
        return per_arm_value
    return bool_metric(payload, "verifier_is_oracle")


def _partial_state_blocked(payload: Mapping[str, Any] | None) -> bool:
    arm = dict_metric(payload, "headline_arm")
    support = dict_metric(arm, "learned_verifier_partial_state_support")
    can_score = bool_metric(support, "can_score")
    status = str_metric(arm, "status") or str_metric(payload, "honest_verdict")
    return can_score is False or "partial_state" in status


def _execution_grounded_read(payload: Mapping[str, Any] | None) -> JsonDict:
    arm = dict_metric(payload, "execution_grounded_arm")
    delta = float_metric(arm, "execution_grounded_guidance_delta")
    if delta is None:
        delta = float_metric(payload, "execution_grounded_guidance_delta")
    verifier_is_oracle = bool_metric(arm, "verifier_is_oracle")
    return {
        "status": str_metric(arm, "status") or ("missing" if not arm else "reported"),
        "execution_grounded_guidance_delta": delta,
        "verifier_is_oracle": verifier_is_oracle,
        "moat_eligible": verifier_is_oracle is False,
        "interpretation": str_metric(arm, "interpretation"),
    }


def guidance_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    headline_oracle = _headline_arm_verifier_is_oracle(payload)
    moat_holds = (
        bool_metric(payload, "diffusiongemma_guidance_moat") is True
        and headline_oracle is False
    )
    partial_blocked = _partial_state_blocked(payload)
    status = (
        "moat_holds"
        if moat_holds
        else ("partial_state_blocked" if partial_blocked else "ties_model_self_guidance")
    )
    return {
        "status": status,
        "guidance_moat_holds": moat_holds,
        "reported_diffusiongemma_guidance_moat": bool_metric(
            payload, "diffusiongemma_guidance_moat"
        ),
        "carnot_minus_rfg_delta": float_metric(payload, "carnot_minus_rfg_delta"),
        "carnot_minus_unguided_delta": float_metric(payload, "carnot_minus_unguided_delta"),
        "guidance_moat_ci95": list_metric(payload, "guidance_moat_ci95"),
        "guidance_ci95_excludes_zero": _ci_excludes_zero(payload, "guidance_moat_ci95"),
        "guidance_changes_selection": bool_metric(payload, "guidance_changes_selection"),
        "headline_arm_verifier_is_oracle": headline_oracle,
        "headline_arm_is_oracle_distinct": headline_oracle is False,
        "learned_partial_state_blocked": partial_blocked,
        "execution_grounded_arm": _execution_grounded_read(payload),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def arcgen_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    holds = (
        bool_metric(payload, "arcgen_cross_family_holds") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "hardens" if holds else "does_not_harden",
        "cross_family_hardens_on_arcgen": holds,
        "reported_arcgen_cross_family_holds": bool_metric(payload, "arcgen_cross_family_holds"),
        "cross_family_delta": float_metric(payload, "cross_family_delta"),
        "cross_family_ci95": list_metric(payload, "cross_family_ci95"),
        "per_substrate_delta": dict_metric(payload, "per_substrate_delta"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def self_learning_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    helps = (
        bool_metric(payload, "online_adaptation_helps") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "helps" if helps else "static_ceiling",
        "online_adaptation_helps": helps,
        "reported_online_adaptation_helps": bool_metric(payload, "online_adaptation_helps"),
        "static_cross_family_delta": float_metric(payload, "static_cross_family_delta"),
        "online_cross_family_delta": float_metric(payload, "online_cross_family_delta"),
        "online_minus_static_ci95": list_metric(payload, "online_minus_static_ci95"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def efficiency_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    cost_ratio = float_metric(payload, "cost_ratio")
    parity = (
        bool_metric(payload, "efficiency_parity_at_lower_cost") is True
        and cost_ratio is not None
        and cost_ratio <= 0.1
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "parity_at_lower_cost" if parity else "not_parity",
        "verifier_efficiency_parity": parity,
        "reported_efficiency_parity_at_lower_cost": bool_metric(
            payload, "efficiency_parity_at_lower_cost"
        ),
        "cost_ratio": cost_ratio,
        "accuracy_energy_verifier": float_metric(payload, "accuracy_energy_verifier"),
        "accuracy_llm_judge": float_metric(payload, "accuracy_llm_judge"),
        "accuracy_delta": float_metric(payload, "accuracy_delta"),
        "accuracy_delta_ci95": list_metric(payload, "accuracy_delta_ci95"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def arc_progress_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    total = int_metric(payload, "total_levels_solved") or int_metric(payload, "total_levels")
    return {
        "status": "included",
        "total_levels": total,
        "total_levels_solved": total,
        "levels_completed": int_metric(payload, "levels_completed"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task"),
        "game_advanced": str_metric(payload, "game_advanced"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def registry_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    gaps = list_metric(payload, "gaps_logged")
    return {
        "status": "included",
        "registry_reconciled": bool_metric(payload, "registry_reconciled"),
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed"),
        "gaps_logged_count": len(gaps),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def hardware_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    return {
        "status": "included",
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed"),
        "kv260_step_taken": str_metric(payload, "kv260_step_taken"),
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken"),
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def diffusiongemma_thesis_state(guidance: Mapping[str, Any]) -> str:
    if guidance.get("guidance_moat_holds") is True:
        return "external_verifier_improves_generation"
    if guidance.get("learned_partial_state_blocked") is True:
        return "partial_state_blocked"
    execution = guidance.get("execution_grounded_arm")
    if isinstance(execution, Mapping):
        delta = execution.get("execution_grounded_guidance_delta")
        if (
            isinstance(delta, (int, float))
            and not isinstance(delta, bool)
            and float(delta) > 0.0
            and execution.get("verifier_is_oracle") is True
        ):
            return "execution_grounded_only"
    return "ties_model_self_guidance"


def _status_part(status: str, clean: str, excluded: str, fallback: str) -> str:
    if status == clean:
        return clean
    if status == "excluded_flagged_adversarial":
        return excluded
    return fallback


def _headline_outcome(
    thesis_state: str,
    arcgen: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    arc_progress: Mapping[str, Any],
    paper_ready: bool,
) -> str:
    arcgen_part = _status_part(str(arcgen.get("status")), "hardens", "excluded_flagged", "not_hardened")
    efficiency_part = "parity" if efficiency.get("verifier_efficiency_parity") is True else "not_parity"
    self_status = str(self_learning.get("status"))
    if self_status == "helps":
        self_part = "helps"
    elif self_status == "excluded_flagged_adversarial":
        self_part = "excluded_flagged"
    else:
        self_part = "static_ceiling"
    total_levels = int(arc_progress.get("total_levels") or 0)
    game = str(arc_progress.get("game_advanced") or "unknown")
    paper = "paper_ready" if paper_ready else "paper_not_ready"
    return (
        f"{thesis_state}_arcgen_{arcgen_part}_efficiency_{efficiency_part}_"
        f"self_learning_{self_part}_arc{total_levels}_game_{game}_{paper}"
    )


def _honest_verdict(
    thesis_state: str,
    arcgen: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    arc_progress: Mapping[str, Any],
) -> str:
    arcgen_part = _status_part(str(arcgen.get("status")), "hardens", "excluded_flagged", "not_hardened")
    efficiency_part = "parity" if efficiency.get("verifier_efficiency_parity") is True else "not_parity"
    total_levels = int(arc_progress.get("total_levels") or 0)
    return (
        f"complete: diffusiongemma_{thesis_state}_arcgen_{arcgen_part}_"
        f"efficiency_{efficiency_part}_arc{total_levels}"
    )


def _oracle_violations(
    guidance: Mapping[str, Any],
    arcgen: Mapping[str, Any],
    efficiency: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    if (
        guidance.get("reported_diffusiongemma_guidance_moat") is True
        and guidance.get("headline_arm_verifier_is_oracle") is not False
    ):
        violations.append("4281_diffusiongemma:headline_guidance")
    if (
        arcgen.get("reported_arcgen_cross_family_holds") is True
        and arcgen.get("verifier_is_oracle") is not False
    ):
        violations.append("4282_arcgen:cross_family")
    if (
        efficiency.get("reported_efficiency_parity_at_lower_cost") is True
        and efficiency.get("verifier_is_oracle") is not False
    ):
        violations.append("4284_efficiency:efficiency")
    return violations


def _blocked_artifact(
    missing: list[JsonDict],
    started_s: float,
    now_s: float,
) -> JsonDict:
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(now_s - started_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "blocked_v396_artifacts_missing",
        "headline_outcome": "blocked_v396_artifacts_missing",
        "guidance_moat_holds": False,
        "cross_family_hardens_on_arcgen": False,
        "verifier_efficiency_parity": False,
        "diffusiongemma_thesis_state": "partial_state_blocked",
        "flagged_artifacts_excluded": [],
        "paper_ready": None,
        "unmet_gates": None,
        "publication_gate": None,
        "verifier_is_oracle_honored": True,
        "oracle_distinct_violations": [],
        "missing_upstream_artifacts": missing,
        "upstream_provenance": [],
        "reproducibility_checksum": BLOCKED_CHECKSUM,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": {
            field: {"principle": principle, "satisfied_by": "blocked precondition"}
            for field, principle in FIELD_PRINCIPLES.items()
        },
    }


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
) -> JsonDict:
    start = time.time() if started_s is None else started_s
    payloads, provenance, exclusions, missing = _read_inputs(root, live_flag_runner, summarize_runner)
    end_for_blocked = time.time() if now_s is None else now_s
    if missing:
        return _blocked_artifact(missing, start, end_for_blocked)

    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = {
        key: clean_payload(payloads.get(key), skipped.get(key, False))
        for key in DEFAULT_UPSTREAMS
    }

    guidance = guidance_read(clean["4281_diffusiongemma"], skipped["4281_diffusiongemma"])
    arcgen = arcgen_read(clean["4282_arcgen"], skipped["4282_arcgen"])
    self_learning = self_learning_read(clean["4283_self_learning"], skipped["4283_self_learning"])
    efficiency = efficiency_read(clean["4284_efficiency"], skipped["4284_efficiency"])
    arc_progress = arc_progress_read(clean["4285_arc_progress"], skipped["4285_arc_progress"])
    registry = registry_read(clean["4287_registry"], skipped["4287_registry"])
    hardware = hardware_read(clean["4288_hardware"], skipped["4288_hardware"])
    thesis_state = diffusiongemma_thesis_state(guidance)
    publication_gate = publication_gate_runner(root)
    paper_ready = bool(publication_gate.get("paper_ready"))
    violations = _oracle_violations(guidance, arcgen, efficiency)
    end = time.time() if now_s is None else now_s

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(thesis_state, arcgen, efficiency, arc_progress),
        "headline_outcome": _headline_outcome(
            thesis_state, arcgen, efficiency, self_learning, arc_progress, paper_ready
        ),
        "guidance_moat_holds": guidance.get("guidance_moat_holds") is True,
        "cross_family_hardens_on_arcgen": arcgen.get("cross_family_hardens_on_arcgen") is True,
        "verifier_efficiency_parity": efficiency.get("verifier_efficiency_parity") is True,
        "diffusiongemma_thesis_state": thesis_state,
        "flagged_artifacts_excluded": exclusions,
        "paper_ready": paper_ready,
        "unmet_gates": list_metric(publication_gate, "unmet_gates"),
        "publication_gate": publication_gate,
        "verifier_is_oracle_honored": not violations,
        "oracle_distinct_violations": violations,
        "diffusiongemma_guidance": guidance,
        "arcgen_cross_family": arcgen,
        "self_learning": self_learning,
        "efficiency": efficiency,
        "arc_progress": arc_progress,
        "registry_read": registry,
        "hardware_read": hardware,
        "missing_upstream_artifacts": [],
        "upstream_provenance": provenance,
        "reproducibility_checksum": checksum_from_provenance(provenance),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": {
            field: {"principle": principle, "satisfied_by": "aggregation logic"}
            for field, principle in FIELD_PRINCIPLES.items()
        },
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if verdict != "blocked_v396_artifacts_missing":
        if not isinstance(verdict, str) or not verdict.startswith(
            ("complete:", "success:", "passed:", "shipped:", "blocked:")
        ):
            raise ValueError("honest_verdict must be terminal-prefixed")
    headline = artifact.get("headline_outcome")
    if not isinstance(headline, str) or not headline:
        raise ValueError("headline_outcome must be a non-empty string")
    for field in (
        "guidance_moat_holds",
        "cross_family_hardens_on_arcgen",
        "verifier_efficiency_parity",
        "verifier_is_oracle_honored",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    if artifact.get("diffusiongemma_thesis_state") not in THESIS_STATES:
        raise ValueError("diffusiongemma_thesis_state is not recognized")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        raise ValueError("flagged_artifacts_excluded must be a list")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    blocked = artifact.get("honest_verdict") == "blocked_v396_artifacts_missing"
    paper_ready = artifact.get("paper_ready")
    if blocked:
        if paper_ready is not None:
            raise ValueError("blocked artifacts must not report paper_ready")
    elif not isinstance(paper_ready, bool):
        raise ValueError("paper_ready must be a bare bool")
    if not isinstance(artifact.get("upstream_provenance"), list):
        raise ValueError("upstream_provenance must be a list")
    principles = artifact.get("field_principles")
    if principles != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required principles")
    if not blocked:
        expected = checksum_from_provenance(artifact["upstream_provenance"])
        if artifact.get("reproducibility_checksum") != expected:
            raise ValueError("reproducibility_checksum does not match upstream sha256 set")
        for row in artifact["upstream_provenance"]:
            if not is_sha256(row.get("sha256")):
                raise ValueError("upstream provenance row has invalid sha256")
            if row.get("skipped") is True and row.get("fields_imported") != []:
                raise ValueError("skipped upstreams must not import fields")


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
) -> Path:
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
        publication_gate_runner=publication_gate_runner,
    )
    validate_artifact(artifact)
    path = root / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
