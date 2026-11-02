"""Build the Exp 4301 v397 capstone aggregation artifact.

Spec refs: REQ-CAPSTONE-4301, SCENARIO-CAPSTONE-4301.
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


OUTPUT_REL_PATH = Path("results/experiment_4301_capstone_v397.json")
EXPERIMENT_ID = 4301
RANDOM_SEED = 4301
SCHEMA = "carnot.capstone_v397_4301.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4301", "SCENARIO-CAPSTONE-4301"]
BLOCKED_CHECKSUM = hashlib.sha256(b"blocked_v397_artifacts_missing").hexdigest()

THESIS_STATES = {
    "cross_generator_moat_closed",
    "cross_generator_open_within_pool_only",
    "in_generation_moat_holds",
    "partial_state_scorer_leaked",
    "partial_state_blocked",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path
    core: bool = True


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4291_cross_generator": Upstream(
        4291, Path("results/experiment_4291_arcgen_cross_generator_nondegenerate.json")
    ),
    "4292_partial_state": Upstream(
        4292, Path("results/experiment_4292_partial_state_diffusion_scorer_build.json")
    ),
    "4293_generation": Upstream(
        4293, Path("results/experiment_4293_diffusiongemma_energy_guided_run_partial_state.json")
    ),
    "4294_efficiency": Upstream(
        4294, Path("results/experiment_4294_verifier_efficiency_harden_strong_judge.json")
    ),
    "4295_self_learning": Upstream(
        4295, Path("results/experiment_4295_self_learning_tier2_fixed_retrieval.json")
    ),
    "4296_arc_progress": Upstream(
        4296, Path("results/experiment_4296_arc_incremental_progress_new_game.json")
    ),
    "4299_registry": Upstream(
        4299, Path("results/experiment_4299_verifier_registry_gaps_hygiene.json"), False
    ),
    "4300_hardware": Upstream(
        4300, Path("results/experiment_4300_hardware_continuity.json"), False
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "cross_generator_moat_closes",
    "in_generation_moat_holds",
    "efficiency_pareto_hardened",
    "verifier_thesis_state",
    "flagged_artifacts_excluded",
    "paper_ready",
    "verifier_is_oracle_honored",
    "reproducibility_checksum",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .397 close-state -- whether the cross-generator "
        "moat closed, the in-generation thesis unblocked, the efficiency win hardened."
    ),
    "headline_outcome": (
        "One honest string aggregating the cross-generator + partial-state + "
        "in-generation + efficiency + self-learning + ARC reads; the single line "
        "the .398 planner frames from."
    ),
    "cross_generator_moat_closes": (
        "BARE bool: did the oracle-distinct selection win transfer to "
        "construction-disjoint generators on a NON-degenerate pool -- the LAST "
        "open axis of the selection moat (NOT the degenerate .396 +1.0)."
    ),
    "in_generation_moat_holds": (
        "BARE bool: did the LEARNED (oracle-distinct) partial-state-guided "
        "DiffusionGemma beat RFG model-self-guidance with CI95-excl-0 -- the §5 "
        "in-generation moat verdict (false if gated off / blocked / ties)."
    ),
    "efficiency_pareto_hardened": (
        "BARE bool: did the energy verifier match/beat a WELL-PROMPTED judge at "
        "<=0.1x cost -- the skeptic-proof §5 efficiency headline."
    ),
    "verifier_thesis_state": (
        "One honest string for the verifier thesis state "
        "(cross_generator_moat_closed / cross_generator_open_within_pool_only / "
        "in_generation_moat_holds / partial_state_scorer_leaked / "
        "partial_state_blocked) -- the framing the .398 planner inherits."
    ),
    "flagged_artifacts_excluded": (
        "List of .397 artifacts excluded for flagged_adversarial -- the "
        "fabrication gate (their numbers are NOT aggregated)."
    ),
    "paper_ready": (
        "From publication_gate.py --json -- the G1-G4 status (FoVer headline "
        "stays the publication target; a verified cross-generator moat or "
        "in-generation win would be a new headline-grade supporting result)."
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
    "4291_cross_generator": [
        "cross_generator_holds",
        "cross_generator_delta",
        "vote_at_1",
        "oracle_at_k",
        "per_substrate_delta",
        "non_degenerate_guards_pass",
        "verifier_is_oracle",
    ],
    "4292_partial_state": [
        "partial_state_scorer_built",
        "partial_state_leak_free",
        "partial_state_auroc",
        "leak_ablation_auroc",
        "verifier_is_oracle",
    ],
    "4293_generation": [
        "diffusiongemma_guidance_moat",
        "carnot_minus_rfg_delta",
        "guidance_moat_ci95",
        "guidance_changes_selection",
        "verifier_is_oracle",
    ],
    "4294_efficiency": [
        "efficiency_pareto_holds",
        "cost_ratio",
        "energy_accuracy",
        "best_prompted_judge_accuracy",
        "accuracy_delta_ci95",
        "verifier_is_oracle",
    ],
    "4295_self_learning": [
        "online_adaptation_helps",
        "static_cross_family_delta",
        "online_cross_family_delta",
        "online_minus_static_ci95",
        "verifier_is_oracle",
    ],
    "4296_arc_progress": [
        "total_levels",
        "total_levels_solved",
        "levels_completed",
        "new_levels_solved_this_task",
        "game_advanced",
    ],
    "4299_registry": ["registry_reconciled", "regression_guard_passed", "gaps_logged"],
    "4300_hardware": [
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


def run_publication_gate(root: Path) -> JsonDict:  # pragma: no cover
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


def _metric_from_top_or_pass_rates(payload: Mapping[str, Any] | None, field: str) -> float | None:
    direct = float_metric(payload, field)
    if direct is not None:
        return direct
    return float_metric(dict_metric(payload, "pass_rates"), field)


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
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "reason": "missing",
                }
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


def _no_import_provenance(provenance: list[JsonDict]) -> list[JsonDict]:
    return [dict(row, fields_imported=[]) for row in provenance]


def cross_generator_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    delta = float_metric(payload, "cross_generator_delta")
    vote_at_1 = _metric_from_top_or_pass_rates(payload, "vote_at_1")
    oracle_at_k = _metric_from_top_or_pass_rates(payload, "oracle_at_k")
    non_degenerate = (
        bool_metric(payload, "non_degenerate_guards_pass") is True
        and vote_at_1 is not None
        and vote_at_1 > 0.05
        and oracle_at_k is not None
        and oracle_at_k < 1.0
        and delta is not None
        and delta < 0.95
    )
    closes = (
        bool_metric(payload, "cross_generator_holds") is True
        and bool_metric(payload, "verifier_is_oracle") is False
        and non_degenerate
    )
    return {
        "status": "closed" if closes else "open_within_pool_only",
        "cross_generator_moat_closes": closes,
        "reported_cross_generator_holds": bool_metric(payload, "cross_generator_holds"),
        "cross_generator_delta": delta,
        "cross_generator_ci95": list_metric(payload, "cross_generator_ci95"),
        "vote_at_1": vote_at_1,
        "oracle_at_k": oracle_at_k,
        "per_substrate_delta": dict_metric(payload, "per_substrate_delta"),
        "reported_non_degenerate_guards_pass": bool_metric(
            payload, "non_degenerate_guards_pass"
        ),
        "non_degenerate_guards_held": non_degenerate,
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def partial_state_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    built = bool_metric(payload, "partial_state_scorer_built") is True
    leak_free = bool_metric(payload, "partial_state_leak_free") is True
    status = "leak_free" if built and leak_free else ("leaked" if built else "blocked")
    return {
        "status": status,
        "partial_state_scorer_built": built,
        "partial_state_leak_free": leak_free,
        "reported_partial_state_scorer_built": bool_metric(
            payload, "partial_state_scorer_built"
        ),
        "reported_partial_state_leak_free": bool_metric(payload, "partial_state_leak_free"),
        "partial_state_auroc": float_metric(payload, "partial_state_auroc"),
        "leak_ablation_auroc": float_metric(payload, "leak_ablation_auroc"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def generation_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    moat = (
        bool_metric(payload, "diffusiongemma_guidance_moat") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "moat_holds" if moat else "ties_or_blocked",
        "in_generation_moat_holds": moat,
        "reported_diffusiongemma_guidance_moat": bool_metric(
            payload, "diffusiongemma_guidance_moat"
        ),
        "carnot_minus_rfg_delta": float_metric(payload, "carnot_minus_rfg_delta"),
        "guidance_moat_ci95": list_metric(payload, "guidance_moat_ci95"),
        "guidance_changes_selection": bool_metric(payload, "guidance_changes_selection"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def efficiency_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    cost_ratio = float_metric(payload, "cost_ratio")
    hardened = (
        bool_metric(payload, "efficiency_pareto_holds") is True
        and cost_ratio is not None
        and cost_ratio <= 0.1
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "hardened" if hardened else "not_hardened",
        "efficiency_pareto_hardened": hardened,
        "reported_efficiency_pareto_holds": bool_metric(payload, "efficiency_pareto_holds"),
        "cost_ratio": cost_ratio,
        "energy_accuracy": float_metric(payload, "energy_accuracy"),
        "best_prompted_judge_accuracy": float_metric(payload, "best_prompted_judge_accuracy"),
        "accuracy_delta_ci95": list_metric(payload, "accuracy_delta_ci95"),
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


def arc_progress_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    total = int_metric(payload, "total_levels") or int_metric(payload, "total_levels_solved")
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


def verifier_thesis_state(
    cross_generator: Mapping[str, Any],
    partial_state: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> str:
    if partial_state.get("partial_state_scorer_built") is not True:
        return "partial_state_blocked"
    if partial_state.get("partial_state_leak_free") is not True:
        return "partial_state_scorer_leaked"
    if generation.get("in_generation_moat_holds") is True:
        return "in_generation_moat_holds"
    if cross_generator.get("cross_generator_moat_closes") is True:
        return "cross_generator_moat_closed"
    return "cross_generator_open_within_pool_only"


def _cross_part(cross_generator: Mapping[str, Any]) -> str:
    if cross_generator.get("cross_generator_moat_closes") is True:
        return "closed"
    if cross_generator.get("status") == "excluded_flagged_adversarial":
        return "excluded_flagged"
    return "open_within_pool_only"


def _partial_part(partial_state: Mapping[str, Any]) -> str:
    status = str(partial_state.get("status"))
    if status == "leak_free":
        return "leak_free"
    if status == "leaked":
        return "leaked"
    if status == "excluded_flagged_adversarial":
        return "excluded_flagged"
    return "blocked"


def _efficiency_part(efficiency: Mapping[str, Any]) -> str:
    if efficiency.get("efficiency_pareto_hardened") is True:
        return "hardened"
    if efficiency.get("status") == "excluded_flagged_adversarial":
        return "excluded_flagged"
    return "not_hardened"


def _self_learning_part(self_learning: Mapping[str, Any]) -> str:
    status = str(self_learning.get("status"))
    if status == "helps":
        return "helps"
    if status == "excluded_flagged_adversarial":
        return "excluded_flagged"
    return "static_ceiling"


def _headline_outcome(
    thesis_state: str,
    cross_generator: Mapping[str, Any],
    partial_state: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    arc_progress: Mapping[str, Any],
    paper_ready: bool,
) -> str:
    total_levels = int(arc_progress.get("total_levels") or 0)
    game = str(arc_progress.get("game_advanced") or "unknown")
    paper = "paper_ready" if paper_ready else "paper_not_ready"
    return (
        f"{thesis_state}_cross_generator_{_cross_part(cross_generator)}_"
        f"partial_state_{_partial_part(partial_state)}_efficiency_"
        f"{_efficiency_part(efficiency)}_self_learning_{_self_learning_part(self_learning)}_"
        f"arc{total_levels}_game_{game}_{paper}"
    )


def _honest_verdict(
    cross_generator: Mapping[str, Any],
    generation: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    arc_progress: Mapping[str, Any],
) -> str:
    cross = "closed" if cross_generator.get("cross_generator_moat_closes") is True else "open"
    gen = "moat" if generation.get("in_generation_moat_holds") is True else "not_moat"
    eff = "hardened" if efficiency.get("efficiency_pareto_hardened") is True else "not_hardened"
    total_levels = int(arc_progress.get("total_levels") or 0)
    return f"complete: v397_cross_generator_{cross}_in_generation_{gen}_efficiency_{eff}_arc{total_levels}"


def _oracle_violations(
    cross_generator: Mapping[str, Any],
    generation: Mapping[str, Any],
    efficiency: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    if (
        cross_generator.get("reported_cross_generator_holds") is True
        and cross_generator.get("verifier_is_oracle") is not False
    ):
        violations.append("4291_cross_generator:cross_generator")
    if (
        generation.get("reported_diffusiongemma_guidance_moat") is True
        and generation.get("verifier_is_oracle") is not False
    ):
        violations.append("4293_generation:in_generation")
    if (
        efficiency.get("reported_efficiency_pareto_holds") is True
        and efficiency.get("verifier_is_oracle") is not False
    ):
        violations.append("4294_efficiency:efficiency_pareto")
    return violations


def _blocked_artifact(
    missing: list[JsonDict],
    provenance: list[JsonDict],
    exclusions: list[JsonDict],
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
        "honest_verdict": "blocked_v397_artifacts_missing",
        "headline_outcome": "blocked_v397_artifacts_missing",
        "cross_generator_moat_closes": False,
        "in_generation_moat_holds": False,
        "efficiency_pareto_hardened": False,
        "verifier_thesis_state": "partial_state_blocked",
        "flagged_artifacts_excluded": exclusions,
        "paper_ready": None,
        "unmet_gates": None,
        "publication_gate": None,
        "verifier_is_oracle_honored": True,
        "oracle_distinct_violations": [],
        "missing_upstream_artifacts": missing,
        "upstream_provenance": _no_import_provenance(provenance),
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
    core_missing = [
        row for row in missing if DEFAULT_UPSTREAMS[str(row["artifact_key"])].core is True
    ]
    if core_missing:
        return _blocked_artifact(missing, provenance, exclusions, start, end_for_blocked)

    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = {
        key: clean_payload(payloads.get(key), skipped.get(key, False))
        for key in DEFAULT_UPSTREAMS
    }

    cross_generator = cross_generator_read(
        clean["4291_cross_generator"], skipped["4291_cross_generator"]
    )
    partial_state = partial_state_read(clean["4292_partial_state"], skipped["4292_partial_state"])
    generation = generation_read(clean["4293_generation"], skipped["4293_generation"])
    efficiency = efficiency_read(clean["4294_efficiency"], skipped["4294_efficiency"])
    self_learning = self_learning_read(clean["4295_self_learning"], skipped["4295_self_learning"])
    arc_progress = arc_progress_read(clean["4296_arc_progress"], skipped["4296_arc_progress"])
    registry = registry_read(clean["4299_registry"], skipped.get("4299_registry", False))
    hardware = hardware_read(clean["4300_hardware"], skipped.get("4300_hardware", False))
    thesis_state = verifier_thesis_state(cross_generator, partial_state, generation)
    publication_gate = publication_gate_runner(root)
    paper_ready = bool(publication_gate.get("paper_ready"))
    violations = _oracle_violations(cross_generator, generation, efficiency)
    end = time.time() if now_s is None else now_s

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(cross_generator, generation, efficiency, arc_progress),
        "headline_outcome": _headline_outcome(
            thesis_state,
            cross_generator,
            partial_state,
            efficiency,
            self_learning,
            arc_progress,
            paper_ready,
        ),
        "cross_generator_moat_closes": cross_generator.get("cross_generator_moat_closes")
        is True,
        "in_generation_moat_holds": generation.get("in_generation_moat_holds") is True,
        "efficiency_pareto_hardened": efficiency.get("efficiency_pareto_hardened") is True,
        "verifier_thesis_state": thesis_state,
        "flagged_artifacts_excluded": exclusions,
        "paper_ready": paper_ready,
        "unmet_gates": list_metric(publication_gate, "unmet_gates"),
        "publication_gate": publication_gate,
        "verifier_is_oracle_honored": not violations,
        "oracle_distinct_violations": violations,
        "cross_generator": cross_generator,
        "partial_state_scorer": partial_state,
        "in_generation": generation,
        "efficiency": efficiency,
        "self_learning": self_learning,
        "arc_progress": arc_progress,
        "registry_read": registry,
        "hardware_read": hardware,
        "missing_upstream_artifacts": missing,
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
    if verdict != "blocked_v397_artifacts_missing":
        if not isinstance(verdict, str) or not verdict.startswith(
            ("complete:", "success:", "passed:", "shipped:", "blocked:")
        ):
            raise ValueError("honest_verdict must be terminal-prefixed")
    headline = artifact.get("headline_outcome")
    if not isinstance(headline, str) or not headline:
        raise ValueError("headline_outcome must be a non-empty string")
    for field in (
        "cross_generator_moat_closes",
        "in_generation_moat_holds",
        "efficiency_pareto_hardened",
        "verifier_is_oracle_honored",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    if artifact.get("verifier_thesis_state") not in THESIS_STATES:
        raise ValueError("verifier_thesis_state is not recognized")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        raise ValueError("flagged_artifacts_excluded must be a list")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    blocked = artifact.get("honest_verdict") == "blocked_v397_artifacts_missing"
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
