"""Exp 4193 registry/gaps hygiene for .388 verifier outcomes.

Spec refs: REQ-VERIFY-4193, SCENARIO-VERIFY-4193.

This runner replays the frozen GAP-4 ARC-1 guard from cached artifacts and
records the .388 efficiency-moat, graded-gate, and sovereign-generator outcomes
from already-written result JSON. It does not run Codex, load GGUF models, use
a GPU, launch TRM training, stop TRM training, or write the stable TRM
checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4153 as exp4153


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "cached_gap4_replay_and_ledger_reconciliation"

EXP4193_ARTIFACT_PATH = "results/experiment_4193_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = exp4153.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4153.ARC1_PROGRAMS_PATH
EXP4186_PATH = "results/experiment_4186_efficiency_moat_verifier_vs_llm_judge.json"
EXP4187_PATH = "results/experiment_4187_gap4_graded_execution_gate_hardening.json"
EXP4188_PATH = "results/experiment_4188_sovereign_local_generator_gap4_self_distill.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
EFFICIENCY_MOAT_GAP_ID = "GAP-MOAT-EFFICIENCY-JUDGE-4186"
GAP4_GRADED_GATE_GAP_ID = "GAP-4-GRADED-GATE-4187"
SOVEREIGN_GENERATOR_GAP_ID = "GAP-SOVEREIGN-LOCAL-GAP4-GENERATOR-4188"
V388_ROLE_ID = "verifier_efficiency_gap4_sovereign_hygiene_4193"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "efficiency_moat",
    "gap4_graded_gate",
    "sovereign_generator",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .388 truth.",
    "regression_guard_passed": (
        "Bare bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched (efficiency moat + GAP-4 graded gate + "
        "sovereign generator)."
    ),
}


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _check_json_resource(repo_root: Path, resource: str, rel_path: str) -> dict[str, Any]:
    path = repo_root / rel_path
    if not path.exists():
        return {"resource": resource, "available": False, "detail": f"missing: {rel_path}"}
    try:
        loaded = base._load_json(path)
    except Exception as exc:  # pragma: no cover - parse exception type varies.
        return {"resource": resource, "available": False, "detail": f"parse_error: {exc}"}
    if not isinstance(loaded, dict):  # pragma: no cover - defensive for corrupted artifacts.
        return {"resource": resource, "available": False, "detail": "not_json_object"}
    return {"resource": resource, "available": True, "detail": rel_path}


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4193: verify cached fixtures, upstream artifacts, and ledgers."""
    base_preflight = exp4153.check_preconditions(repo_root)
    checks = list(base_preflight["checks"]) + [
        _check_json_resource(repo_root, "exp4186_efficiency_moat", EXP4186_PATH),
        _check_json_resource(repo_root, "exp4187_gap4_graded_gate", EXP4187_PATH),
        _check_json_resource(repo_root, "exp4188_sovereign_generator", EXP4188_PATH),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4193: replay GAP-4 ARC-1 from cached artifacts only."""
    return exp4153.replay_gap4_arc1(repo_root)


def _efficiency_moat_status(outcome: dict[str, Any]) -> str:
    if bool(outcome.get("verifier_efficiency_win")):
        return "filled_verifier_efficiency_win"
    return "open_efficiency_moat_not_filled"


def classify_efficiency_moat(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4193: summarize the Exp 4186 efficiency-moat verdict."""
    artifact = base._load_json(repo_root / EXP4186_PATH)
    outcome = {
        "gap_id": EFFICIENCY_MOAT_GAP_ID,
        "status": "",
        "artifact_path": EXP4186_PATH,
        "source_artifacts": [EXP4186_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "verifier_efficiency_win": artifact.get("verifier_efficiency_win") is True,
        "accuracy_parity_vs_judge": dict(artifact.get("accuracy_parity_vs_judge", {})),
        "cost_ratio_vs_judge": dict(artifact.get("cost_ratio_vs_judge", {})),
        "positive_control_confirmed": artifact.get("positive_control_confirmed") is True,
        "arms": dict(artifact.get("arms", {})),
        "model_specs": dict(artifact.get("model_specs", {})),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
        "inference_substrate": str(artifact.get("inference_substrate", "")),
        "missing_discriminator": "none_when_efficiency_win_true",
    }
    outcome["status"] = _efficiency_moat_status(outcome)
    return outcome


def _gap4_graded_status(outcome: dict[str, Any]) -> str:
    if (
        outcome.get("vote_aware_guard_blocked_mispromotion") is True
        and outcome.get("pass2_vote_wins_lost") == 0
        and dict(outcome.get("gross_recovery_ledger", {})).get("recovered") == 4
        and dict(outcome.get("gross_recovery_ledger", {})).get("lost") == 0
    ):
        return "filled_guarded_graded_gate_holds_plus4_minus0"
    return "open_graded_gate_regression"


def classify_gap4_graded_gate(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4193: summarize the Exp 4187 guarded graded gate."""
    artifact = base._load_json(repo_root / EXP4187_PATH)
    outcome = {
        "gap_id": GAP4_GRADED_GATE_GAP_ID,
        "status": "",
        "artifact_path": EXP4187_PATH,
        "source_artifacts": [EXP4187_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "graded_gate_pass2_vs_vote": _numeric_or_none(
            artifact.get("graded_gate_pass2_vs_vote")
        ),
        "gross_recovery_ledger": dict(artifact.get("gross_recovery_ledger", {})),
        "vote_aware_guard_blocked_mispromotion": (
            artifact.get("vote_aware_guard_blocked_mispromotion") is True
        ),
        "vote_aware_guard": dict(artifact.get("vote_aware_guard", {})),
        "pass2_vote_wins_lost": artifact.get("pass2_vote_wins_lost"),
        "pass_at_1": dict(artifact.get("pass_at_1", {})),
        "pass_at_2": dict(artifact.get("pass_at_2", {})),
        "gate_fire_count": artifact.get("gate_fire_count"),
        "guard_block_count": artifact.get("guard_block_count"),
        "band_precision_at_tau": dict(artifact.get("band_precision_at_tau", {})),
        "agreement_confidence_label_only": (
            artifact.get("agreement_confidence_label_only") is True
        ),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
        "inference_substrate": str(artifact.get("inference_substrate", "")),
        "missing_discriminator": "none_for_guarded_arc1_pass2_baseline",
    }
    outcome["status"] = _gap4_graded_status(outcome)
    return outcome


def _sovereign_generator_status(outcome: dict[str, Any]) -> str:
    local = dict(outcome.get("sovereign_pool_pass2", {})).get("LOCAL_HARDENED_GATE")
    vote = dict(outcome.get("sovereign_pool_pass2", {})).get("TRM_VOTE")
    positive = (
        isinstance(local, (int, float))
        and isinstance(vote, (int, float))
        and float(local) > float(vote)
        and int(outcome.get("self_distillation_corpus_size", 0)) > 0
    )
    if positive and bool(outcome.get("flagged_adversarial")):
        return "building_sovereign_local_generator_positive_flagged"
    if positive:
        return "building_sovereign_local_generator_positive"
    return "open_sovereign_local_generator_not_established"


def classify_sovereign_generator(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4193: summarize the Exp 4188 sovereign generator result."""
    artifact = base._load_json(repo_root / EXP4188_PATH)
    outcome = {
        "gap_id": SOVEREIGN_GENERATOR_GAP_ID,
        "status": "",
        "artifact_path": EXP4188_PATH,
        "source_artifacts": [EXP4188_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "local_induction_rate": dict(artifact.get("local_induction_rate", {})),
        "sovereign_pool_pass2": dict(artifact.get("sovereign_pool_pass2", {})),
        "self_distillation_corpus_size": artifact.get("self_distillation_corpus_size"),
        "self_distillation_corpus_path": str(
            artifact.get("self_distillation_corpus_path", "")
        ),
        "no_closed_weight_call": artifact.get("no_closed_weight_call") is True,
        "model_specs": dict(artifact.get("model_specs", {})),
        "prior_local_moe_null": str(artifact.get("prior_local_moe_null", "")),
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "corrigendum_pending": list(artifact.get("corrigendum_pending", [])),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
        "inference_substrate": str(artifact.get("inference_substrate", "")),
        "missing_discriminator": "stronger_local_generator_or_verifier_guided_induction",
    }
    outcome["status"] = _sovereign_generator_status(outcome)
    return outcome


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    efficiency_moat: dict[str, Any],
    gap4_graded_gate: dict[str, Any],
    sovereign_generator: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with the .388 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(
        updated_registry,
        offline_replay,
        efficiency_moat,
        gap4_graded_gate,
        sovereign_generator,
    )
    _ensure_v388_role(
        updated_registry,
        efficiency_moat,
        gap4_graded_gate,
        sovereign_generator,
    )

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4193-efficiency-moat",
        _efficiency_moat_gap_block(efficiency_moat),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4193-gap4-graded-gate",
        _gap4_graded_gate_gap_block(gap4_graded_gate),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4193-sovereign-generator",
        _sovereign_generator_gap_block(sovereign_generator),
    )
    touched = [
        gap_id
        for gap_id in (
            EFFICIENCY_MOAT_GAP_ID,
            GAP4_GRADED_GATE_GAP_ID,
            SOVEREIGN_GENERATOR_GAP_ID,
        )
        if gap_id in updated_gaps
    ]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "efficiency_moat_recorded": EFFICIENCY_MOAT_GAP_ID in touched,
            "gap4_graded_gate_recorded": GAP4_GRADED_GATE_GAP_ID in touched,
            "sovereign_generator_recorded": SOVEREIGN_GENERATOR_GAP_ID in touched,
        },
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    offline_replay: dict[str, Any],
    efficiency_moat: dict[str, Any],
    gap4_graded_gate: dict[str, Any],
    sovereign_generator: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - real/minimal registries include this entry.
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    sovereign_pass2 = sovereign_generator.get("sovereign_pool_pass2", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4193": EXP4193_ARTIFACT_PATH,
            "exp4193_regression_guard_passed": bool(
                offline_replay.get("regression_guard_passed")
            ),
            "exp4193_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4193_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4193_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4193_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4193_verifier_efficiency_win": bool(
                efficiency_moat.get("verifier_efficiency_win")
            ),
            "exp4193_gap4_graded_gate_pass2_vs_vote": gap4_graded_gate.get(
                "graded_gate_pass2_vs_vote"
            ),
            "exp4193_sovereign_local_hardened_gate_pass2": sovereign_pass2.get(
                "LOCAL_HARDENED_GATE"
            ),
            "exp4193_sovereign_self_distillation_corpus_size": sovereign_generator.get(
                "self_distillation_corpus_size"
            ),
        }
    )


def _ensure_v388_role(
    registry: dict[str, Any],
    efficiency_moat: dict[str, Any],
    gap4_graded_gate: dict[str, Any],
    sovereign_generator: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - guarded by _ensure_gap4_eval.
        return
    role = {
        "role_id": V388_ROLE_ID,
        "experiment": EXP4193_ARTIFACT_PATH,
        "role": "registry_gap_ledger_hygiene_v388",
        "status": "efficiency_moat_gap4_graded_sovereign_recorded",
        "efficiency_moat_gap_id": efficiency_moat.get("gap_id"),
        "efficiency_moat_status": efficiency_moat.get("status"),
        "efficiency_moat_artifact": EXP4186_PATH,
        "verifier_efficiency_win": bool(efficiency_moat.get("verifier_efficiency_win")),
        "accuracy_parity_vs_judge": efficiency_moat.get("accuracy_parity_vs_judge", {}),
        "cost_ratio_vs_judge": efficiency_moat.get("cost_ratio_vs_judge", {}),
        "gap4_graded_gate_gap_id": gap4_graded_gate.get("gap_id"),
        "gap4_graded_gate_status": gap4_graded_gate.get("status"),
        "gap4_graded_gate_artifact": EXP4187_PATH,
        "graded_gate_pass2_vs_vote": gap4_graded_gate.get("graded_gate_pass2_vs_vote"),
        "vote_aware_guard_blocked_mispromotion": bool(
            gap4_graded_gate.get("vote_aware_guard_blocked_mispromotion")
        ),
        "sovereign_generator_gap_id": sovereign_generator.get("gap_id"),
        "sovereign_generator_status": sovereign_generator.get("status"),
        "sovereign_generator_artifact": EXP4188_PATH,
        "local_induction_rate": sovereign_generator.get("local_induction_rate", {}),
        "sovereign_pool_pass2": sovereign_generator.get("sovereign_pool_pass2", {}),
        "self_distillation_corpus_size": sovereign_generator.get(
            "self_distillation_corpus_size"
        ),
        "flagged_adversarial": bool(sovereign_generator.get("flagged_adversarial")),
        "eval_exp_4193": EXP4193_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V388_ROLE_ID
    ] + [role]


def _efficiency_moat_gap_block(outcome: dict[str, Any]) -> str:
    accuracy = outcome.get("accuracy_parity_vs_judge", {})
    cost = outcome.get("cost_ratio_vs_judge", {})
    return (
        f"### {EFFICIENCY_MOAT_GAP_ID}: Exp 4193 .388 efficiency moat versus LLM judge\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4186_PATH}`; "
        f"verifier_efficiency_win={str(bool(outcome.get('verifier_efficiency_win'))).lower()}; "
        f"accuracy_parity_vs_judge_delta={accuracy.get('delta')}; "
        f"accuracy_parity_vs_judge_ci95={accuracy.get('ci95')}; "
        f"cost_ratio_wall_clock={cost.get('wall_clock')}; "
        f"wall_clock_x_cheaper={cost.get('wall_clock_x_cheaper')}; "
        f"ten_x_cheaper_on_both_axes={str(bool(cost.get('ten_x_cheaper_on_both_axes'))).lower()}; "
        f"strictly_pareto_dominant={str(bool(cost.get('strictly_pareto_dominant'))).lower()}; "
        f"positive_control_confirmed={str(bool(outcome.get('positive_control_confirmed'))).lower()}.\n"
        "- failure mode: closed for the measured .388 efficiency moat only when the "
        "cheap verifier matches or beats the judge on accuracy while dominating real "
        "cost; otherwise GAP-MOAT remains open.\n"
        "- missing discriminator: none for this measured code-domain efficiency moat "
        "because verifier_efficiency_win=true.\n"
        "- candidate design: preserve the real-cost LLM-judge comparator and the "
        "objective headroom-positive pool for future moat checks.\n"
        "- priority: medium\n"
    )


def _gap4_graded_gate_gap_block(outcome: dict[str, Any]) -> str:
    recovery = outcome.get("gross_recovery_ledger", {})
    return (
        f"### {GAP4_GRADED_GATE_GAP_ID}: Exp 4193 .388 GAP-4 guarded graded gate\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4187_PATH}`; "
        f"graded_gate_pass2_vs_vote={outcome.get('graded_gate_pass2_vs_vote')}; "
        f"gross_recovery_ledger.recovered={recovery.get('recovered')}; "
        f"gross_recovery_ledger.lost={recovery.get('lost')}; "
        f"pass2_vote_wins_lost={outcome.get('pass2_vote_wins_lost')}; "
        f"vote_aware_guard_blocked_mispromotion="
        f"{str(bool(outcome.get('vote_aware_guard_blocked_mispromotion'))).lower()}; "
        f"agreement_confidence_label_only="
        f"{str(bool(outcome.get('agreement_confidence_label_only'))).lower()}.\n"
        "- failure mode: the graded relaxation adds no ARC-1 recovery beyond the exact "
        "baseline, but the guarded policy preserves the +4/-0 pass@2 safety record and "
        "blocks the recorded high-vote-gold mispromotion.\n"
        "- missing discriminator: none for the guarded ARC-1 pass@2 baseline; keep "
        "vote-aware guarding load-bearing for future graded relaxations.\n"
        "- candidate design: use the guarded tau=0.005 graded execution gate while "
        "recording agreement only as a confidence label.\n"
        "- priority: high\n"
    )


def _sovereign_generator_gap_block(outcome: dict[str, Any]) -> str:
    induction = outcome.get("local_induction_rate", {})
    sovereign = outcome.get("sovereign_pool_pass2", {})
    return (
        f"### {SOVEREIGN_GENERATOR_GAP_ID}: Exp 4193 .388 sovereign local GAP-4 generator\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4188_PATH}`; "
        f"local_induction_rate={induction.get('rate')}; "
        f"local_demo_perfect={induction.get('demo_perfect')}; "
        f"local_total={induction.get('total')}; "
        f"sovereign_pool_pass2.LOCAL_HARDENED_GATE={sovereign.get('LOCAL_HARDENED_GATE')}; "
        f"sovereign_pool_pass2.TRM_VOTE={sovereign.get('TRM_VOTE')}; "
        f"sovereign_pool_pass2.delta_vs_vote={sovereign.get('delta_vs_vote')}; "
        f"self_distillation_corpus_size={outcome.get('self_distillation_corpus_size')}; "
        f"no_closed_weight_call={str(bool(outcome.get('no_closed_weight_call'))).lower()}; "
        f"flagged_adversarial={str(bool(outcome.get('flagged_adversarial'))).lower()}.\n"
        "- failure mode: the local generator recovers a positive guarded pass@2 lift and "
        "banks a verifier-labeled corpus, but its induction rate remains far below the "
        "Codex reference and the source artifact carries adversarial caveats.\n"
        "- missing discriminator: stronger local program induction or verifier-guided "
        "generation that surfaces demo-perfect programs without closed-weight calls.\n"
        "- candidate design: continue sovereign generator improvement and self-distill "
        "from verifier-labeled demo-perfect programs before any clean filled claim.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4193") == EXP4193_ARTIFACT_PATH
        and any(role.get("role_id") == V388_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    efficiency_moat: dict[str, Any],
    gap4_graded_gate: dict[str, Any],
    sovereign_generator: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4193 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    needed = {
        EFFICIENCY_MOAT_GAP_ID,
        GAP4_GRADED_GATE_GAP_ID,
        SOVEREIGN_GENERATOR_GAP_ID,
    }
    gaps_complete = needed.issubset(set(gaps_updated))
    prefix = "complete:" if guard_ok and gaps_complete and registry_updated else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4193_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4193_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v388_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"efficiency_{efficiency_moat['status']}_"
            f"gap4_{gap4_graded_gate['status']}_"
            f"sovereign_{sovereign_generator['status']}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "efficiency_moat": efficiency_moat,
        "gap4_graded_gate": gap4_graded_gate,
        "sovereign_generator": sovereign_generator,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4186_PATH,
            EXP4187_PATH,
            EXP4188_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4193", "SCENARIO-VERIFY-4193"],
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    artifact = {
        "experiment": "experiment_4193_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4193_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "registry_updated": False,
        "efficiency_moat": {
            "status": "blocked_precondition",
            "gap_id": EFFICIENCY_MOAT_GAP_ID,
            "verifier_efficiency_win": False,
        },
        "gap4_graded_gate": {
            "status": "blocked_precondition",
            "gap_id": GAP4_GRADED_GATE_GAP_ID,
        },
        "sovereign_generator": {
            "status": "blocked_precondition",
            "gap_id": SOVEREIGN_GENERATOR_GAP_ID,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "preconditions": preflight,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4186_PATH,
            EXP4187_PATH,
            EXP4188_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4193", "SCENARIO-VERIFY-4193"],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4193 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["regression_guard_passed"], bool):
        raise ValueError("regression_guard_passed must be a bare bool")
    if not isinstance(artifact["registry_updated"], bool):
        raise ValueError("registry_updated must be a bare bool")
    if not isinstance(artifact["gaps_updated"], list):
        raise ValueError("gaps_updated must be a list")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4193 principles")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4193 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4193_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    efficiency_moat = classify_efficiency_moat(repo_root)
    gap4_graded_gate = classify_gap4_graded_gate(repo_root)
    sovereign_generator = classify_sovereign_generator(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        efficiency_moat,
        gap4_graded_gate,
        sovereign_generator,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        efficiency_moat=efficiency_moat,
        gap4_graded_gate=gap4_graded_gate,
        sovereign_generator=sovereign_generator,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4193_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
