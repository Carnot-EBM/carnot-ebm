"""Exp 4204 registry/gaps hygiene for .389 verifier-as-reward outcomes.

Spec refs: REQ-VERIFY-4204, SCENARIO-VERIFY-4204.

This runner is an evidence reconciler. It replays the frozen GAP-4 ARC-1
regression guard from cached artifacts and records what the .389 reward
artifacts actually say. That matters because a blocked A-vs-B collection is not
negative proof about the verifier label, but it is also not a positive
training-time result. The ledger must preserve that distinction.
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

EXP4204_ARTIFACT_PATH = "results/experiment_4204_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = exp4153.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4153.ARC1_PROGRAMS_PATH
EXP4197_PATH = "results/experiment_4197_verifier_reward_phase0_headroom_harness_build.json"
EXP4199_PATH = "results/experiment_4199_verifier_reward_decisive_a_vs_b_collect.json"
EXP4200_PATH = "results/experiment_4200_certified_arc_corpus_distill_lift.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
GAP_REWARD_GAP_ID = "GAP-REWARD"
CERTIFIED_CORPUS_GAP_ID = "GAP-REWARD-CERTIFIED-CORPUS-4200"
V389_ROLE_ID = "verifier_reward_gap4_corpus_hygiene_4204"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "verifier_reward_outcome",
    "certified_arc_corpus",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .389 truth.",
    "regression_guard_passed": (
        "BARE bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched (the GAP-REWARD verifier-as-reward axis + "
        "the certified-corpus note)."
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
    except Exception as exc:  # pragma: no cover - exact JSON error type varies.
        return {"resource": resource, "available": False, "detail": f"parse_error: {exc}"}
    if not isinstance(loaded, dict):  # pragma: no cover - defensive for corrupted artifacts.
        return {"resource": resource, "available": False, "detail": "not_json_object"}
    return {"resource": resource, "available": True, "detail": rel_path}


def _first_numeric(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _numeric_or_none(payload.get(key))
        if value is not None:
            return value
    return None


def _first_ci95(payload: dict[str, Any], *keys: str) -> list[float] | None:
    for key in keys:
        value = payload.get(key)
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(_numeric_or_none(item) is not None for item in value)
        ):
            return [float(value[0]), float(value[1])]
    return None


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4204: verify cached replay fixtures, upstream artifacts, and ledgers."""
    base_preflight = exp4153.check_preconditions(repo_root)
    checks = list(base_preflight["checks"]) + [
        _check_json_resource(repo_root, "exp4197_phase0", EXP4197_PATH),
        _check_json_resource(repo_root, "exp4199_a_vs_b", EXP4199_PATH),
        _check_json_resource(repo_root, "exp4200_certified_arc_corpus", EXP4200_PATH),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4204: replay GAP-4 ARC-1 from cached artifacts only."""
    return exp4153.replay_gap4_arc1(repo_root)


def _training_launched_from_exp4199(artifact: dict[str, Any]) -> bool | None:
    for gate in artifact.get("gates_evaluated", []):
        if not isinstance(gate, dict):
            continue
        if str(gate.get("artifact_field", "")).endswith("training_launched"):
            actual = gate.get("actual")
            if isinstance(actual, bool):
                return actual
    return None


def _verifier_reward_status(outcome: dict[str, Any]) -> str:
    ci = outcome.get("a_vs_b_ci95")
    if (
        outcome.get("verifier_label_carries_signal") is True
        and _numeric_or_none(outcome.get("a_vs_b_delta")) is not None
        and isinstance(ci, list)
        and len(ci) == 2
        and float(ci[0]) > 0.0
    ):
        return "filled_verifier_label_carries_signal"
    if outcome.get("training_launched") is False:
        return "blocked_a_vs_b_not_collected_training_not_launched"
    if outcome.get("verifier_label_carries_signal") is False and outcome.get("a_vs_b_delta") == 0.0:
        return "open_no_a_vs_b_signal"
    return "open_a_vs_b_not_decision_grade"


def classify_verifier_reward_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4204: summarize Exp 4197/4199 without fabricating A-vs-B evidence."""
    phase0 = base._load_json(repo_root / EXP4197_PATH)
    a_vs_b = base._load_json(repo_root / EXP4199_PATH)

    phase0_precision = _numeric_or_none(phase0.get("phase0_precision"))
    youden_j = _numeric_or_none(phase0.get("youden_j"))
    training_launched = _training_launched_from_exp4199(a_vs_b)
    if training_launched is None:
        training_launched = bool(a_vs_b.get("training_launched"))

    verifier_label = a_vs_b.get("verifier_label_carries_signal")
    verifier_label_carries_signal = verifier_label is True
    a_vs_b_delta = _first_numeric(a_vs_b, "a_vs_b_delta", "arm_a_vs_b_delta", "rft_a_vs_b_delta")
    a_vs_b_ci95 = _first_ci95(a_vs_b, "a_vs_b_ci95", "arm_a_vs_b_ci95", "rft_a_vs_b_ci95")

    outcome = {
        "gap_id": GAP_REWARD_GAP_ID,
        "status": "",
        "artifact_path": EXP4199_PATH,
        "source_artifacts": [EXP4197_PATH, EXP4199_PATH],
        "honest_verdict": str(a_vs_b.get("honest_verdict", "")),
        "phase0_honest_verdict": str(phase0.get("honest_verdict", "")),
        "phase0_precision": phase0_precision,
        "youden_j": youden_j,
        "phase0_gate_clean": (
            phase0_precision is not None
            and phase0_precision >= 0.85
            and youden_j is not None
            and youden_j > 0.0
            and phase0.get("harness_ready") is True
            and phase0.get("training_headroom_present") is True
        ),
        "training_launched": training_launched,
        "verifier_label_carries_signal": verifier_label_carries_signal,
        "a_vs_b_delta": a_vs_b_delta,
        "a_vs_b_ci95": a_vs_b_ci95,
        "gate_check_summary": str(a_vs_b.get("gate_check_summary", "")),
        "blocked_at_layer": str(a_vs_b.get("blocked_at_layer", "")),
        "status_from_artifact": str(a_vs_b.get("status", "")),
        "missing_discriminator": (
            "decision_grade_a_vs_b_training_signal_beyond_random_label_control"
        ),
    }
    outcome["status"] = _verifier_reward_status(outcome)
    return outcome


def _certified_corpus_status(outcome: dict[str, Any]) -> str:
    size = outcome.get("certified_corpus_size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        return "certified_corpus_empty"
    diagnosis = str(outcome.get("distill_lift_latent_vs_absent", ""))
    if diagnosis == "latent":
        return "certified_corpus_built_distill_lift_latent"
    if diagnosis == "absent":
        return "certified_corpus_built_distill_lift_absent"
    return "certified_corpus_built_distill_lift_uninformative"


def classify_certified_arc_corpus(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4204: summarize Exp 4200 corpus size, precision, and lift read."""
    artifact = base._load_json(repo_root / EXP4200_PATH)
    outcome = {
        "gap_id": CERTIFIED_CORPUS_GAP_ID,
        "status": "",
        "artifact_path": EXP4200_PATH,
        "source_artifacts": [EXP4200_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "certified_corpus_size": artifact.get("certified_corpus_size"),
        "certified_corpus_path": str(artifact.get("certified_corpus_path", "")),
        "certification_precision": dict(artifact.get("certification_precision", {})),
        "local_induction_cold": dict(artifact.get("local_induction_cold", {})),
        "local_induction_with_certified_exemplars": dict(
            artifact.get("local_induction_with_certified_exemplars", {})
        ),
        "distill_lift_ci95": list(artifact.get("distill_lift_ci95", [])),
        "distill_lift_latent_vs_absent": str(artifact.get("invisible_leash_diagnosis", "")),
        "seeded_generation_status": str(artifact.get("seeded_generation_status", "")),
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "corrigendum_pending": list(artifact.get("corrigendum_pending", [])),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
        "inference_substrate": str(artifact.get("inference_substrate", "")),
        "missing_discriminator": (
            "seeded_local_generation_or_lora_distillation_lift_from_certified_arc_corpus"
        ),
    }
    outcome["status"] = _certified_corpus_status(outcome)
    return outcome


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    certified_arc_corpus: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gap text with the .389 reward outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(
        updated_registry, offline_replay, verifier_reward_outcome, certified_arc_corpus
    )
    _ensure_v389_role(updated_registry, verifier_reward_outcome, certified_arc_corpus)

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4204-gap-reward",
        _verifier_reward_gap_block(verifier_reward_outcome),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4204-certified-corpus",
        _certified_corpus_gap_block(certified_arc_corpus),
    )
    touched = [
        gap_id for gap_id in (GAP_REWARD_GAP_ID, CERTIFIED_CORPUS_GAP_ID) if gap_id in updated_gaps
    ]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "verifier_reward_recorded": GAP_REWARD_GAP_ID in touched,
            "certified_corpus_recorded": CERTIFIED_CORPUS_GAP_ID in touched,
        },
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    offline_replay: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    certified_arc_corpus: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - real/minimal registries include this entry.
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4204": EXP4204_ARTIFACT_PATH,
            "exp4204_regression_guard_passed": bool(offline_replay.get("regression_guard_passed")),
            "exp4204_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4204_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4204_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4204_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4204_verifier_label_carries_signal": bool(
                verifier_reward_outcome.get("verifier_label_carries_signal")
            ),
            "exp4204_phase0_precision": verifier_reward_outcome.get("phase0_precision"),
            "exp4204_youden_j": verifier_reward_outcome.get("youden_j"),
            "exp4204_certified_corpus_size": certified_arc_corpus.get("certified_corpus_size"),
            "exp4204_distill_lift_latent_vs_absent": certified_arc_corpus.get(
                "distill_lift_latent_vs_absent"
            ),
        }
    )


def _ensure_v389_role(
    registry: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    certified_arc_corpus: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - guarded by _ensure_gap4_eval.
        return
    role = {
        "role_id": V389_ROLE_ID,
        "experiment": EXP4204_ARTIFACT_PATH,
        "role": "registry_gap_ledger_hygiene_v389",
        "status": "verifier_reward_and_certified_corpus_recorded",
        "verifier_reward_gap_id": verifier_reward_outcome.get("gap_id"),
        "verifier_reward_status": verifier_reward_outcome.get("status"),
        "verifier_reward_artifact": EXP4199_PATH,
        "phase0_precision": verifier_reward_outcome.get("phase0_precision"),
        "youden_j": verifier_reward_outcome.get("youden_j"),
        "training_launched": verifier_reward_outcome.get("training_launched"),
        "verifier_label_carries_signal": bool(
            verifier_reward_outcome.get("verifier_label_carries_signal")
        ),
        "a_vs_b_delta": verifier_reward_outcome.get("a_vs_b_delta"),
        "a_vs_b_ci95": verifier_reward_outcome.get("a_vs_b_ci95"),
        "certified_corpus_gap_id": certified_arc_corpus.get("gap_id"),
        "certified_corpus_status": certified_arc_corpus.get("status"),
        "certified_corpus_artifact": EXP4200_PATH,
        "certified_corpus_size": certified_arc_corpus.get("certified_corpus_size"),
        "certification_precision": certified_arc_corpus.get("certification_precision", {}),
        "distill_lift_ci95": certified_arc_corpus.get("distill_lift_ci95", []),
        "distill_lift_latent_vs_absent": certified_arc_corpus.get("distill_lift_latent_vs_absent"),
        "eval_exp_4204": EXP4204_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V389_ROLE_ID] + [
        role
    ]


def _verifier_reward_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_REWARD_GAP_ID}: Exp 4204 .389 verifier-as-reward A-vs-B axis\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4199_PATH}` with Phase-0 `{EXP4197_PATH}`; "
        f"phase0_precision={outcome.get('phase0_precision')}; "
        f"youden_j={outcome.get('youden_j')}; "
        f"phase0_gate_clean={str(bool(outcome.get('phase0_gate_clean'))).lower()}; "
        f"training_launched={str(bool(outcome.get('training_launched'))).lower()}; "
        f"verifier_label_carries_signal="
        f"{str(bool(outcome.get('verifier_label_carries_signal'))).lower()}; "
        f"a_vs_b_delta={outcome.get('a_vs_b_delta')}; "
        f"a_vs_b_ci95={outcome.get('a_vs_b_ci95')}; "
        f"honest_verdict={outcome.get('honest_verdict')}.\n"
        "- failure mode: the clean Phase-0 operating point exists, but the decisive "
        "A-vs-B collection is blocked because the 3-arm training launch did not "
        "produce a live checkpoint; no reward-signal win is claimable.\n"
        "- missing discriminator: decision-grade evidence that verifier-certified "
        "labels beat same-generator random-label controls on held-out hidden tests.\n"
        "- candidate design: relaunch/resume the stable 3-arm training run, require "
        "gold-control and truncation guards, and promote only if the A-vs-B CI excludes "
        "zero.\n"
        "- priority: high\n"
    )


def _certified_corpus_gap_block(outcome: dict[str, Any]) -> str:
    precision = outcome.get("certification_precision", {})
    return (
        f"### {CERTIFIED_CORPUS_GAP_ID}: Exp 4204 certified ARC corpus distill-lift note\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4200_PATH}`; "
        f"certified_corpus_size={outcome.get('certified_corpus_size')}; "
        f"certification_precision.rate={precision.get('rate')}; "
        f"distill_lift_ci95={outcome.get('distill_lift_ci95')}; "
        f"invisible_leash_diagnosis={outcome.get('distill_lift_latent_vs_absent')}; "
        f"seeded_generation_status={outcome.get('seeded_generation_status')}; "
        f"flagged_adversarial={str(bool(outcome.get('flagged_adversarial'))).lower()}.\n"
        "- failure mode: the GAP-4 certified corpus exists and is high precision, but "
        "the cheap seeded-vs-cold local induction read is uninformative because the "
        "seeded checkpoint is missing; no latent distillation lift is established.\n"
        "- missing discriminator: measured seeded local generation or LoRA distillation "
        "lift from verifier-certified ARC programs.\n"
        "- candidate design: materialize the seeded checkpoint or run the bounded "
        "LoRA-distill follow-up before claiming the certified labels train a better "
        "local generator.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4204") == EXP4204_ARTIFACT_PATH
        and any(role.get("role_id") == V389_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    certified_arc_corpus: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4204 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    needed = {GAP_REWARD_GAP_ID, CERTIFIED_CORPUS_GAP_ID}
    gaps_complete = needed.issubset(set(gaps_updated))
    prefix = "complete:" if guard_ok and gaps_complete and registry_updated else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4204_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4204_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v389_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"reward_{verifier_reward_outcome['status']}_"
            f"certified_corpus_{certified_arc_corpus['status']}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "verifier_reward_outcome": verifier_reward_outcome,
        "certified_arc_corpus": certified_arc_corpus,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4197_PATH,
            EXP4199_PATH,
            EXP4200_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4204", "SCENARIO-VERIFY-4204"],
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    artifact = {
        "experiment": "experiment_4204_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4204_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "registry_updated": False,
        "verifier_reward_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_REWARD_GAP_ID,
            "verifier_label_carries_signal": False,
        },
        "certified_arc_corpus": {
            "status": "blocked_precondition",
            "gap_id": CERTIFIED_CORPUS_GAP_ID,
            "certified_corpus_size": 0,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "preconditions": preflight,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4197_PATH,
            EXP4199_PATH,
            EXP4200_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4204", "SCENARIO-VERIFY-4204"],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4204 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["regression_guard_passed"], bool):
        raise ValueError("regression_guard_passed must be a BARE bool")
    if not isinstance(artifact["registry_updated"], bool):
        raise ValueError("registry_updated must be a bare bool")
    if not isinstance(artifact["gaps_updated"], list):
        raise ValueError("gaps_updated must be a list")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4204 principles")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4204 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4204_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    verifier_reward_outcome = classify_verifier_reward_outcome(repo_root)
    certified_arc_corpus = classify_certified_arc_corpus(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        verifier_reward_outcome,
        certified_arc_corpus,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        verifier_reward_outcome=verifier_reward_outcome,
        certified_arc_corpus=certified_arc_corpus,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4204_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
