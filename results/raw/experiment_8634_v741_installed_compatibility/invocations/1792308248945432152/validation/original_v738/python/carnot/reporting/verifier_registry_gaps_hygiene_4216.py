"""Exp 4216 registry/gaps hygiene for .390 oracle-distinct outcomes.

Spec refs: REQ-VERIFY-4216, SCENARIO-VERIFY-4216.

This runner is a ledger reconciler. It protects the canonical GAP-4 replay
numbers from silent drift and records what the .390 artifacts actually say:
the oracle-distinct A3 gate is blocked, the detector has real AUROC signal, the
reward A-vs-B read has not evaluated yet, and the certified ARC corpus shows no
distill lift. No model, GPU, Codex, GGUF, or TRM training path is touched.
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

EXP4216_ARTIFACT_PATH = "results/experiment_4216_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = exp4153.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4153.ARC1_PROGRAMS_PATH
EXP4208_PATH = "results/experiment_4208_verifier_as_detector_auroc.json"
EXP4209_PATH = "results/experiment_4209_oracle_distinct_arc_verifier_build.json"
EXP4210_PATH = "results/experiment_4210_oracle_distinct_arc_verifier_beats_vote.json"
EXP4211_PATH = "results/experiment_4211_verifier_as_reward_finish_synchronous.json"
EXP4212_PATH = "results/experiment_4212_certified_arc_corpus_distill_lift.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
GAP_ORACLE_DISTINCT_GAP_ID = "GAP-ORACLE-DISTINCT"
DETECTOR_GAP_ID = "GAP-DETECTOR-AUROC-4208"
GAP_REWARD_GAP_ID = "GAP-REWARD"
CERTIFIED_CORPUS_GAP_ID = "GAP-REWARD-CERTIFIED-CORPUS-4212"
V390_ROLE_ID = "oracle_distinct_reward_corpus_hygiene_4216"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "oracle_distinct_outcome",
    "detector_aurocs",
    "verifier_reward_outcome",
    "certified_arc_corpus",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .390 truth.",
    "regression_guard_passed": (
        "BARE bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched (the GAP-ORACLE-DISTINCT frontier entry + "
        "the detector + verifier-as-reward + certified-corpus notes)."
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


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _check_json_resource(repo_root: Path, resource: str, rel_path: str) -> dict[str, Any]:
    path = repo_root / rel_path
    if not path.exists():
        return {"resource": resource, "available": False, "detail": f"missing: {rel_path}"}
    try:
        loaded = base._load_json(path)
    except Exception as exc:  # pragma: no cover - exact JSON exception type varies.
        return {"resource": resource, "available": False, "detail": f"parse_error: {exc}"}
    if not isinstance(loaded, dict):  # pragma: no cover - corrupted artifacts are defensive only.
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
    """REQ-VERIFY-4216: verify cached replay fixtures, upstream artifacts, and ledgers."""
    base_preflight = exp4153.check_preconditions(repo_root)
    checks = list(base_preflight["checks"]) + [
        _check_json_resource(repo_root, "exp4208_detector_auroc", EXP4208_PATH),
        _check_json_resource(repo_root, "exp4209_oracle_distinct_build", EXP4209_PATH),
        _check_json_resource(repo_root, "exp4210_oracle_distinct_a3", EXP4210_PATH),
        _check_json_resource(repo_root, "exp4211_verifier_reward_a_vs_b", EXP4211_PATH),
        _check_json_resource(repo_root, "exp4212_certified_arc_corpus", EXP4212_PATH),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4216: replay GAP-4 ARC-1 from cached artifacts only."""
    return exp4153.replay_gap4_arc1(repo_root)


def _oracle_distinct_status(outcome: dict[str, Any]) -> str:
    if outcome.get("oracle_distinct_beats_vote") is True:
        return "filled_oracle_distinct_beats_vote"
    if outcome.get("selector_trained") is False:
        return "open_a3_blocked_selector_not_trained"
    return "open_a3_not_decision_grade"


def classify_oracle_distinct_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4216: summarize Exp 4210 A3 with the paired Exp 4209 build gate."""
    a3 = base._load_json(repo_root / EXP4210_PATH)
    build = base._load_json(repo_root / EXP4209_PATH)
    verifier_is_oracle = a3.get("verifier_is_oracle")
    if not isinstance(verifier_is_oracle, bool):
        verifier_is_oracle = build.get("verifier_is_oracle") is True

    outcome = {
        "gap_id": GAP_ORACLE_DISTINCT_GAP_ID,
        "status": "",
        "artifact_path": EXP4210_PATH,
        "build_artifact_path": EXP4209_PATH,
        "source_artifacts": [EXP4210_PATH, EXP4209_PATH],
        "honest_verdict": str(a3.get("honest_verdict", "")),
        "build_honest_verdict": str(build.get("honest_verdict", "")),
        "oracle_distinct_beats_vote": a3.get("oracle_distinct_beats_vote") is True,
        "oracle_distinct_delta": _first_numeric(
            a3,
            "oracle_distinct_delta",
            "oracle_distinct_vs_vote_delta",
            "a3_delta_vs_vote",
        ),
        "oracle_distinct_ci95": _first_ci95(
            a3,
            "oracle_distinct_ci95",
            "oracle_distinct_delta_ci95",
            "a3_delta_ci95",
        ),
        "verifier_is_oracle": bool(verifier_is_oracle),
        "selector_trained": build.get("selector_trained") is True,
        "oracle_distinct_auroc": _numeric_or_none(build.get("oracle_distinct_auroc")) or 0.0,
        "oracle_distinct_auroc_ci95": _first_ci95(build, "oracle_distinct_auroc_ci95")
        or [0.0, 0.0],
        "learned_verifier_path": str(build.get("learned_verifier_path", "")),
        "gate_check_summary": str(a3.get("gate_check_summary", "")),
        "blocked_at_layer": str(a3.get("blocked_at_layer", "")),
        "status_from_artifact": str(a3.get("status", "")),
        "missing_discriminator": (
            "learned_verifier_that_beats_vote_where_execution_is_not_the_oracle"
        ),
    }
    outcome["status"] = _oracle_distinct_status(outcome)
    return outcome


def _detector_status(outcome: dict[str, Any]) -> str:
    beats = outcome.get("detection_beats_random_ci95_exclusive_by_domain", {})
    if isinstance(beats, dict) and beats and all(value is True for value in beats.values()):
        return "detector_auroc_recorded_all_domains_ci_exclusive"
    return "detector_auroc_recorded_with_open_domain_caveats"


def classify_detector_aurocs(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4216: summarize Exp 4208 detector AUROCs without making selector claims."""
    artifact = base._load_json(repo_root / EXP4208_PATH)
    outcome = {
        "gap_id": DETECTOR_GAP_ID,
        "status": "",
        "artifact_path": EXP4208_PATH,
        "source_artifacts": [EXP4208_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "detection_auroc_by_domain": dict(artifact.get("detection_auroc_by_domain", {})),
        "detection_auroc_ci95_by_domain": dict(artifact.get("detection_auroc_ci95_by_domain", {})),
        "detection_beats_random_ci95_exclusive_by_domain": dict(
            artifact.get("detection_beats_random_ci95_exclusive_by_domain", {})
        ),
        "n_by_domain": dict(artifact.get("n_by_domain", {})),
        "verifier_is_oracle_by_domain": dict(artifact.get("verifier_is_oracle_by_domain", {})),
        "selector_headroom_by_domain": dict(artifact.get("selector_headroom_by_domain", {})),
        "missing_discriminator": (
            "selection_policy_that_converts_detector_signal_into_vote_beating_ranker"
        ),
    }
    outcome["status"] = _detector_status(outcome)
    return outcome


def _reward_status(outcome: dict[str, Any]) -> str:
    if outcome.get("verifier_label_carries_signal") is True:
        return "filled_verifier_label_carries_signal"
    accumulated = outcome.get("accumulated_n", {})
    if isinstance(accumulated, dict) and accumulated.get("eval") == 0:
        return "open_accumulating_reward_no_eval_yet"
    if _numeric_or_none(outcome.get("a_vs_b_delta")) == 0.0:
        return "open_no_a_vs_b_signal"
    return "open_a_vs_b_not_decision_grade"


def classify_verifier_reward_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4216: summarize Exp 4211 A-vs-B without inventing held-out signal."""
    artifact = base._load_json(repo_root / EXP4211_PATH)
    outcome = {
        "gap_id": GAP_REWARD_GAP_ID,
        "status": "",
        "artifact_path": EXP4211_PATH,
        "source_artifacts": [EXP4211_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "verifier_label_carries_signal": artifact.get("verifier_label_carries_signal") is True,
        "a_vs_b_delta": _first_numeric(artifact, "a_vs_b_delta", "arm_a_vs_b_delta"),
        "a_vs_b_ci95": _first_ci95(artifact, "a_vs_b_ci95", "arm_a_vs_b_ci95"),
        "positive_control_confirmed": artifact.get("positive_control_confirmed") is True,
        "youden_j": _numeric_or_none(artifact.get("youden_j")),
        "accumulated_n": dict(artifact.get("accumulated_n", {})),
        "verifier_is_oracle": artifact.get("verifier_is_oracle") is True,
        "pass_at_1": dict(artifact.get("pass_at_1", {})),
        "truncation_guard": dict(artifact.get("truncation_guard", {})),
        "missing_discriminator": (
            "decision_grade_a_vs_b_training_signal_beyond_random_label_control"
        ),
    }
    outcome["status"] = _reward_status(outcome)
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
    """REQ-VERIFY-4216: summarize Exp 4212 corpus size, precision, and lift read."""
    artifact = base._load_json(repo_root / EXP4212_PATH)
    outcome = {
        "gap_id": CERTIFIED_CORPUS_GAP_ID,
        "status": "",
        "artifact_path": EXP4212_PATH,
        "source_artifacts": [EXP4212_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "certified_corpus_size": artifact.get("certified_corpus_size"),
        "certified_corpus_path": str(artifact.get("certified_corpus_path", "")),
        "certification_precision": dict(artifact.get("certification_precision", {})),
        "local_induction_cold": dict(artifact.get("local_induction_cold", {})),
        "local_induction_with_certified_exemplars": dict(
            artifact.get("local_induction_with_certified_exemplars", {})
        ),
        "distill_lift_delta": _numeric_or_none(artifact.get("distill_lift_delta")),
        "distill_lift_ci95": _first_ci95(artifact, "distill_lift_ci95") or [],
        "distill_lift_latent_vs_absent": str(artifact.get("invisible_leash_diagnosis", "")),
        "seeded_generation_status": str(artifact.get("seeded_generation_status", "")),
        "verifier_is_oracle": artifact.get("verifier_is_oracle") is True,
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
    oracle_distinct_outcome: dict[str, Any],
    detector_aurocs: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    certified_arc_corpus: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gap text with the .390 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(
        updated_registry,
        offline_replay,
        oracle_distinct_outcome,
        detector_aurocs,
        verifier_reward_outcome,
        certified_arc_corpus,
    )
    _ensure_v390_role(
        updated_registry,
        oracle_distinct_outcome,
        detector_aurocs,
        verifier_reward_outcome,
        certified_arc_corpus,
    )

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4216-oracle-distinct",
        _oracle_distinct_gap_block(oracle_distinct_outcome),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4216-detector-auroc",
        _detector_auroc_gap_block(detector_aurocs),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4216-gap-reward",
        _verifier_reward_gap_block(verifier_reward_outcome),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4216-certified-corpus",
        _certified_corpus_gap_block(certified_arc_corpus),
    )
    gap_ids = (
        GAP_ORACLE_DISTINCT_GAP_ID,
        DETECTOR_GAP_ID,
        GAP_REWARD_GAP_ID,
        CERTIFIED_CORPUS_GAP_ID,
    )
    touched = [gap_id for gap_id in gap_ids if gap_id in updated_gaps]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "oracle_distinct_recorded": GAP_ORACLE_DISTINCT_GAP_ID in touched,
            "detector_aurocs_recorded": DETECTOR_GAP_ID in touched,
            "verifier_reward_recorded": GAP_REWARD_GAP_ID in touched,
            "certified_corpus_recorded": CERTIFIED_CORPUS_GAP_ID in touched,
        },
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    detector_aurocs: dict[str, Any],
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
            "eval_exp_4216": EXP4216_ARTIFACT_PATH,
            "exp4216_regression_guard_passed": bool(offline_replay.get("regression_guard_passed")),
            "exp4216_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4216_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4216_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4216_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4216_oracle_distinct_beats_vote": bool(
                oracle_distinct_outcome.get("oracle_distinct_beats_vote")
            ),
            "exp4216_oracle_distinct_auroc": oracle_distinct_outcome.get("oracle_distinct_auroc"),
            "exp4216_oracle_distinct_selector_trained": bool(
                oracle_distinct_outcome.get("selector_trained")
            ),
            "exp4216_detector_arc_auroc": detector_aurocs.get("detection_auroc_by_domain", {}).get(
                "arc"
            ),
            "exp4216_verifier_label_carries_signal": bool(
                verifier_reward_outcome.get("verifier_label_carries_signal")
            ),
            "exp4216_a_vs_b_delta": verifier_reward_outcome.get("a_vs_b_delta"),
            "exp4216_certified_corpus_size": certified_arc_corpus.get("certified_corpus_size"),
            "exp4216_distill_lift_latent_vs_absent": certified_arc_corpus.get(
                "distill_lift_latent_vs_absent"
            ),
        }
    )


def _ensure_v390_role(
    registry: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    detector_aurocs: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    certified_arc_corpus: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - guarded by _ensure_gap4_eval.
        return
    role = {
        "role_id": V390_ROLE_ID,
        "experiment": EXP4216_ARTIFACT_PATH,
        "role": "registry_gap_ledger_hygiene_v390",
        "status": "oracle_distinct_detector_reward_certified_corpus_recorded",
        "oracle_distinct_gap_id": oracle_distinct_outcome.get("gap_id"),
        "oracle_distinct_status": oracle_distinct_outcome.get("status"),
        "oracle_distinct_artifact": EXP4210_PATH,
        "oracle_distinct_build_artifact": EXP4209_PATH,
        "oracle_distinct_beats_vote": bool(
            oracle_distinct_outcome.get("oracle_distinct_beats_vote")
        ),
        "oracle_distinct_delta": oracle_distinct_outcome.get("oracle_distinct_delta"),
        "oracle_distinct_ci95": oracle_distinct_outcome.get("oracle_distinct_ci95"),
        "oracle_distinct_auroc": oracle_distinct_outcome.get("oracle_distinct_auroc"),
        "oracle_distinct_auroc_ci95": oracle_distinct_outcome.get("oracle_distinct_auroc_ci95"),
        "selector_trained": bool(oracle_distinct_outcome.get("selector_trained")),
        "verifier_is_oracle": bool(oracle_distinct_outcome.get("verifier_is_oracle")),
        "detector_gap_id": detector_aurocs.get("gap_id"),
        "detector_status": detector_aurocs.get("status"),
        "detector_auroc_by_domain": detector_aurocs.get("detection_auroc_by_domain", {}),
        "detector_ci95_by_domain": detector_aurocs.get("detection_auroc_ci95_by_domain", {}),
        "verifier_reward_gap_id": verifier_reward_outcome.get("gap_id"),
        "verifier_reward_status": verifier_reward_outcome.get("status"),
        "verifier_reward_artifact": EXP4211_PATH,
        "verifier_label_carries_signal": bool(
            verifier_reward_outcome.get("verifier_label_carries_signal")
        ),
        "a_vs_b_delta": verifier_reward_outcome.get("a_vs_b_delta"),
        "a_vs_b_ci95": verifier_reward_outcome.get("a_vs_b_ci95"),
        "youden_j": verifier_reward_outcome.get("youden_j"),
        "reward_verifier_is_oracle": bool(verifier_reward_outcome.get("verifier_is_oracle")),
        "certified_corpus_gap_id": certified_arc_corpus.get("gap_id"),
        "certified_corpus_status": certified_arc_corpus.get("status"),
        "certified_corpus_artifact": EXP4212_PATH,
        "certified_corpus_size": certified_arc_corpus.get("certified_corpus_size"),
        "certification_precision": certified_arc_corpus.get("certification_precision", {}),
        "distill_lift_delta": certified_arc_corpus.get("distill_lift_delta"),
        "distill_lift_ci95": certified_arc_corpus.get("distill_lift_ci95", []),
        "distill_lift_latent_vs_absent": certified_arc_corpus.get("distill_lift_latent_vs_absent"),
        "corpus_verifier_is_oracle": bool(certified_arc_corpus.get("verifier_is_oracle")),
        "gap_moat_update": "unchanged_a3_blocked_no_oracle_distinct_win",
        "eval_exp_4216": EXP4216_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V390_ROLE_ID] + [
        role
    ]


def _oracle_distinct_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_ORACLE_DISTINCT_GAP_ID}: Exp 4216 .390 oracle-distinct A3 frontier\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4210_PATH}` with build `{EXP4209_PATH}`; "
        f"oracle_distinct_beats_vote={_bool_text(outcome.get('oracle_distinct_beats_vote'))}; "
        f"oracle_distinct_delta={outcome.get('oracle_distinct_delta')}; "
        f"oracle_distinct_ci95={outcome.get('oracle_distinct_ci95')}; "
        f"verifier_is_oracle={_bool_text(outcome.get('verifier_is_oracle'))}; "
        f"selector_trained={_bool_text(outcome.get('selector_trained'))}; "
        f"oracle_distinct_auroc={outcome.get('oracle_distinct_auroc')}; "
        f"oracle_distinct_auroc_ci95={outcome.get('oracle_distinct_auroc_ci95')}; "
        f"honest_verdict={outcome.get('honest_verdict')}. GAP-MOAT unchanged.\n"
        "- failure mode: A3 did not execute because the learned oracle-distinct ARC selector "
        "was not trained from labeled candidates; no vote-beating off-oracle verifier result "
        "is claimable.\n"
        "- missing discriminator: a learned verifier that beats vote where execution is not "
        "the oracle.\n"
        "- candidate design: materialize labeled per-candidate ARC rows, train the "
        "oracle-distinct selector out of fold, then rerun A3 only if selector_trained=true.\n"
        "- priority: high\n"
    )


def _detector_auroc_gap_block(outcome: dict[str, Any]) -> str:
    aurocs = outcome.get("detection_auroc_by_domain", {})
    ci95 = outcome.get("detection_auroc_ci95_by_domain", {})
    oracle = outcome.get("verifier_is_oracle_by_domain", {})
    return (
        f"### {DETECTOR_GAP_ID}: Exp 4216 detector AUROC status note\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4208_PATH}`; "
        f"sudoku={aurocs.get('sudoku')}; code={aurocs.get('code')}; "
        f"math={aurocs.get('math')}; arc={aurocs.get('arc')}; "
        f"ci95_by_domain={ci95}; verifier_is_oracle_by_domain={oracle}.\n"
        "- failure mode: detector AUROC says the verifier can separate good from bad "
        "candidates, but it does not by itself prove a selector beats vote.\n"
        "- missing discriminator: a selection policy that converts detector signal into a "
        "vote-beating ranker on a headroom-present pool.\n"
        "- candidate design: use the detector as a training or calibration source for the "
        "oracle-distinct selector, then measure selection lift with bootstrap CI.\n"
        "- priority: medium\n"
    )


def _verifier_reward_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_REWARD_GAP_ID}: Exp 4216 .390 verifier-as-reward A-vs-B axis\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4211_PATH}`; "
        f"verifier_label_carries_signal="
        f"{_bool_text(outcome.get('verifier_label_carries_signal'))}; "
        f"a_vs_b_delta={outcome.get('a_vs_b_delta')}; "
        f"a_vs_b_ci95={outcome.get('a_vs_b_ci95')}; "
        f"youden_j={outcome.get('youden_j')}; "
        f"positive_control_confirmed={_bool_text(outcome.get('positive_control_confirmed'))}; "
        f"accumulated_n={outcome.get('accumulated_n')}; "
        f"verifier_is_oracle={_bool_text(outcome.get('verifier_is_oracle'))}; "
        f"honest_verdict={outcome.get('honest_verdict')}.\n"
        "- failure mode: the synchronous reward run has not produced held-out A-vs-B eval "
        "rows yet, so the verifier-label reward signal remains unproven.\n"
        "- missing discriminator: decision-grade evidence that verifier-certified labels beat "
        "same-generator random-label controls on held-out hidden tests.\n"
        "- candidate design: continue the accumulate/resume path until eval rows exist, then "
        "promote only if the A-vs-B CI excludes zero with a valid positive control.\n"
        "- priority: high\n"
    )


def _certified_corpus_gap_block(outcome: dict[str, Any]) -> str:
    precision = outcome.get("certification_precision", {})
    return (
        f"### {CERTIFIED_CORPUS_GAP_ID}: Exp 4216 certified ARC corpus distill-lift note\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4212_PATH}`; "
        f"certified_corpus_size={outcome.get('certified_corpus_size')}; "
        f"certification_precision.rate={precision.get('rate')}; "
        f"distill_lift_delta={outcome.get('distill_lift_delta')}; "
        f"distill_lift_ci95={outcome.get('distill_lift_ci95')}; "
        f"invisible_leash_diagnosis={outcome.get('distill_lift_latent_vs_absent')}; "
        f"seeded_generation_status={outcome.get('seeded_generation_status')}; "
        f"verifier_is_oracle={_bool_text(outcome.get('verifier_is_oracle'))}; "
        f"flagged_adversarial={_bool_text(outcome.get('flagged_adversarial'))}.\n"
        "- failure mode: the certified corpus is high precision, but the seeded-vs-cold read "
        "is flat and flagged, so no local distillation lift is established.\n"
        "- missing discriminator: measured seeded local generation or LoRA distillation lift "
        "from verifier-certified ARC programs.\n"
        "- candidate design: materialize a real seeded checkpoint or LoRA-distill follow-up "
        "before claiming certified labels improve a local generator.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4216") == EXP4216_ARTIFACT_PATH
        and any(role.get("role_id") == V390_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    detector_aurocs: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    certified_arc_corpus: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4216 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    needed = {
        GAP_ORACLE_DISTINCT_GAP_ID,
        DETECTOR_GAP_ID,
        GAP_REWARD_GAP_ID,
        CERTIFIED_CORPUS_GAP_ID,
    }
    gaps_complete = needed.issubset(set(gaps_updated))
    prefix = "complete:" if guard_ok and gaps_complete and registry_updated else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4216_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4216_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v390_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"oracle_distinct_{oracle_distinct_outcome['status']}_"
            f"reward_{verifier_reward_outcome['status']}_"
            f"certified_corpus_{certified_arc_corpus['status']}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "oracle_distinct_outcome": oracle_distinct_outcome,
        "detector_aurocs": detector_aurocs,
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
            EXP4208_PATH,
            EXP4209_PATH,
            EXP4210_PATH,
            EXP4211_PATH,
            EXP4212_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4216", "SCENARIO-VERIFY-4216"],
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    artifact = {
        "experiment": "experiment_4216_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4216_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "registry_updated": False,
        "oracle_distinct_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_ORACLE_DISTINCT_GAP_ID,
            "oracle_distinct_beats_vote": False,
            "verifier_is_oracle": False,
        },
        "detector_aurocs": {"status": "blocked_precondition", "gap_id": DETECTOR_GAP_ID},
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
            EXP4208_PATH,
            EXP4209_PATH,
            EXP4210_PATH,
            EXP4211_PATH,
            EXP4212_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4216", "SCENARIO-VERIFY-4216"],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4216 fields before writing the artifact."""
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
        raise ValueError("field_principles must match the required Exp 4216 principles")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4216 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4216_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    oracle_distinct_outcome = classify_oracle_distinct_outcome(repo_root)
    detector_aurocs = classify_detector_aurocs(repo_root)
    verifier_reward_outcome = classify_verifier_reward_outcome(repo_root)
    certified_arc_corpus = classify_certified_arc_corpus(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        oracle_distinct_outcome,
        detector_aurocs,
        verifier_reward_outcome,
        certified_arc_corpus,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        oracle_distinct_outcome=oracle_distinct_outcome,
        detector_aurocs=detector_aurocs,
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
    print(f"Wrote {REPO_ROOT / EXP4216_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
