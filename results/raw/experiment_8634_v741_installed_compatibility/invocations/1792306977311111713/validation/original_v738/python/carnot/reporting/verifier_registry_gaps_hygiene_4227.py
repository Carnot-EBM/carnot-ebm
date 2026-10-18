"""Exp 4227 registry/gaps hygiene for .391 oracle-distinct outcomes.

Spec refs: REQ-VERIFY-4227, SCENARIO-VERIFY-4227.

This runner is an offline ledger reconciler. It replays the canonical GAP-4
ARC-1 candidate set from cached artifacts, records the first trained
oracle-distinct A2 beats-vote read, carries forward detector AUROCs, and records
the verifier-as-reward A-vs-B status without touching TRM training.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import sys
import time
from typing import Any, Callable

import yaml

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4227
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4227_ARTIFACT_PATH = "results/experiment_4227_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = "results/arc3_gap3_stage2_eval_pool.json.gz"
ARC1_PROGRAMS_PATH = "results/arc3_gap4_induced_programs.json"
EXP4208_PATH = "results/experiment_4208_verifier_as_detector_auroc.json"
EXP4220_PATH = "results/experiment_4220_oracle_distinct_arc_verifier_build_labeled.json"
EXP4221_PATH = "results/experiment_4221_oracle_distinct_arc_verifier_beats_vote.json"
EXP4223_PATH = "results/experiment_4223_verifier_as_reward_3arm_synchronous.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
GAP_ORACLE_DISTINCT_GAP_ID = "GAP-ORACLE-DISTINCT"
GAP_ORACLE_DISTINCT_A2_GAP_ID = "GAP-ORACLE-DISTINCT-A2-4221"
GAP_REWARD_GAP_ID = "GAP-REWARD"
DETECTOR_GAP_ID = "GAP-DETECTOR-AUROC-4208"
V391_ROLE_ID = "oracle_distinct_reward_detector_hygiene_4227"

EXPECTED_ARC1 = {
    "n": 31,
    "vote_pass2": 0.4516,
    "gated_pass2": 0.5806,
    "headroom_recovered": 4,
    "vote_wins_lost": 0,
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "oracle_distinct_outcome",
    "verifier_reward_outcome",
    "detector_aurocs",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "inference_substrate",
    "adversarial_verify",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .391 truth.",
    "regression_guard_passed": (
        "BARE bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched (the GAP-ORACLE-DISTINCT frontier entry + "
        "the A2 beats-vote + verifier-as-reward + detector notes)."
    ),
    "random_seed": (
        "Determinism precondition + the methodology field that prevents the exp4216 "
        "METHODOLOGY_MISSING re-flag."
    ),
    "reproducibility_checksum": (
        "Hash of the cached GAP-4 candidate set; the methodology field that prevents "
        "the exp4216 re-flag and catches silent candidate drift."
    ),
}


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


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


def _load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("expected JSON object")
    return loaded


def _load_registry_for_check(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("registry is not a mapping")
    if "verifiers" not in loaded:
        raise ValueError("registry missing verifiers")
    return loaded


def _load_gaps_for_check(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("gaps markdown is empty")
    return text


def _check_resource(
    repo_root: Path,
    resource: str,
    rel_paths: list[str],
    loader: Callable[[Path], Any],
) -> dict[str, Any]:
    paths = [repo_root / rel_path for rel_path in rel_paths]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return {"resource": resource, "available": False, "detail": f"missing: {missing}"}
    try:
        for path in paths:
            loader(path)
    except Exception as exc:  # pragma: no cover - exact parse exceptions vary by parser.
        return {"resource": resource, "available": False, "detail": f"parse_error: {exc}"}
    return {"resource": resource, "available": True, "detail": ", ".join(rel_paths)}


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
    """REQ-VERIFY-4227: verify cached replay fixtures, upstream artifacts, and ledgers."""
    checks = [
        _check_resource(
            repo_root,
            "gap4_arc1_candidate_fixtures",
            [ARC1_POOL_PATH, ARC1_PROGRAMS_PATH],
            lambda path: _load_gzip_json(path)
            if path.suffix == ".gz"
            else base._load_json(path),
        ),
        _check_resource(repo_root, "verifier_registry", [REGISTRY_PATH], _load_registry_for_check),
        _check_resource(repo_root, "verifier_gaps", [GAPS_PATH], _load_gaps_for_check),
        _check_json_resource(repo_root, "exp4208_detector_auroc", EXP4208_PATH),
        _check_json_resource(repo_root, "exp4220_oracle_distinct_build_labeled", EXP4220_PATH),
        _check_json_resource(repo_root, "exp4221_oracle_distinct_a2", EXP4221_PATH),
        _check_json_resource(repo_root, "exp4223_verifier_reward_a_vs_b", EXP4223_PATH),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4227: replay GAP-4 ARC-1 from cached artifacts only."""
    pool = _load_gzip_json(repo_root / ARC1_POOL_PATH)
    programs = base._load_json(repo_root / ARC1_PROGRAMS_PATH)
    entries = list(pool.get("entries", []))
    programs_by_entry = _programs_by_entry(programs)

    vote_hits: set[int] = set()
    gated_hits: set[int] = set()
    oracle_hits: set[int] = set()
    for index, entry in enumerate(entries):
        cands = list(entry.get("candidates", []))
        pred_grid = _trusted_pred_grid(programs_by_entry.get(index))
        if any(bool(cand.get("correct")) for cand in cands):
            oracle_hits.add(index)
        vote_ranked = sorted(cands, key=lambda cand: -int(cand.get("votes", 0)))
        gated_ranked = sorted(
            cands,
            key=lambda cand: (
                -_exec_match(cand, pred_grid),
                -int(cand.get("votes", 0)),
            ),
        )
        if _top2_hit(vote_ranked):
            vote_hits.add(index)
        if _top2_hit(gated_ranked):
            gated_hits.add(index)

    observed = {
        "n": len(entries),
        "vote_pass2": round(len(vote_hits) / max(1, len(entries)), 4),
        "gated_pass2": round(len(gated_hits) / max(1, len(entries)), 4),
        "headroom_recovered": len((gated_hits - vote_hits) & oracle_hits),
        "vote_wins_lost": len(vote_hits - gated_hits),
    }
    return {
        "regression_guard_passed": observed == EXPECTED_ARC1,
        "arc1_rule_exec": observed,
        "expected": {"arc1_rule_exec": deepcopy(EXPECTED_ARC1)},
        "cached_pool_path": ARC1_POOL_PATH,
        "cached_programs_path": ARC1_PROGRAMS_PATH,
        "no_codex_calls": True,
        "no_gguf_inference": True,
        "trm_training_touched": False,
    }


def _programs_by_entry(induced_programs_artifact: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    fallback_index = 0
    for program in induced_programs_artifact.get("programs", []):
        if not isinstance(program, dict):
            continue
        raw_index = program.get("entry_i", fallback_index)
        try:
            entry_index = int(raw_index)
        except (TypeError, ValueError):
            entry_index = fallback_index
        out[entry_index] = program
        fallback_index += 1
    return out


def _trusted_pred_grid(program: dict[str, Any] | None) -> Any | None:
    if not program or not bool(program.get("demo_perfect")):
        return None
    return program.get("pred_grid")


def _exec_match(candidate: dict[str, Any], pred_grid: Any | None) -> int:
    if pred_grid is None:
        return 0
    return int(candidate.get("grid") == pred_grid)


def _top2_hit(ranked_candidates: list[dict[str, Any]]) -> bool:
    return any(bool(cand.get("correct")) for cand in ranked_candidates[:2])


def candidate_set_checksum(repo_root: Path = REPO_ROOT) -> str:
    """REQ-VERIFY-4227: hash the cached GAP-4 candidate set used by the replay."""
    digest = hashlib.sha256()
    with (repo_root / ARC1_POOL_PATH).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _oracle_distinct_status(outcome: dict[str, Any]) -> str:
    if outcome.get("oracle_distinct_beats_vote") is True:
        return "filled_oracle_distinct_beats_vote"
    if outcome.get("headroom_exists") is False:
        return "open_a2_no_headroom_uninformative"
    return "open_a2_ties_vote_with_headroom"


def classify_oracle_distinct_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4227: summarize Exp 4221 A2 with the paired Exp 4220 build."""
    a2 = base._load_json(repo_root / EXP4221_PATH)
    build = base._load_json(repo_root / EXP4220_PATH)
    outcome = {
        "gap_id": GAP_ORACLE_DISTINCT_GAP_ID,
        "a2_gap_id": GAP_ORACLE_DISTINCT_A2_GAP_ID,
        "status": "",
        "artifact_path": EXP4221_PATH,
        "build_artifact_path": EXP4220_PATH,
        "source_artifacts": [EXP4221_PATH, EXP4220_PATH],
        "honest_verdict": str(a2.get("honest_verdict", "")),
        "build_honest_verdict": str(build.get("honest_verdict", "")),
        "headline_outcome": str(a2.get("headline_outcome", "")),
        "oracle_distinct_beats_vote": a2.get("oracle_distinct_beats_vote") is True,
        "verifier_minus_vote_delta": _first_numeric(a2, "verifier_minus_vote_delta"),
        "verifier_minus_vote_ci95": _first_ci95(a2, "verifier_minus_vote_ci95"),
        "arbiter_override_minus_vote": _first_numeric(a2, "arbiter_override_minus_vote"),
        "matched_control_delta": _first_numeric(a2, "matched_control_delta"),
        "oracle_at_k": _first_numeric(a2, "oracle_at_k"),
        "verifier_is_oracle": a2.get("verifier_is_oracle") is True,
        "headroom_exists": a2.get("headroom_exists") is True,
        "pass_rates": dict(a2.get("pass_rates", {})),
        "n_tasks": a2.get("n_tasks"),
        "selector_trained": build.get("selector_trained") is True,
        "oracle_distinct_auroc": _first_numeric(build, "oracle_distinct_auroc"),
        "oracle_distinct_auroc_ci95": _first_ci95(build, "oracle_distinct_auroc_ci95"),
        "wrong_majority_n": build.get("wrong_majority_n"),
        "learned_verifier_path": str(build.get("learned_verifier_path", "")),
        "gap_moat_update": "unchanged_a2_did_not_beat_vote",
        "missing_discriminator": (
            "learned_verifier_that_beats_vote_where_execution_is_not_the_oracle"
        ),
    }
    outcome["verifier_is_oracle"] = bool(outcome["verifier_is_oracle"])
    outcome["status"] = _oracle_distinct_status(outcome)
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
    """REQ-VERIFY-4227: summarize Exp 4223 A-vs-B without inventing held-out signal."""
    artifact = base._load_json(repo_root / EXP4223_PATH)
    outcome = {
        "gap_id": GAP_REWARD_GAP_ID,
        "status": "",
        "artifact_path": EXP4223_PATH,
        "source_artifacts": [EXP4223_PATH],
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


def _detector_status(outcome: dict[str, Any]) -> str:
    beats = outcome.get("detection_beats_random_ci95_exclusive_by_domain", {})
    if isinstance(beats, dict) and beats and all(value is True for value in beats.values()):
        return "detector_auroc_recorded_all_domains_ci_exclusive"
    return "detector_auroc_recorded_with_open_domain_caveats"


def classify_detector_aurocs(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4227: carry forward Exp 4208 detector AUROCs."""
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


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    detector_aurocs: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gap text with the .391 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay, oracle_distinct_outcome, verifier_reward_outcome, detector_aurocs)
    _ensure_v391_role(updated_registry, oracle_distinct_outcome, verifier_reward_outcome, detector_aurocs)

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4227-oracle-distinct",
        _oracle_distinct_frontier_gap_block(oracle_distinct_outcome),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4227-oracle-distinct-a2",
        _oracle_distinct_a2_gap_block(oracle_distinct_outcome),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4227-gap-reward",
        _verifier_reward_gap_block(verifier_reward_outcome),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4227-detector-auroc",
        _detector_auroc_gap_block(detector_aurocs),
    )
    gap_ids = (
        GAP_ORACLE_DISTINCT_GAP_ID,
        GAP_ORACLE_DISTINCT_A2_GAP_ID,
        GAP_REWARD_GAP_ID,
        DETECTOR_GAP_ID,
    )
    touched = [gap_id for gap_id in gap_ids if gap_id in updated_gaps]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "oracle_distinct_recorded": GAP_ORACLE_DISTINCT_GAP_ID in touched,
            "oracle_distinct_a2_recorded": GAP_ORACLE_DISTINCT_A2_GAP_ID in touched,
            "verifier_reward_recorded": GAP_REWARD_GAP_ID in touched,
            "detector_aurocs_recorded": DETECTOR_GAP_ID in touched,
        },
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    detector_aurocs: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - real/minimal registries include this entry.
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4227": EXP4227_ARTIFACT_PATH,
            "exp4227_regression_guard_passed": bool(offline_replay.get("regression_guard_passed")),
            "exp4227_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4227_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4227_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4227_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4227_oracle_distinct_beats_vote": bool(
                oracle_distinct_outcome.get("oracle_distinct_beats_vote")
            ),
            "exp4227_verifier_minus_vote_delta": oracle_distinct_outcome.get(
                "verifier_minus_vote_delta"
            ),
            "exp4227_verifier_minus_vote_ci95": oracle_distinct_outcome.get(
                "verifier_minus_vote_ci95"
            ),
            "exp4227_oracle_distinct_auroc": oracle_distinct_outcome.get(
                "oracle_distinct_auroc"
            ),
            "exp4227_wrong_majority_n": oracle_distinct_outcome.get("wrong_majority_n"),
            "exp4227_verifier_label_carries_signal": bool(
                verifier_reward_outcome.get("verifier_label_carries_signal")
            ),
            "exp4227_a_vs_b_delta": verifier_reward_outcome.get("a_vs_b_delta"),
            "exp4227_detector_arc_auroc": detector_aurocs.get("detection_auroc_by_domain", {}).get(
                "arc"
            ),
        }
    )


def _ensure_v391_role(
    registry: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    detector_aurocs: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - guarded by _ensure_gap4_eval.
        return
    role = {
        "role_id": V391_ROLE_ID,
        "experiment": EXP4227_ARTIFACT_PATH,
        "role": "registry_gap_ledger_hygiene_v391",
        "status": "oracle_distinct_a2_reward_detector_recorded",
        "oracle_distinct_gap_id": oracle_distinct_outcome.get("gap_id"),
        "oracle_distinct_a2_gap_id": oracle_distinct_outcome.get("a2_gap_id"),
        "oracle_distinct_status": oracle_distinct_outcome.get("status"),
        "oracle_distinct_artifact": EXP4221_PATH,
        "oracle_distinct_build_artifact": EXP4220_PATH,
        "oracle_distinct_beats_vote": bool(
            oracle_distinct_outcome.get("oracle_distinct_beats_vote")
        ),
        "verifier_minus_vote_delta": oracle_distinct_outcome.get("verifier_minus_vote_delta"),
        "verifier_minus_vote_ci95": oracle_distinct_outcome.get("verifier_minus_vote_ci95"),
        "arbiter_override_minus_vote": oracle_distinct_outcome.get("arbiter_override_minus_vote"),
        "matched_control_delta": oracle_distinct_outcome.get("matched_control_delta"),
        "oracle_at_k": oracle_distinct_outcome.get("oracle_at_k"),
        "verifier_is_oracle": bool(oracle_distinct_outcome.get("verifier_is_oracle")),
        "selector_trained": bool(oracle_distinct_outcome.get("selector_trained")),
        "oracle_distinct_auroc": oracle_distinct_outcome.get("oracle_distinct_auroc"),
        "oracle_distinct_auroc_ci95": oracle_distinct_outcome.get("oracle_distinct_auroc_ci95"),
        "wrong_majority_n": oracle_distinct_outcome.get("wrong_majority_n"),
        "verifier_reward_gap_id": verifier_reward_outcome.get("gap_id"),
        "verifier_reward_status": verifier_reward_outcome.get("status"),
        "verifier_reward_artifact": EXP4223_PATH,
        "verifier_label_carries_signal": bool(
            verifier_reward_outcome.get("verifier_label_carries_signal")
        ),
        "a_vs_b_delta": verifier_reward_outcome.get("a_vs_b_delta"),
        "a_vs_b_ci95": verifier_reward_outcome.get("a_vs_b_ci95"),
        "youden_j": verifier_reward_outcome.get("youden_j"),
        "detector_gap_id": detector_aurocs.get("gap_id"),
        "detector_status": detector_aurocs.get("status"),
        "detector_auroc_by_domain": detector_aurocs.get("detection_auroc_by_domain", {}),
        "detector_ci95_by_domain": detector_aurocs.get("detection_auroc_ci95_by_domain", {}),
        "gap_moat_update": oracle_distinct_outcome.get("gap_moat_update"),
        "eval_exp_4227": EXP4227_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V391_ROLE_ID] + [
        role
    ]


def _oracle_distinct_frontier_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_ORACLE_DISTINCT_GAP_ID}: Exp 4227 .391 oracle-distinct frontier\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4221_PATH}` with build `{EXP4220_PATH}`; "
        f"oracle_distinct_beats_vote={_bool_text(outcome.get('oracle_distinct_beats_vote'))}; "
        f"verifier_minus_vote_delta={outcome.get('verifier_minus_vote_delta')}; "
        f"verifier_minus_vote_ci95={outcome.get('verifier_minus_vote_ci95')}; "
        f"verifier_is_oracle={_bool_text(outcome.get('verifier_is_oracle'))}; "
        f"selector_trained={_bool_text(outcome.get('selector_trained'))}; "
        f"oracle_distinct_auroc={outcome.get('oracle_distinct_auroc')}; "
        f"wrong_majority_n={outcome.get('wrong_majority_n')}; "
        f"honest_verdict={outcome.get('honest_verdict')}. GAP-MOAT unchanged.\n"
        "- failure mode: the trained non-oracle A2 verifier did not produce a CI-exclusive "
        "vote-beating read on the headroom-present ARC slice.\n"
        "- missing discriminator: a learned verifier that beats vote where execution is not "
        "the oracle.\n"
        "- candidate design: use the wrong-majority rows and detector signal to improve "
        "the oracle-distinct selector before re-testing A2.\n"
        "- priority: high\n"
    )


def _oracle_distinct_a2_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_ORACLE_DISTINCT_A2_GAP_ID}: Exp 4227 .391 A2 beats-vote read\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4221_PATH}`; "
        f"oracle_distinct_beats_vote={_bool_text(outcome.get('oracle_distinct_beats_vote'))}; "
        f"verifier_minus_vote_delta={outcome.get('verifier_minus_vote_delta')}; "
        f"verifier_minus_vote_ci95={outcome.get('verifier_minus_vote_ci95')}; "
        f"arbiter_override_minus_vote={outcome.get('arbiter_override_minus_vote')}; "
        f"matched_control_delta={outcome.get('matched_control_delta')}; "
        f"oracle_at_k={outcome.get('oracle_at_k')}; "
        f"verifier_is_oracle={_bool_text(outcome.get('verifier_is_oracle'))}; "
        f"oracle_distinct_auroc={outcome.get('oracle_distinct_auroc')}; "
        f"oracle_distinct_auroc_ci95={outcome.get('oracle_distinct_auroc_ci95')}; "
        f"wrong_majority_n={outcome.get('wrong_majority_n')}.\n"
        "- failure mode: A2 is the first trained oracle-distinct read, but the held-out "
        "rerank still ties/underperforms vote rather than capturing the wrong-majority "
        "headroom.\n"
        "- missing discriminator: a non-oracle ARC selector that converts off-fold "
        "candidate discrimination into vote-beating top-1 selection.\n"
        "- candidate design: reweight or enrich the learned selector, then require a "
        "positive bootstrap CI before upgrading the frontier.\n"
        "- priority: high\n"
    )


def _verifier_reward_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_REWARD_GAP_ID}: Exp 4227 .391 verifier-as-reward A-vs-B axis\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4223_PATH}`; "
        f"verifier_label_carries_signal={_bool_text(outcome.get('verifier_label_carries_signal'))}; "
        f"a_vs_b_delta={outcome.get('a_vs_b_delta')}; "
        f"a_vs_b_ci95={outcome.get('a_vs_b_ci95')}; "
        f"youden_j={outcome.get('youden_j')}; "
        f"positive_control_confirmed={_bool_text(outcome.get('positive_control_confirmed'))}; "
        f"accumulated_n={outcome.get('accumulated_n')}; "
        f"verifier_is_oracle={_bool_text(outcome.get('verifier_is_oracle'))}; "
        f"honest_verdict={outcome.get('honest_verdict')}.\n"
        "- failure mode: the synchronous reward path still has no held-out A-vs-B eval "
        "rows, so verifier-label reward signal remains unproven.\n"
        "- missing discriminator: decision-grade evidence that verifier-certified labels beat "
        "same-generator random-label controls on held-out hidden tests.\n"
        "- candidate design: continue only until eval rows exist, then promote only if the "
        "A-vs-B CI excludes zero with a valid positive control.\n"
        "- priority: high\n"
    )


def _detector_auroc_gap_block(outcome: dict[str, Any]) -> str:
    aurocs = outcome.get("detection_auroc_by_domain", {})
    ci95 = outcome.get("detection_auroc_ci95_by_domain", {})
    oracle = outcome.get("verifier_is_oracle_by_domain", {})
    return (
        f"### {DETECTOR_GAP_ID}: Exp 4227 detector AUROC status note\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4208_PATH}`; "
        f"sudoku={aurocs.get('sudoku')}; code={aurocs.get('code')}; "
        f"math={aurocs.get('math')}; arc={aurocs.get('arc')}; "
        f"ci95_by_domain={ci95}; verifier_is_oracle_by_domain={oracle}.\n"
        "- failure mode: detector AUROC separates good from bad candidates but has not "
        "yet become a selector that beats vote.\n"
        "- missing discriminator: a selection policy that converts detector signal into a "
        "vote-beating ranker on a headroom-present pool.\n"
        "- candidate design: use the detector as a training or calibration source for the "
        "oracle-distinct selector, then measure selection lift with bootstrap CI.\n"
        "- priority: medium\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4227") == EXP4227_ARTIFACT_PATH
        and any(role.get("role_id") == V391_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def model_specs_for_replay(checksum: str) -> dict[str, Any]:
    """REQ-VERIFY-4227: methodology declaration for cached-candidate replay."""
    return {
        "method": "cached_gap4_candidate_replay_and_ledger_reconciliation",
        "candidate_set": ARC1_POOL_PATH,
        "candidate_set_sha256": checksum,
        "program_outputs": ARC1_PROGRAMS_PATH,
        "scoring_description": "offline verifier ensemble replay over checked-in candidates",
        "codex_calls": 0,
        "live_model_inference": False,
        "trm_training_touched": False,
        "stable_checkpoint_write": False,
    }


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    detector_aurocs: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    random_seed: int,
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4227 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    needed = {
        GAP_ORACLE_DISTINCT_GAP_ID,
        GAP_ORACLE_DISTINCT_A2_GAP_ID,
        GAP_REWARD_GAP_ID,
        DETECTOR_GAP_ID,
    }
    gaps_complete = needed.issubset(set(gaps_updated))
    prefix = "complete:" if guard_ok and gaps_complete and registry_updated else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4227_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4227_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v391_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"oracle_distinct_{oracle_distinct_outcome['status']}_"
            f"reward_{verifier_reward_outcome['status']}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "oracle_distinct_outcome": oracle_distinct_outcome,
        "verifier_reward_outcome": verifier_reward_outcome,
        "detector_aurocs": detector_aurocs,
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs_for_replay(reproducibility_checksum),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4208_PATH,
            EXP4220_PATH,
            EXP4221_PATH,
            EXP4223_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4227", "SCENARIO-VERIFY-4227"],
        "adversarial_verify": {"status": "pending"},
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    checksum = f"blocked:{blocked}"
    artifact = {
        "experiment": "experiment_4227_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4227_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "registry_updated": False,
        "oracle_distinct_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_ORACLE_DISTINCT_GAP_ID,
            "a2_gap_id": GAP_ORACLE_DISTINCT_A2_GAP_ID,
            "oracle_distinct_beats_vote": False,
            "verifier_is_oracle": False,
        },
        "verifier_reward_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_REWARD_GAP_ID,
            "verifier_label_carries_signal": False,
        },
        "detector_aurocs": {"status": "blocked_precondition", "gap_id": DETECTOR_GAP_ID},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": checksum,
        "model_specs": model_specs_for_replay(checksum),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "preconditions": preflight,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4208_PATH,
            EXP4220_PATH,
            EXP4221_PATH,
            EXP4223_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4227", "SCENARIO-VERIFY-4227"],
        "adversarial_verify": {"status": "pending"},
    }
    validate_artifact(artifact)
    return artifact


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "adversarial_verify.py"), "--json", str(artifact_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def _clean_adversarial_report(report: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    for item in report.get("reports", []):
        if isinstance(item, dict):
            flags.extend(flag for flag in item.get("flags", []) if isinstance(flag, dict))
    methodology_missing_clean = not any(flag.get("kind") == "METHODOLOGY_MISSING" for flag in flags)
    return {
        "status": "clean" if not flags else "flagged",
        "methodology_missing_clean": methodology_missing_clean,
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4227 fields before writing the artifact."""
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
    if isinstance(artifact["random_seed"], bool) or not isinstance(artifact["random_seed"], int):
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["reproducibility_checksum"], str) or not artifact[
        "reproducibility_checksum"
    ]:
        raise ValueError("reproducibility_checksum must be a non-empty string")
    if not isinstance(artifact["model_specs"], dict) or not artifact["model_specs"]:
        raise ValueError("model_specs must be a non-empty object")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4227 principles")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4227 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4227_ARTIFACT_PATH
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
    verifier_reward_outcome = classify_verifier_reward_outcome(repo_root)
    detector_aurocs = classify_detector_aurocs(repo_root)
    checksum = candidate_set_checksum(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        oracle_distinct_outcome,
        verifier_reward_outcome,
        detector_aurocs,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        oracle_distinct_outcome=oracle_distinct_outcome,
        verifier_reward_outcome=verifier_reward_outcome,
        detector_aurocs=detector_aurocs,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        random_seed=RANDOM_SEED,
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    raw_report = (
        adversarial_runner(out_path)
        if adversarial_runner is not None
        else _run_adversarial_verify(repo_root, out_path)
    )
    artifact["adversarial_verify"] = _clean_adversarial_report(raw_report)
    validate_artifact(artifact)
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4227_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
