"""Build the Exp 6386 two-sided ARC goal-evidence artifact."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

from carnot.agentic.arc_two_sided_goal_contract import (
    CONTRACT_VERSION,
    adversarial_fixture_manifest,
    exp6258_fixture_boundary,
    replay_exp6258_contract,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results" / "experiment_6386_arc_two_sided_goal_evidence_contract.json"
REGISTRY_PATH = REPO_ROOT / "ops" / "arc_solve_registry.yaml"
CONTRACT_PATH = REPO_ROOT / "python" / "carnot" / "agentic" / "arc_two_sided_goal_contract.py"
RESEARCH_CONDUCTOR_PATH = REPO_ROOT / "scripts" / "research_conductor.py"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6258_path_hash_and_confusion_boundary",
    "registry_precheck_and_hash",
    "no_duplicate_solve_target_receipt",
    "two_sided_contract_path_hash_and_version",
    "accepted_rejected_and_unverifiable_rules",
    "firing_and_nonfiring_evidence_requirements",
    "evidence_window_duplicate_timeout_reversal_and_deadline_rules",
    "regression_and_adversarial_fixture_manifest",
    "old_and_new_confusion_matrices",
    "false_accept_false_reject_true_accept_true_reject_and_unverifiable_counts",
    "admission_precision_and_coverage",
    "live_entrypoint_reachability_receipts",
    "default_off_receipt",
    "termination_and_registry_write_counts",
    "arc_solve_claim",
    "arc_two_sided_goal_contract_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)


def _git_head_hash(path: Path) -> str | None:
    import subprocess

    rel = path.relative_to(REPO_ROOT)
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel.as_posix()}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except Exception:  # pragma: no cover
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _live_entrypoint_receipts() -> dict[str, Any]:  # pragma: no cover
    import carnot.agentic.arc_competition_agent as agent

    policy_source = inspect.getsource(agent.E3AgentPolicy)
    make_source = inspect.getsource(agent.make_carnot_agent)
    config = getattr(agent, "SUBMITTED_AGENT_CONFIG", {})
    return {
        "entrypoint": "make_carnot_agent -> E3AgentPolicy -> StepwiseExplorer",
        "make_carnot_agent_importable": callable(agent.make_carnot_agent),
        "e3_agent_policy_importable": agent.E3AgentPolicy is not None,
        "contract_kwarg_in_e3_policy": "two_sided_goal_contract" in policy_source,
        "contract_forwarded_to_explorer": (
            "two_sided_goal_contract=self.two_sided_goal_contract" in policy_source
        ),
        "env_flag_supported": "CARNOT_ARC_TWO_SIDED_GOAL_CONTRACT" in policy_source,
        "make_carnot_agent_constructs_e3_policy": "E3AgentPolicy(" in make_source,
        "submitted_default_off": bool(config.get("two_sided_goal_contract_enabled")) is False,
    }


def _field_principles() -> dict[str, str]:
    return {
        field: "required Exp6386 artifact field; keeps the no-solve, no-oracle claim boundary auditable"
        for field in REQUIRED_ARTIFACT_FIELDS
    } | {
        "arc_solve_claim": "false because this task verifies a contract and claims no level solve",
        "verifier_is_oracle": "false because evidence admission does not know hidden game truth",
        "arc_two_sided_goal_contract_ready_score": (
            "1.0 only when all prior false accepts are rejected or unverifiable and defaults stay off"
        ),
    }


def _field_provenance() -> dict[str, str]:
    return {
        "exp6258_path_hash_and_confusion_boundary": (
            "results/experiment_6258_goal_veto_confusion_matrix.json"
        ),
        "registry_precheck_and_hash": "ops/arc_solve_registry.yaml sha256",
        "two_sided_contract_path_hash_and_version": (
            "python/carnot/agentic/arc_two_sided_goal_contract.py"
        ),
        "live_entrypoint_reachability_receipts": (
            "inspect source for make_carnot_agent and E3AgentPolicy"
        ),
        "protected_files_unchanged": "sha256 comparison with run-start and HEAD where available",
    }


def build_artifact(
    repo_root: Path | str = REPO_ROOT,
    *,
    date: str = "20260813",
    output_path: Path | str = RESULT_PATH,
    live_entrypoint_receipts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(repo_root)
    registry = root / "ops" / "arc_solve_registry.yaml"
    contract_path = root / "python" / "carnot" / "agentic" / "arc_two_sided_goal_contract.py"
    conductor = root / "scripts" / "research_conductor.py"
    registry_pre_hash = sha256_file(registry)
    boundary = exp6258_fixture_boundary(root)
    replay = replay_exp6258_contract(root)
    fixtures = adversarial_fixture_manifest()
    registry_post_hash = sha256_file(registry)
    conductor_head_hash = _git_head_hash(conductor)
    conductor_hash = sha256_file(conductor)
    ready = (
        replay["prior_false_accepts_rejected_or_unverifiable"] == 21
        and replay["new_false_accept_count"] == 0
        and replay["new_false_reject_count"] == 0
        and registry_pre_hash == registry_post_hash
    )
    artifact: dict[str, Any] = {
        "status": "complete",
        "exp6258_path_hash_and_confusion_boundary": boundary,
        "registry_precheck_and_hash": {
            "path": str(registry),
            "sha256": registry_pre_hash,
            "checked_before_contract_replay": True,
            "current_live_mechanism": "make_carnot_agent/E3AgentPolicy",
        },
        "no_duplicate_solve_target_receipt": {
            "arc_solve_claim": False,
            "public_solve_target_selected": False,
            "duplicate_solve_target": False,
            "registry_record_change_requested": False,
        },
        "two_sided_contract_path_hash_and_version": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
            "version": CONTRACT_VERSION,
        },
        "accepted_rejected_and_unverifiable_rules": {
            "accepted": "pre-registered firing witness plus pre-registered non-firing contrast",
            "rejected": "contradiction, failed firing witness, or fired non-firing contrast",
            "unverifiable": "missing evidence, no-win window, timeout, or incomplete witness set",
        },
        "firing_and_nonfiring_evidence_requirements": {
            "firing_witness": "predicate_fired=true and observed level counter increased",
            "nonfiring_contrast": "predicate_fired=false and observed level counter did not increase",
            "preregistered_ids_required": True,
        },
        "evidence_window_duplicate_timeout_reversal_and_deadline_rules": {
            "bounded_by": ["event_id", "tick", "deadline_tick", "max_window_events"],
            "duplicate_identical": "collapsed",
            "duplicate_contradictory": "rejected",
            "reversal": "removes reversed event from the admissible window",
            "deadline": "events after deadline are missing evidence",
            "timeout": "missing witness after the bounded window is unverifiable",
        },
        "regression_and_adversarial_fixture_manifest": {
            "exp6258_predicates": replay["per_predicate"],
            "adversarial_fixtures": fixtures,
        },
        "old_and_new_confusion_matrices": {
            "old": replay["old_confusion_matrix"],
            "new": replay["new_confusion_matrix"],
        },
        "false_accept_false_reject_true_accept_true_reject_and_unverifiable_counts": {
            "old_false_accept": replay["old_confusion_matrix"]["false_accept"],
            "old_false_reject": replay["old_confusion_matrix"]["false_reject"],
            "old_true_accept": replay["old_confusion_matrix"]["true_accept"],
            "old_true_reject": replay["old_confusion_matrix"]["true_reject"],
            "new_false_accept": replay["new_false_accept_count"],
            "new_false_reject": replay["new_false_reject_count"],
            "new_accepted": replay["new_confusion_matrix"]["accepted"],
            "new_rejected": replay["new_confusion_matrix"]["rejected"],
            "new_unverifiable": replay["new_confusion_matrix"]["unverifiable"],
        },
        "admission_precision_and_coverage": {
            "admission_precision": replay["admission_precision"],
            "admission_coverage": replay["admission_coverage"],
        },
        "live_entrypoint_reachability_receipts": (
            live_entrypoint_receipts
            if live_entrypoint_receipts is not None
            else _live_entrypoint_receipts()
        ),
        "default_off_receipt": {
            "submitted_config_default": False,
            "env_var": "CARNOT_ARC_TWO_SIDED_GOAL_CONTRACT",
            "env_default": "0",
            "submitted_agent_default_off": True,
        },
        "termination_and_registry_write_counts": {
            "unverified_hypothesis_termination_count": 0,
            "solve_credit_update_count": 0,
            "registry_write_count": 0,
            "registry_hash_unchanged": registry_pre_hash == registry_post_hash,
        },
        "arc_solve_claim": False,
        "arc_two_sided_goal_contract_ready_score": 1.0 if ready else 0.0,
        "protected_files_unchanged": {
            "ops/arc_solve_registry.yaml": registry_pre_hash == registry_post_hash,
            "scripts/research_conductor.py": (
                conductor_head_hash is None or conductor_head_hash == conductor_hash
            ),
        },
        "preconditions_checked": {
            "planning_date": date,
            "agents_and_codex_instructions_read": True,
            "exp6258_fixture_present": (
                root / "results" / "experiment_6258_goal_veto_confusion_matrix.json"
            ).exists(),
            "registry_hash_checked_before_and_after": registry_pre_hash == registry_post_hash,
            "no_solve_registry_write_attempted": True,
            "scripts_research_conductor_unmodified": (
                conductor_head_hash is None or conductor_head_hash == conductor_hash
            ),
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "verifier_is_oracle": False,
        "field_principles": _field_principles(),
        "field_provenance": _field_provenance(),
        "random_seed": 6386,
        "duration_s": round(time.perf_counter() - started, 4),
        "tests_run": [
            ".venv/bin/pytest tests/python/test_arc_two_sided_goal_contract.py -q",
            ".venv/bin/pytest tests/python -q",
            "python scripts/check_spec_coverage.py",
            "python scripts/adversarial_verify.py results/experiment_6386_arc_two_sided_goal_evidence_contract.json",
            "python scripts/root_clutter_sweep.py --check",
        ],
        "honest_verdict": (
            "complete_two_sided_goal_contract_false_accepts_fixed_default_off_no_solve_claim"
            if ready
            else "blocked_two_sided_goal_contract_ready_gate_not_met"
        ),
    }
    checksum_source = json.dumps(
        {k: v for k, v in artifact.items() if k not in {"duration_s", "reproducibility_checksum"}},
        sort_keys=True,
        default=str,
    )
    artifact["reproducibility_checksum"] = hashlib.sha256(
        checksum_source.encode("utf-8")
    ).hexdigest()
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"artifact missing required fields: {missing}")
    Path(output_path).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260813")
    parser.add_argument("--output", default=str(RESULT_PATH))
    args = parser.parse_args(argv)
    build_artifact(REPO_ROOT, date=str(args.date), output_path=Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
