"""Tests for Exp 3005 solver-to-validator tree expansion.

Spec refs: REQ-VERIFY-3005, SCENARIO-VERIFY-3005.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import solver_to_validator_tree_expansion_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3005_solver_to_validator_tree_expansion_v1.py"


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        manifest_path=tmp_path / exp.VALIDATOR_MANIFEST_REL_PATH,
        z3_transcript_dir=tmp_path / exp.Z3_TRANSCRIPT_REL_DIR,
        runtime_transcript_dir=tmp_path / exp.RUNTIME_TRANSCRIPT_REL_DIR,
        started_at=10.0,
        clock=lambda: 10.5,
    )


def test_req_verify_3005_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3005: the expansion corpus is OpenSpec anchored."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3005" in spec
    assert "SCENARIO-VERIFY-3005" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "validator_manifest_path" in spec
    assert "partial_viability_checked" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3005_builds_twenty_exact_validator_trees() -> None:
    """SCENARIO-VERIFY-3005: every accepted item has runtime and Z3 authority."""
    items = exp.build_solver_items()
    rejected = exp.build_rejected_constraints()

    assert len(items) >= exp.MIN_SOLVER_ITEMS
    assert {item.source_family for item in items} >= {
        "exp2992_solver_feedback",
        "deterministic_generated",
    }
    assert {row["rejection_reason"] for row in rejected} >= {
        "nondeterministic_test",
        "missing_exact_check",
        "llm_only_label",
    }

    for item in items:
        tree = exp.build_validator_tree(item)
        authorities = {node["authority"] for node in tree["nodes"]}
        full_candidate = exp.candidate_from_item(item)
        full_feedback = exp.evaluate_validator_tree(tree, json.dumps(full_candidate, sort_keys=True))
        flipped = full_candidate | {
            "expected_status": "sat" if full_candidate["expected_status"] == "unsat" else "unsat"
        }
        flipped_feedback = exp.evaluate_validator_tree(tree, json.dumps(flipped, sort_keys=True))

        assert tree["root"]["op"] == "all"
        assert "runtime_json_parser" in authorities
        assert "z3_solver" in authorities
        assert full_feedback["accepted"] is True, item.item_id
        assert full_feedback["llm_judge_used"] is False
        assert not full_feedback["failing_node_ids"]
        assert flipped_feedback["accepted"] is False
        assert {
            "candidate_expected_status_mismatch",
            "reference_status_mismatch",
        } & set(flipped_feedback["rejection_reasons"])


def test_req_verify_3005_partial_viability_accepts_extendable_and_rejects_invalid_prefix() -> None:
    """REQ-VERIFY-3005: partial candidates fail early or remain extendable exactly."""
    for item in exp.build_solver_items()[:5]:
        tree = exp.build_validator_tree(item)
        valid_partial, invalid_partial = exp.partial_candidate_fixtures(item)
        valid_feedback = exp.evaluate_partial_candidate(
            tree,
            json.dumps(valid_partial, sort_keys=True),
        )
        invalid_feedback = exp.evaluate_partial_candidate(
            tree,
            json.dumps(invalid_partial, sort_keys=True),
        )
        malformed_feedback = exp.evaluate_partial_candidate(tree, "{")
        malformed_full_feedback = exp.evaluate_validator_tree(tree, "{")

        assert valid_feedback["accepted"] is True, item.item_id
        assert valid_feedback["extendable_to_reference"] is True
        assert valid_feedback["partial_viability_checked"] is True
        assert invalid_feedback["accepted"] is False, item.item_id
        assert invalid_feedback["extendable_to_reference"] is False
        assert "partial_assertions_not_reference_prefix" in invalid_feedback["rejection_reasons"]
        assert malformed_feedback["accepted"] is False
        assert "json_parse_error" in malformed_feedback["rejection_reasons"]
        assert malformed_full_feedback["accepted"] is False
        assert "json_parse_error" in malformed_full_feedback["rejection_reasons"]


def test_scenario_verify_3005_runner_writes_manifest_and_transcripts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3005: run writes artifact, manifest, and transcript evidence."""
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    manifest_path = tmp_path / artifact["validator_manifest_path"]
    manifest_rows = exp.load_manifest(manifest_path)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["validator_tree_expanded"] is True
    assert artifact["validator_manifest_path"] == str(exp.VALIDATOR_MANIFEST_REL_PATH)
    assert artifact["n_solver_items"] >= exp.MIN_SOLVER_ITEMS
    assert artifact["n_validator_trees"] == artifact["n_solver_items"]
    assert artifact["all_trees_exact_checked"] is True
    assert artifact["partial_viability_checked"] is True
    assert artifact["llm_judge_used"] is False
    assert artifact["honest_verdict"].startswith("ready:")
    assert len(artifact["z3_transcript_paths"]) == artifact["n_solver_items"]
    assert len(artifact["runtime_transcript_paths"]) == artifact["n_solver_items"]
    assert len(manifest_rows) == artifact["n_solver_items"]

    for row in manifest_rows:
        assert row["full_validation"]["accepted"] is True
        assert row["partial_viability"]["valid_partial"]["accepted"] is True
        assert row["partial_viability"]["invalid_partial"]["accepted"] is False
        assert row["validator_tree"]["tree_id"] == row["item_id"]
        assert row["z3_transcript_sha256"] == exp.sha256_file(tmp_path / row["z3_transcript_path"])
        assert row["runtime_transcript_sha256"] == exp.sha256_file(tmp_path / row["runtime_transcript_path"])
        assert "z3_solver" in {node["authority"] for node in row["validator_tree"]["nodes"]}

    for rel_path in artifact["z3_transcript_paths"] + artifact["runtime_transcript_paths"]:
        assert (tmp_path / rel_path).is_file()

    exp.validate_artifact(artifact)


def test_req_verify_3005_validation_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    """REQ-VERIFY-3005: artifact validation enforces corpus and authority gates."""
    artifact = exp.run_experiment(_config(tmp_path))
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "ready: incomplete"})
    with pytest.raises(ValueError, match="llm_judge_used"):
        exp.validate_artifact(artifact | {"llm_judge_used": True})
    with pytest.raises(ValueError, match="at least 20"):
        exp.validate_artifact(artifact | {"n_solver_items": 19})
    with pytest.raises(ValueError, match="n_validator_trees"):
        exp.validate_artifact(artifact | {"n_validator_trees": artifact["n_solver_items"] - 1})
    with pytest.raises(ValueError, match="all_trees_exact_checked"):
        exp.validate_artifact(artifact | {"all_trees_exact_checked": False})
    with pytest.raises(ValueError, match="partial_viability_checked"):
        exp.validate_artifact(artifact | {"partial_viability_checked": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "complete: wrong prefix"})
