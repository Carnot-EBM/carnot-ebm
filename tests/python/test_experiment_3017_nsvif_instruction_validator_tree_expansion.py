"""Tests for Exp 3017 NSVIF instruction validator-tree expansion.

Spec refs: REQ-VERIFY-3017, SCENARIO-VERIFY-3017.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import nsvif_instruction_validator_tree_expansion_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3017_nsvif_instruction_validator_tree_expansion_v1.py"
REQUIRED_CATEGORIES = {
    "required_fields",
    "forbidden_tokens",
    "ordering_constraints",
    "numeric_bounds",
    "simple_transformations",
    "z3_relations",
    "python_ast",
    "runtime_invariants",
}


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        manifest_path=tmp_path / exp.VALIDATOR_MANIFEST_REL_PATH,
        z3_transcript_dir=tmp_path / exp.Z3_TRANSCRIPT_REL_DIR,
        runtime_transcript_dir=tmp_path / exp.RUNTIME_TRANSCRIPT_REL_DIR,
        started_at=10.0,
        clock=lambda: 12.25,
    )


def test_req_verify_3017_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3017: Exp 3017 is OpenSpec anchored and runnable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3017" in spec
    assert "SCENARIO-VERIFY-3017" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "instruction_validator_tree_ready" in spec
    assert "all_authoritative_nodes_exact_checked" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3017_builds_twenty_instruction_validator_trees() -> None:
    """SCENARIO-VERIFY-3017: small instruction items execute through exact nodes."""

    items = exp.build_instruction_items()
    rejected = exp.build_rejected_items()

    assert len(items) >= exp.MIN_INSTRUCTION_ITEMS
    assert {item.category for item in items} >= REQUIRED_CATEGORIES
    assert {row["rejection_reason"] for row in rejected} >= {
        "ambiguous_instruction",
        "nondeterministic_validator",
        "llm_only_label",
    }

    semantic_boundary_seen = False
    for item in items:
        tree = exp.build_validator_tree(item)
        authoritative_nodes = [
            node for node in tree["nodes"] if node.get("authoritative", True)
        ]
        semantic_nodes = [
            node for node in tree["nodes"] if node["kind"] == "semantic_boundary"
        ]
        semantic_boundary_seen = semantic_boundary_seen or bool(semantic_nodes)
        good_feedback = exp.evaluate_validator_tree(tree, item.known_good_candidate)
        bad_feedback = exp.evaluate_validator_tree(tree, item.known_bad_candidate)

        assert tree["root"]["op"] == "all"
        assert authoritative_nodes
        assert all(node["authority"] in exp.EXACT_AUTHORITIES for node in authoritative_nodes)
        assert all(node.get("authoritative") is False for node in semantic_nodes)
        assert good_feedback["accepted"] is True, item.item_id
        assert good_feedback["llm_judge_used"] is False
        assert not good_feedback["failing_node_ids"]
        assert bad_feedback["accepted"] is False, item.item_id
        assert bad_feedback["rejection_reasons"]
        assert bad_feedback["llm_judge_used"] is False

    assert semantic_boundary_seen is True


def test_scenario_verify_3017_runner_writes_manifest_and_transcripts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3017: terminal artifact has replayable exact evidence."""

    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    manifest_path = tmp_path / artifact["validator_manifest_path"]
    manifest_rows = exp.load_manifest(manifest_path)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["instruction_validator_tree_ready"] is True
    assert artifact["validator_manifest_path"] == str(exp.VALIDATOR_MANIFEST_REL_PATH)
    assert artifact["n_instruction_items"] >= exp.MIN_INSTRUCTION_ITEMS
    assert artifact["n_validator_trees"] == artifact["n_instruction_items"]
    assert 0.90 <= artifact["exact_check_coverage"] < 1.0
    assert artifact["all_authoritative_nodes_exact_checked"] is True
    assert artifact["llm_judge_used"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["runtime_transcript_paths"]) == artifact["n_instruction_items"]
    assert len(artifact["z3_transcript_paths"]) >= 4
    assert len(manifest_rows) == artifact["n_instruction_items"]

    for row in manifest_rows:
        assert row["known_good_validation"]["accepted"] is True
        assert row["known_bad_validation"]["accepted"] is False
        assert row["validator_tree"]["tree_id"] == row["item_id"]
        assert row["runtime_transcript_sha256"] == exp.sha256_file(
            tmp_path / row["runtime_transcript_path"]
        )
        assert row["all_authoritative_nodes_exact_checked"] is True
        for node in row["validator_tree"]["nodes"]:
            if node["kind"] == "semantic_boundary":
                assert node["authoritative"] is False
        if row.get("z3_transcript_path"):
            assert row["z3_transcript_sha256"] == exp.sha256_file(
                tmp_path / row["z3_transcript_path"]
            )

    for rel_path in artifact["z3_transcript_paths"] + artifact["runtime_transcript_paths"]:
        assert (tmp_path / rel_path).is_file()

    exp.validate_artifact(artifact)


def test_req_verify_3017_validation_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    """REQ-VERIFY-3017: terminal validation enforces exact authority gates."""

    artifact = exp.run_experiment(_config(tmp_path))
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="llm_judge_used"):
        exp.validate_artifact(artifact | {"llm_judge_used": True})
    with pytest.raises(ValueError, match="at least 20"):
        exp.validate_artifact(artifact | {"n_instruction_items": 19})
    with pytest.raises(ValueError, match="n_validator_trees"):
        exp.validate_artifact(
            artifact | {"n_validator_trees": artifact["n_instruction_items"] - 1}
        )
    with pytest.raises(ValueError, match="exact_check_coverage"):
        exp.validate_artifact(artifact | {"exact_check_coverage": 0.5})
    with pytest.raises(ValueError, match="all_authoritative_nodes_exact_checked"):
        exp.validate_artifact(artifact | {"all_authoritative_nodes_exact_checked": False})
    with pytest.raises(ValueError, match="rejected_items"):
        exp.validate_artifact(artifact | {"rejected_items": []})
    with pytest.raises(ValueError, match="transcript paths"):
        exp.validate_artifact(artifact | {"z3_transcript_paths": []})
    with pytest.raises(ValueError, match="instruction_validator_tree_ready"):
        exp.validate_artifact(artifact | {"instruction_validator_tree_ready": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready: wrong prefix"})


def test_req_verify_3017_malformed_candidates_fail_closed() -> None:
    """REQ-VERIFY-3017: malformed candidates reject by exact local reasons."""

    items = {item.item_id: item for item in exp.build_instruction_items()}

    malformed_json_checks = {
        "if-3017-001": {"json_parse_error"},
        "if-3017-003": {"json_parse_error"},
        "if-3017-005": {"json_parse_error"},
        "if-3017-009": {"json_parse_error"},
    }
    for item_id, expected_reasons in malformed_json_checks.items():
        feedback = exp.evaluate_validator_tree(exp.build_validator_tree(items[item_id]), "{")
        assert feedback["accepted"] is False
        assert expected_reasons <= set(feedback["rejection_reasons"])

    z3_tree = exp.build_validator_tree(items["if-3017-012"])
    z3_parse = exp.evaluate_validator_tree(z3_tree, "{")
    z3_type = exp.evaluate_validator_tree(z3_tree, json.dumps({"x": "4", "y": 6}))
    ast_tree = exp.build_validator_tree(items["if-3017-016"])
    missing_function = exp.evaluate_validator_tree(
        ast_tree,
        "def other(text):\n    return text\n",
    )
    source_status = exp.source_artifact_status(REPO_ROOT)

    assert z3_parse["rejection_reasons"] == ["json_parse_error"]
    assert z3_type["rejection_reasons"] == ["missing_integer_assignment"]
    assert "function_signature_mismatch" in missing_function["rejection_reasons"]
    assert source_status["exp2994"]["present"] is True
    assert source_status["exp3005"]["present"] is True
