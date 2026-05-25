"""Tests for Exp 3084 ReSyn exact fixture-bank generation.

Spec refs: REQ-VERIFY-3084, SCENARIO-VERIFY-3084.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import resyn_exact_fixture_bank_generator_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
REQUIRED_ARTIFACT_FIELDS = {
    "resyn_fixture_bank_ready",
    "exact_fixture_count",
    "family_count",
    "fixture_manifest_path",
    "exact_label_sources",
    "perturbation_families",
    "tests_added_or_reused",
    "preconditions_checked",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
REQUIRED_FIXTURE_FIELDS = {
    "fixture_id",
    "family",
    "task_axis",
    "perturbation_family",
    "leakage_safe_prompt_payload",
    "prompt_payload_sha256",
    "exact_label",
    "label_source",
}


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_sources(root: Path) -> None:
    _write_text(
        root,
        exp.EXP3070_REL_PATH,
        json.dumps({"honest_verdict": "complete: first_token_panel_ready=true"}),
    )
    _write_text(
        root,
        exp.EXP3083_REL_PATH,
        json.dumps(
            {
                "honest_verdict": "complete: protocol ready",
                "experiment_metric_contracts": {
                    "exp3084": {"purpose": "exact_fixture_perturbation_bank"}
                },
            }
        ),
    )
    _write_text(root, Path("CODEX.md"), "Spec First\nWrite Tests First\nVerify\n")
    _write_text(
        root, Path("research-references.md"), "ReSyn exact fixtures and verifier hardness.\n"
    )


def _config(tmp_path: Path) -> exp.FixtureBankConfig:
    return exp.FixtureBankConfig(
        repo_root=tmp_path,
        output_path=tmp_path / exp.OUTPUT_REL_PATH,
        manifest_path=tmp_path / exp.MANIFEST_REL_PATH,
        started_s=10.0,
        clock=lambda: 14.0,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_verify_3084_spec_anchor_exists() -> None:
    """REQ-VERIFY-3084: OpenSpec declares the exact fixture-bank contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3084" in spec
    assert "SCENARIO-VERIFY-3084" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert exp.MANIFEST_REL_PATH.as_posix() in spec
    assert "blocked_exact_label_tooling_missing" in spec
    assert "resyn_fixture_bank_ready" in spec


def test_scenario_verify_3084_writes_exact_fixture_bank(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3084: the generator writes 64+ leakage-safe exact fixtures."""
    _write_sources(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    manifest_path = tmp_path / artifact["fixture_manifest_path"]
    rows = _read_jsonl(manifest_path)

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["resyn_fixture_bank_ready"] is True
    assert artifact["exact_fixture_count"] == 72
    assert artifact["family_count"] == 3
    assert artifact["family_counts"] == {
        "arithmetic_code_assertions": 24,
        "repairable_invalid_candidates": 24,
        "smt_constraints": 24,
    }
    assert manifest_path.is_file()
    assert len(rows) == artifact["exact_fixture_count"]
    assert artifact["fixture_manifest_sha256"] == exp.sha256_file(manifest_path)
    assert set(artifact["exact_label_sources"]) == {
        "json_parser",
        "python_ast_runtime_execution",
        "z3_solver",
    }
    assert {"smt_sat_solving", "smt_unsat_abstention", "python_assertion_repair"} <= set(
        artifact["perturbation_families"]
    )
    assert artifact["task_axes"] == ["abstaining", "repairing", "solving", "verifying"]
    assert artifact["tests_added_or_reused"] == [
        "tests/python/test_experiment_3084_resyn_exact_fixture_bank_generator.py"
    ]
    assert artifact["inference_substrate"]["no_live_llm_inference"] is True
    assert artifact["inference_substrate"]["llm_used_for_labels"] is False
    assert artifact["duration_s"] == pytest.approx(4.0)

    assert all(REQUIRED_FIXTURE_FIELDS <= set(row) for row in rows)
    assert len({row["fixture_id"] for row in rows}) == len(rows)
    assert {row["family"] for row in rows} == {
        "arithmetic_code_assertions",
        "repairable_invalid_candidates",
        "smt_constraints",
    }
    assert {row["task_axis"] for row in rows} == {"abstaining", "repairing", "solving", "verifying"}
    assert all(
        row["prompt_payload_sha256"] == exp.hash_prompt_payload(row["leakage_safe_prompt_payload"])
        for row in rows
    )
    assert not any(
        "answer" in json.dumps(row["leakage_safe_prompt_payload"]).lower() for row in rows
    )

    smt_rows = [row for row in rows if row["family"] == "smt_constraints"]
    assert {row["exact_label"]["solver_status"] for row in smt_rows} == {"sat", "unsat"}
    assert all(row["label_source"] == "z3_solver" for row in smt_rows)

    arithmetic_rows = [row for row in rows if row["family"] == "arithmetic_code_assertions"]
    assert {row["exact_label"]["assertion_passes"] for row in arithmetic_rows} == {True, False}
    assert all(row["label_source"] == "python_ast_runtime_execution" for row in arithmetic_rows)

    repair_rows = [row for row in rows if row["family"] == "repairable_invalid_candidates"]
    assert all(row["exact_label"]["repairable"] is True for row in repair_rows)
    assert all(row["exact_label"]["candidate_valid"] is False for row in repair_rows)
    assert any(row["label_source"] == "json_parser" for row in repair_rows)
    assert any(row["label_source"] == "z3_solver" for row in repair_rows)
    assert any(row["label_source"] == "python_ast_runtime_execution" for row in repair_rows)

    exp.validate_fixture_rows(rows)
    exp.validate_artifact(artifact, rows)
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_verify_3084_exact_authorities_reject_tampering(tmp_path: Path) -> None:
    """REQ-VERIFY-3084: fixture labels are revalidated by exact local authorities."""
    _write_sources(tmp_path)
    artifact = exp.write_artifact(_config(tmp_path))
    rows = _read_jsonl(tmp_path / artifact["fixture_manifest_path"])

    broken_hash = dict(rows[0])
    broken_hash["prompt_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        exp.validate_fixture_rows([broken_hash])

    broken_smt = dict(next(row for row in rows if row["family"] == "smt_constraints"))
    broken_smt["exact_label"] = dict(broken_smt["exact_label"]) | {"solver_status": "sat"}
    with pytest.raises(ValueError, match="SMT label mismatch"):
        exp.validate_fixture_rows([broken_smt])

    broken_arith = dict(next(row for row in rows if row["family"] == "arithmetic_code_assertions"))
    broken_arith["exact_label"] = dict(broken_arith["exact_label"]) | {
        "assertion_passes": not broken_arith["exact_label"]["assertion_passes"]
    }
    with pytest.raises(ValueError, match="arithmetic assertion label mismatch"):
        exp.validate_fixture_rows([broken_arith])

    broken_repair = dict(
        next(row for row in rows if row["family"] == "repairable_invalid_candidates")
    )
    broken_repair["exact_label"] = dict(broken_repair["exact_label"]) | {"repairable": False}
    with pytest.raises(ValueError, match="repair fixture label mismatch"):
        exp.validate_fixture_rows([broken_repair])


def test_req_verify_3084_blocks_when_exact_tooling_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-3084: missing Z3 writes a blocked exact-label diagnostic."""
    _write_sources(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path), z3_module=None)

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["honest_verdict"].startswith("blocked_exact_label_tooling_missing")
    assert artifact["resyn_fixture_bank_ready"] is False
    assert artifact["exact_fixture_count"] == 0
    assert artifact["family_count"] == 0
    assert artifact["fixture_manifest_path"] == exp.MANIFEST_REL_PATH.as_posix()
    assert artifact["exact_label_sources"] == []
    assert artifact["perturbation_families"] == []
    assert artifact["preconditions_checked"]["z3_import"]["ok"] is False
    assert artifact["inference_substrate"]["no_live_llm_inference"] is True
    assert not (tmp_path / exp.MANIFEST_REL_PATH).exists()
    exp.validate_artifact(artifact)


def test_req_verify_3084_artifact_validation_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3084: terminal artifacts cannot claim readiness without evidence."""
    _write_sources(tmp_path)
    artifact = exp.write_artifact(_config(tmp_path))
    rows = _read_jsonl(tmp_path / artifact["fixture_manifest_path"])

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: x"}, rows)
    with pytest.raises(ValueError, match="at least 64 exact fixtures"):
        exp.validate_artifact(artifact | {"exact_fixture_count": 63}, rows[:63])
    with pytest.raises(ValueError, match="at least three families"):
        exp.validate_artifact(artifact | {"family_count": 2}, rows)
    with pytest.raises(ValueError, match="no live LLM inference"):
        bad_substrate = dict(artifact["inference_substrate"]) | {"no_live_llm_inference": False}
        exp.validate_artifact(artifact | {"inference_substrate": bad_substrate}, rows)
    with pytest.raises(ValueError, match="terminal success prefix"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready without prefix"}, rows)


def test_req_verify_3084_safe_arithmetic_eval_rejects_unsupported_ast() -> None:
    """REQ-VERIFY-3084: local arithmetic execution is intentionally bounded."""
    assert exp.safe_eval_arithmetic("(3 + 5) * 2") == 16
    with pytest.raises(ValueError, match="unsupported arithmetic expression"):
        exp.safe_eval_arithmetic("__import__('os').system('true')")


def test_req_verify_3084_defensive_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-3084: malformed rows and unsupported validators fail closed."""
    _write_sources(tmp_path)
    artifact = exp.write_artifact(_config(tmp_path))
    rows = _read_jsonl(tmp_path / artifact["fixture_manifest_path"])

    with pytest.raises(ValueError, match="z3_module is required for SMT fixtures"):
        exp.generate_fixture_rows(z3_module=None)
    with pytest.raises(ValueError, match="z3_module is required for fixture validation"):
        exp.validate_fixture_rows(rows, z3_module=None)

    missing_field = dict(rows[0])
    del missing_field["label_source"]
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_fixture_rows([missing_field])

    leaked_prompt = dict(rows[0])
    leaked_prompt["leakage_safe_prompt_payload"] = dict(
        leaked_prompt["leakage_safe_prompt_payload"]
    ) | {"answer": "SAT"}
    leaked_prompt["prompt_payload_sha256"] = exp.hash_prompt_payload(
        leaked_prompt["leakage_safe_prompt_payload"]
    )
    with pytest.raises(ValueError, match="prompt payload leaks answer"):
        exp.validate_fixture_rows([leaked_prompt])

    unknown_family = dict(rows[0]) | {"family": "unknown_family"}
    with pytest.raises(ValueError, match="unknown fixture family"):
        exp.validate_fixture_rows([unknown_family])

    unknown_repair_source = dict(
        next(row for row in rows if row["family"] == "repairable_invalid_candidates")
    )
    unknown_repair_source["label_source"] = "unknown_source"
    with pytest.raises(ValueError, match="unknown repair label source"):
        exp.validate_fixture_rows([unknown_repair_source])

    with pytest.raises(ValueError, match="LLM labels were not used"):
        bad_substrate = dict(artifact["inference_substrate"]) | {"llm_used_for_labels": True}
        exp.validate_artifact(artifact | {"inference_substrate": bad_substrate}, rows)
    with pytest.raises(ValueError, match="blocked artifact must disclose"):
        exp.validate_artifact(artifact | {"resyn_fixture_bank_ready": False}, rows)
    with pytest.raises(ValueError, match="exact_fixture_count must equal"):
        exp.validate_artifact(artifact | {"exact_fixture_count": len(rows) + 1}, rows)

    assert exp.safe_eval_arithmetic("-(3 + 2)") == -5
    assert exp.safe_eval_arithmetic("+4") == 4
    assert exp.safe_eval_arithmetic("7 // 2") == 3
    assert exp.safe_eval_arithmetic("7 % 3") == 1
    with pytest.raises(ValueError, match="unsupported arithmetic expression"):
        exp.safe_eval_arithmetic("1 +")
    with pytest.raises(ValueError, match="unsupported SMT constraint op"):
        exp._z3_constraint({"op": "gt", "var": "x", "value": 1}, {})
    with pytest.raises(ValueError, match="unsupported SMT constraint op"):
        exp._constraint_strings([{"op": "gt", "var": "x", "value": 1}])

    assert exp._relative_path(Path("/tmp/repo"), Path("/other/manifest.jsonl")) == (
        "/other/manifest.jsonl"
    )
