"""Tests for Exp6506 V561 evidence corrigendum and V562 lineage lock.

Spec refs: REQ-BENCH-6506, SCENARIO-BENCH-6506-ROW-REPLAY,
SCENARIO-BENCH-6506-CORRIGENDUM, SCENARIO-BENCH-6506-EXP6505-NULL,
SCENARIO-BENCH-6506-LINEAGE-LOCK.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6506_v561_evidence_corrigendum_v562_lineage_lock as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.py "
    "-m pytest tests/python/test_experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.py "
    "--fail-under=100 --show-missing"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6506_v561_evidence_corrigendum_v562_lineage_lock --date 20260822"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.json"
)
DOCUMENTATION_COMMAND = "sed -n 1,220p ops/e2e-test-plan.md"
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6506_v561_evidence_corrigendum_v562_lineage_lock --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": DOCUMENTATION_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    result_path = tmp_path_factory.mktemp("exp6506") / mod.RESULT_RELATIVE_PATH.name
    return mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )


def test_req_bench_6506_spec_declares_corrigendum_contract() -> None:
    """REQ-BENCH-6506: OpenSpec owns the corrigendum and lineage lock."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6506") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6506-ROW-REPLAY",
        "SCENARIO-BENCH-6506-CORRIGENDUM",
        "SCENARIO-BENCH-6506-EXP6505-NULL",
        "SCENARIO-BENCH-6506-LINEAGE-LOCK",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "VERDICT_CLASS_MISMATCH",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6506_row_replay_and_corrigendum(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6506-ROW-REPLAY/CORRIGENDUM: Exp6504 is corrected."""

    replay = artifact["exp6504_row_recomputation"]
    correction = artifact["exp6504_corrigendum"]

    assert replay["row_replay_passed"] is True
    assert replay["reported_aggregate_matches_recomputed"] is True
    assert replay["historical_reproducibility_checksum_matches"] is True
    assert replay["raw_regeneration"]["hash_match_count"] == 480
    assert replay["label_recomputation"]["semantic_match_count"] == 480
    assert replay["label_recomputation"]["hash_match_count"] <= 480
    assert replay["replay_recomputation"]["failure_count"] == 0
    assert replay["replay_recomputation"]["hash_match_count"] <= 480
    assert replay["split_recomputation"]["hash_matches"] is True
    assert replay["held_cell_recomputation"]["hash_matches"] is True
    assert replay["stratum_recomputation"]["set_hash_matches"] is True
    assert replay["leakage_recomputation"]["hash_matches"] is True

    assert correction["original_verdict_class"] == "positive"
    assert correction["corrected_verdict_class"] == "circular_positive"
    assert correction["artifact_verdict_class"] == "partial"
    assert correction["positive_scientific_claim_allowed"] is False
    assert correction["eligible_for_v562_exact_branch_raw_label_use"] is True
    assert correction["adversarial_verification_receipt"]["exit_code"] == 1
    assert "VERDICT_CLASS_MISMATCH" in correction["adversarial_verification_receipt"]["flag_kinds"]

    assert artifact["verdict_class"] == "partial"
    assert artifact["v562_exact_branch_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete_")


def test_scenario_bench_6506_exp6505_terminal_null(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6506-EXP6505-NULL: zero-yield accounting is terminal."""

    receipt = artifact["exp6505_terminal_null_receipt"]

    assert receipt["status"] == "complete_null_formal_challenge_mutation_accounting"
    assert receipt["verdict_class"] == "null"
    assert receipt["request_count"] == 3
    assert receipt["terminal_request_count"] == 3
    assert receipt["accepted_mutation_count"] == 0
    assert receipt["challenge_generation_complete_score"] == 1.0
    assert receipt["challenge_pool_ready_score"] == 0.0
    assert receipt["terminal_null_frozen"] is True
    assert receipt["model_invocation_performed_by_exp6506"] is False
    assert receipt["downstream_dependency_allowed"] is False
    assert receipt["reported_aggregate_matches_recomputed"] is True


def test_scenario_bench_6506_lineage_lock_and_attacks(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6506-LINEAGE-LOCK: forbidden inputs fail closed."""

    decisions = {row["scope_id"]: row for row in artifact["lineage_decision_rows"]}
    assert decisions["exp6504_raw_instances"]["decision"] == "allow"
    assert decisions["exp6504_exact_labels"]["decision"] == "allow"
    assert decisions["exp6505_challenge_mutations"]["decision"] == "forbid"
    assert decisions["learned_trajectory_energy"]["decision"] == "forbid"
    assert decisions["hardware_acceleration"]["decision"] == "forbid"
    assert all(row["required_upstream_hash_present"] is True for row in decisions.values())

    attacks = artifact["forbidden_dependency_attack_matrix"]["rows"]
    assert {row["attack_id"] for row in attacks} == set(mod.ATTACK_IDS)
    assert artifact["forbidden_dependency_attack_matrix"]["all_attacks_fail_closed"] is True
    assert all(row["fail_closed"] is True for row in attacks)
    assert all(row["observed_ready_score_if_only_this_attack"] == 0.0 for row in attacks)

    assert mod.classify_lineage_dependency(
        {
            "scope_id": "challenge_pool_laundered_as_branch_advice",
            "upstream_artifact": "exp6505",
            "field": "challenge_pool_ready_score",
            "required_upstream_hash_present": True,
        }
    )["decision"] == "forbid"
    assert mod.classify_lineage_dependency(
        {
            "scope_id": "exp6504_raw_instances",
            "upstream_artifact": "exp6504",
            "field": "raw_instance_rows",
            "required_upstream_hash_present": False,
        }
    )["decision"] == "block"
    assert mod.classify_lineage_dependency(
        {
            "scope_id": "positive_reuse",
            "upstream_artifact": "exp6504",
            "field": "verdict_class",
            "required_upstream_hash_present": True,
        }
    )["decision"] == "forbid"
    unknown = mod.classify_lineage_dependency(
        {
            "scope_id": "fresh_unregistered_side_input",
            "upstream_artifact": "exp6507",
            "field": "side_input",
            "required_upstream_hash_present": True,
        }
    )
    assert unknown["decision"] == "forbid"
    assert unknown["reason"] == "unknown_dependency_fail_closed"
    blocked_status, blocked_verdict = mod._status_verdict(
        0.0,
        {"blocked_reason": "forced_block"},
    )
    assert blocked_status == "blocked_v561_evidence_corrigendum_v562_lineage_lock"
    assert blocked_verdict.endswith("forced_block")


def test_scenario_bench_6506_schema_checksum_and_validation(
    artifact: dict[str, Any],
) -> None:
    """REQ-BENCH-6506: artifact fields, provenance, and checksum validate."""

    result_path = Path(artifact["preconditions_checked"]["result_path"])
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        "json_pointers" in row and "local_reducer" in row
        for row in artifact["field_provenance"].values()
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "verdict_class cannot be positive for oracle replay",
            lambda item: item.__setitem__("verdict_class", "positive"),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
        ),
        (
            "verifier_is_oracle must be true for exact row checks",
            lambda item: item.__setitem__("verifier_is_oracle", False),
        ),
        (
            "exp6505 terminal null receipt mismatch",
            lambda item: item["exp6505_terminal_null_receipt"].__setitem__(
                "challenge_pool_ready_score",
                1.0,
            ),
        ),
        (
            "forbidden_dependency_attack_matrix false accepts",
            lambda item: item["forbidden_dependency_attack_matrix"].__setitem__(
                "all_attacks_fail_closed",
                False,
            ),
        ),
        (
            "v562_exact_branch_ready_score mismatch",
            lambda item: item.__setitem__("v562_exact_branch_ready_score", 0.0),
        ),
        (
            "reproducibility_checksum mismatch",
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        (
            "honest_verdict lacks terminal prefix",
            lambda item: item.__setitem__("honest_verdict", "ready"),
        ),
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)


def test_scenario_bench_6506_main_and_validate_roundtrip(tmp_path: Path) -> None:
    """REQ-BENCH-6506: CLI writes and validates the corrigendum."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260822", "--result-path", str(result_path)]) == 0
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["v562_exact_branch_ready_score"] == 1.0
