"""Tests for Exp6503 V561 source delta and method contract.

Spec refs: REQ-REPORT-6503, SCENARIO-REPORT-6503-GATE,
SCENARIO-REPORT-6503-SOURCES, SCENARIO-REPORT-6503-METHODS,
SCENARIO-REPORT-6503-AUTHORITY, SCENARIO-REPORT-6503-DEPENDENCIES,
SCENARIO-REPORT-6503-SCHEMA.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6503_v561_source_delta_method_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6503_v561_source_delta_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6503_v561_source_delta_method_contract.py "
    "-m pytest tests/python/test_experiment_6503_v561_source_delta_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6503_v561_source_delta_method_contract.py "
    "--fail-under=100 --show-missing"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6503_v561_source_delta_method_contract.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6503_v561_source_delta_method_contract "
    "--date 20260822"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6503_v561_source_delta_method_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6503_v561_source_delta_method_contract.json"
)
DOC_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; assert Path('ops/e2e-test-plan.md').exists()\""
)
TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": DOC_COMMAND, "exit_code": 0},
]


def _receipt(
    url: str,
    body: str,
    *,
    status_code: int = 200,
    error: str | None = None,
) -> mod.JsonDict:
    return {
        "ok": error is None and 200 <= status_code < 400,
        "status_code": status_code,
        "url": url,
        "headers": {"content-type": "application/test"},
        "body": body,
        "error": error,
    }


def _fake_fetcher(url: str, source_id: str) -> mod.JsonDict:
    if source_id.startswith("arxiv_"):
        title = mod.SOURCE_BY_ID[source_id]["title"]
        version = mod.SOURCE_BY_ID[source_id]["fallback_version"]
        return _receipt(
            url,
            (
                f"<html><h1>Title:{title}</h1>"
                f"<div class='submission-history'>[{version}] Tue, 2 Jun 2026</div>"
                "</html>"
            ),
        )
    if source_id == "openreview_clause_predictions":
        return _receipt(
            url,
            json.dumps(
                {
                    "name": "ChallengeRequiredError",
                    "message": "Challenge verification required",
                    "status": 403,
                }
            ),
            status_code=403,
            error="Challenge verification required",
        )
    if source_id.startswith("semantic_scholar_"):
        return _receipt(
            url,
            json.dumps(
                {
                    "message": "Too Many Requests. Please wait and try again.",
                    "code": 429,
                }
            ),
            status_code=429,
            error="HTTP Error 429",
        )
    if source_id == "huggingface_papers":
        return _receipt(url, json.dumps({"error": "Paper not found"}), status_code=404)
    if source_id == "github_neurosat":
        return _receipt(url, "<html><title>GitHub - dmeoli/NeuroSAT</title></html>")
    if source_id == "github_neuralsat":
        return _receipt(url, "<html><title>GitHub - dynaroars/neuralsat</title></html>")
    if source_id == "extropic_z1_update":
        return _receipt(url, "<html><title>From One to One Billion: Torx, Thermalizers, and Z1</title></html>")
    if source_id == "logical_intelligence_kona":
        return _receipt(url, "<html><title>Kona: Energy-Based Models (EBMs) for AI Reasoning</title></html>")
    raise AssertionError(f"unexpected source {source_id}: {url}")


def _source_receipts() -> list[mod.JsonDict]:
    return mod.collect_source_receipts(
        fetcher=_fake_fetcher,
        access_date="2026-08-22",
        network_state={
            "network_required": True,
            "network_used": True,
            "network_available": True,
            "checked_at_utc": "2026-08-22T12:00:00Z",
        },
    )


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        source_receipt_rows=_source_receipts(),
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        access_date="2026-08-22",
    )


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_report_6503_spec_declares_method_contract() -> None:
    """REQ-REPORT-6503: OpenSpec records the Exp6503 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6503") :]
    normalized = " ".join(section.split())

    for token in (
        "SCENARIO-REPORT-6503-GATE",
        "SCENARIO-REPORT-6503-SOURCES",
        "SCENARIO-REPORT-6503-METHODS",
        "SCENARIO-REPORT-6503-AUTHORITY",
        "SCENARIO-REPORT-6503-DEPENDENCIES",
        "SCENARIO-REPORT-6503-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle=false`",
    ):
        assert token in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6503_sources_pin_receipts_and_boundaries() -> None:
    """SCENARIO-REPORT-6503-SOURCES: receipts do not become local claims."""

    rows = _source_receipts()
    by_id = {row["source_id"]: row for row in rows}

    assert len(rows) == len(mod.SOURCE_MANIFEST)
    assert by_id["arxiv_symbolic_certification"]["title"] == (
        "Position: Certified Correctness in Neural Constraint Reasoning Requires Symbolic Integration"
    )
    assert by_id["arxiv_branch_order"]["available_version"] == "arXiv:2603.07176v1"
    assert by_id["arxiv_lns"]["source_class"] == "primary_paper"
    assert by_id["openreview_clause_predictions"]["retrieval_state"] == "blocked_challenge"
    assert by_id["semantic_scholar_ebt"]["retrieval_state"] == "rate_limited"
    assert by_id["huggingface_papers"]["retrieval_state"] == "not_indexed"
    assert by_id["github_neurosat"]["source_class"] == "implementation_reference"
    assert by_id["extropic_z1_update"]["source_class"] == "product_claim"
    assert by_id["logical_intelligence_kona"]["claim_boundary"] == "no_local_runner_or_weights"
    for row in rows:
        assert row["access_date"] == "2026-08-22"
        assert row["stable_url"].startswith("https://")
        assert row["response_sha256"].startswith("sha256:")
        assert row["bounded_carnot_implication"]
        assert "abstract" not in row


def test_scenario_report_6503_methods_authority_and_dependencies(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6503-METHODS/AUTHORITY/DEPENDENCIES: contract is local."""

    artifact = _artifact(tmp_path)
    contract = artifact["method_contract"]
    authority = artifact["authority_boundary"]
    dependencies = {row["dependency_id"]: row for row in artifact["dependency_decision_rows"]}
    methods = {row["method_id"]: row for row in contract["promoted_method_rows"]}

    assert artifact["method_contract_ready_score"] == 1.0
    assert contract["benchmark"]["families"] == [
        "random_3cnf",
        "pseudo_industrial_3cnf",
        "tseitin",
        "pigeonhole",
        "graph_coloring",
        "small_scheduling",
    ]
    assert contract["benchmark"]["minimum_held_cell_size"] == 30
    assert "solver_hardness" in contract["benchmark"]["shift_axes"]
    assert "conflicts" in contract["solver_metrics"]
    assert "propagation_pressure" in contract["structural_features"]
    assert contract["branch_checkpoints"]["conflict_budgets"] == [0, 16, 64, 256]
    assert contract["model_controls"] == [
        "analytic_structural",
        "solver_native_dynamic",
        "random_order",
        "linear",
        "mlp",
        "compact_kan",
        "gnn",
    ]
    assert contract["failure_conditions"]

    assert set(methods) == {
        "symbolic_certification",
        "initial_branch_ranking",
        "clause_prediction_advice",
        "exact_repair_lns",
    }
    assert methods["symbolic_certification"]["local_test_target"] == "exp6504_exact_labels"
    assert methods["initial_branch_ranking"]["local_test_target"] == "exp6507_exp6508_branch_ab"
    assert methods["clause_prediction_advice"]["local_test_target"] == "exp6506_exact_branch_labels"
    assert methods["exact_repair_lns"]["local_test_target"] == "exp6509_exact_repair_lns"
    assert all(row["paper_claim_is_local_evidence"] is False for row in methods.values())

    assert authority["neural_advice_may"] == ["order_search", "select_neighborhood", "abstain"]
    assert authority["neural_advice_must_not"] == [
        "accept_solution",
        "label_solution",
        "release_solution",
        "override_exact_authority",
    ]
    assert authority["exact_authorities"] == [
        "exact_cdcl_solver",
        "exact_csp_repair",
        "executable_validity_check",
    ]

    assert dependencies["z3_existing"]["decision"] == "reuse"
    assert dependencies["local_lns_fixture"]["decision"] == "reuse"
    assert dependencies["external_neurosat_runtime"]["decision"] == "reject"
    assert dependencies["extropic_sdk_or_device"]["decision"] == "reject"
    assert artifact["aggregate_row_recomputation"]["new_runtime_dependency_count"] == 0


def test_scenario_report_6503_gate_artifact_and_per_unit_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6503-GATE/SCHEMA: rows recompute readiness."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    gate = artifact["upstream_gate_receipt"]
    aggregate = artifact["aggregate_row_recomputation"]

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_v561_source_delta_method_contract"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_v561_source_delta_method_contract")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert gate["path"] == mod.UPSTREAM_GATE_RELATIVE_PATH.as_posix()
    assert gate["field"] == "v561_lineage_lock_ready_score"
    assert gate["expected_value"] == 1.0
    assert gate["observed_value"] == 1.0
    assert gate["sha256"].startswith("sha256:")
    assert gate["network_state"]["network_used"] is True
    assert artifact["preconditions_checked"]["upstream_gate_receipt"] == gate
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["gate_check_summary"]["failed_checks"] == []

    assert aggregate["source_receipt_count"] == len(mod.SOURCE_MANIFEST)
    assert aggregate["promoted_method_count"] == 4
    assert aggregate["promoted_methods_with_local_tests"] == 4
    assert aggregate["product_claim_rows"] == 2
    assert aggregate["product_claims_with_no_local_evidence_boundary"] == 2
    assert aggregate["method_contract_ready_score_from_rows"] == 1.0

    row_types = {row["row_type"] for row in artifact["per_unit_rows"]}
    assert {"source", "method", "control", "metric", "boundary", "dependency"} <= row_types
    assert len(artifact["source_delta_rows"]) == len(mod.SOURCE_MANIFEST)
    assert {row["delta_status"] for row in artifact["source_delta_rows"]} >= {
        "unchanged_pinned",
        "unavailable_local_evidence",
    }


def test_scenario_report_6503_validation_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-6503-SCHEMA: malformed artifacts fail closed."""

    artifact = _artifact(tmp_path / "clean")
    assert mod.tests_run_receipts(TESTS_RUN) == TESTS_RUN
    assert all(row["exit_code"] == 0 for row in mod.tests_run_receipts(None))
    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    assert mod._extract_title("<h1>Title:Example</h1>", "fallback") == "Example"
    assert mod._extract_title('{"title": "JSON Title"}', "fallback") == "JSON Title"
    assert (
        mod._extract_title('{"content": {"title": {"value": "Nested Title"}}}', "fallback")
        == "Nested Title"
    )
    assert mod._extract_title("{bad", "fallback") == "fallback"
    assert mod._retrieval_state(403, "Challenge verification required") == "blocked_challenge"
    assert mod._retrieval_state(429, "Too Many Requests") == "rate_limited"
    assert mod._retrieval_state(404, "Paper not found") == "not_indexed"
    assert mod._retrieval_state(200, "") == "available"
    assert mod._retrieval_state(500, "server error") == "blocked"
    assert mod._network_probe(lambda _url, _source_id: _receipt(_url, "ok"))["network_available"] is True
    assert mod._network_probe(lambda _url, _source_id: _receipt(_url, "bad", error="boom"))[
        "network_available"
    ] is False

    original_import = builtins.__import__

    def _no_z3_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "z3":
            raise ImportError("z3 hidden for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_z3_import)
    assert mod.dependency_decision_rows(REPO)[0]["local_available"] is False
    monkeypatch.setattr(builtins, "__import__", original_import)

    bad = deepcopy(artifact)
    del bad["status"]
    _with_checksum(bad)
    assert any("missing required fields" in error for error in mod.validate_artifact(bad))

    bad = deepcopy(artifact)
    bad["unexpected"] = True
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["verdict_class"] = "bad"
    bad["inference_substrate"] = "wrong"
    bad["verifier_is_oracle"] = True
    _with_checksum(bad)
    errors = mod.validate_artifact(bad)
    assert any("unexpected fields" in error for error in errors)
    assert "field_principles must cover exactly required fields" in errors
    assert "field_provenance must cover exactly required fields" in errors
    assert "verdict_class outside closed enum" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors

    bad = deepcopy(artifact)
    bad["method_contract_ready_score"] = 0.0
    bad["gate_check_summary"]["all_gates_passed"] = True
    _with_checksum(bad)
    assert "ready score and gate summary disagree" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "complete: wrong prefix"
    _with_checksum(bad)
    assert "honest_verdict lacks accepted Exp6503 prefix" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["source_receipt_rows"] = []
    _with_checksum(bad)
    assert "source_receipt_rows must cover source manifest" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["method_contract"]["promoted_method_rows"][0]["local_test_target"] = None
    _with_checksum(bad)
    assert "promoted methods must map to local tests" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["authority_boundary"]["learned_advice_can_accept_solution"] = True
    _with_checksum(bad)
    assert "authority boundary must forbid learned acceptance" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["dependency_decision_rows"][0]["decision"] = "add"
    _with_checksum(bad)
    assert "new runtime dependencies are not allowed" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    for row in bad["source_receipt_rows"]:
        if row["source_id"] == "extropic_z1_update":
            row["claim_boundary"] = "product_claim"
    _with_checksum(bad)
    assert "product claims must have no-local-evidence boundary" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:" + "1" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[1]\n", encoding="utf-8")
    assert "artifact must be a JSON object" in mod.validate_artifact(invalid)[0]

    with pytest.raises(ValueError, match="forced validation error"):
        original_validate = mod.validate_artifact
        try:
            mod.validate_artifact = lambda _value: ["forced validation error"]  # type: ignore[method-assign]
            mod.build_artifact(
                repo_root=REPO,
                source_receipt_rows=_source_receipts(),
                write=False,
                duration_s=1.0,
                tests_run=TESTS_RUN,
                access_date="2026-08-22",
            )
        finally:
            mod.validate_artifact = original_validate  # type: ignore[method-assign]

    frozen_rows = _source_receipts()
    monkeypatch.setattr(mod, "collect_source_receipts", lambda **_kwargs: frozen_rows)
    rc = mod.main(["--date", "20260822", "--output", str(tmp_path / "main.json")])
    assert rc == 0
    written = json.loads((tmp_path / "main.json").read_text())
    assert written["method_contract_ready_score"] == 1.0
