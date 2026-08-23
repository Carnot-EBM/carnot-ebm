"""Tests for Exp6515 V564 source-method contract.

Spec refs: REQ-REPORT-6515, SCENARIO-REPORT-6515-SOURCES,
SCENARIO-REPORT-6515-METHODS, SCENARIO-REPORT-6515-AUTHORITY,
SCENARIO-REPORT-6515-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6515_v564_source_method_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6515_v564_source_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6515_v564_source_method_contract.py "
    "-m pytest tests/python/test_experiment_6515_v564_source_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6515_v564_source_method_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6515_v564_source_method_contract.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6515_v564_source_method_contract "
    "--date 20260823"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6515_v564_source_method_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6515_v564_source_method_contract --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
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


def _arxiv_body(source_id: str) -> str:
    source = mod.SOURCE_BY_ID[source_id]
    source_date = source["expected_source_date"]
    title = source["title"]
    return (
        "<html>"
        f"<title>{title}</title>"
        f"<h1>Title:{title}</h1>"
        f"<div class='dateline'>[Submitted on {source_date}]</div>"
        f"<div class='submission-history'>[v1] Thu, {source_date} 12:00:00 UTC</div>"
        "<blockquote class='abstract mathjax'>Abstract: fixture abstract.</blockquote>"
        "</html>"
    )


def _fake_fetcher(url: str, source_id: str) -> mod.JsonDict:
    if source_id == "network_probe":
        return _receipt(url, "ok")
    if source_id in mod.SOURCE_BY_ID and mod.SOURCE_BY_ID[source_id]["source_kind"] == "paper":
        return _receipt(url, _arxiv_body(source_id))
    if source_id == "openreview_solver_advice":
        return _receipt(url, json.dumps({"notes": [{"id": "fixture"}]}))
    if source_id == "semantic_scholar_ebt":
        return _receipt(url, json.dumps({"citationCount": 35, "paperId": "fixture-ebt"}))
    if source_id == "semantic_scholar_arm_ebm":
        return _receipt(url, json.dumps({"message": "Too Many Requests"}), status_code=429, error="HTTP 429")
    if source_id == "huggingface_papers_task_coevolve":
        return _receipt(url, json.dumps({"id": "2608.20169", "title": "Task-CoEvolve"}))
    if source_id == "extropic_z1_status":
        return _receipt(url, "<html><title>From One to One Billion</title>Z1 early access 2027</html>")
    if source_id == "github_ferrotherm":
        return _receipt(url, "<html><title>GitHub - dcharlot-physicalai-bmi/ferrotherm</title></html>")
    if source_id == "github_dibs":
        return _receipt(url, "<html><title>GitHub - shanxierdan/DiBS</title></html>")
    if source_id == "kona_product_page":
        return _receipt(url, "<html><title>Kona: Energy-Based Models for AI Reasoning</title></html>")
    raise AssertionError(f"unexpected fetch {source_id}: {url}")


def _blocked_fetcher(url: str, source_id: str) -> mod.JsonDict:
    if source_id == "task_coevolve":
        return _receipt(url, "Service Unavailable", status_code=503, error="HTTP 503")
    return _fake_fetcher(url, source_id)


def _fake_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
    joined = " ".join(args)
    assert cwd == REPO
    if "sweep_clusters.py" in joined:
        return 0, "http://export.arxiv.org/api/query?search_query=fixture\n", ""
    if "sweep_semscholar.py" in joined:
        return 0, "2608.20169\n2608.20053\n", "# semscholar: fixture\n"
    raise AssertionError(f"unexpected command: {joined}")


def _artifact(tmp_path: Path, *, blocked: bool = False) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        run_date="20260823",
        fetcher=_blocked_fetcher if blocked else _fake_fetcher,
        command_runner=_fake_runner,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_report_6515_spec_declares_source_method_contract() -> None:
    """REQ-REPORT-6515: OpenSpec owns the Exp6515 artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6515") :]
    normalized = " ".join(section.split())

    for token in (
        "SCENARIO-REPORT-6515-SOURCES",
        "SCENARIO-REPORT-6515-METHODS",
        "SCENARIO-REPORT-6515-AUTHORITY",
        "SCENARIO-REPORT-6515-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle=false`",
        "`verdict_class` SHALL be `null` or `partial`",
    ):
        assert token in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6515_sources_verify_primary_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6515-SOURCES: source rows carry boundaries and hashes."""

    artifact = _artifact(tmp_path)
    rows = {row["source_id"]: row for row in artifact["source_rows"]}

    assert len(rows) == len(mod.SOURCE_MANIFEST)
    assert rows["task_coevolve"]["title"] == (
        "Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection"
    )
    assert rows["task_coevolve"]["source_date"] == "2026-08-20"
    assert rows["task_coevolve"]["primary_url_verified"] is True
    assert rows["safety_nets"]["available_code_or_data"] == "open_source_implementation_claimed"
    assert rows["learned_conflicts"]["method"] == "refinement_checked_learned_conflict_reuse"
    assert rows["dibs"]["available_code_or_data"] == "public_repository_without_data_or_checkpoint"
    assert rows["nested_smc"]["method_transfer_status"] == "deferred_non_autoregressive_decoder_control"
    assert rows["chainforge"]["non_transferable_claim"] == "No quantum annealer speed, latency, power, or fidelity claim transfers."
    assert rows["extropic_z1_status"]["source_kind"] == "product"
    assert rows["kona_product_page"]["exact_authority_boundary"] == "Product comparator only; no local runner or weights."
    assert rows["github_ferrotherm"]["source_kind"] == "repository"

    for row in artifact["source_rows"]:
        assert row["source_hash"].startswith("sha256:")
        assert row["carnot_hook"]
        assert row["non_transferable_claim"]
        assert row["exact_authority_boundary"]
        assert row["claimed_evidence"]


def test_scenario_report_6515_methods_authority_and_schema(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6515-METHODS/AUTHORITY/SCHEMA: rows recompute readiness."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    contract = artifact["frozen_method_contract"]
    mappings = {row["source_id"]: row for row in artifact["sota_to_experiment_rows"]}
    aggregate = artifact["aggregate_row_recomputation"]

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_v564_source_method_contract_ready"
    assert artifact["honest_verdict"].startswith("complete_v564_source_method_contract_ready")
    assert artifact["verdict_class"] is None
    assert artifact["v564_method_contract_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    assert contract["feature_contract"]["allowed_features"] == [
        "static_graph_features",
        "partial_assignment_consistency",
        "conflict_pressure",
        "lineage_and_shift_tags",
    ]
    assert contract["control_contract"]["negative_controls"] == [
        "native_dynamic_branching",
        "random_order",
        "fixed_subset_validation",
        "scratch_no_memory",
        "invalid_conflict_veto",
    ]
    assert contract["exact_fallback_contract"]["solver_acceptance_authority"] == "native_exact_solver"
    assert contract["adaptive_sampling_contract"]["release_authority"] == "full_held_audit_only"
    assert contract["exception_table_contract"]["lookup_table_is_authority"] is False
    assert contract["conflict_witness_contract"]["admission_gate"] == "proved_query_refinement_witness"
    assert contract["mapping_cost_contract"]["hardware_claim_allowed"] is False
    assert contract["stop_rule_contract"]["stop_before_outcome_artifact_review"] is True

    assert mappings["task_coevolve"]["target_experiment"] == "Exp6523"
    assert mappings["safety_nets"]["target_experiment"] == "Exp6520"
    assert mappings["learned_conflicts"]["target_experiment"] == "Exp6521-Exp6522"
    assert mappings["dibs"]["target_experiment"] == "Exp6518"
    assert mappings["chainforge"]["target_experiment"] == "Exp6516-Exp6523"
    assert all(row["negative_control"] for row in artifact["sota_to_experiment_rows"])
    assert all(row["falsifiable_metric"] for row in artifact["sota_to_experiment_rows"])
    assert all(row["exact_authority_boundary"] for row in artifact["non_transfer_rows"])

    assert aggregate["required_source_count"] == len(mod.SOURCE_MANIFEST)
    assert aggregate["source_rows_cover_manifest"] is True
    assert aggregate["required_primary_sources_verified"] is True
    assert aggregate["adopted_method_count"] == len(mod.ADOPTED_SOURCE_IDS)
    assert aggregate["adopted_methods_with_source_mapping_control_boundary"] == len(mod.ADOPTED_SOURCE_IDS)
    assert aggregate["ready_score_from_rows"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["preconditions_checked"]["no_high_concurrency_deep_research_harness"] is True
    assert artifact["preconditions_checked"]["outcome_artifact_guard"]["outcome_artifacts_read"] == []
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert {row["row_type"] for row in artifact["per_unit_rows"]} >= {
        "source",
        "mapping",
        "non_transfer",
        "contract",
        "gate",
    }


def test_scenario_report_6515_blocked_source_is_partial_not_fabricated(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6515-SCHEMA: source failure forces partial verdict."""

    artifact = _artifact(tmp_path, blocked=True)
    rows = {row["source_id"]: row for row in artifact["source_rows"]}
    failures = artifact["gate_check_summary"]["failed_checks"]

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked_v564_source_method_contract"
    assert artifact["honest_verdict"].startswith("blocked_v564_source_method_contract")
    assert artifact["verdict_class"] == "partial"
    assert artifact["v564_method_contract_ready_score"] == 0.0
    assert rows["task_coevolve"]["primary_url_verified"] is False
    assert rows["task_coevolve"]["observed_error"] == "HTTP 503"
    assert rows["task_coevolve"]["source_date_verified"] is False
    assert failures
    assert any(item["channel"] == "arxiv_primary" for item in failures)
    assert artifact["primary_source_hashes"]["task_coevolve"].startswith("sha256:")


def test_scenario_report_6515_validation_edges(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6515-SCHEMA: malformed artifacts fail closed."""

    clean_root = tmp_path / "clean"
    artifact = _artifact(clean_root)
    assert mod.tests_run_receipts(TESTS_RUN) == TESTS_RUN
    assert all(row["exit_code"] == 0 for row in mod.tests_run_receipts(None))
    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    assert mod._extract_title('{"title": "JSON Title"}', "fallback") == "JSON Title"
    assert mod._extract_title("<title>HTML Title</title>", "fallback") == "HTML Title"
    assert mod._extract_title("<h1>Title:Arxiv Title</h1>", "fallback") == "Arxiv Title"
    assert mod._extract_title("{bad", "fallback") == "fallback"
    assert mod._extract_source_date("[Submitted on 20 Aug 2026]") == "2026-08-20"
    assert mod._extract_source_date("[v1] Thu, 12 Mar 2026 17:52:12 UTC") == "2026-03-12"
    assert mod._extract_source_date("no date") is None
    assert mod._retrieval_state(200, "") == "available"
    assert mod._retrieval_state(429, "Too Many Requests") == "rate_limited"
    assert mod._retrieval_state(404, "not found") == "not_found"
    assert mod._retrieval_state(503, "server") == "blocked"
    assert mod._citation_count(json.dumps({"citationCount": 7})) == 7
    assert mod._citation_count("{bad") is None
    assert mod._network_probe(_fake_fetcher, "2026-08-23T12:00:00Z")["network_available"] is True
    assert mod._network_probe(
        lambda url, sid: _receipt(url, "bad", status_code=503, error="boom"),
        "2026-08-23T12:00:00Z",
    )["network_available"] is False

    bad = deepcopy(artifact)
    del bad["status"]
    _with_checksum(bad)
    assert any("required field set mismatch" in error for error in mod.validate_artifact(bad))

    bad = deepcopy(artifact)
    bad["unexpected"] = True
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["verdict_class"] = "positive"
    bad["inference_substrate"] = "live_llm"
    bad["verifier_is_oracle"] = True
    _with_checksum(bad)
    errors = mod.validate_artifact(bad)
    assert any("required field set mismatch" in error for error in errors)
    assert "field_principles mismatch" in errors
    assert "field_provenance must cover required fields" in errors
    assert "verdict_class outside Exp6515 enum" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors

    mutations = [
        ("ready score mismatch", lambda item: item.__setitem__("v564_method_contract_ready_score", 0.0)),
        ("source rows must cover manifest", lambda item: item.__setitem__("source_rows", [])),
        (
            "required primary sources must verify",
            lambda item: item["source_rows"][0].__setitem__("primary_url_verified", False),
        ),
        (
            "adopted methods must map to local implementation controls",
            lambda item: item["sota_to_experiment_rows"][0].__setitem__("implementable_local_mapping", ""),
        ),
        (
            "non-transfer rows must forbid unsupported transfer",
            lambda item: item["non_transfer_rows"][0].__setitem__("non_transferable_claim", ""),
        ),
        (
            "learned advice cannot certify or prune",
            lambda item: item["frozen_method_contract"]["authority_contract"].__setitem__(
                "learned_advice_may_prune", True
            ),
        ),
        (
            "protected files changed",
            lambda item: item["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
        ),
        (
            "query receipts must include sweep helper and sequential channels",
            lambda item: item.__setitem__("query_receipts", []),
        ),
        (
            "primary_source_hashes must cover manifest",
            lambda item: item.__setitem__("primary_source_hashes", {}),
        ),
    ]
    for expected, mutate in mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        _with_checksum(broken)
        assert expected in mod.validate_artifact(broken)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    assert mod.main(["--validate", "--result-path", str(clean_root / mod.RESULT_RELATIVE_PATH.name)]) == 0
    assert mod.main(
        [
            "--date",
            "20260823",
            "--result-path",
            str(tmp_path / "cli.json"),
            "--no-live-network",
        ]
    ) == 0
    assert "experiment_6515_v564_source_method_contract.json" in capsys.readouterr().out

    invalid_json_path = tmp_path / "invalid.json"
    invalid_json_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        mod._load_json(invalid_json_path)

    bad_path = tmp_path / "bad_artifact.json"
    broken = deepcopy(artifact)
    del broken["status"]
    _with_checksum(broken)
    bad_path.write_text(json.dumps(broken), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_path)]) == 1

    monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: {"bad": True})
    assert mod.main(["--date", "20260823", "--result-path", str(tmp_path / "bad_cli.json")]) == 1
