"""Tests for Exp5456 guided-decoding tautology corrigendum.

Spec refs: REQ-SAFE-5456, SCENARIO-SAFE-5456.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_5456_guided_decoding_tautology_corrigendum_v496 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5456_guided_decoding_tautology_corrigendum_v496.py -q"
)


def _prior_artifact() -> dict[str, Any]:
    return json.loads((REPO / mod.PRIOR_FLAGGED_ARTIFACT).read_text(encoding="utf-8"))


def _prior_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (REPO / mod.PRIOR_ROW_RESULTS).read_text(encoding="utf-8").splitlines()
    ]


def test_req_safe_5456_spec_declares_guided_corrigendum_contract() -> None:
    """REQ-SAFE-5456: OpenSpec anchors the row-metric audit contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-SAFE-5456") : spec.index("## Implementation Status")
    ]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5456",
        "SCENARIO-SAFE-5456",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.PRIOR_FLAGGED_ARTIFACT),
        str(mod.PRIOR_ROW_RESULTS),
        "guided reward counts",
        "grammar-only baselines",
        "exact final verifier labels",
        "invalid-tautological",
        "metric_independence_checks_passed",
        "verifier_guided_decoding_ready",
        "scripts/research_conductor.py",
        "posthoc_row_metric_audit_no_llm",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_safe_5456_run_writes_clean_corrigendum_and_graph(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5456: corrigendum recomputes readiness from row evidence only."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    graph_path = tmp_path / mod.GRAPH_RELATIVE_PATH
    artifact = mod.run(
        root=REPO,
        result_path=result_path,
        graph_path=graph_path,
        tests_run=[TEST_COMMAND],
        write=True,
    )
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    report = adversarial_verify.verify_artifact(result_path)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact, root=tmp_path, require_graph_file=True)
    assert {flag["kind"] for flag in report["flags"] if flag["severity"] == "critical"} == set()
    assert artifact["prior_flagged_artifact"] == str(mod.PRIOR_FLAGGED_ARTIFACT)
    assert artifact["prior_flagged_adversarial"] is True
    assert [flag["kind"] for flag in artifact["adversarial_flags_found"]] == [
        "TAUTOLOGY",
        "TAUTOLOGY",
    ]
    assert artifact["invalid_tautological_fields"] == [
        "abstention_rate",
        "action_unreachability_rate",
        "guided_validity_delta_vs_unconstrained",
        "semantic_false_accept_rate",
    ]
    assert artifact["recomputed_row_count"] == 12
    assert artifact["exact_final_labels_used"] is True
    assert artifact["guided_decoding_corrigendum_clean"] is True
    assert artifact["guided_decoding_ready_from_independent_rows"] is False
    assert artifact["inference_substrate"] == "posthoc_row_metric_audit_no_llm"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "fresh non-tautological guided-decoding rerun" in artifact["rerun_gate_reason"]

    audit = artifact["row_metric_audit"]
    assert audit["condition_validity"]["unconstrained"]["valid_count"] == 1
    assert audit["condition_validity"]["grammar_only"]["valid_count"] == 4
    assert audit["condition_validity"]["verifier_potential_guided"]["valid_count"] == 2
    assert audit["guided_delta_audit"]["vs_unconstrained"]["rate_delta"] == pytest.approx(0.25)
    assert audit["guided_delta_audit"]["vs_grammar_only"]["rate_delta"] == pytest.approx(-0.5)
    assert audit["guided_condition_false_accept_count"] == 2

    assert graph["nodes"]["prior.abstention_rate"]["classification"] == "invalid-tautological"
    assert graph["nodes"]["prior.semantic_false_accept_rate"]["classification"] == (
        "invalid-tautological"
    )
    assert graph["nodes"]["corrected.guided_delta_vs_grammar_only"]["classification"] == (
        "derived-from-independent"
    )
    assert graph["readiness_dependencies_clean"] is True
    assert set(artifact["independent_metric_fields"]).issuperset(
        {
            "row.condition",
            "row.exact_final_verdict.accepted",
            "corrected.condition_validity",
            "corrected.guided_delta_vs_grammar_only",
        }
    )


def test_req_safe_5456_recomputes_row_metrics_without_prior_scalars() -> None:
    """REQ-SAFE-5456: row audit uses exact final labels, not Exp5444 readiness."""

    metrics = mod.recompute_row_metric_audit(_prior_rows())
    prior = _prior_artifact()
    flags = mod.extract_tautology_flags(prior)
    invalid = mod.invalid_fields_from_tautology_flags(flags)
    graph = mod.build_metric_dependency_graph(
        prior_artifact=prior,
        rows=_prior_rows(),
        row_audit=metrics,
        adversarial_flags=flags,
        invalid_tautological_fields=invalid,
    )

    assert metrics["exact_final_labels_used"] is True
    assert metrics["prior_scalar_fields_used"] == []
    assert metrics["condition_validity"]["verifier_potential_guided"]["rate"] == pytest.approx(0.5)
    assert metrics["condition_validity"]["grammar_only"]["rate"] == pytest.approx(1.0)
    assert metrics["guided_ready_from_rows"] is False
    assert "prior.verifier_guided_decoding_ready" not in graph["readiness_dependencies"]
    assert "prior.metric_independence_checks_passed" not in graph["readiness_dependencies"]
    assert mod.audit_readiness_dependencies(graph) == []


def test_req_safe_5456_regression_rejects_self_validating_readiness_dependency() -> None:
    """REQ-SAFE-5456: readiness cannot depend on the scalar it claims to validate."""

    graph = mod.build_metric_dependency_graph(
        prior_artifact=_prior_artifact(),
        rows=_prior_rows(),
        row_audit=mod.recompute_row_metric_audit(_prior_rows()),
        adversarial_flags=mod.extract_tautology_flags(_prior_artifact()),
        invalid_tautological_fields=[
            "abstention_rate",
            "guided_validity_delta_vs_unconstrained",
        ],
    )
    bad = deepcopy(graph)
    bad["nodes"]["corrected.guided_decoding_ready_from_rows"]["depends_on"].append(
        "prior.verifier_guided_decoding_ready"
    )
    bad["readiness_dependencies"].append("prior.verifier_guided_decoding_ready")

    assert "forbidden readiness dependency" in "; ".join(mod.audit_readiness_dependencies(bad))

    self_dep = deepcopy(graph)
    self_dep["nodes"]["corrected.guided_decoding_ready_from_rows"]["depends_on"].append(
        "corrected.guided_decoding_ready_from_rows"
    )
    assert "self-validating readiness dependency" in "; ".join(
        mod.audit_readiness_dependencies(self_dep)
    )


def test_req_safe_5456_validation_fails_closed_on_missing_or_contradictory_inputs(
    tmp_path: Path,
) -> None:
    """REQ-SAFE-5456: blocked or malformed receipts cannot claim a clean corrigendum."""

    artifact = mod.build_artifact(
        prior_artifact=_prior_artifact(),
        rows=_prior_rows(),
        metric_dependency_graph_path=str(tmp_path / "missing_graph.json"),
        tests_run=[TEST_COMMAND],
    )
    mod.validate_artifact(artifact, root=tmp_path, require_graph_file=False)

    missing = deepcopy(artifact)
    missing.pop("prior_flagged_artifact")
    assert "missing required fields" in "; ".join(
        mod.artifact_schema_errors(missing, root=tmp_path, require_graph_file=False)
    )

    bad_prior = deepcopy(artifact)
    bad_prior["prior_flagged_artifact"] = "results/wrong.json"
    assert "prior_flagged_artifact" in "; ".join(
        mod.artifact_schema_errors(bad_prior, root=tmp_path, require_graph_file=False)
    )

    no_graph = deepcopy(artifact)
    no_graph["guided_decoding_corrigendum_clean"] = True
    assert "metric_dependency_graph_path must point to a written graph" in "; ".join(
        mod.artifact_schema_errors(no_graph, root=tmp_path, require_graph_file=True)
    )

    bad_clean = deepcopy(artifact)
    bad_clean["exact_final_labels_used"] = False
    bad_clean["guided_decoding_corrigendum_clean"] = True
    assert "guided_decoding_corrigendum_clean requires exact final labels" in "; ".join(
        mod.artifact_schema_errors(bad_clean, root=tmp_path, require_graph_file=False)
    )

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate" in "; ".join(
        mod.artifact_schema_errors(bad_substrate, root=tmp_path, require_graph_file=False)
    )

    bad_research_conductor = deepcopy(artifact)
    bad_research_conductor["research_conductor_modified"] = True
    assert "scripts/research_conductor.py" in "; ".join(
        mod.artifact_schema_errors(
            bad_research_conductor,
            root=tmp_path,
            require_graph_file=False,
        )
    )

    with pytest.raises(ValueError, match="prior_flagged_artifact"):
        mod.validate_artifact(bad_prior, root=tmp_path, require_graph_file=False)

    malformed_cases: list[tuple[str, Any, str]] = [
        ("adversarial_flags_found", {}, "adversarial_flags_found"),
        ("invalid_tautological_fields", {}, "invalid_tautological_fields"),
        ("recomputed_row_count", -1, "recomputed_row_count"),
        ("exact_final_labels_used", "yes", "exact_final_labels_used"),
        ("guided_decoding_corrigendum_clean", "yes", "guided_decoding_corrigendum_clean"),
        ("independent_metric_fields", {}, "independent_metric_fields"),
        ("rerun_gate_reason", "", "rerun_gate_reason"),
        ("honest_verdict", "done", "honest_verdict"),
        ("field_principles", {}, "field_principles"),
    ]
    for field, value, expected in malformed_cases:
        malformed = deepcopy(artifact)
        malformed[field] = value
        assert expected in "; ".join(
            mod.artifact_schema_errors(malformed, root=tmp_path, require_graph_file=False)
        )

    clean_without_flags = deepcopy(artifact)
    clean_without_flags["adversarial_flags_found"] = []
    clean_without_flags["guided_decoding_corrigendum_clean"] = True
    assert "prior TAUTOLOGY flags" in "; ".join(
        mod.artifact_schema_errors(clean_without_flags, root=tmp_path, require_graph_file=False)
    )

    clean_without_invalid = deepcopy(artifact)
    clean_without_invalid["invalid_tautological_fields"] = []
    clean_without_invalid["guided_decoding_corrigendum_clean"] = True
    assert "requires invalid fields" in "; ".join(
        mod.artifact_schema_errors(clean_without_invalid, root=tmp_path, require_graph_file=False)
    )

    clean_wrong_status = deepcopy(artifact)
    clean_wrong_status["status"] = "blocked"
    clean_wrong_status["guided_decoding_corrigendum_clean"] = True
    assert "requires complete status" in "; ".join(
        mod.artifact_schema_errors(clean_wrong_status, root=tmp_path, require_graph_file=False)
    )

    clean_zero_rows = deepcopy(artifact)
    clean_zero_rows["recomputed_row_count"] = 0
    clean_zero_rows["guided_decoding_corrigendum_clean"] = True
    assert "requires recomputed rows" in "; ".join(
        mod.artifact_schema_errors(clean_zero_rows, root=tmp_path, require_graph_file=False)
    )

    empty_graph_path = deepcopy(artifact)
    empty_graph_path["metric_dependency_graph_path"] = ""
    assert "metric_dependency_graph_path must be a non-empty string" in "; ".join(
        mod.artifact_schema_errors(empty_graph_path, root=tmp_path, require_graph_file=True)
    )

    relative_graph_path = deepcopy(artifact)
    relative_graph_path["metric_dependency_graph_path"] = "bad_graph.json"
    (tmp_path / "bad_graph.json").write_text("{bad", encoding="utf-8")
    assert "metric_dependency_graph_path is unreadable" in "; ".join(
        mod.artifact_schema_errors(relative_graph_path, root=tmp_path, require_graph_file=True)
    )

    _, good_graph = mod._assemble_artifact_and_graph(  # noqa: SLF001
        prior_artifact=_prior_artifact(),
        rows=_prior_rows(),
        metric_dependency_graph_path="graph.json",
        tests_run=[TEST_COMMAND],
    )
    mismatch_graph = deepcopy(good_graph)
    mismatch_graph["invalid_tautological_fields"] = []
    (tmp_path / "mismatch_graph.json").write_text(json.dumps(mismatch_graph), encoding="utf-8")
    graph_mismatch = deepcopy(artifact)
    graph_mismatch["metric_dependency_graph_path"] = "mismatch_graph.json"
    assert "invalid fields must match" in "; ".join(
        mod.artifact_schema_errors(graph_mismatch, root=tmp_path, require_graph_file=True)
    )

    dirty_graph = deepcopy(good_graph)
    dirty_graph["readiness_dependencies_clean"] = False
    (tmp_path / "dirty_graph.json").write_text(json.dumps(dirty_graph), encoding="utf-8")
    graph_dirty = deepcopy(artifact)
    graph_dirty["metric_dependency_graph_path"] = "dirty_graph.json"
    assert "readiness dependencies must be clean" in "; ".join(
        mod.artifact_schema_errors(graph_dirty, root=tmp_path, require_graph_file=True)
    )


def test_req_safe_5456_blocked_artifact_records_precise_blockers() -> None:
    """REQ-SAFE-5456: missing flags, rows, or exact authority block the audit."""

    no_flags = deepcopy(_prior_artifact())
    no_flags["flagged_adversarial"] = False
    no_flags["corrigendum_pending"] = [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}]
    blocked = mod.build_artifact(
        prior_artifact=no_flags,
        rows=[],
        metric_dependency_graph_path=str(mod.GRAPH_RELATIVE_PATH),
        tests_run=[TEST_COMMAND],
    )

    assert blocked["status"] == "blocked"
    assert blocked["guided_decoding_corrigendum_clean"] is False
    assert blocked["adversarial_flags_found"] == []
    assert set(blocked["blockers"]) >= {
        "prior_artifact_not_flagged_adversarial",
        "prior_tautology_flags_missing",
        "row_evidence_missing",
        "exact_final_labels_missing_or_bypassed",
    }
    assert blocked["honest_verdict"].startswith("blocked:")

    bad_rows = deepcopy(_prior_rows())
    bad_rows[0]["exact_final_verdict"]["authority"] = "model_self_verdict"
    blocked_labels = mod.build_artifact(
        prior_artifact=_prior_artifact(),
        rows=bad_rows,
        metric_dependency_graph_path=str(mod.GRAPH_RELATIVE_PATH),
        tests_run=[TEST_COMMAND],
    )
    assert "exact_final_labels_missing_or_bypassed" in blocked_labels["blockers"]

    assert "readiness_dependency_not_independent" in mod._blockers(  # noqa: SLF001
        _prior_artifact(),
        _prior_rows(),
        mod.extract_tautology_flags(_prior_artifact()),
        mod.recompute_row_metric_audit(_prior_rows()),
        {"readiness_dependencies_clean": False},
        ["unit"],
    )


def test_req_safe_5456_helpers_parse_flags_paths_and_cli(tmp_path: Path, capsys: Any) -> None:
    """REQ-SAFE-5456: helpers and CLI keep the deliverable reproducible."""

    write_false = mod.run(
        root=REPO,
        result_path=tmp_path / "no_write.json",
        graph_path=tmp_path / "no_write_graph.json",
        prior_artifact=_prior_artifact(),
        row_records=_prior_rows(),
        write=False,
    )
    assert write_false["guided_decoding_corrigendum_clean"] is True
    assert not (tmp_path / "no_write.json").exists()

    assert mod.audit_readiness_dependencies({}) == [
        "readiness field missing from dependency graph: corrected.guided_decoding_ready_from_rows"
    ]
    assert mod._rerun_gate_reason({"guided_ready_from_rows": True}, []) == (  # noqa: SLF001
        "no rerun required by this audit"
    )
    assert mod._honest_verdict(True, [], {"guided_ready_from_rows": True}) == (  # noqa: SLF001
        "complete: guided-decoding corrigendum clean with row-derived readiness"
    )
    assert mod._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json(bad_json) == {}  # noqa: SLF001
    assert mod._read_jsonl(tmp_path / "missing.jsonl") == []  # noqa: SLF001
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("{bad}\n", encoding="utf-8")
    assert mod._read_jsonl(bad_jsonl) == []  # noqa: SLF001
    assert mod._normalise_tests_run([])[0]["outcome"] == "not_recorded"  # noqa: SLF001
    assert mod._normalise_test_run({"command": "cmd", "outcome": "passed"}) == {  # noqa: SLF001
        "command": "cmd",
        "outcome": "passed",
    }
    assert mod._normalise_test_run("cmd") == {"command": "cmd", "outcome": "not_recorded"}  # noqa: SLF001
    assert mod._destination(tmp_path, None, Path("x.json")) == tmp_path / "x.json"  # noqa: SLF001
    assert mod._destination(tmp_path, tmp_path / "abs.json", Path("x.json")) == (  # noqa: SLF001
        tmp_path / "abs.json"
    )

    result_path = tmp_path / "cli.json"
    graph_path = tmp_path / "cli_graph.json"
    exit_code = mod.main(
        [
            "--root",
            str(REPO),
            "--result-path",
            str(result_path),
            "--graph-path",
            str(graph_path),
        ]
    )
    printed = capsys.readouterr().out
    artifact = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(printed) == artifact
    assert graph_path.exists()
    mod.validate_artifact(artifact, root=tmp_path, require_graph_file=True)
