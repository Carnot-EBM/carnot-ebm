"""Tests for Exp6483 V559 latent-energy SOTA ingestion.

Spec refs: REQ-INFRA-6483, SCENARIO-INFRA-6483-SOURCE-IDENTITY,
SCENARIO-INFRA-6483-CITATION-VALIDITY,
SCENARIO-INFRA-6483-METHOD-MAPPING,
SCENARIO-INFRA-6483-NO-EXECUTION, SCENARIO-INFRA-6483-ROWS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6483_v559_latent_energy_sota_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _with_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
        output_root=tmp_path,
    )


def _fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in (
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "research-references.md",
        "research-studying.md",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "docs/research-notes/search-layer-literature-2026-06-11.md",
        "scripts/sweep_clusters.py",
        "scripts/sweep_semscholar.py",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        mod.SPEC_RELATIVE_PATH.as_posix(),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    return root


def test_req_infra_6483_spec_declares_source_map_contract() -> None:
    """REQ-INFRA-6483: the OpenSpec section owns the source-map shape."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6483") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6483-SOURCE-IDENTITY",
        "SCENARIO-INFRA-6483-CITATION-VALIDITY",
        "SCENARIO-INFRA-6483-METHOD-MAPPING",
        "SCENARIO-INFRA-6483-NO-EXECUTION",
        "SCENARIO-INFRA-6483-ROWS",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.NOTE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_infra_6483_source_identity_rows_cover_required_areas() -> None:
    """SCENARIO-INFRA-6483-SOURCE-IDENTITY: primary rows keep exact identity."""

    rows = mod.primary_source_rows()
    assert len(rows) >= 5
    assert len({row["source_id"] for row in rows}) == len(rows)
    areas = {row["relevance_area"] for row in rows}
    assert {
        "ebm_verification",
        "neural_constraints",
        "probabilistic_hardware",
        "hallucination_detection",
        "kan",
        "energy_guided_decisions",
        "continual_constraint_learning",
    } <= areas

    for row in rows:
        assert row["url"].startswith("https://")
        assert row["date"]
        assert row["title"]
        assert row["query_route"]
        assert row["citation_validity"]["resolvable_url"] is True
        assert row["citation_validity"]["source_class"] == "primary"
        assert row["claim_boundary"]
        assert row["checked_utc"] <= mod.SOURCE_CUTOFF_UTC
        assert row["execution_claim"] is False


def test_scenario_infra_6483_citation_validity_rows_do_not_invent_counts() -> None:
    """SCENARIO-INFRA-6483-CITATION-VALIDITY: secondary rows preserve limits."""

    rows = mod.secondary_source_rows()
    surfaces = {row["surface"] for row in rows}
    assert {
        "Semantic Scholar EBT",
        "Semantic Scholar ARM-EBM",
        "OpenReview",
        "Hugging Face Papers",
        "GitHub",
        "Extropic",
        "Logical Intelligence",
    } <= surfaces
    for row in rows:
        assert row["url"].startswith("https://")
        assert row["checked_utc"] <= mod.SOURCE_CUTOFF_UTC
        assert row["execution_claim"] is False
        assert row["citation_count_policy"] in {"observed_count", "not_applicable", "not_returned"}
        if row["citation_count_policy"] == "observed_count":
            assert isinstance(row["observed_citation_count"], int)
            assert row["observed_citation_count"] >= 0
        else:
            assert row["observed_citation_count"] is None


def test_scenario_infra_6483_methods_map_to_current_falsifiable_tests() -> None:
    """SCENARIO-INFRA-6483-METHOD-MAPPING: selected methods map to code."""

    primary_ids = {row["source_id"] for row in mod.primary_source_rows()}
    rows = mod.method_mapping_rows()
    assert 3 <= len(rows) <= 5
    for row in rows:
        assert row["source_id"] in primary_ids
        assert row["source_url"].startswith("https://")
        assert (REPO / row["current_carnot_surface"]).exists()
        assert row["expected_test"].startswith("tests/python/")
        assert row["failure_boundary"]
        assert row["retired_scope_risk"]
        assert row["candidate_next_task"].startswith("Exp")
        assert row["execution_claim"] is False


def test_scenario_infra_6483_no_execution_artifact_and_aggregates(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6483-NO-EXECUTION and ROWS: artifact validates from rows."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
        output_root=tmp_path,
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["no_execution_claim"] is True
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"]
    )
    assert artifact["aggregate_row_recomputation"]["primary_source_count"] >= 5
    assert artifact["aggregate_row_recomputation"]["method_mapping_count"] >= 3
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    bad = deepcopy(artifact)
    bad["no_execution_claim"] = False
    assert "no_execution_claim must be true" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["method_mapping_rows"] = bad["method_mapping_rows"][:2]
    bad["per_unit_rows"] = [
        row for row in bad["per_unit_rows"] if row.get("row_type") != "method_mapping"
    ][:2] + [row for row in bad["per_unit_rows"] if row.get("row_type") != "method_mapping"]
    assert "method mapping gate failed" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "live_model_execution"
    assert "inference_substrate mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = True
    assert "verifier_is_oracle must be false for paper claims" in mod.validate_artifact(
        _with_checksum(bad)
    )


def test_scenario_infra_6483_materializes_note_ledger_and_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-INFRA-6483-ROWS: note, ledger, and JSON render from rows."""

    monkeypatch.delenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", raising=False)
    artifact = _artifact(tmp_path)
    assert mod.utc_now_iso().endswith("Z")
    assert mod.default_tests_run() == _passing_tests()

    note = mod.research_note_text(artifact)
    assert "## Source Rows" in note
    assert "## Method Map" in note
    assert artifact["primary_source_rows"][0]["source_id"] in note

    block = mod.ledger_block(artifact)
    assert "EXP6483-V559-LATENT-ENERGY-SOTA-INGESTION-START" in block
    assert mod.replace_or_append_marked_block(
        "alpha\n",
        block,
        start_marker="<!-- EXP6483-V559-LATENT-ENERGY-SOTA-INGESTION-START -->",
        end_marker="<!-- EXP6483-V559-LATENT-ENERGY-SOTA-INGESTION-END -->",
    ).endswith(block)
    replaced = mod.replace_or_append_marked_block(
        "alpha\n<!-- EXP6483-V559-LATENT-ENERGY-SOTA-INGESTION-START -->\nold\n"
        "<!-- EXP6483-V559-LATENT-ENERGY-SOTA-INGESTION-END -->\nomega\n",
        block,
        start_marker="<!-- EXP6483-V559-LATENT-ENERGY-SOTA-INGESTION-START -->",
        end_marker="<!-- EXP6483-V559-LATENT-ENERGY-SOTA-INGESTION-END -->",
    )
    assert "old" not in replaced
    assert "omega" in replaced

    root = _fake_repo(tmp_path)
    materialized = mod.materialize_research_outputs(root, artifact)
    assert materialized["study_ledger_changed"] is True
    assert Path(materialized["note_path"]).read_text(encoding="utf-8") == note
    assert "INGESTED" in Path(materialized["study_ledger_path"]).read_text(encoding="utf-8")
    materialized_again = mod.materialize_research_outputs(root, artifact)
    assert materialized_again["study_ledger_changed"] is False

    result_path = tmp_path / "artifact.json"
    written = mod.write_artifact(artifact, result_path)
    assert written == result_path
    assert mod.validate_artifact(written) == []
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert mod.validate_artifact(tmp_path / "missing.json") == ["artifact missing"]


def test_scenario_infra_6483_validation_reports_schema_and_row_errors(tmp_path: Path) -> None:
    """REQ-INFRA-6483: invalid source and mapping rows fail explicitly."""

    artifact = _artifact(tmp_path)
    missing = deepcopy(artifact)
    missing.pop("status")
    assert mod.validate_artifact(missing) == ["missing required field: status"]

    bad = deepcopy(artifact)
    bad["primary_source_rows"] = bad["primary_source_rows"][:4]
    assert "primary source gate failed" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["primary_source_rows"][0]["url"] = "http://example.invalid/source"
    assert "primary source URL is not resolvable: arxiv_2608_20337" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["primary_source_rows"][0]["citation_validity"]["source_class"] = "secondary"
    assert "primary source class mismatch: arxiv_2608_20337" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["primary_source_rows"][0]["claim_boundary"] = ""
    assert "missing claim boundary: arxiv_2608_20337" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["primary_source_rows"][0]["execution_claim"] = True
    assert "primary_source_rows contains execution claim: arxiv_2608_20337" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["secondary_source_rows"][0]["observed_citation_count"] = None
    assert (
        "observed citation count missing: semantic_scholar_ebt_2507_02092"
        in mod.validate_artifact(_with_checksum(bad))
    )

    bad = deepcopy(artifact)
    bad["secondary_source_rows"][2]["observed_citation_count"] = 7
    assert "citation count invented: openreview_ebt_page" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["method_mapping_rows"][0]["source_id"] = "missing_source"
    assert "mapping source missing: m1_anytime_valid_cache_promotion" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["method_mapping_rows"][0]["failure_boundary"] = ""
    assert (
        "mapping missing failure_boundary: m1_anytime_valid_cache_promotion"
        in mod.validate_artifact(_with_checksum(bad))
    )


def test_scenario_infra_6483_validation_reports_aggregate_and_metadata_errors(
    tmp_path: Path,
    ) -> None:
    """SCENARIO-INFRA-6483-NO-EXECUTION: reducers catch summary drift."""

    artifact = _artifact(tmp_path)

    bad = deepcopy(artifact)
    bad["aggregate_row_recomputation"] = {}
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["gate_check_summary"] = {}
    assert "gate_check_summary mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["status"] = "blocked_manual_check"
    assert "blocked status requires gate_check_summary failed_gates" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["per_unit_rows"] = []
    aggregates = mod.recompute_aggregates_from_rows(bad["per_unit_rows"])
    bad["aggregate_row_recomputation"] = aggregates
    bad["gate_check_summary"] = mod.gate_check_summary(aggregates)
    assert "complete status with failed gates" in mod.validate_artifact(_with_checksum(bad))

    for field, message in (
        ("field_principles", "field_principles must cover exactly required fields"),
        ("field_provenance", "field_provenance must cover exactly required fields"),
    ):
        bad = deepcopy(artifact)
        bad[field] = {}
        assert message in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["source_cutoff_utc"] = "2026-08-21T00:00:00Z"
    assert "source_cutoff_utc mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["random_seed"] = 1
    assert "random_seed mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "maybe"
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["primary_source_rows"][0]["title"] = "changed title"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)


def test_scenario_infra_6483_run_and_cli_paths_are_bounded(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """SCENARIO-INFRA-6483-NO-EXECUTION: command paths write only source artifacts."""

    monkeypatch.delenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", raising=False)
    root = _fake_repo(tmp_path)
    result_path = tmp_path / "result.json"
    artifact = mod.run(
        date="20260821",
        result_path=result_path,
        root=root,
        tests_run=_passing_tests(),
    )
    assert artifact["status"] == "complete"
    assert result_path.exists()
    assert (root / mod.NOTE_RELATIVE_PATH).exists()
    assert mod.validate_artifact(result_path) == []

    try:
        mod.run(date="20260820", result_path=tmp_path / "wrong.json", root=root)
    except ValueError as exc:
        assert "expected --date 20260821" in str(exc)
    else:  # pragma: no cover - the assertion above must raise
        raise AssertionError("wrong date did not fail")

    original_build_artifact = mod.build_artifact

    def invalid_artifact(**kwargs: Any) -> dict[str, Any]:
        built = original_build_artifact(**kwargs)
        built["no_execution_claim"] = False
        built["reproducibility_checksum"] = mod.payload_checksum(built)
        return built

    monkeypatch.setattr(mod, "build_artifact", invalid_artifact)
    try:
        mod.run(date="20260821", result_path=tmp_path / "invalid.json", root=root)
    except ValueError as exc:
        assert "no_execution_claim must be true" in str(exc)
    else:  # pragma: no cover - the invalid artifact must raise
        raise AssertionError("invalid artifact did not fail")
    monkeypatch.setattr(mod, "build_artifact", original_build_artifact)

    monkeypatch.setattr(mod, "find_repo_root", lambda start=None: root)
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert mod.main(["--validate", "--result-path", str(tmp_path / "absent.json")]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False

    cli_result = tmp_path / "cli-result.json"
    assert mod.main(["--date", "20260821", "--result-path", str(cli_result)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "complete"
    assert Path(printed["result_path"]) == cli_result
    assert cli_result.exists()
