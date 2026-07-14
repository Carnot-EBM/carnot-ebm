"""Tests for the Exp5706 V510 transition receipt.

Spec refs: REQ-CAPSTONE-5706, SCENARIO-CAPSTONE-5706,
SCENARIO-CAPSTONE-5706-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5706-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5706_transition_v510 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _v509_payloads() -> dict[Path, JsonDict]:
    base: dict[Path, JsonDict] = {
        mod.EXP5636_TRANSITION_PATH: {
            "schema": "carnot.experiment_5636.transition_v509.v1",
            "status": "complete",
            "experiment_id": "exp5636-transition-v509",
            "honest_verdict": "complete: v509 transition loaded",
        },
        mod.EXP5637_SOURCE_PATH: {
            "schema": "carnot.experiment_5637.v509_source_delta_ingestion.v1",
            "status": "complete",
            "experiment_id": "exp5637-v509-source-delta-ingestion",
            "honest_verdict": "complete: source delta",
        },
        mod.EXP5638_SCHEMA_PATH: {
            "schema": "carnot.experiment_5638.fr11_gate_schema_corrigendum.v1",
            "status": "complete",
            "task_id": "exp5638-fr11-gate-schema-corrigendum",
            "honest_verdict": "complete: hash_bound_scalar_gate_contract_ready",
        },
        mod.EXP5639_AUDIT_PATH: {
            "schema": "carnot.experiment_5639.anytime_valid_csl_independent_audit.v1",
            "status": "complete",
            "task_id": "exp5639-anytime-valid-csl-independent-audit",
            "honest_verdict": "complete: anytime_valid_csl_independent_audit_ready",
            "fr11_independent_promotion_ready_score": 1.0,
        },
        mod.EXP5640_SHADOW_PATH: {
            "schema": "carnot.experiment_5640.fr11_shadow_pipeline_integration.v1",
            "status": "complete",
            "experiment_id": "exp5640-fr11-shadow-pipeline-integration",
            "honest_verdict": "complete: fr11_shadow_ready_not_default_enabled",
            "fr11_shadow_integration_ready_score": 1.0,
            "default_enabled": False,
        },
        mod.EXP5641_ARC_MODEL_PATH: {
            "schema": "carnot.exp5641.arc_counterexample_executable_model.v1",
            "status": "blocked",
            "experiment_id": "exp5641-arc-counterexample-executable-model",
            "honest_verdict": "blocked: counterexample_patched_executable_model_retired_terminal",
            "executable_model_ready_score": 0.0,
        },
        mod.EXP5642_ARC_LIVE_AB_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        mod.EXP5643_ARC_LEVEL_PATH: {
            "schema": "arc_live_self_discovery_levelup_attempt.v4",
            "status": "complete",
            "experiment_id": "exp5643-arc-live-self-discovery-levelup-v509",
            "honest_verdict": "complete: no_new_arc_level_banked_lf52_L8_bounded_live_attempt_v509",
            "registry_count_before": 177,
            "registry_count_after": 177,
            "registry_delta": 0,
        },
        mod.EXP5644_TWO_AXIS_EXACT_PATH: {
            "schema": "carnot.experiment_5644.two_axis_parallel_tempering_exact_audit.v1",
            "status": "complete",
            "experiment_id": "exp5644-two-axis-parallel-tempering-exact-audit",
            "honest_verdict": "complete: exact two-axis beta-lambda label-exchange invariant audit ready",
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5645_TWO_AXIS_QUALITY_PATH: {
            "schema": "carnot.experiment_5645.two_axis_tempering_hard_constraint_quality.v1",
            "status": "blocked",
            "experiment_id": "exp5645-two-axis-tempering-hard-constraint-quality",
            "honest_verdict": "blocked: two-axis constraint-penalty exchange did not clear every preregistered quality promotion gate",
            "two_axis_quality_ready_score": 0.0,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5646_RUST_PARITY_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
    }
    base[mod.EXP5647_CAPSTONE_PATH] = {
        "schema": "carnot.experiment_5647.v509_capstone_reconciliation.v1",
        "status": "complete",
        "experiment_id": "exp5647-v509-capstone-reconciliation",
        "honest_verdict": (
            "complete: v509 reconciled; fr11_promoted=True; "
            "fr11_shadow_opt_in=True; arc_registry_delta=0; "
            "two_axis_quality_promoted=False"
        ),
        "fr11_independent_promotion_status": {"promoted": True},
        "fr11_shadow_integration_status": {"ready": True, "default_enabled": False},
        "arc_registry_count_after": 177,
        "arc_solve_provenance": {"registry_delta": 0},
        "one_axis_replica_exchange_preserved": True,
        "two_axis_quality_status": {"promoted": False},
        "rust_parity_status": {"promoted": False, "gate_skipped": True},
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "retirements_applied": [
            {
                "scope": "arc_counterexample_executable_model_exp5641",
                "manifest_update_required": True,
                "manifest_updated": False,
                "reason": "exp5641_executable_model_not_ready",
            },
            {
                "scope": "two_axis_quality_extension_exp5645",
                "manifest_update_required": True,
                "manifest_updated": False,
                "reason": "quality_gate_failed_or_material_regression_present",
            },
        ],
    }
    return base


def _outer_payloads() -> dict[Path, JsonDict]:
    return {
        rel_path: {
            "schema": f"carnot.{task_id}.v1",
            "experiment": task_id.replace("-", "_"),
            "honest_verdict": f"complete: {task_id} fixture verdict",
        }
        for task_id, rel_path in mod.OUTER_LOOP_ARTIFACT_PATHS.items()
    }


def _manifest_payload(include_retirements: bool = True) -> JsonDict:
    extras: list[JsonDict] = []
    if include_retirements:
        extras = [dict(entry) for entry in mod.REQUIRED_MANIFEST_RETIREMENTS]
    return {
        "retired": [],
        "retired_experiments": [],
        "retired_extras": extras,
    }


def _make_root(
    root: Path,
    *,
    include_manifest_retirements: bool = True,
    omit: Path | None = None,
    malformed: Path | None = None,
) -> None:
    _write_text(root, "AGENTS.md")
    _write_text(root, "CODEX.md")
    _write_text(root, "CLAUDE.md")
    _write_text(root, "research-program.md")
    _write_text(root, "openspec/change-proposals/research-roadmap-vNEXT.md")
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH)
    _write_json(
        root,
        mod.ROADMAP_RELATIVE_PATH,
        {
            "milestone": mod.CURRENT_MILESTONE,
            "tasks": [{"id": task_id} for task_id in mod.CURRENT_TASK_IDS],
        },
    )
    _write_json(root, "ops/arc_solve_registry.yaml", {"reproducible_total_levels": 177})
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        yaml.safe_dump(_manifest_payload(include_manifest_retirements), sort_keys=False),
        encoding="utf-8",
    )
    for rel_path, payload in {**_v509_payloads(), **_outer_payloads()}.items():
        if rel_path == omit:
            continue
        if rel_path == malformed:
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json\n", encoding="utf-8")
            continue
        _write_json(root, rel_path, payload)


def test_req_capstone_5706_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5706: OpenSpec declares the V510 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5706") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in [*mod.V509_ARTIFACT_PATHS.values(), *mod.OUTER_LOOP_ARTIFACT_PATHS.values()]:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5706_live_repo_archives_v509_and_allocates_v510() -> None:
    """SCENARIO-CAPSTONE-5706: live repo transition preserves terminal evidence."""

    artifact = mod.run_transition(
        root=REPO,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["source_capstone_hash"] == mod.path_sha256(REPO / mod.EXP5647_CAPSTONE_PATH)
    assert set(artifact["v509_task_verdicts"]) == set(mod.V509_TASK_IDS)
    assert artifact["fr11_promoted"] is True
    assert artifact["fr11_shadow_default_enabled"] is False
    assert artifact["arc_registry_count"] == 177
    assert artifact["arc_registry_delta"] == 0
    assert artifact["one_axis_replica_exchange_promoted"] is True
    assert artifact["two_axis_quality_promoted"] is False
    assert artifact["timing_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["current_task_range"] == mod.CURRENT_TASK_RANGE

    before_scopes = {row["scope"] for row in artifact["missing_retirements_before"]}
    assert before_scopes == {
        "arc_counterexample_executable_model_exp5641",
        "two_axis_quality_extension_exp5645",
    }
    applied_scopes = {row["scope"] for row in artifact["retirements_applied"]}
    assert {
        "arc_counterexample_patched_transition_model_exp5641",
        "two_axis_beta_lambda_tempering_extension_exp5645",
    } <= applied_scopes
    assert all(row["manifest_entry_present"] is True for row in artifact["retirements_applied"])

    assert set(artifact["outer_loop_snapshot"]) == set(mod.OUTER_LOOP_TASK_IDS)
    assert artifact["outer_loop_snapshot"]["exp5703-sp80-candidate-stack-mechanism-trace"][
        "gap_id"
    ] == "GAP-5703"
    assert artifact["dependency_id_validation"]["valid"] is True
    assert artifact["unconditional_arc_path"]["task_id"] == "exp5713-arc-live-levelup-attempt"
    assert artifact["unconditional_arc_path"]["structured_gate_required"] is False
    assert artifact["protected_file_checks"]["research-roadmap.yaml"]["unchanged"] is True
    assert artifact["protected_file_checks"]["scripts/research_conductor.py"]["unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5706_missing_inputs_or_manifest_debt_block(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5706-MISSING-MALFORMED: bad inputs fail closed."""

    _make_root(
        tmp_path,
        include_manifest_retirements=False,
        omit=mod.OUTER_LOOP_ARTIFACT_PATHS["exp5705-full-precision-27b-vs-4bit-quant-ab"],
        malformed=mod.EXP5641_ARC_MODEL_PATH,
    )
    artifact = mod.run_transition(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.EXP5641_ARC_MODEL_PATH.as_posix() in artifact["malformed_artifacts"]
    assert (
        mod.OUTER_LOOP_ARTIFACT_PATHS["exp5705-full-precision-27b-vs-4bit-quant-ab"].as_posix()
        in artifact["missing_artifacts"]
    )
    assert {row["scope"] for row in artifact["manifest_debt_after"]} == {
        "arc_counterexample_patched_transition_model_exp5641",
        "two_axis_beta_lambda_tempering_extension_exp5645",
    }
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5706_validation_rejects_overclaims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5706-FIELD-PRINCIPLES: validation catches drift."""

    _make_root(tmp_path)
    artifact = mod.run_transition(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert mod.validate_artifact(artifact) == []

    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"honest_verdict": mod.FIELD_PRINCIPLES["honest_verdict"]}}
    )
    assert "fr11_promoted" in mod.validate_artifact({**artifact, "fr11_promoted": False})
    assert "fr11_shadow_default_enabled" in mod.validate_artifact(
        {**artifact, "fr11_shadow_default_enabled": True}
    )
    assert "arc_registry_count" in mod.validate_artifact({**artifact, "arc_registry_count": 176})
    assert "arc_registry_delta" in mod.validate_artifact({**artifact, "arc_registry_delta": 1})
    assert "one_axis_replica_exchange_promoted" in mod.validate_artifact(
        {**artifact, "one_axis_replica_exchange_promoted": False}
    )
    assert "two_axis_quality_promoted" in mod.validate_artifact(
        {**artifact, "two_axis_quality_promoted": True}
    )
    assert "timing_claimed" in mod.validate_artifact({**artifact, "timing_claimed": True})
    assert "hardware_speedup_claimed" in mod.validate_artifact(
        {**artifact, "hardware_speedup_claimed": True}
    )
    assert "current_task_range" in mod.validate_artifact(
        {**artifact, "current_task_range": "exp5706-exp5717"}
    )
    assert "outer_loop_snapshot" in mod.validate_artifact({**artifact, "outer_loop_snapshot": {}})
    assert "retired_scopes" in mod.validate_artifact(
        {**artifact, "retired_scopes": [{"scope": "generic_arc_models"}]}
    )
    bad_dependency = dict(artifact["dependency_map"])
    bad_dependency["exp5716-v510-capstone"]["depends_on"] = ["exp9999-missing"]
    assert "dependency_map" in mod.validate_artifact({**artifact, "dependency_map": bad_dependency})
    bad_gate = dict(artifact["gate_map"])
    bad_gate["exp5709-fr11-prospective-shadow-stream"] = [{"upstream": "exp9999-missing"}]
    assert "gate_map" in mod.validate_artifact({**artifact, "gate_map": bad_gate})
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )


def test_scenario_capstone_5706_helpers_and_cli_emit_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5706: writer, CLI, and helpers are deterministic."""

    _make_root(tmp_path)
    validation = [
        {"command": ".venv/bin/pytest tests/python/test_experiment_5706_transition_v510.py -q", "exit_code": 0, "status": "passed"},
        {"command": ".venv/bin/pytest tests/python -q", "exit_code": 2, "status": "failed_pre_existing_debt"},
    ]

    artifact = mod.write_transition(
        root=tmp_path,
        validation_results=validation,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["pre_existing_broad_suite_debt"][0]["exit_code"] == 2

    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    output_path = tmp_path / "custom" / "transition.json"
    assert (
        mod.main(
            [
                "--root",
                str(tmp_path),
                "--output",
                str(output_path),
                "--validation-results",
                str(validation_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID
    assert mod._load_validation_results(None) == mod.DEFAULT_VALIDATION_RESULTS
    assert mod._load_validation_results(validation_path) == validation
    bad_validation = tmp_path / "bad-validation.json"
    bad_validation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_validation_results(bad_validation)


def test_scenario_capstone_5706_defensive_paths_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-5706-MISSING-MALFORMED: helpers report malformed state."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    payload, meta = mod._read_json_any(list_json)
    assert payload == {}
    assert meta["error"] == "not_json_object"
    missing_payload, missing_meta = mod._read_json_any(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_meta["error"] == "missing"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad\n", encoding="utf-8")
    bad_payload, bad_meta = mod._read_json_any(bad_json)
    assert bad_payload == {}
    assert bad_meta["error"] == "malformed_json"
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":\n", encoding="utf-8")
    manifest, manifest_meta = mod._read_manifest(bad_yaml)
    assert manifest == {}
    assert manifest_meta["error"] == "malformed_yaml"
    missing_manifest, missing_manifest_meta = mod._read_manifest(tmp_path / "missing.yaml")
    assert missing_manifest == {}
    assert missing_manifest_meta["error"] == "missing"
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    list_manifest, list_manifest_meta = mod._read_manifest(list_yaml)
    assert list_manifest == {}
    assert list_manifest_meta["error"] == "not_yaml_mapping"
    assert mod._status_for_meta({"exists": False}) == "missing"
    assert mod._status_for_meta({"exists": True, "loadable": False}) == "malformed"
    assert mod._status_for_meta({"exists": True, "loadable": True}) == "present"
    assert mod._terminal_prefix_ok("complete: done") is True
    assert mod._terminal_prefix_ok("blocked: resource") is True
    assert mod._terminal_prefix_ok("maybe") is False
    assert mod._registry_count_from_yaml({}) is None
    assert mod._registry_count_from_yaml({"reproducible_total_levels": True}) is None
    assert mod._registry_count_from_yaml({"reproducible_total_levels": 177}) == 177
    assert mod._registry_count_from_yaml({"reproducible_total_levels": "177"}) == 177
    assert mod._manifest_entries({"retired_extras": "bad"}) == []
    assert mod._missing_retirements_before({"retirements_applied": "bad"}) == []
    bad_dependency = mod._valid_dependency_ids(
        {"exp9999-new": {"depends_on": ["exp5706-transition-v510"]}},
        {"exp9998-new": [{"upstream": "exp9997-new"}]},
    )
    assert bad_dependency["valid"] is False
    assert bad_dependency["invalid_ids"] == ["exp9997-new", "exp9998-new", "exp9999-new"]

    _make_root(tmp_path)
    capstone_path = tmp_path / mod.EXP5647_CAPSTONE_PATH
    capstone = json.loads(capstone_path.read_text(encoding="utf-8"))
    capstone["arc_solve_provenance"] = []
    capstone["fr11_independent_promotion_status"] = []
    capstone["fr11_shadow_integration_status"] = []
    capstone["two_axis_quality_status"] = []
    capstone["rust_parity_status"] = []
    capstone_path.write_text(json.dumps(capstone) + "\n", encoding="utf-8")
    malformed_nested = mod.run_transition(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert malformed_nested["fr11_promoted"] is False
    assert malformed_nested["arc_registry_delta"] == 0

    _make_root(tmp_path)
    valid_artifact = mod.run_transition(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    bad_principles = dict(valid_artifact["field_principles"])
    bad_principles["honest_verdict"] = "wrong"
    assert "field_principles" in mod.validate_artifact(
        {**valid_artifact, "field_principles": bad_principles}
    )
    bad_v509 = dict(valid_artifact["v509_task_verdicts"])
    bad_v509.pop("exp5636-transition-v509")
    assert "v509_task_verdicts" in mod.validate_artifact(
        {**valid_artifact, "v509_task_verdicts": bad_v509}
    )
    assert "retired_scopes" in mod.validate_artifact(
        {**valid_artifact, "retired_scopes": [{"scope": "other", "note": "generic_replica_exchange"}]}
    )

    monkeypatch.setattr(mod, "run_transition", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError):
        mod.write_transition(root=tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["schema"])
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])
