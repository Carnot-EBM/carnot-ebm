"""RED-first tests for the matched ARC adapter-withheld transfer pilot.

Spec refs: REQ-ARC-WMTE-7186 and SCENARIO-ARC-WMTE-7186-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7186_v633_arc_withheld_transfer as exp


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _registry_rows() -> list[dict[str, object]]:
    return [
        {
            "game": "r11l",
            "reproducibility": "reproduced",
            "full_game_clear": True,
            "levels_reproduced": 6,
        },
        {
            "game": "not-ready",
            "reproducibility": "pending",
            "full_game_clear": False,
            "levels_reproduced": 0,
        },
        {
            "game": "ls20",
            "reproducibility": "reproduced",
            "full_game_clear": True,
            "levels_reproduced": 7,
        },
        {
            "game": "wa30",
            "reproducibility": "reproduced",
            "full_game_clear": True,
            "levels_reproduced": 9,
        },
    ]


def _cell(seed: int, arm: str, levels: int) -> dict[str, object]:
    adapter_allowed = arm == "adapter_visible_control"
    return {
        "game": "ls20",
        "seed": seed,
        "arm": arm,
        "status": "complete",
        "entrypoint": exp.REAL_ENTRYPOINT,
        "policy_class": "E3AgentPolicy",
        "actions": 120,
        "levels": levels,
        "elapsed_s": 30.0,
        "generator_invocation_count": 1,
        "live_qwen_invoked": True,
        "supervisor_outcome": "completed",
        "adapter_access": {
            "policy": (
                "selected_adapter_allowed" if adapter_allowed else "selected_adapter_denied"
            ),
            "selected_adapter": "ls20",
            "recipe_module_loaded_before_denial": False,
            "denied_attempt_count": 1 if not adapter_allowed else 0,
            "allowed_access_count": 1 if adapter_allowed else 0,
            "proved": True,
        },
        "gpu_receipt_id": f"gpu-{seed}-{arm}",
        "raw_receipt_id": f"raw-{seed}-{arm}",
    }


def _complete_rows() -> list[dict[str, object]]:
    return [
        _cell(exp.SEEDS[0], "adapter_withheld", 0),
        _cell(exp.SEEDS[0], "adapter_visible_control", 0),
        _cell(exp.SEEDS[1], "adapter_withheld", 0),
        _cell(exp.SEEDS[1], "adapter_visible_control", 0),
    ]


def test_req_arc_wmte_7186_spec_precedes_implementation() -> None:
    """REQ-ARC-WMTE-7186 defines all named interventions and artifact fields."""

    text = SPEC.read_text(encoding="utf-8").split("## REQ-ARC-WMTE-7186", 1)[1]
    for scenario in (
        "PRECONDITIONS",
        "ROTATION",
        "CONFIGURATION",
        "ACCESS",
        "RUNTIME",
        "NULL",
        "NONCLAIM",
    ):
        assert f"SCENARIO-ARC-WMTE-7186-{scenario}" in text
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in text
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)


def test_scenario_rotation_selects_next_adaptered_registry_row() -> None:
    """SCENARIO-ARC-WMTE-7186-ROTATION selects ls20 after Exp7144's r11l."""

    selected = exp.select_rotated_game(
        _registry_rows(), adapter_games={"r11l", "ls20", "wa30"}, previous_game="r11l"
    )
    assert selected == {
        "game": "ls20",
        "eligibility_rank": 2,
        "registry_index": 2,
        "levels_reproduced_precheck": 7,
        "selection_rule": exp.ROTATION_RULE,
    }

    wrapped = exp.select_rotated_game(
        _registry_rows(), adapter_games={"r11l", "ls20", "wa30"}, previous_game="wa30"
    )
    assert wrapped["game"] == "r11l"
    with pytest.raises(ValueError, match="no_registry_eligible"):
        exp.select_rotated_game(_registry_rows(), adapter_games=set())
    with pytest.raises(ValueError, match="previous_rotation_game_not_eligible"):
        exp.select_rotated_game(
            _registry_rows(), adapter_games={"r11l", "ls20"}, previous_game="absent"
        )


def test_scenario_configuration_diff_is_only_access_policy() -> None:
    """SCENARIO-ARC-WMTE-7186-CONFIGURATION keeps both policy builds matched."""

    serialized = exp.serialize_common_configuration(
        game="ls20", effective_flags={"goal_energy": True, "tool_loop": False}
    )
    left, right = exp.construct_policy_configurations(serialized)
    assert left is not right
    assert left["effective_flags"] is not right["effective_flags"]
    assert left["policy_class"] == right["policy_class"] == "E3AgentPolicy"
    assert left["seed_schedule"] == right["seed_schedule"] == list(exp.SEEDS)
    assert exp.configuration_diff_rows(left, right) == [
        {
            "field": "adapter_access_policy",
            "adapter_withheld": "selected_adapter_denied",
            "adapter_visible_control": "selected_adapter_allowed",
            "permitted": True,
        }
    ]
    assert exp.matched_configuration_gate(left, right)["passed"] is True


def test_scenario_configuration_extra_difference_disqualifies() -> None:
    """SCENARIO-ARC-WMTE-7186-CONFIGURATION rejects any second difference."""

    serialized = exp.serialize_common_configuration(game="ls20", effective_flags={"x": True})
    left, right = exp.construct_policy_configurations(serialized)
    right["generator_settings"]["temperature"] = 0.25
    gate = exp.matched_configuration_gate(left, right)
    assert gate["passed"] is False
    assert gate["expected_value"] == ["adapter_access_policy"]
    assert gate["observed_value"] == [
        "adapter_access_policy",
        "generator_settings.temperature",
    ]
    with pytest.raises(ValueError, match="must_be_an_object"):
        exp.construct_policy_configurations("[]")


def test_scenario_access_receipts_require_real_denial_and_control_use() -> None:
    """SCENARIO-ARC-WMTE-7186-ACCESS rejects labels without executed access."""

    rows = _complete_rows()
    receipts = [
        deepcopy(row["adapter_access"]) | {"seed": row["seed"], "arm": row["arm"]} for row in rows
    ]
    assert exp.adapter_access_gate(receipts)["passed"] is True

    forged = deepcopy(receipts)
    forged[1]["allowed_access_count"] = 0
    failed = exp.adapter_access_gate(forged)
    assert failed["passed"] is False
    assert failed["observed_value"]["control_access_receipt_count"] == 1


def test_scenario_complete_rows_report_seed_paired_loss_and_nonclaim() -> None:
    """SCENARIO-ARC-WMTE-7186-RUNTIME retains zeros and no solve claim."""

    rows = _complete_rows()
    rows[1]["levels"] = 1
    pairs = exp.paired_transfer_rows(rows)
    assert pairs == [
        {
            "game": "ls20",
            "seed": exp.SEEDS[0],
            "adapter_withheld_levels": 0,
            "adapter_visible_control_levels": 1,
            "transfer_loss_levels": 1,
        },
        {
            "game": "ls20",
            "seed": exp.SEEDS[1],
            "adapter_withheld_levels": 0,
            "adapter_visible_control_levels": 0,
            "transfer_loss_levels": 0,
        },
    ]
    artifact = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=12.5,
        checks=[exp.gate_check("preflight", "repo", "all", True, True)],
        source_hashes={"source": "sha256:" + "1" * 64},
        selected_game={"game": "ls20"},
        configuration_rows=[
            {
                "field": "adapter_access_policy",
                "adapter_withheld": "selected_adapter_denied",
                "adapter_visible_control": "selected_adapter_allowed",
                "permitted": True,
            }
        ],
        cell_rows=_complete_rows(),
        model_specs=[{"hf_id": exp.MODEL_ID, "quantization": exp.QUANTIZATION}],
        gpu_receipts={"provenance_ok": True},
        runner_receipt={"model_count": 1, "runner": "native_llama.cpp_server"},
        raw_manifest=[{"receipt_id": row["raw_receipt_id"]} for row in rows],
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_transfer_complete_score"] == 1
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["new_solve_claimed"] is False
    assert artifact["registered_level_increment"] == 0
    assert artifact["per_game_results"] == [
        {
            "game": "ls20",
            "seed": exp.SEEDS[0],
            "adapter_withheld_levels": 0,
            "adapter_visible_control_levels": 0,
            "transfer_loss_levels": 0,
        },
        {
            "game": "ls20",
            "seed": exp.SEEDS[1],
            "adapter_withheld_levels": 0,
            "adapter_visible_control_levels": 0,
            "transfer_loss_levels": 0,
        },
    ]
    assert exp.validate_artifact(artifact) == []


def test_scenario_blocked_source_preflight_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7186-PRECONDITIONS keeps a missing runner actionable."""

    root = tmp_path / "repo"
    for relative in exp.REQUIRED_SOURCE_PATHS:
        if relative == exp.MISSING_PLANNER_SOURCE_PATH:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == exp.SPEC_PATH:
            path.write_text("## REQ-ARC-WMTE-7186: fixture\n", encoding="utf-8")
        elif relative == Path("research-roadmap.yaml"):
            path.write_text(
                "- id: exp7186-arc-withheld-transfer\n"
                "  milestone: 2026.09.633\n"
                "  deliverable: results/experiment_7186_v633_arc_withheld_transfer.json\n",
                encoding="utf-8",
            )
        else:
            path.write_text("fixture\n", encoding="utf-8")
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoints" / "running.json"
    raw_dir = tmp_path / "raw"
    called = []
    artifact = exp.run_experiment(
        root=root,
        run_date=exp.RUN_DATE,
        result_path=output,
        checkpoint_path=checkpoint,
        raw_dir=raw_dir,
        measurement_runner=lambda *_args, **_kwargs: called.append(True),
    )
    assert called == []
    assert output.is_file()
    assert checkpoint.is_file()
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["execution_venue"] == "host"
    assert artifact["execution_host"]
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "required_source_bytes"
    observed = artifact["gate_check_summary"]["observed_value"]
    assert observed[str(exp.MISSING_PLANNER_SOURCE_PATH)] == 0
    assert exp.validate_artifact(json.loads(output.read_text(encoding="utf-8"))) == []


def test_scenario_partial_or_mismatched_cells_cannot_score_complete() -> None:
    """SCENARIO-ARC-WMTE-7186-NULL requires every executed cell and receipt."""

    rows = _complete_rows()[:-1]
    summary = exp.completion_gate(rows, adapter_gate_passed=True, configuration_gate_passed=True)
    assert summary["passed"] is False
    assert summary["observed_value"]["complete_cell_count"] == 3


def test_terminal_classification_covers_disqualified_partial_and_positive() -> None:
    """REQ-ARC-WMTE-7186 keeps terminal evidence classes distinct."""

    passed = [exp.gate_check("preflight", "repo", "all", True, True)]
    common = {
        "run_date": exp.RUN_DATE,
        "duration_s": 1.0,
        "checks": passed,
        "source_hashes": {"source": "sha256:" + "1" * 64},
    }
    disqualified = exp.build_terminal_artifact(**common)
    assert disqualified["verdict_class"] == "disqualified"

    configuration = [
        {
            "field": "adapter_access_policy",
            "adapter_withheld": "selected_adapter_denied",
            "adapter_visible_control": "selected_adapter_allowed",
            "permitted": True,
        }
    ]
    partial = exp.build_terminal_artifact(
        **common,
        configuration_rows=configuration,
        cell_rows=_complete_rows()[:-1],
        gpu_receipts={"provenance_ok": True},
        runner_receipt={"model_count": 1},
    )
    assert partial["verdict_class"] == "partial"
    assert partial["inference_substrate_class"] == "model_bounded_generation"

    positive_rows = _complete_rows()
    positive_rows[1]["levels"] = 1
    positive = exp.build_terminal_artifact(
        **common,
        configuration_rows=configuration,
        cell_rows=positive_rows,
        gpu_receipts={"provenance_ok": True},
        runner_receipt={"model_count": 1},
    )
    assert positive["verdict_class"] == "positive"
    assert positive["arc_transfer_complete_score"] == 1


def test_validator_reports_each_structural_failure(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7186 rejects altered terminal projections and claims."""

    assert "missing_fields:" in exp.validate_artifact({})[0]
    base = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        checks=[exp.gate_check("missing", "repo", "file", True, False)],
        source_hashes={},
    )
    path = tmp_path / "artifact.json"
    exp.atomic_write(path, base)
    assert exp.validate_artifact(path) == []

    changed = deepcopy(base)
    changed.update(
        field_principles={},
        run_date="20260909",
        MODEL_SPECS=[],
        execution_venue="icbfl1",
        verdict_class="unknown",
        status="unknown",
        gate_check_summary={},
        reproducibility_checksum="bad",
        new_solve_claimed=True,
        registered_level_increment=1,
    )
    errors = exp.validate_artifact(changed)
    assert {
        "field_principles_mismatch",
        "run_date_mismatch",
        "model_specs_declaration_mismatch",
        "execution_venue_invalid",
        "verdict_class_invalid",
        "status_invalid",
        "gate_summary_mismatch",
        "reproducibility_checksum_invalid",
        "new_solve_claim_forbidden",
        "registry_increment_forbidden",
    } <= set(errors)

    blocked = deepcopy(base)
    blocked["status"] = "complete"
    blocked["inference_substrate_class"] = "model_full_generation"
    blocked["rows"] = [{}]
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    errors = exp.validate_artifact(blocked)
    assert "blocked_terminal_inconsistent" in errors
    assert "blocked_rows_not_empty" in errors

    complete = deepcopy(base)
    complete["verdict_class"] = "null"
    complete["status"] = "partial"
    complete["arc_transfer_complete_score"] = 1
    complete["reproducibility_checksum"] = exp.artifact_checksum(complete)
    assert "complete_score_inconsistent" in exp.validate_artifact(complete)

    no_summary = deepcopy(base)
    no_summary["gate_check_summary"] = None
    no_summary["reproducibility_checksum"] = exp.artifact_checksum(no_summary)
    assert "gate_summary_missing" in exp.validate_artifact(no_summary)

    checksum = deepcopy(base)
    checksum["duration_s"] = 2.0
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(checksum)


def test_source_oserror_and_parser_defaults_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7186-PRECONDITIONS converts unreadable source to missing."""

    first = (tmp_path / exp.REQUIRED_SOURCE_PATHS[0]).resolve()
    original_stat = Path.stat

    def guarded_stat(path: Path, *args: object, **kwargs: object) -> object:
        if path.absolute() == first:
            raise OSError("unreadable fixture")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    sizes, hashes = exp._snapshot_sources(tmp_path)
    assert sizes[str(exp.REQUIRED_SOURCE_PATHS[0])] == 0
    assert hashes[str(exp.REQUIRED_SOURCE_PATHS[0])] == "missing"
    monkeypatch.undo()
    assert exp._task_identity("") == {"id": None, "milestone": None, "deliverable": None}
    args = exp.parse_args([])
    assert args.date == exp.RUN_DATE
    assert args.result_path == exp.RESULT_PATH


def test_terminal_self_validation_failure_stops_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7186 never publishes an artifact that fails its own validator."""

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_invalid"])
    with pytest.raises(ValueError, match="terminal_artifact_invalid:forced_invalid"):
        exp.run_experiment(
            root=tmp_path,
            run_date=exp.RUN_DATE,
            result_path=tmp_path / "result.json",
            checkpoint_path=tmp_path / "checkpoints" / "running.json",
            raw_dir=tmp_path / "raw",
        )
    assert not (tmp_path / "result.json").exists()
