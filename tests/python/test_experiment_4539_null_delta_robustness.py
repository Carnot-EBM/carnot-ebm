"""Tests for Exp 4539 null-delta robustness.

Spec refs: REQ-CAPSTONE-4539, SCENARIO-CAPSTONE-4539,
REQ-ARC-WMTE-4524, SCENARIO-ARC-WMTE-4524.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot import experiment_4531_capstone_v418 as capstone_mod
from carnot import experiment_4539_null_delta_robustness as runner_mod
from carnot.reporting import null_delta_robustness_4539 as mod
from scripts import adversarial_verify as av
from scripts import summarize_artifact as sa


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
CAPSTONE_SPEC = REPO / "openspec/capabilities/capstone/spec.md"
ARC_SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
EXP4524_FIXTURE = REPO / "results/experiment_4524_reach_deeper_levels.json"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, Any]]:
    return []


def _summary_codes(codes: dict[str, int]):
    def _runner(path: Path, _root: Path) -> int:
        return int(codes.get(path.name, 0))

    return _runner


def _minimal_payloads(a2_payload: JsonDict) -> dict[str, JsonDict]:
    return {
        "A1_forward_walk_navigation": {
            "honest_verdict": "complete: forward_walk_no_reduction_honest_null",
            "flagged_adversarial": False,
            "median_actions_on_core_control": 7761.5,
            "median_actions_on_core_best": 7761.5,
            "core_solves_preserved": True,
            "nav_diagnostics_before_after": {
                "before": {"reset_replay_steps": 4576, "forward_walk_hits": 26},
                "after": {"reset_replay_steps": 4533, "forward_walk_hits": 34},
            },
            "chosen_submitted_config": "unchanged",
            "leaderboard_submission": False,
        },
        "A2_reach_deeper_levels": a2_payload,
        "A2_stop_after_levelup": {
            "honest_verdict": "success: stop_after_levelup_core_actions_2825_below_control",
            "median_actions_on_core_control": 7761.5,
            "median_actions_on_core_best": 2825.5,
            "core_solves_preserved": True,
            "levels_per_game_preserved": {"passed": True, "lost_level_depth_games": []},
            "leaderboard_submission": False,
        },
        "A3_levelup_attempt": {
            "honest_verdict": "success: cd82_L2_offline_reproduced",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "target_game": "cd82",
            "target_level": 2,
            "reproduction_gate": {
                "game": "cd82",
                "claimed_level": 2,
                "reached_level": 2,
                "reproduced": True,
            },
            "registry_update": {
                "prior_total_declared": 48,
                "new_total_declared": 50,
                "reconciled_total_delta": 2,
                "banked_levels": 1,
                "updated": True,
            },
        },
        "A4_integration": {
            "honest_verdict": "complete: no_lever_raises_core_efficiency_honest_null",
            "flagged_adversarial": False,
            "core_efficiency_baseline": capstone_mod.CORE_EFFICIENCY_BASELINE,
            "core_efficiency_integrated": capstone_mod.CORE_EFFICIENCY_BASELINE,
            "core_solves_preserved": True,
            "ready_for_operator_submit": False,
            "operator_submission_performed": False,
            "heldout_solve_rate": 0.0,
            "gate_result": {
                "current": {
                    "median_actions_on_core": 2824.5,
                    "core_efficiency": capstone_mod.CORE_EFFICIENCY_BASELINE,
                    "solved_count": 4,
                },
                "pass": True,
            },
            "nav_diagnostics": {
                "integrated_config": "unchanged",
                "reset_replay_steps_integrated": 4500,
                "reset_replay_steps_candidate_after": 4533,
            },
        },
    }


def _write_payloads(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / capstone_mod.DEFAULT_UPSTREAMS[key].path, payload)
    registry = root / capstone_mod.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\nreproducible_total_levels: 50\n"
        "prior_submitted_baseline_levels: 13\n",
        encoding="utf-8",
    )


def _a2_live_flags(path: Path) -> list[dict[str, Any]]:
    if path.name == "experiment_4524_reach_deeper_levels.json":
        return av.verify_artifact(path)["flags"]
    return []


def test_req_capstone_4539_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4539: OpenSpec declares the null-delta robustness contract."""

    capstone_spec = CAPSTONE_SPEC.read_text(encoding="utf-8")
    arc_spec = ARC_SPEC.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4539" in capstone_spec
    assert "SCENARIO-CAPSTONE-4539" in capstone_spec
    assert "results/experiment_4539_null_delta_robustness.json" in capstone_spec
    assert "null_delta_methodology_note" in arc_spec
    assert "efficiency_delta" in arc_spec


def test_scenario_capstone_4539_summarizer_classifies_exp4524_fixture() -> None:
    """SCENARIO-CAPSTONE-4539: exp4524's annotated null-delta TAUTOLOGY is readable."""

    payload = json.loads(EXP4524_FIXTURE.read_text(encoding="utf-8"))
    flags = av.verify_artifact(EXP4524_FIXTURE)["flags"]

    assert payload["efficiency_delta"] == 0.0
    assert payload["null_delta_methodology_note"]
    assert [
        flag for flag in flags
        if flag["kind"] == "TAUTOLOGY" and flag["severity"] == "critical"
    ], "the verifier TAUTOLOGY check must remain active"

    classification = sa.classify_known_false_positive_null_delta(payload, flags)

    assert classification is not None
    assert classification["kind"] == "KNOWN_FALSE_POSITIVE_NULL_DELTA_TAUTOLOGY"
    assert classification["field_pair"] == [
        "core_efficiency_baseline",
        "core_efficiency_best",
    ]
    assert "corrigendum" in classification["corrigendum_note"].lower()
    diagnosis = sa.readable_diagnosis_context(payload, flags)
    assert diagnosis is not None
    assert diagnosis["barrier_diagnosis"]["root_cause"] == "depth_cap"
    assert diagnosis["corrigendum"]["field_pair"] == classification["field_pair"]


def test_scenario_capstone_4539_unrelated_tautology_still_excluded(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4539: unrelated equal metrics remain excluded."""

    artifact_path = tmp_path / "unrelated_tautology.json"
    unrelated = {
        "honest_verdict": "complete: fabricated_collision_fixture",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "flagged_adversarial": True,
        "metric_alpha": 0.812345,
        "metric_beta": 0.812345,
        "efficiency_delta": 0.0,
        "null_delta_methodology_note": "A note is not enough when the equal fields are unrelated.",
        "duration_s": 1.0,
    }
    _write_json(artifact_path, unrelated)
    flags = av.verify_artifact(artifact_path)["flags"]

    assert any(flag["kind"] == "TAUTOLOGY" for flag in flags)
    assert sa.classify_known_false_positive_null_delta(unrelated, flags) is None
    assert sa.readable_diagnosis_context(unrelated, flags) is None


def test_scenario_capstone_4539_capstone_reads_exp4524_barrier_with_corrigendum(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4539: capstone reads exp4524 diagnosis but no headline win."""

    a2_payload = json.loads(EXP4524_FIXTURE.read_text(encoding="utf-8"))
    _write_payloads(tmp_path, _minimal_payloads(a2_payload))

    artifact = capstone_mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=11.0,
        live_flag_runner=_a2_live_flags,
        summarize_runner=_summary_codes({"experiment_4524_reach_deeper_levels.json": 2}),
    )

    diagnosis = artifact["a2_l1_l2_barrier_diagnosis"]
    assert diagnosis["status"] == "corrigendum_known_false_positive_null_delta"
    assert diagnosis["cleanly_reportable"] is True
    assert diagnosis["what_blocks_deeper_levels"] == "depth_cap"
    assert diagnosis["induction_not_engaged"] is True
    assert diagnosis["corrigendum"]["field_pair"] == [
        "core_efficiency_baseline",
        "core_efficiency_best",
    ]
    assert artifact["scorecard"]["core_efficiency"]["moved"] is False
    cited = {
        row["artifact_key"]: row
        for row in artifact["cited_upstream_artifacts"]
    }
    assert cited["A2_reach_deeper_levels"]["fields_imported"] == []
    assert artifact["upstream_provenance"][1]["diagnosis_context_read"] is True


def test_scenario_capstone_4539_capstone_excludes_fabrication_style_tautology(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4539: genuine unrelated TAUTOLOGY stays excluded."""

    a2_payload = {
        "honest_verdict": "complete: unrelated_collision_fixture",
        "flagged_adversarial": True,
        "core_efficiency_baseline": capstone_mod.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": capstone_mod.CORE_EFFICIENCY_BASELINE,
        "efficiency_delta": 0.0,
        "null_delta_methodology_note": "The note does not cover unrelated metrics.",
        "metric_alpha": 0.812345,
        "metric_beta": 0.812345,
        "barrier_diagnosis": {
            "root_cause": "depth_cap",
            "actionable_next_step": "do not read this from a fabricated collision",
        },
    }
    _write_payloads(tmp_path, _minimal_payloads(a2_payload))

    artifact = capstone_mod.build_artifact(
        tmp_path,
        started_s=20.0,
        now_s=21.0,
        live_flag_runner=lambda path: [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": (
                    "metric_alpha=0.812345 and metric_beta=0.812345 agree "
                    "to >5 sig figs. Two distinct metrics matching this precisely "
                    "is more likely a bug than a finding."
                ),
            }
        ]
        if path.name == "experiment_4524_reach_deeper_levels.json"
        else [],
        summarize_runner=_summary_codes({"experiment_4524_reach_deeper_levels.json": 2}),
    )

    assert artifact["a2_l1_l2_barrier_diagnosis"] == {
        "status": "excluded_flagged_adversarial",
        "cleanly_reportable": False,
        "what_blocks_deeper_levels": None,
        "what_to_build_next": "not_cleanly_reportable_from_flagged_artifact",
    }
    assert artifact["upstream_provenance"][1]["diagnosis_context_read"] is False


def test_req_capstone_4539_payload_fields_and_validation(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4539: result artifact records the robustness proof."""

    payload = mod.build_payload(
        REPO,
        tests_added_pass={
            "command": ".venv/bin/pytest tests/python/test_experiment_4539_null_delta_robustness.py -q --no-cov",
            "passed": True,
        },
    )

    assert payload["honest_verdict"] == "shipped: null_delta_false_positive_robustness_added"
    assert payload["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert payload["robustness_mechanism"]["carve_out"] == (
        "annotated_control_vs_treatment_null_delta"
    )
    assert payload["genuine_tautology_still_excluded"]["passed"] is True
    assert payload["preconditions_checked"]["summarize_artifact_help_exits_0"] is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]

    mod.validate_artifact(payload)
    output_path = mod.write_payload(tmp_path, payload, started_s=100.0, now_s=lambda: 100.25)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["duration_s"] == 0.25
    assert persisted["compute_bound"] is False
    mod.validate_artifact(persisted)

    invalid = dict(persisted)
    invalid["genuine_tautology_still_excluded"] = {"passed": False}
    with pytest.raises(ValueError, match="genuine_tautology_still_excluded"):
        mod.validate_artifact(invalid)


def test_req_capstone_4539_defensive_branches_and_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4539: blocked/error paths fail closed without fabricating success."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json_object(list_json)  # noqa: SLF001

    assert mod._tests_passed(True) is True  # noqa: SLF001
    assert mod._tests_passed("not-a-result") is False  # noqa: SLF001
    assert mod._blocked_reason({"summarize_artifact_help_exits_0": False}) == (  # noqa: SLF001
        "complete: null_delta_robustness_partial_blocked_summarize_artifact_help_exits_0"
    )
    ready = {
        "summarize_artifact_help_exits_0": True,
        "summarize_artifact_import_ok": True,
        "adversarial_verify_import_ok": True,
        "capstone_module_import_ok": True,
        "spec_has_req_4539": True,
        "spec_has_scenario_4539": True,
        "exp4524_fixture_exists": True,
        "scripts_research_conductor_modified": True,
    }
    assert mod._blocked_reason(ready) == (  # noqa: SLF001
        "complete: null_delta_robustness_partial_protected_conductor_modified"
    )

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )
    assert mod._git_path_modified(tmp_path, "scripts/research_conductor.py") is False  # noqa: SLF001

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("summarize", 1)
        ),
    )
    assert mod._summarize_help_exits_0(tmp_path) is False  # noqa: SLF001

    monkeypatch.setattr(
        mod,
        "check_preconditions",
        lambda root: {
            "summarize_artifact_help_exits_0": False,
            "summarize_artifact_import_ok": True,
            "adversarial_verify_import_ok": True,
            "capstone_module_import_ok": True,
            "spec_has_req_4539": True,
            "spec_has_scenario_4539": True,
            "exp4524_fixture_exists": True,
            "scripts_research_conductor_modified": False,
        },
    )
    blocked = mod.build_payload(tmp_path, tests_added_pass=False)
    assert blocked["honest_verdict"].startswith("complete: null_delta_robustness_partial_blocked")

    monkeypatch.setattr(
        mod,
        "check_preconditions",
        lambda root: {
            "summarize_artifact_help_exits_0": True,
            "summarize_artifact_import_ok": True,
            "adversarial_verify_import_ok": True,
            "capstone_module_import_ok": True,
            "spec_has_req_4539": True,
            "spec_has_scenario_4539": True,
            "exp4524_fixture_exists": True,
            "scripts_research_conductor_modified": False,
        },
    )
    valid = mod.build_payload(
        REPO,
        tests_added_pass={
            "command": "fixture",
            "passed": True,
        },
    )
    pending = mod.build_payload(REPO, tests_added_pass=False)
    assert pending["honest_verdict"] == (
        "complete: null_delta_robustness_partial_tests_or_assertions_pending"
    )
    invalid_cases = [
        ("__delete__", None, "missing required fields"),
        ("honest_verdict", "not-terminal", "honest_verdict"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("field_principles", [], "field_principles"),
        ("__drop_principle__", None, "missing field principle"),
        ("robustness_mechanism", [], "robustness_mechanism"),
        ("__headline_false__", None, "headline_numbers_remain_quarantined"),
        ("genuine_tautology_still_excluded", [], "genuine_tautology_still_excluded"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("__shipped_genuine_false__", None, "genuine_tautology_still_excluded"),
        ("__shipped_capstone_false__", None, "capstone_reads_diagnosis"),
        ("tests_added_pass", False, "tests_added_pass"),
        ("duration_s", "slow", "duration_s"),
        ("compute_bound", "no", "compute_bound"),
    ]
    for field, value, message in invalid_cases:
        changed = json.loads(json.dumps(valid))
        if field == "__delete__":
            changed.pop("honest_verdict")
        elif field == "__drop_principle__":
            changed["field_principles"].pop("honest_verdict")
        elif field == "__headline_false__":
            changed["robustness_mechanism"]["headline_numbers_remain_quarantined"] = False
        elif field == "__shipped_genuine_false__":
            changed["genuine_tautology_still_excluded"]["passed"] = False
        elif field == "__shipped_capstone_false__":
            changed["robustness_mechanism"]["capstone_reads_diagnosis"] = False
        else:
            changed[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(changed)

    output_path = tmp_path / mod.OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True)
    monkeypatch.setattr(mod, "build_payload", lambda root, tests_added_pass: valid)
    monkeypatch.setattr(mod, "write_payload", lambda root, payload, started_s: output_path)
    assert mod.run(tmp_path) == output_path


def test_req_capstone_4539_experiment_entrypoint_prints_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CAPSTONE-4539: requested direct script command has a thin entrypoint."""

    output_path = tmp_path / mod.OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"honest_verdict": "shipped: ok"}\n', encoding="utf-8")

    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_mod, "run", lambda root: output_path)

    assert runner_mod.main() == 0
    assert '"honest_verdict": "shipped: ok"' in capsys.readouterr().out
