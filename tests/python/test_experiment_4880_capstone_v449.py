"""Tests for REQ-CAPSTONE-4880 / SCENARIO-CAPSTONE-4880."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4880_capstone_v449 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1(*, flagged: bool = False, fork: str | None = None) -> JsonDict:
    return {
        "experiment_id": 4871,
        "honest_verdict": "complete_generation_wall_fork_probe",
        "fork_verdict": fork,
        "per_game_fork": {
            "cd82": {
                "game": "cd82",
                "engine_heldout_accuracy": 0.25,
                "planned_bucket": "NEVER_ENUMERATED",
                "migrated": False,
            }
        },
        "coverage_migration_count": 0,
        "median_engine_heldout_accuracy": 0.0,
        "positive_control_game": "tu93",
        "positive_control_migrated": False,
        "planner_blind_to_banked_answer": True,
        "n_games_measured": 9,
        "generator_backend": "gpu0_cuda",
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "inference_substrate": "live_llm_inference",
        "flagged_adversarial": flagged,
    }


def _a1b(*, gate_skipped: bool = False) -> JsonDict:
    return {
        "experiment_id": 4872,
        "honest_verdict": "complete_cegis_gate_skipped_a1_high_accuracy"
        if gate_skipped
        else "complete_cegis_no_heldout_accuracy_lift_residual_positive_control_failed",
        "cegis_heldout_accuracy_delta_median": None if gate_skipped else 0.0,
        "cegis_heldout_accuracy_delta_ci95": [] if gate_skipped else [0.0, 0.0],
        "per_game_accuracy_delta": {},
        "positive_control_passed": False,
        "delta_on_truly_heldout_split": True,
        "live_path_reachable": True,
        "n_games_measured": 0 if gate_skipped else 9,
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "inference_substrate": "live_llm_inference",
    }


def _b1(
    *,
    diagnostic: bool = False,
    live_gpu0: bool = True,
    planner_blind: bool = True,
    positive_control: bool = False,
    numbers_match: bool = False,
    source_fork: str | None = None,
    a1b_trustworthy: bool = True,
    a1b_status: str = "ran",
) -> JsonDict:
    return {
        "experiment": "experiment_4876_fork_probe_inducer_audit",
        "experiment_id": 4876,
        "honest_verdict": "complete_a1_a1b_audited",
        "a1_source_fork_verdict": source_fork,
        "a1_source_honest_verdict": "complete_generation_wall_fork_probe",
        "a1_source_n_games_measured": 9,
        "a1_ran_live_on_gpu0": live_gpu0,
        "a1_genuinely_diagnostic": diagnostic,
        "planner_blind_confirmed": planner_blind,
        "positive_control_confirmed": positive_control,
        "numbers_match_fork": numbers_match,
        "a1_failure_reasons": [
            name
            for name, ok in {
                "a1_not_genuinely_diagnostic": diagnostic,
                "a1_not_live_on_gpu0": live_gpu0,
                "planner_not_blind": planner_blind,
                "positive_control_not_migrated": positive_control,
                "numbers_do_not_match_fork": numbers_match,
            }.items()
            if not ok
        ],
        "a1b_delta_median": None if a1b_status == "gate_skipped" else 0.0,
        "a1b_delta_ci95": [] if a1b_status == "gate_skipped" else [0.0, 0.0],
        "a1b_delta_trustworthy": a1b_trustworthy,
        "a1b_failure_reasons": [] if a1b_trustworthy else ["a1b_not_trustworthy"],
        "a1b_residual_reasons": ["a1b_positive_control_failed"],
        "checks": {
            "a1_numbers_match_fork": {
                "computed_fork_verdict": "INDUCER_CEILING",
                "passed": numbers_match,
                "reported_fork_verdict": source_fork,
            },
            "a1b_delta": {
                "status": a1b_status,
                "computed_median": None if a1b_status == "gate_skipped" else 0.0,
                "computed_ci95": [] if a1b_status == "gate_skipped" else [0.0, 0.0],
                "passed": a1b_trustworthy,
            },
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _levelup() -> JsonDict:
    return {
        "honest_verdict": "success_s5i5_levelup_banked",
        "target_game": "s5i5",
        "new_levels_banked": 1,
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "registry_update": {
            "prior_total_declared": 66,
            "new_total_declared": 67,
            "updated": True,
        },
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": True,
        "inference_substrate": "offline_arcade_reproduction_gate_no_llm",
    }


def _self_play() -> JsonDict:
    return {
        "honest_verdict": "success_self_play_checkpoint_refreshed",
        "target_game": "re86",
        "verifier_checkpoint_refreshed": True,
        "checkpoint_path": "models/arc_verifier_re86.json",
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "reproduction_gate": {"game": "re86", "claimed_level": 2, "reproduced": True},
        "search_state_count": 56,
        "solve_provenance": "live_agent_self_discovery",
        "inference_substrate": "live_llm_inference",
    }


def _heldout() -> JsonDict:
    return {
        "experiment_id": 4875,
        "honest_verdict": "complete_heldout_first_win_0.04_flat_genuine_null",
        "heldout_first_win_rate": 0.04,
        "first_win_baseline": 0.04,
        "prior_best_heldout_first_win_rate": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_first_win_delta_vs_prior_best": 0.0,
        "heldout_variant_attempts": 100,
        "heldout_first_win_ci": {"ci95": [0.0, 0.0]},
        "positive_control_passed": True,
        "parity_test_green": True,
        "checkpoint_emitted": True,
        "live_agent_ran": True,
        "generator_backend": "gpu0_cuda",
        "solve_provenance": "development_proxy",
        "inference_substrate": "live_llm_inference",
    }


def _package() -> JsonDict:
    return {
        "honest_verdict": "success_submission_package_ready",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"dry_build_ran": True, "package_builds": True},
        "agent_config_resolution": {"resolved": True},
        "model_path_resolution": {"resolved": True},
        "packaging_requirements_crosscheck": {"ok": True},
        "ready_package_regression_check": {"ok": True},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _hardware() -> JsonDict:
    return {
        "honest_verdict": "success_kv260_continuity_ok",
        "kv260_ssh_reachable": True,
        "board_state": {"captured": True, "hostname": "kv260", "uio_device_count": 5},
        "next_forward_step": "keep continuity rotation",
        "inference_substrate": "hardware_smoke",
    }


def _sota() -> JsonDict:
    return {
        "honest_verdict": "success_sota_ingestion_v450_frontier_mapped",
        "aimed_at_fork_verdict": "INDUCER_CEILING",
        "flagged_for_v450": [{"candidate": "test_time_dynamics_adaptation_loop"}],
        "methods_mapped": [{"method": "Test-time world-model and dynamics adaptation loop"}],
        "arxiv_ids_cited": ["2506.02918"],
        "sota_to_experiment_mapping_note": {"root_cause": "world-model inducer quality"},
        "upstream_artifacts": {"a1b_delta_median": 0.0},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(
    *,
    a1: JsonDict | None = None,
    a1b: JsonDict | None = None,
    b1: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "A1": a1 or _a1(),
        "A1B": a1b or _a1b(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": _self_play(),
        "HELDOUT": _heldout(),
        "B1_AUDIT": b1 or _b1(),
        "PACKAGE": _package(),
        "HARDWARE": _hardware(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {source: f"sha256:{source.lower()}" for source in mod.UPSTREAM_SOURCES}


def _summaries(*, a1_code: int = 0) -> dict[str, mod.SummarizerResult]:
    return {
        source: mod.SummarizerResult(
            ["summarize", spec.relative_path],
            a1_code if source == "A1" else 0,
            "LIVE re-check: CRITICAL" if source == "A1" and a1_code >= 2 else "clean",
            "",
        )
        for source, spec in mod.UPSTREAM_SOURCES.items()
    }


def test_req_capstone_4880_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4880: OpenSpec declares the .449 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4880_default_a1_non_test_but_a1b_trusted() -> None:
    """SCENARIO-CAPSTONE-4880: B1 gates void A1 while preserving trusted A1b delta."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 67},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    a1 = artifact["a1_generation_wall_fork_verdict"]
    assert artifact["honest_verdict"] == "complete_a1_generation_wall_non_test_capstone_ready"
    assert a1["verdict"] == "non_test_b1_untrusted"
    assert a1["fork_verdict"] is None
    assert a1["computed_fork_verdict"] == "INDUCER_CEILING"
    assert a1["trust_checks"] == {
        "a1_ran_live_on_gpu0": True,
        "planner_blind": True,
        "positive_control_migrated": False,
        "numbers_match_fork": False,
    }
    assert a1["trust_failure_reasons"] == [
        "a1_genuinely_diagnostic",
        "positive_control_migrated",
        "numbers_match_fork",
    ]
    assert artifact["a1b_inducer_swing"] == {
        "source": "A1B",
        "experiment_id": 4872,
        "b1_experiment_id": 4876,
        "ran": True,
        "gate_skipped": False,
        "cegis_heldout_accuracy_delta_median": 0.0,
        "delta_ci95": [0.0, 0.0],
        "a1b_delta_trustworthy": True,
        "b1_failure_reasons": [],
        "residual_reasons": ["a1b_positive_control_failed"],
        "status": "ran",
        "positive_control_passed": False,
        "delta_on_truly_heldout_split": True,
        "n_games_measured": 9,
    }
    assert artifact["scored_lever_state"] == {
        "level_up_banked": True,
        "heldout_first_win_rate": 0.04,
        "live_agent_ran": True,
        "submission_package_ready": True,
    }
    assert artifact["levelup_bank"]["reproducible_total_levels_after"] == 67
    assert artifact["self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert artifact["heldout_readiness"]["generator_backend"] == "gpu0_cuda"
    assert artifact["submission_package_state"]["decision"] == "package_ready_operator_only"
    assert artifact["hardware_continuity"]["decision"] == "kv260_reachable"
    assert artifact["sota_handoff"]["decision"] == "v450_frontier_handoff"
    assert artifact["sota_handoff"]["aimed_at_fork_verdict"] == "INDUCER_CEILING"
    assert artifact["reproducible_total_levels"] == 67
    assert len(artifact["cited_upstream_artifacts"]) == len(mod.UPSTREAM_SOURCES)
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4880_trusted_fork_and_gate_skipped_a1b() -> None:
    """SCENARIO-CAPSTONE-4880: trusted A1 headlines; gate-skipped A1b is not a null."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(fork="GUIDANCE_WALL"),
            a1b=_a1b(gate_skipped=True),
            b1=_b1(
                diagnostic=True,
                positive_control=True,
                numbers_match=True,
                source_fork="GUIDANCE_WALL",
                a1b_status="gate_skipped",
            ),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 67},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == (
        "complete_a1_generation_wall_guidance_wall_capstone_ready"
    )
    assert artifact["a1_generation_wall_fork_verdict"]["b1_trusted"] is True
    assert artifact["a1_generation_wall_fork_verdict"]["next_450_pivot"] == "guided_planner"
    assert artifact["a1b_inducer_swing"]["ran"] is False
    assert artifact["a1b_inducer_swing"]["gate_skipped"] is True
    assert artifact["a1b_inducer_swing"]["cegis_heldout_accuracy_delta_median"] is None
    assert mod.validate_artifact(artifact) == []


def test_run_capstone_invokes_summarizer_and_blocks_on_preconditions(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4880-BLOCKED-PRECONDITION: runtime checks fail closed."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 67\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4880\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")
    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        return mod.SummarizerResult(["summarize", relative_path], 0, "clean", "")

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)
    assert sorted(called) == sorted(spec.relative_path for spec in mod.UPSTREAM_SOURCES.values())
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"]["registry"]["yaml_loadable"] is True
    assert mod.validate_artifact(artifact) == []

    missing_root = tmp_path / "missing"
    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(missing_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    missing_registry = missing_root / mod.REGISTRY_RELATIVE_PATH
    missing_registry.parent.mkdir(parents=True, exist_ok=True)
    missing_registry.write_text(
        "schema_version: 1\nreproducible_total_levels: 67\n", encoding="utf-8"
    )
    missing_spec = missing_root / mod.SPEC_RELATIVE_PATH
    missing_spec.parent.mkdir(parents=True, exist_ok=True)
    missing_spec.write_text("REQ-CAPSTONE-4880\n", encoding="utf-8")
    missing_summarizer = missing_root / mod.SUMMARIZER_RELATIVE_PATH
    missing_summarizer.parent.mkdir(parents=True, exist_ok=True)
    missing_summarizer.write_text("# placeholder\n", encoding="utf-8")

    blocked = mod.run_capstone(root=missing_root, summarizer=summarizer)
    assert blocked["honest_verdict"] == "blocked_upstreams_missing"
    assert blocked["a1_generation_wall_fork_verdict"] == {}
    assert blocked["a1b_inducer_swing"] == {}
    assert blocked["cited_upstream_artifacts"] == []
    assert blocked["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(blocked) == []

    bad_registry_root = tmp_path / "bad_registry"
    for key, payload in _artifacts().items():
        _write_json(bad_registry_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    bad_registry = bad_registry_root / mod.REGISTRY_RELATIVE_PATH
    bad_registry.parent.mkdir(parents=True, exist_ok=True)
    bad_registry.write_text("schema_version: [unterminated\n", encoding="utf-8")
    bad_spec = bad_registry_root / mod.SPEC_RELATIVE_PATH
    bad_spec.parent.mkdir(parents=True, exist_ok=True)
    bad_spec.write_text("REQ-CAPSTONE-4880\n", encoding="utf-8")
    bad_summarizer = bad_registry_root / mod.SUMMARIZER_RELATIVE_PATH
    bad_summarizer.parent.mkdir(parents=True, exist_ok=True)
    bad_summarizer.write_text("# placeholder\n", encoding="utf-8")

    registry_blocked = mod.run_capstone(root=bad_registry_root, summarizer=summarizer)
    assert registry_blocked["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert registry_blocked["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(registry_blocked) == []


def test_validate_artifact_rejects_schema_violations_and_helpers() -> None:
    """SCENARIO-CAPSTONE-4880: malformed scorecards fail validation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 67},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    critical_summary = mod.SummarizerResult(["summarize"], 2, "CRITICAL", "")

    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "not terminal"}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_model"}
    )
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "67"}
    )
    assert "missing_principle:honest_verdict" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert "invalid_a1_generation_wall_fork_verdict" in mod.validate_artifact(
        {**artifact, "a1_generation_wall_fork_verdict": {"verdict": "maybe"}}
    )
    assert "invalid_a1b_inducer_swing" in mod.validate_artifact(
        {**artifact, "a1b_inducer_swing": {"ran": "yes"}}
    )
    assert "invalid_scored_lever_state" in mod.validate_artifact(
        {**artifact, "scored_lever_state": {"level_up_banked": "yes"}}
    )
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4871}]}
    )
    assert "invalid_flagged_artifacts_skipped" in mod.validate_artifact(
        {**artifact, "flagged_artifacts_skipped": [{"experiment_id": 4871}]}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert mod._experiment_id("A1", {"experiment_id": True}) == 4871
    assert mod._experiment_id("HARDWARE", {"experiment": 4878}) == 4878
    assert mod._is_skipped(_a1(flagged=True), None) == "flagged_adversarial"
    assert mod._is_skipped(_a1(), critical_summary) == "live_critical_recheck"
    assert mod._a1_generation_wall_fork_verdict(None, None, None, None) == {}
    assert mod._a1b_inducer_swing(None, None) == {}
    assert mod._trust_checks(None) == {
        "a1_ran_live_on_gpu0": False,
        "planner_blind": False,
        "positive_control_migrated": False,
        "numbers_match_fork": False,
    }
    assert mod._levelup_bank(None) == {}
    assert mod._self_play_checkpoint(None) == {}
    assert mod._heldout_readiness(None) == {}
    assert mod._submission_package_state(None) == {}
    assert mod._hardware_continuity(None) == {}
    assert mod._sota_handoff(None) == {}
    assert mod._first_blocker(
        summarizer_present=False,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=True,
        upstream_preconditions={},
    ) == "missing_summarizer"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=False,
        registry_loadable=True,
        spec_has_req=True,
        upstream_preconditions={},
    ) == "missing_registry"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=False,
        upstream_preconditions={},
    ) == "spec_missing_req_4880"

    invalid_fork = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(fork="UNKNOWN"), b1=_b1(source_fork="UNKNOWN")),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 67},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert invalid_fork["a1_generation_wall_fork_verdict"]["verdict"] == "non_test_b1_untrusted"

    flagged = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(flagged=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 67},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(a1_code=2),
        duration_s=0.001,
    )
    assert flagged["flagged_artifacts_skipped"] == [
        {
            "source": "A1",
            "experiment_id": 4871,
            "path": mod.UPSTREAM_SOURCES["A1"].relative_path,
            "reason": "flagged_adversarial",
            "sha256": "sha256:a1",
        }
    ]
    assert 4871 not in {row["experiment_id"] for row in flagged["cited_upstream_artifacts"]}
