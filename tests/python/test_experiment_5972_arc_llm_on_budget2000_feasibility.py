"""Focused tests for Exp5972's LLM-on budget-2000 feasibility artifact.

Spec refs: REQ-ARC-LLB2-5972,
SCENARIO-ARC-LLB2-5972-PRECONDITION-BLOCK,
SCENARIO-ARC-LLB2-5972-SEALED-LIVE-CELLS,
SCENARIO-ARC-LLB2-5972-PROJECTION-NO-FLAG-FLIP.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from carnot import experiment_5972_arc_llm_on_budget2000_feasibility as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "agentic-harness" / "spec.md"


def _preconditions(ok: bool = True) -> list[dict]:
    return [
        {"name": "registry_precheck", "available": True, "detail": "all target games cleared"},
        {
            "name": "mandated_qwen_gguf_cache",
            "available": ok,
            "detail": "cached" if ok else "missing",
        },
        {"name": "llama_cpp_cuda_offload", "available": ok, "detail": "CUDA"},
        {"name": "live_e3_path", "available": True, "detail": "make_carnot_agent/E3AgentPolicy"},
        {"name": "time_and_resource_budget", "available": True, "detail": "bounded"},
    ]


def _file_hashes() -> dict:
    return {
        "ops/arc_solve_registry.yaml": {
            "before_sha256": "sha256:registry",
            "after_sha256": "sha256:registry",
            "unchanged": True,
        },
        "python/carnot/agentic/arc_competition_agent.py": {
            "before_sha256": "sha256:agent",
            "after_sha256": "sha256:agent",
            "unchanged": True,
        },
    }


def _completed_rows() -> list[dict]:
    rows = []
    for index, cell in enumerate(mod.freeze_game_arm_seed_budget_and_timeout()["cells"]):
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "game": cell["game"],
                "arm": cell["arm"],
                "seed": cell["seed"],
                "budget": cell["budget"],
                "terminal_state": "completed",
                "generator_valid": True,
                "timeout": False,
                "elapsed_s": 100.0 + index,
                "model_load_s": 10.0 if index == 0 else 0.0,
                "llm": {
                    "calls": 2,
                    "prompt_tokens": 100 + index,
                    "completion_tokens": 30 + index,
                    "total_tokens": 130 + 2 * index,
                },
                "actions": 2000 if index % 2 else 1180,
                "progress": {"levels_completed": index % 3},
                "levels": index % 3,
                "plan_channel_openings": 1 if cell["game"] == "lp85" else 0,
                "gpu": {"peak_vram_mb": 22100 + index, "utilization_pct_mean": 71.0},
            }
        )
    return rows


def test_spec_contains_req_and_scenarios_for_5972():
    """REQ-ARC-LLB2-5972: implementation tests must be anchored to the spec."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    for token in (
        "REQ-ARC-LLB2-5972",
        "SCENARIO-ARC-LLB2-5972-PRECONDITION-BLOCK",
        "SCENARIO-ARC-LLB2-5972-SEALED-LIVE-CELLS",
        "SCENARIO-ARC-LLB2-5972-PROJECTION-NO-FLAG-FLIP",
        "results/experiment_5972_arc_llm_on_budget2000_feasibility.json",
    ):
        assert token in text


def test_sealed_cells_are_frozen_to_gain_games_lp85_pair_seed_and_budget():
    """SCENARIO-ARC-LLB2-5972-SEALED-LIVE-CELLS: cells and limits freeze before outcomes."""

    seal = mod.freeze_game_arm_seed_budget_and_timeout()
    cells = seal["cells"]
    assert seal["seed"] == 20260804
    assert seal["budget"] == 2000
    assert [cell["game"] for cell in cells[:7]] == ["dc22", "ft09", "s5i5", "su15", "lf52", "r11l", "cd82"]
    assert all(cell["arm"] == mod.GAIN_GAME_ARM for cell in cells[:7])
    lp85 = [cell for cell in cells if cell["game"] == "lp85"]
    assert [cell["arm"] for cell in lp85] == [
        "S_minus_frontier_llmon",
        "S_llmon_healthy_control",
    ]
    assert len({cell["cell_id"] for cell in cells}) == len(cells) == 9
    assert all(cell["budget"] == 2000 and cell["seed"] == 20260804 for cell in cells)


def test_blocked_artifact_is_complete_qwen_only_and_does_not_invent_rows():
    """SCENARIO-ARC-LLB2-5972-PRECONDITION-BLOCK: missing Qwen/CUDA blocks before live cells."""

    artifact = mod.build_artifact(
        preconditions=_preconditions(ok=False),
        rows=[],
        registry_hashes=_file_hashes()["ops/arc_solve_registry.yaml"],
        protected_file_hashes=_file_hashes(),
        duration_s=1.25,
    )
    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["model_specs"][0]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["model_specs"][0]["substitution_allowed"] is False
    assert "Qwen3.5" not in str(artifact["model_specs"])
    expected = artifact["expected_completed_missing_errored_timed_out_and_generator_invalid_cells"]
    assert expected["planned"] == 9
    assert expected["completed"] == 0
    assert expected["missing"] == 9
    assert artifact["per_cell_calls_tokens_actions_progress_levels_plan_channel_time_and_gpu_metrics"] == []
    assert artifact["no_automatic_flag_change_receipt"]["max_actions_changed"] is False
    assert artifact["no_new_solve_credit_receipt"]["registry_update_requested"] is False


def test_terminal_state_accounting_requires_one_terminal_state_per_planned_cell():
    """SCENARIO-ARC-LLB2-5972-SEALED-LIVE-CELLS: every cell has one honest terminal state."""

    cells = mod.freeze_game_arm_seed_budget_and_timeout()["cells"]
    rows = [
        {"cell_id": cells[0]["cell_id"], "terminal_state": "completed"},
        {"cell_id": cells[1]["cell_id"], "terminal_state": "timed_out"},
        {"cell_id": cells[2]["cell_id"], "terminal_state": "generator_invalid"},
        {"cell_id": cells[3]["cell_id"], "terminal_state": "errored"},
    ]
    counts = mod.account_cell_terminal_states(cells, rows)
    assert counts == {
        "planned": 9,
        "completed": 1,
        "missing": 5,
        "errored": 1,
        "timed_out": 1,
        "generator_invalid": 1,
        "unexpected": 0,
    }
    with pytest.raises(ValueError, match="duplicate terminal row"):
        mod.account_cell_terminal_states(cells, [rows[0], rows[0]])


def test_projection_uses_game_bootstrap_upper_bound_and_keeps_flag_receipt():
    """SCENARIO-ARC-LLB2-5972-PROJECTION-NO-FLAG-FLIP: feasibility is an interval, not advice."""

    projection = mod.twenty_five_game_projection(
        _completed_rows(),
        n_boot=250,
        seed=123,
        load_amortization_s=30.0,
    )
    assert projection["games_projected"] == 25
    assert projection["bootstrap_unit"] == "game"
    assert projection["upper_bound_s"] >= projection["mean_s"]
    assert projection["fits_12h_at_upper_bound"] is True

    slow = deepcopy(_completed_rows())
    for row in slow:
        row["elapsed_s"] = 4000.0
    too_slow = mod.twenty_five_game_projection(slow, n_boot=64, seed=5, load_amortization_s=0.0)
    assert too_slow["fits_12h_at_upper_bound"] is False


def test_validate_artifact_requires_required_fields_and_principles():
    """REQ-ARC-LLB2-5972: required fields carry principle provenance."""

    artifact = mod.build_artifact(
        preconditions=_preconditions(ok=True),
        rows=_completed_rows(),
        registry_hashes=_file_hashes()["ops/arc_solve_registry.yaml"],
        protected_file_hashes=_file_hashes(),
        duration_s=3.5,
    )
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_FIELDS).issubset(artifact)
    for field in mod.FIELD_PRINCIPLES:
        assert artifact["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]

    broken = deepcopy(artifact)
    broken.pop("model_specs")
    with pytest.raises(ValueError, match="missing required field"):
        mod.validate_artifact(broken)

    bad_principle = deepcopy(artifact)
    bad_principle["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_principle)


def test_reproducibility_checksum_excludes_duration_only():
    """REQ-ARC-LLB2-5972: checksum hashes evidence, not wall-clock noise."""

    artifact = mod.build_artifact(
        preconditions=_preconditions(ok=True),
        rows=_completed_rows(),
        registry_hashes=_file_hashes()["ops/arc_solve_registry.yaml"],
        protected_file_hashes=_file_hashes(),
        duration_s=3.5,
    )
    changed_duration = deepcopy(artifact)
    changed_duration["duration_s"] = 999.0
    assert mod.reproducibility_checksum(artifact) == mod.reproducibility_checksum(changed_duration)

    changed_rows = deepcopy(artifact)
    changed_rows["per_cell_calls_tokens_actions_progress_levels_plan_channel_time_and_gpu_metrics"][0][
        "elapsed_s"
    ] += 1.0
    assert mod.reproducibility_checksum(artifact) != mod.reproducibility_checksum(changed_rows)


def test_helper_branches_for_files_prior_rows_and_censored_projection(tmp_path: Path):
    """REQ-ARC-LLB2-5972: missing inputs and censored cells are explicit, never silent zeros."""

    missing = mod.file_hash_record(tmp_path / "absent.txt")
    assert missing == {
        "path": str(tmp_path / "absent.txt"),
        "exists": False,
        "sha256": None,
        "bytes": None,
    }
    present = tmp_path / "present.txt"
    present.write_text("abc", encoding="utf-8")
    assert mod.file_hash_record(present)["sha256"] == mod.sha256_bytes(b"abc")

    assert mod.default_budget400_receipt(tmp_path)["exists"] is False
    assert mod._blocked_reason(_preconditions(ok=True)) == "none"
    assert mod._test_exit_map([{"command": "cmd", "exit_code": 7}]) == {"cmd": 7}

    rows = [
        {"game": "a", "terminal_state": "errored", "elapsed_s": 10.0},
        {"game": "b", "terminal_state": "completed", "generator_valid": False, "elapsed_s": 10.0},
        {"game": "c", "terminal_state": "completed", "timeout": True, "elapsed_s": 10.0},
    ]
    assert mod.twenty_five_game_projection(rows)["projection_status"] == (
        "unavailable_no_complete_valid_cells"
    )


def test_terminal_accounting_counts_unexpected_and_unknown_as_errored():
    """SCENARIO-ARC-LLB2-5972-SEALED-LIVE-CELLS: malformed rows do not disappear."""

    cells = mod.freeze_game_arm_seed_budget_and_timeout()["cells"]
    counts = mod.account_cell_terminal_states(
        cells,
        [
            {"cell_id": "not-in-seal", "terminal_state": "completed"},
            {"cell_id": cells[0]["cell_id"], "terminal_state": "strange"},
        ],
    )
    assert counts["unexpected"] == 1
    assert counts["errored"] == 1


def test_checkpoint_resume_loads_only_sealed_rows_and_keeps_order(tmp_path: Path):
    """SCENARIO-ARC-LLB2-5972-SEALED-LIVE-CELLS: resume evidence is sealed-cell evidence."""

    cells = mod.freeze_game_arm_seed_budget_and_timeout()["cells"]
    first = {
        "cell_id": cells[0]["cell_id"],
        "terminal_state": "completed",
        "game": cells[0]["game"],
    }
    second = {
        "cell_id": cells[1]["cell_id"],
        "terminal_state": "completed",
        "game": cells[1]["game"],
    }
    unexpected = {"cell_id": "not-in-seal", "terminal_state": "completed", "game": "xx"}
    (tmp_path / f"{cells[1]['cell_id']}.json").write_text(
        mod.json.dumps(second), encoding="utf-8"
    )
    (tmp_path / f"{cells[0]['cell_id']}.json").write_text(
        mod.json.dumps(first), encoding="utf-8"
    )
    (tmp_path / "stray.json").write_text(mod.json.dumps(unexpected), encoding="utf-8")
    (tmp_path / "malformed.json").write_text("{", encoding="utf-8")
    (tmp_path / f"{cells[2]['cell_id']}.json.tmp").write_text("partial", encoding="utf-8")

    rows, receipt = mod.load_checkpoint_rows(tmp_path, cells)
    assert [row["cell_id"] for row in rows] == [cells[0]["cell_id"], cells[1]["cell_id"]]
    assert receipt["loaded_count"] == 2
    assert receipt["ignored_unsealed_count"] == 1
    assert receipt["ignored_malformed_count"] == 1
    assert receipt["missing_count"] == len(cells) - 2
    assert cells[2]["cell_id"] in receipt["missing_cell_ids"]
    assert [row["cell_id"] for row in mod.order_rows_by_seal(cells, [second, first])] == [
        cells[0]["cell_id"],
        cells[1]["cell_id"],
    ]


def test_build_artifact_underpowered_and_infeasible_verdicts_are_distinct():
    """SCENARIO-ARC-LLB2-5972-PROJECTION-NO-FLAG-FLIP: verdicts name data support."""

    underpowered = mod.build_artifact(
        preconditions=_preconditions(ok=True),
        rows=_completed_rows()[:2],
        registry_hashes=_file_hashes()["ops/arc_solve_registry.yaml"],
        protected_file_hashes=_file_hashes(),
        duration_s=2.0,
    )
    assert underpowered["status"] == "complete_underpowered"
    assert underpowered["honest_verdict"].startswith("complete_underpowered:")

    slow = _completed_rows()
    for row in slow:
        row["elapsed_s"] = 4000.0
    infeasible = mod.build_artifact(
        preconditions=_preconditions(ok=True),
        rows=slow,
        registry_hashes=_file_hashes()["ops/arc_solve_registry.yaml"],
        protected_file_hashes=_file_hashes(),
        duration_s=2.0,
    )
    assert infeasible["status"] == "complete_infeasible"
    assert infeasible["honest_verdict"].startswith("complete_infeasible:")


def test_validate_artifact_rejects_bad_prefix_model_substitution_flags_credit_and_checksum():
    """REQ-ARC-LLB2-5972: schema validation refuses the known ways to overclaim."""

    artifact = mod.build_artifact(
        preconditions=_preconditions(ok=True),
        rows=_completed_rows(),
        registry_hashes=_file_hashes()["ops/arc_solve_registry.yaml"],
        protected_file_hashes=_file_hashes(),
        duration_s=3.5,
    )

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance must be a mapping"):
        mod.validate_artifact(bad_provenance_type)

    bad_prefix = deepcopy(artifact)
    bad_prefix["honest_verdict"] = "complete: vague"
    bad_prefix["reproducibility_checksum"] = mod.reproducibility_checksum(bad_prefix)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_prefix)

    bad_model = deepcopy(artifact)
    bad_model["model_specs"][0]["hf_id"] = "unsloth/gemma-4-31B-it-GGUF"
    bad_model["reproducibility_checksum"] = mod.reproducibility_checksum(bad_model)
    with pytest.raises(ValueError, match="mandated Qwen3.6"):
        mod.validate_artifact(bad_model)

    legacy = deepcopy(artifact)
    legacy["model_specs"].append({"hf_id": "Qwen3.5-9B-MTP"})
    legacy["reproducibility_checksum"] = mod.reproducibility_checksum(legacy)
    with pytest.raises(ValueError, match="legacy model"):
        mod.validate_artifact(legacy)

    flag_flip = deepcopy(artifact)
    flag_flip["no_automatic_flag_change_receipt"]["max_actions_changed"] = True
    flag_flip["reproducibility_checksum"] = mod.reproducibility_checksum(flag_flip)
    with pytest.raises(ValueError, match="MAX_ACTIONS"):
        mod.validate_artifact(flag_flip)

    credit = deepcopy(artifact)
    credit["no_new_solve_credit_receipt"]["registry_update_requested"] = True
    credit["reproducibility_checksum"] = mod.reproducibility_checksum(credit)
    with pytest.raises(ValueError, match="registry solve credit"):
        mod.validate_artifact(credit)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:not-real"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum)
