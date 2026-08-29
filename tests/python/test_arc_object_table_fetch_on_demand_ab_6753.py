"""Focused tests for the paired object-table fetch-on-demand experiment.

Spec refs: REQ-ARC-WMTE-6753 and SCENARIO-ARC-WMTE-6753-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_6753_object_table_fetch_on_demand_ab as exp


def _model(tmp_path: Path, index: int) -> dict:
    spec = exp.MODEL_SPECS[index]
    path = tmp_path / spec["filename"]
    path.write_bytes(f"fixture-{spec['model_id']}".encode())
    return {
        **spec,
        "resolved": True,
        "model_path": str(path),
        "model_size_bytes": path.stat().st_size,
        "model_sha256": exp.sha256_file(path),
    }


def _science_row(
    tmp_path: Path,
    game: str,
    seed: int,
    arm: str,
    *,
    change_fidelity: float,
    prompt_tokens: int,
    transition_utility: float,
) -> dict:
    model = _model(tmp_path, 0)
    treatment = arm == exp.TREATMENT_ARM
    events = (
        [
            {
                "turn": 0,
                "parsed_tool": "find_objects",
                "parsed_arguments": {
                    "t": 0,
                    "which": "before",
                    "predicate_code": "def accept(obj):\n    return True",
                    "max_objects": 8,
                },
                "dispatch_result": {"ok": True, "objects": [], "response_bytes": 64},
                "bounded_response": "<tool_response>{\"ok\":true}</tool_response>",
            },
            {
                "turn": 1,
                "parsed_tool": "run_engine_on_transitions",
                "parsed_arguments": {"code": "def engine(grid, action, data): return grid"},
                "dispatch_result": {"ok": True, "change_fidelity": change_fidelity},
                "bounded_response": "<tool_response>{\"ok\":true}</tool_response>",
            },
        ]
        if treatment
        else []
    )
    accounting = exp.fetch_accounting(events)
    row = {
        "row_id": f"science:{game}:{seed}:{arm}",
        "row_kind": "science",
        "game": game,
        "seed": seed,
        "arm": arm,
        "arm_order": [exp.BASELINE_ARM, exp.TREATMENT_ARM],
        "model_id": model["model_id"],
        "model_role": model["role"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "production_route": exp.PRODUCTION_ROUTE,
        "context_requested": exp.CONTEXT_REQUESTED,
        "context_observed_by_model": exp.CONTEXT_REQUESTED,
        "gpu_receipt": {
            "assigned_device": {"physical_index": 0, "uuid": "GPU-0", "name": "RTX 3090"},
            "gpu_layers": {"requested": 999, "offloaded": 66, "total": 66},
            "peak_vram_mb": 18_000,
        },
        "live_model_invoked": True,
        "raw_prompt_sha256": exp.sha256_text(f"prompt:{game}:{seed}:{arm}"),
        "prompt_tokens": prompt_tokens,
        "tool_events": events,
        **accounting,
        "transition_result": {
            "n_heldout": 3,
            "true_changed_cells": 10,
            "correct_changed_cells": 6,
            "spurious_changed_cells": 2,
        },
        "change_fidelity": change_fidelity,
        "transition_utility": transition_utility,
        "duration_s": 1.0,
        "actions": [],
        "stop_reason": "early_stop_stable_best",
        "failure_class": None,
        "solve_claim": False,
    }
    row["row_sha256"] = exp.row_checksum(row)
    return row


def _sidecar_row(
    tmp_path: Path,
    arm: str,
    *,
    game: str = exp.GAME_IDS[0],
    seed: int = exp.SEEDS[0],
) -> dict:
    model = _model(tmp_path, 1)
    row = {
        "row_id": f"sidecar:{game}:{seed}:{arm}",
        "row_kind": "transport_sidecar",
        "game": game,
        "seed": seed,
        "arm": arm,
        "arm_order": [exp.BASELINE_ARM, exp.TREATMENT_ARM],
        "model_id": model["model_id"],
        "model_role": model["role"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "production_route": exp.PRODUCTION_ROUTE,
        "context_requested": exp.CONTEXT_REQUESTED,
        "context_observed_by_model": exp.CONTEXT_REQUESTED,
        "gpu_receipt": {
            "assigned_device": {"physical_index": 0, "uuid": "GPU-0", "name": "RTX 3090"},
            "gpu_layers": {"requested": 999, "offloaded": 41, "total": 41},
            "peak_vram_mb": 22_000,
        },
        "live_model_invoked": True,
        "raw_prompt_sha256": exp.sha256_text(f"sidecar:{arm}"),
        "prompt_tokens": 800 if arm == exp.BASELINE_ARM else 400,
        "tool_events": [],
        **exp.fetch_accounting([]),
        "transition_result": None,
        "change_fidelity": None,
        "transition_utility": None,
        "duration_s": 1.0,
        "actions": [],
        "stop_reason": "bounded_transport_complete",
        "failure_class": None,
        "solve_claim": False,
    }
    row["row_sha256"] = exp.row_checksum(row)
    return row


def _passing_preflight(models: list[dict]) -> dict:
    return {
        "all_passed": True,
        "checks": [
            {
                "check": "exp6752_ready",
                "expected": True,
                "observed": True,
                "passed": True,
            },
            {
                "check": "registry_no_new_or_duplicate_solve_target",
                "expected": True,
                "observed": {"experiment_target": None},
                "passed": True,
            },
        ],
        "source_receipts": {},
    }


def test_req_arc_wmte_6753_freezes_denominator_models_and_budget() -> None:
    """REQ-ARC-WMTE-6753 fixes all science pairs and keeps sidecars separate."""
    plan = exp.science_plan()
    sidecars = exp.sidecar_plan()
    assert len(plan) == 20 * 3 * 2
    assert len({(row["game"], row["seed"], row["arm"]) for row in plan}) == len(plan)
    assert set(exp.SEEDS) == {6100, 6101, 6102}
    assert [row["arm"] for row in plan[:2]] == [exp.BASELINE_ARM, exp.TREATMENT_ARM]
    assert [row["arm"] for row in plan[2:4]] == [exp.TREATMENT_ARM, exp.BASELINE_ARM]
    assert len(sidecars) == 2
    assert {row["row_kind"] for row in sidecars} == {"transport_sidecar"}
    assert {row["model_id"] for row in plan} == {"unsloth/Qwen3.8-27B-GGUF"}
    assert {row["model_id"] for row in sidecars} == {"unsloth/Qwen3.6-35B-A3B-GGUF"}
    design = exp.frozen_design()
    assert design["context_requested"] == 32_768
    assert design["noninferiority_margin"] == pytest.approx(0.259909)
    assert design["science_budgets"] == exp.SCIENCE_BUDGETS
    assert design["sidecar_budgets"] == exp.SIDECAR_BUDGETS
    assert design["solve_target"] is None


def test_scenario_arc_wmte_6753_arm_environment_isolation(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6753-ARM-ISOLATION changes only table presence."""
    model = _model(tmp_path, 0)
    base = {"KEEP": "yes", "CARNOT_ARC_OBJECT_PERCEPTION": "stale"}
    planned = {"game": "ls20", "seed": 6100, "row_kind": "science"}
    baseline = exp.worker_environment(base, model, {**planned, "arm": exp.BASELINE_ARM})
    treatment = exp.worker_environment(base, model, {**planned, "arm": exp.TREATMENT_ARM})
    assert baseline["CARNOT_ARC_OBJECT_PERCEPTION"] == "1"
    assert treatment["CARNOT_ARC_OBJECT_PERCEPTION"] == "0"
    difference = {
        key for key in baseline | treatment if baseline.get(key) != treatment.get(key)
    }
    assert difference == {"CARNOT_ARC_OBJECT_PERCEPTION"}
    assert baseline["CARNOT_ARC_INDUCE_N_CTX"] == "32768"
    assert baseline["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
    assert baseline["CARNOT_ARC_GENERATOR_SEED"] == "6100"


def test_scenario_arc_wmte_6753_prompt_construction_exact_removal() -> None:
    """SCENARIO-ARC-WMTE-6753-ARM-ISOLATION proves prompt-byte isolation."""
    object_table = "OBJECT STRUCTURE\nrow-1\nrow-2"
    schemas = "TOOLS\nfind_objects\nrun_engine_on_transitions"
    baseline = f"shared-prefix\n{object_table}\nshared-suffix\n{schemas}"
    treatment = f"shared-prefix\n\nshared-suffix\n{schemas}"
    receipt = exp.prompt_isolation_receipt(baseline, treatment, object_table, schemas)
    assert receipt["only_object_table_removed"] is True
    assert receipt["inline_object_table_chars"] == len(object_table)
    assert receipt["tool_schema_sha256"] == exp.sha256_text(schemas)
    with pytest.raises(ValueError, match="more than the inline object table"):
        exp.prompt_isolation_receipt(baseline, treatment + "!", object_table, schemas)
    with pytest.raises(ValueError, match="exactly once"):
        exp.prompt_isolation_receipt(baseline + object_table, treatment, object_table, schemas)


def test_scenario_arc_wmte_6753_useful_fetch_accounting() -> None:
    """SCENARIO-ARC-WMTE-6753-USEFUL-FETCH requires returned evidence and later code."""
    fetch = {
        "turn": 0,
        "parsed_tool": "find_objects",
        "dispatch_result": {"ok": True},
        "bounded_response": "<tool_response>{}</tool_response>",
    }
    later_good = {
        "turn": 1,
        "parsed_tool": "run_engine_on_transitions",
        "dispatch_result": {"ok": True},
    }
    assert exp.fetch_accounting([fetch, later_good]) == {
        "find_objects_attempts": 1,
        "find_objects_successes": 1,
        "fetched_evidence_entered_later_reasoning": True,
        "useful_fetches": 1,
    }
    assert exp.fetch_accounting([{**fetch, "bounded_response": ""}, later_good])["useful_fetches"] == 0
    assert exp.fetch_accounting([fetch])["useful_fetches"] == 0
    assert exp.fetch_accounting([fetch, {**later_good, "dispatch_result": {"ok": False}}])[
        "useful_fetches"
    ] == 0


def test_req_arc_wmte_6753_transition_result_uses_net_changed_cells() -> None:
    """REQ-ARC-WMTE-6753 fixes transition utility and keeps its raw evidence."""
    result = exp.transition_result(
        n_heldout=4,
        true_changed_cells=10,
        correct_changed_cells=7,
        spurious_changed_cells=2,
    )
    assert result["transition_utility"] == pytest.approx(0.5)
    assert result["transition_result"]["correct_changed_cells"] == 7
    assert exp.transition_result(
        n_heldout=0,
        true_changed_cells=0,
        correct_changed_cells=0,
        spurious_changed_cells=0,
    )["transition_utility"] == 0.0


def test_scenario_arc_wmte_6753_pairing_excludes_sidecar(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6753-PAIRING-AND-SIDECAR clusters at the game."""
    rows = []
    for game in ("g1", "g2"):
        rows.extend(
            [
                _science_row(
                    tmp_path,
                    game,
                    1,
                    exp.BASELINE_ARM,
                    change_fidelity=0.50,
                    prompt_tokens=1000,
                    transition_utility=0.20,
                ),
                _science_row(
                    tmp_path,
                    game,
                    1,
                    exp.TREATMENT_ARM,
                    change_fidelity=0.45,
                    prompt_tokens=600,
                    transition_utility=0.30,
                ),
            ]
        )
    hostile_sidecar = _sidecar_row(tmp_path, exp.BASELINE_ARM)
    hostile_sidecar["change_fidelity"] = -99.0
    hostile_sidecar["prompt_tokens"] = 999_999
    stats = exp.paired_statistics(
        rows + [hostile_sidecar], game_ids=("g1", "g2"), seeds=(1,), n_resamples=200
    )
    assert stats["n_games_paired"] == 2
    assert stats["n_seed_pairs"] == 2
    assert stats["change_fidelity_delta"] == pytest.approx(-0.05)
    assert stats["change_fidelity_ci95"] == pytest.approx([-0.05, -0.05])
    assert stats["transition_utility_delta"] == pytest.approx(0.10)
    assert stats["mean_prompt_token_savings"] == pytest.approx(400.0)
    assert stats["fetch_rate"] == 1.0
    assert stats["useful_fetch_rate"] == 1.0
    assert stats["harmful_regressions"] == []
    assert stats["noninferiority_passed"] is True


def test_req_arc_wmte_6753_pairing_fails_closed_on_missing_or_duplicate_rows(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6753 never changes the denominator to hide a missing pair."""
    baseline = _science_row(
        tmp_path,
        "g1",
        1,
        exp.BASELINE_ARM,
        change_fidelity=0.1,
        prompt_tokens=100,
        transition_utility=0.0,
    )
    with pytest.raises(ValueError, match="missing science row"):
        exp.paired_statistics([baseline], game_ids=("g1",), seeds=(1,), n_resamples=10)
    with pytest.raises(ValueError, match="duplicate science row"):
        exp.paired_statistics(
            [baseline, deepcopy(baseline)], game_ids=("g1",), seeds=(1,), n_resamples=10
        )


def test_scenario_arc_wmte_6753_completion_not_solve(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6753-COMPLETION-NOT-SOLVE separates completion and adoption."""
    science_plan = [
        {
            "row_id": f"science:g1:1:{arm}",
            "row_kind": "science",
            "game": "g1",
            "seed": 1,
            "arm": arm,
            "arm_order": [exp.BASELINE_ARM, exp.TREATMENT_ARM],
            "model_id": exp.MODEL_SPECS[0]["model_id"],
        }
        for arm in exp.ARMS
    ]
    sidecar_plan = [
        {
            "row_id": f"sidecar:g1:1:{arm}",
            "row_kind": "transport_sidecar",
            "game": "g1",
            "seed": 1,
            "arm": arm,
            "arm_order": [exp.BASELINE_ARM, exp.TREATMENT_ARM],
            "model_id": exp.MODEL_SPECS[1]["model_id"],
        }
        for arm in exp.ARMS
    ]
    rows = [
        _science_row(
            tmp_path,
            "g1",
            1,
            exp.BASELINE_ARM,
            change_fidelity=0.8,
            prompt_tokens=1000,
            transition_utility=0.5,
        ),
        _science_row(
            tmp_path,
            "g1",
            1,
            exp.TREATMENT_ARM,
            change_fidelity=0.0,
            prompt_tokens=500,
            transition_utility=0.0,
        ),
        _sidecar_row(tmp_path, exp.BASELINE_ARM, game="g1", seed=1),
        _sidecar_row(tmp_path, exp.TREATMENT_ARM, game="g1", seed=1),
    ]
    completion = exp.completion_and_adoption(
        rows,
        science_plan_rows=science_plan,
        sidecar_plan_rows=sidecar_plan,
        game_ids=("g1",),
        seeds=(1,),
        n_resamples=20,
    )
    assert completion["object_table_ab_completed"] is True
    assert completion["adoption_gate_passed"] is False
    assert completion["solve_claim"] is False
    assert completion["change_fidelity_delta"] == pytest.approx(-0.8)


def test_req_arc_wmte_6753_blocked_artifact_keeps_full_denominator(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6753 writes every planned blocked row without invoking a model."""
    models = [_model(tmp_path, 0), _model(tmp_path, 1)]
    calls: list[str] = []

    def blocked(_: list[dict]) -> dict:
        return {
            "all_passed": False,
            "checks": [
                {
                    "check": "exp6752_ready",
                    "expected": True,
                    "observed": False,
                    "passed": False,
                }
            ],
            "source_receipts": {},
        }

    artifact = exp.run(
        result_path=tmp_path / "blocked.json",
        resolver=lambda: models,
        preflight_fn=blocked,
        worker_runner=lambda *_: calls.append("called"),
        clock=iter((1_000_000_000, 2_000_000_000)).__next__,
    )
    assert calls == []
    assert len(artifact["rows"]) == 20 * 3 * 2 + 2
    assert all(row["failure_class"] == "preflight_blocked:exp6752_ready" for row in artifact["rows"])
    assert artifact["honest_verdict"] == "complete_blocked_object_table_ab:exp6752_ready"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["object_table_ab_completed"] is False
    assert artifact["adoption_gate_passed"] is False
    assert artifact["solve_claim"] is False
    assert artifact["live_model_invoked"] is False
    assert artifact["gate_check_summary"] == blocked(models)["checks"]
    assert set(artifact) <= set(artifact["field_principles"])
    assert exp.validate_artifact(artifact) == []
    assert json.loads((tmp_path / "blocked.json").read_text()) == artifact


def test_req_arc_wmte_6753_row_validation_detects_provenance_attacks(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6753 rejects missing CUDA, context, hashes, and solve claims."""
    row = _science_row(
        tmp_path,
        "g1",
        1,
        exp.BASELINE_ARM,
        change_fidelity=0.5,
        prompt_tokens=100,
        transition_utility=0.2,
    )
    assert exp.row_evidence_errors(row) == []
    attacks = {
        "context_observed_by_model": 16_384,
        "prompt_tokens": 0,
        "raw_prompt_sha256": None,
        "solve_claim": True,
        "model_id": exp.MODEL_SPECS[1]["model_id"],
    }
    for field, value in attacks.items():
        mutated = deepcopy(row)
        mutated[field] = value
        mutated["row_sha256"] = exp.row_checksum(mutated)
        assert exp.row_evidence_errors(mutated)
    mutated = deepcopy(row)
    mutated["gpu_receipt"]["gpu_layers"]["offloaded"] = 0
    mutated["row_sha256"] = exp.row_checksum(mutated)
    assert exp.row_evidence_errors(mutated)


def test_req_arc_wmte_6753_defensive_pure_branches(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6753 reports malformed schemas, pairs, rows, and JSON plainly."""
    with pytest.raises(ValueError, match="tool schemas"):
        exp.prompt_isolation_receipt("xTAB", "x", "TAB", "SCHEMA")

    treatment = _science_row(
        tmp_path,
        "g1",
        1,
        exp.TREATMENT_ARM,
        change_fidelity=0.1,
        prompt_tokens=10,
        transition_utility=0.0,
    )
    with pytest.raises(ValueError, match="missing science row"):
        exp.paired_statistics([treatment], game_ids=("g1",), seeds=(1,), n_resamples=2)

    row = _science_row(
        tmp_path,
        "g1",
        1,
        exp.BASELINE_ARM,
        change_fidelity=0.1,
        prompt_tokens=10,
        transition_utility=0.0,
    )
    mutations = {
        "production_route": "helper",
        "context_requested": 1,
        "live_model_invoked": False,
        "failure_class": "failed",
        "change_fidelity": None,
        "transition_utility": None,
    }
    for field, value in mutations.items():
        changed = deepcopy(row)
        changed[field] = value
        changed["row_sha256"] = exp.row_checksum(changed)
        assert field in exp.row_evidence_errors(changed) or exp.row_evidence_errors(changed)
    changed = deepcopy(row)
    changed["row_sha256"] = "sha256:wrong"
    assert "row_sha256" in exp.row_evidence_errors(changed)

    duplicate_plan = [
        {
            "row_id": row["row_id"],
            "row_kind": "science",
            "game": "g1",
            "seed": 1,
            "arm": exp.BASELINE_ARM,
        }
    ]
    receipt = exp.completion_and_adoption(
        [row, deepcopy(row)],
        science_plan_rows=duplicate_plan,
        sidecar_plan_rows=[],
        game_ids=("g1",),
        seeds=(1,),
        n_resamples=2,
    )
    assert receipt["row_completion_receipt"]["duplicate_row_ids"] == [row["row_id"]]

    object_path = tmp_path / "object.json"
    object_path.write_text('{"ok": true}')
    assert exp._load_json(object_path) == {"ok": True}
    list_path = tmp_path / "list.json"
    list_path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        exp._load_json(list_path)


def test_req_arc_wmte_6753_resolve_models_and_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6753 checks exact Exp6752 models and frozen prior evidence."""
    model_rows = []
    for index, spec in enumerate(exp.MODEL_SPECS):
        path = tmp_path / spec["filename"]
        path.write_bytes(f"model-{index}".encode())
        model_rows.append(
            {
                **spec,
                "model_path": str(path),
                "model_sha256": exp.sha256_file(path),
                "required_vram_mb": 100 + index,
            }
        )
    preflight = {
        "arc_context_tool_preflight_ready": True,
        "models_used": model_rows,
        "rows": [
            {
                "model_id": row["model_id"],
                "context_observed_by_model": exp.CONTEXT_REQUESTED,
                "gpu_layers": {"offloaded": 40},
            }
            for row in model_rows
        ],
    }
    prior = {
        "preregistration": {"content": {"roster": list(exp.GAME_IDS)}},
        "random_seed": exp.SEEDS[0],
        "NOISE_FLOOR_within_arm_replicate_spread": {
            "mean_spread": exp.NONINFERIORITY_MARGIN
        },
    }
    preflight_path = tmp_path / "6752.json"
    prior_path = tmp_path / "prior.json"
    registry_path = tmp_path / "registry.yaml"
    codex_path = tmp_path / "CODEX.md"
    preflight_path.write_text(json.dumps(preflight))
    prior_path.write_text(json.dumps(prior))
    registry_path.write_text("solves:\n  ls20: registered\n")
    codex_path.write_text("Last reconciled: 2026-08-26\n")
    monkeypatch.setattr(exp, "PREFLIGHT_PATH", preflight_path)
    monkeypatch.setattr(exp, "PRIOR_PATH", prior_path)
    monkeypatch.setattr(exp, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(exp, "CODEX_PATH", codex_path)
    monkeypatch.setattr(
        exp,
        "nvidia_smi_inventory",
        lambda: {
            "devices": [
                {
                    "index": 0,
                    "uuid": "GPU-0",
                    "name": "RTX 3090",
                    "memory_free_mb": 24_000,
                }
            ]
        },
    )
    from llama_cpp import llama_cpp

    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", lambda: True)
    resolved = exp.resolve_model_specs()
    assert [row["model_sha256"] for row in resolved] == [
        row["model_sha256"] for row in model_rows
    ]
    checked = exp.live_preflight(resolved)
    assert checked["all_passed"] is True
    assert all(row["passed"] is True for row in checked["checks"])
    assert set(checked["source_receipts"]) == {
        "exp6752",
        "prior_20260801",
        "solve_registry",
        "codex_instructions",
    }

    monkeypatch.setattr(
        llama_cpp,
        "llama_supports_gpu_offload",
        lambda: (_ for _ in ()).throw(RuntimeError("cuda probe")),
    )
    cuda_blocked = exp.live_preflight(resolved)
    cuda_check = next(row for row in cuda_blocked["checks"] if row["check"] == "llama_cpp_cuda_offload")
    assert cuda_check["passed"] is False
    assert "RuntimeError" in cuda_check["observed"]

    monkeypatch.setattr(exp, "PREFLIGHT_PATH", tmp_path / "missing.json")
    blocked = exp.live_preflight(resolved)
    assert blocked["all_passed"] is False
    assert blocked["checks"][0]["check"] == "required_source_artifacts"


def test_req_arc_wmte_6753_gpu_and_scoring_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6753 retains CUDA receipts and held-transition utility."""
    log_path = tmp_path / "server.log"
    log_path.write_text("offloaded 66/66 layers to GPU\n")
    proposer = SimpleNamespace(
        n_gpu_layers=999,
        _stderr_log_path=log_path,
        _proc=SimpleNamespace(pid=77),
        server_props=lambda: {"total_slots": 1},
        observed_total_slots=lambda: 1,
        observed_model_path=lambda: "/cache/model.gguf",
        observed_n_ctx=lambda: exp.CONTEXT_REQUESTED,
    )
    monkeypatch.setattr(
        exp,
        "nvidia_smi_inventory",
        lambda: {"devices": [{"index": 0, "uuid": "GPU-0", "name": "RTX 3090"}]},
    )
    receipt = exp._gpu_receipt(proposer, 0, 123)
    assert receipt["gpu_layers"]["offloaded"] == 66
    assert receipt["server_pid"] == 77
    assert receipt["context_observed_by_model"] == exp.CONTEXT_REQUESTED
    proposer._stderr_log_path = tmp_path
    assert exp._gpu_receipt(proposer, 0, 123)["gpu_layers"]["offloaded"] == 0

    assert exp._pid_vram_mb(None) == 0
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="bad\n77, 123\n77, nope\n78, 999\n"),
    )
    assert exp._pid_vram_mb(77) == 123
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    assert exp._pid_vram_mb(77) == 0

    held = [
        SimpleNamespace(
            grid=np.asarray([[0, 0], [0, 0]]),
            next_grid=np.asarray([[1, 0], [0, 0]]),
            action=1,
            data=None,
            level_before=0,
            level_after=0,
        )
    ]
    assert exp._score_written_engine("g", held, tmp_path)["failure_class"] == (
        "model_no_world_model_file"
    )
    target = tmp_path / "g" / "world_model.py"
    target.parent.mkdir()
    target.write_text("def nope():\n    return None\n")
    assert exp._score_written_engine("g", held, tmp_path)["failure_class"].startswith(
        "model_engine_unusable"
    )
    target.write_text(
        "import numpy as np\n"
        "def engine(grid, action, data):\n"
        "    out = np.asarray(grid).copy()\n"
        "    out[0, 0] = 1\n"
        "    return out\n"
    )
    scored = exp._score_written_engine("g", held, tmp_path)
    assert scored["ok"] is True
    assert scored["change_fidelity"] == 1.0
    assert scored["transition_utility"] == 1.0


@pytest.mark.memory_watchdog_skip
def test_req_arc_wmte_6753_window_and_public_policy_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-6753 rebuilds the prior split and enters through the public E3 factory."""
    from carnot.agentic import arc_actions_to_progress as progress
    from carnot.agentic import arc_world_model_trust_energy as trust
    from carnot.agentic import arc_competition_agent as competition

    transition = SimpleNamespace(name="row")
    monkeypatch.setattr(progress, "build_progress_window", lambda game: ([transition], [], 2))
    monkeypatch.setattr(trust, "_split_prefix_heldout", lambda rows: (rows, rows))
    windows = exp._build_windows(("g",))
    assert windows == {"g": {"shown": [transition], "held": [transition], "cell": 2}}
    monkeypatch.setattr(progress, "build_progress_window", lambda game: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        exp._build_windows(("g",))

    proposer = object()

    def factory(base_cls, *, cascade, proposer):
        class Agent(base_cls):
            def __init__(self, game: str) -> None:
                super().__init__(game)
                self._policy = competition.E3AgentPolicy(game, proposer=proposer)

        assert cascade is True
        return Agent

    monkeypatch.setattr(competition, "make_carnot_agent", factory)
    assert exp._public_policy_proposer("g", proposer) is proposer

    def wrong_policy(base_cls, *, cascade, proposer):
        class Agent(base_cls):
            def __init__(self, game: str) -> None:
                super().__init__(game)
                self._policy = object()

        return Agent

    monkeypatch.setattr(competition, "make_carnot_agent", wrong_policy)
    with pytest.raises(RuntimeError, match="did not construct"):
        exp._public_policy_proposer("g", proposer)

    substitute = object()

    def wrong_proposer(base_cls, *, cascade, proposer):
        class Agent(base_cls):
            def __init__(self, game: str) -> None:
                super().__init__(game)
                self._policy = competition.E3AgentPolicy(game, proposer=substitute)

        return Agent

    monkeypatch.setattr(competition, "make_carnot_agent", wrong_proposer)
    with pytest.raises(RuntimeError, match="substituted"):
        exp._public_policy_proposer("g", proposer)


def test_req_arc_wmte_6753_live_row_instrumentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6753 records prompts, tool evidence, metrics, and sidecar isolation."""
    from carnot.agentic import arc_induction_tool_loop as loop

    model = _model(tmp_path, 0)
    output_root = tmp_path / "e3"
    output_root.mkdir()
    planned = {
        "row_id": "science:g:1:table_inline",
        "row_kind": "science",
        "game": "g",
        "seed": 1,
        "arm": exp.BASELINE_ARM,
        "arm_order": list(exp.ARMS),
        "model_id": model["model_id"],
    }
    window = {"shown": [object()], "held": [object()], "cell": 1}

    class Proposer:
        def __init__(self) -> None:
            self._proc = SimpleNamespace(pid=77)
            self.last_tool_loop_stats = {}

        def induce(self, game, shown, cell):
            return loop.induce_with_tool_loop(self, game, shown, cell)

    proposer = Proposer()

    def fake_post(proposer, messages, **kwargs):
        return {"usage": {"prompt_tokens": 321}, "choices": [{"message": {"content": ""}}]}

    def fake_loop(proposer, game, shown, cell, *, tool_event_sink=None, **kwargs):
        loop._post_chat(proposer, [{"role": "user", "content": "PROMPT"}], turn=0)
        tool_event_sink.extend(
            [
                {
                    "turn": 0,
                    "parsed_tool": "find_objects",
                    "dispatch_result": {"ok": True},
                    "bounded_response": "<tool_response>{}</tool_response>",
                },
                {
                    "turn": 1,
                    "parsed_tool": "run_engine_on_transitions",
                    "dispatch_result": {"ok": True},
                },
            ]
        )
        proposer.last_tool_loop_stats = {
            "terminated_by": "best_zero_mismatches",
            "prompt_tokens_per_turn": [321],
        }
        return True, "ok"

    monkeypatch.setattr(loop, "_post_chat", fake_post)
    monkeypatch.setattr(loop, "induce_with_tool_loop", fake_loop)
    monkeypatch.setattr(exp, "_public_policy_proposer", lambda game, proposer: proposer)
    monkeypatch.setattr(exp, "_pid_vram_mb", lambda pid: 123)
    monkeypatch.setattr(
        exp,
        "_gpu_receipt",
        lambda proposer, device, peak: {
            "context_observed_by_model": exp.CONTEXT_REQUESTED,
            "gpu_layers": {"requested": 999, "offloaded": 66, "total": 66},
            "assigned_device": {"physical_index": 0, "uuid": "GPU-0", "name": "RTX"},
            "peak_vram_mb": peak,
        },
    )
    monkeypatch.setattr(
        exp,
        "_score_written_engine",
        lambda game, held, root: {
            "ok": True,
            "transition_result": {"n_heldout": 1},
            "change_fidelity": 0.5,
            "transition_utility": 0.25,
            "verifier_metrics": {"accuracy": 0.0},
            "engine_sha256": "sha256:engine",
        },
    )
    row, prompt = exp._run_live_row(planned, model, proposer, window, output_root)
    assert prompt == "PROMPT"
    assert row["prompt_tokens"] == 321
    assert row["useful_fetches"] == 1
    assert row["change_fidelity"] == 0.5
    assert row["failure_class"] is None
    assert row["stop_reason"] == "best_zero_mismatches"

    sidecar_model = _model(tmp_path, 1)
    sidecar = {
        **planned,
        "row_id": "sidecar:g:1:fetch_on_demand",
        "row_kind": "transport_sidecar",
        "arm": exp.TREATMENT_ARM,
        "model_id": sidecar_model["model_id"],
    }

    def unsuccessful_loop(proposer, game, shown, cell, *, tool_event_sink=None, **kwargs):
        loop._post_chat(proposer, [{"role": "user", "content": "SIDE"}], turn=0)
        proposer.last_tool_loop_stats = {"prompt_tokens_per_turn": [222]}
        return False, "turn_cap"

    monkeypatch.setattr(loop, "induce_with_tool_loop", unsuccessful_loop)
    sidecar_row, _ = exp._run_live_row(
        sidecar, sidecar_model, proposer, window, output_root
    )
    assert sidecar_row["failure_class"] is None
    assert sidecar_row["change_fidelity"] is None
    assert sidecar_row["prompt_tokens"] == 321

    monkeypatch.setattr(
        exp,
        "_public_policy_proposer",
        lambda game, proposer: (_ for _ in ()).throw(RuntimeError("route")),
    )
    failed, prompt = exp._run_live_row(planned, model, proposer, window, output_root)
    assert prompt == ""
    assert failed["failure_class"].startswith("live_row_exception:RuntimeError")

    monkeypatch.setattr(exp, "_public_policy_proposer", lambda game, proposer: proposer)
    monkeypatch.setattr(loop, "induce_with_tool_loop", fake_loop)
    monkeypatch.setattr(
        exp,
        "_score_written_engine",
        lambda game, held, root: {"ok": False, "failure_class": "bad_engine"},
    )
    unusable, _ = exp._run_live_row(planned, model, proposer, window, output_root)
    assert unusable["failure_class"] == "bad_engine"


def test_req_arc_wmte_6753_live_batch_prompt_pair_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6753 proves paired prompt isolation and checkpoints retained rows."""
    from carnot.agentic import arc_executable_world_model as world
    from carnot.agentic import arc_induction_tools as tools

    model = _model(tmp_path, 0)
    monkeypatch.setenv("CARNOT_ARC_E3_DIR", str(tmp_path / "e3"))
    monkeypatch.setattr(exp, "_build_windows", lambda games: {"g": {"shown": [], "held": [], "cell": 1}})
    monkeypatch.setattr(world, "_free_port", lambda: 1234)
    monkeypatch.setattr(world, "objects_block", lambda shown: "ROWS")
    monkeypatch.setattr(tools, "render_tool_schemas_for_prompt", lambda: "SCHEMAS")

    class FakeProposer:
        stopped = False

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr(world, "LocalGGUFProposer", FakeProposer)
    object_table = (
        "OBJECT STRUCTURE (same frames, connected-component view -- use object shape ids to "
        "track objects across the deltas above):\nROWS"
    )

    def row_runner(planned, model, proposer, window, output_root):
        prompt = (
            f"HEAD{object_table}TAILSCHEMAS"
            if planned["arm"] == exp.BASELINE_ARM
            else "HEADTAILSCHEMAS"
        )
        row = {
            **planned,
            "failure_class": None,
            "solve_claim": False,
        }
        row["row_sha256"] = exp.row_checksum(row)
        return row, prompt

    monkeypatch.setattr(exp, "_run_live_row", row_runner)
    plans = [
        {
            "row_id": f"science:g:1:{arm}",
            "row_kind": "science",
            "game": "g",
            "seed": 1,
            "arm": arm,
            "arm_order": list(exp.ARMS),
            "model_id": model["model_id"],
        }
        for arm in exp.ARMS
    ]
    checkpoint = tmp_path / "checkpoint.json"
    rows = exp.run_live_batch(model, plans, checkpoint_path=checkpoint)
    assert all(row["prompt_isolation_receipt"]["only_object_table_removed"] for row in rows)
    assert json.loads(checkpoint.read_text()) == rows

    monkeypatch.setattr(
        exp,
        "prompt_isolation_receipt",
        lambda *args: (_ for _ in ()).throw(ValueError("attack")),
    )
    attacked = exp.run_live_batch(model, plans)
    assert all(row["failure_class"] == "prompt_arm_isolation_failed" for row in attacked)


def test_req_arc_wmte_6753_batch_subprocess_terminal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6753 retains success, crash, timeout, and malformed worker outputs."""
    model = _model(tmp_path, 0)
    plans = exp.science_plan()[:2]

    def completed(command, **kwargs):
        output = Path(command[command.index("--worker-output") + 1])
        output.write_text(json.dumps([{"row_id": plans[0]["row_id"]}]))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(exp.subprocess, "run", completed)
    assert exp.run_model_batch_subprocess(model, plans) == [{"row_id": plans[0]["row_id"]}]

    def malformed(command, **kwargs):
        output = Path(command[command.index("--worker-output") + 1])
        output.write_text("{}")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(exp.subprocess, "run", malformed)
    with pytest.raises(ValueError, match="row list"):
        exp.run_model_batch_subprocess(model, plans)

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stderr="worker died"),
    )
    crashed = exp.run_model_batch_subprocess(model, plans)
    assert len(crashed) == 2
    assert all(row["failure_class"].startswith("worker_process_failed:returncode=7") for row in crashed)

    def timed_out(command, **kwargs):
        output = Path(command[command.index("--worker-output") + 1])
        partial = exp._failed_row(plans[0], model, "live_partial")
        output.write_text(json.dumps([partial]))
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(exp.subprocess, "run", timed_out)
    timed = exp.run_model_batch_subprocess(model, plans)
    assert [row["row_id"] for row in timed] == [row["row_id"] for row in plans]
    assert timed[1]["failure_class"].startswith("worker_process_failed:TimeoutExpired")


def _full_passing_rows(tmp_path: Path) -> list[dict]:
    rows = []
    for planned in exp.science_plan():
        rows.append(
            _science_row(
                tmp_path,
                planned["game"],
                planned["seed"],
                planned["arm"],
                change_fidelity=0.5,
                prompt_tokens=1000 if planned["arm"] == exp.BASELINE_ARM else 500,
                transition_utility=0.25,
            )
        )
        rows[-1]["arm_order"] = planned["arm_order"]
        rows[-1]["row_sha256"] = exp.row_checksum(rows[-1])
    rows.extend(
        [
            _sidecar_row(tmp_path, exp.BASELINE_ARM),
            _sidecar_row(tmp_path, exp.TREATMENT_ARM),
        ]
    )
    return rows


def test_req_arc_wmte_6753_complete_artifact_and_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6753 reduces complete rows and detects artifact-level attacks."""
    models = [_model(tmp_path, 0), _model(tmp_path, 1)]
    rows = _full_passing_rows(tmp_path)
    artifact = exp.build_artifact(
        rows=rows,
        models=models,
        preflight=_passing_preflight(models),
        started_ns=1_000,
        finished_ns=2_000,
    )
    assert artifact["object_table_ab_completed"] is True
    assert artifact["adoption_gate_passed"] is True
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"] == "complete_object_table_ab_adopt_fetch_on_demand"
    assert exp.validate_artifact(artifact) == []

    attacks = {
        "schema": "bad",
        "inference_substrate": "cpu",
        "context_requested": 1,
        "solve_claim": True,
        "verdict_class": "unknown",
    }
    expected = {
        "schema": "schema",
        "inference_substrate": "inference_substrate",
        "context_requested": "context_requested",
        "solve_claim": "solve_claim",
        "verdict_class": "verdict_class",
    }
    for field, value in attacks.items():
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected[field] in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["extra"] = True
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "field_principles_incomplete" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"] = list(reversed(changed["rows"]))
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "row_denominator_or_order" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["mean_prompt_token_savings"] = -1
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "reduction:mean_prompt_token_savings" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum" in exp.validate_artifact(changed)

    no_adopt = deepcopy(rows)
    for row in no_adopt:
        if row["row_kind"] == "science" and row["arm"] == exp.TREATMENT_ARM:
            row["prompt_tokens"] = 1_500
            row["row_sha256"] = exp.row_checksum(row)
    null_artifact = exp.build_artifact(
        rows=no_adopt,
        models=models,
        preflight=_passing_preflight(models),
        started_ns=1,
        finished_ns=2,
    )
    assert null_artifact["verdict_class"] == "null"

    reduction = exp.completion_and_adoption([])
    monkeypatch.setattr(exp, "completion_and_adoption", lambda rows: reduction)
    partial = exp.build_artifact(
        rows=[],
        models=models,
        preflight=_passing_preflight(models),
        started_ns=1,
        finished_ns=2,
    )
    assert partial["verdict_class"] == "partial"


def test_req_arc_wmte_6753_blocked_validator_attacks(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6753 detects blocked completion and failure-shape attacks."""
    models = [_model(tmp_path, 0), _model(tmp_path, 1)]
    blocked = {
        "all_passed": False,
        "checks": [{"check": "gate", "observed": False, "passed": False}],
        "source_receipts": {},
    }
    rows = [exp._failed_row(row, models[0], "preflight_blocked:gate") for row in exp.science_plan()]
    rows.extend(exp._failed_row(row, models[1], "preflight_blocked:gate") for row in exp.sidecar_plan())
    artifact = exp.build_artifact(
        rows=rows,
        models=models,
        preflight=blocked,
        started_ns=1,
        finished_ns=2,
    )
    changed = deepcopy(artifact)
    changed["object_table_ab_completed"] = True
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "blocked_completion" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"][0]["failure_class"] = "other"
    changed["rows"][0]["row_sha256"] = exp.row_checksum(changed["rows"][0])
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "blocked_rows" in exp.validate_artifact(changed)


def test_req_arc_wmte_6753_run_live_branch_and_invalid_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6753 runs both isolated batches and refuses an invalid artifact."""
    models = [_model(tmp_path, 0), _model(tmp_path, 1)]
    rows = _full_passing_rows(tmp_path)
    science = [row for row in rows if row["row_kind"] == "science"]
    sidecars = [row for row in rows if row["row_kind"] == "transport_sidecar"]
    calls: list[str] = []

    def worker(model, plans):
        calls.append(model["model_id"])
        return deepcopy(science if model["model_id"] == models[0]["model_id"] else sidecars)

    result = exp.run(
        result_path=tmp_path / "complete.json",
        resolver=lambda: models,
        preflight_fn=_passing_preflight,
        worker_runner=worker,
        clock=iter((1, 2)).__next__,
    )
    assert calls == [model["model_id"] for model in models]
    assert result["object_table_ab_completed"] is True
    assert json.loads((tmp_path / "complete.json").read_text()) == result

    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["attack"])
    with pytest.raises(ValueError, match="invalid Exp6753"):
        exp.run(
            result_path=tmp_path / "invalid.json",
            resolver=lambda: models,
            preflight_fn=lambda values: {
                "all_passed": False,
                "checks": [{"check": "gate", "passed": False}],
                "source_receipts": {},
            },
            clock=iter((1, 2)).__next__,
        )


def test_req_arc_wmte_6753_worker_entry_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-6753 keeps the repository CLI and owned worker entry auditable."""
    model = _model(tmp_path, 0)
    job_path = tmp_path / "job.json"
    output_path = tmp_path / "rows.json"
    job_path.write_text(json.dumps({"model": model, "planned_rows": exp.science_plan()[:1]}))
    monkeypatch.setattr(exp, "run_live_batch", lambda model, rows, checkpoint_path: [{"ok": True}])
    assert exp._worker_entry(job_path, output_path) == 0
    assert json.loads(output_path.read_text()) == [{"ok": True}]

    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        exp,
        "_worker_entry",
        lambda job, output: calls.append((job, output)) or 0,
    )
    assert exp.main(["--worker-job", str(job_path), "--worker-output", str(output_path)]) == 0
    assert calls == [(job_path, output_path)]
    with pytest.raises(SystemExit):
        exp.main(["--worker-job", str(job_path)])

    monkeypatch.setattr(
        exp,
        "run",
        lambda: {
            "object_table_ab_completed": True,
            "adoption_gate_passed": False,
            "honest_verdict": "complete_object_table_ab_do_not_adopt",
        },
    )
    assert exp.main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["completed"] is True
    assert printed["adopted"] is False
