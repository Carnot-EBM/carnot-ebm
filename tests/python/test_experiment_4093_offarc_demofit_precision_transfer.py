"""Tests for Exp 4093 OFF-ARC demo-fit precision replay.

Spec refs: REQ-VERIFY-4093, SCENARIO-VERIFY-4093.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

import exp4093_offarc_demofit_precision_transfer as runner


def _candidate(
    code: str,
    *,
    draw_index: int,
    visible: bool,
    hidden: bool,
) -> dict[str, Any]:
    return {
        "code": code,
        "draw_index": draw_index,
        "status": "ok",
        "visible_passes": [visible],
        "hidden_passes": [hidden],
        "visible_outputs": [2 if visible else 0],
        "generation_seconds": 0.01,
        "truncated": False,
        "error": None,
    }


def _exec(
    code: str,
    func_name: str,
    args: tuple[Any, ...],
    _timeout: float,
) -> tuple[Any, Exception | None]:
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)  # noqa: S102 - fixture execution for verifier tests.
        return namespace[func_name](*args), None
    except Exception as exc:
        return None, exc


def _precision_pool() -> dict[str, list[dict[str, Any]]]:
    correct = "def double(x):\n    return x * 2\n"
    overfit = "def double(x):\n    if x == 1:\n        return 2\n    return 0\n"
    wrong_visible = "def double(x):\n    return -1\n"
    return {
        "Task/0": [
            _candidate(correct, draw_index=0, visible=True, hidden=True),
            _candidate(overfit, draw_index=1, visible=True, hidden=False),
            _candidate(correct, draw_index=2, visible=True, hidden=True),
        ],
        "Task/1": [
            _candidate(correct, draw_index=0, visible=True, hidden=True),
            _candidate(wrong_visible, draw_index=1, visible=False, hidden=False),
        ],
    }


def _mutation_probes() -> dict[str, list[runner.MutationProbe]]:
    return {
        "Task/0": [runner.MutationProbe(func_name="double", args=(2,), expected=4)],
        "Task/1": [runner.MutationProbe(func_name="double", args=(3,), expected=6)],
    }


def test_req_4093_spec_declared() -> None:
    # REQ-VERIFY-4093: OpenSpec declares the offline precision replay contract.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4093",
        "SCENARIO-VERIFY-4093",
        "exp4093_offarc_demofit_precision_transfer.py",
        "P(hidden-pass | visible-pass)",
        "demofit_precision_raw",
        "demofit_precision_filtered",
        "filter_recall",
        "verifier_ensemble_against_cached_candidates",
    ):
        assert marker in spec


def test_req_4093_scores_raw_and_filtered_precision(tmp_path: Path) -> None:
    # REQ-VERIFY-4093: raw demo-fit precision and mutation-filter recall are candidate-level.
    metrics = runner.score_candidate_pool(
        _precision_pool(),
        mutation_probes_by_task=_mutation_probes(),
        executor=_exec,
    )
    assert metrics["n_tasks_scored"] == 2
    assert metrics["n_visible_pass_candidates"] == 4
    assert metrics["n_visible_hidden_pass_candidates"] == 3
    assert metrics["n_filtered_candidates"] == 3
    assert metrics["demofit_precision_raw"] == pytest.approx(0.75)
    assert metrics["demofit_precision_filtered"] == pytest.approx(1.0)
    assert metrics["filter_recall"] == pytest.approx(0.75)
    assert metrics["filter_raises_precision"] is True

    artifact = runner.build_artifact(
        metrics=metrics,
        preconditions_checked=[{"resource": "candidate_pool_cache", "available": True}],
        duration_s=0.25,
        candidate_pool_source=tmp_path / "pool.json",
    )
    runner.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: offarc_demofit_precision_0.75_filter_raises_to_1.00"
    )
    assert artifact["primitive_is_domain_general"] is True
    assert artifact["n_codex_calls"] == 0
    assert artifact["inference_substrate"] == runner.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"]


def test_scenario_4093_no_headroom_verdict_and_validation(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4093: saturated visible-pass rows get no-headroom, not a transfer claim.
    code = "def double(x):\n    return x * 2\n"
    pool = {"Task/0": [_candidate(code, draw_index=0, visible=True, hidden=True)]}
    metrics = runner.score_candidate_pool(
        pool,
        mutation_probes_by_task={"Task/0": [runner.MutationProbe("double", (2,), 4)]},
        executor=_exec,
    )
    artifact = runner.build_artifact(
        metrics=metrics,
        preconditions_checked=[],
        duration_s=0.1,
        candidate_pool_source=tmp_path / "pool.json",
    )
    runner.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: offarc_no_headroom_visible_pass_equals_hidden_pass"
    assert artifact["primitive_is_domain_general"] is False

    for field, value, message in (
        ("honest_verdict", "success: unsupported", "terminal prefix"),
        ("demofit_precision_raw", "0.75", "bare float"),
        ("primitive_is_domain_general", 1, "bare bool"),
        ("n_tasks_scored", 1.5, "bare int"),
        ("n_codex_calls", 1, "zero for offline replay"),
        ("random_seed", False, "bare int"),
        ("reproducibility_checksum", "", "non-empty"),
        ("inference_substrate", "live_llm_inference", "cached candidates"),
    ):
        poisoned = dict(artifact)
        poisoned[field] = value
        with pytest.raises(ValueError, match=message):
            runner.validate_artifact(poisoned)


def test_run_blocks_when_cached_pool_missing(tmp_path: Path) -> None:
    # REQ-VERIFY-4093: missing cached candidates write blocked_<resource> with zero calls.
    output = tmp_path / "experiment_4093.json"
    artifact = runner.run(
        candidate_artifact_path=tmp_path / "missing-4068.json",
        candidate_checkpoint_path=tmp_path / "missing-checkpoint.json",
        legacy_checkpoint_path=tmp_path / "missing-legacy.json",
        output_path=output,
        sandbox_importer=lambda: True,
        task_probe_builder=lambda: pytest.fail("task probes should not load"),
    )
    runner.validate_artifact(artifact)
    assert output.exists()
    assert artifact["honest_verdict"] == "blocked_cached_candidate_pool_missing"
    assert artifact["n_tasks_scored"] == 0
    assert artifact["n_codex_calls"] == 0


def test_run_writes_complete_artifact_from_fixture_cache(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4093: run() replays cached candidates and writes the terminal JSON.
    source = tmp_path / "experiment_4068.json"
    source.write_text(
        json.dumps({"candidate_pool": _precision_pool()}),
        encoding="utf-8",
    )
    output = tmp_path / "experiment_4093.json"
    artifact = runner.run(
        candidate_artifact_path=source,
        candidate_checkpoint_path=tmp_path / "unused.checkpoint.json",
        legacy_checkpoint_path=tmp_path / "unused-legacy.checkpoint.json",
        output_path=output,
        sandbox_importer=lambda: True,
        task_probe_builder=lambda: _mutation_probes(),
        executor=_exec,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["honest_verdict"] == (
        "complete: offarc_demofit_precision_0.75_filter_raises_to_1.00"
    )
    assert artifact["candidate_pool_source"] == str(source)


def test_cache_loader_falls_through_to_checkpoint_and_normalizes(tmp_path: Path) -> None:
    # REQ-VERIFY-4093: cache loading accepts checkpoint evaluations and skips malformed rows.
    bad_artifact = tmp_path / "bad.json"
    bad_artifact.write_text("{", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "evaluations_by_task": {
                    "Task/0": [_candidate("def f():\n    return 1\n", draw_index=0, visible=True, hidden=True)],
                    "Task/1": {"not": "a list"},
                }
            }
        ),
        encoding="utf-8",
    )
    pool, source = runner.load_cached_candidate_pool(
        bad_artifact,
        checkpoint,
        tmp_path / "missing-legacy.json",
    )
    assert source == checkpoint
    assert sorted(pool) == ["Task/0"]
    assert runner._pool_from_payload({"no_pool": True}) == {}


def test_build_mutation_probes_covers_aliases_and_probe_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SCENARIO-VERIFY-4093: public-derived mutation probes are deterministic and bounded.
    fake_runner = SimpleNamespace(
        base=SimpleNamespace(
            _mutate_public_arg=lambda arg: arg if arg == "same" else arg + 1
        ),
        _canonical_code=lambda row, corpus: f"{corpus}:{row['name']}",
        _canonical_expected=lambda code, func_name, args: (args[0] != 1, args[0] * 2),
    )
    monkeypatch.setitem(sys.modules, "offarc_power_evalplus_run", fake_runner)
    tasks = [
        SimpleNamespace(
            task_id="Mbpp/7",
            visible_tests=[
                SimpleNamespace(args=("same",), func_name="double"),
                SimpleNamespace(args=(0,), func_name="double"),
                SimpleNamespace(args=(1,), func_name="double"),
                SimpleNamespace(args=(2,), func_name="double"),
            ],
        ),
        SimpleNamespace(task_id="Missing/0", visible_tests=[]),
    ]
    probes = runner.build_mutation_probes(
        tasks,
        {"mbpp-7": ({"name": "fixture"}, "evalplus_mbpp")},
        max_probes_per_task=2,
    )
    assert probes["Mbpp/7"] == [
        runner.MutationProbe("double", (2,), 4),
        runner.MutationProbe("double", (3,), 6),
    ]
    assert probes["Missing/0"] == []
    assert runner._task_aliases("mbpp-7") == ["mbpp-7", "Mbpp/7"]


def test_scoring_and_verdict_edge_cases(tmp_path: Path) -> None:
    # REQ-VERIFY-4093: edge verdicts stay honest for no-visible and no-raise cases.
    skipped = runner.score_candidate_pool(
        {"Task/skip": [{"code": "def double(x):\n    return x\n"}]},
        mutation_probes_by_task={},
        executor=_exec,
    )
    assert skipped["n_tasks_scored"] == 0

    no_visible = runner.score_candidate_pool(
        {"Task/0": [_candidate("def double(x):\n    return 0\n", draw_index=0, visible=False, hidden=False)]},
        mutation_probes_by_task={},
        executor=_exec,
    )
    artifact = runner.build_artifact(
        metrics=no_visible,
        preconditions_checked=[],
        duration_s=0.0,
        candidate_pool_source=tmp_path / "pool.json",
    )
    assert artifact["honest_verdict"] == "complete: offarc_no_headroom_no_visible_pass_candidates"
    assert artifact["filter_recall"] == 0.0

    same_precision = runner.score_candidate_pool(
        {
            "Task/0": [
                _candidate("def double(x):\n    return x * 2\n", draw_index=0, visible=True, hidden=True),
                _candidate("def double(x):\n    return x * 2\n", draw_index=1, visible=True, hidden=False),
                {"code": "def double(x):\n    return x\n", "draw_index": 3},
            ]
        },
        mutation_probes_by_task={"Task/0": [runner.MutationProbe("double", (2,), 4)]},
        executor=_exec,
    )
    no_raise = runner.build_artifact(
        metrics=same_precision,
        preconditions_checked=[],
        duration_s=0.0,
        candidate_pool_source=tmp_path / "pool.json",
    )
    assert no_raise["honest_verdict"] == "complete: offarc_demofit_precision_0.50_filter_no_raise_0.50"
    assert runner._mutation_agrees(
        {"code": "def double(x):\n    return x\n"},
        [],
        executor=_exec,
        timeout=0.1,
    ) is False
    assert runner._mutation_agrees(
        {"code": ""},
        [runner.MutationProbe("double", (2,), 4)],
        executor=_exec,
        timeout=0.1,
    ) is False


def test_run_blocks_for_sandbox_and_probe_failures(tmp_path: Path) -> None:
    # REQ-VERIFY-4093: sandbox/probe resource misses become blocked artifacts.
    source = tmp_path / "experiment_4068.json"
    source.write_text(json.dumps({"candidate_pool": _precision_pool()}), encoding="utf-8")

    sandbox_blocked = runner.run(
        candidate_artifact_path=source,
        candidate_checkpoint_path=tmp_path / "unused.checkpoint.json",
        legacy_checkpoint_path=tmp_path / "unused-legacy.checkpoint.json",
        output_path=tmp_path / "sandbox.json",
        sandbox_importer=lambda: False,
        task_probe_builder=lambda: pytest.fail("task probes should not load"),
    )
    assert sandbox_blocked["honest_verdict"] == "blocked_sandbox_unavailable"

    probe_blocked = runner.run(
        candidate_artifact_path=source,
        candidate_checkpoint_path=tmp_path / "unused.checkpoint.json",
        legacy_checkpoint_path=tmp_path / "unused-legacy.checkpoint.json",
        output_path=tmp_path / "probes.json",
        sandbox_importer=lambda: True,
        task_probe_builder=lambda: (_ for _ in ()).throw(RuntimeError("no probes")),
    )
    assert probe_blocked["honest_verdict"] == "blocked_public_mutation_probes_unavailable"


def test_validate_artifact_rejects_missing_required_field(tmp_path: Path) -> None:
    # REQ-VERIFY-4093: malformed artifacts fail closed.
    artifact = runner.build_artifact(
        metrics=runner.score_candidate_pool(
            _precision_pool(),
            mutation_probes_by_task=_mutation_probes(),
            executor=_exec,
        ),
        preconditions_checked=[],
        duration_s=0.0,
        candidate_pool_source=tmp_path / "pool.json",
    )
    artifact.pop("filter_recall")
    with pytest.raises(ValueError, match="missing required field"):
        runner.validate_artifact(artifact)
