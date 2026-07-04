"""Tests for Exp 5211 GAP-4 SOTA local candidate expansion.

Spec refs: REQ-REPORT-5211, SCENARIO-REPORT-5211,
SCENARIO-REPORT-5211-BLOCKED-SOTA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5211_gap4_sota_local_candidate_expansion_v477 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _task(index: int, *, task_id: str | None = None) -> JsonDict:
    base = index % 3
    return {
        "task_id": task_id or f"human_replay:test:{index}",
        "source": "unit",
        "demos": [
            {
                "input": [[base, base + 1], [base + 2, base + 3]],
                "output": [[base + 1, base + 2], [base + 3, base + 4]],
            },
            {
                "input": [[base + 4, base + 5]],
                "output": [[base + 5, base + 6]],
            },
        ],
        "test_input": [[base + 7, base + 8]],
        "test_shape": [1, 2],
    }


def _plus_one_code() -> str:
    return """
```python
def transform(grid):
    return [[int(cell) + 1 for cell in row] for row in grid]
```
"""


def _specs() -> list[JsonDict]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/qwen.gguf",
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "model_path": "/models/gemma26.gguf",
        },
    ]


def test_req_report_5211_spec_declares_candidate_expansion_contract() -> None:
    """REQ-REPORT-5211: OpenSpec declares the v477 candidate-pool contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5211",
        "SCENARIO-REPORT-5211",
        "SCENARIO-REPORT-5211-BLOCKED-SOTA",
        mod.RESULT_RELATIVE_PATH,
        mod.CHECKPOINT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_report_5211_feasibility_guard_accepts_only_safe_demo_perfect_code() -> None:
    """REQ-REPORT-5211: parse, restricted exec, demo-fit, and shape guards all apply."""

    task = _task(0)
    ok = mod.guard_candidate(task, mod.extract_transform_code(_plus_one_code()) or "")
    assert ok.accepted is True
    assert ok.reason == "accepted"
    assert ok.demo_perfect is True
    assert ok.output_shape_matches is True

    for bad_code, reason in (
        ("import os\ndef transform(grid):\n    return grid", "forbidden_ast"),
        ("def transform(grid):\n    return open('/tmp/x').read()", "forbidden_ast"),
        ("def transform(grid):\n    return [[0, 0], [0, 0]]", "demo_mismatch"),
        ("def nope(grid):\n    return grid", "missing_transform"),
    ):
        result = mod.guard_candidate(task, bad_code)
        assert result.accepted is False
        assert result.reason == reason


def test_scenario_report_5211_repair_uses_demo_failures_and_checkpoints_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5211: invalid candidates are repaired or rejected with checkpoints."""

    accepted, attempts = mod.process_task_row(
        task=_task(1),
        raw_text="no code here",
        model_spec=_specs()[0],
        checkpoint_path=tmp_path / mod.CHECKPOINT_RELATIVE_PATH,
        prior_events=[],
        repair_budget=2,
    )

    checkpoint = json.loads((tmp_path / mod.CHECKPOINT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert accepted["accepted"] is True
    assert accepted["guard_status"] == "accepted"
    assert accepted["repair_strategy"] == "demo_lookup_same_shape"
    assert attempts == 1
    assert checkpoint["events"][0]["accepted"] is True
    assert "test_output" not in json.dumps(accepted)

    rejected, attempts = mod.process_task_row(
        task=_task(2),
        raw_text="def transform(grid):\n    return [[0, 0], [0, 0]]",
        model_spec=_specs()[0],
        checkpoint_path=tmp_path / mod.CHECKPOINT_RELATIVE_PATH,
        prior_events=checkpoint["events"],
        repair_budget=0,
    )

    checkpoint = json.loads((tmp_path / mod.CHECKPOINT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert rejected["accepted"] is False
    assert rejected["guard_status"] == "demo_mismatch"
    assert attempts == 0
    assert len(checkpoint["events"]) == 2
    assert checkpoint["events"][1]["accepted"] is False


def test_scenario_report_5211_run_builds_usable_pool_without_significance_claim(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5211: >=120 feasible rows set the exp5212 bare gate only."""

    tasks = [_task(i) for i in range(125)]
    prompts: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        cached_pair_loader=lambda: _specs(),
        task_loader=lambda _root, _exclude, _budget: tasks,
        text_generator_factory=lambda _spec: (lambda prompt: prompts.append(prompt) or _plus_one_code()),
        max_live_prompts=1,
        repair_budget=2,
        task_budget=125,
    )

    assert len(prompts) == 1
    assert "test_output" not in prompts[0]
    assert artifact["candidate_pool_n"] == 120
    assert artifact["gap4_expansion_usable"] is True
    assert artifact["sota_gguf_resolved"] is True
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert artifact["model_specs"] == _specs()
    assert artifact["accepted_rows"] == 120
    assert artifact["rejected_rows"] == 0
    assert artifact["repair_attempts"] == 119
    assert artifact["leakage_audit_passed"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert "significance" not in artifact["honest_verdict"].lower()
    assert artifact["honest_verdict"].startswith("complete_")
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    checkpoint = json.loads((tmp_path / mod.CHECKPOINT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert len(checkpoint["events"]) == 120
    assert checkpoint["accepted_count"] == 120


def test_scenario_report_5211_blocked_sota_cache_writes_bare_gate_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5211-BLOCKED-SOTA: missing SOTA cache blocks honestly."""

    artifact = mod.run(
        root=tmp_path,
        cached_pair_loader=lambda: None,
        task_loader=lambda _root, _exclude, _budget: [_task(0)],
        text_generator_factory=lambda _spec: (lambda _prompt: _plus_one_code()),
    )

    assert artifact["candidate_pool_n"] == 0
    assert artifact["gap4_expansion_usable"] is False
    assert artifact["sota_gguf_resolved"] is False
    assert artifact["accepted_rows"] == 0
    assert artifact["models_used"] == []
    assert artifact["honest_verdict"].startswith("blocked_sota_gguf_not_cached")
    assert artifact["legacy_tiny_fallback_expected_quality"] == "poor"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5211_loads_frame_transition_tasks_and_excludes_exp5197(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-5211: source tasks are same-shape and not Exp 5197 scored IDs."""

    shard_dir = tmp_path / mod.HUMAN_REPLAY_RELATIVE_DIR / "shards"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "train-00000.jsonl"
    rows = [
        {"env": "g", "source_row_index": 7, "step_index": 0, "frame": [[0, 0], [0, 0]]},
        {"env": "g", "source_row_index": 7, "step_index": 1, "frame": [[1, 0], [0, 0]]},
        {"env": "g", "source_row_index": 7, "step_index": 2, "frame": [[1, 2], [0, 0]]},
        {"env": "g", "source_row_index": 7, "step_index": 3, "frame": [[1, 2], [3, 0]]},
        {"env": "g", "source_row_index": 7, "step_index": 4, "frame": [[1, 2], [3, 4]]},
    ]
    shard.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    tasks = mod.load_frame_transition_tasks(tmp_path, exclude_task_ids={"human_replay:g:7:0"}, budget=4)

    assert [task["task_id"] for task in tasks] == ["human_replay:g:7:1"]
    assert len(tasks[0]["demos"]) == 2
    assert tasks[0]["test_shape"] == [1, 2]
    assert tasks[0]["test_input"] == [[3, 0]]
    assert "test_output" not in tasks[0]


def test_artifact_schema_rejects_wrapped_gate_fields_and_leakage() -> None:
    """REQ-REPORT-5211: schema checks protect bare gates and leakage audit."""

    artifact = mod.build_artifact(
        events=[
            {
                "accepted": True,
                "task_id": f"safe-{idx}",
                "code": "def transform(grid):\n    return grid",
                "guard_status": "accepted",
                "live_prompted": idx == 0,
                "model_hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            }
            for idx in range(120)
        ],
        model_specs=_specs(),
        models_used=[],
        sota_gguf_resolved=True,
        repair_attempts=0,
        source_task_budget=120,
        source_task_count=120,
        checkpoint_path=mod.CHECKPOINT_RELATIVE_PATH,
        duration_s=1.0,
    )
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert mod.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["candidate_pool_n"] = {"value": 120}
    bad["gap4_expansion_usable"] = {"value": True}
    bad["leakage_audit_passed"] = True
    bad["candidate_rows"] = [{"task_id": "x", "code": "def transform(grid):\n    return test_output"}]
    bad["honest_verdict"] = "success_gap4_significance_claim"
    bad["inference_substrate"] = "cached_only"
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(bad)

    assert "candidate_pool_n_bare_int" in errors
    assert "gap4_expansion_usable_bare_bool" in errors
    assert "leakage_audit_passed_false" in errors
    assert "honest_verdict_no_significance_claim" in errors
    assert "inference_substrate" in errors
    assert "field_principles" in errors
    assert "reproducibility_checksum" in errors

    with pytest.raises(ValueError):
        mod.write_artifact(Path("/tmp"), bad)


def test_req_report_5211_defensive_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-5211: defensive helper branches remain deterministic."""

    assert mod._shape(7) == []
    assert mod._shape([1, 2]) == [2]
    assert mod._normalize_grid(7) == [[7]]
    assert mod._normalize_grid([1, 2]) == [[1, 2]]
    assert mod._normalize_grid(__import__("numpy").array([[1]])).__class__ is list
    assert mod.extract_transform_code(123) is None  # type: ignore[arg-type]
    assert mod.extract_transform_code("prefix\ndef transform(grid):\n    return grid") == (
        "def transform(grid):\n    return grid"
    )

    task = _task(0)
    assert mod.guard_candidate(task, "def transform(:").reason == "syntax_error"
    assert mod.guard_candidate(
        task,
        "def transform(grid):\n    subprocess.run([])\n    return grid",
    ).reason == "forbidden_ast"
    assert mod.guard_candidate(task, "def transform(grid):\n    subprocess\n    return grid").reason == (
        "forbidden_ast"
    )
    assert mod._forbidden_ast_reason(
        __import__("ast").parse("def transform(grid):\n    len(grid)\n    return grid")
    ) is None
    assert mod.guard_candidate(task, "def transform(grid):\n    return grid.open()").reason == (
        "forbidden_ast"
    )
    assert mod.guard_candidate(
        task, "raise RuntimeError('x')\ndef transform(grid):\n    return grid"
    ).reason.startswith("runtime_error")
    assert mod.guard_candidate(task, "def transform(grid):\n    return [[0]]").reason == (
        "demo_shape_mismatch"
    )
    assert mod.guard_candidate(task, "def transform(grid):\n    raise ValueError('x')").reason == (
        "runtime_error"
    )
    test_shape_bad = (
        "def transform(grid):\n"
        "    if grid == [[7, 8]]:\n"
        "        return [[8, 9, 10]]\n"
        "    return [[int(cell) + 1 for cell in row] for row in grid]\n"
    )
    assert mod.guard_candidate(task, test_shape_bad).reason == "test_shape_mismatch"
    assert mod.repair_candidate(task, "", mod.GuardResult(False, "x"), repair_index=1)[1] == (
        "demo_lookup_same_shape_retry"
    )

    assert mod.load_checkpoint(tmp_path / "missing.json") == []
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert mod._read_json(broken) == {}
    assert list(mod._rows_from_exp5197_payload([{"task": "a"}, {"no": "task"}])) == ["a"]
    assert list(mod._rows_from_exp5197_payload({"rows": [{"task": "b"}]})) == ["b"]
    assert list(mod._rows_from_exp5197_payload("bad")) == []
    exp5197 = tmp_path / mod.EXP5197_RELATIVE_PATH
    exp5197.parent.mkdir(parents=True)
    exp5197.write_text(json.dumps({"scaleup_rows": [{"task": "c"}]}), encoding="utf-8")
    assert "c" in mod.load_exp5197_scored_task_ids(tmp_path)

    assert mod._crop_changed_rows([[1]], [[1], [2]]) is None
    assert mod._crop_changed_rows([[1]], [[1]]) is None
    rows = [
        {"env": "g", "source_row_index": 1, "step_index": 0, "frame": [[0, 0]]},
        {"env": "g", "source_row_index": 1, "step_index": 1, "frame": [[1, 0]]},
    ]
    assert mod._emit_transition_tasks(rows, exclude_task_ids=set(), budget=1) == []
    enough = [
        {"env": "g", "source_row_index": 1, "step_index": 0, "frame": [[0, 0]]},
        {"env": "g", "source_row_index": 1, "step_index": 1, "frame": [[1, 0]]},
        {"env": "g", "source_row_index": 1, "step_index": 2, "frame": [[1, 2]]},
        {"env": "g", "source_row_index": 1, "step_index": 3, "frame": [[3, 2]]},
        {"env": "g", "source_row_index": 1, "step_index": 4, "frame": [[3, 4]]},
    ]
    assert len(mod._emit_transition_tasks(enough, exclude_task_ids=set(), budget=1)) == 1

    shard_dir = tmp_path / "loader" / mod.HUMAN_REPLAY_RELATIVE_DIR / "shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "train-00000.jsonl").write_text(
        "{bad json\n" + json.dumps({"no": "frame"}) + "\n" + "\n".join(json.dumps(row) for row in enough),
        encoding="utf-8",
    )
    assert mod.load_frame_transition_tasks(tmp_path / "loader", set(), budget=0) == []

    assert mod._audit_no_leakage([{"task_id": "x", "code": "def transform(grid):\n    return grid"}])
    assert not mod._audit_no_leakage(
        [{"task_id": "x", "code": "def transform(grid):\n    return grid"}],
        exp5197_task_ids={"x"},
    )
    assert not mod._audit_no_leakage(
        [{"task_id": "x", "code": "import os\ndef transform(grid):\n    return grid"}]
    )
    assert not mod._audit_no_leakage(
        [{"task_id": "x", "code": "def transform(grid):\n    return grid", "test_output": [[1]]}]
    )

    good = mod.build_artifact(
        events=[],
        model_specs=[],
        models_used=[],
        sota_gguf_resolved=False,
        repair_attempts=0,
        source_task_budget=0,
        source_task_count=0,
        checkpoint_path=mod.CHECKPOINT_RELATIVE_PATH,
        duration_s=0.0,
    )
    missing = dict(good)
    missing.pop("duration_s")
    missing["reproducibility_checksum"] = mod.payload_checksum(missing)
    assert "missing required field duration_s" in mod.artifact_schema_errors(missing)
    wrong = dict(good)
    wrong["candidate_pool_n"] = 1
    wrong["accepted_rows"] = 0
    wrong["gap4_expansion_usable"] = True
    wrong["model_specs"] = "bad"
    wrong["models_used"] = "bad"
    wrong["honest_verdict"] = "bad"
    wrong["reproducibility_checksum"] = mod.payload_checksum(wrong)
    errors = mod.artifact_schema_errors(wrong)
    assert "accepted_rows" in errors
    assert "blocked_candidate_pool_n_zero" in errors
    assert "gap4_expansion_usable" in errors
    assert "model_specs" in errors
    assert "models_used" in errors
    assert "honest_verdict_terminal_prefix" in errors

    assert mod._env_int("EXP5211_DOES_NOT_EXIST", 3) == 3
    os_environ = __import__("os").environ
    os_environ["EXP5211_TEST_INT"] = "bad"
    assert mod._env_int("EXP5211_TEST_INT", 3) == 3
    os_environ["EXP5211_TEST_INT"] = "-1"
    assert mod._env_int("EXP5211_TEST_INT", 3) == 3
    os_environ["EXP5211_TEST_INT"] = "4"
    assert mod._env_int("EXP5211_TEST_INT", 3) == 4

    existing, _ = mod.process_task_row(
        task=_task(90, task_id="existing"),
        raw_text=_plus_one_code(),
        model_spec=_specs()[0],
        checkpoint_path=tmp_path / "resume.json",
        prior_events=[],
        repair_budget=0,
        generation_error="boom",
    )
    artifact = mod.run(
        root=tmp_path,
        cached_pair_loader=lambda: _specs(),
        task_loader=lambda _root, _exclude, _budget: [dict(_task(90, task_id="existing")), _task(91)],
        text_generator_factory=lambda _spec: (_ for _ in ()).throw(RuntimeError("factory")),
        max_live_prompts=1,
        task_budget=2,
    )
    assert artifact["generation_errors"][0].startswith("generator_factory:RuntimeError")

    checkpoint_path = tmp_path / mod.CHECKPOINT_RELATIVE_PATH
    mod.write_checkpoint(checkpoint_path, [existing])
    artifact = mod.run(
        root=tmp_path,
        cached_pair_loader=lambda: _specs(),
        task_loader=lambda _root, _exclude, _budget: [dict(_task(90, task_id="existing")), _task(92)],
        text_generator_factory=lambda _spec: (lambda _prompt: (_ for _ in ()).throw(ValueError("gen"))),
        max_live_prompts=1,
        task_budget=2,
    )
    assert any(error.startswith("ValueError:gen") for error in artifact["generation_errors"])
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert mod._default_cached_pair_loader()
