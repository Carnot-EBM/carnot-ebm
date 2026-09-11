"""Tests for the V636 cumulative adapter-withheld ARC session.

Spec refs: REQ-ARC-WMTE-7221 and SCENARIO-ARC-WMTE-7221-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7206_v635_arc_volume_a as base
from carnot import experiment_7221_v636_arc_session as exp


def _artifact(
    rows: list[dict[str, object]], *, complete: object = 1, experiment_id: int = 7206
) -> dict[str, object]:
    payload: dict[str, object] = {
        "arc_session_complete_score": complete,
        "experiment_id": experiment_id,
        "flagged_adversarial": False,
        "cumulative_induction_rows": rows,
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = base.artifact_checksum(payload)
    return payload


def _row(identity: str, session: str, seed: int) -> dict[str, object]:
    return {
        "induction_id": "sha256:" + identity * 64,
        "source_session_id": session,
        "seed": seed,
        "source_authenticated": True,
        "engaged": True,
        "tool_calls_total": 1,
        "tool_gap_events": [],
    }


def test_req_7221_spec_and_frozen_runtime_contract() -> None:
    """REQ-ARC-WMTE-7221 freezes the V636 identity, model, and bounded session."""

    spec = (exp.REPO_ROOT / base.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-ARC-WMTE-7221:" in spec
    assert "SCENARIO-ARC-WMTE-7221-HISTORICAL-IDENTITY" in spec
    assert exp.TASK_ID == "exp7221-arc-session"
    assert exp.EXPERIMENT_ID == 7221
    assert exp.MILESTONE == "2026.09.636"
    assert exp.RUN_DATE == "20260911"
    assert exp.RANDOM_SEED == 7_221_001
    assert exp.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
    assert (
        exp.ACTION_BUDGET,
        exp.SESSION_TIMEOUT_S,
        exp.INDUCTION_TIMEOUT_S,
        exp.N_CTX,
        exp.COMPLETION_BUDGET,
    ) == (4000, 3600, 2400, 49152, 4096)


def test_scenario_7221_configuration_is_scoped_and_restored() -> None:
    """SCENARIO-ARC-WMTE-7221-CONFIGURATION resolves current defaults at call time."""

    original = {
        name: getattr(base, name)
        for name in (
            "TASK_ID",
            "EXPERIMENT_ID",
            "RANDOM_SEED",
            "SCHEMA",
            "DRIVING_REQUIREMENT",
            "RESULT_PATH",
            "CHECKPOINT_PATH",
            "RAW_DIR",
            "SIBLING_PATH",
            "SIBLING_TASK_ID",
            "_optional_sibling",
        )
    }
    with exp.configured_runtime() as configured:
        assert configured.TASK_ID == exp.TASK_ID
        assert configured.RANDOM_SEED == 7_221_001
        assert configured.DRIVING_REQUIREMENT == "REQ-ARC-WMTE-7221"
        assert configured.RESULT_PATH == Path("results/experiment_7221_v636_arc_session.json")
        assert configured.CHECKPOINT_PATH.parent == Path(
            "results/checkpoints/experiment_7221_v636_arc_session"
        )
        assert configured.RAW_DIR == Path("results/raw/experiment_7221")
        assert configured._optional_sibling is exp.authenticated_v635_cumulative_rows
        configured.configure_reused_driver()
        env = configured.reused.session_environment(
            {},
            model_path="/cache/model.gguf",
            gpu_index=1,
            port=9123,
            raw_dir=Path("/tmp/exp7221-raw"),
        )
        assert env["CARNOT_FORCE_LIVE"] == "1"
        assert env["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
        assert env["CARNOT_ARC_INDUCE_N_CTX"] == "49152"
        assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "4096"
        assert env["CARNOT_ARC_RANDOM_SEED"] == "7221001"
        assert "CARNOT_ARC_SUPERVISOR_TOOL_ARM" not in env
    assert {name: getattr(base, name) for name in original} == original


def test_scenario_7221_configuration_restores_after_failure() -> None:
    """SCENARIO-ARC-WMTE-7221-CONFIGURATION cannot leak on an exception."""

    original_seed = base.RANDOM_SEED
    with pytest.raises(RuntimeError, match="fixture"):
        with exp.configured_runtime():
            assert base.RANDOM_SEED == exp.RANDOM_SEED
            raise RuntimeError("fixture")
    assert base.RANDOM_SEED == original_seed


def test_scenario_7221_historical_ids_are_authenticated_and_not_relabelled(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7221-HISTORICAL-IDENTITY keeps retained V635 identity."""

    row_a = _row("a", "exp7206-arc-volume-a", 7_206_001)
    row_b = _row("b", "exp7206-arc-volume-a", 7_206_001)
    path_a = tmp_path / exp.V635_ARTIFACT_PATHS[0]
    path_b = tmp_path / exp.V635_ARTIFACT_PATHS[1]
    path_a.parent.mkdir(parents=True)
    path_a.write_text(json.dumps(_artifact([row_a])), encoding="utf-8")
    path_b.write_text(
        json.dumps(_artifact([row_a, row_b], experiment_id=7207)), encoding="utf-8"
    )
    source_hashes: dict[str, object] = {}

    receipt, rows = exp.authenticated_v635_cumulative_rows(tmp_path, source_hashes)

    assert receipt["state"] == "authenticated"
    assert receipt["accepted_unique_inductions"] == 2
    assert receipt["excluded_inductions"] == 0
    assert rows == [row_a, row_b]
    assert rows[1]["source_session_id"] == "exp7206-arc-volume-a"
    assert rows[1]["seed"] == 7_206_001
    assert set(source_hashes) == {path.as_posix() for path in exp.V635_ARTIFACT_PATHS}


def test_scenario_7221_preflight_rejects_quarantine_before_consumption(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7221-PREFLIGHT never consumes a quarantined complete value."""

    good = _artifact([_row("a", "exp7206-arc-volume-a", 7_206_001)])
    bad = deepcopy(good)
    bad["flagged_adversarial"] = True
    for index, payload in enumerate((good, bad)):
        path = tmp_path / exp.V635_ARTIFACT_PATHS[index]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    receipt, rows = exp.authenticated_v635_cumulative_rows(tmp_path, {})

    assert receipt["state"] == "rejected_required_source"
    rejected = receipt["sources"][1]
    assert rejected["quarantined"] is True
    assert rejected["completion_value"] == "not_consumed"
    assert rejected["consumed"] is False
    assert rows == []


def test_scenario_7221_historical_merge_replays_shipped_reducer() -> None:
    """SCENARIO-ARC-WMTE-7221-HISTORICAL-IDENTITY uses repaired ID deduplication."""

    old = _row("a", "pre_repair_label", 7_206_001)
    duplicate = deepcopy(old)
    current = _row("c", exp.TASK_ID, exp.RANDOM_SEED)
    merged, summary = base.merge_cumulative_rows(
        [("v635", [old, duplicate]), (exp.TASK_ID, [current])]
    )
    assert [row["induction_id"] for row in merged] == [
        old["induction_id"],
        current["induction_id"],
    ]
    assert merged[0]["source_session_id"] == "pre_repair_label"
    assert merged[0]["seed"] == 7_206_001
    assert summary["cumulative_unique_inductions"] == 2


def test_req_7221_contract_and_field_principles_match_roadmap() -> None:
    """REQ-ARC-WMTE-7221 authenticates the task and its exact evidence principles."""

    with exp.configured_runtime() as configured:
        observed = configured._task_contract(exp.REPO_ROOT / configured.ROADMAP_PATH)
        assert json.loads(json.dumps(observed)) == json.loads(
            json.dumps(exp.EXPECTED_TASK_CONTRACT)
        )
        assert configured.FIELD_PRINCIPLES["sample_size_budget"] == (
            "Historical, new, excluded and accepted counts; target 10 remains cumulative."
        )
        assert configured.FIELD_PRINCIPLES["cumulative_induction_rows"] == (
            "Deduplicate hashes without relabeling historical rows."
        )


def test_scenario_7221_terminal_delegate_and_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7221-TERMINAL delegates and exposes the thin entrypoint."""

    seen: dict[str, object] = {}

    def fake_run_experiment(**kwargs: object) -> dict[str, object]:
        seen.update(task=base.TASK_ID, seed=base.RANDOM_SEED, result=kwargs["result_path"])
        return {"status": "complete", "arc_session_complete_score": 1}

    monkeypatch.setattr(base, "run_experiment", fake_run_experiment)
    result = exp.run_experiment(
        root=tmp_path,
        run_date="20260911",
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        raw_dir=tmp_path / "raw",
    )
    assert result["arc_session_complete_score"] == 1
    assert seen == {
        "task": exp.TASK_ID,
        "seed": exp.RANDOM_SEED,
        "result": tmp_path / "result.json",
    }
    monkeypatch.setattr(exp, "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(exp.REPO_ROOT / exp.WRAPPER_PATH), run_name="__main__")
    assert stopped.value.code == 0
