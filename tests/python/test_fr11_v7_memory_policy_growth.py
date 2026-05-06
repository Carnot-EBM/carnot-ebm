"""Tests for Exp 1447 FR-11 v7 memory-policy growth.

Spec: REQ-LEARN-1447, SCENARIO-LEARN-1447, SCENARIO-LEARN-1448.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.pipeline.session_memory import SessionMemory
from carnot.reporting import fr11_v7_memory_policy_growth as mod


def _write_checkpoint(path: Path, *, secl_threshold: float = 0.500001) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=np.zeros(128, dtype=np.float32),
            bias=np.asarray([10.0], dtype=np.float32),
            secl_bin_values=np.full(10, 0.5, dtype=np.float32),
            secl_global_value=np.asarray([0.5], dtype=np.float32),
            secl_n_bins=np.asarray([10], dtype=np.int32),
            dvi_incorrect_threshold=np.asarray([0.72], dtype=np.float32),
            secl_confidence_threshold=np.asarray([secl_threshold], dtype=np.float32),
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _exp1395(*, baseline_count: int = 1) -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": baseline_count,
        "memory_updates": {"promoted": ["dvi_v2:fover:baseline"], "demoted": []},
    }


def _exp1432(checkpoint_path: Path, *, deployed: bool = True) -> dict[str, Any]:
    return {
        "status": "complete" if deployed else "blocked",
        "dvi_v3_deployed": deployed,
        "dvi_v3_checkpoint_path": str(checkpoint_path),
        "nonforgetting_rate": 1.0,
    }


def _exp1446(
    *,
    changes_policy: bool = True,
    expected: int = 1,
    fresh_threshold: float = 0.5,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "recommended_v7_policy": {
            "policy_name": "fr11_v7_asymmetric_fresh_threshold",
            "changes_exp1433_policy": changes_policy,
            "fresh_secl_confidence_threshold": fresh_threshold,
            "replay_nonforgetting_secl_confidence_threshold": 0.500001,
            "dvi_incorrect_threshold": 0.72,
            "expected_promotions_under_v7_policy": expected,
        },
    }


def _fover_row(case_id: str, label: str) -> dict[str, str]:
    return {
        "question_id": case_id,
        "step_text": f"{label} FoVer trace for {case_id}",
        "label": label,
        "source": "unit_fover",
    }


def test_req_learn_1447_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1447-1: bootstrap artifact is visible before source loading."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["model_specs"] == mod.MODEL_SPECS
    assert written["policy_changes_applied"] == []
    assert written["fresh_verified_sample_count"] is None
    assert written["session_memory_updated"] is None
    assert written["honest_verdict"] == "in_progress"


def test_req_learn_1447_requires_changed_v7_policy() -> None:
    """REQ-LEARN-1447-2: unchanged Exp 1433 policy cannot launch v7."""

    with pytest.raises(ValueError, match="recommended_v7_policy"):
        mod.load_v7_policy({"status": "complete"})

    with pytest.raises(ValueError, match="changes_exp1433_policy"):
        mod.load_v7_policy(_exp1446(changes_policy=False))


def test_req_learn_1447_candidate_source_counts_unusable_rows() -> None:
    """REQ-LEARN-1447-3: local FoVer source counts skipped candidate rows."""

    loaded = mod.fresh_candidates_from_local_fover(
        [
            {"step_text": "missing id", "label": "incorrect"},
            {"question_id": "known", "step_text": "known", "label": "incorrect"},
            {"question_id": "bad_label", "step_text": "bad", "label": "unknown"},
            {"question_id": "empty_text", "label": "incorrect"},
            _fover_row("fresh", "incorrect"),
        ],
        exclude_case_ids={"known"},
    )

    assert [case.case_id for case in loaded.cases] == ["fresh"]
    assert loaded.counts["missing_case_id"] == 1
    assert loaded.counts["novelty_threshold"] == 1
    assert loaded.counts["unusable_candidate"] == 2


def test_scenario_learn_1447_promotes_dedupes_and_persists_memory(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1447: verified v7 promotions persist before delta is positive."""

    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    session_dir = tmp_path / "session"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(baseline_count=1),
        exp1432_artifact=_exp1432(checkpoint_path),
        exp1446_artifact=_exp1446(expected=1),
        fover_rows=[
            _fover_row("baseline", "incorrect"),
            _fover_row("new_bad", "incorrect"),
            _fover_row("new_bad", "incorrect"),
            _fover_row("new_good", "correct"),
        ],
        session_memory_dir=session_dir,
        project_root="/repo",
        commands_run=["unit command"],
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["candidate_policy"]["candidate_source"] == (
        "local_verified_fover_rows_not_exp1395_promoted_deduped"
    )
    assert artifact["candidate_supply_count"] == 2
    assert artifact["new_promoted_count"] == 1
    assert artifact["memory_entries_added"] == 1
    assert artifact["fresh_verified_sample_count"] == 2
    assert artifact["self_learning_delta_overall"] == 1
    assert artifact["session_memory_updated"] is True
    assert artifact["retire_if_zero_growth_repeats"] is False
    assert artifact["memory_updates"]["promoted"] == ["dvi_v7:fover:new_bad"]
    assert artifact["memory_updates"]["duplicate_candidate_rows_skipped"] == 1

    loaded = SessionMemory(str(session_dir), mod.SESSION_MEMORY_MODEL_ID).load()
    assert loaded is not None
    case_memory, _, _ = loaded
    entries = case_memory.entries()
    assert len(entries) == 1
    assert entries[0].provenance[0].case_id == "new_bad"


def test_scenario_learn_1448_zero_growth_sets_retirement_gate(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1448: changed policy with no persisted growth is retired."""

    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(baseline_count=1),
        exp1432_artifact=_exp1432(checkpoint_path),
        exp1446_artifact=_exp1446(expected=0),
        fover_rows=[_fover_row("new_good", "correct")],
        session_memory_dir=tmp_path / "session",
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["new_promoted_count"] == 0
    assert artifact["memory_entries_added"] == 0
    assert artifact["self_learning_delta_overall"] == 0
    assert artifact["session_memory_updated"] is False
    assert artifact["retire_if_zero_growth_repeats"] is True
    assert artifact["next_root_cause"] == "dvi_state_mismatch_after_v7_policy_change"
    assert artifact["honest_verdict"] == (
        "fr11_v7_zero_growth_after_changed_policy_retire_dvi_state_mismatch"
    )


def test_req_learn_1447_threshold_rejection_and_nonforgetting_replay(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1447-3/4: fresh and replay thresholds are measured separately."""

    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1395_artifact={
            **_exp1395(baseline_count=1),
            "memory_updates": {
                "promoted": ["dvi_v2:fover:baseline"],
                "demoted": ["dvi_v2:fover:replay_bad", "dvi_v2:fover:replay_bad"],
            },
        },
        exp1432_artifact=_exp1432(checkpoint_path),
        exp1446_artifact=_exp1446(expected=0, fresh_threshold=0.500001),
        fover_rows=[
            _fover_row("new_bad", "incorrect"),
            _fover_row("replay_bad", "incorrect"),
        ],
        session_memory_dir=tmp_path / "session",
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["memory_updates"]["rejection_reason_counts"]["dvi_threshold"] == 2
    assert artifact["nonforgetting_rate"] == 1.0
    assert artifact["next_root_cause"] == "fresh_threshold_still_blocks_candidates"


def test_req_learn_1447_replay_loader_handles_bad_demoted_payload() -> None:
    """REQ-LEARN-1447-4: malformed replay memory degrades to no replay cases."""

    assert (
        mod.replay_cases_from_exp1395(
            {"memory_updates": {"demoted": "not-a-list"}},
            [_fover_row("case", "incorrect")],
        )
        == []
    )
    replay = mod.replay_cases_from_exp1395(
        {
            "memory_updates": {
                "demoted": [
                    "dvi_v2:fover:missing",
                    "dvi_v2:fover:bad_label",
                    "dvi_v2:fover:empty_text",
                    "dvi_v2:fover:usable",
                ]
            }
        },
        [
            {"question_id": "bad_label", "step_text": "bad", "label": "unknown"},
            {"question_id": "empty_text", "label": "incorrect"},
            _fover_row("usable", "incorrect"),
        ],
    )
    assert [case.case_id for case in replay] == ["usable"]


def test_req_learn_1447_persistence_failure_keeps_delta_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-1447-5: failed SessionMemory reload prevents positive growth."""

    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    _write_checkpoint(checkpoint_path)

    monkeypatch.setattr(mod.SessionMemory, "load", lambda self: None)
    artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(baseline_count=1),
        exp1432_artifact=_exp1432(checkpoint_path),
        exp1446_artifact=_exp1446(expected=1),
        fover_rows=[_fover_row("new_bad", "incorrect")],
        session_memory_dir=tmp_path / "session",
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["memory_entries_added"] == 0
    assert artifact["next_root_cause"] == "session_memory_persistence_failed"
    assert artifact["retire_if_zero_growth_repeats"] is True


def test_req_learn_1447_run_loads_sources_and_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1447-6: run writes the required complete artifact fields."""

    results = tmp_path / "results"
    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    fover_path = tmp_path / "fover.jsonl"
    out_path = results / mod.OUTPUT_FILE
    _write_checkpoint(checkpoint_path)
    _write_json(results / mod.EXP1395_FILE, _exp1395(baseline_count=1))
    _write_json(results / mod.EXP1432_FILE, _exp1432(checkpoint_path))
    _write_json(results / mod.EXP1446_FILE, _exp1446(expected=1))
    _write_jsonl(fover_path, [_fover_row("new_bad", "incorrect")])

    artifact = mod.run(
        exp1395_path=results / mod.EXP1395_FILE,
        exp1432_path=results / mod.EXP1432_FILE,
        exp1446_path=results / mod.EXP1446_FILE,
        fover_path=fover_path,
        out_path=out_path,
        session_memory_dir=tmp_path / "session",
        project_root=tmp_path,
        commands_run=["pytest test"],
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["commands_run"] == ["pytest test"]


def test_req_learn_1447_validation_rejects_unpersisted_positive_delta() -> None:
    """REQ-LEARN-1447-5: positive delta requires persisted SessionMemory entries."""

    artifact = mod.write_in_progress_artifact(Path("/tmp/nonpersistent_exp1447.json"))
    mod.validate_artifact(artifact)
    artifact.update(
        {
            "status": "complete",
            "fresh_verified_sample_count": 2,
            "new_promoted_count": 1,
            "self_learning_delta_overall": 1,
            "nonforgetting_rate": 1.0,
            "session_memory_updated": False,
            "memory_entries_added": 0,
            "retire_if_zero_growth_repeats": False,
            "honest_verdict": "bad",
        }
    )
    with pytest.raises(AssertionError, match="persisted SessionMemory"):
        mod.validate_artifact(artifact)

    artifact["session_memory_updated"] = True
    artifact["memory_entries_added"] = 2
    with pytest.raises(AssertionError, match="memory_entries_added"):
        mod.validate_artifact(artifact)

    missing = dict(artifact)
    missing["memory_entries_added"] = 1
    del missing["model_specs"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)


def test_req_learn_1447_validation_rejects_other_bad_terminal_invariants() -> None:
    """REQ-LEARN-1447-5/6: validation protects delta, nonforgetting, and retire flags."""

    base = mod.write_in_progress_artifact(Path("/tmp/nonpersistent_exp1447_more.json"))
    base.update(
        {
            "status": "complete",
            "fresh_verified_sample_count": 2,
            "new_promoted_count": 1,
            "self_learning_delta_overall": 0,
            "nonforgetting_rate": 1.0,
            "session_memory_updated": True,
            "memory_entries_added": 1,
            "retire_if_zero_growth_repeats": False,
            "honest_verdict": "bad",
        }
    )
    with pytest.raises(AssertionError, match="self_learning_delta_overall"):
        mod.validate_artifact(base)

    stale_flag = dict(base)
    stale_flag["self_learning_delta_overall"] = 1
    stale_flag["session_memory_updated"] = False
    with pytest.raises(AssertionError, match="persisted SessionMemory"):
        mod.validate_artifact(stale_flag)

    low_nonforgetting = dict(base)
    low_nonforgetting["self_learning_delta_overall"] = 1
    low_nonforgetting["nonforgetting_rate"] = 0.5
    with pytest.raises(AssertionError, match="preserved nonforgetting"):
        mod.validate_artifact(low_nonforgetting)

    bad_retire = dict(base)
    bad_retire["self_learning_delta_overall"] = 1
    bad_retire["retire_if_zero_growth_repeats"] = True
    with pytest.raises(AssertionError, match="cannot set retire"):
        mod.validate_artifact(bad_retire)

    zero_no_retire = dict(base)
    zero_no_retire["new_promoted_count"] = 0
    zero_no_retire["self_learning_delta_overall"] = 0
    zero_no_retire["session_memory_updated"] = False
    zero_no_retire["memory_entries_added"] = 0
    with pytest.raises(AssertionError, match="retirement gate"):
        mod.validate_artifact(zero_no_retire)

    stale_positive_flag = dict(zero_no_retire)
    stale_positive_flag["retire_if_zero_growth_repeats"] = True
    stale_positive_flag["session_memory_updated"] = True
    with pytest.raises(AssertionError, match="session_memory_updated"):
        mod.validate_artifact(stale_positive_flag)


def test_req_learn_1447_root_cause_and_blocked_verdict_helpers() -> None:
    """REQ-LEARN-1447-6: zero-growth verdicts name the next root cause."""

    assert (
        mod.measure_v7_nonforgetting_rate(
            exp1432_artifact={},
            exp1395_artifact={},
            fover_rows=[],
            activation=mod.v6.DviV3Activation(
                active=False,
                path=None,
                blocker="missing",
                nonforgetting_rate=None,
                state=None,
            ),
            policy=mod.V7Policy("p", 0.5, 0.5, 0.72, 0),
        )
        is None
    )
    assert (
        mod._next_root_cause(
            status="blocked",
            candidates=[],
            memory_updates={},
            promoted_variants=[],
            memory_entries_added=0,
            nonforgetting_preserved=False,
        )
        == "dvi_v3_inactive_or_unavailable"
    )
    assert (
        mod._next_root_cause(
            status="complete",
            candidates=[],
            memory_updates={},
            promoted_variants=[],
            memory_entries_added=0,
            nonforgetting_preserved=True,
        )
        == "no_local_verified_candidates_after_novelty_filter"
    )
    assert (
        mod._next_root_cause(
            status="complete",
            candidates=[mod.dvi.DviCase("c", "text", 1, "unit")],
            memory_updates={},
            promoted_variants=[],
            memory_entries_added=0,
            nonforgetting_preserved=False,
        )
        == "replay_nonforgetting_not_preserved"
    )
    assert (
        mod._next_root_cause(
            status="complete",
            candidates=[mod.dvi.DviCase("c", "text", 1, "unit")],
            memory_updates={"rejection_reason_counts": {}},
            promoted_variants=[],
            memory_entries_added=0,
            nonforgetting_preserved=True,
        )
        == "no_promotable_candidates_after_v7_policy_change"
    )
    assert (
        mod._honest_verdict(
            status="blocked",
            delta=0,
            next_root_cause="dvi_v3_inactive_or_unavailable",
            nonforgetting_preserved=False,
        )
        == "fr11_v7_blocked_dvi_v3_inactive"
    )
    assert (
        mod._honest_verdict(
            status="complete",
            delta=0,
            next_root_cause="fresh_threshold_still_blocks_candidates",
            nonforgetting_preserved=True,
        )
        == "fr11_v7_zero_growth_after_changed_policy_retire_fresh_threshold_still_blocks_candidates"
    )
