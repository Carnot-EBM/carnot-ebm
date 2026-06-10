"""Tests for Exp 3999 GAP-4 precision confirmation v2.

Spec refs: REQ-VERIFY-3999, SCENARIO-VERIFY-3999.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import experiment_3999_gap4_precision_confirmation_v2 as exp


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _entry(task: str, demo_value: int, test_value: int) -> JsonDict:
    return {
        "task": task,
        "demos": [{"input": [[demo_value]], "output": [[demo_value]]}],
        "test_input": [[test_value]],
        "candidates": [],
    }


def _arc_gold(entries: list[JsonDict], gold_by_task: dict[str, int]) -> tuple[JsonDict, JsonDict]:
    challenges: JsonDict = {}
    solutions: JsonDict = {}
    for entry in entries:
        task = str(entry["task"])
        challenges[task] = {"test": [{"input": entry["test_input"]}]}
        solutions[task] = [[[gold_by_task[task]]]]
    return challenges, solutions


def _constant_code(demo_value: int, pred_value: int) -> str:
    return (
        "def transform(grid):\n"
        "    if int(grid[0, 0]) == %d:\n"
        "        return np.array([[%d]])\n"
        "    return np.array([[%d]])\n"
    ) % (demo_value, demo_value, pred_value)


def test_req_verify_3999_spec_anchor_exists() -> None:
    """REQ-VERIFY-3999: OpenSpec declares the precision-confirmation contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-3999" in spec
    assert "SCENARIO-VERIFY-3999" in spec
    assert "protocol_preregistered" in spec
    assert "blocked_eval_pool_unreadable" in spec
    assert "n_agreement_events>=19" in spec


def test_req_verify_3999_real_pool_selects_new_clean_nonprior_tasks() -> None:
    """REQ-VERIFY-3999: the committed task set is new clean ARC-2 tasks only."""

    pool = exp.load_eval_pool(Path("results/arc3_gap4_arc2_eval_pool.json.gz"))
    chain_artifact = json.loads(
        Path("results/arc3_gap4_arc2_chain_ensemble.json").read_text(encoding="utf-8")
    )

    assert exp.clean_new_tasks(pool["entries"], chain_artifact) == [
        "21897d95",
        "269e22fb",
        "28a6681f",
        "2c181942",
        "3a25b0d8",
        "3dc255db",
        "a6f40cea",
        "b9e38dc0",
        "dd6b8c4b",
    ]


def test_req_verify_3999_primary_binomial_gate() -> None:
    """REQ-VERIFY-3999: the primary gate is the pre-registered n>=19, >=14-gold rule."""

    assert exp.primary_gate_passed(14, 19) is True
    assert exp.primary_gate_passed(13, 19) is False
    assert exp.primary_gate_passed(14, 18) is False

    assert exp.verdict_for(14, 19, True) == "success: gap4_precision_confirmed_14of19_gold"
    assert (
        exp.verdict_for(13, 19, False)
        == "complete: gap4_agreement_confidence_label_only_13of19"
    )


def test_scenario_verify_3999_preregisters_before_any_induction_and_scores_agreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3999: mocked k=3 fresh chains emit bare precision fields."""

    entries = [
        _entry("gold_agree", 0, 8),
        _entry("wrong_majority", 2, 8),
        _entry("no_agreement", 3, 8),
    ]
    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    chain_path = tmp_path / "results" / "arc3_gap4_arc2_chain_ensemble.json"
    output_path = tmp_path / "results" / "experiment_3999_gap4_precision_confirmation_v2.json"
    transcript_dir = tmp_path / "results" / "experiment_3999_gap4_precision_transcripts"
    challenges_path = tmp_path / "arc-agi_evaluation2_challenges.json"
    solutions_path = tmp_path / "arc-agi_evaluation2_solutions.json"

    _write_gzip_json(pool_path, {"entries": entries})
    _write_json(chain_path, {"preregistration": {"tasks": ["prior"]}})
    challenges, solutions = _arc_gold(
        entries,
        {"gold_agree": 1, "wrong_majority": 9, "no_agreement": 2},
    )
    _write_json(challenges_path, challenges)
    _write_json(solutions_path, solutions)

    arm_preds = {
        "gold_agree": [1, 1, 1],
        "wrong_majority": [5, 5, 6],
        "no_agreement": [2, 3, 4],
    }
    seen_prereg: list[JsonDict] = []

    def fake_induce(
        task_name: str,
        demos: list[JsonDict],
        test_input: list[list[int]],
        iters: int,
        timeout: int,
        transcripts_dir: str,
    ) -> JsonDict:
        del test_input, iters, timeout
        assert output_path.exists()
        prereg = json.loads(output_path.read_text(encoding="utf-8"))
        assert prereg["protocol_preregistered"] is True
        assert prereg["n_agreement_events"] == 0
        seen_prereg.append(prereg)

        arm_idx = int(Path(transcripts_dir).name.replace("arm", "")) - 1
        Path(transcripts_dir).mkdir(parents=True, exist_ok=True)
        (Path(transcripts_dir) / f"{task_name}_iter0.txt").write_text(
            "===== PROMPT =====\nDemo pairs and test input only\n===== RAW OUTPUT =====\n"
            "```python\n# clean transcript\n```\n",
            encoding="utf-8",
        )
        pred = arm_preds[task_name][arm_idx]
        code = _constant_code(int(demos[0]["input"][0][0]), pred)
        return {
            "task": task_name,
            "demo_fit": 1.0,
            "demo_perfect": True,
            "pred_hash": "unused",
            "pred_grid": [[pred]],
            "n_calls": 1,
            "codex_seconds": 1.0,
            "history": [{"iter": 0, "status": "graded", "demo_fit": 1.0, "codex_s": 1.0}],
            "code": code,
        }

    monkeypatch.setattr(exp, "induce_program", fake_induce)

    artifact = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        chain_artifact_path=chain_path,
        output_path=output_path,
        transcripts_dir=transcript_dir,
        challenges_path=challenges_path,
        solutions_path=solutions_path,
        codex_available_override=True,
        workers=1,
    )

    assert seen_prereg
    assert output_path.exists()
    assert artifact["protocol_preregistered"] is True
    assert artifact["preregistration"]["task_set"] == [
        "gold_agree",
        "no_agreement",
        "wrong_majority",
    ]
    assert artifact["n_agreement_events"] == 2
    assert artifact["n_gold_given_agreement"] == 1
    assert artifact["primary_gate_passed"] is False
    assert artifact["agreement_is_selector_not_label"] is False
    assert artifact["fresh_arm_base_rate"] == pytest.approx(0.4444)
    assert artifact["precision_vs_fresharm_base"] == pytest.approx(0.0556)
    assert artifact["sibling_disagreement_tripwire_gold_rate"] == 1.0
    assert artifact["total_codex_calls"] == 9
    assert artifact["total_codex_seconds"] == 9.0
    assert artifact["leak_clean"] is True
    assert artifact["honest_verdict"] == "complete: gap4_agreement_confidence_label_only_1of2"
    assert artifact["missing_verifier_gaps"] == [
        {
            "task": "wrong_majority",
            "input_idx": 0,
            "failure_mode": "agreement_wrong_but_unanimity_tripwire_abstains",
            "missing_discriminator": "GAP-5 demo-underdetermination sibling-input tripwire",
        }
    ]
    exp.validate_artifact(artifact)


def test_scenario_verify_3999_blocks_without_codex_or_pool(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3999: blocked preconditions do not fabricate metrics."""

    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    _write_gzip_json(pool_path, {"entries": [_entry("n0", 1, 2)]})
    blocked_output = tmp_path / "results" / "blocked_codex_written.json"

    written_blocked = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        output_path=blocked_output,
        codex_available_override=False,
    )

    assert blocked_output.exists()
    assert written_blocked["honest_verdict"] == "blocked_codex_unavailable"

    blocked_codex = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        output_path=tmp_path / "results" / "blocked_codex.json",
        codex_available_override=False,
        write=False,
    )

    assert blocked_codex["honest_verdict"] == "blocked_codex_unavailable"
    assert blocked_codex["protocol_preregistered"] is False
    assert blocked_codex["n_agreement_events"] == 0
    assert blocked_codex["total_codex_calls"] == 0

    blocked_pool = exp.run(
        root=tmp_path,
        pool_path=tmp_path / "results" / "missing.json.gz",
        output_path=tmp_path / "results" / "blocked_pool.json",
        codex_available_override=True,
        write=False,
    )

    assert blocked_pool["honest_verdict"] == "blocked_eval_pool_unreadable"
    assert blocked_pool["n_gold_given_agreement"] == 0


def test_req_verify_3999_defensive_helpers_cover_residual_gap_paths() -> None:
    """REQ-VERIFY-3999: helper fallbacks preserve honest residual-gap accounting."""

    assert exp.selected_tasks_from_chain_artifact(
        {"per_task": [{"task": "b"}, {"task": "a"}]}
    ) == ["a", "b"]
    assert exp._history_iter0_demo_perfect([]) is False
    assert exp.gold_for_entry({"task": "missing", "test_input": [[1]], "candidates": []}, {}, {}) is None

    fallback_entry = {
        "task": "bad_unanimous",
        "test_input": [[8]],
        "candidates": [{"correct": True, "grid": [[9]]}],
    }
    assert exp.gold_for_entry(fallback_entry, {}, {}).tolist() == [[9]]

    arms = [
        {
            "source": "fresh_chain1",
            "demo_perfect": True,
            "predictions": [{"pred_hash": exp._grid_hash([[5]]), "pred_grid": [[5]]}],
        },
        {
            "source": "fresh_chain2",
            "demo_perfect": True,
            "predictions": [{"pred_hash": exp._grid_hash([[5]]), "pred_grid": [[5]]}],
        },
        {
            "source": "fresh_chain3",
            "demo_perfect": False,
            "predictions": [{"pred_hash": None, "pred_grid": None}],
        },
    ]
    row = exp._agreement_for_input("bad_unanimous", 0, arms, exp.gold_for_entry(fallback_entry, {}, {}))
    assert row["agreement"] is True
    assert row["agreed_is_gold"] is False
    assert row["n_demo_perfect_arms"] == 2

    null_row = exp._agreement_for_input(
        "none",
        0,
        [
            {
                "source": "fresh_chain1",
                "demo_perfect": True,
                "predictions": [{"pred_hash": None, "pred_grid": None}],
            }
        ],
        None,
    )
    assert null_row["agreement"] is False

    summary = exp._summarize_records(
        [{"task": "bad_unanimous", "arms": arms, "n_calls": 0, "codex_seconds": 0.0}],
        {"bad_unanimous": [fallback_entry]},
        {},
        {},
    )
    assert summary["missing_verifier_gaps"] == [
        {
            "task": "bad_unanimous",
            "input_idx": 0,
            "failure_mode": "agreement_wrong_and_tripwire_keeps",
            "missing_discriminator": "residual rule underdetermination beyond sibling unanimity",
        }
    ]


def test_req_verify_3999_validation_rejects_non_bare_schema_fields() -> None:
    """REQ-VERIFY-3999: required artifact fields stay bare scalars."""

    artifact = exp.blocked_artifact(
        "blocked_codex_unavailable",
        [{"resource": "codex", "available": False}, {"resource": "eval_pool", "available": True}],
        0.5,
    )
    exp.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad = dict(artifact, honest_verdict="maybe")
    with pytest.raises(ValueError, match="terminal prefix"):
        exp.validate_artifact(bad)
    bad = dict(artifact, protocol_preregistered="true")
    with pytest.raises(ValueError, match="protocol_preregistered"):
        exp.validate_artifact(bad)
    bad = dict(artifact, n_agreement_events=True)
    with pytest.raises(ValueError, match="n_agreement_events"):
        exp.validate_artifact(bad)
    bad = dict(artifact, precision_vs_fresharm_base=True)
    with pytest.raises(ValueError, match="precision_vs_fresharm_base"):
        exp.validate_artifact(bad)
    bad = dict(artifact, missing_verifier_gaps={})
    with pytest.raises(ValueError, match="missing_verifier_gaps"):
        exp.validate_artifact(bad)
    bad = dict(artifact, leak_clean="yes")
    with pytest.raises(ValueError, match="leak_clean"):
        exp.validate_artifact(bad)
    bad = dict(artifact, inference_substrate=3)
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad)
