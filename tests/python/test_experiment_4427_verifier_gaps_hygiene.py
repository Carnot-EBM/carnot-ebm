"""Tests for Exp 4427 verifier-gaps hygiene.

Spec refs: REQ-VERIFY-4427, SCENARIO-VERIFY-4427.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4427_verifier_gaps_hygiene as exp4427


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_gaps_text() -> str:
    return (
        "# Verifier Gaps\n\n"
        "### GAP-4421-S5I5-MARKER-COVERAGE\n"
        "- status: open\n"
        "- evidence: prior s5i5 marker coverage was not offline reproduced.\n"
        "- failure mode: marker coverage rule was ungrounded.\n"
        "- missing discriminator: grounded marker coverage predicate.\n"
        "- candidate design: derive marker coverage clicks from the visible target markers.\n"
        "- priority: high\n\n"
        "### GAP-4422-TR87-GLYPH-REWRITE-PERCEPTION\n"
        "- status: open\n"
        "- evidence: prior tr87 glyph rewrite rule was not yet grounded.\n"
        "- failure mode: glyph rewrite relation was not selectable.\n"
        "- missing discriminator: glyph rewrite perception predicate.\n"
        "- candidate design: segment glyphs and replay the rewrite predicate.\n"
        "- priority: high\n"
    )


def _write_minimal_repo(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / exp4427.GAPS_RELATIVE_PATH).write_text(_minimal_gaps_text(), encoding="utf-8")
    _write_json(
        root / exp4427.EXP4421_PATH,
        {
            "experiment": "experiment_4421_config_rule_solve_unseen",
            "target_game": "s5i5",
            "honest_verdict": "success_s5i5_L1_offline_reproduced",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "new_levels_reproduced": 1,
            "missing_verifier_gaps": [],
            "result_path": exp4427.EXP4421_PATH,
        },
    )
    _write_json(
        root / exp4427.EXP4422_PATH,
        {
            "experiment": "experiment_4422_glyph_rewrite_perception",
            "target_game": "tr87",
            "honest_verdict": "success_glyph_rewrite_perception_tr87_grounded_reproduced",
            "offline_reproduced": True,
            "reproduced_levels": 6,
            "false_positive_rate": 0.0,
            "fires_on_win": True,
            "result_path": exp4427.EXP4422_PATH,
        },
    )
    _write_json(
        root / exp4427.EXP4423_PATH,
        {
            "experiment": "experiment_4423_generic_first_contact_breadth",
            "target_game": "g50t",
            "honest_verdict": "partial: generic_first_contact_g50t_routed_missing_verifier_gap_logged",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "missing_verifier_gaps": [
                {
                    "gap_id": "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT",
                    "game": "g50t",
                    "status": "open",
                    "failure_mode": "needs_per_game_RE",
                    "missing_discriminator": (
                        "selectable verifier that distinguishes the target's winning delta "
                        "from the explored non-winning states"
                    ),
                    "candidate_design": (
                        "adapt Exp 4421 config-rule predicate grounding to this game's visible toggles"
                    ),
                    "loop_result_summary": {
                        "offline_reproduced": False,
                        "reproduced_levels": 0,
                        "mode": "standing_arc_loop_routing_only",
                    },
                }
            ],
            "result_path": exp4427.EXP4423_PATH,
        },
    )
    _write_json(
        root / exp4427.EXP4424_PATH,
        {
            "experiment": "experiment_4424_deeper_solved_game",
            "game": "sc25",
            "target_level": 2,
            "prior_best_level": 1,
            "honest_verdict": "complete: sc25_L2_hud_cleanup_fixed_reproduction_gap",
            "offline_reproduced": False,
            "reproduced_levels": 1,
            "new_levels_reproduced": 0,
            "residual_failing_mechanic": "sc25_l2_route_search_still_missing_after_hud_cleanup",
            "result_path": exp4427.EXP4424_PATH,
        },
    )
    _write_json(
        root / exp4427.EXP4425_PATH,
        {
            "experiment": "experiment_4425_config_rule_vocabulary_transfer",
            "honest_verdict": "complete: vocabulary_transfer_null",
            "logged_gaps": ["missing_vocabulary_seeded_repeat_bench"],
        },
    )
    _write_json(
        root / exp4427.EXP4426_PATH,
        {
            "experiment": "experiment_4426_arc_registry_repro_audit",
            "honest_verdict": "complete: registry_repro_audit",
            "milestone_409_reproduction_gates": [
                {
                    "experiment": "exp4421",
                    "artifact": exp4427.EXP4421_PATH,
                    "offline_reproduced": True,
                    "reproduced_levels": 1,
                    "new_levels_counted": 1,
                    "reproduction_gated": True,
                },
                {
                    "experiment": "exp4422",
                    "artifact": exp4427.EXP4422_PATH,
                    "offline_reproduced": True,
                    "reproduced_levels": 6,
                    "new_levels_counted": 0,
                    "reproduction_gated": True,
                },
                {
                    "experiment": "exp4423",
                    "artifact": exp4427.EXP4423_PATH,
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "new_levels_counted": 0,
                    "reproduction_gated": True,
                },
                {
                    "experiment": "exp4424",
                    "artifact": exp4427.EXP4424_PATH,
                    "offline_reproduced": False,
                    "reproduced_levels": 1,
                    "new_levels_counted": 0,
                    "reproduction_gated": True,
                },
            ],
        },
    )


def test_req_verify_4427_spec_declares_artifact_contract() -> None:
    """REQ-VERIFY-4427: OpenSpec declares the .409 gap hygiene contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4427",
        "SCENARIO-VERIFY-4427",
        "python/carnot/experiment_4427_verifier_gaps_hygiene.py",
        exp4427.RESULT_RELATIVE_PATH,
        "honest_verdict",
        "inference_substrate",
        "build_target_for_410",
        "cpu_artifact_reconciliation_no_llm",
    ):
        assert marker in spec
    for field in exp4427.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_4427_reconciles_missing_and_filled_gaps(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4427: .409 gaps become the .410 build target."""

    _write_minimal_repo(tmp_path)

    artifact = exp4427.run(tmp_path, now=lambda: 100.0)

    exp4427.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == exp4427.INFERENCE_SUBSTRATE
    assert artifact["verifier_gaps_reconciled"] is True
    assert artifact["build_target_for_410"]["gap_id"] == "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT"
    assert artifact["build_target_for_410"]["priority"] == "high"
    assert [gap["gap_id"] for gap in artifact["filled_gaps"]] == [
        "GAP-4421-S5I5-MARKER-COVERAGE",
        "GAP-4422-TR87-GLYPH-REWRITE-PERCEPTION",
    ]
    assert "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT" in [
        gap["gap_id"] for gap in artifact["appended_gaps"]
    ]
    assert "GAP-4424-SC25-L2-ROUTE-SEARCH" in [gap["gap_id"] for gap in artifact["appended_gaps"]]

    ledger = (tmp_path / exp4427.GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "### GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT" in ledger
    assert "- missing discriminator: selectable verifier that distinguishes" in ledger
    assert "- candidate design: adapt Exp 4421 config-rule predicate grounding" in ledger
    assert "- priority: high" in ledger
    assert "- build target for .410 planner: true" in ledger
    assert "### GAP-4424-SC25-L2-ROUTE-SEARCH" in ledger
    assert "- priority: medium" in ledger
    assert "### GAP-4421-S5I5-MARKER-COVERAGE\n- status: filled (exp4421_s5i5_marker_coverage)" in ledger
    assert (
        "### GAP-4422-TR87-GLYPH-REWRITE-PERCEPTION\n"
        "- status: filled (exp4422_tr87_glyph_rewrite_perception)"
    ) in ledger

    written = json.loads((tmp_path / exp4427.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_verify_4427_blocks_without_mutating_when_source_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-4427: missing required artifacts block and preserve the ledger."""

    _write_minimal_repo(tmp_path)
    original = (tmp_path / exp4427.GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    (tmp_path / exp4427.EXP4423_PATH).unlink()

    artifact = exp4427.run(tmp_path, now=lambda: 100.0)

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["verifier_gaps_reconciled"] is False
    assert artifact["appended_gaps"] == []
    assert artifact["filled_gaps"] == []
    assert artifact["build_target_for_410"] == {}
    assert (tmp_path / exp4427.GAPS_RELATIVE_PATH).read_text(encoding="utf-8") == original
    exp4427.validate_artifact(artifact)


def test_req_verify_4427_artifact_schema_rejects_malformed_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4427: schema validation catches fabricated terminal artifacts."""

    _write_minimal_repo(tmp_path)
    artifact = exp4427.run(tmp_path, now=lambda: 100.0)

    for field in exp4427.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4427.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal"):
        exp4427.validate_artifact({**artifact, "honest_verdict": "clean"})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4427.validate_artifact({**artifact, "inference_substrate": "llm"})
    with pytest.raises(ValueError, match="verifier_gaps_reconciled"):
        exp4427.validate_artifact({**artifact, "verifier_gaps_reconciled": "true"})
    with pytest.raises(ValueError, match="build_target_for_410"):
        exp4427.validate_artifact({**artifact, "build_target_for_410": []})
    with pytest.raises(ValueError, match="appended_gaps"):
        exp4427.validate_artifact({**artifact, "appended_gaps": {}})
    with pytest.raises(ValueError, match="build_target_for_410"):
        exp4427.validate_artifact({**artifact, "verifier_gaps_reconciled": True, "build_target_for_410": {}})
    with pytest.raises(ValueError, match="random_seed"):
        exp4427.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4427.validate_artifact({**artifact, "reproducibility_checksum": "not-a-sha"})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4427.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4427"]})


def test_req_verify_4427_defensive_helpers_cover_malformed_inputs(tmp_path: Path) -> None:
    """REQ-VERIFY-4427: defensive branches stay deterministic and CPU-only."""

    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bad.json").write_text("[]\n", encoding="utf-8")
    payload, error = exp4427._load_json(tmp_path, "bad.json")
    assert payload == {}
    assert error == "JSON top-level is not an object"
    assert exp4427._file_sha256(tmp_path / "missing.json") == ""
    assert exp4427._as_int("not-int") == 0

    preconditions, _payloads = exp4427.check_preconditions(tmp_path)
    assert preconditions["ok"] is False
    assert preconditions["ledger"]["readable"] is False

    assert exp4427._priority_from_headroom({"priority": "low"}, {}) == "low"
    assert exp4427._priority_from_headroom({}, {"offline_reproduced": False, "reproduced_levels": 0}) == "high"
    assert exp4427._priority_from_headroom({}, {"offline_reproduced": True, "reproduced_levels": 1}) == "medium"
    assert exp4427._headroom_score({}, {"target_level": 3, "reproduced_levels": 1}) == 2
    assert exp4427._headroom_score({}, {"offline_reproduced": False}) == 1
    assert exp4427._headroom_score({}, {"offline_reproduced": True}) == 0

    emitted = exp4427.collect_emitted_gaps(
        {
            "exp4423": {
                "missing_verifier_gaps": [
                    "not-a-gap",
                    {"gap_id": ""},
                    {
                        "gap_id": "GAP-LOW",
                        "priority": "low",
                        "missing_discriminator": "x",
                        "candidate_design": "y",
                    },
                ]
            }
        }
    )
    assert [gap["gap_id"] for gap in emitted] == ["GAP-LOW"]
    assert exp4427.collect_residual_gaps(
        {"exp4424": {"offline_reproduced": True, "residual_failing_mechanic": "filled"}}
    ) == []
    assert exp4427.choose_build_target([]) == {}

    filled_gap = {
        "gap_id": "GAP-FILLED-ABSENT",
        "status": "filled (fixture)",
        "evidence": "fixture evidence",
        "failure_mode": "old",
        "missing_discriminator": "x",
        "candidate_design": "y",
        "priority": "medium",
        "headroom": 0,
        "movement": "filled",
    }
    ledger, appended = exp4427.reconcile_ledger(
        "# Verifier Gaps\n",
        open_gaps=[],
        filled_gaps=[filled_gap],
        build_target={},
    )
    assert [gap["gap_id"] for gap in appended] == ["GAP-FILLED-ABSENT"]
    assert "- status: filled (fixture)" in ledger

    no_status, replaced = exp4427._replace_existing_gap_status(
        "### GAP-NO-STATUS\n- evidence: old\n",
        {"gap_id": "GAP-NO-STATUS", "status": "filled (fixture)", "evidence": "new"},
    )
    assert replaced is True
    assert "### GAP-NO-STATUS\n- status: filled (fixture)" in no_status
    unchanged, replaced = exp4427._replace_existing_gap_status(
        "# no matching gap\n",
        {"gap_id": "GAP-MISSING", "status": "filled (fixture)", "evidence": "new"},
    )
    assert unchanged == "# no matching gap\n"
    assert replaced is False

    open_gap = {
        "gap_id": "GAP-OPEN-MARKED",
        "status": "open",
        "evidence": "fixture",
        "failure_mode": "needs selector",
        "missing_discriminator": "selector",
        "candidate_design": "build selector",
        "priority": "high",
        "headroom": 1,
        "movement": "newly_logged",
    }
    marked, _ = exp4427.reconcile_ledger(
        "# Verifier Gaps\n",
        open_gaps=[open_gap],
        filled_gaps=[],
        build_target=open_gap,
    )
    refreshed, appended = exp4427.reconcile_ledger(
        marked,
        open_gaps=[{**open_gap, "candidate_design": "updated selector"}],
        filled_gaps=[],
        build_target=open_gap,
    )
    assert appended == []
    assert "updated selector" in refreshed
