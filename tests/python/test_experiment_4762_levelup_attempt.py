"""Tests for Exp 4762 ARC level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4762,
SCENARIO-ARC-WMTE-4762-ROTATION-PRECHECK,
SCENARIO-ARC-WMTE-4762-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4762-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_4762_levelup_attempt as exp4762


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: re86
  reproducibility: reproduced
  levels_reproduced: 2
- game: sb26
  reproducibility: reproduced
  levels_reproduced: 2
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
- game: lf52
  reproducibility: reproduced
  levels_reproduced: 2
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 2
reproducible_total_levels: 65
"""


def _loop_result(game: str, reached_level: int, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "solve_provenance": "development_proxy",
        "inference_substrate": "standing_arc_loop_offline_no_quota",
        "learned_verifier_checkpoint": f"models/arc_verifier_{game}.json",
        "reproduction_gate": {
            "game": game,
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": ["{\"action\":1}"],
    }


def test_req_arc_wmte_4762_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4762: OpenSpec declares the 4762 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4762.SPEC_REFS:
        assert ref in spec
    assert exp4762.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4762.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4762_rotation_precheck_prefers_public_deepen() -> None:
    """SCENARIO-ARC-WMTE-4762-ROTATION-PRECHECK: L2 public games become deepen attempts."""

    registry = yaml.safe_load(_registry_text())
    selected = exp4762.select_rotation_attempts(registry)

    assert [row["game"] for row in selected[:4]] == ["re86", "sb26", "bp35", "lf52"]
    assert all(row["reason"] == "preferred_public_already_reproduced_deepen" for row in selected[:4])
    assert all(row["target_level"] == 3 for row in selected[:4])
    assert selected[4]["game"] == "ka59"
    assert selected[4]["reason"] == "shallowest_adaptered_fallback"


def test_scenario_arc_wmte_4762_same_depth_reproduction_is_not_a_bank() -> None:
    """SCENARIO-ARC-WMTE-4762-REPRODUCTION-GATE: same-depth gates retire with no bank."""

    attempt = exp4762.summarize_loop_attempt(
        game="re86",
        prior_level=2,
        target_level=3,
        loop_result=_loop_result("re86", 2),
        loop_result_path="results/arc_loop_solve_re86.json",
    )

    assert attempt["offline_reproduced_existing_depth"] is True
    assert attempt["new_levels_banked"] == 0
    assert attempt["residual_cause"] == "reproduced_existing_or_lower_level"
    assert "same-depth" in attempt["dead_end"]


def test_req_arc_wmte_4762_builds_no_bank_artifact_without_fabrication() -> None:
    """REQ-ARC-WMTE-4762: no-bank artifact keeps the registry total unchanged."""

    registry = yaml.safe_load(_registry_text())
    attempts = [
        exp4762.summarize_loop_attempt(
            game=game,
            prior_level=2,
            target_level=3,
            loop_result=_loop_result(game, 2),
            loop_result_path=f"results/arc_loop_solve_{game}.json",
        )
        for game in ("re86", "sb26", "bp35", "lf52")
    ]
    attempts.append(
        exp4762.summarize_timed_no_gate(
            game="dc22",
            prior_level=2,
            target_level=3,
            elapsed_s=115.0,
            loop_result_path="results/arc_loop_solve_dc22.json",
        )
    )

    artifact = exp4762.build_artifact(
        registry=registry,
        attempts=attempts,
        preconditions_checked={
            "AGENTS.md": True,
            "CODEX.md": True,
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "induction_needed": False,
        },
    )

    assert artifact["honest_verdict"] == "complete_re86_no_new_level_residual_existing_depth"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_update"]["updated"] is False
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 65
    assert artifact["schema_errors"] == []
    assert exp4762.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4762_success_requires_strictly_new_reproduced_level() -> None:
    """REQ-ARC-WMTE-4762: success requires a gate above prior registry depth."""

    registry = yaml.safe_load(_registry_text())
    attempt = exp4762.summarize_loop_attempt(
        game="re86",
        prior_level=2,
        target_level=3,
        loop_result=_loop_result("re86", 3),
        loop_result_path="results/arc_loop_solve_re86.json",
    )

    artifact = exp4762.build_artifact(
        registry=registry,
        attempts=[attempt],
        preconditions_checked={"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}},
    )

    assert artifact["honest_verdict"] == "success_re86_L3_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["registry_update"]["updated"] is True
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 66
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4762_schema_guards_required_fields() -> None:
    """REQ-ARC-WMTE-4762: schema validation rejects missing principles and bad checksums."""

    registry = yaml.safe_load(_registry_text())
    attempt = exp4762.summarize_loop_attempt(
        game="re86",
        prior_level=2,
        target_level=3,
        loop_result=_loop_result("re86", 2),
        loop_result_path="results/arc_loop_solve_re86.json",
    )
    artifact = exp4762.build_artifact(
        registry=registry,
        attempts=[attempt],
        preconditions_checked={"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}},
    )

    missing = dict(artifact)
    missing.pop("honest_verdict")
    assert "missing_field:honest_verdict" in exp4762.artifact_schema_errors(missing)

    wrong_principle = dict(artifact)
    wrong_principle["field_principles"] = dict(artifact["field_principles"])
    wrong_principle["field_principles"]["honest_verdict"] = "wrong"
    assert "missing_principle:honest_verdict" in exp4762.artifact_schema_errors(wrong_principle)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    assert "invalid_reproducibility_checksum" in exp4762.artifact_schema_errors(bad_checksum)

    stale_checksum = dict(artifact)
    stale_checksum["reproducibility_checksum"] = "0" * 64
    assert "checksum_mismatch" in exp4762.artifact_schema_errors(stale_checksum)

    bad_prefix = dict(artifact)
    bad_prefix["honest_verdict"] = "partial_re86"
    assert "honest_verdict_missing_terminal_prefix" in exp4762.artifact_schema_errors(bad_prefix)

    bad_provenance = dict(artifact)
    bad_provenance["solve_provenance"] = "outer_loop_re"
    assert "solve_provenance_mismatch" in exp4762.artifact_schema_errors(bad_provenance)

    bad_substrate = dict(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate_mismatch" in exp4762.artifact_schema_errors(bad_substrate)

    bad_oracle = dict(artifact)
    bad_oracle["verifier_is_oracle"] = False
    assert "verifier_is_oracle_must_be_true" in exp4762.artifact_schema_errors(bad_oracle)

    fabricated_bank = dict(artifact)
    fabricated_bank["new_levels_banked"] = 1
    fabricated_bank["reproducibility_checksum"] = exp4762.stable_checksum(fabricated_bank)
    assert "bank_without_offline_reproduction" in exp4762.artifact_schema_errors(fabricated_bank)

    fabricated_repro = dict(artifact)
    fabricated_repro["offline_reproduced"] = True
    fabricated_repro["reproducibility_checksum"] = exp4762.stable_checksum(fabricated_repro)
    assert "offline_reproduced_true_without_new_bank" in exp4762.artifact_schema_errors(fabricated_repro)


def test_req_arc_wmte_4762_handles_malformed_registry_and_empty_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4762: malformed registry data cannot fabricate progress."""

    registry = {
        "games": [
            "bad-row",
            {"levels_reproduced": 7},
            {"game": "re86", "levels_reproduced": "bad"},
        ],
        "reproducible_total_levels": object(),
    }

    assert exp4762.registry_levels(registry) == {"re86": 0}
    assert exp4762.registry_total_levels(registry) == 0
    monkeypatch.setattr(exp4762, "EXTRA_PROBED_TARGETS", ("dc22",))
    assert all(row["game"] != "dc22" for row in exp4762.select_rotation_attempts(registry))

    empty_artifact = exp4762.build_artifact(
        registry=registry,
        attempts=[],
        preconditions_checked={"offline_arcade": {"ok": True}},
    )
    assert empty_artifact["honest_verdict"] == "complete_none_no_new_level_residual_no_attempts"
    assert empty_artifact["schema_errors"] == []

    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4762.load_registry(registry_file) == {}


def test_req_arc_wmte_4762_failed_gate_records_residual() -> None:
    """SCENARIO-ARC-WMTE-4762-REPRODUCTION-GATE: failed gates stay residuals."""

    failed = _loop_result("re86", 3, reproduced=False)
    failed["reproduction_gate"] = {"reached_level": object(), "reproduced": False}

    attempt = exp4762.summarize_loop_attempt(
        game="re86",
        prior_level=2,
        target_level=3,
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_re86.json",
    )
    artifact = exp4762.build_artifact(
        registry=yaml.safe_load(_registry_text()),
        attempts=[attempt],
        preconditions_checked={"offline_arcade": {"ok": True}},
    )

    assert attempt["reached_level"] == 0
    assert attempt["residual_cause"] == "offline_reproduction_failed"
    assert "offline_reproduction_failed" in attempt["dead_end"]
    assert artifact["honest_verdict"] == "complete_re86_no_new_level_residual_offline_reproduction_failed"


def test_scenario_arc_wmte_4762_stable_artifact_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4762-STABLE-ARTIFACT: writer emits deterministic JSON."""

    payload = {
        "z": 1,
        "a": {"b": 2},
    }
    out = tmp_path / exp4762.RESULT_RELATIVE_PATH

    exp4762.write_artifact(payload, path=out)

    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert out.read_text(encoding="utf-8").startswith("{\n  \"a\"")


def test_req_arc_wmte_4762_main_writes_terminal_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-4762: main writes a stable terminal artifact from loop results."""

    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / exp4762.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    for game in ("re86", "sb26", "bp35", "lf52"):
        (tmp_path / "results" / f"arc_loop_solve_{game}.json").write_text(
            json.dumps(_loop_result(game, 2)),
            encoding="utf-8",
        )

    monkeypatch.setattr(exp4762, "REPO", tmp_path)
    monkeypatch.setattr(exp4762, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4762, "REGISTRY", tmp_path / exp4762.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(exp4762, "ARTIFACT", tmp_path / exp4762.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp4762, "check_preconditions", lambda: {"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}})

    assert exp4762.main([]) == 0

    written = json.loads((tmp_path / exp4762.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"].startswith("complete_re86_no_new_level")
    assert any(row["game"] == "ka59" and row["residual_cause"] == "time_budget_no_terminal_gate" for row in written["attempted_games"])
    assert written["schema_errors"] == []
