"""Tests for Exp 3681 G2 reproducer prep for operator re-freeze.

Spec: REQ-PUBLISH-040, SCENARIO-PUBLISH-040, SCENARIO-PUBLISH-040B.
"""

from __future__ import annotations

import json
import importlib
import types
from pathlib import Path
from typing import Any

import pytest

from scripts import reproduce_fover_headline as reproducer
from scripts import experiment_3681_g2_reproducer_prep_operator_refreeze_package as exp3681


def _exp3680_fixture(*, confirmed: bool = True) -> dict[str, Any]:
    per_seed_lc = [0.018314, 0.013386, 0.029904, 0.030056, 0.019084]
    return {
        "honest_verdict": (
            "complete: dependency_aware_g1_rigor_confirmed_headline_candidate_exceeds_frozen_0_9131"
            if confirmed
            else "complete: dependency_aware_no_significant_gain_under_g1_protocol_frozen_headline_stands"
        ),
        "dependency_aware_g1_rigor_confirmed": confirmed,
        "production_auroc_dependency_aware": 0.925328,
        "production_auroc_ci": {
            "point": 0.924869,
            "ci95": [0.91699, 0.932891],
            "bootstrap_seeds": [42, 137, 271, 314, 1729],
            "n_bootstrap_per_seed": 200,
        },
        "learning_contribution_dependency_aware": 0.022149,
        "per_seed_results": [
            {
                "seed": seed,
                "production_auroc_dependency_aware": production,
                "learning_contribution_dependency_aware": lc,
            }
            for seed, production, lc in zip(
                [42, 137, 271, 314, 1729],
                [0.924802, 0.92095, 0.916566, 0.930296, 0.934028],
                per_seed_lc,
                strict=True,
            )
        ],
        "random_seed": 42,
        "random_seeds_used": [42, 137, 271, 314, 1729],
        "reproducibility_checksum": "exp3680-checksum",
    }


def _frozen_result(*, in_ci: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: FoVer memory-leakage v3 measured",
        "condition_a_production_auroc_mean": 0.9131 if in_ci else 0.80,
        "condition_b_architecture_only_auroc_mean": 0.8947,
        "learning_contribution_ci95": {
            "mean": 0.0185 if in_ci else 0.001,
            "low": 0.0125,
            "high": 0.0245,
        },
        "reproducibility_checksum": "frozen-checksum",
    }


def _candidate_result(*, in_ci: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: dependency-aware candidate recomputed",
        "production_auroc_dependency_aware": 0.925328 if in_ci else 0.80,
        "production_auroc_ci": {"point": 0.924869, "ci95": [0.91699, 0.932891]},
        "learning_contribution_dependency_aware": 0.022149 if in_ci else 0.002,
        "learning_contribution_ci95": {
            "mean": 0.022149,
            "low": 0.012868,
            "high": 0.03143,
        },
        "candidate_production_auroc_in_exp3680_ci": in_ci,
        "candidate_learning_contribution_in_exp3680_ci": in_ci,
        "candidate_reproduction_asserts_in_ci": in_ci,
        "reproducibility_checksum": "candidate-checksum",
    }


def _publication_gate(*, paper_ready: bool = False) -> dict[str, Any]:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {
                "pass": True,
                "source": "experiment_2850_fover_dual_condition_integrity_v4.json",
                "detail": "FoVer dual-condition AUROC artifact present",
            },
            "G2": {"pass": False, "detail": "no independent reproducer"},
            "G3": {"pass": True, "detail": "clean"},
            "G4": {
                "pass": True,
                "source": "experiment_2850_fover_dual_condition_integrity_v4.json",
                "detail": "random_seed/seeds=True, reproducibility_checksum=True",
            },
        },
        "unmet_gates": ["G2"],
    }


def test_reproducer_derives_candidate_bounds_from_exp3680_fixture() -> None:
    """REQ-PUBLISH-040: candidate assertions use Exp 3680-derived bounds."""

    bounds = reproducer.dependency_aware_candidate_bounds_from_artifact(_exp3680_fixture())

    assert bounds["production_auroc_dependency_aware"]["ci95"] == [0.91699, 0.932891]
    assert bounds["production_auroc_dependency_aware"]["point"] == 0.924869
    assert bounds["learning_contribution_dependency_aware"]["point"] == 0.022149
    assert bounds["learning_contribution_dependency_aware"]["ci95"] == pytest.approx(
        [0.012868, 0.03143]
    )


def test_reproducer_candidate_ci_check_requires_both_numbers() -> None:
    """SCENARIO-PUBLISH-040: the candidate path asserts AUROC and learning CI."""

    bounds = reproducer.dependency_aware_candidate_bounds_from_artifact(_exp3680_fixture())
    assert reproducer.check_dependency_aware_candidate_ci(_candidate_result(), bounds) == (
        True,
        True,
    )
    assert reproducer.check_dependency_aware_candidate_ci(
        _candidate_result(in_ci=False),
        bounds,
    ) == (False, False)


def test_reproducer_candidate_helpers_cover_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-PUBLISH-040: additive candidate helpers load, run, and fail closed."""

    exp3680_path = tmp_path / reproducer.EXP3680_CANDIDATE_REL_PATH
    exp3680_path.parent.mkdir(parents=True)
    exp3680_path.write_text(json.dumps(_exp3680_fixture()), encoding="utf-8")
    assert reproducer.load_dependency_aware_candidate_bounds(tmp_path)[
        "production_auroc_dependency_aware"
    ]["ci95"] == [0.91699, 0.932891]
    assert reproducer._seed_t_ci95([0.2]) == {"mean": 0.2, "low": 0.2, "high": 0.2}
    with pytest.raises(ValueError, match="at least one seed"):
        reproducer._seed_t_ci95([])
    with pytest.raises(ValueError, match="production_auroc_ci"):
        reproducer.dependency_aware_candidate_bounds_from_artifact({})
    with pytest.raises(ValueError, match="two bounds"):
        reproducer.dependency_aware_candidate_bounds_from_artifact(
            {"production_auroc_ci": {"point": 0.9, "ci95": [0.8]}}
        )
    with pytest.raises(ValueError, match="learning contribution"):
        reproducer.dependency_aware_candidate_bounds_from_artifact(
            {
                "production_auroc_ci": {"point": 0.9, "ci95": [0.8, 1.0]},
                "per_seed_results": [],
            }
        )
    without_lc_point = _exp3680_fixture()
    without_lc_point.pop("learning_contribution_dependency_aware")
    assert reproducer.dependency_aware_candidate_bounds_from_artifact(without_lc_point)[
        "learning_contribution_dependency_aware"
    ]["point"] == pytest.approx(0.022149)

    from carnot.verify import dependency_aware_dual_condition_integrity as dep_mod

    monkeypatch.setattr(dep_mod, "build_artifact", lambda *args, **kwargs: _exp3680_fixture())
    candidate = reproducer.run_dependency_aware_candidate_reproduction(tmp_path, seeds=(42,), n_examples=10)
    assert candidate["candidate_reproduction_asserts_in_ci"] is True
    assert candidate["learning_contribution_ci95"]["mean"] == pytest.approx(0.022149)

    monkeypatch.setattr(
        reproducer,
        "run_dependency_aware_candidate_reproduction",
        lambda root: {
            **_candidate_result(),
            "candidate_exp3680_assertion_bounds": reproducer.load_dependency_aware_candidate_bounds(
                tmp_path
            ),
        },
    )
    assert reproducer.main(["--dependency-aware-candidate"]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out

    monkeypatch.setattr(
        reproducer,
        "run_dependency_aware_candidate_reproduction",
        lambda root: {
            **_candidate_result(in_ci=False),
            "candidate_exp3680_assertion_bounds": reproducer.load_dependency_aware_candidate_bounds(
                tmp_path
            ),
        },
    )
    assert reproducer.main(["--dependency-aware-candidate"]) == 1
    assert "RESULT: FAIL" in capsys.readouterr().out

    monkeypatch.setattr(
        reproducer,
        "run_dependency_aware_candidate_reproduction",
        lambda root: {"honest_verdict": "complete: blocked_fixture"},
    )
    assert reproducer.main(["--dependency-aware-candidate"]) == 1
    assert "BLOCKED" in capsys.readouterr().err

    monkeypatch.setattr(reproducer, "run_reproduction", lambda root: _frozen_result())
    assert reproducer.main([]) == 0
    assert "FoVer headline reproduces" in capsys.readouterr().out
    monkeypatch.setattr(reproducer, "run_reproduction", lambda root: _frozen_result(in_ci=False))
    assert reproducer.main([]) == 1
    assert "RESULT: FAIL" in capsys.readouterr().out
    monkeypatch.setattr(
        reproducer,
        "run_reproduction",
        lambda root: {"honest_verdict": "blocked_fr11_state_files", "blocked_resources": ["x"]},
    )
    assert reproducer.main([]) == 1
    assert "BLOCKED" in capsys.readouterr().err


@pytest.mark.parametrize(
    (
        "scenario",
        "g1_confirmed",
        "reproducer_importable",
        "reproducer_extended",
        "frozen_green",
        "candidate_in_ci",
        "north_star_unmodified",
        "ci_workflow_unmodified",
        "frozen_headline_unchanged",
        "expected_verdict",
    ),
    [
        pytest.param(
            "package_ready",
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            exp3681.READY_VERDICT,
            id="package_ready",
        ),
        pytest.param(
            "candidate_not_confirmed",
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            exp3681.BLOCKED_VERDICT,
            id="candidate_not_confirmed",
        ),
        pytest.param(
            "blocked",
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            exp3681.BLOCKED_VERDICT,
            id="blocked",
        ),
    ],
)
def test_exp3681_classifies_honest_outcomes_without_single_success_string(
    scenario: str,
    g1_confirmed: bool,
    reproducer_importable: bool,
    reproducer_extended: bool,
    frozen_green: bool,
    candidate_in_ci: bool,
    north_star_unmodified: bool,
    ci_workflow_unmodified: bool,
    frozen_headline_unchanged: bool,
    expected_verdict: str,
) -> None:
    """SCENARIO-PUBLISH-040/B: anti-poison outcomes include ready and blocked."""

    assert scenario in {"package_ready", "candidate_not_confirmed", "blocked"}
    assert (
        exp3681.classify_honest_verdict(
            g1_candidate_confirmed=g1_confirmed,
            reproducer_importable=reproducer_importable,
            reproducer_extended=reproducer_extended,
            existing_0_9131_reproduction_still_green=frozen_green,
            candidate_reproduction_asserts_in_ci=candidate_in_ci,
            north_star_unmodified_assert=north_star_unmodified,
            ci_workflow_unmodified_assert=ci_workflow_unmodified,
            frozen_headline_unchanged_assert=frozen_headline_unchanged,
        )
        == expected_verdict
    )


def test_exp3681_builds_operator_only_package_artifact(tmp_path: Path) -> None:
    """REQ-PUBLISH-040: artifact records operator actions and non-edits."""

    exp3680 = _exp3680_fixture()
    candidate = _candidate_result()
    artifact = exp3681.build_artifact(
        repo_root=tmp_path,
        started_s=10.0,
        now_s=25.0,
        exp3680_artifact=exp3680,
        reproducer_importable=True,
        reproducer_extended=True,
        frozen_reproduction_result=_frozen_result(),
        candidate_reproduction_result=candidate,
        publication_gate_before=_publication_gate(paper_ready=False),
        publication_gate_after=_publication_gate(paper_ready=False),
        north_star_hash_before="north",
        north_star_hash_after="north",
        ci_workflow_hash_before="workflow",
        ci_workflow_hash_after="workflow",
        github_run_triggered=False,
    )

    assert artifact["honest_verdict"] == exp3681.READY_VERDICT
    for field in exp3681.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert artifact["inference_substrate"] == exp3681.INFERENCE_SUBSTRATE
    assert artifact["existing_0_9131_reproduction_still_green"] is True
    assert artifact["candidate_reproduction_asserts_in_ci"] is True
    assert artifact["north_star_unmodified_assert"] is True
    assert artifact["ci_workflow_unmodified_assert"] is True
    assert artifact["frozen_headline_unchanged_assert"] is True
    assert artifact["draft_ci_workflow_assertion_bounds"] == (
        exp3681.draft_ci_workflow_assertion_bounds(exp3680)
    )
    assert all(step.startswith("OPERATOR-ACTION:") for step in artifact["operator_checklist"])
    assert any("0.9131 stays the headline" in step for step in artifact["operator_checklist"])


def test_exp3681_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PUBLISH-040: helper branches stay honest on missing or bad inputs."""

    assert exp3681._sha256_file(tmp_path / "missing") == "missing"
    assert exp3681.load_exp3680_artifact(tmp_path) == {}
    path = tmp_path / exp3681.EXP3680_REL_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_exp3680_fixture()), encoding="utf-8")
    assert exp3681.load_exp3680_artifact(tmp_path)["dependency_aware_g1_rigor_confirmed"] is True

    assert exp3681.reproducer_has_candidate_extension(None) is False
    fake_module = types.ModuleType("fake_reproducer")
    fake_source = tmp_path / "fake_reproducer.py"
    fake_source.write_text("--dependency-aware-candidate\n", encoding="utf-8")
    fake_module.__file__ = str(fake_source)
    fake_module.run_dependency_aware_candidate_reproduction = lambda root: _candidate_result()
    fake_module.check_dependency_aware_candidate_ci = lambda result, bounds: (True, True)
    fake_module.dependency_aware_candidate_bounds_from_artifact = lambda artifact: {}
    fake_module.run_reproduction = lambda root: _frozen_result()
    fake_module.check_acceptance_ci = reproducer.check_acceptance_ci
    assert exp3681.reproducer_has_candidate_extension(fake_module) is True
    assert exp3681.run_frozen_reproduction(fake_module, tmp_path)["reproducibility_checksum"] == (
        "frozen-checksum"
    )
    assert exp3681.run_candidate_reproduction(fake_module, tmp_path)["reproducibility_checksum"] == (
        "candidate-checksum"
    )
    assert exp3681.frozen_reproduction_green(None, _frozen_result()) is False
    assert exp3681.frozen_reproduction_green(fake_module, {"honest_verdict": "blocked_x"}) is False

    original_import = importlib.import_module

    def fake_import(name: str) -> types.ModuleType:
        if name == "scripts.reproduce_fover_headline":
            raise ImportError("no reproducer")
        return original_import(name)

    assert exp3681.reproducer_import_status()[0] is True
    monkeypatch.setattr(exp3681.importlib, "import_module", fake_import)
    assert exp3681.reproducer_import_status() == (False, None)

    pub_script = tmp_path / "scripts" / "publication_gate.py"
    pub_script.parent.mkdir()
    pub_script.write_text(
        "def evaluate():\n    return {'paper_ready': False, 'gates': {}}\n",
        encoding="utf-8",
    )
    assert exp3681.evaluate_publication_gate(tmp_path)["paper_ready"] is False
    monkeypatch.setattr(exp3681.importlib.util, "spec_from_file_location", lambda *args: None)
    assert exp3681.evaluate_publication_gate(tmp_path)["paper_ready"] is None

    frozen = tmp_path / exp3681.FROZEN_SOURCE_REL_PATH
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(
        json.dumps({"condition_a_production_auroc_mean": 0.9131336}),
        encoding="utf-8",
    )
    assert exp3681.publication_gate_reads_frozen_0_9131(tmp_path, _publication_gate()) is True
    assert exp3681.publication_gate_reads_frozen_0_9131(
        tmp_path,
        {"gates": {"G1": {"detail": "still 0.9131"}, "G4": {}}},
    ) is True
    assert exp3681._blocked_result("fixture", RuntimeError("boom"))["error"] == "RuntimeError: boom"

    with pytest.raises(ValueError, match="missing required"):
        exp3681.validate_artifact({})
    valid = exp3681.build_artifact(
        repo_root=tmp_path,
        started_s=0.0,
        now_s=1.0,
        exp3680_artifact={},
        reproducer_importable=False,
        reproducer_extended=False,
        frozen_reproduction_result={},
        candidate_reproduction_result={},
        publication_gate_before=_publication_gate(),
        publication_gate_after=_publication_gate(),
        north_star_hash_before="a",
        north_star_hash_after="a",
        ci_workflow_hash_before="b",
        ci_workflow_hash_after="b",
        github_run_triggered=False,
        reproducer_module=fake_module,
    )
    assert valid["honest_verdict"] == exp3681.BLOCKED_VERDICT
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3681.validate_artifact({**valid, "honest_verdict": "complete: invented"})
    with pytest.raises(ValueError, match="field_principles"):
        exp3681.validate_artifact({**valid, "field_principles": []})
    with pytest.raises(ValueError, match="missing field principles"):
        exp3681.validate_artifact({**valid, "field_principles": {}})
    with pytest.raises(ValueError, match="bare boolean"):
        exp3681.validate_artifact({**valid, "reproducer_extended": 1})
    with pytest.raises(ValueError, match="operator_checklist"):
        exp3681.validate_artifact({**valid, "operator_checklist": "bad"})
    with pytest.raises(ValueError, match="OPERATOR-ACTION"):
        exp3681.validate_artifact({**valid, "operator_checklist": ["do it"]})


def test_exp3681_write_artifact_failure_and_main_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-PUBLISH-040B: write failures and precondition blocks stay terminal."""

    (tmp_path / "ops").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "ops" / "north-star.md").write_text("FoVer AUROC 0.9131\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "reproduce-fover-headline.yml").write_text(
        "run: python3 scripts/reproduce_fover_headline.py\n",
        encoding="utf-8",
    )
    exp3680_path = tmp_path / exp3681.EXP3680_REL_PATH
    exp3680_path.parent.mkdir(parents=True)
    exp3680_path.write_text(json.dumps(_exp3680_fixture()), encoding="utf-8")

    monkeypatch.setattr(exp3681, "reproducer_import_status", lambda: (True, reproducer))
    monkeypatch.setattr(exp3681, "run_frozen_reproduction", lambda module, root: (_ for _ in ()).throw(RuntimeError("frozen")))
    monkeypatch.setattr(exp3681, "run_candidate_reproduction", lambda module, root: (_ for _ in ()).throw(RuntimeError("candidate")))
    monkeypatch.setattr(exp3681, "evaluate_publication_gate", lambda root: _publication_gate())
    failed_output = exp3681.write_artifact(tmp_path, output_path="results/failed.json")
    failed = json.loads(failed_output.read_text(encoding="utf-8"))
    assert failed["honest_verdict"] == exp3681.BLOCKED_VERDICT
    assert failed["frozen_reproduction_result"]["error"] == "RuntimeError: frozen"
    assert failed["candidate_reproduction_result"]["error"] == "RuntimeError: candidate"

    exp3680_path.write_text(json.dumps(_exp3680_fixture(confirmed=False)), encoding="utf-8")
    blocked_output = exp3681.write_artifact(tmp_path, output_path="results/blocked.json")
    blocked = json.loads(blocked_output.read_text(encoding="utf-8"))
    assert blocked["frozen_reproduction_result"]["honest_verdict"] == "blocked_preconditions_not_met"

    result_path = tmp_path / "results" / "main.json"

    def fake_write(root: Path) -> Path:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"honest_verdict": exp3681.READY_VERDICT}), encoding="utf-8")
        return result_path

    monkeypatch.setattr(exp3681, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp3681, "write_artifact", fake_write)
    assert exp3681.main() == 0
    assert exp3681.READY_VERDICT in capsys.readouterr().out


def test_exp3681_write_artifact_preserves_operator_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PUBLISH-040: writing the package leaves north-star and CI YAML untouched."""

    (tmp_path / "ops").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "ops" / "north-star.md").write_text("FoVer AUROC 0.9131\n", encoding="utf-8")
    workflow = tmp_path / ".github" / "workflows" / "reproduce-fover-headline.yml"
    workflow.write_text("run: python3 scripts/reproduce_fover_headline.py\n", encoding="utf-8")
    exp3680_path = tmp_path / "results" / "experiment_3680_dependency_aware_dual_condition_integrity.json"
    exp3680_path.parent.mkdir()
    exp3680_path.write_text(json.dumps(_exp3680_fixture()), encoding="utf-8")

    monkeypatch.setattr(exp3681, "reproducer_import_status", lambda: (True, reproducer))
    monkeypatch.setattr(exp3681, "run_frozen_reproduction", lambda module, root: _frozen_result())
    monkeypatch.setattr(
        exp3681,
        "run_candidate_reproduction",
        lambda module, root: _candidate_result(),
    )
    monkeypatch.setattr(
        exp3681,
        "evaluate_publication_gate",
        lambda root: _publication_gate(paper_ready=False),
    )

    output = exp3681.write_artifact(tmp_path, started_s=0.0, now_s=3.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["honest_verdict"] == exp3681.READY_VERDICT
    assert (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8") == "FoVer AUROC 0.9131\n"
    assert workflow.read_text(encoding="utf-8") == "run: python3 scripts/reproduce_fover_headline.py\n"
    assert saved["ci_workflow_unmodified_assert"] is True
