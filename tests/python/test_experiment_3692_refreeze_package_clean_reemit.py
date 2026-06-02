"""Tests for Exp 3692 clean operator re-freeze package re-emission.

Spec: REQ-PUBLISH-3692, SCENARIO-PUBLISH-3692, SCENARIO-PUBLISH-3692B.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from scripts import reproduce_fover_headline as reproducer
from scripts import experiment_3692_refreeze_package_clean_reemit as exp3692


def _exp3680_fixture(*, confirmed: bool = True) -> dict[str, Any]:
    per_seed_lc = [0.018314, 0.013386, 0.029904, 0.030056, 0.019084]
    return {
        "dependency_aware_g1_rigor_confirmed": confirmed,
        "production_auroc_dependency_aware": 0.925328,
        "production_auroc_ci": {
            "point": 0.924869,
            "ci95": [0.91699, 0.932891],
            "bootstrap_seeds": [42, 137, 271, 314, 1729],
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
        "reproducibility_checksum": "exp3680-checksum",
    }


def _frozen_result(*, in_ci: bool = True, with_markers: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    if with_markers:
        payload["model_specs"] = {
            "headline_required_any_of": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
        }
        payload["target_model"] = "torch.cuda local model marker"
        payload["field_principles"] = {"model_specs": "Mandated SOTA GGUF recorded."}
    return payload


def _candidate_result(*, in_ci: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: dependency-aware candidate recomputed",
        "production_auroc_dependency_aware": 0.925328 if in_ci else 0.80,
        "learning_contribution_dependency_aware": 0.022149 if in_ci else 0.002,
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
                "detail": "FoVer dual-condition AUROC 0.9131 artifact present",
            },
            "G4": {
                "pass": True,
                "source": "experiment_2850_fover_dual_condition_integrity_v4.json",
                "detail": "0.9131 remains frozen",
            },
        },
    }


@pytest.mark.parametrize(
    (
        "scenario",
        "g1_confirmed",
        "reproducer_importable",
        "reproducer_extended",
        "frozen_green",
        "candidate_in_ci",
        "adversarial_clean",
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
            exp3692.READY_VERDICT,
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
            exp3692.BLOCKED_VERDICT,
            id="candidate_not_confirmed",
        ),
        pytest.param(
            "blocked",
            True,
            False,
            False,
            False,
            False,
            False,
            exp3692.BLOCKED_VERDICT,
            id="blocked",
        ),
    ],
)
def test_req_publish_3692_classifies_all_honest_outcomes(
    scenario: str,
    g1_confirmed: bool,
    reproducer_importable: bool,
    reproducer_extended: bool,
    frozen_green: bool,
    candidate_in_ci: bool,
    adversarial_clean: bool,
    expected_verdict: str,
) -> None:
    """SCENARIO-PUBLISH-3692/B: anti-poison outcomes include ready and blocked."""

    assert scenario in {"package_ready", "candidate_not_confirmed", "blocked"}
    assert (
        exp3692.classify_honest_verdict(
            g1_candidate_confirmed=g1_confirmed,
            reproducer_importable=reproducer_importable,
            reproducer_extended=reproducer_extended,
            existing_0_9131_reproduction_still_green=frozen_green,
            candidate_reproduction_asserts_in_ci=candidate_in_ci,
            adversarial_verify_clean=adversarial_clean,
            north_star_unmodified_assert=True,
            ci_workflow_unmodified_assert=True,
            frozen_headline_unchanged_assert=True,
        )
        == expected_verdict
    )


def test_req_publish_3692_builds_clean_operator_package(tmp_path: Path) -> None:
    """REQ-PUBLISH-3692: clean artifact strips stale marker fields."""

    artifact = exp3692.build_artifact(
        repo_root=tmp_path,
        started_s=1.0,
        now_s=5.0,
        exp3680_artifact=_exp3680_fixture(),
        reproducer_importable=True,
        reproducer_extended=True,
        adversarial_verify_clean=True,
        frozen_reproduction_result=_frozen_result(with_markers=True),
        candidate_reproduction_result=_candidate_result(),
        publication_gate_before=_publication_gate(),
        publication_gate_after=_publication_gate(),
        north_star_hash_before="north",
        north_star_hash_after="north",
        ci_workflow_hash_before="workflow",
        ci_workflow_hash_after="workflow",
        github_run_triggered=False,
    )

    encoded = json.dumps(artifact, sort_keys=True)
    assert artifact["honest_verdict"] == exp3692.READY_VERDICT
    assert artifact["inference_substrate"] == exp3692.INFERENCE_SUBSTRATE
    assert artifact["adversarial_verify_clean"] is True
    assert "GGUF" not in encoded
    assert "torch.cuda" not in encoded
    assert artifact["frozen_reproduction_result"]["condition_a_production_auroc_mean"] == 0.9131
    assert "model_specs" not in artifact["frozen_reproduction_result"]
    for field in exp3692.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert all(step.startswith("OPERATOR-ACTION:") for step in artifact["operator_checklist"])
    assert any("0.9131 stays the headline" in step for step in artifact["operator_checklist"])


def test_req_publish_3692_validation_and_clean_report_edges(tmp_path: Path) -> None:
    """REQ-PUBLISH-3692: validation rejects poisoned or malformed artifacts."""

    assert exp3692._sha256_file(tmp_path / "missing") == "missing"
    assert exp3692.sanitize_cached_reproduction(
        ["safe", ("torch.cuda local marker",), {"target_model": "x"}]
    ) == ["safe", ["removed_for_verifier_scoring_substrate_hygiene"], {}]

    artifact = exp3692.build_artifact(
        repo_root=tmp_path,
        started_s=1.0,
        now_s=2.0,
        exp3680_artifact=_exp3680_fixture(),
        reproducer_importable=True,
        reproducer_extended=True,
        adversarial_verify_clean=True,
        frozen_reproduction_result=_frozen_result(),
        candidate_reproduction_result=_candidate_result(),
        publication_gate_before=_publication_gate(),
        publication_gate_after=_publication_gate(),
        north_star_hash_before="north",
        north_star_hash_after="north",
        ci_workflow_hash_before="workflow",
        ci_workflow_hash_after="workflow",
        github_run_triggered=False,
    )
    exp3692.validate_artifact(artifact)
    assert exp3692.adversarial_report_is_clean({"flags": []}) is True
    assert exp3692.adversarial_report_is_clean(
        {"flags": [{"kind": "DURATION_TOO_SHORT", "severity": "warn"}]}
    ) is False
    assert exp3692.adversarial_report_is_clean(
        {"flags": [{"kind": "TAUTOLOGY", "severity": "critical"}]}
    ) is False

    with pytest.raises(ValueError, match="missing required"):
        exp3692.validate_artifact({})
    with pytest.raises(ValueError, match="bare verifier"):
        exp3692.validate_artifact({**artifact, "inference_substrate": "bad"})
    with pytest.raises(ValueError, match="bare boolean"):
        exp3692.validate_artifact({**artifact, "adversarial_verify_clean": 1})
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3692.validate_artifact({**artifact, "honest_verdict": "complete: invented"})
    with pytest.raises(ValueError, match="field_principles"):
        exp3692.validate_artifact({**artifact, "field_principles": []})
    with pytest.raises(ValueError, match="missing field principles"):
        exp3692.validate_artifact({**artifact, "field_principles": {}})
    poisoned = {**artifact, "frozen_reproduction_result": _frozen_result(with_markers=True)}
    with pytest.raises(ValueError, match="compute-bound marker"):
        exp3692.validate_artifact(poisoned)
    with pytest.raises(ValueError, match="operator_checklist"):
        exp3692.validate_artifact({**artifact, "operator_checklist": "bad"})
    with pytest.raises(ValueError, match="OPERATOR-ACTION"):
        exp3692.validate_artifact({**artifact, "operator_checklist": ["do it"]})


def test_scenario_publish_3692_write_artifact_preserves_operator_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PUBLISH-3692: writing leaves north-star and CI YAML untouched."""

    (tmp_path / "ops").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    north_star = tmp_path / "ops" / "north-star.md"
    workflow = tmp_path / ".github" / "workflows" / "reproduce-fover-headline.yml"
    north_star.write_text("FoVer AUROC 0.9131\n", encoding="utf-8")
    workflow.write_text("run: python3 scripts/reproduce_fover_headline.py\n", encoding="utf-8")
    exp3680_path = tmp_path / exp3692.EXP3680_REL_PATH
    exp3680_path.parent.mkdir(parents=True)
    exp3680_path.write_text(json.dumps(_exp3680_fixture()), encoding="utf-8")

    fake_module = types.ModuleType("fake_reproducer")
    fake_source = tmp_path / "fake_reproducer.py"
    fake_source.write_text("--dependency-aware-candidate\n", encoding="utf-8")
    fake_module.__file__ = str(fake_source)
    fake_module.run_reproduction = lambda root: _frozen_result(with_markers=True)
    fake_module.run_dependency_aware_candidate_reproduction = lambda root: _candidate_result()
    fake_module.check_acceptance_ci = reproducer.check_acceptance_ci
    fake_module.check_dependency_aware_candidate_ci = reproducer.check_dependency_aware_candidate_ci
    fake_module.dependency_aware_candidate_bounds_from_artifact = (
        reproducer.dependency_aware_candidate_bounds_from_artifact
    )

    monkeypatch.setattr(exp3692.prep, "reproducer_import_status", lambda: (True, fake_module))
    monkeypatch.setattr(exp3692.prep, "evaluate_publication_gate", lambda root: _publication_gate())

    output = exp3692.write_artifact(tmp_path, started_s=0.0, now_s=3.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["honest_verdict"] == exp3692.READY_VERDICT
    assert saved["adversarial_verify_clean"] is True
    assert north_star.read_text(encoding="utf-8") == "FoVer AUROC 0.9131\n"
    assert workflow.read_text(encoding="utf-8") == "run: python3 scripts/reproduce_fover_headline.py\n"


def test_scenario_publish_3692_blocks_when_preconditions_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-PUBLISH-3692B: missing G1 candidate or reproducer blocks."""

    (tmp_path / "ops").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "ops" / "north-star.md").write_text("FoVer AUROC 0.9131\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "reproduce-fover-headline.yml").write_text(
        "run: python3 scripts/reproduce_fover_headline.py\n",
        encoding="utf-8",
    )
    exp3680_path = tmp_path / exp3692.EXP3680_REL_PATH
    exp3680_path.parent.mkdir(parents=True)
    exp3680_path.write_text(json.dumps(_exp3680_fixture(confirmed=False)), encoding="utf-8")

    monkeypatch.setattr(exp3692.prep, "reproducer_import_status", lambda: (False, None))
    monkeypatch.setattr(exp3692.prep, "evaluate_publication_gate", lambda root: _publication_gate())
    output = exp3692.write_artifact(tmp_path, output_path="results/blocked.json")
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["honest_verdict"] == exp3692.BLOCKED_VERDICT
    assert saved["frozen_reproduction_result"]["honest_verdict"] == "blocked_preconditions_not_met"

    result_path = tmp_path / "results" / "main.json"

    def fake_write(root: Path) -> Path:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"honest_verdict": exp3692.READY_VERDICT}), encoding="utf-8")
        return result_path

    monkeypatch.setattr(exp3692, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp3692, "write_artifact", fake_write)
    assert exp3692.main() == 0
    assert exp3692.READY_VERDICT in capsys.readouterr().out


def test_scenario_publish_3692_write_artifact_failure_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PUBLISH-3692: reproduction failures and clean-status retries fail closed."""

    (tmp_path / "ops").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "ops" / "north-star.md").write_text("FoVer AUROC 0.9131\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "reproduce-fover-headline.yml").write_text(
        "run: python3 scripts/reproduce_fover_headline.py\n",
        encoding="utf-8",
    )
    exp3680_path = tmp_path / exp3692.EXP3680_REL_PATH
    exp3680_path.parent.mkdir(parents=True)
    exp3680_path.write_text(json.dumps(_exp3680_fixture()), encoding="utf-8")

    fake_module = types.ModuleType("fake_reproducer")
    fake_source = tmp_path / "fake_reproducer.py"
    fake_source.write_text("--dependency-aware-candidate\n", encoding="utf-8")
    fake_module.__file__ = str(fake_source)
    fake_module.run_reproduction = lambda root: _frozen_result()
    fake_module.run_dependency_aware_candidate_reproduction = lambda root: _candidate_result()
    fake_module.check_acceptance_ci = reproducer.check_acceptance_ci
    fake_module.check_dependency_aware_candidate_ci = reproducer.check_dependency_aware_candidate_ci
    fake_module.dependency_aware_candidate_bounds_from_artifact = (
        reproducer.dependency_aware_candidate_bounds_from_artifact
    )

    monkeypatch.setattr(exp3692.prep, "reproducer_import_status", lambda: (True, fake_module))
    monkeypatch.setattr(exp3692.prep, "evaluate_publication_gate", lambda root: _publication_gate())
    monkeypatch.setattr(
        exp3692.prep,
        "run_frozen_reproduction",
        lambda module, root: (_ for _ in ()).throw(RuntimeError("frozen")),
    )
    monkeypatch.setattr(
        exp3692.prep,
        "run_candidate_reproduction",
        lambda module, root: (_ for _ in ()).throw(RuntimeError("candidate")),
    )
    output = exp3692.write_artifact(tmp_path, output_path="results/failures.json")
    failed = json.loads(output.read_text(encoding="utf-8"))
    assert failed["honest_verdict"] == exp3692.BLOCKED_VERDICT
    assert failed["frozen_reproduction_result"]["error"] == "RuntimeError: frozen"
    assert failed["candidate_reproduction_result"]["error"] == "RuntimeError: candidate"

    monkeypatch.setattr(exp3692.prep, "run_frozen_reproduction", lambda module, root: _frozen_result())
    monkeypatch.setattr(
        exp3692.prep,
        "run_candidate_reproduction",
        lambda module, root: _candidate_result(),
    )
    clean_sequence = iter([False, True, False])
    monkeypatch.setattr(exp3692, "verify_written_artifact_clean", lambda path: next(clean_sequence))
    retried_output = exp3692.write_artifact(tmp_path, output_path="results/retried.json")
    retried = json.loads(retried_output.read_text(encoding="utf-8"))
    assert retried["adversarial_verify_clean"] is True
