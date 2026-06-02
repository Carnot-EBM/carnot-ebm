"""Tests for Exp 3704 re-freeze candidate disambiguation.

Spec: REQ-PUBLISH-3704, SCENARIO-PUBLISH-3704.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion as exp3704


def _score_shift(labels: np.ndarray, shift: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=len(labels))
    return noise + labels.astype(float) * shift


def _panel(seed: int, *, shifts: dict[str, float]) -> exp3704.CandidateScorePanel:
    labels = np.asarray([0] * 500 + [1] * 500, dtype=np.int64)
    return exp3704.CandidateScorePanel(
        seed=seed,
        labels=labels,
        dependency_scores=_score_shift(labels, shifts["dependency"], seed=seed),
        external_scores=_score_shift(labels, shifts["external"], seed=seed + 10_000),
        fusion_scores=_score_shift(labels, shifts["fusion"], seed=seed + 20_000),
        carnot_current_scores=_score_shift(labels, shifts["carnot"], seed=seed + 30_000),
        dependency_architecture_scores=_score_shift(labels, shifts["dependency_arch"], seed=seed + 40_000),
        external_architecture_scores=_score_shift(labels, shifts["external_arch"], seed=seed + 50_000),
        fusion_architecture_scores=_score_shift(labels, shifts["fusion_arch"], seed=seed + 60_000),
        subset_sha256=f"subset-{seed}",
    )


def _artifact_kwargs(tmp_path: Path, panels: list[exp3704.CandidateScorePanel]) -> dict[str, Any]:
    return {
        "repo_root": tmp_path,
        "panels": panels,
        "started_s": 1.0,
        "now_s": 4.0,
        "preconditions": [{"resource": "synthetic_fixture", "available": True, "detail": "ok"}],
        "publication_gate_before": {
            "paper_ready": False,
            "gates": {"G1": {"detail": "FoVer AUROC 0.9131"}},
        },
        "publication_gate_after": {
            "paper_ready": False,
            "gates": {"G1": {"detail": "FoVer AUROC 0.9131"}},
        },
        "north_star_hash_before": "north",
        "north_star_hash_after": "north",
        "ci_workflow_hash_before": "workflow",
        "ci_workflow_hash_after": "workflow",
        "github_run_triggered": False,
        "adversarial_verify_clean": True,
        "random_seed": 3704,
        "bootstrap_seeds": [42, 137],
        "n_bootstrap": 8,
    }


@pytest.mark.parametrize(
    ("scenario", "shifts", "expected_refreeze", "expected_verdict"),
    [
        pytest.param(
            "winner_beats_frozen_package_reemitted",
            {
                "dependency": 1.72,
                "external": 1.95,
                "fusion": 2.16,
                "carnot": 1.70,
                "dependency_arch": 1.20,
                "external_arch": 1.25,
                "fusion_arch": 1.30,
            },
            True,
            exp3704.SUCCESS_TEMPLATE,
            id="winner_beats_frozen_package_reemitted",
        ),
        pytest.param(
            "no_candidate_beats_frozen",
            {
                "dependency": 1.15,
                "external": 1.25,
                "fusion": 1.35,
                "carnot": 1.10,
                "dependency_arch": 1.00,
                "external_arch": 1.00,
                "fusion_arch": 1.00,
            },
            False,
            exp3704.NO_CANDIDATE_VERDICT,
            id="no_candidate_beats_frozen",
        ),
        pytest.param(
            "blocked",
            {},
            False,
            exp3704.BLOCKED_VERDICT,
            id="blocked",
        ),
    ],
)
def test_honest_outcomes_are_parametrized(
    tmp_path: Path,
    scenario: str,
    shifts: dict[str, float],
    expected_refreeze: bool,
    expected_verdict: str,
) -> None:
    """REQ-PUBLISH-3704: synthetic outcomes cover winner/no-win/blocked."""

    if scenario == "blocked":
        artifact = exp3704.blocked_artifact(
            started_s=1.0,
            now_s=2.0,
            random_seed=3704,
            preconditions=[{"resource": "synthetic_fixture", "available": False, "detail": "missing"}],
        )
    else:
        panels = [_panel(seed, shifts=shifts) for seed in [42, 137, 271, 314, 1729]]
        artifact = exp3704.build_artifact_from_panels(**_artifact_kwargs(tmp_path, panels))

    exp3704.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(expected_verdict)
    assert artifact["refreeze_package_reemitted_for_winner"] is expected_refreeze
    assert artifact["strongest_candidate_beats_frozen"] is expected_refreeze
    assert artifact["adversarial_verify_clean"] is (scenario != "blocked")
    assert artifact["north_star_unmodified_assert"] is (scenario != "blocked")
    assert artifact["frozen_headline_unchanged_assert"] is (scenario != "blocked")
    assert type(artifact["strongest_candidate_beats_frozen"]) is bool

    if scenario != "blocked":
        assert artifact["dependency_aware_auroc"] != artifact["external_comparator_auroc"]
        assert artifact["external_comparator_auroc"] != artifact["fusion_auroc"]
        assert artifact["winner_vs_runnerup_delta_ci"]["ci95"][0] <= artifact["winner_vs_runnerup_delta_ci"]["point"]
        assert artifact["winner_vs_runnerup_delta_ci"]["delong_p"] <= 1.0

    if expected_refreeze:
        assert artifact["strongest_candidate"] in {"dependency_aware", "external", "fusion"}
        assert all(step.startswith("OPERATOR-ACTION:") for step in artifact["operator_checklist"])
        assert any("0.9131 stays the headline" in step for step in artifact["operator_checklist"])
    else:
        assert artifact["operator_checklist"] == []


def test_fusion_crossfit_scores_are_distinct_and_weighted() -> None:
    """SCENARIO-PUBLISH-3704: fusion composes dependency and external signals."""

    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    dependency = np.asarray([0.05, 0.15, 0.45, 0.55, 0.80, 0.90])
    external = np.asarray([0.10, 0.20, 0.30, 0.60, 0.70, 0.95])

    result = exp3704.fusion_crossfit_scores(
        labels=labels,
        dependency_scores=dependency,
        external_scores=external,
        random_seed=7,
        n_folds=3,
    )

    assert result.scores.shape == dependency.shape
    assert np.isfinite(result.scores).all()
    assert not np.allclose(result.scores, dependency)
    assert not np.allclose(result.scores, external)
    assert 0.05 <= result.mean_alpha <= 0.95
    assert len(result.fold_alphas) == 3


def test_condition_rows_are_converted_to_candidate_panels(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PUBLISH-3704: condition rows produce all three candidate vectors."""

    labels = [0, 0, 1, 1]
    row = exp3704.exp3680.ConditionScoreRows(
        seed=42,
        labels=labels,
        production_scores_by_verifier={
            name: [0.1, 0.2, 0.7, 0.8] for name in exp3704.exp3644.VERIFIER_NAMES
        },
        architecture_scores_by_verifier={
            name: [0.2, 0.3, 0.6, 0.7] for name in exp3704.exp3644.VERIFIER_NAMES
        },
        subset_sha256="subset",
    )

    def fake_score_weighting_panel(**kwargs: Any) -> dict[str, np.ndarray]:
        matrix = np.asarray(kwargs["score_matrix"], dtype=float)
        base = matrix[:, 0]
        return {
            "dependency_aware_proper": base + 0.03,
            "carnot_current": base,
            "weaver_style": base - 0.01,
            "unweighted": base - 0.02,
        }

    class FakeExternal:
        scores = np.asarray([0.11, 0.19, 0.72, 0.81])

    class FakeFusion:
        scores = np.asarray([0.10, 0.18, 0.74, 0.84])
        mean_alpha = 0.55
        fold_alphas = [0.55, 0.55]

    monkeypatch.setattr(exp3704.exp3667, "score_weighting_panel", fake_score_weighting_panel)
    monkeypatch.setattr(
        exp3704.exp3693,
        "cig_deentangled_crossfit_scores",
        lambda **kwargs: FakeExternal(),
    )
    monkeypatch.setattr(exp3704, "fusion_crossfit_scores", lambda **kwargs: FakeFusion())

    panel = exp3704.panel_from_condition_row(row)

    assert panel.seed == 42
    assert panel.subset_sha256 == "subset"
    assert panel.fusion_alpha_mean == 0.55
    assert panel.dependency_scores[2] > panel.carnot_current_scores[2]


def test_validate_artifact_rejects_copy_bugs_and_mutations(tmp_path: Path) -> None:
    """REQ-PUBLISH-3704: schema validation enforces de-tautology and non-edits."""

    panels = [
        _panel(
            seed,
            shifts={
                "dependency": 1.72,
                "external": 1.95,
                "fusion": 2.16,
                "carnot": 1.70,
                "dependency_arch": 1.20,
                "external_arch": 1.25,
                "fusion_arch": 1.30,
            },
        )
        for seed in [42, 137, 271, 314, 1729]
    ]
    artifact = exp3704.build_artifact_from_panels(**_artifact_kwargs(tmp_path, panels))

    with pytest.raises(ValueError, match="missing required"):
        exp3704.validate_artifact({k: v for k, v in artifact.items() if k != "fusion_auroc"})
    with pytest.raises(ValueError, match="bare boolean"):
        exp3704.validate_artifact({**artifact, "strongest_candidate_beats_frozen": "true"})
    with pytest.raises(ValueError, match="distinct AUROC"):
        bad_copy = {
            **artifact,
            "fusion_auroc": artifact["external_comparator_auroc"],
            "score_vector_checksums": {
                **artifact["score_vector_checksums"],
                "fusion": artifact["score_vector_checksums"]["external_comparator"],
            },
        }
        bad_copy["strongest_candidate_auroc"] = {
            **artifact["strongest_candidate_auroc"],
            "value": bad_copy[artifact["strongest_candidate_auroc"]["source_field"]],
        }
        exp3704.validate_artifact(
            bad_copy
        )
    with pytest.raises(ValueError, match="operator_checklist"):
        exp3704.validate_artifact({**artifact, "operator_checklist": ["edit the file"]})
    with pytest.raises(ValueError, match="north_star_unmodified_assert"):
        exp3704.validate_artifact({**artifact, "north_star_unmodified_assert": False})


def test_write_artifact_preserves_operator_files_and_main_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-PUBLISH-3704: write path records non-edits and main prints verdict."""

    (tmp_path / "ops").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "ops" / "north-star.md").write_text("FoVer AUROC 0.9131\n", encoding="utf-8")
    workflow = tmp_path / ".github" / "workflows" / "reproduce-fover-headline.yml"
    workflow.write_text("assert: 0.9131\n", encoding="utf-8")

    panels = [
        _panel(
            seed,
            shifts={
                "dependency": 1.72,
                "external": 1.95,
                "fusion": 2.16,
                "carnot": 1.70,
                "dependency_arch": 1.20,
                "external_arch": 1.25,
                "fusion_arch": 1.30,
            },
        )
        for seed in [42, 137, 271, 314, 1729]
    ]
    monkeypatch.setattr(exp3704, "score_candidate_panels", lambda root: panels)
    monkeypatch.setattr(
        exp3704,
        "probe_preconditions",
        lambda root: [{"resource": "synthetic_fixture", "available": True, "detail": "ok"}],
    )
    monkeypatch.setattr(
        exp3704,
        "evaluate_publication_gate",
        lambda root: {"paper_ready": False, "gates": {"G1": {"detail": "FoVer AUROC 0.9131"}}},
    )
    monkeypatch.setattr(
        exp3704,
        "run_adversarial_verify_report",
        lambda path: {"flag_count": 0, "max_severity": None, "flags": []},
    )
    monkeypatch.setattr(
        exp3704.exp3667,
        "bootstrap_auroc_ci",
        lambda labels, scores, *, seeds, n_bootstrap: {
            "point": round(exp3704.exp3644.tie_aware_auroc(labels, scores), 6),
            "ci95": [0.91, 0.96],
            "bootstrap_seeds": list(seeds),
            "n_bootstrap_per_seed": n_bootstrap,
        },
    )
    monkeypatch.setattr(
        exp3704,
        "_paired_delta_with_delong",
        lambda labels, first_scores, second_scores, *, seeds, n_bootstrap, first, second: {
            "point": 0.02,
            "ci95": [0.01, 0.03],
            "bootstrap_seeds": list(seeds),
            "n_bootstrap_per_seed": n_bootstrap,
            "winner": first,
            "comparison": second,
            "delong_p": 0.01,
            "delong": {"p_value": 0.01},
        },
    )

    output = exp3704.write_artifact(tmp_path)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["adversarial_verify_clean"] is True
    assert artifact["north_star_unmodified_assert"] is True
    assert artifact["ci_workflow_unmodified_assert"] is True
    assert artifact["github_actions_run_triggered"] is False
    assert (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8") == "FoVer AUROC 0.9131\n"
    assert workflow.read_text(encoding="utf-8") == "assert: 0.9131\n"

    monkeypatch.setattr(exp3704, "REPO_ROOT", tmp_path)
    assert exp3704.main() == 0
    assert "complete: refreeze_disambiguated_winner_" in capsys.readouterr().out


def test_io_glue_branches_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PUBLISH-3704: IO wrappers cover blocked and import branches."""

    monkeypatch.setattr(
        exp3704.exp3693,
        "probe_preconditions",
        lambda root, n_examples: [{"resource": "base", "available": True, "detail": "ok"}],
    )
    assert exp3704.probe_preconditions(tmp_path)[-1]["available"] is True

    monkeypatch.setattr(exp3704.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("x")))
    assert exp3704.probe_preconditions(tmp_path)[-1]["available"] is False

    monkeypatch.setattr(exp3704.exp3680, "discover_fr11_state_files", lambda root: ["state"])

    class FakeRow:
        pass

    monkeypatch.setattr(
        exp3704.exp3680,
        "score_dual_condition_rows",
        lambda root, seed, n_examples, state_files: FakeRow(),
    )
    monkeypatch.setattr(
        exp3704,
        "panel_from_condition_row",
        lambda row: _panel(
            42,
            shifts={
                "dependency": 1.72,
                "external": 1.95,
                "fusion": 2.16,
                "carnot": 1.70,
                "dependency_arch": 1.20,
                "external_arch": 1.25,
                "fusion_arch": 1.30,
            },
        ),
    )
    assert len(exp3704.score_candidate_panels(tmp_path)) == len(exp3704.DEFAULT_RANDOM_SEEDS)

    pub = tmp_path / "scripts" / "publication_gate.py"
    pub.parent.mkdir()
    pub.write_text("def evaluate():\n    return {'paper_ready': False}\n", encoding="utf-8")
    assert exp3704.evaluate_publication_gate(tmp_path)["paper_ready"] is False
    monkeypatch.setattr(exp3704.importlib.util, "spec_from_file_location", lambda name, path: None)
    assert exp3704.evaluate_publication_gate(tmp_path)["paper_ready"] is None

    adv = tmp_path / "scripts" / "adversarial_verify.py"
    adv.write_text(
        "def verify_artifact(path):\n    return {'flag_count': 0, 'flags': []}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exp3704, "REPO_ROOT", tmp_path)
    monkeypatch.undo()
    monkeypatch.setattr(exp3704, "REPO_ROOT", tmp_path)
    assert exp3704.run_adversarial_verify_report(tmp_path / "artifact.json")["flag_count"] == 0
    monkeypatch.setattr(exp3704.importlib.util, "spec_from_file_location", lambda name, path: None)
    with pytest.raises(RuntimeError, match="could not import"):
        exp3704.run_adversarial_verify_report(tmp_path / "artifact.json")
    assert exp3704.adversarial_report_is_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp3704.adversarial_report_is_clean({"flags": [{"kind": "TAUTOLOGY"}]}) is False


def test_write_artifact_blocked_and_scoring_exception_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PUBLISH-3704: write_artifact fails closed before scoring."""

    (tmp_path / "ops").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "ops" / "north-star.md").write_text("FoVer AUROC 0.9131\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "reproduce-fover-headline.yml").write_text(
        "assert: 0.9131\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        exp3704,
        "evaluate_publication_gate",
        lambda root: {"paper_ready": False, "gates": {"G1": {"detail": "FoVer AUROC 0.9131"}}},
    )
    monkeypatch.setattr(
        exp3704,
        "probe_preconditions",
        lambda root: [{"resource": "synthetic_fixture", "available": False, "detail": "missing"}],
    )

    output = exp3704.write_artifact(tmp_path)
    assert json.loads(output.read_text(encoding="utf-8"))["honest_verdict"] == exp3704.BLOCKED_VERDICT

    monkeypatch.setattr(
        exp3704,
        "probe_preconditions",
        lambda root: [{"resource": "synthetic_fixture", "available": True, "detail": "ok"}],
    )
    monkeypatch.setattr(
        exp3704,
        "score_candidate_panels",
        lambda root: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    output = exp3704.write_artifact(tmp_path)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == exp3704.BLOCKED_VERDICT
    assert artifact["preconditions_checked"][-1]["resource"] == "candidate_panel_scoring"


def test_validation_defensive_branches(tmp_path: Path) -> None:
    """REQ-PUBLISH-3704: validators reject malformed synthetic artifacts."""

    panels = [
        _panel(
            seed,
            shifts={
                "dependency": 1.72,
                "external": 1.95,
                "fusion": 2.16,
                "carnot": 1.70,
                "dependency_arch": 1.20,
                "external_arch": 1.25,
                "fusion_arch": 1.30,
            },
        )
        for seed in [42, 137, 271, 314, 1729]
    ]
    artifact = exp3704.build_artifact_from_panels(**_artifact_kwargs(tmp_path, panels))

    invalid_cases = [
        ({**artifact, "honest_verdict": "complete: bad"}, "unsupported"),
        ({**artifact, "inference_substrate": "live_llm_inference"}, "inference_substrate"),
        ({**artifact, "field_principles": None}, "field_principles"),
        ({**artifact, "field_principles": {"honest_verdict": "x"}}, "missing field principles"),
        ({**artifact, "operator_checklist": "bad"}, "operator_checklist"),
        ({**artifact, "n_seeds": 4}, "n_seeds"),
        ({**artifact, "n_examples": 999}, "n_examples"),
        ({**artifact, "dependency_aware_auroc": 1.0}, "leak guard"),
        ({**artifact, "fusion_auroc": 1.2}, "fusion_auroc"),
        ({**artifact, "winner_vs_runnerup_delta_ci": None}, "winner_vs_runnerup_delta_ci"),
        (
            {**artifact, "winner_vs_runnerup_delta_ci": {"point": 0.1, "ci95": [0.2, 0.3], "delong_p": 0.1}},
            "contain its point",
        ),
        (
            {**artifact, "winner_vs_runnerup_delta_ci": {"point": 0.1, "ci95": ["x", 0.3], "delong_p": 0.1}},
            "bounds",
        ),
        (
            {**artifact, "winner_vs_runnerup_delta_ci": {"point": 0.1, "ci95": [0.0], "delong_p": 0.1}},
            "include point",
        ),
        (
            {**artifact, "winner_vs_runnerup_delta_ci": {"point": 0.1, "ci95": [0.0, 0.2], "delong_p": 2.0}},
            "DeLong",
        ),
    ]
    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            exp3704.validate_artifact(bad_artifact)

    with pytest.raises(ValueError, match="operator_checklist"):
        exp3704.validate_artifact(
            {
                **artifact,
                "refreeze_package_reemitted_for_winner": True,
                "operator_checklist": [],
            }
        )
    with pytest.raises(ValueError, match="frozen_headline_unmodified|frozen_headline_unchanged"):
        exp3704.validate_artifact(
            {
                **artifact,
                "refreeze_package_reemitted_for_winner": True,
                "frozen_headline_unchanged_assert": False,
                "operator_checklist": exp3704.build_operator_checklist("fusion", artifact_path=Path("x.json")),
            }
        )
    with pytest.raises(ValueError, match="OPERATOR-ACTION"):
        exp3704.validate_artifact(
            {
                **artifact,
                "refreeze_package_reemitted_for_winner": False,
                "operator_checklist": exp3704.build_operator_checklist("fusion", artifact_path=Path("x.json")),
            }
        )
    with pytest.raises(ValueError, match="adversarial_verify_clean"):
        exp3704.validate_artifact(
            {
                **artifact,
                "refreeze_package_reemitted_for_winner": True,
                "adversarial_verify_clean": False,
                "operator_checklist": exp3704.build_operator_checklist("fusion", artifact_path=Path("x.json")),
            }
        )
    with pytest.raises(ValueError, match="score vectors"):
        exp3704.validate_artifact(
            {
                **artifact,
                "score_vector_checksums": {
                    **artifact["score_vector_checksums"],
                    "fusion": artifact["score_vector_checksums"]["external_comparator"],
                },
            }
        )


def test_low_level_error_helpers() -> None:
    """REQ-PUBLISH-3704: low-level helpers fail closed on malformed inputs."""

    labels = np.asarray([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="same length"):
        exp3704._assert_score_lengths(labels, {"bad": np.asarray([0.1])})
    with pytest.raises(ValueError, match="finite"):
        exp3704._assert_score_lengths(labels, {"bad": np.asarray([0.1, float("nan")])})
    with pytest.raises(ValueError, match="binary"):
        exp3704._require_binary_labels(np.asarray([0, 0, 0]))
    assert np.all(exp3704._minmax01(np.asarray([3.0, 3.0])) == 0.5)
    assert exp3704._round_metric(None) is None
    assert exp3704._sha256_file(Path("/tmp/definitely_missing_exp3704_file")) == "missing"
    with pytest.raises(ValueError, match="candidate panel"):
        exp3704.build_artifact_from_panels(
            repo_root=Path("/tmp"),
            panels=[],
            started_s=1.0,
            now_s=2.0,
            preconditions=[],
            publication_gate_before={"paper_ready": False},
            publication_gate_after={"paper_ready": False},
            north_star_hash_before="a",
            north_star_hash_after="a",
            ci_workflow_hash_before="b",
            ci_workflow_hash_after="b",
            github_run_triggered=False,
            adversarial_verify_clean=True,
        )
