"""Tests for Exp 3715 re-freeze disambiguation clean corrigendum.

Spec: REQ-PUBLISH-3715, SCENARIO-PUBLISH-3715.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import experiment_3715_refreeze_disambiguation_clean_corrigendum as exp3715


def _exp3704_fixture() -> dict[str, Any]:
    return {
        "artifact": "experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion",
        "schema": "carnot.refreeze_disambiguation.v1",
        "honest_verdict": (
            "complete: refreeze_disambiguated_no_candidate_beats_frozen_"
            "headline_stays_0_9131"
        ),
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": (
                    "external_comparator_auroc=0.928737 and "
                    "strongest_candidate_auroc=0.928737 agree"
                ),
            }
        ],
        "dependency_aware_auroc": 0.924869,
        "external_comparator_auroc": 0.928737,
        "fusion_auroc": 0.928462,
        "carnot_current_auroc": 0.91303,
        "frozen_headline_auroc": 0.9131,
        "strongest_candidate": "external",
        "strongest_candidate_auroc": 0.928737,
        "winner_vs_runnerup_delta_ci": {
            "point": 0.000275,
            "ci95": [-0.000044, 0.000625],
            "delong_p": 0.129713,
            "winner": "external",
            "comparison": "fusion",
        },
        "winner_vs_frozen_delta_ci": {
            "point": 0.015706,
            "ci95": [0.01225, 0.019332],
            "delong_p": 8.59387e-18,
            "winner": "external",
            "comparison": "frozen_0_9131_carnot_current_vector",
        },
        "candidate_ranking": [
            {"candidate": "external", "pooled_auroc": 0.928737},
            {"candidate": "fusion", "pooled_auroc": 0.928462},
            {"candidate": "dependency_aware", "pooled_auroc": 0.924869},
        ],
        "candidate_auroc_ci95": {
            "dependency_aware": {"ci95": [0.91699, 0.932891]},
            "external": {"ci95": [0.921543, 0.936042]},
            "fusion": {"ci95": [0.921115, 0.93591]},
        },
        "score_vector_checksums": {
            "dependency_aware": "dep",
            "external_comparator": "ext",
            "fusion": "fus",
            "carnot_current": "cur",
        },
        "n_seeds": 5,
        "n_examples": 1000,
        "n_pooled_examples": 5000,
        "random_seed": 3704,
        "random_seeds_used": [42, 137, 271, 314, 1729],
        "bootstrap_seeds": [42, 137, 271, 314, 1729],
        "n_bootstrap_per_seed": 200,
        "reproducibility_checksum": "exp3704-checksum",
        "duration_s": 67.315771,
        "north_star_unmodified_assert": True,
        "frozen_headline_unchanged_assert": True,
        "ci_workflow_unmodified_assert": True,
        "github_actions_run_triggered": False,
        "publication_gate_paper_ready_before": True,
        "publication_gate_paper_ready_after": True,
        "adversarial_verify_clean": False,
    }


def _gate_fixture(*, paper_ready: bool = True) -> dict[str, Any]:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {
                "pass": True,
                "detail": "FoVer dual-condition production AUROC 0.9131",
            }
        },
    }


@pytest.mark.parametrize(
    ("outcome", "exp3704_payload", "expected_verdict", "expected_clean"),
    [
        pytest.param(
            "corrigendum_clean_no_candidate_beats_frozen",
            _exp3704_fixture(),
            exp3715.SUCCESS_VERDICT,
            True,
            id="corrigendum_clean_no_candidate_beats_frozen",
        ),
        pytest.param(
            "blocked",
            None,
            exp3715.BLOCKED_VERDICT,
            False,
            id="blocked",
        ),
    ],
)
def test_honest_outcomes_are_parametrized_on_synthetic_fixtures(
    tmp_path: Path,
    outcome: str,
    exp3704_payload: dict[str, Any] | None,
    expected_verdict: str,
    expected_clean: bool,
) -> None:
    """REQ-PUBLISH-3715: clean and blocked outcomes are both explicit."""

    if outcome == "blocked":
        artifact = exp3715.blocked_artifact(
            started_s=1.0,
            now_s=1.25,
            exp3704_path=tmp_path / "missing.json",
        )
    else:
        source = tmp_path / "experiment_3704.json"
        source.write_text(json.dumps(exp3704_payload), encoding="utf-8")
        artifact = exp3715.build_corrigendum_artifact(
            exp3704=exp3704_payload,
            exp3704_path=source,
            exp3704_sha256="source-sha",
            north_star_hash_before="north",
            north_star_hash_after="north",
            ci_workflow_hash_before="workflow",
            ci_workflow_hash_after="workflow",
            publication_gate_before=_gate_fixture(),
            publication_gate_after=_gate_fixture(),
            adversarial_verify_clean=True,
            adversarial_verify_report={"flag_count": 0, "flags": []},
            started_s=1.0,
            now_s=1.25,
        )

    exp3715.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["adversarial_verify_clean"] is expected_clean
    assert type(artifact["no_candidate_beats_frozen"]) is bool

    if outcome != "blocked":
        assert artifact["no_candidate_beats_frozen"] is True
        assert artifact["strongest_candidate"] == "external"
        assert artifact["strongest_candidate_value_field"] == "external_comparator_auroc"
        assert "strongest_candidate_auroc" not in artifact
        assert artifact["external_comparator_auroc"] == 0.928737
        assert artifact["candidate_ranking"][0] == {
            "candidate": "external",
            "auroc_field": "external_comparator_auroc",
        }
        assert artifact["north_star_unmodified_assert"] is True
        assert artifact["ci_reproducer_not_triggered_assert"] is True
        assert artifact["frozen_headline_unchanged_assert"] is True
        assert artifact["acceptance_gate"]["passed"] is True


def test_validate_rejects_duplicate_auroc_alias_and_bad_pointers(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3715: schema validation enforces de-tautology."""

    source = tmp_path / "experiment_3704.json"
    payload = _exp3704_fixture()
    source.write_text(json.dumps(payload), encoding="utf-8")
    artifact = exp3715.build_corrigendum_artifact(
        exp3704=payload,
        exp3704_path=source,
        exp3704_sha256="source-sha",
        north_star_hash_before="north",
        north_star_hash_after="north",
        ci_workflow_hash_before="workflow",
        ci_workflow_hash_after="workflow",
        publication_gate_before=_gate_fixture(),
        publication_gate_after=_gate_fixture(),
        adversarial_verify_clean=True,
        adversarial_verify_report={"flag_count": 0, "flags": []},
        started_s=1.0,
        now_s=1.25,
    )

    invalid_cases = [
        ({k: v for k, v in artifact.items() if k != "fusion_auroc"}, "missing required"),
        ({**artifact, "strongest_candidate_auroc": 0.928737}, "strongest_candidate_auroc"),
        ({**artifact, "honest_verdict": "complete: unsupported"}, "unsupported honest_verdict"),
        ({**artifact, "strongest_candidate": "fusion"}, "source field"),
        ({**artifact, "strongest_candidate_value_field": "fusion_auroc"}, "source field"),
        (
            {**artifact, "candidate_ranking": [{"candidate": "external", "pooled_auroc": 0.928737}]},
            "candidate_ranking",
        ),
        ({**artifact, "no_candidate_beats_frozen": "true"}, "bare boolean"),
        ({**artifact, "no_candidate_beats_frozen": False}, "no_candidate_beats_frozen"),
        ({**artifact, "dependency_aware_auroc": 1.2}, "dependency_aware_auroc"),
        ({**artifact, "fusion_auroc": artifact["external_comparator_auroc"]}, "duplicate"),
        ({**artifact, "carnot_current_auroc": 0.9}, "carnot_current_auroc"),
        ({**artifact, "frozen_headline_auroc": 0.9}, "frozen_headline_auroc"),
        (
            {
                **artifact,
                "strongest_candidate": "dependency_aware",
                "strongest_candidate_value_field": "dependency_aware_auroc",
            },
            "strongest_candidate does not match",
        ),
        (
            {
                **artifact,
                "strongest_candidate": "bogus",
                "strongest_candidate_value_field": "dependency_aware_auroc",
            },
            "strongest_candidate must",
        ),
        ({**artifact, "field_principles": {"honest_verdict": "x"}}, "missing field principles"),
        ({**artifact, "field_principles": None}, "field_principles"),
        ({**artifact, "inference_substrate": "live_llm_inference"}, "inference_substrate"),
        ({**artifact, "correction_note": "missing audit detail"}, "correction_note"),
        ({**artifact, "acceptance_gate": {"passed": True}}, "acceptance_gate"),
        (
            {
                **artifact,
                "publication_gate_paper_ready_after": False,
                "frozen_headline_unchanged_assert": True,
            },
            "frozen_headline_unchanged_assert",
        ),
        ({**artifact, "ci_reproducer_not_triggered_assert": False}, "ci_reproducer"),
        (
            {
                **artifact,
                "acceptance_gate": {**artifact["acceptance_gate"], "passed": False},
            },
            "acceptance_gate passed",
        ),
        (
            {
                **artifact,
                "candidate_ranking": [
                    "external",
                    {"candidate": "fusion", "auroc_field": "fusion_auroc"},
                    {
                        "candidate": "dependency_aware",
                        "auroc_field": "dependency_aware_auroc",
                    },
                ],
            },
            "rows must be objects",
        ),
        (
            {
                **artifact,
                "candidate_ranking": [
                    {"candidate": "external", "auroc_field": "external_comparator_auroc", "x": "y"},
                    {"candidate": "fusion", "auroc_field": "fusion_auroc"},
                    {
                        "candidate": "dependency_aware",
                        "auroc_field": "dependency_aware_auroc",
                    },
                ],
            },
            "must not duplicate",
        ),
        (
            {
                **artifact,
                "candidate_ranking": [
                    {"candidate": "external", "auroc_field": "fusion_auroc"},
                    {"candidate": "fusion", "auroc_field": "fusion_auroc"},
                    {
                        "candidate": "dependency_aware",
                        "auroc_field": "dependency_aware_auroc",
                    },
                ],
            },
            "source field mismatch",
        ),
        (
            {
                **artifact,
                "candidate_ranking": [
                    {"candidate": "external", "auroc_field": "external_comparator_auroc"},
                    {"candidate": "external", "auroc_field": "external_comparator_auroc"},
                    {
                        "candidate": "dependency_aware",
                        "auroc_field": "dependency_aware_auroc",
                    },
                ],
            },
            "cover every candidate",
        ),
    ]
    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            exp3715.validate_artifact(bad_artifact)

    with pytest.raises(ValueError, match="unsupported"):
        exp3715.build_corrigendum_artifact(
            exp3704={**payload, "strongest_candidate": "bogus"},
            exp3704_path=source,
            exp3704_sha256="source-sha",
            north_star_hash_before="north",
            north_star_hash_after="north",
            ci_workflow_hash_before="workflow",
            ci_workflow_hash_after="workflow",
            publication_gate_before=_gate_fixture(),
            publication_gate_after=_gate_fixture(),
            adversarial_verify_clean=True,
            adversarial_verify_report={"flag_count": 0, "flags": []},
            started_s=1.0,
            now_s=1.25,
        )


def test_write_artifact_preserves_operator_files_and_finalizes_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-PUBLISH-3715: write path records non-edits and verifier cleanliness."""

    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    north_star = tmp_path / "ops" / "north-star.md"
    workflow = tmp_path / ".github" / "workflows" / "reproduce-fover-headline.yml"
    north_star.write_text("FoVer AUROC 0.9131\n", encoding="utf-8")
    workflow.write_text("assert 0.9131\n", encoding="utf-8")
    exp3704 = tmp_path / exp3715.EXP3704_REL_PATH
    exp3704.write_text(json.dumps(_exp3704_fixture()), encoding="utf-8")

    monkeypatch.setattr(exp3715, "evaluate_publication_gate", lambda repo_root: _gate_fixture())
    monkeypatch.setattr(
        exp3715,
        "run_adversarial_verify_report",
        lambda path: {"flag_count": 0, "max_severity": None, "flags": []},
    )

    output = exp3715.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["adversarial_verify_clean"] is True
    assert artifact["no_candidate_beats_frozen"] is True
    assert north_star.read_text(encoding="utf-8") == "FoVer AUROC 0.9131\n"
    assert workflow.read_text(encoding="utf-8") == "assert 0.9131\n"

    monkeypatch.setattr(exp3715, "REPO_ROOT", tmp_path)
    assert exp3715.main() == 0
    assert exp3715.SUCCESS_VERDICT in capsys.readouterr().out


def test_write_artifact_blocks_when_exp3704_is_unavailable(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3715: missing Exp 3704 fails closed."""

    output = exp3715.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp3715.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3715.BLOCKED_VERDICT
    assert artifact["adversarial_verify_clean"] is False
    assert artifact["no_candidate_beats_frozen"] is False


def test_import_glue_and_fallback_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PUBLISH-3715: IO helpers are deterministic and fail closed."""

    results = tmp_path / "results"
    results.mkdir()
    fallback = results / "experiment_3704_alt.json"
    fallback.write_text("{}", encoding="utf-8")
    assert exp3715.find_exp3704_artifact(tmp_path) == fallback
    fallback.unlink()
    assert exp3715.find_exp3704_artifact(tmp_path) is None

    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    publication_gate = script_dir / "publication_gate.py"
    publication_gate.write_text(
        "def evaluate():\n"
        "    return {'paper_ready': True, 'gates': {'G1': {'source': 'fixture'}}}\n",
        encoding="utf-8",
    )
    assert exp3715.evaluate_publication_gate(tmp_path)["paper_ready"] is True
    monkeypatch.setattr(exp3715.importlib.util, "spec_from_file_location", lambda name, path: None)
    assert exp3715.evaluate_publication_gate(tmp_path)["paper_ready"] is None
    monkeypatch.undo()

    adversarial = script_dir / "adversarial_verify.py"
    adversarial.write_text(
        "def verify_artifact(path):\n"
        "    return {'flag_count': 0, 'flags': [], 'path': str(path)}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exp3715, "REPO_ROOT", tmp_path)
    assert exp3715.run_adversarial_verify_report(tmp_path / "artifact.json")["flag_count"] == 0
    monkeypatch.setattr(exp3715.importlib.util, "spec_from_file_location", lambda name, path: None)
    with pytest.raises(RuntimeError, match="could not import"):
        exp3715.run_adversarial_verify_report(tmp_path / "artifact.json")
    monkeypatch.undo()

    assert exp3715.adversarial_report_is_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp3715._publication_gate_source({}) is None
    assert exp3715._sha256_file(tmp_path / "missing") == "missing"
    assert exp3715._clean_delta_ci(None) is None
    assert exp3715._candidate_ci_bounds({}) == {}
    with pytest.raises(ValueError, match="finite number"):
        exp3715._finite_float("not-a-number")

    sparse = _exp3704_fixture()
    sparse.pop("candidate_ranking")
    sparse.pop("candidate_auroc_ci95")
    sparse.pop("winner_vs_frozen_delta_ci")
    source = tmp_path / "experiment_3704_sparse.json"
    source.write_text(json.dumps(sparse), encoding="utf-8")
    artifact = exp3715.build_corrigendum_artifact(
        exp3704=sparse,
        exp3704_path=source,
        exp3704_sha256="source-sha",
        north_star_hash_before="north",
        north_star_hash_after="north",
        ci_workflow_hash_before="workflow",
        ci_workflow_hash_after="workflow",
        publication_gate_before=_gate_fixture(),
        publication_gate_after=_gate_fixture(),
        adversarial_verify_clean=True,
        adversarial_verify_report={"flag_count": 0, "flags": []},
        started_s=1.0,
        now_s=1.25,
    )
    assert artifact["candidate_ranking"][0]["candidate"] == "external"
    assert artifact["candidate_auroc_ci95_bounds_from_exp3704"] == {}
    assert artifact["paired_delta_evidence_from_exp3704"]["winner_vs_frozen_delta_ci"] is None
