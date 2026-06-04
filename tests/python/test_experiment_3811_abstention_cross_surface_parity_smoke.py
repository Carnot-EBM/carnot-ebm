"""Tests for Exp 3811 abstention cross-surface parity smoke.

Spec: REQ-SPOE-3811, SCENARIO-SPOE-3811.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from carnot.reporting import abstention_cross_surface_parity_smoke_3811 as exp3811


ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / ".venv/bin/python"
EXP3771_PATH = ROOT / "results/experiment_3771_certified_abstention_operating_point.json"
SCRIPT_PATH = ROOT / "scripts/experiment_3811_abstention_cross_surface_parity_smoke.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("experiment_3811", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scenario_spoe_3811_real_surfaces_agree_on_confident_and_abstain() -> None:
    """SCENARIO-SPOE-3811: real verify API, CLI, and HTTP surfaces match."""

    candidates = exp3811.select_cached_fover_candidates(ROOT, n_candidates=10)
    comparison = exp3811.compare_surfaces(
        ROOT,
        candidates,
        executable=str(VENV_PYTHON),
        certified_threshold_path=EXP3771_PATH,
    )

    assert comparison["surfaces_compared"] == ["verify_api", "cli", "http_rest"]
    assert comparison["all_surfaces_agree"] is True
    assert comparison["mismatches"] == []
    assert comparison["n_candidates_compared"] == 10

    verdicts = {row["verdict"] for row in comparison["canonical_rows"]}
    assert {"confident", "abstain"}.issubset(verdicts)
    for row in comparison["canonical_rows"]:
        assert row["coverage"] == pytest.approx(0.998218)
        assert row["risk"] == pytest.approx(0.037646)
        assert row["delta"] == pytest.approx(0.05)
        assert row["threshold"] == pytest.approx(0.733216)


def test_req_spoe_3811_run_writes_required_complete_artifact(
    tmp_path: Path,
) -> None:
    """REQ-SPOE-3811: complete parity artifact records required bare fields."""

    output_path = tmp_path / "results/experiment_3811.json"
    candidates = [
        {
            "candidate_id": "cached-confident",
            "domain": "math",
            "text": "cached FoVer confident row",
            "confidence_error": 0.0,
            "ensemble_energy": 0.36,
            "expected_verdict": "confident",
        },
        {
            "candidate_id": "cached-abstain",
            "domain": "math",
            "text": "cached FoVer abstain row",
            "confidence_error": 0.0,
            "ensemble_energy": 0.65,
            "expected_verdict": "abstain",
        },
    ]
    comparison = {
        "surfaces_compared": ["verify_api", "cli", "http_rest"],
        "all_surfaces_agree": True,
        "n_candidates_compared": len(candidates),
        "mismatches": [],
        "canonical_rows": [
            {
                "candidate_id": "cached-confident",
                "verdict": "confident",
                "coverage": 0.998218,
                "risk": 0.037646,
                "delta": 0.05,
                "threshold": 0.733216,
            },
            {
                "candidate_id": "cached-abstain",
                "verdict": "abstain",
                "coverage": 0.998218,
                "risk": 0.037646,
                "delta": 0.05,
                "threshold": 0.733216,
            },
        ],
        "surface_rows": {},
    }

    artifact = exp3811.run(
        ROOT,
        output_path=output_path,
        executable=str(VENV_PYTHON),
        candidate_selector=lambda _root, _n_candidates: candidates,
        surface_runner=lambda _root, rows, **_kwargs: {
            **comparison,
            "n_candidates_compared": len(rows),
        },
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(exp3811.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"] == (
        "complete: abstention_cross_surface_parity_smoke_all_surfaces_agree_"
        "true_n2_verify_api_cli_http_rest_no_surface_drift"
    )
    assert artifact["inference_substrate"] == exp3811.INFERENCE_SUBSTRATE
    assert artifact["surfaces_compared"] == ["verify_api", "cli", "http_rest"]
    assert artifact["all_surfaces_agree"] is True
    assert artifact["n_candidates_compared"] == 2
    assert artifact["mismatches"] == []
    assert artifact["certified_threshold_used"] == pytest.approx(0.733216)
    assert artifact["tests_assert_real_behavior"] is True
    assert artifact["model_specs"]["verifiers"] == list(exp3811.SCORING_VERIFIERS)
    assert artifact["random_seed"] == exp3811.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 16


def test_req_spoe_3811_script_main_routes_to_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3811: requested script entrypoint delegates to the runner."""

    script = _load_script()
    output_path = tmp_path / "artifact.json"
    calls: dict[str, object] = {}

    def fake_run(root: Path, *, output_path: Path | None, executable: str | None) -> dict[str, object]:
        calls["root"] = root
        calls["output_path"] = output_path
        calls["executable"] = executable
        return {
            "honest_verdict": (
                "complete: abstention_cross_surface_parity_smoke_all_surfaces_agree_"
                "true_n10_verify_api_cli_http_rest_no_surface_drift"
            )
        }

    monkeypatch.setattr(script.exp3811, "run", fake_run)

    assert script.main(["--output", str(output_path), "--executable", "python"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["honest_verdict"].startswith("complete:")
    assert calls == {
        "root": ROOT,
        "output_path": output_path,
        "executable": "python",
    }


def test_req_spoe_3811_http_gate_blocks_without_parity_claims(
    tmp_path: Path,
) -> None:
    """REQ-SPOE-3811: absent Exp 3810 HTTP landing blocks gracefully."""

    output_path = tmp_path / "results/experiment_3811.json"
    exp3810_path = tmp_path / "results/experiment_3810.json"
    exp3810_path.parent.mkdir(parents=True, exist_ok=True)
    exp3810_path.write_text(json.dumps({"http_rest_surface_added": False}), encoding="utf-8")

    artifact = exp3811.run(
        ROOT,
        output_path=output_path,
        executable=str(VENV_PYTHON),
        exp3810_path=exp3810_path,
        candidate_selector=lambda _root, _n_candidates: (_ for _ in ()).throw(
            AssertionError("candidate selection must not run when HTTP gate blocks")
        ),
    )

    assert artifact["honest_verdict"] == "blocked_http_surface_not_landed"
    assert artifact["all_surfaces_agree"] is False
    assert artifact["n_candidates_compared"] == 0
    assert artifact["mismatches"] == []
    assert artifact["surfaces_compared"] == ["verify_api", "cli", "http_rest"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_spoe_3811_mismatch_reports_drift_details() -> None:
    """REQ-SPOE-3811: cross-surface mismatch is surfaced, not hidden."""

    surface_rows = {
        "verify_api": {
            "row-1": {
                "candidate_id": "row-1",
                "verdict": "confident",
                "coverage": 0.998218,
                "risk": 0.037646,
                "delta": 0.05,
                "threshold": 0.733216,
            }
        },
        "cli": {
            "row-1": {
                "candidate_id": "row-1",
                "verdict": "abstain",
                "coverage": 0.998218,
                "risk": 0.037646,
                "delta": 0.05,
                "threshold": 0.733216,
            }
        },
        "http_rest": {
            "row-1": {
                "candidate_id": "row-1",
                "verdict": "confident",
                "coverage": 0.998218,
                "risk": 0.037646,
                "delta": 0.05,
                "threshold": 0.733216,
            }
        },
    }

    comparison = exp3811.compare_normalized_surface_rows(surface_rows)

    assert comparison["all_surfaces_agree"] is False
    assert comparison["mismatches"] == [
        {
            "candidate_id": "row-1",
            "field": "verdict",
            "values_by_surface": {
                "verify_api": "confident",
                "cli": "abstain",
                "http_rest": "confident",
            },
        }
    ]


def test_req_spoe_3811_defensive_paths_and_blocker_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3811: blocked resources and drift branches stay explicit."""

    with pytest.raises(RuntimeError, match="CLI surface failed"):
        exp3811.run_cli_surface(ROOT, [], "/bin/false")

    monkeypatch.setattr(exp3811, "_post_json", lambda *_args, **_kwargs: (500, {"bad": True}))
    with pytest.raises(RuntimeError, match="HTTP surface failed"):
        exp3811.run_http_surface(ROOT, [], EXP3771_PATH)

    assert exp3811.normalize_http_response({"scores": [1]}) == {}
    with pytest.raises(ValueError, match="missing certified_abstention"):
        exp3811.normalize_verify_api_response({"scores": [{"candidate_id": "bad"}]})

    row = {
        "candidate_id": "row-1",
        "verdict": "confident",
        "coverage": 0.998218,
        "risk": 0.037646,
        "delta": 0.05,
        "threshold": 0.733216,
    }
    presence = exp3811.compare_normalized_surface_rows(
        {"verify_api": {"row-1": row}, "cli": {}, "http_rest": {"row-1": row}}
    )
    assert presence["mismatches"][0]["field"] == "presence"

    assert exp3811.first_blocker({"not-a-map": "ok"}) is None
    assert exp3811.first_blocker({"certified_threshold": {"available": False}}) == (
        "blocked_no_certified_threshold"
    )
    assert exp3811.first_blocker({"fover_corpus": {"available": False}}) == (
        "blocked_fover_corpus_missing"
    )
    assert exp3811.first_blocker({"package_import_and_surfaces": {"available": False}}) == (
        "blocked_surface_import"
    )
    assert exp3811.first_blocker({"upstream_exp3779_verify_api": {"available": False}}) == (
        "blocked_upstream_artifact_missing"
    )
    assert exp3811.first_blocker({"unexpected": {"available": False}}) == "blocked_unexpected"

    bad_fover = tmp_path / "bad_fover.json"
    bad_fover.write_text("{", encoding="utf-8")
    preconditions, config = exp3811.check_preconditions(
        ROOT,
        executable=str(tmp_path / "python"),
        certified_threshold_path=tmp_path / "missing_threshold.json",
        exp3779_path=tmp_path / "missing_3779.json",
        exp3789_path=tmp_path / "missing_3789.json",
        exp3810_path=tmp_path / "missing_3810.json",
        fover_corpus_path=bad_fover,
    )
    assert config is None
    assert preconditions["interpreter"]["available"] is False
    assert preconditions["certified_threshold"]["available"] is False
    assert preconditions["http_rest_surface"]["available"] is False
    assert preconditions["fover_corpus"]["available"] is False

    preconditions, _ = exp3811.check_preconditions(
        ROOT,
        executable=str(VENV_PYTHON),
        certified_threshold_path=EXP3771_PATH,
        exp3779_path=ROOT / "results/experiment_3779_abstention_operating_point_product_wiring.json",
        exp3789_path=ROOT / "results/experiment_3789_abstention_cli_batch_surface.json",
        exp3810_path=ROOT / "results/experiment_3810_abstention_http_rest_surface_v2.json",
        fover_corpus_path=tmp_path / "missing_fover.json",
    )
    assert preconditions["fover_corpus"]["available"] is False

    fover_path = tmp_path / "data/fover_corpus_v4.json"
    fover_path.parent.mkdir(parents=True, exist_ok=True)
    fover_path.write_text("[]", encoding="utf-8")
    all_ids = list(exp3811.FIXED_CONFIDENT_EXAMPLE_IDS + exp3811.FIXED_ABSTAIN_EXAMPLE_IDS)
    examples = [
        exp3811.spd.LabeledDetectorExample(
            domain="math",
            label=0,
            ensemble_energy=0.1,
            confidence_error=0.0,
            example_id=example_id,
        )
        for example_id in [*all_ids, "zz-extra"]
    ]
    monkeypatch.setattr(
        exp3811.spd,
        "load_cached_labeled_examples",
        lambda _root, **_kwargs: (examples, {"math": {"status": "loaded"}}),
    )
    assert exp3811.select_cached_fover_candidates(tmp_path, n_candidates=11)[-1][
        "candidate_id"
    ] == "zz-extra"

    monkeypatch.setattr(
        exp3811.spd,
        "load_cached_labeled_examples",
        lambda _root, **_kwargs: ([], {"math": {"status": "loaded"}}),
    )
    with pytest.raises(FileNotFoundError, match="cached FoVer examples missing"):
        exp3811.select_cached_fover_candidates(tmp_path)

    monkeypatch.setattr(
        exp3811.spd,
        "load_cached_labeled_examples",
        lambda _root, **_kwargs: ([], {"math": {"status": "missing"}}),
    )
    with pytest.raises(FileNotFoundError, match="fover_corpus_v4"):
        exp3811.select_cached_fover_candidates(tmp_path)

    monkeypatch.setattr(
        exp3811,
        "REQUIRED_ARTIFACT_FIELDS",
        (*exp3811.REQUIRED_ARTIFACT_FIELDS, "missing_field"),
    )
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp3811.build_artifact(
            verdict="blocked_test",
            duration_s=0.0,
            threshold_config=None,
            preconditions={},
            comparison=None,
            candidates=[],
            root=ROOT,
            output_path=tmp_path / "out.json",
            threshold_path=EXP3771_PATH,
            exp3779_path=ROOT / "results/experiment_3779_abstention_operating_point_product_wiring.json",
            exp3789_path=ROOT / "results/experiment_3789_abstention_cli_batch_surface.json",
            exp3810_path=ROOT / "results/experiment_3810_abstention_http_rest_surface_v2.json",
            fover_corpus_path=ROOT / "data/fover_corpus_v4.json",
        )
