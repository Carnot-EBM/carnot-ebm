"""Tests for Exp 3778 FR-11 continuous self-learning v18.

Spec: REQ-LEARN-3778, SCENARIO-LEARN-3778.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.fr11 import continuous_self_learning_v18 as exp3778
from carnot.fr11.continuous_self_learning_v18 import (
    BLOCKED_CORPUS_VERDICT,
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    SUCCESS_VERDICT,
    apply_consolidated_deltas,
    build_artifact,
    build_artifact_from_scores,
    consolidate_domain_deltas,
    validate_artifact,
    write_artifact,
)
from carnot.fr11.tier2_memory import Tier2ThresholdMemory


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_NAMES = exp3778.VERIFIER_NAMES


def _score_fixture() -> tuple[list[int], dict[str, list[float]]]:
    labels = [1] * 32 + [0] * 32
    return labels, {
        "fr11_session_memory": [0.82] * 24 + [0.61] * 8 + [0.21] * 24 + [0.42] * 8,
        "tier0r_curry_howard": [0.91] * 32 + [0.11] * 32,
        "tier0s_arithmetic_gap": [0.72] * 20 + [0.41] * 12 + [0.31] * 20 + [0.62] * 12,
        "tier0u_logical_consistency": [0.56] * 32 + [0.44] * 32,
    }


def test_req_learn_3778_spec_declares_tier2_constraint_memory_contract() -> None:
    """REQ-LEARN-3778: OpenSpec declares the v18 Tier-2 contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3778" in spec
    assert "SCENARIO-LEARN-3778" in spec
    assert "Tier2ThresholdMemory.update_domain_delta" in spec
    assert "no corpus to consolidate from" in spec


def test_scenario_learn_3778_persisted_deltas_round_trip_and_anti_poison(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3778: persisted deltas are the applied Tier-2 memory."""

    labels, scores_by_verifier = _score_fixture()
    db_path = tmp_path / "tier2.db"
    deltas = consolidate_domain_deltas(
        labels=labels,
        scores_by_domain=scores_by_verifier,
        db_path=db_path,
    )
    memory = Tier2ThresholdMemory(str(db_path))

    assert set(deltas) == set(VERIFIER_NAMES)
    for name in VERIFIER_NAMES:
        expected = (sum(scores_by_verifier[name]) / len(scores_by_verifier[name])) - 0.5
        assert deltas[name] == pytest.approx(expected)
        assert memory.get_domain_delta(name) == pytest.approx(deltas[name])

    adjusted = apply_consolidated_deltas(
        scores_by_domain=scores_by_verifier,
        db_path=db_path,
    )
    raw_first = scores_by_verifier["fr11_session_memory"][0]
    assert adjusted[0, 0] == pytest.approx(raw_first - deltas["fr11_session_memory"])

    memory.update_domain_delta("fr11_session_memory", [1.0] * 32, [1] * 32)
    poisoned = apply_consolidated_deltas(
        scores_by_domain=scores_by_verifier,
        db_path=db_path,
    )
    assert poisoned[0, 0] != pytest.approx(adjusted[0, 0])
    assert poisoned[0, 0] == pytest.approx(raw_first - 0.5)


def test_req_learn_3778_artifact_persists_state_and_preserves_no_regression(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3778-4/5/6/7: artifact uses persisted deltas and gates AUROC."""

    labels, scores_by_verifier = _score_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=10.0,
        now_s=10.4,
        db_path=tmp_path / "tier2.db",
        repo_root=tmp_path,
        memory_contribution_reference=0.0184712,
        frozen_ci=(0.9, 1.0),
    )

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == SUCCESS_VERDICT
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["per_domain_threshold_deltas"]["fr11_session_memory"] != 0.0
    assert artifact["ensemble_auroc_under_consolidation"] == pytest.approx(1.0)
    assert artifact["auroc_within_frozen_ci"] is True
    assert artifact["memory_contribution_preserved"] is True
    assert artifact["is_tier2_not_tier1"] is True
    assert artifact["tracker_state_persisted"] is True
    assert artifact["tier2_memory_state_path"] == "tier2.db"
    assert artifact["model_specs"]["verifiers"] == list(VERIFIER_NAMES)
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)
    assert "live_llm_inference" not in json.dumps(artifact)


def test_req_learn_3778_empty_fallback_does_not_fabricate_deltas(tmp_path: Path) -> None:
    """REQ-LEARN-3778-3: absent corpus records an honest blocked artifact."""

    output = write_artifact(
        tmp_path,
        output_path="results/blocked.json",
        db_path="results/blocked.db",
        labels=[],
        scores_by_verifier={},
        started_s=2.0,
        now_s=2.1,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == BLOCKED_CORPUS_VERDICT
    assert artifact["per_domain_threshold_deltas"] == {}
    assert artifact["ensemble_auroc_under_consolidation"] is None
    assert artifact["auroc_within_frozen_ci"] is False
    assert artifact["memory_contribution_preserved"] is False
    assert artifact["tracker_state_persisted"] is False
    assert not (tmp_path / "results/blocked.db").exists()


def test_req_learn_3778_build_artifact_scores_fover_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3778-1/2/6: runnable path checks preconditions and writes."""

    labels, scores_by_verifier = _score_fixture()
    corpus_path = tmp_path / "data/fover_corpus.jsonl"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text("{}\n" * 64, encoding="utf-8")
    monkeypatch.setattr(
        exp3778,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {
                "resource": "cached_traces_with_per_verifier_scores_and_labels",
                "available": True,
                "detail": "fixture",
            }
        ],
    )
    monkeypatch.setattr(
        exp3778,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    monkeypatch.setattr(
        exp3778,
        "load_memory_contribution_reference",
        lambda root: 0.0184712,
    )

    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.2,
        n_examples=64,
        db_path="results/tier2.db",
        frozen_ci=(0.9, 1.0),
    )
    validate_artifact(artifact)
    assert artifact["honest_verdict"] == SUCCESS_VERDICT
    assert artifact["preconditions_checked"][1]["resource"] == "fover_corpus_absolute_path"
    assert artifact["model_specs"]["corpus_absolute_path"] == str(corpus_path.resolve())
    assert artifact["methodology"]["lineage"] == "v18_tier2_constraint_memory_not_v17_tier1_precision_tracker"

    output = write_artifact(
        tmp_path,
        output_path="results/out.json",
        db_path="results/state.db",
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=0.2,
        frozen_ci=(0.9, 1.0),
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(saved)
    assert saved["tier2_memory_state_path"] == "results/state.db"

    missing = build_artifact(tmp_path / "missing", started_s=0.0, now_s=0.1)
    validate_artifact(missing)
    assert missing["honest_verdict"] == BLOCKED_CORPUS_VERDICT
    assert missing["preconditions_checked"][1]["available"] is False


def test_req_learn_3778_schema_and_input_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3778-2/4/6: schema and score guards are strict."""

    labels, scores_by_verifier = _score_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=0.1,
        db_path=tmp_path / "tier2.db",
        repo_root=tmp_path,
        memory_contribution_reference=0.0184712,
        frozen_ci=(0.9, 1.0),
    )

    assert exp3778.memory_contribution_preserved(None) is False
    assert exp3778.memory_contribution_preserved(0.0184712) is True
    assert exp3778._round(None) is None

    with pytest.raises(ValueError, match="missing required artifact fields"):
        bad = dict(artifact)
        bad.pop("honest_verdict")
        validate_artifact(bad)
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(dict(artifact, field_principles=None))
    with pytest.raises(ValueError, match="missing field principles"):
        validate_artifact(dict(artifact, field_principles={}))
    with pytest.raises(ValueError, match="inference_substrate"):
        validate_artifact(dict(artifact, inference_substrate="live_llm_inference"))
    with pytest.raises(ValueError, match="forbidden inference marker"):
        validate_artifact(dict(artifact, methodology={"bad": "CUDA"}))
    with pytest.raises(ValueError, match="per_domain_threshold_deltas"):
        validate_artifact(dict(artifact, per_domain_threshold_deltas=[]))
    with pytest.raises(ValueError, match="all four verifier domains"):
        validate_artifact(
            dict(artifact, per_domain_threshold_deltas={"fr11_session_memory": 0.0})
        )
    with pytest.raises(ValueError, match="auroc_within_frozen_ci"):
        validate_artifact(dict(artifact, auroc_within_frozen_ci="true"))
    with pytest.raises(ValueError, match="memory_contribution_preserved"):
        validate_artifact(dict(artifact, memory_contribution_preserved="true"))
    with pytest.raises(ValueError, match="is_tier2_not_tier1"):
        validate_artifact(dict(artifact, is_tier2_not_tier1=False))
    with pytest.raises(ValueError, match="tier2_lookup_is_cpu_memory"):
        validate_artifact(dict(artifact, tier2_lookup_is_cpu_memory="model_retrain"))
    with pytest.raises(ValueError, match="tracker_state_persisted"):
        validate_artifact(dict(artifact, tracker_state_persisted="yes"))
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="ensemble_auroc_under_consolidation"):
        validate_artifact(dict(artifact, ensemble_auroc_under_consolidation=None))
    with pytest.raises(ValueError, match="per_domain_threshold_deltas values"):
        validate_artifact(
            dict(
                artifact,
                per_domain_threshold_deltas=dict(
                    artifact["per_domain_threshold_deltas"],
                    fr11_session_memory="bad",
                ),
            )
        )
    with pytest.raises(ValueError, match="ensemble_auroc_under_consolidation"):
        validate_artifact(dict(artifact, ensemble_auroc_under_consolidation=1.5))
    with pytest.raises(ValueError, match="auroc_within_frozen_ci"):
        validate_artifact(dict(artifact, auroc_within_frozen_ci=False))
    with pytest.raises(ValueError, match="memory_contribution_preserved"):
        validate_artifact(dict(artifact, memory_contribution_preserved=False))
    with pytest.raises(ValueError, match="tracker_state_persisted"):
        validate_artifact(dict(artifact, tracker_state_persisted=False))
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        validate_artifact(dict(artifact, honest_verdict="complete: wrong"))

    blocked = exp3778._blocked_artifact(
        duration_s=0.1,
        random_seed=3778,
        db_output=tmp_path / "missing.db",
        repo_root=tmp_path,
        preconditions=[],
        verdict=BLOCKED_CORPUS_VERDICT,
    )
    validate_artifact(blocked)
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, per_domain_threshold_deltas={"fake": 0.1}))
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, ensemble_auroc_under_consolidation=0.5))

    with pytest.raises(ValueError, match="labels and verifier scores"):
        build_artifact_from_scores(
            labels=[1, 0],
            scores_by_verifier={name: [0.9] for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            db_path=tmp_path / "short.db",
        )
    with pytest.raises(ValueError, match="both binary classes"):
        build_artifact_from_scores(
            labels=[1] * 64,
            scores_by_verifier={name: [0.9] * 64 for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            db_path=tmp_path / "one_class.db",
        )
    with pytest.raises(ValueError, match="labels and verifier scores"):
        consolidate_domain_deltas(
            labels=[1, 0],
            scores_by_domain={name: [0.9] for name in VERIFIER_NAMES},
            db_path=tmp_path / "mismatch.db",
        )
    with pytest.raises(ValueError, match="adjusted verifier score matrix"):
        apply_consolidated_deltas(
            scores_by_domain={name: [float("nan")] for name in VERIFIER_NAMES},
            db_path=tmp_path / "nan.db",
        )

    assert exp3778.auroc_within_or_above_frozen_ci(float("nan")) is False
    assert exp3778._relative_path(tmp_path / "x.db", None) == (tmp_path / "x.db").as_posix()
    outside = tmp_path.parent / "outside.db"
    assert exp3778._relative_path(outside, tmp_path) == outside.as_posix()


def test_req_learn_3778_blocked_and_reference_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3778-2/3: blocked resources and references are explicit."""

    labels, scores_by_verifier = _score_fixture()
    corpus_path = tmp_path / "data/fover_corpus.jsonl"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text("{}\n" * 64, encoding="utf-8")

    monkeypatch.setattr(
        exp3778,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {"resource": "text_scoring_verifiers", "available": False, "detail": "missing"}
        ],
    )
    scoring_blocked = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_examples=64,
    )
    validate_artifact(scoring_blocked)
    assert scoring_blocked["honest_verdict"] == exp3778.BLOCKED_SCORING_VERDICT

    monkeypatch.setattr(
        exp3778,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {
                "resource": "cached_traces_with_per_verifier_scores_and_labels",
                "available": True,
                "detail": "fixture",
            }
        ],
    )

    def _raise_score(root: Path, n_examples: int, random_seed: int) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("cached scores unavailable")

    monkeypatch.setattr(exp3778, "score_fover_corpus", _raise_score)
    failed_scoring = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_examples=64,
    )
    validate_artifact(failed_scoring)
    assert failed_scoring["honest_verdict"] == exp3778.BLOCKED_SCORING_VERDICT
    assert failed_scoring["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"

    output = write_artifact(
        tmp_path / "missing",
        output_path="results/no_args.json",
        db_path="results/no_args.db",
        started_s=0.0,
        now_s=0.1,
    )
    no_args = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(no_args)
    assert no_args["honest_verdict"] == BLOCKED_CORPUS_VERDICT

    assert exp3778.load_memory_contribution_reference(tmp_path) is None
    reference_path = tmp_path / exp3778.EXP2837_REL_PATH
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text("{bad-json", encoding="utf-8")
    assert exp3778.load_memory_contribution_reference(tmp_path) is None
    reference_path.write_text(json.dumps({"learning_contribution": "bad"}), encoding="utf-8")
    assert exp3778.load_memory_contribution_reference(tmp_path) is None
    reference_path.write_text(json.dumps({"learning_contribution": 0.0184712}), encoding="utf-8")
    assert exp3778.load_memory_contribution_reference(tmp_path) == pytest.approx(0.0184712)

    blocked_interpreter = exp3778._blocked_verdict(
        [{"resource": "interpreter_runtime", "available": False}]
    )
    assert blocked_interpreter == exp3778.BLOCKED_INTERPRETER_VERDICT
    blocked_corpus = exp3778._blocked_verdict(
        [{"resource": "fover_corpus_absolute_path", "available": False}]
    )
    assert blocked_corpus == BLOCKED_CORPUS_VERDICT
    blocked_scoring = exp3778._blocked_verdict(
        [{"resource": "text_scoring_verifiers", "available": False}]
    )
    assert blocked_scoring == exp3778.BLOCKED_SCORING_VERDICT

    real_import = exp3778.importlib.import_module

    def _fake_import(name: str):
        if name == "yaml":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(exp3778.importlib, "import_module", _fake_import)
    precondition = exp3778._interpreter_precondition()
    assert precondition["available"] is False
    assert "missing=yaml" in precondition["detail"]

    monkeypatch.setattr(
        exp3778,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
