"""Tests for Exp 3862 graph-grounding fact-verifier prototype.

Spec: REQ-VERIFY-3862, SCENARIO-VERIFY-3862,
SCENARIO-VERIFY-3862-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.verify import graph_grounding_probe as mod


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _facts_rows() -> list[dict[str, Any]]:
    return [
        {
            "answer": "Anne Frank died of typhus in Bergen-Belsen.",
            "evidence_passage": "Anne Frank died of typhus in the Bergen-Belsen camp.",
            "is_hallucination": 0,
            "model_confidence": 0.7,
            "question": "fixture",
        },
        {
            "answer": "Margot Frank died before Anne.",
            "evidence_passage": "The exact dates remain unclear. Margot died before Anne.",
            "is_hallucination": 0,
            "model_confidence": 0.6,
            "question": "fixture",
        },
        {
            "answer": "Anne Frank died before February 7, 2022.",
            "evidence_passage": "Witnesses said Anne had symptoms before February 7, 1945.",
            "is_hallucination": 1,
            "model_confidence": 0.8,
            "question": "fixture",
        },
        {
            "answer": "The Bergen-Belsen camp was liberated in Australia.",
            "evidence_passage": "The Bergen-Belsen concentration camp was liberated in April 1945.",
            "is_hallucination": 1,
            "model_confidence": 0.9,
            "question": "fixture",
        },
    ]


def test_req_verify_3862_extracts_triples_and_scores_context_alignment() -> None:
    """REQ-VERIFY-3862: claim triples align against a context graph."""

    verifier = mod.GraphGroundingProbe()
    grounded = verifier.score_claim(
        "Anne Frank died of typhus in Bergen-Belsen.",
        "Anne Frank died of typhus in the Bergen-Belsen camp.",
    )
    ungrounded = verifier.score_claim(
        "Anne Frank died before February 7, 2022.",
        "Witnesses said Anne had symptoms before February 7, 1945.",
    )

    assert verifier.triple_extraction_method == "rule_based"
    assert grounded.claim_triples
    assert grounded.context_graph.triples
    assert grounded.energy < ungrounded.energy
    assert ungrounded.missing_key_tokens
    assert "heuristic" in (mod.__doc__ or "").lower()
    assert "not a model-backed" in (mod.__doc__ or "").lower()
    assert verifier.verify("Anne Frank died of typhus.", "Anne Frank died of typhus.") == (
        verifier.score_claim("Anne Frank died of typhus.", "Anne Frank died of typhus.").energy
    )


def test_req_verify_3862_fallback_and_private_alignment_edges() -> None:
    """REQ-VERIFY-3862: fallback token scoring handles sparse text safely."""

    verifier = mod.GraphGroundingProbe()
    sparse = verifier.score_claim("a", "")
    no_relation = verifier.extract_triples("water freezes quickly")

    assert sparse.claim_triples == ()
    assert sparse.energy == pytest.approx(0.0)
    assert no_relation[0].relation == "related_to"
    assert no_relation[0].subject == "water freez quickly"
    assert mod._best_triple_alignment(no_relation[0], sparse.context_graph) == 0.0
    assert mod._support_ratio((), frozenset()) == 1.0
    assert mod._stem("berries") == "berry"
    assert mod._jaccard(set(), set()) == 1.0
    assert mod._jaccard({"x"}, set()) == 0.0


def test_scenario_verify_3862_artifact_signal_and_no_signal_gates() -> None:
    """SCENARIO-VERIFY-3862: AUROC and delta select honest terminal verdicts."""

    rows = _facts_rows()
    signal = mod.build_artifact_from_rows(
        rows,
        graph_scores=[0.05, 0.20, 0.80, 0.95],
        math_scores=[0.80, 0.70, 0.30, 0.20],
        started_s=10.0,
        now_s=12.0,
    )
    mod.validate_artifact(signal)
    assert signal["graph_grounding_auroc"] == pytest.approx(1.0)
    assert signal["math_ensemble_auroc_on_facts"] == pytest.approx(0.0)
    assert signal["facts_catch_delta"] == pytest.approx(1.0)
    assert type(signal["facts_catch_delta"]) is float
    assert signal["honest_verdict"].startswith(
        "complete: graph_grounding_prototype_SIGNAL_"
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(signal["field_principles"])

    no_signal = mod.build_artifact_from_rows(
        rows,
        graph_scores=[0.60, 0.60, 0.60, 0.60],
        math_scores=[0.05, 0.10, 0.90, 0.95],
        started_s=10.0,
        now_s=11.0,
    )
    mod.validate_artifact(no_signal)
    assert no_signal["graph_grounding_auroc"] == pytest.approx(0.5)
    assert no_signal["facts_catch_delta"] == pytest.approx(-0.5)
    assert no_signal["honest_verdict"].startswith(
        "complete: graph_grounding_prototype_NO_SIGNAL_"
    )


def test_req_verify_3862_scores_rows_without_overrides() -> None:
    """REQ-VERIFY-3862: default scoring uses graph and math-bound score paths."""

    rows = _facts_rows()
    graph_scores = mod.score_rows_graph_grounding(rows)
    math_scores = mod.score_rows_math_bound_ensemble(rows)
    artifact = mod.build_artifact_from_rows(rows, started_s=0.0, now_s=1.0)

    mod.validate_artifact(artifact)
    assert len(graph_scores) == len(rows)
    assert len(math_scores) == len(rows)
    assert all(0.0 <= score <= 1.0 for score in graph_scores + math_scores)
    assert artifact["n_facts_items"] == len(rows)


def test_scenario_verify_3862_blocked_when_no_facts_corpus(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3862-BLOCKED: missing corpus does not fabricate scores."""

    artifact = mod.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=2.0,
        download_fn=lambda _root: None,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_facts_corpus_not_available"
    assert artifact["n_facts_items"] == 0
    assert artifact["graph_grounding_auroc"] is None
    assert artifact["math_ensemble_auroc_on_facts"] is None
    assert artifact["facts_catch_delta"] == 0.0


def test_scenario_verify_3862_blocked_when_import_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3862-BLOCKED: import failure gets its own blocked verdict."""

    monkeypatch.setattr(
        mod.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("boom")),
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_carnot_verify_import"
    assert artifact["preconditions_checked"][0]["available"] is False


def test_req_verify_3862_write_artifact_uses_cached_ragtruth_fixture(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3862: top-level writer persists the required JSON artifact."""

    _write_jsonl(tmp_path / "data" / "real_factual_corpus_ragtruth.jsonl", _facts_rows())
    output = mod.write_artifact(
        tmp_path,
        graph_scores=[0.05, 0.20, 0.80, 0.95],
        math_scores=[0.80, 0.70, 0.30, 0.20],
        tests_run=["pytest tests/python/test_experiment_3862_graph_grounding_fact_verifier.py"],
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert output.name == "experiment_3862_graph_grounding_fact_verifier_prototype_v2.json"
    assert saved["n_facts_items"] == 4
    assert saved["tests_run"] == [
        "pytest tests/python/test_experiment_3862_graph_grounding_fact_verifier.py"
    ]


def test_req_verify_3862_loads_balanced_rows_and_download_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3862: corpus loading skips malformed rows and balances labels."""

    rows = [{"answer": "missing label", "evidence_passage": "x"}] + _facts_rows()
    path = tmp_path / "facts.jsonl"
    _write_jsonl(path, rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    loaded = mod.load_facts_rows(path, sample_size=2)

    monkeypatch.delenv("CARNOT_FACTS_CORPUS_URL", raising=False)
    assert mod.try_download_small_facts_corpus(tmp_path) is None
    assert len(loaded) == 2
    assert [row["is_hallucination"] for row in loaded] == [0, 1]
    assert mod._read_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._display_path(tmp_path, None) == "unknown"
    assert mod._display_path(tmp_path, Path("/outside/file.jsonl")) == "/outside/file.jsonl"


def test_req_verify_3862_gpu_helper_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-3862: GPU precondition records free-vs-unavailable honestly."""

    def run_free(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess("nvidia-smi", 0, stdout="100, 40000, 0\n")

    def run_busy(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            "nvidia-smi", 0, stdout="bad line\n20000, 40000, 50\n"
        )

    def run_missing(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess("nvidia-smi", 1, stdout="")

    monkeypatch.setattr(mod.subprocess, "run", run_free)
    assert mod._free_gpu_for_gguf() is True
    monkeypatch.setattr(mod.subprocess, "run", run_busy)
    assert mod._free_gpu_for_gguf() is False
    monkeypatch.setattr(mod.subprocess, "run", run_missing)
    assert mod._free_gpu_for_gguf() is False


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(lambda artifact: artifact.pop("duration_s"), "missing", id="missing"),
        pytest.param(
            lambda artifact: artifact.__setitem__("honest_verdict", "maybe"),
            "honest_verdict",
            id="bad_verdict",
        ),
        pytest.param(
            lambda artifact: artifact.__setitem__("triple_extraction_method", "magic"),
            "triple_extraction_method",
            id="bad_method",
        ),
        pytest.param(
            lambda artifact: artifact.__setitem__("facts_catch_delta", "0.1"),
            "facts_catch_delta",
            id="bad_delta_type",
        ),
        pytest.param(
            lambda artifact: artifact.__setitem__("verifier_authenticity_disclosed", "true"),
            "verifier_authenticity_disclosed",
            id="bad_auth_type",
        ),
        pytest.param(
            lambda artifact: artifact.__setitem__("n_facts_items", -1),
            "n_facts_items",
            id="bad_n",
        ),
        pytest.param(
            lambda artifact: artifact.__setitem__("field_principles", None),
            "field_principles",
            id="bad_principles",
        ),
        pytest.param(
            lambda artifact: artifact["field_principles"].pop("duration_s"),
            "field_principles missing",
            id="missing_principle",
        ),
        pytest.param(
            lambda artifact: artifact.__setitem__("duration_s", -0.1),
            "duration_s",
            id="bad_duration",
        ),
        pytest.param(
            lambda artifact: artifact.__setitem__("graph_grounding_auroc", 1.5),
            "graph_grounding_auroc",
            id="bad_auc",
        ),
    ],
)
def test_req_verify_3862_validate_artifact_rejects_schema_errors(
    mutate: Any,
    match: str,
) -> None:
    """REQ-VERIFY-3862: schema validation rejects malformed required fields."""

    artifact = mod.build_artifact_from_rows(
        _facts_rows(),
        graph_scores=[0.05, 0.20, 0.80, 0.95],
        math_scores=[0.80, 0.70, 0.30, 0.20],
        started_s=10.0,
        now_s=12.0,
    )
    mutate(artifact)

    with pytest.raises(ValueError, match=match):
        mod.validate_artifact(artifact)
