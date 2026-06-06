"""Tests for Exp 3886 defabricated graph-grounding fact verifier.

Spec refs: REQ-VERIFY-3886, SCENARIO-VERIFY-3886,
SCENARIO-VERIFY-3886-BLOCKED.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import graph_grounding_fact_verifier_defabricated as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _facts_rows() -> list[dict[str, Any]]:
    return [
        {
            "answer": "Anne Frank died of typhus in Bergen-Belsen.",
            "evidence_passage": "Anne Frank died of typhus in the Bergen-Belsen camp.",
            "is_hallucination": 0,
            "model_confidence": 0.6,
        },
        {
            "answer": "Kevone Charleston was shot by FBI agents in Georgia.",
            "evidence_passage": "Kevone Charleston was shot by FBI agents in Georgia.",
            "is_hallucination": 0,
            "model_confidence": 0.4,
        },
        {
            "answer": "Anne Frank died before February 7, 2022.",
            "evidence_passage": "Witnesses said Anne and Margot had symptoms before February 7, 1945.",
            "is_hallucination": 1,
            "model_confidence": 0.3,
        },
        {
            "answer": "The suspect injured two suspects during the chase.",
            "evidence_passage": "Two FBI agents were injured and the suspect was shot.",
            "is_hallucination": 1,
            "model_confidence": 0.2,
        },
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _graph(
    *,
    entities: tuple[str, ...],
    relations: tuple[mod.RelationTriple, ...],
    raw: str = "{}",
) -> mod.ExtractedGraph:
    return mod.ExtractedGraph(entities=entities, relations=relations, raw_response=raw)


class FakeGraphExtractor:
    def __init__(self, pairs: list[mod.GraphExtractionResult]) -> None:
        self.pairs = pairs
        self.calls: list[tuple[str, str, int]] = []

    def extract_pair(self, answer: str, evidence: str, item_index: int) -> mod.GraphExtractionResult:
        self.calls.append((answer, evidence, item_index))
        return self.pairs[item_index]


def _pairs() -> list[mod.GraphExtractionResult]:
    grounded_1 = mod.GraphExtractionResult(
        response_graph=_graph(
            entities=("Anne Frank", "typhus", "Bergen-Belsen"),
            relations=(mod.RelationTriple("Anne Frank", "died_of", "typhus"),),
            raw='{"response": "grounded"}',
        ),
        evidence_graph=_graph(
            entities=("Anne Frank", "typhus", "Bergen-Belsen camp"),
            relations=(mod.RelationTriple("Anne Frank", "died_of", "typhus"),),
            raw='{"evidence": "grounded"}',
        ),
        prompt_sha256="p0",
        completion_tokens=12,
    )
    grounded_2 = mod.GraphExtractionResult(
        response_graph=_graph(
            entities=("Kevone Charleston", "FBI agents", "Georgia"),
            relations=(mod.RelationTriple("Kevone Charleston", "shot_by", "FBI agents"),),
        ),
        evidence_graph=_graph(
            entities=("Kevone Charleston", "FBI agents", "Georgia"),
            relations=(mod.RelationTriple("Kevone Charleston", "shot_by", "FBI agents"),),
        ),
        prompt_sha256="p1",
        completion_tokens=9,
    )
    ungrounded_1 = mod.GraphExtractionResult(
        response_graph=_graph(
            entities=("Anne Frank", "February 7, 2022"),
            relations=(mod.RelationTriple("Anne Frank", "died_before", "February 7, 2022"),),
        ),
        evidence_graph=_graph(
            entities=("Anne Frank", "February 7, 1945"),
            relations=(mod.RelationTriple("Anne Frank", "symptoms_before", "February 7, 1945"),),
        ),
        prompt_sha256="p2",
        completion_tokens=10,
    )
    ungrounded_2 = mod.GraphExtractionResult(
        response_graph=_graph(
            entities=("suspect", "two suspects", "chase"),
            relations=(mod.RelationTriple("suspect", "injured", "two suspects"),),
        ),
        evidence_graph=_graph(
            entities=("two FBI agents", "suspect", "shot"),
            relations=(mod.RelationTriple("two FBI agents", "injured_during", "chase"),),
        ),
        prompt_sha256="p3",
        completion_tokens=11,
    )
    return [grounded_1, grounded_2, ungrounded_1, ungrounded_2]


def _json_from_pair(pair: mod.GraphExtractionResult) -> str:
    def graph_payload(graph: mod.ExtractedGraph) -> dict[str, Any]:
        return {
            "entities": list(graph.entities),
            "relations": [
                {"subject": rel.subject, "relation": rel.relation, "object": rel.object}
                for rel in graph.relations
            ],
        }

    return json.dumps(
        {
            "response": graph_payload(pair.response_graph),
            "evidence": graph_payload(pair.evidence_graph),
        }
    )


class FakeLlama:
    def __init__(self, outputs: list[str] | None = None, **kwargs: Any) -> None:
        self.outputs = list(outputs or [])
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        text = self.outputs.pop(0) if self.outputs else "not json"
        return {"choices": [{"text": text}], "usage": {"completion_tokens": 5}}


def test_req_verify_3886_spec_anchor_exists() -> None:
    """REQ-VERIFY-3886: the defabricated run is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3886" in spec
    assert "SCENARIO-VERIFY-3886" in spec
    assert "blocked_graph_verifier_not_invoked" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.PER_ITEM_REL_PATH.as_posix() in spec


def test_req_verify_3886_hallugraph_entity_relation_cfi_scores() -> None:
    """REQ-VERIFY-3886: HalluGraph EG/RP/CFI produces bounded hallucination scores."""

    grounded = mod.compute_hallugraph_score(_pairs()[0])
    ungrounded = mod.compute_hallugraph_score(_pairs()[2])
    empty_response = mod.compute_hallugraph_score(
        mod.GraphExtractionResult(
            response_graph=_graph(entities=(), relations=()),
            evidence_graph=_graph(entities=("Anne Frank",), relations=()),
            prompt_sha256="empty",
            completion_tokens=0,
        )
    )

    assert grounded.entity_grounding == pytest.approx(1.0)
    assert grounded.relation_preservation == pytest.approx(1.0)
    assert grounded.composite_fidelity_index == pytest.approx(1.0)
    assert grounded.hallucination_score == pytest.approx(0.0)
    assert ungrounded.hallucination_score > grounded.hallucination_score
    assert ungrounded.missing_entities
    assert ungrounded.unsupported_relations
    assert empty_response.composite_fidelity_index == pytest.approx(1.0)


def test_scenario_verify_3886_writes_scores_and_signal_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3886: live-model scores persist and gate the facts signal."""

    scores_path = tmp_path / mod.PER_ITEM_REL_PATH
    artifact = mod.build_artifact_from_rows(
        _facts_rows(),
        graph_extractor=FakeGraphExtractor(_pairs()),
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf", "loader": "fake-llama"},
        per_item_scores_path=scores_path,
        preconditions_checked=[mod.PreconditionCheck("cuda_available", True, "ok")],
        math_scores=[0.8, 0.7, 0.2, 0.1],
        started_s=10.0,
        now_s=75.0,
        tests_run=["pytest tests/python/test_experiment_3886_graph_grounding_fact_verifier_defabricated.py"],
    )

    mod.validate_artifact(artifact)
    per_item_rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["honest_verdict"].startswith(
        "complete: graph_grounding_FACTS_SIGNAL_REPRODUCED_delta"
    )
    assert artifact["model_invoked"] is True
    assert artifact["duration_s"] == pytest.approx(65.0)
    assert artifact["n_items"] == 4
    assert artifact["facts_catch_delta"] > 0.05
    assert type(artifact["facts_catch_delta"]) is float
    assert artifact["per_item_scores_path"] == mod.PER_ITEM_REL_PATH.as_posix()
    assert len(per_item_rows) == 4
    assert all("graph_score" in row and "math_baseline_score" in row for row in per_item_rows)
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])
    assert all(isinstance(artifact["field_principles"][field], str) for field in mod.REQUIRED_PRINCIPLE_FIELDS)


def test_scenario_verify_3886_no_signal_terminal_verdict(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3886: non-positive graph delta is reported as no signal."""

    artifact = mod.build_artifact_from_rows(
        _facts_rows(),
        graph_extractor=FakeGraphExtractor(_pairs()),
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        per_item_scores_path=tmp_path / mod.PER_ITEM_REL_PATH,
        math_scores=[0.1, 0.2, 0.8, 0.9],
        started_s=0.0,
        now_s=65.0,
    )

    mod.validate_artifact(artifact)
    assert artifact["facts_catch_delta"] <= 0.05
    assert artifact["honest_verdict"].startswith("complete: graph_grounding_NO_SIGNAL_")


def test_scenario_verify_3886_blocks_sub60_duration_after_scores(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3886-BLOCKED: sub-60s runs cannot claim model invocation."""

    artifact = mod.build_artifact_from_rows(
        _facts_rows(),
        graph_extractor=FakeGraphExtractor(_pairs()),
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        per_item_scores_path=tmp_path / mod.PER_ITEM_REL_PATH,
        math_scores=[0.8, 0.7, 0.2, 0.1],
        started_s=0.0,
        now_s=59.9,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_graph_verifier_not_invoked"
    assert artifact["model_invoked"] is False
    assert artifact["facts_catch_delta"] == 0.0
    assert artifact["graph_auroc"] is None
    assert artifact["per_item_scores_path"] == mod.PER_ITEM_REL_PATH.as_posix()


@pytest.mark.parametrize(
    ("completed", "model_present", "corpus", "expected"),
    [
        (subprocess.CompletedProcess("python", 1, stderr="no cuda"), True, True, "blocked_no_cuda"),
        (subprocess.CompletedProcess("python", 0, stdout="ok"), False, True, "blocked_model_not_cached"),
        (subprocess.CompletedProcess("python", 0, stdout="ok"), True, False, "blocked_facts_corpus_missing"),
    ],
)
def test_scenario_verify_3886_preconditions_fail_closed(
    tmp_path: Path,
    completed: subprocess.CompletedProcess[str],
    model_present: bool,
    corpus: bool,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3886-BLOCKED: missing live resources produce terminal blockers."""

    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: object())
    model_path = tmp_path / "model.gguf"
    if model_present:
        model_path.write_bytes(b"gguf")
    if corpus:
        _write_jsonl(tmp_path / "data" / "real_factual_corpus_ragtruth.jsonl", _facts_rows())

    preflight = mod.probe_preconditions(
        mod.ExperimentConfig(repo_root=tmp_path),
        command_runner=lambda *_args, **_kwargs: completed,
        resolve_gguf=lambda _hf_id: str(model_path) if model_present else None,
    )

    assert preflight.blocked_reason == expected
    artifact = mod.build_blocked_artifact(
        reason=preflight.blocked_reason or "blocked_fixture",
        preconditions_checked=preflight.checks,
        duration_s=1.0,
        model_specs=preflight.model_specs,
    )
    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected
    assert artifact["model_invoked"] is False


def test_req_verify_3886_json_parser_and_artifact_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-3886: model JSON parsing and artifact validation fail closed."""

    parsed = mod.parse_model_graph_response(
        """
        text before
        {"response": {"entities": ["Anne Frank"], "relations": [{"subject": "Anne Frank", "relation": "died of", "object": "typhus"}]},
         "evidence": {"entities": ["Anne Frank", "typhus"], "relations": [{"subject": "Anne Frank", "predicate": "died of", "object": "typhus"}]}}
        trailing text
        """
    )
    malformed = mod.parse_model_graph_response("not json")

    assert parsed.response_graph.entities == ("Anne Frank",)
    assert parsed.response_graph.relations[0].relation == "died of"
    assert parsed.evidence_graph.relations[0].relation == "died of"
    assert parsed.completion_tokens == 0
    assert malformed.response_graph.raw_response == "not json"

    artifact = mod.build_blocked_artifact(
        reason="blocked_fixture",
        preconditions_checked=[],
        duration_s=0.5,
        model_specs={},
    )
    bad = dict(artifact)
    bad["facts_catch_delta"] = {"value": 0.0}
    with pytest.raises(ValueError, match="bare float"):
        mod.validate_artifact(bad)

    bad = dict(artifact)
    bad["field_principles"] = {"facts_catch_delta": {"principle": "wrapped"}}
    with pytest.raises(ValueError, match="principle strings"):
        mod.validate_artifact(bad)


def test_req_verify_3886_live_extractor_seam_and_fallback() -> None:
    """REQ-VERIFY-3886: live extractor parses model output and has a bounded fallback."""

    fake = FakeLlama([_json_from_pair(_pairs()[0])])
    factory_kwargs: dict[str, Any] = {}
    extractor = mod.LlamaGraphExtractor(
        {"model_path": "fixture.gguf", "n_gpu_layers": 0, "n_ctx": 256, "n_batch": 8},
        llama_factory=lambda **kwargs: (factory_kwargs.update(kwargs) or fake),
        max_tokens=32,
    )
    parsed = extractor.extract_pair("Anne Frank died of typhus.", "Anne Frank died of typhus.", 0)

    fallback_extractor = mod.LlamaGraphExtractor(
        {"model_path": "fixture.gguf"},
        llama_factory=lambda **_kwargs: FakeLlama(["not json"]),
        max_tokens=8,
    )
    fallback = fallback_extractor.extract_pair(
        "Anne Frank died of typhus.",
        "Anne Frank died of typhus.",
        1,
    )

    assert factory_kwargs["model_path"] == "fixture.gguf"
    assert fake.calls[0]["kwargs"]["max_tokens"] == 32
    assert parsed.response_graph.entities[0] == "Anne Frank"
    assert parsed.completion_tokens == 5
    assert fallback.response_graph.relations
    assert mod._graph_empty(mod.ExtractedGraph(entities=(), relations=())) is True
    assert "truncated" in mod.graph_extraction_prompt("a" * 1300, "b" * 3300)


def test_scenario_verify_3886_run_experiment_success_and_inference_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3886: run_experiment writes complete or blocked artifacts."""

    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: object())
    _write_jsonl(tmp_path / "data" / "real_factual_corpus_ragtruth.jsonl", _facts_rows())
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    outputs = [_json_from_pair(pair) for pair in _pairs()]
    clock_values = iter([10.0, 75.0])

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            sample_size=4,
            clock=lambda: next(clock_values),
        ),
        write=True,
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess("python", 0, stdout="ok"),
        resolve_gguf=lambda _hf_id: str(model_path),
        llama_factory=lambda **kwargs: FakeLlama(outputs, **kwargs),
    )

    assert artifact["model_invoked"] is True
    assert (tmp_path / mod.OUTPUT_REL_PATH).is_file()
    assert (tmp_path / mod.PER_ITEM_REL_PATH).is_file()

    fail_clock = iter([0.0, 1.0])
    failed = mod.run_experiment(
        mod.ExperimentConfig(repo_root=tmp_path, sample_size=4, clock=lambda: next(fail_clock)),
        write=False,
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess("python", 0, stdout="ok"),
        resolve_gguf=lambda _hf_id: str(model_path),
        llama_factory=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert failed["honest_verdict"] == "blocked_llama_cpp_inference_failed"


def test_req_verify_3886_helpers_and_validation_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-3886: helper branches and schema rejections are covered."""

    config = mod.ExperimentConfig(
        repo_root=tmp_path,
        output_path=Path("custom/out.json"),
        per_item_scores_path=Path("custom/scores.jsonl"),
        started_at=3.0,
        clock=lambda: 9.0,
    )
    assert config.resolved_output_path() == tmp_path / "custom" / "out.json"
    assert config.resolved_per_item_scores_path() == tmp_path / "custom" / "scores.jsonl"
    assert config.venv_python() == tmp_path / ".venv" / "bin" / "python"
    assert config.start_time() == 3.0

    artifact = mod.build_artifact_from_rows(
        [],
        graph_extractor=FakeGraphExtractor([]),
        model_specs={},
        per_item_scores_path=tmp_path / mod.PER_ITEM_REL_PATH,
        now_s=2.0,
    )
    assert artifact["honest_verdict"] == "blocked_facts_corpus_missing"

    assert mod._probe_cuda(
        config,
        command_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    ).available is False
    monkeypatch.setattr(
        mod.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("boom")) if name == "llama_cpp" else object(),
    )
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    _write_jsonl(tmp_path / "data" / "real_factual_corpus_ragtruth.jsonl", _facts_rows())
    preflight = mod.probe_preconditions(
        mod.ExperimentConfig(repo_root=tmp_path),
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess("python", 0, stdout="ok"),
        resolve_gguf=lambda _hf_id: str(model_path),
    )
    assert preflight.blocked_reason == "blocked_llama_cpp_not_installed"

    fallback_specs, fallback_checks = mod._resolve_model(
        lambda hf_id: None if "Qwen" in hf_id else str(model_path)
    )
    assert fallback_specs["fallback_used"] is True
    assert len(fallback_checks) == 2
    assert mod._model_path_available(None) is False
    assert mod._corpus_has_labels(None, 4) is False
    assert mod.resolve_facts_corpus(tmp_path) == tmp_path / "data" / "real_factual_corpus_ragtruth.jsonl"

    assert mod._parse_graph("bad", raw_response="raw").entities == ()
    parsed = mod._parse_graph(
        {
            "entities": ["x", "x", ""],
            "relations": ["bad", {"subject": "x"}, {"subject": "x", "verb": "near", "target": "y"}],
        },
        raw_response="raw",
    )
    assert parsed.entities == ("x",)
    assert parsed.relations == (mod.RelationTriple("x", "near", "y"),)
    assert mod._first_json_object("prefix {bad") is None
    assert mod._first_json_object("{bad}") is None
    assert mod._first_json_object('{"x": "quote \\\\" ok"}') is None

    assert mod._entity_supported("", ()) is True
    assert mod._entity_supported("Anne Frank", ("", "noise")) is False
    assert mod._entity_supported("a b c d e f g", ("a b c d e f h",)) is True
    assert mod._entity_supported("Anne Frank", ("noise", "Margot Frank House")) is False
    assert mod._relation_supported(
        mod.RelationTriple("Anne", "died of", "typhus"),
        (mod.RelationTriple("Margot", "died of", "typhus"),),
        ("Anne", "typhus"),
    ) is False
    assert mod._relation_supported(
        mod.RelationTriple("Anne", "died of", "typhus"),
        (mod.RelationTriple("Anne", "died of", "lice"),),
        ("Anne", "typhus"),
    ) is False
    assert mod._relation_supported(
        mod.RelationTriple("Anne Frank", "died of typhus in camp", "Bergen-Belsen"),
        (
                mod.RelationTriple(
                    "Anne Frank",
                    "died of typhus camp",
                    "Bergen Belsen",
                ),
        ),
        ("Anne Frank", "Bergen Belsen"),
    ) is True
    assert mod._relation_supported(
        mod.RelationTriple("Anne", "died of", "typhus"),
        (mod.RelationTriple("Anne", "died from", "typhus"),),
        ("Anne", "typhus"),
    ) is False
    assert mod._relation_supported(
        mod.RelationTriple("Anne", "died of", "typhus"),
        (mod.RelationTriple("Anne", "died of", "typhus"),),
        ("Anne", "typhus"),
    ) is True
    assert mod._jaccard(set(), set()) == 1.0
    assert mod._jaccard({"x"}, set()) == 0.0
    assert mod._finite_triplets([1], [float("nan")], [0.1]) == ([], [], [])
    assert mod._checks_to_dicts([{"resource": "x", "available": True, "detail": "ok"}])[0]["resource"] == "x"
    assert mod._repo_path(tmp_path, Path("/outside")) == Path("/outside")
    assert mod._artifact_path(None) == ""
    assert mod._artifact_path(Path("/outside/file.json")) == "/outside/file.json"
    assert mod._extract_text("raw") == "raw"
    assert mod._extract_text({"choices": [{"message": {"content": "chat"}}]}) == "chat"
    assert mod._extract_text({"choices": []}) == ""
    assert mod._extract_text({"choices": ["bad"]}) == ""
    assert mod._extract_text({"choices": [{"message": "bad"}]}) == ""
    assert mod._extract_text(object()) == ""
    assert mod._completion_tokens({"usage": {"completion_tokens": "bad"}}, "x y") == 0
    assert mod._completion_tokens({}, "x y") == 2
    assert mod._clamp01(-1.0) == 0.0
    assert mod._clamp01(2.0) == 1.0

    invalids = [
        (dict(artifact, honest_verdict="bad"), "honest_verdict"),
        (dict(artifact, model_invoked="false"), "bare bool"),
        (dict(artifact, n_items=-1), "n_items"),
        (dict(artifact, per_item_scores_path=5), "per_item_scores_path"),
        (dict(artifact, field_principles=[]), "field_principles"),
        (dict(artifact, field_principles={"facts_catch_delta": "x"}), "missing required"),
        (dict(artifact, graph_auroc=2.0), "graph_auroc"),
        (dict(artifact, duration_s=-1), "duration_s"),
        (dict(artifact, honest_verdict="complete: fixture", model_invoked=False), "complete artifacts"),
        (dict(artifact, model_invoked=True, duration_s=1.0), "duration_s>=60"),
    ]
    missing = dict(artifact)
    missing.pop("n_items")
    invalids.append((missing, "missing required"))
    for bad_artifact, match in invalids:
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad_artifact)


def test_scenario_verify_3886_import_failure_and_blocked_run_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3886-BLOCKED: import and preflight blockers are persisted."""

    monkeypatch.setattr(
        mod.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("boom")) if name == "carnot.verify" else object(),
    )
    preflight = mod.probe_preconditions(
        mod.ExperimentConfig(repo_root=tmp_path),
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess("python", 0, stdout="ok"),
        resolve_gguf=lambda _hf_id: None,
    )
    assert preflight.blocked_reason == "blocked_carnot_verify_import"

    blocked = mod.run_experiment(
        mod.ExperimentConfig(repo_root=tmp_path, clock=lambda: 1.0),
        write=True,
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess("python", 1, stderr="no cuda"),
        resolve_gguf=lambda _hf_id: None,
    )
    assert blocked["honest_verdict"] == "blocked_no_cuda"
    assert (tmp_path / mod.OUTPUT_REL_PATH).is_file()


def test_req_verify_3886_main_uses_cli_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-3886: module CLI prints artifact path and terminal verdict."""

    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda _config, write=True: {
            "honest_verdict": "blocked_fixture",
        },
    )

    assert mod.main(["--repo-root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert str(tmp_path / mod.OUTPUT_REL_PATH) in captured.out
    assert "blocked_fixture" in captured.out
