"""Tests for Exp 2932 citation hallucination field verifier.

Spec: REQ-VERIFY-2932, SCENARIO-VERIFY-2932.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import citation_hallucination_field_verifier as exp


MANDATED = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _spec() -> dict[str, Any]:
    return {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED,
        "gpu": 0,
        "model_path": "/tmp/gemma4-26b.gguf",
    }


def _citation_text(case: exp.CitationCase) -> str:
    fields = case.citation
    return exp.format_citation_line(
        {
            "seed_id": fields.seed_id,
            "title": fields.title,
            "authors": fields.authors,
            "year": fields.year,
            "venue": fields.venue,
            "arxiv_id": fields.arxiv_id,
            "url": fields.url,
        }
    )


def test_req_verify_2932_spec_and_fixture_cover_required_mutations() -> None:
    """REQ-VERIFY-2932: fixture is spec-backed and covers all citation fields."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    cases = exp.build_citation_fixture()
    prompts = [exp.prompt_for_case(case) for case in cases]
    mutation_fields = {case.mutation_field for case in cases if case.mutation_field}

    assert "REQ-VERIFY-2932" in spec
    assert "SCENARIO-VERIFY-2932" in spec
    assert "research-references.md" in spec
    assert len(cases) == 30
    assert len({case.case_id for case in cases}) == len(cases)
    assert mutation_fields == {"title", "authors", "year", "venue", "arxiv_id", "url"}
    assert {case.expected_taxonomy for case in cases} == {
        "real",
        "hallucinated-field",
        "nonexistent-seed",
    }
    assert sum(case.expected_taxonomy == "real" for case in cases) >= 5
    assert all(case.source_document == "research-references.md" for case in cases)
    assert all("CITE[" in prompt for prompt in prompts)
    assert all("Do not use web search" in prompt for prompt in prompts)


def test_scenario_verify_2932_extraction_and_field_classification() -> None:
    """SCENARIO-VERIFY-2932: raw citation text maps to deterministic taxonomy."""

    cases = exp.build_citation_fixture()
    real = next(case for case in cases if case.expected_taxonomy == "real")
    mutated = next(case for case in cases if case.mutation_field == "arxiv_id")
    nonexistent = next(case for case in cases if case.expected_taxonomy == "nonexistent-seed")

    real_row = exp.evaluate_raw_output(
        real,
        _citation_text(real),
        cases,
        generation_metadata={"model_hf_id": MANDATED, "raw_response_path": "real.txt"},
    )
    mutated_row = exp.evaluate_raw_output(
        mutated,
        _citation_text(mutated),
        cases,
        generation_metadata={"model_hf_id": MANDATED, "raw_response_path": "mutated.txt"},
    )
    nonexistent_row = exp.evaluate_raw_output(
        nonexistent,
        _citation_text(nonexistent),
        cases,
        generation_metadata={"model_hf_id": MANDATED, "raw_response_path": "fake.txt"},
    )
    ambiguous = exp.verify_citation(
        exp.extract_citations("Gladstone et al. (2026) discuss scalable learners.")[0],
        cases,
    )

    assert real_row["extraction_success"] is True
    assert real_row["taxonomy"] == "real"
    assert real_row["field_match_count"] == real_row["field_comparison_count"]
    assert real_row["taxonomy_correct"] is True

    assert mutated_row["taxonomy"] == "hallucinated-field"
    assert mutated_row["taxonomy_correct"] is True
    assert mutated_row["mismatched_fields"] == ["arxiv_id"]
    assert mutated_row["field_match_count"] == mutated_row["field_comparison_count"] - 1

    assert nonexistent_row["taxonomy"] == "nonexistent-seed"
    assert nonexistent_row["taxonomy_correct"] is True
    assert nonexistent_row["matched_seed_id"] is None

    assert ambiguous["taxonomy"] == "potential/ambiguous"
    assert ambiguous["mismatched_fields"] == []


def test_req_verify_2932_parser_handles_json_missing_and_lookup_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-2932: deterministic extraction covers JSON and fallback matching."""

    cases = exp.build_citation_fixture(tmp_path / "missing-research-references.md")
    real = next(case for case in cases if case.expected_taxonomy == "real")
    citation = real.citation
    json_text = json.dumps(
        {
            "citations": [
                {
                    "title": citation.title,
                    "author": [citation.authors],
                    "year": "2026",
                    "journal": citation.venue,
                    "doi": citation.arxiv_id,
                    "url": citation.url,
                }
            ]
        }
    )
    list_text = json.dumps(
        [
            {
                "seed_id": citation.seed_id,
                "title": citation.title,
                "authors": citation.authors,
                "year": citation.year,
                "venue": citation.venue,
                "arxiv": citation.arxiv_id,
                "url": citation.url,
            }
        ]
    )
    object_text = json.dumps(
        {
            "citation_id": citation.seed_id,
            "title": citation.title,
            "authors": citation.authors,
            "year": citation.year,
            "booktitle": citation.venue,
            "id": citation.arxiv_id,
            "url": citation.url,
        }
    )
    bad_segment_text = (
        f"CITE[note without equals; title={citation.title}; authors={citation.authors}; "
        f"year={citation.year}; venue={citation.venue}; arxiv_id={citation.arxiv_id}; "
        f"url={citation.url}]"
    )

    assert exp.verify_citation(exp.extract_citations(json_text)[0], cases)["taxonomy"] == "real"
    assert exp.verify_citation(exp.extract_citations(list_text)[0], cases)["taxonomy"] == "real"
    assert exp.verify_citation(exp.extract_citations(object_text)[0], cases)["taxonomy"] == "real"
    assert exp.verify_citation(exp.extract_citations(bad_segment_text)[0], cases)["taxonomy"] == "real"
    assert exp.verify_citation(
        exp.CitationFields(url=citation.url, title=citation.title, year=None),
        cases,
    )["taxonomy"] == "potential/ambiguous"
    assert exp.verify_citation(exp.CitationFields(arxiv_id=citation.arxiv_id), cases)[
        "matched_seed_id"
    ] == citation.seed_id
    assert exp.verify_citation(exp.CitationFields(title=citation.title), cases)[
        "matched_seed_id"
    ] == citation.seed_id
    assert exp.extract_citations("No citation here.") == []

    no_extract = exp.evaluate_raw_output(real, "No citation here.", cases)
    assert no_extract["extraction_success"] is False
    assert no_extract["taxonomy"] == "potential/ambiguous"


def test_req_verify_2932_run_uses_cached_pair_first_then_writes_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-2932: model selection tries cached_sota_pair before fallback."""

    cached_calls: list[dict[str, Any]] = []
    resolved_ids: list[str] = []

    def cached_pair(**kwargs: Any) -> None:
        cached_calls.append(kwargs)
        return None

    def resolver(hf_id: str) -> str | None:
        resolved_ids.append(hf_id)
        return "/tmp/gemma4-26b.gguf" if hf_id == MANDATED else None

    def fake_collection(
        spec: dict[str, Any],
        cases: list[exp.CitationCase],
        config: exp.ExperimentConfig,
    ) -> dict[str, Any]:
        rows = []
        config.response_dir().mkdir(parents=True, exist_ok=True)
        for index, case in enumerate(cases):
            output_text = _citation_text(case)
            raw_path = config.response_dir() / f"{case.case_id}.txt"
            raw_path.write_text(output_text, encoding="utf-8")
            rows.append(
                {
                    "case_id": case.case_id,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "model_path": spec["model_path"],
                    "gpu_index": spec["gpu"],
                    "prompt_hash": exp.sha256_text(exp.prompt_for_case(case)),
                    "per_case_seed": exp.RANDOM_SEED + index,
                    "generation_source": "fake_live_sota_llamacpp_citation",
                    "output_text": output_text,
                    "raw_response_path": str(raw_path),
                    "raw_response_sha256": exp.sha256_text(output_text),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
            )
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "model_used": True,
                "blocker": None,
                "live_inference_duration_s": 1.0,
            },
            "rows": rows,
        }

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            fixture_path=tmp_path / exp.FIXTURE_FILENAME,
            raw_response_dir=tmp_path / exp.RAW_RESPONSE_DIRNAME,
            started_at=10.0,
            clock=lambda: 13.0,
        ),
        cached_pair_provider=cached_pair,
        individual_model_resolver=resolver,
        collect_model_outputs_fn=fake_collection,
    )

    persisted = json.loads((tmp_path / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    fixture = json.loads((tmp_path / exp.FIXTURE_FILENAME).read_text(encoding="utf-8"))

    assert persisted == artifact
    assert cached_calls == [{"gpu_indices": (0, 1)}]
    assert MANDATED in resolved_ids
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["citation_verifier_ready"] is True
    assert artifact["random_seed"] == 2932
    assert artifact["models_used"] == [MANDATED]
    assert artifact["model_specs"][0]["hf_id"] == MANDATED
    assert artifact["fixture_path"].endswith(exp.FIXTURE_FILENAME)
    assert len(fixture["cases"]) == 30
    assert artifact["n_citation_cases"] == 30
    assert artifact["extraction_success_rate"] == pytest.approx(1.0)
    assert artifact["field_match_accuracy"] == pytest.approx(150 / 174)
    assert artifact["hallucination_detection_accuracy"] == pytest.approx(1.0)
    assert artifact["taxonomy_counts"] == {
        "real": 5,
        "potential/ambiguous": 0,
        "hallucinated-field": 24,
        "nonexistent-seed": 1,
    }
    assert len(artifact["per_case_results"]) == 30
    assert artifact["raw_response_dir"].endswith(exp.RAW_RESPONSE_DIRNAME)
    assert artifact["inference_substrate"] == "live_llm_inference_plus_deterministic_verifier"
    assert artifact["duration_s"] == pytest.approx(3.0)
    assert artifact["run_date"] == "20260523"
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_verify_2932_cached_pair_path_and_low_accuracy_verdict(tmp_path: Path) -> None:
    """REQ-VERIFY-2932: cached SOTA pair path and partial verdict are auditable."""

    cases = exp.build_citation_fixture()

    def noisy_collection(
        spec: dict[str, Any],
        fixture_cases: list[exp.CitationCase],
        config: exp.ExperimentConfig,
    ) -> dict[str, Any]:
        rows = []
        for index, case in enumerate(fixture_cases):
            output_text = _citation_text(case) if index % 2 == 0 else "No citation here."
            rows.append(
                {
                    "case_id": case.case_id,
                    "model_hf_id": spec["hf_id"],
                    "generation_source": "fake_noisy_collection",
                    "output_text": output_text,
                    "raw_response_path": "",
                }
            )
        return {
            "summary": {"hf_id": spec["hf_id"], "model_used": True, "blocker": None},
            "rows": rows,
        }

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            fixture_path=tmp_path / exp.FIXTURE_FILENAME,
            raw_response_dir=tmp_path / exp.RAW_RESPONSE_DIRNAME,
            started_at=0.0,
            clock=lambda: 1.0,
        ),
        cached_pair_provider=lambda **_: [_spec(), {"hf_id": "legacy/model", "model_path": "/tmp/x"}],
        individual_model_resolver=lambda _hf_id: pytest.fail("fallback resolver should not run"),
        collect_model_outputs_fn=noisy_collection,
    )

    assert cases[0].case_id == artifact["per_case_results"][0]["case_id"]
    assert artifact["models_used"] == [MANDATED]
    assert artifact["extraction_success_rate"] == pytest.approx(0.5)
    assert artifact["honest_verdict"] == "complete:citation_field_verifier_partial"


def test_req_verify_2932_missing_sota_cache_blocks_without_collection(tmp_path: Path) -> None:
    """REQ-VERIFY-2932: no cached mandated GGUF writes an honest blocked artifact."""

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            fixture_path=tmp_path / exp.FIXTURE_FILENAME,
            raw_response_dir=tmp_path / exp.RAW_RESPONSE_DIRNAME,
            started_at=1.0,
            clock=lambda: 1.5,
        ),
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=lambda *_args, **_kwargs: pytest.fail("collector must not run"),
    )

    assert artifact["honest_verdict"] == "blocked_sota_gguf_cache_missing"
    assert artifact["citation_verifier_ready"] is False
    assert artifact["models_used"] == []
    assert artifact["n_citation_cases"] == 30
    assert artifact["per_case_results"] == []
    assert artifact["extraction_success_rate"] == 0.0
    assert artifact["field_match_accuracy"] == 0.0
    assert artifact["hallucination_detection_accuracy"] == 0.0
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)


def test_scenario_verify_2932_llamacpp_collector_accepts_injected_backend(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2932: live collector captures raw text and hashes."""

    cases = exp.build_citation_fixture()[:1]

    class FakeLlama:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            assert "CITE[" in prompt
            assert kwargs["max_tokens"] == 160
            return {"choices": [{"text": _citation_text(cases[0])}]}

    result = exp.collect_model_outputs_llamacpp(
        _spec(),
        cases,
        exp.ExperimentConfig(raw_response_dir=tmp_path / "raw"),
        llama_cls=FakeLlama,
        monotonic_clock=lambda: 100.0,
    )

    row = result["rows"][0]
    assert result["summary"]["model_used"] is True
    assert result["summary"]["blocker"] is None
    assert row["case_id"] == cases[0].case_id
    assert row["raw_response_sha256"] == exp.sha256_text(_citation_text(cases[0]))
    assert Path(row["raw_response_path"]).read_text(encoding="utf-8") == _citation_text(cases[0])
    assert row["generation_source"] == "live_sota_llamacpp_citation"


def test_scenario_verify_2932_helper_edges_are_deterministic() -> None:
    """SCENARIO-VERIFY-2932: helper edges preserve stable deterministic behavior."""

    assert exp.aggregate_results([])["field_match_accuracy"] == 0.0
    assert exp.compute_reproducibility_checksum(
        fixture_cases=[{"case_id": "mapping-case"}],
        model_specs=[],
        raw_outputs=[],
    ) == exp.compute_reproducibility_checksum(
        fixture_cases=[{"case_id": "mapping-case"}],
        model_specs=[],
        raw_outputs=[],
    )
    assert exp._json_ready(exp.CitationFields(seed_id="x"))["seed_id"] == "x"  # noqa: SLF001
    assert exp._list_of_mappings(None) == []  # noqa: SLF001
    assert exp._citations_from_json_obj("not a mapping") == []  # noqa: SLF001
    assert exp._citations_from_json_obj({"irrelevant": True}) == []  # noqa: SLF001
    assert exp._llama_output_text({"choices": [{"message": {"content": "content"}}]}) == "content"  # noqa: SLF001
    assert exp._llama_output_text("plain text") == "plain text"  # noqa: SLF001
    assert exp._year_value(None) is None  # noqa: SLF001
    assert exp._honest_verdict(  # noqa: SLF001
        {"extraction_success_rate": 0.49, "hallucination_detection_accuracy": 1.0}
    ) == "complete:low_extraction_success"
