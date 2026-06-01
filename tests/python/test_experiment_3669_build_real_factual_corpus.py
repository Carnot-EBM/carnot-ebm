"""Tests for Exp 3669 real factual corpus builder.

Spec refs: REQ-REPORT-3669, SCENARIO-REPORT-3669,
SCENARIO-REPORT-3669-DEGENERATE, SCENARIO-REPORT-3669-BLOCKED.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import real_factual_corpus_ragtruth_3669 as mod


def _ragtruth_payload(
    *,
    n_sources: int = 120,
    benchmark: str = "RAGTruth",
) -> mod.BenchmarkPayload:
    source_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    places = [
        ("Evergreen University", "Lake Clara", "1876", "Solstice Hall"),
        ("Harborview Museum", "Mason Shipyard", "1912", "North Gallery"),
        ("Redwood Observatory", "Mount Talus", "1964", "Cedar Dome"),
        ("Civic Botanical Archive", "Elena Park", "1938", "Glasshouse Wing"),
        ("Pioneer Transit Center", "Market Square", "2004", "Blue Line Depot"),
    ]
    for idx in range(n_sources):
        name, location, year, feature = places[idx % len(places)]
        source_id = f"source-{idx:04d}"
        source_rows.append(
            {
                "source_id": source_id,
                "task_type": "QA",
                "source": "local-encyclopedia",
                "source_info": {
                    "question": f"Where is {name} located and when did it open?",
                    "passages": (
                        f"{name} is located beside {location}. The public records "
                        f"state that it opened in {year}. The site includes {feature} "
                        "and hosts community research visits each spring."
                    ),
                },
                "prompt": (
                    f"Briefly answer using only the provided passages: Where is {name} "
                    "located and when did it open?"
                ),
            }
        )
        response_rows.extend(
            [
                {
                    "id": f"resp-{idx:04d}-a",
                    "source_id": source_id,
                    "model": "gpt-4-0613",
                    "temperature": 0.3,
                    "quality": "good",
                    "split": "test",
                    "labels": [],
                    "response": (
                        f"{name} is located beside {location} and opened in {year}."
                    ),
                },
                {
                    "id": f"resp-{idx:04d}-b",
                    "source_id": source_id,
                    "model": "llama-2-7b-chat",
                    "temperature": 0.9,
                    "quality": "good",
                    "split": "test",
                    "labels": [
                        {
                            "start": 42,
                            "end": 58,
                            "text": "opened in 1999",
                            "label_type": "Evident Conflict",
                            "implicit_true": False,
                        }
                    ],
                    "response": (
                        f"{name} is next to {location}, but it opened in 1999 and "
                        f"was designed around {feature}."
                    ),
                },
            ]
        )
    return mod.BenchmarkPayload(
        benchmark=benchmark,
        version="fixture-2026-06-01",
        responses=response_rows,
        sources=source_rows,
        source_urls=("fixture://ragtruth-response", "fixture://ragtruth-source"),
        from_cache=False,
    )


def _headroom_confidence(row: dict[str, Any]) -> float:
    """SCENARIO-REPORT-3669: confidence has signal but not perfect separation."""
    if row["model"] == "gpt-4-0613":
        year = row["answer"].rsplit(" ", 1)[-1].strip(".")
        return {"1876": 0.91, "1912": 0.78, "1964": 0.63, "1938": 0.49, "2004": 0.35}[year]
    source_name = row["question"].split(" located", 1)[0].replace("Where is ", "")
    return {
        "Evergreen University": 0.22,
        "Harborview Museum": 0.38,
        "Redwood Observatory": 0.54,
        "Civic Botanical Archive": 0.70,
        "Pioneer Transit Center": 0.86,
    }[source_name]


def _perfect_confidence(row: dict[str, Any]) -> float:
    """SCENARIO-REPORT-3669-DEGENERATE: confidence is a perfect detector."""
    return 0.99 if int(row["is_hallucination"]) == 0 else 0.01


def test_req_report_3669_spec_anchor_declares_real_corpus_builder() -> None:
    """REQ-REPORT-3669: OpenSpec declares the real corpus builder first."""
    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )
    assert "REQ-REPORT-3669" in spec
    assert "SCENARIO-REPORT-3669" in spec
    assert "SCENARIO-REPORT-3669-DEGENERATE" in spec
    assert "SCENARIO-REPORT-3669-BLOCKED" in spec
    assert "experiment_3669_build_real_factual_corpus.json" in spec


@pytest.mark.parametrize(
    (
        "case_name",
        "payload",
        "network_ok",
        "confidence_fn",
        "expected_verdict",
        "expected_built",
        "expected_non_degenerate",
    ),
    [
        pytest.param(
            "corpus_built",
            _ragtruth_payload(),
            True,
            _headroom_confidence,
            mod.VERDICT_RAGTRUTH_NON_DEGENERATE,
            True,
            True,
            id="corpus_built",
        ),
        pytest.param(
            "degenerate_confidence_perfect",
            _ragtruth_payload(),
            True,
            _perfect_confidence,
            mod.VERDICT_DEGENERATE_CONFIDENCE,
            False,
            False,
            id="degenerate_confidence_perfect",
        ),
        pytest.param(
            "blocked",
            None,
            False,
            _headroom_confidence,
            mod.VERDICT_BLOCKED_NO_CORPUS,
            False,
            False,
            id="blocked",
        ),
    ],
)
def test_scenarios_report_3669_parametrize_honest_outcomes(
    tmp_path: Path,
    case_name: str,
    payload: mod.BenchmarkPayload | None,
    network_ok: bool,
    confidence_fn: mod.ConfidenceFn,
    expected_verdict: str,
    expected_built: bool,
    expected_non_degenerate: bool,
) -> None:
    """SCENARIO-REPORT-3669: built, degenerate, and blocked outcomes stay honest."""
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        source_loader=lambda: payload,
        network_checker=lambda: network_ok,
        confidence_fn=confidence_fn,
        duration_s=2.5,
    )

    mod.validate_artifact(artifact)
    assert case_name in {"corpus_built", "degenerate_confidence_perfect", "blocked"}
    assert artifact["honest_verdict"] == expected_verdict
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["real_factual_corpus_built"] is expected_built
    assert artifact["corpus_non_degenerate"] is expected_non_degenerate
    assert type(artifact["real_factual_corpus_built"]) is bool
    assert type(artifact["corpus_non_degenerate"]) is bool
    assert artifact["duration_s"] == 2.5

    result_path = tmp_path / mod.OUTPUT_REL_PATH
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact

    corpus_path = tmp_path / mod.CORPUS_REL_PATH
    if case_name == "blocked":
        assert artifact["n_examples"] == 0
        assert artifact["class_balance"] == {"correct": 0.0, "hallucinated": 0.0}
        assert not corpus_path.exists()
        return

    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 240
    assert {int(row["is_hallucination"]) for row in rows} == {0, 1}
    assert all(set(row) == set(mod.REQUIRED_CORPUS_FIELDS) for row in rows)
    assert all("Evergreen University" not in row["evidence_passage"] or "Lake Clara" in row["evidence_passage"] for row in rows)
    assert artifact["sample_examples"][0]["answer"].startswith("Evergreen University")
    assert "placeholder" not in artifact["sample_examples"][0]["answer"].lower()

    if case_name == "corpus_built":
        assert artifact["acceptance_gate"]["passed"] is True
        assert artifact["source_benchmark"].startswith("RAGTruth")
        assert 0.0 < artifact["confidence_baseline_auroc"] < 0.95
        assert artifact["class_balance"]["correct"] >= 0.20
        assert artifact["class_balance"]["hallucinated"] >= 0.20
    if case_name == "degenerate_confidence_perfect":
        assert artifact["acceptance_gate"]["passed"] is False
        assert artifact["confidence_baseline_auroc"] >= 0.95


def test_scenario_report_3669_fallback_benchmark_gets_fallback_verdict(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3669: FELM/HaluEval fallback provenance is explicit."""
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        source_loader=lambda: _ragtruth_payload(benchmark="HaluEval"),
        network_checker=lambda: True,
        confidence_fn=_headroom_confidence,
        duration_s=1.25,
    )

    assert artifact["honest_verdict"] == mod.VERDICT_FALLBACK_NON_DEGENERATE
    assert artifact["real_factual_corpus_built"] is True
    assert artifact["source_benchmark"].startswith("HaluEval")


def test_req_report_3669_projection_filters_quality_placeholders_and_implicit_true() -> None:
    """REQ-REPORT-3669: projection uses human labels and rejects poisoned rows."""
    payload = _ragtruth_payload(n_sources=1)
    payload.responses.extend(
        [
            {
                "id": "bad-quality",
                "source_id": "source-0000",
                "model": "gpt-3.5-turbo-0613",
                "temperature": 0.7,
                "quality": "truncated",
                "labels": [],
                "response": "This truncated answer should not enter the corpus.",
            },
            {
                "id": "implicit-true",
                "source_id": "source-0000",
                "model": "gpt-3.5-turbo-0613",
                "temperature": 0.7,
                "quality": "good",
                "labels": [{"text": "community visits", "implicit_true": True}],
                "response": "Evergreen University also hosts community research visits.",
            },
            {
                "id": "placeholder",
                "source_id": "source-0000",
                "model": "gpt-3.5-turbo-0613",
                "temperature": 0.7,
                "quality": "good",
                "labels": [],
                "response": "R17",
            },
        ]
    )

    records = mod.project_ragtruth_payload(payload, confidence_fn=lambda _row: 0.5)

    assert [record["is_hallucination"] for record in records] == [0, 1, 0]
    assert all(record["answer"] != "R17" for record in records)
    assert all("truncated answer" not in record["answer"] for record in records)
    assert records[-1]["answer"].startswith("Evergreen University also hosts")


def test_req_report_3669_default_loader_network_cache_and_fallback_paths(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3669: source loading prefers RAGTruth, then cache, then HaluEval."""
    payload = _ragtruth_payload(n_sources=1)
    response_text = "\n".join(json.dumps(row) for row in payload.responses) + "\n"
    source_text = "\n".join(json.dumps(row) for row in payload.sources) + "\n"

    def fetcher(url: str) -> str:
        if url == mod.RAGTRUTH_RESPONSE_URL:
            return response_text
        if url == mod.RAGTRUTH_SOURCE_URL:
            return source_text
        raise AssertionError(url)

    loaded = mod.default_source_loader(tmp_path, network_ok=True, fetcher=fetcher)
    assert loaded is not None
    assert loaded.benchmark == "RAGTruth"
    assert loaded.from_cache is False
    assert (tmp_path / mod.RAGTRUTH_RESPONSE_CACHE).is_file()
    assert (tmp_path / mod.RAGTRUTH_SOURCE_CACHE).is_file()

    cached = mod.default_source_loader(
        tmp_path,
        network_ok=False,
        fetcher=lambda _url: pytest.fail("cache path should not fetch"),
    )
    assert cached is not None
    assert cached.from_cache is True
    assert cached.responses[0]["id"].startswith("resp-")

    cache_root = tmp_path / "halueval-only"
    fallback_row = {
        "question": "Which city is the capital of France?",
        "answer": "Paris is the capital of France.",
        "is_hallucination": 0,
        "evidence_passage": "France's capital city is Paris.",
        "model_confidence": 0.72,
    }
    fallback_path = cache_root / mod.LOCAL_HALUEVAL_V3_PATH
    fallback_path.parent.mkdir(parents=True)
    fallback_path.write_text(json.dumps(fallback_row) + "\n", encoding="utf-8")
    fallback = mod.default_source_loader(cache_root, network_ok=False, fetcher=lambda _url: "")
    assert fallback is not None
    assert fallback.benchmark == "HaluEval"
    assert fallback.responses == [fallback_row]
    assert mod.read_jsonl_text('\n{"kept": true}\n["ignored"]\n') == [{"kept": True}]

    assert mod.default_source_loader(tmp_path / "empty", network_ok=False, fetcher=lambda _url: "") is None


def test_req_report_3669_default_loader_fetch_failure_uses_cache(tmp_path: Path) -> None:
    """REQ-REPORT-3669: failed primary fetch falls through to cached RAGTruth."""
    payload = _ragtruth_payload(n_sources=1)
    (tmp_path / mod.RAGTRUTH_RESPONSE_CACHE).parent.mkdir(parents=True)
    (tmp_path / mod.RAGTRUTH_RESPONSE_CACHE).write_text(
        "\n".join(json.dumps(row) for row in payload.responses) + "\n",
        encoding="utf-8",
    )
    (tmp_path / mod.RAGTRUTH_SOURCE_CACHE).write_text(
        "\n".join(json.dumps(row) for row in payload.sources) + "\n",
        encoding="utf-8",
    )

    def broken_fetch(_url: str) -> str:
        raise OSError("offline")

    loaded = mod.default_source_loader(tmp_path, network_ok=True, fetcher=broken_fetch)
    assert loaded is not None
    assert loaded.benchmark == "RAGTruth"
    assert loaded.from_cache is True


def test_req_report_3669_projection_edges_and_default_confidence() -> None:
    """REQ-REPORT-3669: projection handles source variants without label leakage."""
    payload = mod.BenchmarkPayload(
        benchmark="RAGTruth",
        version="fixture-edge",
        sources=[
            {
                "source_id": "plain",
                "source_info": "Madrid is the capital city of Spain.",
                "prompt": "Question: Which city is Spain's capital?",
            },
            {
                "source_id": "context",
                "source_info": {"context": "Rome is the capital city of Italy."},
                "prompt": "Question: Which city is Italy's capital?",
            },
            {
                "source_id": "document",
                "source_info": {"document": "Berlin is the capital city of Germany."},
                "prompt": "Question: Which city is Germany's capital?",
            },
            {
                "source_id": "dict",
                "source_info": {"note": "Lisbon is the capital city of Portugal."},
                "prompt": "Question: Which city is Portugal's capital?",
            },
        ],
        responses=[
            {
                "source_id": "plain",
                "response": "Madrid is Spain's capital.",
                "labels": "not-a-list",
                "model": "gpt-3.5-turbo-0613",
                "temperature": "not-a-number",
            },
            {
                "source_id": "context",
                "response": "Milan is Italy's capital.",
                "labels": ["span-label"],
                "model": "llama-2-70b-chat",
                "temperature": float("nan"),
            },
            {
                "source_id": "document",
                "response": "Berlin is likely Germany's capital, according to the record.",
                "labels": [],
                "model": "llama-2-13b-chat",
                "temperature": 0.4,
            },
            {
                "source_id": "dict",
                "response": "Lisbon is Portugal's capital.",
                "labels": [],
                "model": "mistral-7B-instruct",
                "temperature": 0.6,
            },
            {
                "source_id": "missing",
                "response": "This row lacks a source.",
                "labels": [],
                "model": "unknown-model",
            },
            {
                "source_id": "plain",
                "response": "",
                "labels": [],
                "model": "gpt-4-0613",
            },
        ],
        source_urls=(),
    )

    records = mod.project_ragtruth_payload(payload)

    assert mod._question_from_prompt("Standalone prompt with no question marker.") == (
        "Standalone prompt with no question marker."
    )
    assert [row["is_hallucination"] for row in records] == [0, 1, 0, 0]
    assert records[0]["question"] == "Which city is Spain's capital?"
    assert records[0]["evidence_passage"] == "Madrid is the capital city of Spain."
    assert records[3]["evidence_passage"].startswith("{")
    assert all(0.0 <= row["model_confidence"] <= 1.0 for row in records)
    assert mod.default_model_confidence(
        {"model": "gpt-4-0613", "answer": "A concise supported answer.", "temperature": 0.2}
    ) > mod.default_model_confidence(
        {"model": "unknown", "answer": " ".join(["verbose"] * 400), "temperature": 1.0}
    )
    assert mod.default_model_confidence(
        {
            "model": "llama-2-7b-chat",
            "answer": "It may be around the date suggested by the archive.",
            "temperature": 0.7,
        }
    ) > 0.0


def test_req_report_3669_existing_v3_projection_and_rejection_edges() -> None:
    """REQ-REPORT-3669: cached v3 fallback rows are normalized or rejected."""
    payload = mod.BenchmarkPayload(
        benchmark="HaluEval",
        version="fixture-v3",
        responses=[
            {
                "question": "Which city is the capital of Japan?",
                "answer": "Tokyo is the capital of Japan.",
                "is_hallucination": False,
                "evidence_passage": "Japan's capital city is Tokyo.",
                "model_confidence": 0.8,
            },
            {
                "question": "",
                "answer": "Paris.",
                "is_hallucination": 0,
                "evidence_passage": "France's capital city is Paris.",
                "model_confidence": 0.8,
            },
            {
                "question": "Which city is the capital of Italy?",
                "answer": "Q7",
                "is_hallucination": 0,
                "evidence_passage": "Italy's capital city is Rome.",
                "model_confidence": 0.8,
            },
        ],
        sources=[],
    )

    records = mod.project_ragtruth_payload(payload, confidence_fn=lambda _row: 1.7)
    assert records == [
        {
            "question": "Which city is the capital of Japan?",
            "answer": "Tokyo is the capital of Japan.",
            "is_hallucination": 0,
            "evidence_passage": "Japan's capital city is Tokyo.",
            "model_confidence": 1.0,
        }
    ]


def test_req_report_3669_metrics_checksum_and_validation_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3669: AUROC, checksums, examples, and validators are explicit."""
    assert mod.binary_auroc([], []) == 0.0
    assert mod.binary_auroc([0, 1], [0.8, 0.2]) == 0.0
    assert mod.binary_auroc([0, 1], [0.2, 0.8]) == 1.0
    assert mod.binary_auroc([0, 1, 1], [0.5, 0.5, 0.8]) == 0.75
    with pytest.raises(ValueError, match="same length"):
        mod.binary_auroc([0], [0.1, 0.2])
    assert mod.has_placeholder_token("H19") is True
    assert mod.has_placeholder_token("Harborview Museum opened in 1912.") is False

    payload = _ragtruth_payload(n_sources=3)
    records = mod.project_ragtruth_payload(payload, confidence_fn=_headroom_confidence)
    validation = mod.validate_records(records, min_examples=6)
    assert validation.n_examples == 6
    assert validation.class_balance["correct"] == pytest.approx(0.5)
    assert validation.class_balance["hallucinated"] == pytest.approx(0.5)
    assert validation.placeholder_tokens_rejected is True

    examples = mod.sample_examples(records, limit=2)
    assert len(examples) == 2
    assert examples[0]["question"].startswith("Where is")
    assert mod.reproducibility_checksum({"records": records}) == mod.reproducibility_checksum(
        {"records": records}
    )

    bad_records = [
        {
            "question": "Which city is the capital of Spain?",
            "answer": "Madrid",
            "is_hallucination": 0,
            "evidence_passage": "Spain's capital city is Madrid.",
            "model_confidence": 0.9,
            "extra": "invalid",
        },
        {
            "question": "Which city is the capital of France?",
            "answer": "R44",
            "is_hallucination": 0,
            "evidence_passage": "",
            "model_confidence": 0.8,
        },
    ]
    bad_validation = mod.validate_records(bad_records, min_examples=3)
    assert bad_validation.corpus_non_degenerate is False
    assert set(bad_validation.degeneracy_reasons) == {
        "n_examples_below_200",
        "class_balance_below_20_percent",
        "schema_invalid_or_missing_text",
        "placeholder_tokens_present",
    }

    invalid = {
        "honest_verdict": "complete: invalid",
        "real_factual_corpus_built": {"value": True},
    }
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(invalid)

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        source_loader=lambda: None,
        network_checker=lambda: False,
        duration_s=0.0,
    )
    bad_bool = dict(artifact, real_factual_corpus_built={"value": False})
    with pytest.raises(ValueError, match="bare boolean"):
        mod.validate_artifact(bad_bool)
    bad_verdict = dict(artifact, honest_verdict="blocked")
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)
    bad_principles = dict(artifact, field_principles=None)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)
    incomplete_principles = dict(artifact, field_principles={})
    with pytest.raises(ValueError, match="field_principles missing"):
        mod.validate_artifact(incomplete_principles)
    negative_n = dict(artifact, n_examples=-1)
    with pytest.raises(ValueError, match="n_examples"):
        mod.validate_artifact(negative_n)
    bad_confidence = dict(artifact, confidence_baseline_auroc=1.5)
    with pytest.raises(ValueError, match="confidence_baseline_auroc"):
        mod.validate_artifact(bad_confidence)
    bad_duration = dict(artifact, duration_s=-0.1)
    with pytest.raises(ValueError, match="duration_s"):
        mod.validate_artifact(bad_duration)


def test_req_report_3669_run_experiment_returns_output_path(tmp_path: Path) -> None:
    """REQ-REPORT-3669: run_experiment writes the terminal artifact path."""
    out_path = mod.run_experiment(
        repo_root=tmp_path,
        source_loader=lambda: None,
        network_checker=lambda: False,
        duration_s=0.0,
    )

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert json.loads(out_path.read_text(encoding="utf-8"))["honest_verdict"] == (
        mod.VERDICT_BLOCKED_NO_CORPUS
    )


def test_scenario_report_3669_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3669: conductor entrypoint script delegates to the module."""
    script = Path("scripts/experiment_3669_build_real_factual_corpus.py")
    text = script.read_text(encoding="utf-8") if script.exists() else ""
    assert "real_factual_corpus_ragtruth_3669" in text
    assert "main" in text
    spec = importlib.util.spec_from_file_location("exp3669_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main is mod.main
