"""REQ-REPORT-7646: Source features retain custody and honest unknowns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_7646_v667_source_feature_corpus as corpus


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def model_row(group: str, source: str, answer: str, role: str = "fit") -> dict:
    return {
        "component_hash": group,
        "role": role,
        "learning_partition": "fit_optimization",
        "complete_source": source,
        "source_sha256": digest(source),
        "complete_answer": answer,
        "answer_sha256": digest(answer),
        "answer_sentences": [
            {
                "text": answer,
                "byte_start": 0,
                "byte_end": len(answer.encode()),
                "text_sha256": digest(answer),
            }
        ],
    }


def test_req_report_7646_feature_arms_and_unknown_denominator() -> None:
    """SCENARIO-REPORT-7646-COVERAGE checks three arms without dropping prose."""
    source_a = "```python file=a.py\n1 | def f():\n2 |     return 1\n```"
    source_b = "```python file=b.py\n1 | def g():\n2 |     return 2\n```"
    rows = [
        model_row("a", source_a, "In `a.py`, `f` is defined at line 1."),
        model_row("b", source_b, "The function does something useful."),
    ]
    result = corpus.extract_role(rows, "fit", {"a": "b", "b": "a"})
    assert len(result) == 6
    original = result[0]
    assert original["unit_id"] == "a"
    assert original["arm"] == "original_source"
    assert original["checked_predicates"] == 1
    assert original["witnesses"][0]["status"] == "supported"
    assert original["source_sha256"] == digest(source_a)
    assert result[1]["arm"] == "evidence_erasure"
    assert result[1]["unknown_predicates"] == 1
    assert result[2]["arm"] == "within_role_derangement"
    assert result[2]["source_sha256"] == digest(source_b)
    assert result[3]["unknown_predicates"] == 1
    assert result[3]["unchecked_prose"] == 1
    assert all(row["denominator"] == 1 for row in result)


def test_req_report_7646_rejects_label_and_hash_drift() -> None:
    """SCENARIO-REPORT-7646-ISOLATION rejects evaluator fields in predictor rows."""
    row = model_row("a", "text", "claim")
    corpus.validate_model_row(row)
    with pytest.raises(ValueError, match="label"):
        corpus.validate_model_row({**row, "label": 1})
    with pytest.raises(ValueError, match="source_hash"):
        corpus.validate_model_row({**row, "source_sha256": "sha256:wrong"})
    with pytest.raises(ValueError, match="answer_offset"):
        corpus.validate_model_row(
            {**row, "answer_sentences": [{**row["answer_sentences"][0], "byte_start": 1}]}
        )


def test_req_report_7646_cold_replay_rejects_group_role_and_source_mutations(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7646-CUSTODY reconstructs exact rows from role inputs."""
    source = "```python file=a.py\n1 | def f():\n2 |     pass\n```"
    source2 = "```python file=b.py\n1 | def g():\n2 |     pass\n```"
    inputs = [model_row("a", source, "In `a.py`, `f` exists."), model_row("b", source2, "x")]
    model_path = tmp_path / "fit_model_inputs.jsonl"
    model_path.write_text("".join(json.dumps(r) + "\n" for r in inputs))
    rows = corpus.extract_role(inputs, "fit", {"a": "b", "b": "a"})
    features = tmp_path / "fit_features.jsonl"
    features.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        "roles": {
            "fit": {
                "input_path": str(model_path),
                "feature_path": str(features),
                "input_sha256": corpus.sha256_file(model_path),
                "feature_sha256": corpus.sha256_file(features),
                "group_ids": ["a", "b"],
            }
        },
        "derangement": {"a": "b", "b": "a"},
    }
    corpus.cold_reconstruct(manifest, tmp_path)
    for mutation in (
        lambda m: m["roles"]["fit"]["group_ids"].append("a"),
        lambda m: m["roles"]["fit"].update(input_sha256="sha256:wrong"),
        lambda m: m["roles"]["fit"].update(group_ids=["b", "a"]),
    ):
        changed = json.loads(json.dumps(manifest))
        mutation(changed)
        with pytest.raises(ValueError):
            corpus.cold_reconstruct(changed, tmp_path)
    changed = json.loads(json.dumps(manifest))
    changed["roles"]["tune"] = changed["roles"].pop("fit")
    with pytest.raises(ValueError):
        corpus.cold_reconstruct(changed, tmp_path)


def test_req_report_7646_real_protocol_custody() -> None:
    """SCENARIO-REPORT-7646-CUSTODY validates all inherited sidecar hashes."""
    authenticated = corpus.authenticate_inputs(corpus.ROOT)
    assert not [check for check in authenticated["checks"] if not check["passed"]]
    assert authenticated["protocol"]["scored_group_count"] == 240
    assert authenticated["protocol"]["pilot_group_count"] == 8
