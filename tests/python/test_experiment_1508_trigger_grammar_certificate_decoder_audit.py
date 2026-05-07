"""Tests for Exp 1508 trigger+grammar certificate decoder audit.

Spec: REQ-VERIFY-1508, SCENARIO-VERIFY-1508.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu
from carnot.eval import cctu_trigger_certificate_export as exp1493
from carnot.verify import trigger_grammar_certificate_decoder as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ready_exp1507(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "complete",
            "verifier_induction_ready": True,
            "induction_manifest_path": "induction.jsonl",
            "honest_verdict": "complete: test gate ready",
        },
    )


def _selected_induction_manifest(path: Path) -> None:
    _write_jsonl(
        path,
        [
            {
                "row_type": "candidate",
                "candidate_name": "certificate_transcript_consistency",
                "compiled": True,
                "compiled_dsl": {
                    "kind": "safe_dsl_verifier",
                    "target": {"source": "certificate"},
                    "rules": [{"path": "parser_result.parsed", "op": "is_true"}],
                },
            },
            {
                "row_type": "selected_set_summary",
                "candidate_names": ["certificate_transcript_consistency"],
                "verifier_false_accept_rate": 0.0,
            },
        ],
    )


def _schema_rows(path: Path, cases: list[cctu.BenchmarkCase]) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        row = exp1493.build_manifest_row(
            case,
            {
                "case_id": case.case_id,
                "lane": exp1493.TRIGGER_LANE,
                "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                "generation_source": "live_sota_llamacpp",
                "output_text": exp1493.certificate_text_for_case(
                    case,
                    lane=exp1493.TRIGGER_LANE,
                    reasoning_text="schema-only baseline reasoning",
                ),
                "elapsed_seconds": 0.01,
                "blocker": None,
            },
        )
        rows.append(row)
    _write_jsonl(path, rows)


def test_req_verify_1508_exact_gbnf_and_parser_validate_cctu_certificate() -> None:
    """REQ-VERIFY-1508: grammar rows preserve trigger state and CCTU validation."""

    case = cctu.build_benchmark_cases()[0]
    grammar = exp.build_exact_certificate_gbnf(case)
    generation_row = {
        "case_id": case.case_id,
        "decoder_mode": exp.TRIGGER_GRAMMAR_MODE,
        "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
        "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
        "generation_source": "live_sota_llamacpp",
        "reasoning_text": f"I solved it.\n{exp.TRIGGER_TOKEN}",
        "certificate_body": json.dumps(exp1493.certificate_for_case(case), sort_keys=True),
        "grammar_backend": exp.EXACT_GBNF_BACKEND,
        "elapsed_seconds": 0.01,
        "blocker": None,
    }

    row = exp.build_grammar_manifest_row(case, generation_row)

    assert grammar.startswith("root ::= ")
    assert case.case_id in grammar
    assert row["decoder_mode"] == exp.TRIGGER_GRAMMAR_MODE
    assert row["trigger_token_present"] is True
    assert row["parser_result"]["parsed"] is True
    assert row["deterministic_validation_passed"] is True
    assert row["verifier_result"]["false_accept"] is False


def test_req_verify_1508_schema_rows_convert_and_metrics_compare_modes(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1508: aggregate rates compare trigger+grammar to schema-only rows."""

    cases = cctu.build_benchmark_cases()[:2]
    schema_manifest = tmp_path / "schema.jsonl"
    _schema_rows(schema_manifest, cases[:1])
    schema_rows = exp.load_schema_only_rows(schema_manifest, case_ids={cases[0].case_id})
    grammar_rows = [
        exp.build_grammar_manifest_row(
            cases[0],
            {
                "case_id": cases[0].case_id,
                "decoder_mode": exp.TRIGGER_GRAMMAR_MODE,
                "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                "generation_source": "live_sota_llamacpp",
                "reasoning_text": f"reason\n{exp.TRIGGER_TOKEN}",
                "certificate_body": json.dumps(
                    exp1493.certificate_for_case(cases[0]),
                    sort_keys=True,
                ),
                "grammar_backend": exp.EXACT_GBNF_BACKEND,
                "elapsed_seconds": 0.01,
                "blocker": None,
            },
        ),
        exp.build_grammar_manifest_row(
            cases[1],
            {
                "case_id": cases[1].case_id,
                "decoder_mode": exp.TRIGGER_GRAMMAR_MODE,
                "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                "generation_source": "live_sota_llamacpp",
                "reasoning_text": "missing trigger",
                "certificate_body": "",
                "grammar_backend": exp.EXACT_GBNF_BACKEND,
                "elapsed_seconds": 0.01,
                "blocker": "missing_trigger_token",
            },
        ),
    ]

    metrics = exp.aggregate_manifest_metrics([*schema_rows, *grammar_rows])

    assert schema_rows[0]["decoder_mode"] == exp.SCHEMA_ONLY_MODE
    assert schema_rows[0]["schema_source_lane"] == exp1493.TRIGGER_LANE
    assert metrics["trigger_token_presence_rate"] == pytest.approx(0.5)
    assert metrics["grammar_parse_rate"] == pytest.approx(0.5)
    assert metrics["schema_only_parse_rate"] == pytest.approx(1.0)
    assert metrics["grammar_validation_rate"] == pytest.approx(0.5)
    assert metrics["schema_only_validation_rate"] == pytest.approx(1.0)
    assert metrics["verifier_false_accept_rate"] == pytest.approx(0.0)


def test_scenario_verify_1508_runner_writes_ready_artifact_and_manifest(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1508: runner writes rows, metrics, and required fields."""

    cases = cctu.build_benchmark_cases()[:2]
    exp1507_path = tmp_path / "exp1507.json"
    induction_manifest = tmp_path / "induction.jsonl"
    schema_manifest = tmp_path / "schema.jsonl"
    output_path = tmp_path / "experiment_1508.json"
    decoder_manifest = tmp_path / "decoder_1508.jsonl"
    _ready_exp1507(exp1507_path)
    _selected_induction_manifest(induction_manifest)
    _schema_rows(schema_manifest, cases)

    def fake_collect(
        spec: dict[str, Any],
        selected_cases: list[cctu.BenchmarkCase],
    ) -> dict[str, Any]:
        rows = []
        for case in selected_cases:
            rows.append(
                {
                    "case_id": case.case_id,
                    "decoder_mode": exp.TRIGGER_GRAMMAR_MODE,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "generation_source": "live_sota_llamacpp",
                    "reasoning_text": f"Free solve for {case.case_id}.\n{exp.TRIGGER_TOKEN}",
                    "certificate_body": json.dumps(
                        exp1493.certificate_for_case(case),
                        sort_keys=True,
                    ),
                    "grammar_backend": exp.EXACT_GBNF_BACKEND,
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
            )
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_used": True,
                "blocker": None,
                "grammar_backend": exp.EXACT_GBNF_BACKEND,
            },
            "rows": rows,
        }

    artifact = exp.run_experiment(
        output_path=output_path,
        decoder_manifest_path=decoder_manifest,
        exp1507_artifact_path=exp1507_path,
        induction_manifest_path=induction_manifest,
        schema_only_manifest_path=schema_manifest,
        run_date="20260507",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_grammar_outputs_fn=fake_collect,
        gpu_probe_fn=lambda: {"nvidia_smi_available": True, "gpu_count": 1},
        max_cases=2,
        tests_run=["focused pytest"],
    )
    manifest_rows = [
        json.loads(line) for line in decoder_manifest.read_text(encoding="utf-8").splitlines()
    ]

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["certificate_decoder_ready"] is True
    assert artifact["gated_inputs_present"] is True
    assert artifact["cases_attempted"] == 2
    assert artifact["grammar_backend"] == exp.EXACT_GBNF_BACKEND
    assert artifact["trigger_token_presence_rate"] == pytest.approx(1.0)
    assert artifact["grammar_parse_rate"] == pytest.approx(1.0)
    assert artifact["schema_only_parse_rate"] == pytest.approx(1.0)
    assert artifact["grammar_validation_rate"] == pytest.approx(1.0)
    assert artifact["schema_only_validation_rate"] == pytest.approx(1.0)
    assert artifact["verifier_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["decoder_manifest_path"] == str(decoder_manifest)
    assert artifact["models_used"] == [exp.MANDATED_MODEL_SPECS[0]["hf_id"]]
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert [row["decoder_mode"] for row in manifest_rows] == [
        exp.SCHEMA_ONLY_MODE,
        exp.SCHEMA_ONLY_MODE,
        exp.TRIGGER_GRAMMAR_MODE,
        exp.TRIGGER_GRAMMAR_MODE,
    ]


def test_req_verify_1508_runner_blocks_when_exp1507_gate_is_not_ready(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1508: absent verifier induction gate writes a terminal artifact."""

    exp1507_path = tmp_path / "exp1507_blocked.json"
    _write_json(
        exp1507_path,
        {"status": "blocked", "verifier_induction_ready": False},
    )
    artifact = exp.run_experiment(
        output_path=tmp_path / "blocked.json",
        decoder_manifest_path=tmp_path / "blocked.jsonl",
        exp1507_artifact_path=exp1507_path,
        induction_manifest_path=tmp_path / "missing_induction.jsonl",
        schema_only_manifest_path=tmp_path / "missing_schema.jsonl",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_grammar_outputs_fn=lambda _spec, _cases: pytest.fail("gate must stop collection"),
        gpu_probe_fn=lambda: {"nvidia_smi_available": False, "gpu_count": 0},
    )

    assert artifact["status"] == "blocked"
    assert artifact["certificate_decoder_ready"] is False
    assert artifact["gated_inputs_present"] is False
    assert artifact["cases_attempted"] == 0
    assert artifact["blockers"] == [f"exp1507_not_ready:{exp1507_path}"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1508_live_collector_uses_llama_grammar_injection() -> None:
    """REQ-VERIFY-1508: live collector separates trigger reasoning from grammar tail."""

    class FakeGrammar:
        grammars: list[str] = []

        @classmethod
        def from_string(cls, grammar: str, verbose: bool = False) -> "FakeGrammar":
            del verbose
            cls.grammars.append(grammar)
            return cls()

    class FakeLlama:
        calls: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append({"prompt": prompt, "kwargs": kwargs, "init": self.kwargs})
            case_id = prompt.split("Case: ", 1)[1].split("\n", 1)[0]
            case = next(c for c in cctu.build_benchmark_cases() if c.case_id == case_id)
            if "grammar" not in kwargs:
                return {"choices": [{"text": f"Solved locally.\n{exp.TRIGGER_TOKEN}"}]}
            return {
                "choices": [
                    {
                        "text": json.dumps(
                            exp1493.certificate_for_case(case),
                            sort_keys=True,
                        )
                    }
                ]
            }

        def close(self) -> None:
            pass

    cases = cctu.build_benchmark_cases()[:1]
    collection = exp.collect_live_grammar_outputs(
        {**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"},
        cases,
        llama_importer=lambda: (True, FakeLlama, None),
        grammar_importer=lambda: (True, FakeGrammar, None),
        env_preparer=lambda: {},
    )
    missing_grammar = exp.collect_live_grammar_outputs(
        {**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"},
        cases,
        llama_importer=lambda: (True, FakeLlama, None),
        grammar_importer=lambda: (False, None, "grammar missing"),
        env_preparer=lambda: {},
    )

    assert collection["summary"]["model_used"] is True
    assert collection["rows"][0]["grammar_backend"] == exp.EXACT_GBNF_BACKEND
    assert collection["rows"][0]["reasoning_text"].endswith(exp.TRIGGER_TOKEN)
    assert len(FakeLlama.calls) == 2
    assert "grammar" not in FakeLlama.calls[0]["kwargs"]
    assert FakeLlama.calls[1]["kwargs"]["grammar"] is not None
    assert FakeGrammar.grammars[0].startswith("root ::= ")
    assert missing_grammar["summary"]["blocker"] == "grammar missing"
