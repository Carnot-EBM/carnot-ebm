"""Tests for Exp 1507 safe-DSL verifier induction pack.

Spec: REQ-VERIFY-1507, SCENARIO-VERIFY-1507.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import safe_dsl_verifier_induction as exp


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sample_manifest_paths(tmp_path: Path) -> tuple[Path, Path]:
    certificate_path = tmp_path / "certificates.jsonl"
    validator_path = tmp_path / "validators.jsonl"
    _write_jsonl(
        certificate_path,
        [
            {
                "case_id": "cctu-good",
                "family": "arithmetic",
                "lane": "trigger_certificate",
                "deterministic_validation_passed": True,
                "parser_result": {"parsed": True},
                "validator_result": {
                    "case_id_valid": True,
                    "final_answer_valid": True,
                    "tool_call_structure_valid": True,
                    "tool_result_consistent": True,
                    "verifier_outcome_valid": True,
                },
                "verifier_result": {"base_valid": True, "false_accept": False},
            },
            {
                "case_id": "cctu-bad",
                "family": "arithmetic",
                "lane": "trigger_certificate",
                "deterministic_validation_passed": False,
                "parser_result": {"parsed": True},
                "validator_result": {
                    "case_id_valid": True,
                    "final_answer_valid": False,
                    "tool_call_structure_valid": True,
                    "tool_result_consistent": True,
                    "verifier_outcome_valid": False,
                },
                "verifier_result": {"base_valid": False, "false_accept": False},
            },
        ],
    )
    _write_jsonl(
        validator_path,
        [
            {
                "prompt_id": "validator-good",
                "family": "json_schema",
                "validator_compiled": True,
                "known_good_passed": True,
                "known_bad_rejected": True,
                "false_accept": False,
                "false_reject": False,
                "compiled_validator": {"kind": "json_schema"},
            },
            {
                "prompt_id": "validator-bad",
                "family": "json_schema",
                "validator_compiled": True,
                "known_good_passed": True,
                "known_bad_rejected": False,
                "false_accept": True,
                "false_reject": False,
                "compiled_validator": {"kind": "json_schema"},
            },
        ],
    )
    return certificate_path, validator_path


def _certificate_candidate() -> dict[str, Any]:
    return {
        "name": "certificate_transcript_consistency",
        "kind": "safe_dsl_verifier",
        "target": {"source": "certificate"},
        "rules": [
            {"path": "parser_result.parsed", "op": "is_true"},
            {"path": "validator_result.case_id_valid", "op": "is_true"},
            {"path": "validator_result.final_answer_valid", "op": "is_true"},
            {"path": "validator_result.tool_call_structure_valid", "op": "is_true"},
            {"path": "validator_result.tool_result_consistent", "op": "is_true"},
            {"path": "validator_result.verifier_outcome_valid", "op": "is_true"},
        ],
    }


def _validator_candidate() -> dict[str, Any]:
    return {
        "name": "compiled_validator_sanity",
        "kind": "safe_dsl_verifier",
        "target": {"source": "validator"},
        "rules": [
            {"path": "validator_compiled", "op": "is_true"},
            {"path": "known_good_passed", "op": "is_true"},
            {"path": "known_bad_rejected", "op": "is_true"},
        ],
    }


def test_req_verify_1507_loads_labeled_certificate_and_validator_rows(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1507: manifests become labeled rows for false-accept accounting."""

    certificate_path, validator_path = _sample_manifest_paths(tmp_path)
    loaded = exp.load_labeled_rows(
        certificate_manifest_path=certificate_path,
        validator_manifest_path=validator_path,
    )

    assert loaded.blockers == []
    assert len(loaded.rows) == 4
    assert [row.row_id for row in loaded.rows] == [
        "certificate:cctu-good:trigger_certificate:0",
        "certificate:cctu-bad:trigger_certificate:1",
        "validator:validator-good:0",
        "validator:validator-bad:1",
    ]
    assert sum(row.label_accept for row in loaded.rows) == 2
    assert exp.baseline_validator_coverage_rate(loaded.rows) == pytest.approx(0.5)


def test_req_verify_1507_safe_dsl_compiler_rejects_python_and_unsafe_io() -> None:
    """REQ-VERIFY-1507: generated Python, imports, eval/exec, and filesystem access fail closed."""

    unsafe_outputs = [
        "import os\n" + json.dumps(_certificate_candidate()),
        json.dumps({**_certificate_candidate(), "description": "call eval('1+1')"}),
        json.dumps({**_certificate_candidate(), "description": "open('/tmp/x').read()"}),
        json.dumps({**_certificate_candidate(), "description": "requests.get('https://x')"}),
        json.dumps({**_certificate_candidate(), "description": "random.random()"}),
    ]

    for output in unsafe_outputs:
        compiled = exp.compile_candidate_from_model_output(output)
        assert compiled.compiled is False
        assert "unsafe" in str(compiled.failure_reason)


def test_req_verify_1507_compiler_and_predicate_fail_closed_edges(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1507: malformed DSL shapes and nonmatching predicates stay closed."""

    certificate_path, validator_path = _sample_manifest_paths(tmp_path)
    rows = exp.load_labeled_rows(
        certificate_manifest_path=certificate_path,
        validator_manifest_path=validator_path,
    ).rows
    base = _certificate_candidate()
    malformed_candidates = [
        {"name": "bad-kind", "kind": "python"},
        {**base, "target": "certificate"},
        {**base, "target": {"source": "certificate", "path": "/tmp"}},
        {**base, "target": {"source": ""}},
        {**base, "rules": []},
        {**base, "rules": ["not an object"]},
        {**base, "rules": [{"path": "row_id", "op": "exists", "extra": True}]},
        {**base, "rules": [{"path": "label_accept", "op": "exists"}]},
        {**base, "rules": [{"path": "row_id", "op": "regex"}]},
        {**base, "rules": [{"path": "row_id", "op": "equals"}]},
        {**base, "rules": [{"path": "row_id", "op": "equals", "value": {"bad": "dict"}}]},
    ]

    assert exp.parse_candidate_proposals("no json here") == []
    assert exp.compile_candidate_from_model_output("no json here").failure_reason == (
        "no_json_candidate_object"
    )
    leading_counts = (
        "{not-json}\n"
        '{"certificate": 40, "validator": 30}\n'
        + json.dumps({"candidates": [_certificate_candidate()]})
    )
    assert exp.compile_candidate_from_model_output(leading_counts).compiled is True
    assert [exp.compile_candidate(candidate).compiled for candidate in malformed_candidates] == [
        False
    ] * len(malformed_candidates)

    uncompiled = exp.compile_candidate(malformed_candidates[0])
    assert exp.score_candidate(uncompiled, rows)["compiled"] is False
    assert exp.candidate_accepts_row(uncompiled, rows[0]) is False

    equals_not_null_and_false = {
        "name": "equals_not_null_false",
        "kind": "safe_dsl_verifier",
        "target": {"source": "certificate"},
        "rules": [
            {"path": "source", "op": "equals", "value": "certificate"},
            {"path": "family", "op": "not_null"},
            {"path": "verifier_result.false_accept", "op": "is_false"},
        ],
    }
    missing_path = {
        "name": "missing_trigger_field",
        "kind": "safe_dsl_verifier",
        "target": {"source": "validator"},
        "rules": [{"path": "trigger_token_present", "op": "is_true"}],
    }

    assert exp.candidate_accepts_row(exp.compile_candidate(equals_not_null_and_false), rows[0])
    assert not exp.candidate_accepts_row(exp.compile_candidate(missing_path), rows[2])
    assert exp._rule_accepts(True, {"op": "unsupported"}) is False
    assert exp._unsafe_reason(["safe", {"nested": "exec('x')"}]).startswith("unsafe")
    assert exp._canonical({"b": (2, 3), "a": [1]}) == {"a": [1], "b": [2, 3]}


def test_scenario_verify_1507_search_selects_compact_zero_false_accept_set(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1507: set search maximizes coverage with zero false accepts."""

    certificate_path, validator_path = _sample_manifest_paths(tmp_path)
    rows = exp.load_labeled_rows(
        certificate_manifest_path=certificate_path,
        validator_manifest_path=validator_path,
    ).rows
    broad_bad = {
        "name": "accepts_every_json_row",
        "kind": "safe_dsl_verifier",
        "target": {"source": "*"},
        "rules": [{"path": "row_id", "op": "exists"}],
    }
    compiled = [
        exp.compile_candidate(_certificate_candidate()),
        exp.compile_candidate(_validator_candidate()),
        exp.compile_candidate(broad_bad),
    ]
    scores = [exp.score_candidate(candidate, rows) for candidate in compiled]
    selected = exp.search_compact_verifier_set(scores, rows)

    assert selected["candidate_names"] == [
        "certificate_transcript_consistency",
        "compiled_validator_sanity",
    ]
    assert selected["verifier_set_size"] == 2
    assert selected["verifier_coverage_rate"] == pytest.approx(1.0)
    assert selected["verifier_false_accept_rate"] == pytest.approx(0.0)

    inconsistent = exp.search_compact_verifier_set(
        [
            {
                "name": "inconsistent",
                "compiled": True,
                "true_accept_count": 1,
                "false_accept_count": 0,
                "true_accept_row_ids": ["certificate:cctu-good:trigger_certificate:0"],
                "false_accept_row_ids": ["validator:validator-bad:1"],
            }
        ],
        rows,
    )
    assert inconsistent["candidate_names"] == []


def test_scenario_verify_1507_runner_writes_manifest_and_ready_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1507: runner writes candidate rows, summary, and terminal artifact."""

    certificate_path, validator_path = _sample_manifest_paths(tmp_path)

    def fake_collect(spec: dict[str, Any], rows: list[exp.LabeledVerifierRow]) -> dict[str, Any]:
        assert len(rows) == 4
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_used": True,
                "blocker": None,
            },
            "rows": [
                {
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "generation_source": "live_sota_llamacpp",
                    "output_text": json.dumps(
                        {"candidates": [_certificate_candidate(), _validator_candidate()]},
                        sort_keys=True,
                    ),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
            ],
        }

    output_path = tmp_path / "experiment_1507.json"
    induction_manifest_path = tmp_path / "induction_1507.jsonl"
    artifact = exp.run_experiment(
        output_path=output_path,
        induction_manifest_path=induction_manifest_path,
        certificate_manifest_path=certificate_path,
        validator_manifest_path=validator_path,
        run_date="20260507",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_candidate_proposals_fn=fake_collect,
        gpu_probe_fn=lambda: {"nvidia_smi_available": True, "gpu_count": 1},
        tests_run=["focused pytest"],
    )
    manifest_rows = [
        json.loads(line) for line in induction_manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["verifier_induction_ready"] is True
    assert artifact["labeled_rows_loaded"] == 4
    assert artifact["candidate_verifiers_proposed"] == 2
    assert artifact["candidate_verifiers_compiled"] == 2
    assert artifact["verifier_compile_rate"] == pytest.approx(1.0)
    assert artifact["verifier_set_size"] == 2
    assert artifact["verifier_coverage_rate"] == pytest.approx(1.0)
    assert artifact["verifier_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["baseline_validator_coverage_rate"] == pytest.approx(0.5)
    assert artifact["induction_manifest_path"] == str(induction_manifest_path)
    assert artifact["models_used"] == [exp.MANDATED_MODEL_SPECS[0]["hf_id"]]
    assert artifact["honest_verdict"].startswith("complete:")
    assert [row["row_type"] for row in manifest_rows] == [
        "candidate",
        "candidate",
        "selected_set_summary",
    ]


def test_req_verify_1507_runner_blocks_when_required_manifests_are_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1507: missing source manifests are terminal blockers with paths."""

    artifact = exp.run_experiment(
        output_path=tmp_path / "blocked.json",
        induction_manifest_path=tmp_path / "blocked.jsonl",
        certificate_manifest_path=tmp_path / "missing_certificates.jsonl",
        validator_manifest_path=tmp_path / "missing_validators.jsonl",
        run_date="20260507",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_candidate_proposals_fn=lambda _spec, _rows: pytest.fail("must not collect"),
        gpu_probe_fn=lambda: {"nvidia_smi_available": False, "gpu_count": 0},
    )

    assert artifact["status"] == "blocked"
    assert artifact["verifier_induction_ready"] is False
    assert artifact["labeled_rows_loaded"] == 0
    assert artifact["blockers"] == [
        f"missing_certificate_manifest:{tmp_path / 'missing_certificates.jsonl'}",
        f"missing_validator_manifest:{tmp_path / 'missing_validators.jsonl'}",
    ]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1507_runner_blocks_without_model_or_live_candidates(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1507: model unavailability and empty live proposals do not become headline results."""

    certificate_path, validator_path = _sample_manifest_paths(tmp_path)
    no_model = exp.run_experiment(
        output_path=tmp_path / "no_model.json",
        induction_manifest_path=tmp_path / "no_model.jsonl",
        certificate_manifest_path=certificate_path,
        validator_manifest_path=validator_path,
        run_date="20260507",
        model_specs=[],
        gpu_probe_fn=lambda: {"nvidia_smi_available": False, "gpu_count": 0},
    )

    def fake_empty_collect(_spec: dict[str, Any], _rows: list[exp.LabeledVerifierRow]) -> dict[str, Any]:
        return {
            "summary": {
                "hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                "model_used": False,
                "blocker": "load_failed",
            },
            "rows": [],
        }

    no_candidates = exp.run_experiment(
        output_path=tmp_path / "no_candidates.json",
        induction_manifest_path=tmp_path / "no_candidates.jsonl",
        certificate_manifest_path=certificate_path,
        validator_manifest_path=validator_path,
        run_date="20260507",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_candidate_proposals_fn=fake_empty_collect,
        gpu_probe_fn=lambda: {"nvidia_smi_available": False, "gpu_count": 0},
    )

    assert no_model["status"] == "blocked"
    assert no_model["blockers"] == ["no_mandated_sota_gguf_model_available"]
    assert no_candidates["status"] == "blocked"
    assert no_candidates["blockers"] == [
        "load_failed",
        "live_sota_candidate_generation_unavailable",
    ]


def test_req_verify_1507_runner_records_unparseable_live_output(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1507: live model text without safe-DSL JSON is still auditable."""

    certificate_path, validator_path = _sample_manifest_paths(tmp_path)

    def fake_unparseable(_spec: dict[str, Any], _rows: list[exp.LabeledVerifierRow]) -> dict[str, Any]:
        return {
            "summary": {
                "hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                "model_used": True,
                "blocker": None,
            },
            "rows": [
                {
                    "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                    "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                    "generation_source": "live_sota_llamacpp",
                    "output_text": "I cannot return JSON.",
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
            ],
        }

    manifest_path = tmp_path / "unparseable.jsonl"
    artifact = exp.run_experiment(
        output_path=tmp_path / "unparseable.json",
        induction_manifest_path=manifest_path,
        certificate_manifest_path=certificate_path,
        validator_manifest_path=validator_path,
        run_date="20260507",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_candidate_proposals_fn=fake_unparseable,
        gpu_probe_fn=lambda: {"nvidia_smi_available": True, "gpu_count": 1},
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "blocked"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["candidate_verifiers_proposed"] == 1
    assert rows[0]["compile_failure_reason"] == "no_json_candidate_object"
