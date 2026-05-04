"""Tests for Exp 1283 certificate grammar backend bakeoff.

Spec: REQ-VERIFY-1283, SCENARIO-VERIFY-1283
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import certificate_grammar_backend_bakeoff as exp


def test_certificate_schema_and_validator_cover_req1283_fields() -> None:
    """REQ-VERIFY-1283-2: the minimal certificate schema has required fields."""

    schema = exp.certificate_schema()
    sample = exp.sample_certificate()
    valid, errors = exp.validate_certificate(sample, schema)
    invalid, invalid_errors = exp.validate_certificate({"claims": []}, schema)

    assert schema["required"] == [
        "claims",
        "equations",
        "final_answer",
        "confidence",
        "verifier_routes",
        "proof_numbers",
    ]
    assert valid is True
    assert errors == []
    assert invalid is False
    assert "missing equations" in invalid_errors
    assert "missing final_answer" in invalid_errors
    assert exp.bounded_vocab_constraint_count(schema) == 9


def test_probe_backend_records_availability_and_failure_reasons_for_req1283() -> None:
    """REQ-VERIFY-1283-3/4: backend probes record import, CLI, and blockers."""

    definitions = [
        exp.BackendDefinition(
            name="available_cli_backend",
            import_name="available_pkg",
            cli_candidates=("available-cli",),
            schema_support="json_schema",
            unsupported_features=("cross-field equality",),
            constrained_generation=True,
            priority=10,
            help_markers=("--grammar",),
        ),
        exp.BackendDefinition(
            name="missing_backend",
            import_name="missing_pkg",
            cli_candidates=("missing-cli",),
            schema_support="json_schema",
            unsupported_features=("cross-field equality",),
            constrained_generation=True,
            priority=20,
            help_markers=("--grammar",),
        ),
    ]

    def import_checker(name: str) -> bool:
        return name == "available_pkg"

    def cli_finder(name: str) -> str | None:
        return "/usr/bin/available-cli" if name == "available-cli" else None

    def help_runner(_path: str) -> str:
        return "usage: available-cli --grammar FILE"

    records = exp.probe_backends(
        definitions,
        import_checker=import_checker,
        cli_finder=cli_finder,
        help_runner=help_runner,
    )

    assert records[0]["name"] == "available_cli_backend"
    assert records[0]["import_available"] is True
    assert records[0]["cli_available"] is True
    assert records[0]["cli_supports_grammar"] is True
    assert records[0]["available"] is True
    assert records[0]["failure_reason"] is None
    assert records[1]["name"] == "missing_backend"
    assert records[1]["available"] is False
    assert records[1]["failure_reason"] == "import_and_cli_absent"


def test_backend_selection_blocks_when_only_validation_fallback_exists_for_req1283() -> None:
    """REQ-VERIFY-1283-6: post-hoc validation alone does not unblock decoding."""

    records = [
        {
            "name": "xgrammar",
            "available": False,
            "constrained_generation": True,
            "priority": 30,
        },
        {
            "name": "pure_python_validation",
            "available": True,
            "constrained_generation": False,
            "priority": 100,
        },
    ]

    selected = exp.select_backend(records)

    assert selected["name"] == "pure_python_validation"
    assert selected["grammar_backend_available"] is False
    assert selected["fallback_only"] is True


def test_backend_selection_prefers_lowest_friction_generation_backend_for_req1283() -> None:
    """REQ-VERIFY-1283-3: available constrained-generation backends win."""

    records = [
        {"name": "xgrammar", "available": True, "constrained_generation": True, "priority": 30},
        {
            "name": "llama_cpp_gbnf",
            "available": True,
            "constrained_generation": True,
            "priority": 10,
        },
        {
            "name": "pure_python_validation",
            "available": True,
            "constrained_generation": False,
            "priority": 100,
        },
    ]

    selected = exp.select_backend(records)

    assert selected["name"] == "llama_cpp_gbnf"
    assert selected["grammar_backend_available"] is True
    assert selected["fallback_only"] is False


def test_build_bakeoff_artifact_contains_required_notes_for_req1283() -> None:
    """REQ-VERIFY-1283-5: artifact includes CDoT, STATIC, ABS, and risk notes."""

    schema = exp.certificate_schema()
    records = [
        {
            "name": "llguidance",
            "available": True,
            "constrained_generation": True,
            "priority": 20,
            "import_available": True,
            "cli_available": False,
            "schema_support": "json_schema",
            "unsupported_features": ["context-sensitive proof-number consistency"],
            "estimated_overhead": "not_measured_no_model_inference",
            "failure_reason": None,
        },
        {
            "name": "pure_python_validation",
            "available": True,
            "constrained_generation": False,
            "priority": 100,
            "import_available": True,
            "cli_available": False,
            "schema_support": "post_hoc_validation",
            "unsupported_features": ["token-level constrained generation"],
            "estimated_overhead": {"validation_1000_docs_ms": 1.0},
            "failure_reason": None,
        },
    ]

    artifact = exp.build_bakeoff_artifact(records, schema=schema, run_date="20260504")

    assert artifact["experiment"] == "1283_certificate_grammar_backend_bakeoff"
    assert artifact["schema"] == "certificate_grammar_backend_bakeoff_v1"
    assert artifact["run_date"] == "20260504"
    assert artifact["status"] == "complete"
    assert artifact["llm_inference_run"] is False
    assert artifact["grammar_backend_available"] is True
    assert artifact["grammar_backend_selected"] == "llguidance"
    assert artifact["bounded_vocab_constraint_count"] == 9
    assert artifact["automata_fallback_viable"] is True
    assert artifact["dfa_checkable_fields"] == schema["required"]
    assert "context-sensitive" in artifact["cdot_expressiveness_note"]
    assert "trie" in artifact["static_trie_note"].lower()
    assert artifact["structure_snowballing_risk"] == "medium"
    assert artifact["honest_verdict"] == "selected_llguidance"


def test_run_bakeoff_writes_in_progress_then_complete_scenario1283(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-1283: runner writes the required final artifact."""

    output_path = tmp_path / "experiment_1283_certificate_grammar_backend_bakeoff.json"
    statuses: list[str] = []
    original = exp.write_in_progress_artifact

    def tracking_write(path: Path | str, *, run_date: str = exp.RUN_DATE) -> dict:
        artifact = original(path, run_date=run_date)
        statuses.append(json.loads(Path(path).read_text(encoding="utf-8"))["status"])
        return artifact

    monkeypatch.setattr(exp, "write_in_progress_artifact", tracking_write)

    artifact = exp.run_bakeoff(
        output_path=output_path,
        run_date="20260504",
        import_checker=lambda name: name == "xgrammar",
        cli_finder=lambda _name: None,
        help_runner=lambda _path: "",
        overhead_timer=exp.ConstantStepTimer(),
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert statuses == ["in_progress"]
    assert persisted == artifact
    assert persisted["status"] == "complete"
    assert persisted["grammar_backend_selected"] == "xgrammar"
    assert persisted["grammar_backend_available"] is True
    assert persisted["backend_probes"]["pure_python_validation"]["estimated_overhead"][
        "validation_1000_docs_ms"
    ] == pytest.approx(1.0)


def test_validation_edge_cases_cover_req1283_fallback_errors() -> None:
    """REQ-VERIFY-1283-3: pure-Python fallback reports bounded-schema errors."""

    schema = exp.certificate_schema()
    payload = exp.sample_certificate()
    payload.update(
        {
            "claims": [
                {"id": "", "text": ""},
                "not an object",
                {"id": "c3"},
                {"id": "c4", "text": "x" * 321},
                {"id": "c5", "text": "ok"},
                {"id": "c6", "text": "ok"},
                {"id": "c7", "text": "ok"},
                {"id": "c8", "text": "ok"},
                {"id": "c9", "text": "too many"},
            ],
            "equations": [{"lhs": 3, "relation": "about", "rhs": ""}],
            "final_answer": 4,
            "confidence": float("inf"),
            "verifier_routes": [{"claim_id": "c1", "verifier": "unknown"}],
            "proof_numbers": ["not a number"],
        }
    )

    valid, errors = exp.validate_certificate(payload, schema)
    low_confidence = dict(exp.sample_certificate(), confidence=-0.1)
    high_confidence = dict(exp.sample_certificate(), confidence=1.1)
    array_type_error = dict(exp.sample_certificate(), proof_numbers="not an array")

    assert valid is False
    assert "claims must contain at most 8 items" in errors
    assert "claims[0].id is too short" in errors
    assert "claims[0].id does not match pattern ^c[0-9]+$" in errors
    assert "claims[0].text is too short" in errors
    assert "claims[1] must be object" in errors
    assert "missing claims[2].text" in errors
    assert "claims[3].text is too long" in errors
    assert "equations[0].lhs must be string" in errors
    assert "equations[0].relation must be one of ['=', '!=', '<=', '>=']" in errors
    assert "equations[0].rhs is too short" in errors
    assert "final_answer must be string" in errors
    assert "confidence must be finite" in errors
    assert "verifier_routes[0].verifier must be one of" in "\n".join(errors)
    assert "proof_numbers[0] must be number" in errors
    assert "confidence below minimum 0.0" in exp.validate_certificate(low_confidence, schema)[1]
    assert "confidence above maximum 1.0" in exp.validate_certificate(high_confidence, schema)[1]
    assert "proof_numbers must be array" in exp.validate_certificate(array_type_error, schema)[1]


def test_probe_helper_edges_cover_req1283_absent_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1283-4: helper edges distinguish import-only and CLI-only gaps."""

    import_only = exp.BackendDefinition(
        name="import_only",
        import_name="missing_import_only",
        cli_candidates=(),
        schema_support="schema",
        unsupported_features=(),
        constrained_generation=True,
        priority=1,
    )
    cli_only = exp.BackendDefinition(
        name="cli_only",
        import_name=None,
        cli_candidates=("missing-cli-only",),
        schema_support="schema",
        unsupported_features=(),
        constrained_generation=True,
        priority=2,
    )

    monkeypatch.setattr(exp.shutil, "which", lambda name: f"/bin/{name}" if name == "tool" else None)

    class Completed:
        stdout = "stdout --grammar"
        stderr = "stderr"

    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: Completed())

    assert exp._module_available("json") is True
    assert exp._module_available("definitely_missing_carnot_backend_1283") is False
    assert exp._find_cli("tool") == "/bin/tool"
    assert "--grammar" in exp._help_text("/bin/tool")
    assert exp._failure_reason(import_only, False, None) == "import_absent"
    assert exp._failure_reason(cli_only, False, None) == "cli_absent"
    assert exp.select_backend([])["name"] == "none"

    def raise_oserror(*_args: object, **_kwargs: object) -> object:
        raise OSError("boom")

    monkeypatch.setattr(exp.subprocess, "run", raise_oserror)

    assert exp._help_text("/bin/tool") == ""
