"""Tests for REQ-VERIFY-7208 and SCENARIO-VERIFY-7208-*.

Writer tests use temporary fixture directories. They do not change the checked-in
research record.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7208_v635_span_fixture as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"


def _relation(text: str, subject: str, predicate: str, obj: str, polarity: str = "positive"):
    """Make one relation whose offsets point at exact public UTF-8 bytes."""

    encoded = text.encode("utf-8")
    subject_bytes = subject.encode("utf-8")
    object_bytes = obj.encode("utf-8")
    subject_start = encoded.index(subject_bytes)
    object_start = encoded.rindex(object_bytes)
    return {
        "sentence_index": 0,
        "subject_start": subject_start,
        "subject_end": subject_start + len(subject_bytes),
        "predicate": predicate,
        "object_start": object_start,
        "object_end": object_start + len(object_bytes),
        "polarity": polarity,
    }


def _completion(relation):
    """Wrap one tuple in the frozen known-outcome completion shape."""

    return {"outcome": "known", "relations": [relation]}


def test_req_verify_7208_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7208 declares every focused scenario and required field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7208") :]
    for scenario in (
        "COMPILER",
        "GRAMMARS",
        "PANEL",
        "AUTHORITY",
        "DIAGNOSIS",
        "PREFLIGHT",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7208-{scenario}" in section
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_verify_7208_panel_has_frozen_balanced_splits() -> None:
    """SCENARIO-VERIFY-7208-PANEL fixes all bases and four-row groups."""

    panel = exp.build_panel()
    public = panel["public_rows"]
    authority = panel["authority_rows"]

    assert len(public) == len(authority) == 320
    assert len({row["base_id"] for row in authority}) == 80
    assert all(set(row) == {"unit_id", "source_text", "claim_text"} for row in public)
    assert [row["unit_id"] for row in public] == [row["unit_id"] for row in authority]
    assert len({row["unit_id"] for row in public}) == 320
    assert {row["variant"] for row in authority} == set(exp.VARIANTS)
    assert sum(row["split"] == "test" for row in authority) == 256

    counts = {}
    for row in authority:
        key = (row["split"], row["relation_family"])
        counts.setdefault(key, set()).add(row["base_id"])
    assert {key: len(value) for key, value in counts.items()} == {
        **{("canary", family): 2 for family in exp.FAMILIES},
        **{("development", family): 2 for family in exp.FAMILIES},
        **{("test", family): 16 for family in exp.FAMILIES},
    }
    assert panel["split_manifest"]["construction_seed"] == 7_208_001
    assert panel["split_manifest"]["surface_seed"] == 7_208_002
    assert len(set(panel["split_manifest"]["split_hashes"].values())) == 3
    assert all(row["alpha_rename_base_id"] for row in authority)


def test_scenario_verify_7208_compiler_accepts_exact_public_references() -> None:
    """SCENARIO-VERIFY-7208-COMPILER compiles a public tuple without labels."""

    text = "Aster precedes Brin."
    relation = _relation(text, "Aster", "precedes", "Brin")
    grammar = exp.compile_grammar(text.encode(), "claim", "reference")
    compiled = exp.compile_completion(
        text.encode(),
        _completion(relation),
        "claim",
        grammar["grammar_sha256"],
    )

    assert compiled["outcome"] == "known"
    assert compiled["relations"][0]["subject_surface"] == "Aster"
    assert compiled["relations"][0]["object_surface"] == "Brin"
    assert compiled["relations"][0]["predicate"] == "precedes"
    assert compiled["errors"] == []
    assert "expected_decision" not in json.dumps(grammar)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("out_of_range", "span_out_of_range"),
        ("wrong_sentence", "span_outside_sentence"),
        ("wrong_type", "type_mismatch"),
        ("unsupported_predicate", "unsupported_predicate"),
        ("invalid_polarity", "invalid_polarity"),
        ("extra_field", "completion_shape"),
        ("excess_claim", "relation_count"),
        ("cross_document", "grammar_request_mismatch"),
    ),
)
def test_scenario_verify_7208_compiler_rejects_reference_mutations(
    mutation: str, error: str
) -> None:
    """SCENARIO-VERIFY-7208-COMPILER rejects each bounded attack."""

    text = "Aster precedes Brin. Cora follows Dain."
    relation = _relation(text, "Aster", "precedes", "Brin")
    completion = _completion(relation)
    grammar = exp.compile_grammar(text.encode(), "claim", "reference")
    grammar_hash = grammar["grammar_sha256"]
    if mutation == "out_of_range":
        relation["subject_end"] = len(text.encode()) + 1
    elif mutation == "wrong_sentence":
        relation["object_start"] = text.encode().index(b"Cora")
        relation["object_end"] = relation["object_start"] + 4
    elif mutation == "wrong_type":
        relation["subject_start"] = text.encode().index(b"precedes")
        relation["subject_end"] = relation["subject_start"] + len(b"precedes")
    elif mutation == "unsupported_predicate":
        relation["predicate"] = "touches"
    elif mutation == "invalid_polarity":
        relation["polarity"] = "maybe"
    elif mutation == "extra_field":
        completion["label"] = "supported"
    elif mutation == "excess_claim":
        completion["relations"].append(deepcopy(relation))
    else:
        grammar_hash = exp.compile_grammar(b"Eris precedes Fenn.", "claim", "reference")[
            "grammar_sha256"
        ]

    result = exp.compile_completion(text.encode(), completion, "claim", grammar_hash)

    assert result["outcome"] == "unknown"
    assert error in result["errors"]


def test_scenario_verify_7208_compiler_handles_unknown_and_source_bound() -> None:
    """SCENARIO-VERIFY-7208-COMPILER keeps unknown explicit and caps source tuples."""

    text = b"Aster precedes Brin."
    grammar = exp.compile_grammar(text, "source", "reference")
    unknown = exp.compile_completion(
        text, {"outcome": "unknown", "relations": []}, "source", grammar["grammar_sha256"]
    )
    invalid_unknown = exp.compile_completion(
        text,
        {
            "outcome": "unknown",
            "relations": [_relation(text.decode(), "Aster", "precedes", "Brin")],
        },
        "source",
        grammar["grammar_sha256"],
    )
    excess = exp.compile_completion(
        text,
        {
            "outcome": "known",
            "relations": [_relation(text.decode(), "Aster", "precedes", "Brin")] * 5,
        },
        "source",
        grammar["grammar_sha256"],
    )

    assert unknown == {"outcome": "unknown", "relations": [], "errors": []}
    assert "unknown_with_relations" in invalid_unknown["errors"]
    assert "relation_count" in excess["errors"]


def test_scenario_verify_7208_grammars_share_schema_but_not_hidden_truth() -> None:
    """SCENARIO-VERIFY-7208-GRAMMARS restricts references from public bytes only."""

    first = b"Aster precedes Brin."
    second = b"Cora precedes Dain."
    syntax_first = exp.compile_grammar(first, "claim", "grammar_only")
    syntax_second = exp.compile_grammar(second, "claim", "grammar_only")
    reference_first = exp.compile_grammar(first, "claim", "reference")
    reference_second = exp.compile_grammar(second, "claim", "reference")

    assert syntax_first["grammar"] == syntax_second["grammar"]
    assert reference_first["grammar"] != reference_second["grammar"]
    assert syntax_first["tuple_schema_sha256"] == reference_first["tuple_schema_sha256"]
    assert syntax_first["token_budget"] == reference_first["token_budget"]
    assert syntax_first["model_settings"] == reference_first["model_settings"]
    assert exp.grammar_errors(syntax_first["grammar"]) == []
    assert exp.grammar_errors(reference_first["grammar"]) == []
    assert "supported" not in reference_first["grammar"]
    assert "contradicted" not in reference_first["grammar"]
    assert first.hex() not in reference_first["grammar"]


def test_scenario_verify_7208_authority_and_candidate_agree_on_panel() -> None:
    """SCENARIO-VERIFY-7208-AUTHORITY independently checks every semantic edit."""

    panel = exp.build_panel()
    execution = exp.execute_panel(panel["public_rows"], panel["authority_rows"])

    assert len(execution) == 320
    assert all(row["metric"] == 1 and row["error"] is None for row in execution)
    observed = {row["variant"]: set() for row in panel["authority_rows"]}
    for row in execution:
        observed[row["variant"]].add(row["prediction"])
    assert observed == {
        "supported": {"supported"},
        "reversal": {"contradicted"},
        "joint_support": {"supported"},
        "support_removed": {"unknown"},
    }
    assert exp.AUTHORITY_IMPORTS_CANDIDATE is False


def test_scenario_verify_7208_authority_covers_negation_and_bad_language() -> None:
    """SCENARIO-VERIFY-7208-AUTHORITY interprets negation without candidate code."""

    assert (
        exp.authority_decision("Aster does not precede Brin.", "Aster precedes Brin.")
        == "contradicted"
    )
    assert exp.authority_decision("Aster precedes Brin.", "Aster follows Brin.") == "contradicted"
    assert exp.authority_decision("uncontrolled text", "Aster precedes Brin.") == "unknown"


def test_scenario_verify_7208_diagnosis_reproduces_raw_counts() -> None:
    """SCENARIO-VERIFY-7208-DIAGNOSIS derives old failures from authenticated bytes."""

    rows = exp.diagnose_exp7196(REPO)
    by_type = {row["call_type"]: row for row in rows}

    assert by_type["claim"]["denominator"] == 192
    assert by_type["claim"]["invalid_count"] == 192
    assert by_type["claim"]["truncated_count"] == 192
    assert by_type["source"]["denominator"] == 192
    assert by_type["source"]["invalid_count"] == 144
    assert by_type["source"]["truncated_count"] == 144
    assert by_type["claim"]["missing_terminator_count"] == 192
    assert by_type["claim"]["format_overhead_bytes"] > 0
    assert all(row["manifest_rows_authenticated"] == row["denominator"] for row in rows)
    assert all("larger_budget_assumed" in row and not row["larger_budget_assumed"] for row in rows)


def test_scenario_verify_7208_lexical_control_retains_saturated_cells() -> None:
    """REQ-VERIFY-7208 reports all lexical rows and same-token blind pairs."""

    panel = exp.build_panel()
    rows = exp.lexical_control(panel["public_rows"], panel["authority_rows"])

    assert len(rows) == 320
    assert sum(row["metric"] for row in rows) == 240
    assert sum(row["split"] == "test" and row["metric"] for row in rows) == 192
    assert sum(row["matched_semantic_pair_indistinguishable"] for row in rows) == 160
    assert all(row["arm"] == "frozen_lexical_token_membership" for row in rows)


def test_scenario_verify_7208_mutations_cover_compiler_executor_boundary() -> None:
    """REQ-VERIFY-7208 stores every required adverse boundary outcome."""

    rows = exp.span_mutation_checks()
    names = {row["mutation"] for row in rows}

    assert {
        "out_of_range",
        "cross_document",
        "wrong_sentence",
        "type_mismatch",
        "unsupported_predicate",
        "invalid_polarity",
        "excess_source_relations",
        "direction_reversal",
        "negation",
        "support_removed",
        "grammar_serialization",
    } <= names
    assert all(row["passed"] for row in rows)


def test_scenario_verify_7208_preflight_unwrap_and_quarantine_are_fail_closed() -> None:
    """SCENARIO-VERIFY-7208-PREFLIGHT unwraps only declared wrappers."""

    assert exp._unwrap({"principle": "why", "value": 1}) == 1
    arbitrary = {"value": 1, "other": "not a field wrapper"}
    assert exp._unwrap(arbitrary) is arbitrary
    assert exp._is_quarantined({"flagged_adversarial": {"principle": "why", "value": True}})
    assert not exp._is_quarantined({"flagged_adversarial": {"value": True, "other": False}})


def test_scenario_verify_7208_artifact_writes_and_cold_validates(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7208-ARTIFACT seals sidecars before readiness."""

    artifact = exp.build_artifact(REPO, exp.RUN_DATE, output_root=tmp_path)
    public_path = tmp_path / exp.PUBLIC_VIEW_PATH
    authority_path = tmp_path / exp.AUTHORITY_SIDECAR_PATH
    manifest_path = tmp_path / exp.FIXTURE_MANIFEST_PATH

    assert artifact["status"] == "complete"
    assert artifact["span_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["execution_venue"] == "host"
    assert artifact["execution_host"]
    assert public_path.is_file() and authority_path.is_file() and manifest_path.is_file()
    assert exp.validate_artifact(artifact, REPO, output_root=tmp_path) == []

    changed = deepcopy(artifact)
    changed["sample_size_budget"]["completed_rows"] = 319
    assert "sample_size_budget" in exp.validate_artifact(changed, REPO, output_root=tmp_path)


def test_scenario_verify_7208_artifact_blocks_missing_external_input(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7208-PREFLIGHT gives an exact terminal block."""

    missing = tmp_path / "missing-exp7196.json"
    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        output_root=tmp_path / "outputs",
        path_overrides={"exp7196_artifact": missing},
    )

    assert artifact["status"] == "blocked"
    assert artifact["span_fixture_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "source_exists",
        "upstream": "exp7196_artifact",
        "field": "path",
        "expected_value": "existing_file",
        "observed_value": str(missing),
    }
    assert exp.validate_artifact(artifact, REPO, output_root=tmp_path / "outputs") == []


def test_req_verify_7208_tokenizer_receipt_defers_without_embedded_loader() -> None:
    """REQ-VERIFY-7208 leaves measured token sizes unknown when no loader exists."""

    receipt = exp.measure_completion_tokens([b"{}", b'{"outcome":"unknown","relations":[]}'], None)

    assert receipt["measurement_status"] == "deferred_to_canary"
    assert receipt["minimum_completion_tokens"] is None
    assert receipt["maximum_completion_tokens"] is None
    measured = exp.measure_completion_tokens([b"a", b"abcd"], lambda value: list(value))
    assert measured["measurement_status"] == "measured_embedded_gguf_tokenizer"
    assert measured["minimum_completion_tokens"] == 1
    assert measured["maximum_completion_tokens"] == 4


def test_req_verify_7208_loads_only_embedded_vocabulary(tmp_path: Path) -> None:
    """REQ-VERIFY-7208 uses GGUF vocabulary-only mode and can defer a missing file."""

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fixture")
    calls = []

    class FakeVocabulary:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def tokenize(self, value, add_bos=False):
            assert add_bos is False
            return list(value)

        def close(self):
            calls.append("closed")

    owner, tokenizer, receipt = exp.load_embedded_tokenizer(
        {"model_specs": [{"path": str(model_path), "sha256": "sha256:model"}]},
        loader=FakeVocabulary,
    )
    assert owner is not None and tokenizer is not None
    assert tokenizer(b"ab") == [97, 98]
    assert calls[0]["vocab_only"] is True
    assert calls[0]["model_path"] == str(model_path)
    assert receipt["embedded_tokenizer_available"] is True
    owner.close()
    assert calls[-1] == "closed"

    owner, tokenizer, receipt = exp.load_embedded_tokenizer(
        {"model_specs": [{"path": str(tmp_path / "missing.gguf")}]}, loader=FakeVocabulary
    )
    assert owner is tokenizer is None
    assert receipt["embedded_tokenizer_available"] is False
    owner, tokenizer, receipt = exp.load_embedded_tokenizer(
        {"model_specs": [{"path": str(model_path)}]},
        loader=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad vocab")),
    )
    assert owner is tokenizer is None
    assert receipt["defer_reason"] == "vocabulary_load_failed:RuntimeError"

    fake_module = type("FakeModule", (), {"Llama": FakeVocabulary})
    assert exp.resolve_tokenizer_loader(lambda name: fake_module) is FakeVocabulary

    def missing_import(name):
        raise ImportError(name)

    assert exp.resolve_tokenizer_loader(missing_import) is None

    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        output_root=tmp_path / "measured",
        tokenizer_loader=FakeVocabulary,
    )
    size = artifact["grammar_contract"]["completion_size_receipt"]
    assert size["measurement_status"] == "measured_embedded_gguf_tokenizer"
    assert (
        artifact["grammar_contract"]["embedded_tokenizer_receipt"]["embedded_tokenizer_available"]
        is True
    )
    assert calls[-1] == "closed"


def test_req_verify_7208_artifact_checksum_excludes_runtime_only_fields() -> None:
    """REQ-VERIFY-7208 checksum binds evidence but not measured duration."""

    value = {"duration_s": 1.0, "reproducibility_checksum": "old", "rows": [1]}
    changed = {"duration_s": 9.0, "reproducibility_checksum": "new", "rows": [1]}
    assert exp.artifact_checksum(value) == exp.artifact_checksum(changed)
    changed["rows"] = [2]
    assert exp.artifact_checksum(value) != exp.artifact_checksum(changed)


def test_scenario_verify_7208_compiler_defensive_shapes_and_grammar_lint() -> None:
    """SCENARIO-VERIFY-7208-COMPILER rejects malformed shapes and grammar text."""

    text = b"Aster precedes Brin."
    relation = _relation(text.decode(), "Aster", "precedes", "Brin")
    grammar_hash = exp.compile_grammar(text, "claim", "reference")["grammar_sha256"]
    with pytest.raises(ValueError, match="unsupported grammar"):
        exp.compile_grammar(text, "bad", "reference")
    assert exp.grammar_errors("") == ["gbnf_rule_shape"]
    duplicate = 'root ::= item\nroot ::= item\nitem ::= "supported"\n'
    assert set(exp.grammar_errors(duplicate)) == {"gbnf_duplicate_rule", "authority_leak"}
    assert "grammar_utf8" in exp.grammar_errors('root ::= "\ud800"')
    assert exp.extract_public_completion(b"uncontrolled text.", "claim")["outcome"] == "unknown"
    five = b"Aster precedes Brin. " * 5
    assert exp.extract_public_completion(five, "source")["outcome"] == "unknown"
    assert exp.compile_completion(text, _completion(relation), "bad", grammar_hash)["errors"] == [
        "call_type"
    ]
    malformed = (
        {"outcome": "bad", "relations": []},
        {"outcome": "known", "relations": "bad"},
        {"outcome": "known", "relations": [{"sentence_index": 0}]},
    )
    assert exp.compile_completion(text, malformed[0], "claim", grammar_hash)["errors"] == [
        "completion_shape"
    ]
    assert exp.compile_completion(text, malformed[1], "claim", grammar_hash)["errors"] == [
        "completion_shape"
    ]
    assert (
        "relation_shape"
        in exp.compile_completion(text, malformed[2], "claim", grammar_hash)["errors"]
    )
    for field, value, expected in (
        ("sentence_index", True, "sentence_index"),
        ("subject_start", True, "span_out_of_range"),
        ("sentence_index", 9, "sentence_index"),
    ):
        changed = deepcopy(relation)
        changed[field] = value
        assert (
            expected
            in exp.compile_completion(text, _completion(changed), "claim", grammar_hash)["errors"]
        )
    assert exp._canonical_relation("follows", "Aster", "Brin") == (
        "precedes",
        "Brin",
        "Aster",
    )
    assert exp._candidate_decision("uncontrolled text.", "Aster precedes Brin.") == {
        "decision": "unknown",
        "abstention": True,
        "errors": ["explicit_unknown"],
    }


def test_scenario_verify_7208_execution_emits_observed_heartbeat(monkeypatch) -> None:
    """REQ-VERIFY-7208 reports elapsed work in a long execution loop."""

    panel = exp.build_panel()
    moments = iter((0.0, 61.0))
    events = []
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(exp, "_progress", lambda *args: events.append(args))

    rows = exp.execute_panel(panel["public_rows"][:1], panel["authority_rows"][:1])

    assert rows[0]["metric"] == 1
    assert events == [(4, "heartbeat", "executed=1/1 elapsed_s=61.0")]


def test_scenario_verify_7208_raw_receipt_authentication_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7208-DIAGNOSIS rejects missing or changed raw bytes."""

    path = tmp_path / "call.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="raw completion missing"):
        exp._stored_completion(path, {"row_sha256": "x", "raw_output_sha256": "y"})

    completion = {"raw_output": "{}"}
    path.write_text(json.dumps({"completion": completion}), encoding="utf-8")
    with pytest.raises(ValueError, match="raw row hash mismatch"):
        exp._stored_completion(path, {"row_sha256": "x", "raw_output_sha256": "y"})
    with pytest.raises(ValueError, match="raw output hash mismatch"):
        exp._stored_completion(
            path,
            {
                "row_sha256": exp.sha256_bytes(exp.canonical_json(completion).encode()),
                "raw_output_sha256": "x",
            },
        )


def _write_synthetic_raw_tree(tmp_path: Path, escaped: bool = False) -> None:
    """Write a tiny authenticated raw tree for diagnosis failure-path tests."""

    raw_dir = tmp_path / exp.RAW_MANIFEST_PATH.parent
    raw_dir.mkdir(parents=True)
    raw_rows = []
    schedule = []
    for index, call_type in enumerate(("source", "claim")):
        raw = '{\n"abcdefghijkl":1,\n"abcdefghijkl":1,\n"finalpadding":2\n}'
        completion = {
            "call_id": f"000:{call_type}",
            "call_type": call_type,
            "raw_output": raw,
            "parse_status": "valid",
            "parsed_output": {"relations": []},
            "truncated": False,
        }
        call_path = raw_dir / f"call_{index:03d}.json"
        call_path.write_text(json.dumps({"completion": completion}), encoding="utf-8")
        raw_rows.append(
            {
                "call_id": completion["call_id"],
                "call_type": call_type,
                "path": str((tmp_path / "escaped.json") if escaped and index == 0 else call_path),
                "row_sha256": exp.sha256_bytes(exp.canonical_json(completion).encode()),
                "raw_output_sha256": exp.sha256_bytes(raw.encode()),
            }
        )
        schedule.append(
            {"call_id": completion["call_id"], "model_input": {f"{call_type}_text": "visible"}}
        )
    manifest = {"raw_rows": raw_rows, "schedule": schedule}
    manifest_path = tmp_path / exp.RAW_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifact_path = tmp_path / exp.EXP7196_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps({"source_artifact_hashes": {"raw_manifest": exp.sha256_file(manifest_path)}}),
        encoding="utf-8",
    )


def test_scenario_verify_7208_diagnosis_counts_semantic_and_repeated_bytes(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7208-DIAGNOSIS attributes valid but empty semantic output."""

    _write_synthetic_raw_tree(tmp_path)
    rows = exp.diagnose_exp7196(tmp_path)
    assert all(row["semantic_error_count"] == 1 for row in rows)
    assert all(row["repeated_output_count"] == 1 for row in rows)

    artifact_path = tmp_path / exp.EXP7196_PATH
    artifact_path.write_text(
        json.dumps({"source_artifact_hashes": {"raw_manifest": "changed"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        exp.diagnose_exp7196(tmp_path)


def test_scenario_verify_7208_diagnosis_rejects_escaped_raw_path(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7208-DIAGNOSIS never follows a manifest outside its raw directory."""

    _write_synthetic_raw_tree(tmp_path, escaped=True)
    with pytest.raises(ValueError, match="escaped"):
        exp.diagnose_exp7196(tmp_path)


def _copy_json_with_change(source: Path, target: Path, **changes) -> Path:
    """Copy one JSON source to a temporary path and change named top-level fields."""

    value = json.loads(source.read_text(encoding="utf-8"))
    value.update(changes)
    target.write_text(json.dumps(value), encoding="utf-8")
    return target


def test_scenario_verify_7208_precondition_failure_paths(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-VERIFY-7208-PREFLIGHT diagnoses each independently failed gate."""

    checks, _, _ = exp._preconditions(REPO, "wrong", tmp_path / "date", None)
    assert checks[-1]["check"] == "run_date"

    no_spec = tmp_path / "no-spec.md"
    no_spec.write_text("no requirement", encoding="utf-8")
    checks, _, _ = exp._preconditions(
        REPO, exp.RUN_DATE, tmp_path / "spec", {"constraint_spec": no_spec}
    )
    assert checks[-1]["check"] == "driving_spec"

    original_executor = exp.execute_relation
    monkeypatch.setattr(exp, "execute_relation", None)
    checks, _, _ = exp._preconditions(REPO, exp.RUN_DATE, tmp_path / "imports", None)
    assert checks[-1]["check"] == "required_imports"
    monkeypatch.setattr(exp, "execute_relation", original_executor)

    original_access = exp.os.access
    denied = tmp_path / "denied"
    monkeypatch.setattr(
        exp.os,
        "access",
        lambda path, mode: False if Path(path) == denied else original_access(path, mode),
    )
    checks, _, _ = exp._preconditions(REPO, exp.RUN_DATE, denied, None)
    assert checks[-1]["check"] == "output_destination"
    monkeypatch.setattr(exp.os, "access", original_access)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    checks, _, _ = exp._preconditions(
        REPO, exp.RUN_DATE, tmp_path / "parse", {"exp7196_artifact": malformed}
    )
    assert checks[-1]["check"] == "source_parse"

    quarantined = _copy_json_with_change(
        REPO / exp.EXP7196_PATH, tmp_path / "quarantined.json", flagged_adversarial=True
    )
    checks, _, _ = exp._preconditions(
        REPO, exp.RUN_DATE, tmp_path / "quarantine", {"exp7196_artifact": quarantined}
    )
    assert checks[-1]["check"] == "structured_quarantine"

    excluded = tmp_path / "excluded.yaml"
    excluded.write_text("excluded: exp7196-qwen-atomic-capture\n", encoding="utf-8")
    checks, _, _ = exp._preconditions(
        REPO, exp.RUN_DATE, tmp_path / "excluded", {"exclusion_manifest": excluded}
    )
    assert checks[-1]["check"] == "exclusion_manifest"

    bad_gate = _copy_json_with_change(
        REPO / exp.EXP7196_PATH, tmp_path / "bad-gate.json", atomic_capture_complete_score=0
    )
    checks, _, _ = exp._preconditions(
        REPO, exp.RUN_DATE, tmp_path / "gate", {"exp7196_artifact": bad_gate}
    )
    assert checks[-1]["check"] == "producer_gate_fields"

    bad_checksum = _copy_json_with_change(
        REPO / exp.EXP7196_PATH, tmp_path / "bad-checksum.json", reproducibility_checksum="bad"
    )
    checks, _, _ = exp._preconditions(
        REPO, exp.RUN_DATE, tmp_path / "checksum", {"exp7196_artifact": bad_checksum}
    )
    assert checks[-1]["check"] == "upstream_authentication"


def test_scenario_verify_7208_validator_reports_terminal_mutations(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7208-ARTIFACT reports all changed terminal evidence."""

    artifact = exp.build_artifact(REPO, exp.RUN_DATE, output_root=tmp_path)
    changed = deepcopy(artifact)
    changed.update(
        {
            "field_principles": {},
            "run_date": "wrong",
            "MODEL_SPECS": ["model"],
            "execution_venue": "gpu",
            "verifier_is_oracle": False,
            "duration_s": True,
            "random_seed": 1,
            "inference_substrate_class": "aggregation",
            "inference_substrate": "wrong",
            "rows": [],
            "lexical_control_rows": [],
            "span_mutation_rows": [],
            "span_fixture_ready_score": 0,
            "verdict_class": "positive",
            "grammar_contract": {},
            "split_manifest": {},
            "source_artifact_hashes": {},
        }
    )
    errors = set(exp.validate_artifact(changed, REPO, output_root=tmp_path))
    assert {
        "field_principles",
        "run_date",
        "model_contract",
        "execution_identity",
        "oracle_declaration",
        "duration_s",
        "random_seed",
        "reproducibility_checksum",
        "inference_substrate_class",
        "inference_substrate",
        "rows",
        "lexical_control_rows",
        "span_mutation_rows",
        "readiness_terminal_state",
        "split_manifest",
        "semantic_replay",
        "lexical_replay",
        "grammar_contract",
        "fixture_source_hashes",
    } <= errors
    assert exp.validate_artifact([]) == ["artifact_mapping"]
    assert exp.validate_artifact({})[0].startswith("missing_required_field:")

    blocked = deepcopy(artifact)
    blocked.update(
        {
            "status": "blocked",
            "verdict_class": "positive",
            "span_fixture_ready_score": 1,
            "gate_check_summary": {},
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert set(exp.validate_artifact(blocked)) == {
        "blocked_terminal_state",
        "gate_check_summary",
    }
    running = deepcopy(artifact)
    running["status"] = "running"
    running["reproducibility_checksum"] = exp.artifact_checksum(running)
    assert exp.validate_artifact(running) == ["status"]
    assert "sealed_fixture_files" in exp.validate_artifact(
        artifact, REPO, output_root=tmp_path / "absent"
    )


def test_scenario_verify_7208_validator_detects_sidecar_mutations(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7208-ARTIFACT rejects changed public, authority, and manifest bytes."""

    artifact = exp.build_artifact(REPO, exp.RUN_DATE, output_root=tmp_path)
    public_path = tmp_path / exp.PUBLIC_VIEW_PATH
    authority_path = tmp_path / exp.AUTHORITY_SIDECAR_PATH
    manifest_path = tmp_path / exp.FIXTURE_MANIFEST_PATH
    public_rows = exp._read_jsonl(public_path)
    authority_rows = exp._read_jsonl(authority_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_rows[0]["source_text"] = "Eris precedes Fenn."
    authority_rows[0]["expected_decision"] = "unknown"
    manifest["public_view_sha256"] = "changed"
    public_path.write_bytes(exp.jsonl_bytes(public_rows))
    authority_path.write_bytes(exp.jsonl_bytes(authority_rows))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = set(exp.validate_artifact(artifact, REPO, output_root=tmp_path))
    assert {"public_view", "authority_sidecar", "fixture_manifest_hashes"} <= errors
    invalid_jsonl = tmp_path / "invalid.jsonl"
    invalid_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        exp._read_jsonl(invalid_jsonl)


def test_req_verify_7208_cli_paths(monkeypatch, capsys, tmp_path: Path) -> None:
    """REQ-VERIFY-7208 keeps the executable wrapper and terminal error path bounded."""

    with pytest.raises(argparse.ArgumentTypeError):
        exp._date_argument("wrong")
    ready = exp._base_artifact(exp.RUN_DATE)
    ready.update(
        {
            "status": "complete",
            "honest_verdict": "complete_test",
            "span_fixture_ready_score": 1,
        }
    )
    monkeypatch.setattr(exp, "build_artifact", lambda root, date, **kwargs: ready)
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: [])
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert "complete_test" in capsys.readouterr().out
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: ["bad"])
    assert exp.main(["--date", exp.RUN_DATE]) == 1
    assert "invalid artifact" in capsys.readouterr().err

    monkeypatch.setattr(exp, "main", lambda: 0)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(REPO / exp.WRAPPER_PATH), run_name="__main__")
    assert caught.value.code == 0


def test_scenario_verify_7208_builder_refuses_invalid_terminal_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-VERIFY-7208-ARTIFACT refuses invalid complete and blocked writes."""

    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="invalid blocked"):
        exp.build_artifact(
            REPO,
            exp.RUN_DATE,
            output_root=tmp_path / "blocked",
            path_overrides={"exp7196_artifact": tmp_path / "missing.json"},
        )
    with pytest.raises(ValueError, match="invalid Exp7208"):
        exp.build_artifact(REPO, exp.RUN_DATE, output_root=tmp_path / "complete")
