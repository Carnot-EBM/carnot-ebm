"""Focused tests for the environment-indexed proof grammar.

Spec refs: REQ-VERIFY-6769 and SCENARIO-VERIFY-6769-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6769_environment_indexed_proof_grammar_v2 as exp


def _inputs() -> tuple[dict, dict]:
    """Load the frozen panel and exact stream without changing either file."""

    return (
        exp.load_json_object(exp.REPO_ROOT / exp.UPSTREAM_PANEL_PATH),
        exp.load_json_object(exp.REPO_ROOT / exp.UPSTREAM_STREAM_PATH),
    )


def _small_cnf() -> dict:
    """Use a small CNF whose complete SAT witness is easy to inspect."""

    return {"n_vars": 3, "clauses": [[1], [2, -1], [3, -2]]}


def _passing_runner(command: str, _root: Path) -> dict:
    """Return a deterministic passing receipt for unit tests of the runner."""

    output = "TOTAL 100 0 100%" if "coverage report" in command else "passed"
    return {"command": command, "exit_code": 0, "stdout": output, "stderr": "", "duration_s": 0.01}


def test_req_verify_6769_preconditions_fail_closed_with_observed_values() -> None:
    """REQ-VERIFY-6769 checks the panel, backend, and exact authority first."""

    panel, _stream = _inputs()
    ready = exp.evaluate_preconditions(
        panel,
        backend_receipt={"available": True, "backend": "llama_cpp.LlamaGrammar"},
        checker_receipt={"available": True, "checker": "exact_check_constraints"},
    )

    assert ready["all_passed"] is True
    assert exp.first_failed_check(ready)["check"] == "all_preconditions"
    assert [row["check"] for row in ready["checks"]] == [
        "exp6768_targetable_panel_ready",
        "exp6768_minimum_rows",
        "exp6768_held_family_coverage",
        "exp6768_target_class_coverage",
        "local_grammar_backend_import",
        "exact_checker_import",
    ]

    cases = [
        (
            "exp6768_targetable_panel_ready",
            lambda value: value.__setitem__("targetable_panel_ready", False),
        ),
        ("exp6768_minimum_rows", lambda value: value.__setitem__("targetable_row_count", 35)),
        (
            "exp6768_held_family_coverage",
            lambda value: value["counts_by_family"].pop("ladder_tseitin"),
        ),
        (
            "exp6768_target_class_coverage",
            lambda value: value["counts_by_error_class"].pop("invalid_clause"),
        ),
    ]
    for expected, mutate in cases:
        changed = deepcopy(panel)
        mutate(changed)
        summary = exp.evaluate_preconditions(
            changed,
            backend_receipt={"available": True},
            checker_receipt={"available": True},
        )
        assert exp.first_failed_check(summary)["check"] == expected

    backend_block = exp.evaluate_preconditions(
        panel,
        backend_receipt={"available": False, "error": "missing"},
        checker_receipt={"available": True},
    )
    checker_block = exp.evaluate_preconditions(
        panel,
        backend_receipt={"available": True},
        checker_receipt={"available": False, "error": "missing"},
    )
    assert exp.first_failed_check(backend_block)["check"] == "local_grammar_backend_import"
    assert exp.first_failed_check(checker_block)["check"] == "exact_checker_import"


def test_scenario_verify_6769_gamma_contains_only_answer_blind_runtime_state() -> None:
    """REQ-VERIFY-6769 binds symbols and state without answer authority."""

    gamma = exp.RuntimeGamma.from_cnf(_small_cnf())
    snapshot = gamma.snapshot()

    assert snapshot == {
        "variable_symbols": ["x1", "x2", "x3"],
        "clause_symbols": ["c1", "c2", "c3"],
        "binary_domain": ["0", "1"],
        "claim_branch": None,
        "remaining_required_slots": [],
        "used_symbols": [],
        "completion_state": "claim_required",
    }
    assert set(snapshot).isdisjoint(exp.FORBIDDEN_AUTHORITY_FIELDS)
    assert set(exp.GAMMA_SCHEMA["fields"]) == set(snapshot)
    assert exp.exact_authority_features_in_mechanism() == []

    text = "SAT x3=1 x1=1 x2=1"
    tokens = exp.lex_candidate(text)
    assert "".join(tokens) == text
    assert exp.lex_candidate("ＳＡＴ x１=１") == ["ＳＡＴ", " ", "x１", "=", "１"]


def test_scenario_verify_6769_sat_mask_blocks_ghosts_values_duplicates_and_stop() -> None:
    """SCENARIO-VERIFY-6769-MASK rejects SAT attacks before emission."""

    cnf = _small_cnf()
    valid = exp.EnvironmentIndexedProofMask(exp.RuntimeGamma.from_cnf(cnf)).replay(
        "SAT x3=1 x1=1 x2=1"
    )
    assert valid["terminal_reachable"] is True
    assert valid["candidate_reachable"] is True
    assert valid["emitted_output"] == "SAT x3=1 x1=1 x2=1"
    assert valid["final_gamma"]["remaining_required_slots"] == []
    assert valid["final_gamma"]["completion_state"] == "complete"
    assert valid["mask_invocation_count"] == len(exp.lex_candidate(valid["candidate"])) + 1
    assert "sat_select_remaining_variable" in valid["policies_invoked"]
    session = exp.EnvironmentIndexedProofMask(exp.RuntimeGamma.from_cnf(cnf))
    session.replay("ABSTAIN")
    assert session.allowed_tokens() == ("complete", set())

    attacks = {
        "SAT x9=0 x1=1 x2=1 x3=1": "x9",
        "SAT x1=2 x2=1 x3=1": "2",
        "SAT x1=1 x1=0 x2=1 x3=1": "x1",
        "SAT x1=1": exp.EOS_TOKEN,
        "ＳＡＴ x1=1 x2=1 x3=1": "ＳＡＴ",
        "SAT UNSAT c1": "UNSAT",
    }
    for candidate, blocked_token in attacks.items():
        receipt = exp.EnvironmentIndexedProofMask(exp.RuntimeGamma.from_cnf(cnf)).replay(candidate)
        assert receipt["candidate_reachable"] is False
        assert receipt["terminal_reachable"] is False
        assert receipt["blocked_token"] == blocked_token
        assert receipt["accepted_tokens"][-1:] != [blocked_token]
        assert receipt["mask_invocation_count"] > 0


def test_scenario_verify_6769_unsat_abstain_and_static_cfg_comparison() -> None:
    """SCENARIO-VERIFY-6769-MASK resolves all branches and exposes CFG limits."""

    cnf = _small_cnf()
    vocabulary = set(exp.lex_candidate("UNSAT c99,c1,c1")) | {exp.EOS_TOKEN}
    static = exp.StaticCFGProofMask(vocabulary)

    assert (
        exp.EnvironmentIndexedProofMask(exp.RuntimeGamma.from_cnf(cnf)).replay("UNSAT c1,c2,c3")[
            "terminal_reachable"
        ]
        is True
    )
    assert (
        exp.EnvironmentIndexedProofMask(exp.RuntimeGamma.from_cnf(cnf)).replay("UNSAT c99")[
            "blocked_token"
        ]
        == "c99"
    )
    assert (
        exp.EnvironmentIndexedProofMask(exp.RuntimeGamma.from_cnf(cnf)).replay("UNSAT c1,c1")[
            "blocked_token"
        ]
        == "c1"
    )
    abstain = exp.EnvironmentIndexedProofMask(exp.RuntimeGamma.from_cnf(cnf)).replay("ABSTAIN")
    assert abstain["terminal_reachable"] is True
    assert abstain["final_gamma"]["claim_branch"] == "ABSTAIN"

    assert static.replay("UNSAT c99")["terminal_reachable"] is True
    assert static.replay("UNSAT c1,c1")["terminal_reachable"] is True
    assert exp.compile_static_cfg()["compiled"] is True
    assert (
        exp.StaticCFGProofMask(vocabulary).replay("ABSTAIN extra")["candidate_reachable"] is False
    )


def test_scenario_verify_6769_draft_renderer_preserves_only_complete_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-6769-DRAFT never consults or invents exact evidence."""

    monkeypatch.setattr(
        exp.frozen,
        "exact_check_constraints",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("checker called")),
    )
    cnf = _small_cnf()
    complete = exp.render_draft_conditioned("SAT x3=1 x1=1 x2=1", exp.RuntimeGamma.from_cnf(cnf))
    incomplete = exp.render_draft_conditioned("SAT x1=1", exp.RuntimeGamma.from_cnf(cnf))
    invalid_core = exp.render_draft_conditioned("UNSAT c99", exp.RuntimeGamma.from_cnf(cnf))

    assert complete["rendered_output"] == "SAT x3=1 x1=1 x2=1"
    assert complete["intent_preserved"] is True
    assert complete["terminal_reachable"] is True
    assert incomplete["rendered_output"] == "ABSTAIN"
    assert invalid_core["rendered_output"] == "ABSTAIN"
    assert incomplete["intent_preserved"] is False
    assert complete["exact_checker_calls"] == incomplete["exact_checker_calls"] == 0
    assert complete["synthesized_slots"] == incomplete["synthesized_slots"] == 0
    assert complete["draft_sha256"] == exp.sha256_text("SAT x3=1 x1=1 x2=1")


def test_scenario_verify_6769_support_witnesses_remain_exact_valid_per_size_bin() -> None:
    """SCENARIO-VERIFY-6769-SUPPORT keeps one SAT and UNSAT witness per bin."""

    _panel, stream = _inputs()
    fixtures = exp.build_support_fixtures(stream)

    assert {(row["size_bin"], row["claim"]) for row in fixtures} == {
        ("small", "SAT"),
        ("small", "UNSAT"),
        ("medium", "SAT"),
        ("medium", "UNSAT"),
    }
    for fixture in fixtures:
        receipt = exp.EnvironmentIndexedProofMask(exp.RuntimeGamma.from_cnf(fixture["cnf"])).replay(
            fixture["candidate"]
        )
        checked = exp.exact_check_output(fixture["candidate"], fixture["cnf"])
        assert receipt["candidate_reachable"] is True
        assert receipt["terminal_reachable"] is True
        assert checked["valid"] is True

    with pytest.raises(ValueError, match="missing_support_witness"):
        exp.build_support_fixtures({"rows": stream["rows"][:-1]})
    missing_claim = deepcopy(stream)
    for row in missing_claim["rows"]:
        if row["size_bin"] == "small" and row["label"] == "SAT":
            row["label"] = "UNKNOWN"
    with pytest.raises(ValueError, match="missing_support_witness:small:SAT"):
        exp.build_support_fixtures(missing_claim)


def test_req_verify_6769_rows_cover_each_mode_fixture_and_attack() -> None:
    """REQ-VERIFY-6769 retains one comparison row for every case and mode."""

    panel, stream = _inputs()
    cases = exp.build_cases(panel, stream)
    rows, compile_receipt = exp.build_rows(cases)
    case_ids = {case["case_id"] for case in cases}

    assert compile_receipt["compiled"] is True
    assert len(rows) == len(cases) * len(exp.MODES)
    assert {row["mode"] for row in rows} == set(exp.MODES)
    assert all(
        {row["case_id"] for row in rows if row["mode"] == mode} == case_ids for mode in exp.MODES
    )
    assert {case["target_class"] for case in cases if case["case_kind"] == "attack"} >= {
        *exp.TARGET_CLASSES,
        "extra_text",
        "confusable",
        "adversarial_prefix",
        "support_collapse",
    }
    dynamic_rows = [
        row for row in rows if row["mode"] in {"environment_indexed", "draft_conditioned"}
    ]
    assert all(row["mask_invocation_count"] > 0 for row in dynamic_rows)
    assert all(row["ghost_symbols_emitted"] == [] for row in dynamic_rows)
    assert all(row["row_sha256"] == exp.row_checksum(row) for row in rows)

    collapse_rows = [row for row in rows if row["target_class"] == "support_collapse"]
    assert len(collapse_rows) == len(exp.MODES)
    assert all(row["candidate_reachable"] is True for row in collapse_rows)
    assert all(row["exact_check"]["valid"] is True for row in collapse_rows)


def test_scenario_verify_6769_artifact_is_row_derived_and_rejects_drift() -> None:
    """SCENARIO-VERIFY-6769-ARTIFACT recomputes every readiness input."""

    panel, stream = _inputs()
    preconditions = exp.evaluate_preconditions(
        panel,
        backend_receipt={"available": True, "backend": "llama_cpp.LlamaGrammar"},
        checker_receipt={"available": True, "checker": "exact_check_constraints"},
    )
    cases = exp.build_cases(panel, stream)
    rows, compile_receipt = exp.build_rows(cases)
    artifact = exp.build_artifact(
        date="20260830",
        duration_s=0.5,
        rows=rows,
        source_artifact_sha256="sha256:panel",
        source_stream_artifact_sha256="sha256:stream",
        preconditions=preconditions,
        static_cfg_compile_receipt=compile_receipt,
        deterministic_replay=True,
        verification_receipts=exp.verification_rows(
            exp.VERIFICATION_COMMANDS, _passing_runner, exp.REPO_ROOT
        ),
        code_files={"module": "sha256:code", "test": "sha256:test"},
    )

    assert artifact["dynamic_proof_grammar_ready"] is True
    assert artifact["valid_sat_reachable"] is True
    assert artifact["valid_unsat_reachable"] is True
    assert artifact["runtime_mask_invocation_count"] > 0
    assert artifact["no_ghost_violations"] == 0
    assert artifact["exact_authority_features_in_grammar"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(artifact) == set(artifact["field_principles"])
    assert (
        exp.validate_artifact(artifact, code_files={"module": "sha256:code", "test": "sha256:test"})
        == []
    )

    mutations = [
        ("runtime_mask_invocation_count", 0, "aggregate_recomputation_mismatch"),
        ("no_ghost_violations", 1, "aggregate_recomputation_mismatch"),
        ("dynamic_proof_grammar_ready", False, "readiness_gate_mismatch"),
        ("verifier_is_oracle", True, "verifier_is_oracle_mismatch"),
        ("verdict_class", "invalid", "verdict_class_invalid"),
    ]
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(
            changed, code_files={"module": "sha256:code", "test": "sha256:test"}
        )

    changed = deepcopy(artifact)
    changed["rows"][0]["emitted_output"] = "tampered"
    assert set(
        exp.validate_artifact(changed, code_files={"module": "sha256:code", "test": "sha256:test"})
    ) >= {
        "row_checksum_mismatch",
        "reproducibility_checksum_mismatch",
    }
    schema_drift = deepcopy(artifact)
    schema_drift["field_principles"].pop("title")
    schema_drift["inference_substrate"] = "wrong"
    schema_drift["gamma_schema"] = {}
    assert set(
        exp.validate_artifact(
            schema_drift, code_files={"module": "sha256:code", "test": "sha256:test"}
        )
    ) >= {"field_principles_mismatch", "inference_substrate_mismatch", "gamma_schema_mismatch"}
    gate_drift = deepcopy(artifact)
    gate_drift["gate_check_summary"]["checks"][0]["observed"] = False
    assert "gate_check_summary_mismatch" in exp.validate_artifact(
        gate_drift, code_files={"module": "sha256:code", "test": "sha256:test"}
    )
    authority_drift = deepcopy(artifact)
    authority_drift["exact_authority_features_in_grammar"] = ["answer_label"]
    assert "exact_authority_separation_mismatch" in exp.validate_artifact(
        authority_drift, code_files={"module": "sha256:code", "test": "sha256:test"}
    )
    assert exp.validate_artifact({}) == [
        "missing_required_fields:" + ",".join(sorted(exp.ARTIFACT_FIELDS))
    ]


def test_scenario_verify_6769_blocked_artifact_and_run_are_atomic(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6769-ARTIFACT publishes complete blocked evidence."""

    blocked = exp.build_blocked_artifact(
        date="20260830",
        duration_s=0.1,
        failed_check="local_grammar_backend_import",
        expected=True,
        observed={"available": False},
        source_artifact_sha256="sha256:panel",
        source_stream_artifact_sha256="not_checked",
        preconditions={"all_passed": False, "checks": []},
        code_files={"module": "sha256:code", "test": "sha256:test"},
    )
    assert blocked["status"] == "complete_blocked_dynamic_grammar_v2"
    assert blocked["honest_verdict"].startswith("complete_blocked_dynamic_grammar_v2")
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]["failed_check"] == "local_grammar_backend_import"
    assert (
        exp.validate_artifact(blocked, code_files={"module": "sha256:code", "test": "sha256:test"})
        == []
    )

    results = tmp_path / "results"
    results.mkdir()
    (results / exp.UPSTREAM_PANEL_PATH.name).write_text("[]", encoding="utf-8")
    artifact = exp.run("20260830", tmp_path, verification_runner=_passing_runner)
    written = json.loads((results / exp.RESULT_PATH.name).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["gate_check_summary"]["failed_check"] == "exp6768_json_object"

    panel, _stream = _inputs()
    gate_root = tmp_path / "gate"
    gate_results = gate_root / "results"
    gate_results.mkdir(parents=True)
    changed_panel = deepcopy(panel)
    changed_panel["targetable_panel_ready"] = False
    (gate_results / exp.UPSTREAM_PANEL_PATH.name).write_text(json.dumps(changed_panel))
    gate_block = exp.run("20260830", gate_root, verification_runner=_passing_runner)
    assert gate_block["gate_check_summary"]["failed_check"] == "exp6768_targetable_panel_ready"

    stream_root = tmp_path / "stream"
    stream_results = stream_root / "results"
    stream_results.mkdir(parents=True)
    (stream_results / exp.UPSTREAM_PANEL_PATH.name).write_text(json.dumps(panel))
    (stream_results / exp.UPSTREAM_STREAM_PATH.name).write_text("[]")
    stream_block = exp.run("20260830", stream_root, verification_runner=_passing_runner)
    assert stream_block["gate_check_summary"]["failed_check"] == "exp6744_json_object"

    changed = deepcopy(blocked)
    changed["rows"] = [{}]
    changed["dynamic_proof_grammar_ready"] = True
    changed["verdict_class"] = "partial"
    changed["honest_verdict"] = "wrong"
    assert set(
        exp.validate_artifact(changed, code_files={"module": "sha256:code", "test": "sha256:test"})
    ) >= {
        "blocked_rows_invalid",
        "blocked_readiness_mismatch",
        "blocked_verdict_class_mismatch",
        "blocked_verdict_prefix_mismatch",
    }
    with pytest.raises(ValueError):
        exp.write_json_atomic(
            tmp_path / "bad.json",
            changed,
            code_files={"module": "sha256:code", "test": "sha256:test"},
        )


def test_req_verify_6769_run_writes_ready_artifact_with_live_masks(tmp_path: Path) -> None:
    """REQ-VERIFY-6769 runs the complete local no-LLM fixture path."""

    results = tmp_path / "results"
    results.mkdir()
    for source in (exp.UPSTREAM_PANEL_PATH, exp.UPSTREAM_STREAM_PATH):
        shutil.copy2(exp.REPO_ROOT / source, results / source.name)

    artifact = exp.run("20260830", tmp_path, verification_runner=_passing_runner)
    written = exp.load_json_object(results / exp.RESULT_PATH.name)

    assert written == artifact
    assert artifact["dynamic_proof_grammar_ready"] is True
    assert artifact["inference_substrate"] == "deterministic_automaton_no_llm"
    assert artifact["source_artifact_sha256"] == exp.sha256_file(
        results / exp.UPSTREAM_PANEL_PATH.name
    )
    assert artifact["duration_s"] > 0.0
    assert exp.validate_artifact(artifact) == []


def test_req_verify_6769_verification_failure_prevents_readiness(tmp_path: Path) -> None:
    """REQ-VERIFY-6769 does not report ready when focused verification fails."""

    panel, stream = _inputs()
    preconditions = exp.evaluate_preconditions(
        panel,
        backend_receipt={"available": True},
        checker_receipt={"available": True},
    )
    rows, compile_receipt = exp.build_rows(exp.build_cases(panel, stream))

    def failing_runner(command: str, _root: Path) -> dict:
        return {
            "command": command,
            "exit_code": 1 if command == exp.FOCUSED_COMMAND else 0,
            "stdout": "TOTAL 100 0 100%" if "coverage report" in command else "failed",
            "stderr": "",
            "duration_s": 0.01,
        }

    artifact = exp.build_artifact(
        date="20260830",
        duration_s=0.2,
        rows=rows,
        source_artifact_sha256="sha256:panel",
        source_stream_artifact_sha256="sha256:stream",
        preconditions=preconditions,
        static_cfg_compile_receipt=compile_receipt,
        deterministic_replay=True,
        verification_receipts=exp.verification_rows(
            exp.VERIFICATION_COMMANDS, failing_runner, tmp_path
        ),
        code_files={"module": "sha256:code", "test": "sha256:test"},
    )
    assert artifact["dynamic_proof_grammar_ready"] is False
    assert artifact["verdict_class"] == "partial"
    assert "focused_tests" in artifact["gate_check_summary"]["failed_checks"]


def test_req_verify_6769_spec_cli_and_helper_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6769 keeps the spec, CLI, and strict helper contract visible."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md").read_text()
    assert "REQ-VERIFY-6769" in spec
    assert "SCENARIO-VERIFY-6769-MASK" in spec
    assert exp.parse_args([]).date == "20260830"
    assert exp.parse_args(["--date", "20260901"]).date == "20260901"
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object required"):
        exp.load_json_object(array)
    assert exp.first_failed_check({"checks": [{"check": "x", "passed": False}]})["check"] == "x"

    base = exp.TokenMaskSession()
    assert base.state_snapshot() is None
    with pytest.raises(NotImplementedError):
        base.allowed_tokens()
    with pytest.raises(NotImplementedError):
        base.advance("SAT")

    real_import = exp.importlib.import_module
    monkeypatch.setattr(
        exp.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("absent")),
    )
    assert exp.backend_import_receipt()["available"] is False
    assert exp.compile_static_cfg()["compiled"] is False
    monkeypatch.setattr(exp.importlib, "import_module", real_import)

    assert exp.exact_check_output("ABSTAIN", _small_cnf())["attempted"] is False
    monkeypatch.setattr(
        exp.encoder_a,
        "encode_certificate",
        lambda _parsed: (_ for _ in ()).throw(ValueError("rejected")),
    )
    assert exp.exact_check_output("SAT x1=1 x2=1 x3=1", _small_cnf())["reason"] == (
        "encoder_rejected:rejected"
    )

    monkeypatch.setattr(
        exp.inspect,
        "getsource",
        lambda _value: "def probe():\n    return exact_check_constraints\n",
    )
    assert exp.exact_authority_features_in_mechanism() == ["exact_check_constraints"]
