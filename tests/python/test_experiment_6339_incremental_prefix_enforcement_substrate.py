"""Tests for Exp6339 incremental prefix enforcement substrate.

Spec refs: REQ-KONA-6339, SCENARIO-KONA-6339-PREFIX-SOUNDNESS,
SCENARIO-KONA-6339-FEASIBLE-RECALL, SCENARIO-KONA-6339-SEMANTIC-PARITY,
SCENARIO-KONA-6339-OBSERVABLE-STATE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6326_restricted_policy_contract_compiler as exp6326
from carnot import experiment_6339_incremental_prefix_enforcement_substrate as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6339_incremental_prefix_enforcement_substrate "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6339_incremental_prefix_enforcement_substrate.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6339_incremental_prefix_enforcement_substrate.py "
    "-m pytest tests/python/test_experiment_6339_incremental_prefix_enforcement_substrate.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6339_incremental_prefix_enforcement_substrate.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6339_incremental_prefix_enforcement_substrate.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6339_incremental_prefix_enforcement_substrate.json"
)
TEST_COMMANDS = [
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def test_req_kona_6339_spec_declares_prefix_contract() -> None:
    """REQ-KONA-6339: OpenSpec anchors the prefix substrate and artifact."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-KONA-6339") :]

    for marker in (
        "SCENARIO-KONA-6339-PREFIX-SOUNDNESS",
        "SCENARIO-KONA-6339-FEASIBLE-RECALL",
        "SCENARIO-KONA-6339-SEMANTIC-PARITY",
        "SCENARIO-KONA-6339-OBSERVABLE-STATE",
        "results/experiment_6339_incremental_prefix_enforcement_substrate.json",
        "JIT SMT prefix-feasibility checker",
        "`hidden_state_access_count`, `generated_label_count`, and",
    ):
        assert marker in section


def test_scenario_kona_6339_observable_parser_state_is_deterministic() -> None:
    """SCENARIO-KONA-6339-OBSERVABLE-STATE: state features are observable."""

    prefix = (
        b"policy p\n"
        b"states: s0, s1;\n"
        b"actions: a0, a1;\n"
        b"rule s0 -> a0;\n"
    )
    first = mod.incremental_parse(prefix).to_dict()
    second = mod.incremental_parse(prefix).to_dict()

    assert first == second
    assert first["status"] == "viable"
    assert first["phase"] == "in_rules"
    assert first["declared_state_count"] == 2
    assert first["declared_action_count"] == 2
    assert first["rule_count"] == 1
    assert first["missing_state_actions"] == ["s1"]
    assert "rule" in first["expected_next"]
    assert first["observable_only"] is True
    assert first["hidden_state_fields"] == []

    split = mod.incremental_parse(
        b"policy p\nstates: s0;\nactions: a"
    ).to_dict()
    assert split["status"] == "viable"
    assert split["trailing_fragment"] == "actions: a"

    bad_utf8 = mod.incremental_parse(b"policy p\n\xff").to_dict()
    assert bad_utf8["status"] == "rejected"
    assert bad_utf8["error_reason"] == "invalid_utf8"


def test_scenario_kona_6339_completed_programs_match_exp6326() -> None:
    """SCENARIO-KONA-6339-SEMANTIC-PARITY: complete programs share semantics."""

    for fixture in exp6326.build_fixture_manifest():
        policy = exp6326.parse_policy(fixture.fallback_program)
        for variant in exp6326.normalization_variants(policy):
            parsed = mod.parse_completed_program(variant)
            exp_policy = exp6326.parse_policy(variant)
            assert parsed.accepted is True
            assert parsed.normalized_source == exp6326.normalize_policy(exp_policy)
            assert parsed.semantic_hash == exp6326.semantic_hash(exp_policy)

    bad = mod.parse_completed_program("policy p\nstates: s0;\nactions: a0;\nrule s0 -> a9;\nend\n")
    assert bad.accepted is False
    assert bad.error_reason == "unknown_action"


def test_scenario_kona_6339_jit_checker_accept_reject_and_timeout() -> None:
    """SCENARIO-KONA-6339-PREFIX-SOUNDNESS: the JIT checker is fail-closed."""

    checker = mod.PrefixFeasibilityChecker(timeout_ms=mod.DEFAULT_TIMEOUT_MS)
    valid = "policy p\nstates: s0;\nactions: a0;\nrule s0 -> a0;\n"
    accepted = checker.check(valid)
    assert accepted.verdict == "accept"
    assert accepted.fail_closed is False
    assert accepted.feasible is True
    assert accepted.solver == "z3"

    rejected = checker.check("policy p\nstates: s0;\nactions: a0;\nrule s0 -> a9;\n")
    assert rejected.verdict == "reject"
    assert rejected.fail_closed is True
    assert rejected.feasible is False

    timed_out = mod.PrefixFeasibilityChecker(timeout_ms=0).check(valid)
    assert timed_out.verdict == "timeout"
    assert timed_out.fail_closed is True
    assert timed_out.feasible is None


def test_scenarios_kona_6339_exhaustive_prefix_results_pass() -> None:
    """SCENARIO-KONA-6339-PREFIX-SOUNDNESS and FEASIBLE-RECALL: exhaustive pass."""

    manifest = mod.build_prefix_fixture_manifest()
    results = mod.exhaustive_prefix_results(manifest)

    assert results["exhaustive_prefix_count"] > 100
    assert results["feasible_infeasible_and_timeout_counts"]["timeout"] == 0
    assert results["prefix_soundness_results"]["all_passed"] is True
    assert results["feasible_completion_recall_results"]["all_passed"] is True
    assert results["completed_program_semantic_parity_results"]["all_passed"] is True
    assert results["parser_state_determinism_results"]["all_passed"] is True
    assert results["verification_calls_time_and_cost_distribution"]["call_count"] == results[
        "exhaustive_prefix_count"
    ]
    assert results["verification_cost_error_table"]["false_reject_count"] == 0
    assert results["verification_cost_error_table"]["false_accept_count"] == 0


def test_scenario_kona_6339_adversarial_prefix_controls_fail_closed() -> None:
    """REQ-KONA-6339: adversarial prefixes are bounded and fail closed."""

    attacks = mod.adversarial_prefix_results()

    assert attacks["all_passed"] is True
    assert attacks["invalid_utf8"]["verdict"] == "reject"
    assert attacks["token_split"]["verdict"] == "accept"
    assert attacks["whitespace_aliases"]["semantic_hashes_match"] is True
    assert attacks["prefix_bomb"]["verdict"] in {"reject", "timeout"}
    assert attacks["solver_timeout"]["verdict"] == "timeout"
    assert attacks["unknown_symbols"]["verdict"] == "reject"
    assert attacks["normalization_collisions"]["collision_found"] is False


def test_req_kona_6339_parser_and_checker_edge_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KONA-6339: defensive parser and cost branches stay explicit."""

    assert mod.incremental_lex("Policy").to_dict()["tokens"][0]["kind"] == "UNKNOWN"
    assert mod.incremental_lex(b"a" * (mod.MAX_PREFIX_BYTES + 1)).error_reason == "prefix_too_long"
    assert mod.parse_completed_program(b"\xff").to_dict()["error_reason"] == "invalid_utf8"
    assert mod._percentile([], 0.5) == 0.0
    assert json.loads(mod._canonical_json({"items": ("a",)})) == {"items": ["a"]}

    parser_cases = {
        "policy p\n!\n": "unknown_syntax",
        "policy p\n\n": None,
        "bad p\n": "unknown_syntax",
        "policy p\nstates: s0;\nstates: s1;\n": "duplicate_states",
        "policy p\nstates: s-;\n": "invalid_identifier",
        "policy p\nrule s1 -> a0;\nstates: s0;\n": "unknown_state",
        "policy p\nrule s0 -> a1;\nactions: a0;\n": "unknown_action",
        "policy p\nrule s0 -> a0;\nrule s0 -> a0;\n": "duplicate_rule",
        "policy p\nstates: s0;\nrule s1 -> a0;\n": "unknown_state",
        "policy p\nchoose s0 -> a0;\n": "unknown_syntax",
        (
            "policy p\nstates: s0;\nactions: a0;\nrule s0 -> a0;\n"
            "end\nrule s0 -> a0;\n"
        ): "unknown_syntax",
        (
            "policy p\nstates: s0, s1;\nactions: a0;\n"
            "rule s0 -> a0;\nend\n"
        ): "missing_state_actions",
    }
    for source, reason in parser_cases.items():
        state = mod.incremental_parse(source)
        if reason is None:
            assert state.status == "viable"
        else:
            assert state.status == "rejected"
            assert state.error_reason == reason

    assert mod.incremental_parse("policy p\nactions: a0;\n").phase == "after_actions"
    state = mod.incremental_parse("")
    assert mod._fragment_is_viable(" # only comment", state) is True
    assert mod.incremental_parse("policy p\n$").error_reason == "unknown_syntax"

    class UnsatSolver:
        def set(self, **_: object) -> None:
            return None

        def add(self, *_: object) -> None:
            return None

        def check(self) -> object:
            return mod.z3.unsat

    with monkeypatch.context() as patcher:
        patcher.setattr(mod.z3, "Solver", lambda: UnsatSolver())
        result = mod.PrefixFeasibilityChecker().check("policy p\n")
        assert result.verdict == "reject"
        assert result.reason == "smt_unsat"

    class UnknownSolver(UnsatSolver):
        def check(self) -> object:
            return mod.z3.unknown

    with monkeypatch.context() as patcher:
        patcher.setattr(mod.z3, "Solver", lambda: UnknownSolver())
        result = mod.PrefixFeasibilityChecker().check("policy p\n")
        assert result.verdict == "timeout"
        assert result.reason == "smt_unknown_or_timeout"

    source = mod._completed_sources()[0]
    with monkeypatch.context() as patcher:
        patcher.setattr(
            mod,
            "parse_completed_program",
            lambda _: mod.CompletedProgramReceipt(False, None, None, "forced"),
        )
        assert mod._completed_program_semantic_parity({source})["mismatch_count"] == 1

    class FakeState:
        def __init__(self, value: int) -> None:
            self.value = value

        def to_dict(self) -> dict[str, int]:
            return {"value": self.value}

    counter = {"value": 0}

    def changing_parse(_: str) -> FakeState:
        counter["value"] += 1
        return FakeState(counter["value"])

    with monkeypatch.context() as patcher:
        patcher.setattr(mod, "incremental_parse", changing_parse)
        assert mod._parser_state_determinism(["policy p\n"])["mismatch_count"] == 1

    with monkeypatch.context() as patcher:
        patcher.setattr(mod.exp6326, "semantic_hash", lambda _: "sha256:constant")
        assert mod._normalization_collision_results()["collision_found"] is True


def test_req_kona_6339_artifact_schema_sidecars_and_validation(tmp_path: Path) -> None:
    """REQ-KONA-6339: terminal artifact carries every required receipt."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    data_dir = tmp_path / "sidecars"
    artifact = mod.run(
        date="20260812",
        result_path=result_path,
        data_dir=data_dir,
        duration_s=2.5,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["status"] == "complete_ready"
    assert artifact["prefix_enforcement_substrate_ready_score"] == 1.0
    assert artifact["hidden_state_access_count"] == 0
    assert type(artifact["hidden_state_access_count"]) is int
    assert artifact["generated_label_count"] == 0
    assert type(artifact["generated_label_count"]) is int
    assert artifact["llm_call_count"] == 0
    assert type(artifact["llm_call_count"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["exact_oracle_claim_boundary"]["oracle_distinct_verifier_claim"] is False
    assert artifact["parser_state_observable_only_receipt"]["observable_only"] is True
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert artifact["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
        assert artifact["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]

    for receipt_field in (
        "parser_state_schema_path_and_hash",
        "prefix_fixture_manifest_path_and_hash",
    ):
        receipt = artifact[receipt_field]
        assert (REPO / receipt["path"]).exists()
        assert mod.sha256_file(REPO / receipt["path"]) == receipt["sha256"]


def test_req_kona_6339_validator_rejects_false_readiness(tmp_path: Path) -> None:
    """REQ-KONA-6339: artifact validation rejects readiness laundering."""

    artifact = mod.run(
        date="20260812",
        result_path=tmp_path / "artifact.json",
        data_dir=tmp_path / "data",
        duration_s=0.1,
        write=False,
    )

    missing = deepcopy(artifact)
    missing.pop("jit_smt_prefix_checker_path_and_hash")
    with pytest.raises(ValueError, match="jit_smt_prefix_checker_path_and_hash"):
        mod.validate_artifact(missing)

    bad_count = deepcopy(artifact)
    bad_count["llm_call_count"] = False
    bad_count["reproducibility_checksum"] = mod.payload_checksum(bad_count)
    with pytest.raises(ValueError, match="llm_call_count"):
        mod.validate_artifact(bad_count)

    bad_score = deepcopy(artifact)
    bad_score["feasible_completion_recall_results"]["all_passed"] = False
    bad_score["prefix_enforcement_substrate_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = mod.payload_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)

    blocked = deepcopy(artifact)
    blocked["feasible_completion_recall_results"]["all_passed"] = False
    blocked["prefix_enforcement_substrate_ready_score"] = 0.0
    blocked["status"] = "blocked"
    blocked["honest_verdict"] = mod._honest_verdict("blocked")
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    assert mod.validate_artifact(blocked) is True


def test_req_kona_6339_cli_writes_requested_paths(tmp_path: Path) -> None:
    """REQ-KONA-6339: the required module command writes the artifact."""

    result_path = tmp_path / "cli_artifact.json"
    data_dir = tmp_path / "cli_data"

    assert mod.main(
        [
            "--date",
            "20260812",
            "--result-path",
            str(result_path),
            "--data-dir",
            str(data_dir),
        ]
    ) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete_ready"
    assert payload["prefix_enforcement_substrate_ready_score"] == 1.0
