"""Tests for Exp6326 restricted policy contract compiler.

Spec refs: REQ-KONA-6326, SCENARIO-KONA-6326-CANONICAL-PARSER,
SCENARIO-KONA-6326-FACTOR-EXACTNESS,
SCENARIO-KONA-6326-FALLBACK-AND-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6326_restricted_policy_contract_compiler as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6326_restricted_policy_contract_compiler "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6326_restricted_policy_contract_compiler.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6326_restricted_policy_contract_compiler.py "
    "-m pytest tests/python/test_experiment_6326_restricted_policy_contract_compiler.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6326_restricted_policy_contract_compiler.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6326_restricted_policy_contract_compiler.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6326_restricted_policy_contract_compiler.json"
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


def test_req_kona_6326_spec_declares_artifact_contract() -> None:
    """REQ-KONA-6326: OpenSpec anchors the DSL, checker, and artifact."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-KONA-6326") :]

    for marker in (
        "SCENARIO-KONA-6326-CANONICAL-PARSER",
        "SCENARIO-KONA-6326-FACTOR-EXACTNESS",
        "SCENARIO-KONA-6326-FALLBACK-AND-ATTACKS",
        "results/experiment_6326_restricted_policy_contract_compiler.json",
        "bounded typed DSL",
        "verifier_is_oracle=true",
    ):
        assert marker in section


def test_scenario_kona_6326_parser_totality_and_canonical_property() -> None:
    """SCENARIO-KONA-6326-CANONICAL-PARSER: syntax is total and canonical."""

    canonical_program = mod.parse_policy(
        """
        # comments and order do not affect semantics
        policy access
        actions: allow, challenge, deny;
        states: guest, member, banned;
        rule member -> challenge;
        rule banned -> deny;
        rule guest -> allow;
        end
        """
    )
    reordered_program = mod.parse_policy(
        """
        policy renamed
        states: banned, guest, member;
        actions: deny, allow, challenge;
        rule banned -> deny;

        # same finite map as the first program
        rule guest -> allow;
        rule member -> challenge;
        end
        """
    )

    assert mod.normalize_policy(canonical_program) == mod.normalize_policy(reordered_program)
    assert mod.semantic_hash(canonical_program) == mod.semantic_hash(reordered_program)

    for variant in mod.normalization_variants(canonical_program):
        assert mod.normalize_policy(mod.parse_policy(variant)) == mod.normalize_policy(
            canonical_program
        )

    rejection_cases = {
        "policy p\n": "unknown_syntax",
        "not_policy p\nstates: s0;\nactions: a0;\nrule s0 -> a0;\nend\n": "unknown_syntax",
        "policy p\nstates: s0;\nactions: a0;\nrule s0 -> a0;\n": "unknown_syntax",
        "policy p\nstates: s0;\nstates: s1;\nactions: a0;\nrule s0 -> a0;\nend\n": "duplicate_states",
        "policy p\nactions: a0;\nrule s0 -> a0;\nend\n": "missing_states",
        "policy p\nstates: s0;\nrule s0 -> a0;\nend\n": "missing_actions",
        "policy p\nstates: s0, ;\nactions: a0;\nrule s0 -> a0;\nend\n": "unknown_syntax",
        "policy p\nstates: s0, s0;\nactions: a0;\nrule s0 -> a0;\nend\n": "duplicate_identifier",
        "policy p\nstates: s-;\nactions: a0;\nrule s -> a0;\nend\n": "invalid_identifier",
        "policy p\nstates: s0;\nactions: a0;\nunknown s0 -> a0;\nend\n": "unknown_syntax",
        "policy p\nstates: s0;\nactions: a0;\nrule s0 -> a0;\nrule s0 -> a0;\nend\n": "duplicate_rule",
        "policy p\nstates: s0,s1;\nactions: a0;\nrule s0 -> a0;\nend\n": "missing_state_actions",
        "policy p\nstates: s0;\nactions: a0;\nrule s1 -> a0;\nend\n": "unknown_state",
        "policy p\nstates: s0;\nactions: a0;\nrule s0 -> a1;\nend\n": "unknown_action",
        mod.program_text(
            name="too_many",
            states=("s0", "s1", "s2", "s3", "s4"),
            actions=("a0",),
            mapping={"s0": "a0", "s1": "a0", "s2": "a0", "s3": "a0", "s4": "a0"},
        ): "state_bound",
    }
    for source, reason in rejection_cases.items():
        with pytest.raises(mod.PolicySyntaxError, match=reason):
            mod.parse_policy(source)


def test_req_kona_6326_contract_schema_rejects_malformed_inputs() -> None:
    """REQ-KONA-6326: contract schema rejects malformed finite domains."""

    valid = {
        "family": "unit",
        "split": "development",
        "states": ("s0", "s1"),
        "actions": ("a0", "a1"),
        "clauses": ({"kind": "same_action", "state": "s0", "other_state": "s1", "weight": 1},),
    }
    contract = mod.validate_contract(valid)
    factors = mod.compile_contract_to_factors(contract)
    same = mod.parse_policy(
        "policy p\nstates: s0, s1;\nactions: a0, a1;\nrule s0 -> a0;\nrule s1 -> a0;\nend\n"
    )
    different = mod.parse_policy(
        "policy p\nstates: s0, s1;\nactions: a0, a1;\nrule s0 -> a0;\nrule s1 -> a1;\nend\n"
    )

    assert factors[0].satisfied(same) is True
    assert mod.exact_contract_energy(different, contract) == 1
    assert mod.factor_energy(different, factors) == 1

    malformed_contracts: list[tuple[dict[str, object], type[Exception], str]] = [
        ({**valid, "split": "future"}, mod.ContractValidationError, "unknown_split"),
        ({**valid, "clauses": "bad"}, mod.ContractValidationError, "clauses_type"),
        ({**valid, "clauses": ("bad",)}, mod.ContractValidationError, "clause_type"),
        ({**valid, "family": 7}, mod.ContractValidationError, "invalid_family"),
        ({**valid, "family": "Bad"}, mod.ContractValidationError, "invalid_family"),
        ({**valid, "states": "s0"}, mod.ContractValidationError, "states_type"),
        ({**valid, "states": ("s0", "s0")}, mod.ContractValidationError, "duplicate_states"),
        ({**valid, "states": ()}, mod.PolicySyntaxError, "state_bound_empty"),
        (
            {
                **valid,
                "clauses": (
                    {"kind": "require_action", "state": "missing", "action": "a0", "weight": 1},
                ),
            },
            mod.ContractValidationError,
            "unknown_state",
        ),
        (
            {
                **valid,
                "clauses": (
                    {"kind": "require_action", "state": "s0", "action": "missing", "weight": 1},
                ),
            },
            mod.ContractValidationError,
            "unknown_action",
        ),
        (
            {
                **valid,
                "clauses": (
                    {"kind": "require_action", "state": "s0", "action": "a0", "weight": 0},
                ),
            },
            mod.ContractValidationError,
            "invalid_weight",
        ),
        (
            {
                **valid,
                "clauses": ({"kind": "allow_actions", "state": "s0", "actions": (), "weight": 1},),
            },
            mod.ContractValidationError,
            "empty_action_set",
        ),
        (
            {
                **valid,
                "clauses": ({"kind": "unknown_kind", "state": "s0", "action": "a0", "weight": 1},),
            },
            mod.ContractValidationError,
            "unknown_clause_kind",
        ),
    ]
    for payload, exc_type, reason in malformed_contracts:
        with pytest.raises(exc_type, match=reason):
            mod.validate_contract(payload)

    with pytest.raises(mod.PolicySyntaxError, match="invalid_identifier"):
        mod._require_identifier(7, "identifier", syntax=True)


def test_scenario_kona_6326_factor_energy_matches_exact_violations_property() -> None:
    """SCENARIO-KONA-6326-FACTOR-EXACTNESS: factors equal exact violations."""

    for fixture in mod.build_fixture_manifest():
        contract = mod.validate_contract(fixture.contract)
        factors = mod.compile_contract_to_factors(contract)
        policies = mod.enumerate_policy_semantics(contract.states, contract.actions)
        assert policies

        for factor in factors:
            assert set(factor.scope) <= set(contract.states)
            assert factor.weight > 0

        for policy in policies:
            assert mod.factor_energy(policy, factors) == mod.exact_contract_energy(
                policy, contract
            )

    summary = mod.factor_energy_exactness_results(mod.build_fixture_manifest())
    assert summary["all_passed"] is True
    assert summary["checked_policy_count"] > 100
    assert summary["mismatch_count"] == 0


def test_scenario_kona_6326_fallbacks_are_hash_pinned_and_attacks_fail(tmp_path: Path) -> None:
    """SCENARIO-KONA-6326-FALLBACK-AND-ATTACKS: fallbacks pass and attacks fail."""

    data_dir = tmp_path / "data"
    sidecars = mod.write_sidecars(data_dir, date="20260812")
    fallback_results = mod.verify_fallbacks(mod.build_fixture_manifest(), sidecars.fallbacks)
    attack_results = mod.attack_control_results(
        mod.build_fixture_manifest(),
        sidecars.fallbacks,
        test_file_text=Path(__file__).read_text(encoding="utf-8"),
    )

    assert fallback_results["all_passed"] is True
    assert fallback_results["verified_family_count"] == len(mod.FAMILY_ORDER)
    assert attack_results["all_attacks_failed_closed"] is True
    assert attack_results["vacuous_contract"]["rejected"] is True
    assert attack_results["parser_default"]["rejected"] is True
    assert attack_results["validator_mutation"]["detected"] is True
    assert attack_results["test_deletion"]["detected"] is True
    assert attack_results["fallback_laundering"]["rejected"] is True
    assert attack_results["hash_swap"]["rejected"] is True
    assert attack_results["nondeterministic_normalization"]["detected"] is False

    for fallback in sidecars.fallbacks.values():
        assert fallback.path.exists()
        assert mod.sha256_file(fallback.path) == fallback.source_sha256
        assert mod.parse_policy(fallback.path.read_text(encoding="utf-8"))

    bad_source_result = mod.verify_fallback_program(
        mod.build_fixture_manifest()[0],
        "policy broken\n",
        expected_source_sha256="sha256:" + "0" * 64,
        expected_semantic_hash="sha256:" + "1" * 64,
    )
    assert bad_source_result["verified"] is False
    assert bad_source_result["reason"].startswith("unknown_syntax")


def test_req_kona_6326_artifact_schema_and_replayable_sidecars(tmp_path: Path) -> None:
    """REQ-KONA-6326: terminal artifact carries all required receipts."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    data_dir = tmp_path / "sidecars"
    artifact = mod.run(
        date="20260812",
        result_path=result_path,
        data_dir=data_dir,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["status"] == "complete_ready"
    assert artifact["contract_guard_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["generated_label_count"] == 0
    assert type(artifact["generated_label_count"]) is int
    assert artifact["hidden_state_access_count"] == 0
    assert type(artifact["hidden_state_access_count"]) is int
    assert artifact["external_text_scorer_count"] == 0
    assert type(artifact["external_text_scorer_count"]) is int
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["factor_energy_exactness_results"]["all_passed"] is True
    assert artifact["parser_rejection_and_totality_results"]["all_passed"] is True
    assert artifact["exhaustive_contract_results_by_family"]["all_families_passed"] is True
    assert artifact[
        "vacuous_contract_parser_default_validator_mutation_test_deletion_fallback_laundering_and_hash_swap_results"
    ]["all_attacks_failed_closed"] is True
    assert artifact["exact_oracle_claim_boundary"]["oracle_distinct_verifier_claim"] is False
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert artifact["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
        assert artifact["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]

    for receipt_field in (
        "dsl_grammar_path_and_hash",
        "contract_schema_path_and_hash",
        "fixture_manifest_path_and_hash",
    ):
        receipt = artifact[receipt_field]
        assert Path(receipt["path"]).exists()
        assert mod.sha256_file(Path(receipt["path"])) == receipt["sha256"]


def test_req_kona_6326_validator_fails_closed_for_false_readiness(tmp_path: Path) -> None:
    """REQ-KONA-6326: artifact validation rejects readiness laundering."""

    artifact = mod.run(
        date="20260812",
        result_path=tmp_path / "artifact.json",
        data_dir=tmp_path / "data",
        duration_s=0.1,
        write=False,
    )

    missing = deepcopy(artifact)
    missing.pop("factor_compiler_path_and_hash")
    with pytest.raises(ValueError, match="factor_compiler_path_and_hash"):
        mod.validate_artifact(missing)

    bad_count = deepcopy(artifact)
    bad_count["generated_label_count"] = False
    bad_count["reproducibility_checksum"] = mod.payload_checksum(bad_count)
    with pytest.raises(ValueError, match="generated_label_count"):
        mod.validate_artifact(bad_count)

    bad_score = deepcopy(artifact)
    bad_score["factor_energy_exactness_results"]["all_passed"] = False
    bad_score["contract_guard_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = mod.payload_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)

    blocked = deepcopy(artifact)
    blocked["factor_energy_exactness_results"]["all_passed"] = False
    blocked["contract_guard_ready_score"] = 0.0
    blocked["status"] = "blocked"
    blocked["honest_verdict"] = mod._honest_verdict("blocked")
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    assert mod.validate_artifact(blocked) is True

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)


def test_req_kona_6326_cli_writes_requested_paths(tmp_path: Path) -> None:
    """REQ-KONA-6326: the required module command writes the artifact."""

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
    assert payload["contract_guard_ready_score"] == 1.0
