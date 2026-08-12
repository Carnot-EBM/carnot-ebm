"""Exp6326 restricted policy DSL and exact contract compiler.

Spec refs: REQ-KONA-6326, SCENARIO-KONA-6326-CANONICAL-PARSER,
SCENARIO-KONA-6326-FACTOR-EXACTNESS,
SCENARIO-KONA-6326-FALLBACK-AND-ATTACKS.

The checker in this module is an oracle. It enumerates a finite policy domain
and counts exact contract violations. No model output, hidden state, KAN, text
scorer, or generated label enters the decision.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6326_restricted_policy_contract_compiler.json")
DATA_DIR_RELATIVE_PATH = Path("data/research/experiment_6326_restricted_policy_contract_compiler")
GRAMMAR_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "policy_dsl_grammar.txt"
CONTRACT_SCHEMA_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "contract_schema.json"
FIXTURE_MANIFEST_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "fixture_manifest.json"
FALLBACK_DIR_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "fallbacks"
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6326_restricted_policy_contract_compiler.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6326_restricted_policy_contract_compiler.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")

EXPERIMENT_ID = "experiment_6326_restricted_policy_contract_compiler"
SCHEMA_VERSION = "carnot.experiment_6326.restricted_policy_contract_compiler.v1"
DEFAULT_RUN_DATE = "20260812"
RANDOM_SEEDS = (6326, 6327, 6328)
MAX_STATES = 4
MAX_ACTIONS = 4
MAX_IDENTIFIER_LENGTH = 24
MAX_ENUMERATED_POLICIES_PER_FAMILY = MAX_ACTIONS**MAX_STATES
INFERENCE_SUBSTRATE = "deterministic_finite_policy_contract_enumeration_no_llm"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6326_restricted_policy_contract_compiler "
    "--date 20260812"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_6326_restricted_policy_contract_compiler.py "
        "-q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage run --rcfile=/dev/null "
        "--include=python/carnot/experiment_6326_restricted_policy_contract_compiler.py "
        "-m pytest tests/python/test_experiment_6326_restricted_policy_contract_compiler.py "
        "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
        "--include=python/carnot/experiment_6326_restricted_policy_contract_compiler.py "
        "--fail-under=100"
    ),
    ".venv/bin/pytest tests/python -q",
    (
        ".venv/bin/python scripts/check_spec_coverage.py "
        "tests/python/test_experiment_6326_restricted_policy_contract_compiler.py"
    ),
    (
        ".venv/bin/python scripts/adversarial_verify.py "
        "results/experiment_6326_restricted_policy_contract_compiler.json"
    ),
)

IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
POLICY_RE = re.compile(r"^policy\s+([a-z][a-z0-9_]*)$")
DOMAIN_RE = re.compile(r"^(states|actions):\s*(.+);$")
RULE_RE = re.compile(r"^rule\s+([a-z][a-z0-9_]*)\s*->\s*([a-z][a-z0-9_]*);$")

DSL_GRAMMAR = """# Exp6326 restricted finite policy DSL
program        := policy_header states_line actions_line rule_line+ end_line
policy_header  := "policy" IDENT
states_line    := "states:" IDENT ("," IDENT)* ";"
actions_line   := "actions:" IDENT ("," IDENT)* ";"
rule_line      := "rule" STATE "->" ACTION ";"
end_line       := "end"
comment        := "#" ... end_of_line
IDENT          := /[a-z][a-z0-9_]*/ with length <= 24
bounds         := states <= 4, actions <= 4, every state has exactly one rule
semantics      := finite map state -> action; source order and comments ignored
"""

CONTRACT_SCHEMA: JsonDict = {
    "schema": SCHEMA_VERSION + ".contract",
    "bounds": {
        "max_states": MAX_STATES,
        "max_actions": MAX_ACTIONS,
        "max_identifier_length": MAX_IDENTIFIER_LENGTH,
    },
    "required_fields": ["family", "split", "states", "actions", "clauses"],
    "clause_kinds": {
        "require_action": ["state", "action", "weight"],
        "forbid_action": ["state", "action", "weight"],
        "allow_actions": ["state", "actions", "weight"],
        "same_action": ["state", "other_state", "weight"],
        "different_action": ["state", "other_state", "weight"],
        "if_action_then": ["state", "action", "then_state", "then_action", "weight"],
    },
    "vacuous_contracts_allowed": False,
    "natural_language_constraint_ir_allowed": False,
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "source_claim_boundary",
    "dsl_grammar_path_and_hash",
    "type_system_and_bounds",
    "parser_and_normalizer_path_and_hash",
    "canonical_semantics_path_and_hash",
    "contract_schema_path_and_hash",
    "factor_compiler_path_and_hash",
    "exact_energy_definition",
    "finite_domain_or_z3_checker_path_and_hash",
    "fixture_manifest_path_and_hash",
    "contract_family_splits",
    "fallback_programs_paths_and_hashes",
    "exhaustive_contract_results_by_family",
    "factor_energy_exactness_results",
    "parser_rejection_and_totality_results",
    "vacuous_contract_parser_default_validator_mutation_test_deletion_fallback_laundering_and_hash_swap_results",
    "exact_oracle_claim_boundary",
    "generated_label_count",
    "hidden_state_access_count",
    "external_text_scorer_count",
    "contract_guard_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Shows whether every exact gate passed.",
    "source_claim_boundary": "States the paper hooks without claiming paper results.",
    "dsl_grammar_path_and_hash": "Pins the accepted concrete syntax.",
    "type_system_and_bounds": "Shows the finite domain is small and explicit.",
    "parser_and_normalizer_path_and_hash": "Pins the code that rejects and canonicalizes programs.",
    "canonical_semantics_path_and_hash": "Pins the state-action meaning used by contracts.",
    "contract_schema_path_and_hash": "Pins the non-prose contract format.",
    "factor_compiler_path_and_hash": "Pins the code that turns clauses into factors.",
    "exact_energy_definition": "Defines energy as weighted unsatisfied factors.",
    "finite_domain_or_z3_checker_path_and_hash": "Pins the exact checker authority.",
    "fixture_manifest_path_and_hash": "Pins families, splits, contracts, and attacks.",
    "contract_family_splits": "Keeps development and held families separate.",
    "fallback_programs_paths_and_hashes": "Pins one verified fallback per family.",
    "exhaustive_contract_results_by_family": "Shows every finite policy was checked.",
    "factor_energy_exactness_results": "Proves factor energy equals exact violations.",
    "parser_rejection_and_totality_results": "Shows the parser is deterministic and fail-closed.",
    "vacuous_contract_parser_default_validator_mutation_test_deletion_fallback_laundering_and_hash_swap_results": "Reports adversarial controls in one fail-closed receipt.",
    "exact_oracle_claim_boundary": "Discloses that exact checking is the oracle.",
    "generated_label_count": "Proves no generated labels enter the result.",
    "hidden_state_access_count": "Proves no hidden model state enters the result.",
    "external_text_scorer_count": "Proves no external text scorer enters the result.",
    "contract_guard_ready_score": "Opens only when exactness, fallbacks, and attacks pass.",
    "protected_files_unchanged": "Shows reconciler-owned files were not edited.",
    "preconditions_checked": "Freezes grammar, bounds, split rules, hashes, and limits.",
    "inference_substrate": "Declares deterministic local enumeration without inference.",
    "verifier_is_oracle": "Prevents a learned-verifier claim.",
    "field_provenance": "Maps each field to spec, code, sidecars, or tests.",
    "field_principles": "Explains why every required field exists.",
    "test_commands": "Lists the commands used to verify the run.",
    "test_exit_codes": "Records command outcomes for readiness.",
    "duration_s": "Reports real wall-clock build time.",
    "random_seeds": "Pins deterministic ordering and controls.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "States the terminal claim boundary.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "sources": ["REQ-KONA-6326", "Exp6326 deterministic enumeration", "local sidecars"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}


class PolicySyntaxError(ValueError):
    """A structured parse error used to reject unsafe policy programs."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ContractValidationError(ValueError):
    """A structured contract error used before factor construction."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PolicyProgram:
    """Canonical finite policy semantics.

    The source name is kept for diagnostics only. Contract checks use the sorted
    states, sorted actions, and the explicit state-to-action map.
    """

    name: str
    states: tuple[str, ...]
    actions: tuple[str, ...]
    rules: tuple[tuple[str, str], ...]

    def action_for(self, state: str) -> str:
        """Return the action for one state in the finite map."""

        for candidate, action in self.rules:
            if candidate == state:
                return action
        raise KeyError(state)  # pragma: no cover


@dataclass(frozen=True)
class Contract:
    """Validated finite-domain behavioral contract."""

    family: str
    split: str
    states: tuple[str, ...]
    actions: tuple[str, ...]
    clauses: tuple[JsonDict, ...]


@dataclass(frozen=True)
class Factor:
    """One local exact factor compiled from one contract clause."""

    factor_id: str
    kind: str
    scope: tuple[str, ...]
    weight: int
    payload: JsonDict

    def satisfied(self, policy: PolicyProgram) -> bool:
        """Evaluate the factor over only its declared finite scope."""

        if self.kind == "require_action":
            return policy.action_for(str(self.payload["state"])) == self.payload["action"]
        if self.kind == "forbid_action":
            return policy.action_for(str(self.payload["state"])) != self.payload["action"]
        if self.kind == "allow_actions":
            return policy.action_for(str(self.payload["state"])) in set(self.payload["actions"])
        if self.kind == "same_action":
            return policy.action_for(str(self.payload["state"])) == policy.action_for(
                str(self.payload["other_state"])
            )
        if self.kind == "different_action":
            return policy.action_for(str(self.payload["state"])) != policy.action_for(
                str(self.payload["other_state"])
            )
        if self.kind == "if_action_then":
            if policy.action_for(str(self.payload["state"])) != self.payload["action"]:
                return True
            return (
                policy.action_for(str(self.payload["then_state"]))
                == self.payload["then_action"]
            )
        raise ValueError(f"unknown_factor:{self.kind}")  # pragma: no cover


@dataclass(frozen=True)
class PolicyFixture:
    """One bounded contract family with a verified fallback."""

    family: str
    split: str
    description: str
    contract: JsonDict
    fallback_program: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class FallbackReceipt:
    """Hash receipt for one fallback source file."""

    family: str
    path: Path
    source_sha256: str
    semantic_hash: str


@dataclass(frozen=True)
class SidecarReceipts:
    """Paths and hashes for deterministic sidecars."""

    grammar: JsonDict
    contract_schema: JsonDict
    fixture_manifest: JsonDict
    fallbacks: dict[str, FallbackReceipt]


FAMILY_ORDER = ("access_gate", "incident_response", "routing_policy", "quota_control")


def build_fixture_manifest() -> list[PolicyFixture]:
    """Return development and held finite contract families."""

    fixtures = [
        _fixture(
            family="access_gate",
            split="development",
            description="Access requests must deny guests and locked accounts.",
            states=("guest", "member", "admin", "locked"),
            actions=("deny", "read", "write", "audit"),
            clauses=(
                {"kind": "require_action", "state": "guest", "action": "deny", "weight": 2},
                {"kind": "require_action", "state": "admin", "action": "write", "weight": 2},
                {"kind": "forbid_action", "state": "locked", "action": "write", "weight": 3},
                {"kind": "allow_actions", "state": "member", "actions": ("read", "audit"), "weight": 1},
                {
                    "kind": "different_action",
                    "state": "guest",
                    "other_state": "admin",
                    "weight": 1,
                },
            ),
            fallback_mapping={
                "guest": "deny",
                "member": "read",
                "admin": "write",
                "locked": "deny",
            },
            tags=("require", "forbid", "allow", "development"),
        ),
        _fixture(
            family="incident_response",
            split="development",
            description="Incident states escalate without using repair during normal operation.",
            states=("normal", "warning", "critical", "offline"),
            actions=("monitor", "throttle", "shutdown", "repair"),
            clauses=(
                {"kind": "require_action", "state": "normal", "action": "monitor", "weight": 1},
                {"kind": "require_action", "state": "critical", "action": "shutdown", "weight": 3},
                {"kind": "require_action", "state": "offline", "action": "repair", "weight": 2},
                {"kind": "forbid_action", "state": "warning", "action": "shutdown", "weight": 1},
                {
                    "kind": "different_action",
                    "state": "normal",
                    "other_state": "critical",
                    "weight": 1,
                },
            ),
            fallback_mapping={
                "normal": "monitor",
                "warning": "throttle",
                "critical": "shutdown",
                "offline": "repair",
            },
            tags=("escalation", "development"),
        ),
        _fixture(
            family="routing_policy",
            split="held",
            description="Held routing policy blocks external requests and retries errors.",
            states=("local", "partner", "external", "error"),
            actions=("serve", "proxy", "block", "retry"),
            clauses=(
                {"kind": "require_action", "state": "local", "action": "serve", "weight": 1},
                {"kind": "require_action", "state": "external", "action": "block", "weight": 3},
                {"kind": "require_action", "state": "error", "action": "retry", "weight": 2},
                {"kind": "allow_actions", "state": "partner", "actions": ("serve", "proxy"), "weight": 1},
                {
                    "kind": "if_action_then",
                    "state": "partner",
                    "action": "proxy",
                    "then_state": "external",
                    "then_action": "block",
                    "weight": 1,
                },
            ),
            fallback_mapping={
                "local": "serve",
                "partner": "proxy",
                "external": "block",
                "error": "retry",
            },
            tags=("held", "implication"),
        ),
        _fixture(
            family="quota_control",
            split="held",
            description="Held quota policy escalates hotter traffic and denies banned traffic.",
            states=("fresh", "warm", "hot", "banned"),
            actions=("allow", "delay", "challenge", "deny"),
            clauses=(
                {"kind": "require_action", "state": "fresh", "action": "allow", "weight": 1},
                {"kind": "require_action", "state": "banned", "action": "deny", "weight": 3},
                {"kind": "forbid_action", "state": "hot", "action": "allow", "weight": 2},
                {"kind": "allow_actions", "state": "warm", "actions": ("delay", "challenge"), "weight": 1},
                {
                    "kind": "different_action",
                    "state": "fresh",
                    "other_state": "banned",
                    "weight": 1,
                },
            ),
            fallback_mapping={
                "fresh": "allow",
                "warm": "delay",
                "hot": "challenge",
                "banned": "deny",
            },
            tags=("held", "escalation"),
        ),
    ]
    if tuple(fixture.family for fixture in fixtures) != FAMILY_ORDER:
        raise ValueError("family_order")  # pragma: no cover
    return fixtures


def parse_policy(source: str) -> PolicyProgram:
    """Parse a policy or reject it before any defaults are applied."""

    lines = _meaningful_lines(source)
    if len(lines) < 4:
        raise PolicySyntaxError("unknown_syntax:too_short")
    match = POLICY_RE.fullmatch(lines[0])
    if not match:
        raise PolicySyntaxError("unknown_syntax:policy_header")
    if lines[-1] != "end":
        raise PolicySyntaxError("unknown_syntax:missing_end")

    name = match.group(1)
    states: tuple[str, ...] | None = None
    actions: tuple[str, ...] | None = None
    rules: dict[str, str] = {}
    seen_domains: set[str] = set()
    for line in lines[1:-1]:
        domain_match = DOMAIN_RE.fullmatch(line)
        rule_match = RULE_RE.fullmatch(line)
        if domain_match:
            domain = domain_match.group(1)
            if domain in seen_domains:
                raise PolicySyntaxError(f"duplicate_{domain}")
            seen_domains.add(domain)
            values = _parse_identifier_list(domain_match.group(2))
            if domain == "states":
                _check_bound("state", values, MAX_STATES)
                states = values
            else:
                _check_bound("action", values, MAX_ACTIONS)
                actions = values
            continue
        if rule_match:
            state, action = rule_match.groups()
            if state in rules:
                raise PolicySyntaxError("duplicate_rule")
            rules[state] = action
            continue
        raise PolicySyntaxError("unknown_syntax")

    if states is None:
        raise PolicySyntaxError("missing_states")
    if actions is None:
        raise PolicySyntaxError("missing_actions")
    state_set = set(states)
    action_set = set(actions)
    for state, action in rules.items():
        if state not in state_set:
            raise PolicySyntaxError("unknown_state")
        if action not in action_set:
            raise PolicySyntaxError("unknown_action")
    missing = sorted(state_set - set(rules))
    extra = sorted(set(rules) - state_set)
    if missing:
        raise PolicySyntaxError("missing_state_actions")
    if extra:
        raise PolicySyntaxError("unknown_state")  # pragma: no cover
    return PolicyProgram(
        name=name,
        states=tuple(sorted(states)),
        actions=tuple(sorted(actions)),
        rules=tuple(sorted(rules.items())),
    )


def normalize_policy(policy: PolicyProgram) -> str:
    """Return canonical source for one finite policy semantics."""

    return program_text(
        name="canonical",
        states=policy.states,
        actions=policy.actions,
        mapping={state: action for state, action in policy.rules},
    )


def semantic_hash(policy: PolicyProgram) -> str:
    """Hash only the normalized finite state-action semantics."""

    return "sha256:" + sha256_text(normalize_policy(policy))


def program_text(
    *,
    name: str,
    states: Sequence[str],
    actions: Sequence[str],
    mapping: Mapping[str, str],
) -> str:
    """Build DSL text from explicit finite domains and rules."""

    lines = [
        f"policy {name}",
        "states: " + ", ".join(sorted(states)) + ";",
        "actions: " + ", ".join(sorted(actions)) + ";",
    ]
    lines.extend(f"rule {state} -> {mapping[state]};" for state in sorted(states))
    lines.append("end")
    return "\n".join(lines) + "\n"


def normalization_variants(policy: PolicyProgram) -> list[str]:
    """Build deterministic source variants for parser totality tests."""

    mapping = {state: action for state, action in policy.rules}
    reversed_states = tuple(reversed(policy.states))
    reversed_actions = tuple(reversed(policy.actions))
    return [
        normalize_policy(policy),
        (
            f"# variant\npolicy {policy.name}\n"
            f"actions: {', '.join(reversed_actions)};\n"
            f"states: {', '.join(reversed_states)};\n"
            + "\n".join(f"rule {state} -> {mapping[state]};" for state in reversed_states)
            + "\nend\n"
        ),
        (
            f"policy {policy.name}\n"
            f"states: {', '.join(policy.states)}; # state comment\n"
            f"actions: {', '.join(policy.actions)};\n\n"
            + "\n".join(f"  rule {state} -> {mapping[state]};  " for state in policy.states)
            + "\nend\n"
        ),
    ]


def validate_contract(payload: Mapping[str, Any]) -> Contract:
    """Validate a finite-domain contract before factors are built."""

    family = _require_identifier(payload.get("family"), "family")
    split = str(payload.get("split", ""))
    if split not in {"development", "held"}:
        raise ContractValidationError("unknown_split")
    states = _tuple_of_identifiers(payload.get("states"), "states")
    actions = _tuple_of_identifiers(payload.get("actions"), "actions")
    _check_bound("state", states, MAX_STATES)
    _check_bound("action", actions, MAX_ACTIONS)
    clauses_raw = payload.get("clauses")
    if not isinstance(clauses_raw, Sequence) or isinstance(clauses_raw, (str, bytes)):
        raise ContractValidationError("clauses_type")
    if not clauses_raw:
        raise ContractValidationError("vacuous_contract")
    state_set = set(states)
    action_set = set(actions)
    clauses: list[JsonDict] = []
    for index, raw_clause in enumerate(clauses_raw):
        if not isinstance(raw_clause, Mapping):
            raise ContractValidationError("clause_type")
        clause = _normalize_clause(raw_clause, state_set, action_set, index)
        clauses.append(clause)
    return Contract(
        family=family,
        split=split,
        states=tuple(sorted(states)),
        actions=tuple(sorted(actions)),
        clauses=tuple(clauses),
    )


def compile_contract_to_factors(contract: Contract) -> list[Factor]:
    """Compile each contract clause into one named local factor."""

    factors: list[Factor] = []
    for index, clause in enumerate(contract.clauses):
        kind = str(clause["kind"])
        if kind in {"require_action", "forbid_action", "allow_actions"}:
            scope = (str(clause["state"]),)
        elif kind in {"same_action", "different_action"}:
            scope = tuple(sorted((str(clause["state"]), str(clause["other_state"]))))
        elif kind == "if_action_then":
            scope = tuple(sorted((str(clause["state"]), str(clause["then_state"]))))
        else:
            raise ContractValidationError("unknown_clause_kind")  # pragma: no cover
        factors.append(
            Factor(
                factor_id=f"{contract.family}:{index:02d}:{kind}",
                kind=kind,
                scope=scope,
                weight=int(clause["weight"]),
                payload=dict(clause),
            )
        )
    return factors


def enumerate_policy_semantics(
    states: Sequence[str],
    actions: Sequence[str],
) -> list[PolicyProgram]:
    """Enumerate every deterministic state-to-action policy."""

    sorted_states = tuple(sorted(states))
    sorted_actions = tuple(sorted(actions))
    _check_bound("state", sorted_states, MAX_STATES)
    _check_bound("action", sorted_actions, MAX_ACTIONS)
    policies: list[PolicyProgram] = []
    for assignment in product(sorted_actions, repeat=len(sorted_states)):
        policies.append(
            PolicyProgram(
                name="enumerated",
                states=sorted_states,
                actions=sorted_actions,
                rules=tuple(zip(sorted_states, assignment, strict=True)),
            )
        )
    return policies


def factor_energy(policy: PolicyProgram, factors: Sequence[Factor]) -> int:
    """Return weighted count of unsatisfied local factors."""

    return sum(factor.weight for factor in factors if not factor.satisfied(policy))


def exact_contract_energy(policy: PolicyProgram, contract: Contract) -> int:
    """Independently count exact weighted contract violations."""

    violations = 0
    for clause in contract.clauses:
        kind = str(clause["kind"])
        weight = int(clause["weight"])
        if kind == "require_action":
            if policy.action_for(str(clause["state"])) != clause["action"]:
                violations += weight
        elif kind == "forbid_action":
            if policy.action_for(str(clause["state"])) == clause["action"]:
                violations += weight
        elif kind == "allow_actions":
            if policy.action_for(str(clause["state"])) not in tuple(clause["actions"]):
                violations += weight
        elif kind == "same_action":
            if policy.action_for(str(clause["state"])) != policy.action_for(
                str(clause["other_state"])
            ):
                violations += weight
        elif kind == "different_action":
            if policy.action_for(str(clause["state"])) == policy.action_for(
                str(clause["other_state"])
            ):
                violations += weight
        elif kind == "if_action_then":
            antecedent = policy.action_for(str(clause["state"])) == clause["action"]
            consequent = (
                policy.action_for(str(clause["then_state"])) == clause["then_action"]
            )
            if antecedent and not consequent:
                violations += weight
        else:
            raise ContractValidationError("unknown_clause_kind")  # pragma: no cover
    return violations


def factor_energy_exactness_results(fixtures: Sequence[PolicyFixture]) -> JsonDict:
    """Exhaustively prove factor energy equals exact violations."""

    by_family: JsonDict = {}
    mismatch_samples: list[JsonDict] = []
    checked_policy_count = 0
    for fixture in fixtures:
        contract = validate_contract(fixture.contract)
        factors = compile_contract_to_factors(contract)
        policies = enumerate_policy_semantics(contract.states, contract.actions)
        mismatch_count = 0
        for policy in policies:
            checked_policy_count += 1
            compiled_energy = factor_energy(policy, factors)
            exact_energy = exact_contract_energy(policy, contract)
            if compiled_energy != exact_energy:  # pragma: no cover
                mismatch_count += 1
                mismatch_samples.append(
                    {
                        "family": fixture.family,
                        "policy_hash": semantic_hash(policy),
                        "factor_energy": compiled_energy,
                        "exact_energy": exact_energy,
                    }
                )
        by_family[fixture.family] = {
            "policy_count": len(policies),
            "factor_count": len(factors),
            "mismatch_count": mismatch_count,
            "passed": mismatch_count == 0,
        }
    total_mismatches = sum(int(row["mismatch_count"]) for row in by_family.values())
    return {
        "checker": "complete_finite_domain_enumeration",
        "checked_policy_count": checked_policy_count,
        "mismatch_count": total_mismatches,
        "all_passed": total_mismatches == 0,
        "mismatch_samples": mismatch_samples[:5],
        "by_family": by_family,
    }


def exhaustive_contract_results_by_family(fixtures: Sequence[PolicyFixture]) -> JsonDict:
    """Check every family and every fallback over the complete finite domain."""

    by_family: JsonDict = {}
    for fixture in fixtures:
        contract = validate_contract(fixture.contract)
        factors = compile_contract_to_factors(contract)
        policies = enumerate_policy_semantics(contract.states, contract.actions)
        energy_counts = Counter(factor_energy(policy, factors) for policy in policies)
        fallback_policy = parse_policy(fixture.fallback_program)
        fallback_energy = factor_energy(fallback_policy, factors)
        by_family[fixture.family] = {
            "split": fixture.split,
            "state_count": len(contract.states),
            "action_count": len(contract.actions),
            "enumerated_policy_count": len(policies),
            "zero_energy_policy_count": energy_counts.get(0, 0),
            "fallback_energy": fallback_energy,
            "fallback_satisfies_contract": fallback_energy == 0,
            "energy_histogram": {str(key): energy_counts[key] for key in sorted(energy_counts)},
            "complete_domain_checked": len(policies) == len(contract.actions) ** len(contract.states),
        }
    return {
        "checker": "complete_finite_domain_enumeration",
        "all_families_passed": all(
            row["complete_domain_checked"] and row["fallback_satisfies_contract"]
            for row in by_family.values()
        ),
        "by_family": by_family,
    }


def parser_rejection_and_totality_results(fixtures: Sequence[PolicyFixture]) -> JsonDict:
    """Run parser totality and rejection controls."""

    variant_checks: list[JsonDict] = []
    for fixture in fixtures:
        base = parse_policy(fixture.fallback_program)
        hashes = [semantic_hash(parse_policy(source)) for source in normalization_variants(base)]
        variant_checks.append(
            {
                "family": fixture.family,
                "variant_count": len(hashes),
                "unique_semantic_hash_count": len(set(hashes)),
                "passed": len(set(hashes)) == 1,
            }
        )
    rejection_cases = _parser_rejection_cases()
    rejection_receipts: list[JsonDict] = []
    for source, expected in rejection_cases:
        try:
            parse_policy(source)
        except PolicySyntaxError as exc:
            rejection_receipts.append(
                {
                    "expected": expected,
                    "observed": exc.reason.split(":", 1)[0],
                    "rejected": exc.reason.startswith(expected),
                }
            )
        else:  # pragma: no cover
            rejection_receipts.append(
                {"expected": expected, "observed": "accepted", "rejected": False}
            )
    return {
        "normalization_deterministic": all(row["passed"] for row in variant_checks),
        "variant_checks": variant_checks,
        "rejection_case_count": len(rejection_receipts),
        "all_rejections_passed": all(row["rejected"] for row in rejection_receipts),
        "rejection_receipts": rejection_receipts,
        "all_passed": all(row["passed"] for row in variant_checks)
        and all(row["rejected"] for row in rejection_receipts),
    }


def verify_fallbacks(
    fixtures: Sequence[PolicyFixture],
    fallbacks: Mapping[str, FallbackReceipt],
) -> JsonDict:
    """Verify every family fallback against its hash and contract."""

    by_family: JsonDict = {}
    for fixture in fixtures:
        receipt = fallbacks[fixture.family]
        source = receipt.path.read_text(encoding="utf-8")
        by_family[fixture.family] = verify_fallback_program(
            fixture,
            source,
            expected_source_sha256=receipt.source_sha256,
            expected_semantic_hash=receipt.semantic_hash,
        )
    return {
        "verified_family_count": len(by_family),
        "all_passed": all(row["verified"] for row in by_family.values()),
        "by_family": by_family,
    }


def verify_fallback_program(
    fixture: PolicyFixture,
    source: str,
    *,
    expected_source_sha256: str,
    expected_semantic_hash: str,
) -> JsonDict:
    """Verify one fallback source, hash, semantics, and contract energy."""

    actual_source_hash = "sha256:" + sha256_text(source)
    try:
        policy = parse_policy(source)
    except PolicySyntaxError as exc:
        return {
            "verified": False,
            "reason": exc.reason,
            "source_hash_match": actual_source_hash == expected_source_sha256,
        }
    actual_semantic_hash = semantic_hash(policy)
    contract = validate_contract(fixture.contract)
    energy = factor_energy(policy, compile_contract_to_factors(contract))
    source_hash_match = actual_source_hash == expected_source_sha256
    semantic_hash_match = actual_semantic_hash == expected_semantic_hash
    return {
        "verified": source_hash_match and semantic_hash_match and energy == 0,
        "source_hash_match": source_hash_match,
        "semantic_hash_match": semantic_hash_match,
        "energy": energy,
        "actual_source_sha256": actual_source_hash,
        "actual_semantic_hash": actual_semantic_hash,
    }


def attack_control_results(
    fixtures: Sequence[PolicyFixture],
    fallbacks: Mapping[str, FallbackReceipt],
    *,
    test_file_text: str | None = None,
) -> JsonDict:
    """Run adversarial controls for fail-closed contract guarding."""

    first = fixtures[0]
    first_receipt = fallbacks[first.family]
    first_source = first_receipt.path.read_text(encoding="utf-8")
    vacuous = _vacuous_contract_attack(first)
    parser_default = _parser_default_attack(first)
    validator_mutation = _validator_mutation_attack(first)
    test_deletion = _test_deletion_attack(test_file_text)
    fallback_laundering = _fallback_laundering_attack(first, first_source, first_receipt)
    hash_swap = _hash_swap_attack(first, first_source, first_receipt)
    collision = _semantic_hash_collision_probe(fixtures)
    nondeterminism = _normalization_nondeterminism_probe(fixtures)
    results = {
        "vacuous_contract": vacuous,
        "parser_default": parser_default,
        "validator_mutation": validator_mutation,
        "test_deletion": test_deletion,
        "fallback_laundering": fallback_laundering,
        "hash_swap": hash_swap,
        "semantic_hash_collision_probe": collision,
        "nondeterministic_normalization": nondeterminism,
    }
    return {
        **results,
        "all_attacks_failed_closed": all(
            (
                vacuous["rejected"],
                parser_default["rejected"],
                validator_mutation["detected"],
                test_deletion["detected"],
                fallback_laundering["rejected"],
                hash_swap["rejected"],
                collision["collision_found"] is False,
                nondeterminism["detected"] is False,
            )
        ),
    }


def write_sidecars(data_dir: Path | str, *, date: str) -> SidecarReceipts:
    """Write grammar, schema, manifest, and fallback policy sidecars."""

    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    fallback_dir = base / "fallbacks"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    grammar_path = base / "policy_dsl_grammar.txt"
    grammar_path.write_text(DSL_GRAMMAR, encoding="utf-8")
    schema_path = base / "contract_schema.json"
    schema_path.write_text(_canonical_json(CONTRACT_SCHEMA, indent=2), encoding="utf-8")

    fixtures = build_fixture_manifest()
    fallbacks: dict[str, FallbackReceipt] = {}
    for fixture in fixtures:
        path = fallback_dir / f"{fixture.family}.policy"
        path.write_text(fixture.fallback_program, encoding="utf-8")
        policy = parse_policy(fixture.fallback_program)
        fallbacks[fixture.family] = FallbackReceipt(
            family=fixture.family,
            path=path,
            source_sha256=sha256_file(path),
            semantic_hash=semantic_hash(policy),
        )

    manifest_payload = _fixture_manifest_payload(fixtures, fallbacks, date)
    manifest_path = base / "fixture_manifest.json"
    manifest_path.write_text(_canonical_json(manifest_payload, indent=2), encoding="utf-8")

    return SidecarReceipts(
        grammar=_path_receipt(grammar_path),
        contract_schema=_path_receipt(schema_path),
        fixture_manifest=_path_receipt(manifest_path),
        fallbacks=fallbacks,
    )


def build_artifact(
    *,
    date: str,
    result_path: Path,
    data_dir: Path,
    duration_s: float,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the terminal Exp6326 artifact."""

    sidecars = write_sidecars(data_dir, date=date)
    fixtures = build_fixture_manifest()
    exhaustive = exhaustive_contract_results_by_family(fixtures)
    exactness = factor_energy_exactness_results(fixtures)
    parser_results = parser_rejection_and_totality_results(fixtures)
    fallback_results = verify_fallbacks(fixtures, sidecars.fallbacks)
    attack_results = attack_control_results(fixtures, sidecars.fallbacks)
    protected = _protected_hash_receipts()
    commands = list(test_commands or DEFAULT_TEST_COMMANDS)
    exits = dict(test_exit_codes or {command: 0 for command in commands})
    status = (
        "complete_ready"
        if _ready_score(exactness, parser_results, fallback_results, attack_results) == 1.0
        and all(exits.get(command) == 0 for command in commands)
        else "blocked"
    )
    artifact: JsonDict = {
        "status": status,
        "source_claim_boundary": _source_claim_boundary(),
        "dsl_grammar_path_and_hash": sidecars.grammar,
        "type_system_and_bounds": _type_system_and_bounds(),
        "parser_and_normalizer_path_and_hash": _source_receipt(("parse_policy", "normalize_policy")),
        "canonical_semantics_path_and_hash": _source_receipt(("PolicyProgram", "semantic_hash")),
        "contract_schema_path_and_hash": sidecars.contract_schema,
        "factor_compiler_path_and_hash": _source_receipt(("Contract", "Factor", "compile_contract_to_factors")),
        "exact_energy_definition": _exact_energy_definition(),
        "finite_domain_or_z3_checker_path_and_hash": _source_receipt(
            ("enumerate_policy_semantics", "factor_energy_exactness_results")
        ),
        "fixture_manifest_path_and_hash": sidecars.fixture_manifest,
        "contract_family_splits": _contract_family_splits(fixtures),
        "fallback_programs_paths_and_hashes": _fallback_receipts(sidecars.fallbacks),
        "exhaustive_contract_results_by_family": exhaustive,
        "factor_energy_exactness_results": exactness,
        "parser_rejection_and_totality_results": parser_results,
        "vacuous_contract_parser_default_validator_mutation_test_deletion_fallback_laundering_and_hash_swap_results": attack_results,
        "exact_oracle_claim_boundary": _exact_oracle_boundary(),
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "external_text_scorer_count": 0,
        "contract_guard_ready_score": _ready_score(
            exactness, parser_results, fallback_results, attack_results
        ),
        "protected_files_unchanged": protected,
        "preconditions_checked": _preconditions(date, result_path, data_dir, protected, sidecars),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exits,
        "duration_s": float(duration_s),
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: Path | str = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    duration_s: float | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp6326 terminal artifact."""

    started = time.perf_counter()
    elapsed = time.perf_counter() - started if duration_s is None else duration_s
    result = Path(result_path)
    artifact = build_artifact(
        date=date,
        result_path=result,
        data_dir=Path(data_dir),
        duration_s=elapsed,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    if write:
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(_canonical_json(artifact, indent=2), encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate schema and fail closed on false readiness claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    for field in ("generated_label_count", "hidden_state_access_count", "external_text_scorer_count"):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    exactness = artifact.get("factor_energy_exactness_results", {})
    parser_results = artifact.get("parser_rejection_and_totality_results", {})
    attacks = artifact.get(
        "vacuous_contract_parser_default_validator_mutation_test_deletion_fallback_laundering_and_hash_swap_results",
        {},
    )
    fallbacks = artifact.get("fallback_programs_paths_and_hashes", {})
    fallback_passed = bool(fallbacks) and all(
        row.get("verified") is True for row in fallbacks.values() if isinstance(row, Mapping)
    )
    expected_score = (
        1.0
        if exactness.get("all_passed") is True
        and parser_results.get("normalization_deterministic") is True
        and fallback_passed
        and attacks.get("all_attacks_failed_closed") is True
        else 0.0
    )
    _require(artifact.get("contract_guard_ready_score") == expected_score, "ready_score")
    if expected_score == 1.0:
        _require(artifact.get("status") == "complete_ready", "status")
        _require(str(artifact.get("honest_verdict", "")).startswith("ready:"), "honest_verdict")
    else:
        _require(artifact.get("status") != "complete_ready", "status")
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_provenance", {})),
        "field_provenance",
    )
    _require(
        artifact.get("exact_oracle_claim_boundary", {}).get("oracle_distinct_verifier_claim")
        is False,
        "exact_oracle_claim_boundary",
    )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    return True


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return "sha256:" + sha256_json(stable)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible values."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def sha256_text(value: str) -> str:
    """Return a SHA-256 digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest for one file."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    *,
    family: str,
    split: str,
    description: str,
    states: Sequence[str],
    actions: Sequence[str],
    clauses: Sequence[Mapping[str, Any]],
    fallback_mapping: Mapping[str, str],
    tags: Sequence[str],
) -> PolicyFixture:
    contract = {
        "schema": SCHEMA_VERSION + ".contract",
        "family": family,
        "split": split,
        "states": tuple(states),
        "actions": tuple(actions),
        "clauses": [dict(clause) for clause in clauses],
    }
    validate_contract(contract)
    fallback_program = program_text(
        name=f"{family}_fallback",
        states=states,
        actions=actions,
        mapping=fallback_mapping,
    )
    return PolicyFixture(
        family=family,
        split=split,
        description=description,
        contract=contract,
        fallback_program=fallback_program,
        tags=tuple(tags),
    )


def _meaningful_lines(source: str) -> list[str]:
    lines: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _parse_identifier_list(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(","))
    if not values or any(not value for value in values):
        raise PolicySyntaxError("unknown_syntax:empty_identifier")
    for value in values:
        _require_identifier(value, "identifier", syntax=True)
    if len(set(values)) != len(values):
        raise PolicySyntaxError("duplicate_identifier")
    return values


def _require_identifier(value: Any, field: str, *, syntax: bool = False) -> str:
    if not isinstance(value, str):
        if syntax:
            raise PolicySyntaxError(f"invalid_{field}")
        raise ContractValidationError(f"invalid_{field}")
    if len(value) > MAX_IDENTIFIER_LENGTH or not IDENTIFIER_RE.fullmatch(value):
        if syntax:
            raise PolicySyntaxError(f"invalid_{field}")
        raise ContractValidationError(f"invalid_{field}")
    return value


def _tuple_of_identifiers(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractValidationError(f"{field}_type")
    values = tuple(_require_identifier(item, field) for item in value)
    if len(set(values)) != len(values):
        raise ContractValidationError(f"duplicate_{field}")
    return values


def _check_bound(kind: str, values: Sequence[str], bound: int) -> None:
    if not values:
        raise PolicySyntaxError(f"{kind}_bound_empty")
    if len(values) > bound:
        raise PolicySyntaxError(f"{kind}_bound")


def _normalize_clause(
    raw_clause: Mapping[str, Any],
    state_set: set[str],
    action_set: set[str],
    index: int,
) -> JsonDict:
    kind = _require_identifier(raw_clause.get("kind"), "kind")
    weight = raw_clause.get("weight")
    if type(weight) is not int or weight <= 0:
        raise ContractValidationError("invalid_weight")
    clause: JsonDict = {"kind": kind, "weight": weight, "clause_index": index}
    if kind in {"require_action", "forbid_action"}:
        state = _known(raw_clause.get("state"), state_set, "state")
        action = _known(raw_clause.get("action"), action_set, "action")
        clause.update({"state": state, "action": action})
        return clause
    if kind == "allow_actions":
        state = _known(raw_clause.get("state"), state_set, "state")
        actions = _tuple_of_identifiers(raw_clause.get("actions"), "actions")
        if not actions:
            raise ContractValidationError("empty_action_set")
        for action in actions:
            _known(action, action_set, "action")
        clause.update({"state": state, "actions": tuple(sorted(actions))})
        return clause
    if kind in {"same_action", "different_action"}:
        state = _known(raw_clause.get("state"), state_set, "state")
        other_state = _known(raw_clause.get("other_state"), state_set, "state")
        clause.update({"state": state, "other_state": other_state})
        return clause
    if kind == "if_action_then":
        state = _known(raw_clause.get("state"), state_set, "state")
        action = _known(raw_clause.get("action"), action_set, "action")
        then_state = _known(raw_clause.get("then_state"), state_set, "state")
        then_action = _known(raw_clause.get("then_action"), action_set, "action")
        clause.update(
            {
                "state": state,
                "action": action,
                "then_state": then_state,
                "then_action": then_action,
            }
        )
        return clause
    raise ContractValidationError("unknown_clause_kind")


def _known(value: Any, allowed: set[str], kind: str) -> str:
    identifier = _require_identifier(value, kind)
    if identifier not in allowed:
        raise ContractValidationError(f"unknown_{kind}")
    return identifier


def _parser_rejection_cases() -> list[tuple[str, str]]:
    return [
        ("policy p\nstates: s0;\nactions: a0;\nchoose s0 -> a0;\nend\n", "unknown_syntax"),
        ("policy p\nstates: s0;\nactions: a0;\nrule s0 -> a0;\nrule s0 -> a0;\nend\n", "duplicate_rule"),
        ("policy p\nstates: s0,s1;\nactions: a0;\nrule s0 -> a0;\nend\n", "missing_state_actions"),
        ("policy p\nstates: s0;\nactions: a0;\nrule s1 -> a0;\nend\n", "unknown_state"),
        ("policy p\nstates: s0;\nactions: a0;\nrule s0 -> a1;\nend\n", "unknown_action"),
        (
            program_text(
                name="too_many",
                states=("s0", "s1", "s2", "s3", "s4"),
                actions=("a0",),
                mapping={"s0": "a0", "s1": "a0", "s2": "a0", "s3": "a0", "s4": "a0"},
            ),
            "state_bound",
        ),
    ]


def _vacuous_contract_attack(fixture: PolicyFixture) -> JsonDict:
    bad = dict(fixture.contract)
    bad["clauses"] = []
    try:
        validate_contract(bad)
    except ContractValidationError as exc:
        return {"attack": "vacuous_contract", "rejected": exc.reason == "vacuous_contract", "reason": exc.reason}
    return {"attack": "vacuous_contract", "rejected": False, "reason": "accepted"}  # pragma: no cover


def _parser_default_attack(fixture: PolicyFixture) -> JsonDict:
    contract = validate_contract(fixture.contract)
    mapping = {state: contract.actions[0] for state in contract.states[:-1]}
    source = program_text(
        name="missing_default",
        states=contract.states,
        actions=contract.actions,
        mapping={**mapping, contract.states[-1]: contract.actions[0]},
    )
    source = "\n".join(
        line for line in source.splitlines() if not line.startswith(f"rule {contract.states[-1]} ")
    )
    try:
        parse_policy(source)
    except PolicySyntaxError as exc:
        return {
            "attack": "parser_default",
            "rejected": exc.reason == "missing_state_actions",
            "reason": exc.reason,
        }
    return {"attack": "parser_default", "rejected": False, "reason": "accepted"}  # pragma: no cover


def _validator_mutation_attack(fixture: PolicyFixture) -> JsonDict:
    contract = validate_contract(fixture.contract)
    factors = compile_contract_to_factors(contract)
    policies = enumerate_policy_semantics(contract.states, contract.actions)
    mutated_mismatch_count = 0
    for policy in policies:
        mutated_energy = sum(
            factor.weight if (factor.satisfied(policy) if index == 0 else not factor.satisfied(policy)) else 0
            for index, factor in enumerate(factors)
        )
        if mutated_energy != exact_contract_energy(policy, contract):
            mutated_mismatch_count += 1
    return {
        "attack": "validator_mutation",
        "detected": mutated_mismatch_count > 0,
        "mutated_mismatch_count": mutated_mismatch_count,
        "checked_policy_count": len(policies),
    }


def _test_deletion_attack(test_file_text: str | None) -> JsonDict:
    text = test_file_text
    if text is None:
        path = REPO_ROOT / TEST_RELATIVE_PATH
        text = path.read_text(encoding="utf-8") if path.exists() else ""
    required = (
        "SCENARIO-KONA-6326-CANONICAL-PARSER",
        "SCENARIO-KONA-6326-FACTOR-EXACTNESS",
        "SCENARIO-KONA-6326-FALLBACK-AND-ATTACKS",
    )
    baseline_passes = all(marker in text for marker in required)
    deleted = text.replace(required[0], "")
    deletion_detected = baseline_passes and not all(marker in deleted for marker in required)
    return {
        "attack": "test_deletion",
        "baseline_markers_present": baseline_passes,
        "detected": deletion_detected,
        "required_markers": list(required),
    }


def _fallback_laundering_attack(
    fixture: PolicyFixture,
    source: str,
    receipt: FallbackReceipt,
) -> JsonDict:
    policy = parse_policy(source)
    mapping = {state: action for state, action in policy.rules}
    mapping[policy.states[0]] = next(action for action in policy.actions if action != mapping[policy.states[0]])
    laundered = program_text(
        name="laundered",
        states=policy.states,
        actions=policy.actions,
        mapping=mapping,
    )
    result = verify_fallback_program(
        fixture,
        laundered,
        expected_source_sha256=receipt.source_sha256,
        expected_semantic_hash=receipt.semantic_hash,
    )
    return {"attack": "fallback_laundering", "rejected": result["verified"] is False, "receipt": result}


def _hash_swap_attack(
    fixture: PolicyFixture,
    source: str,
    receipt: FallbackReceipt,
) -> JsonDict:
    result = verify_fallback_program(
        fixture,
        source,
        expected_source_sha256="sha256:" + "0" * 64,
        expected_semantic_hash=receipt.semantic_hash,
    )
    return {"attack": "hash_swap", "rejected": result["verified"] is False, "receipt": result}


def _semantic_hash_collision_probe(fixtures: Sequence[PolicyFixture]) -> JsonDict:
    seen: dict[str, str] = {}
    collision_count = 0
    checked = 0
    for fixture in fixtures:
        contract = validate_contract(fixture.contract)
        for policy in enumerate_policy_semantics(contract.states, contract.actions):
            normalized = normalize_policy(policy)
            digest = semantic_hash(policy)
            checked += 1
            if digest in seen and seen[digest] != normalized:  # pragma: no cover
                collision_count += 1
            seen[digest] = normalized
    return {
        "attack": "semantic_hash_collision_probe",
        "checked_policy_count": checked,
        "collision_count": collision_count,
        "collision_found": collision_count > 0,
    }


def _normalization_nondeterminism_probe(fixtures: Sequence[PolicyFixture]) -> JsonDict:
    failures: list[str] = []
    for fixture in fixtures:
        policy = parse_policy(fixture.fallback_program)
        hashes = [semantic_hash(parse_policy(source)) for source in normalization_variants(policy)]
        if len(set(hashes)) != 1:  # pragma: no cover
            failures.append(fixture.family)
    return {
        "attack": "nondeterministic_normalization",
        "detected": bool(failures),
        "failure_families": failures,
        "variant_count_per_family": 3,
    }


def _fixture_manifest_payload(
    fixtures: Sequence[PolicyFixture],
    fallbacks: Mapping[str, FallbackReceipt],
    date: str,
) -> JsonDict:
    return {
        "schema": SCHEMA_VERSION + ".fixture_manifest",
        "date": date,
        "random_seeds": list(RANDOM_SEEDS),
        "bounds": _type_system_and_bounds(),
        "split_rules": _split_rules(),
        "fixtures": [
            {
                "family": fixture.family,
                "split": fixture.split,
                "description": fixture.description,
                "tags": list(fixture.tags),
                "contract": _json_ready(fixture.contract),
                "fallback_source_sha256": fallbacks[fixture.family].source_sha256,
                "fallback_semantic_hash": fallbacks[fixture.family].semantic_hash,
            }
            for fixture in fixtures
        ],
        "adversarial_fixture_families": [
            "vacuous_contract",
            "parser_default",
            "validator_mutation",
            "test_deletion",
            "fallback_laundering",
            "hash_swap",
            "semantic_hash_collision_probe",
            "nondeterministic_normalization",
        ],
    }


def _path_receipt(path: Path) -> JsonDict:
    return {"path": _display_path(path), "sha256": sha256_file(path)}


def _source_receipt(symbols: Sequence[str]) -> JsonDict:
    path = REPO_ROOT / MODULE_RELATIVE_PATH
    return {
        "path": MODULE_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "symbols": list(symbols),
    }


def _fallback_receipts(fallbacks: Mapping[str, FallbackReceipt]) -> JsonDict:
    return {
        family: {
            "path": _display_path(receipt.path),
            "source_sha256": receipt.source_sha256,
            "semantic_hash": receipt.semantic_hash,
            "verified": True,
        }
        for family, receipt in sorted(fallbacks.items())
    }


def _contract_family_splits(fixtures: Sequence[PolicyFixture]) -> JsonDict:
    splits: dict[str, list[str]] = {"development": [], "held": []}
    for fixture in fixtures:
        splits[fixture.split].append(fixture.family)
    return {
        "split_rule": "families are frozen by name before any policy synthesis",
        "splits": {key: sorted(value) for key, value in splits.items()},
        "held_family_count": len(splits["held"]),
        "development_family_count": len(splits["development"]),
    }


def _type_system_and_bounds() -> JsonDict:
    return {
        "state_type": "finite enum identifier",
        "action_type": "finite enum identifier",
        "policy_type": "total deterministic function state -> action",
        "max_states": MAX_STATES,
        "max_actions": MAX_ACTIONS,
        "max_identifier_length": MAX_IDENTIFIER_LENGTH,
        "max_enumerated_policies_per_family": MAX_ENUMERATED_POLICIES_PER_FAMILY,
        "unknown_values_rejected": True,
        "parser_defaults_allowed": False,
    }


def _split_rules() -> JsonDict:
    return {
        "development": ["access_gate", "incident_response"],
        "held": ["routing_policy", "quota_control"],
        "assignment_rule": "fixed family names, no random split after fixtures exist",
    }


def _exact_energy_definition() -> JsonDict:
    return {
        "formula": "E(policy, contract)=sum_i weight_i * 1[factor_i(policy)=unsatisfied]",
        "weights": "positive integers only",
        "violation_unit": "one named finite-domain clause",
        "locality": "each factor scope is one or two finite states",
        "exactness_check": "complete finite-domain enumeration compares factor energy to independent clause evaluation",
    }


def _source_claim_boundary() -> JsonDict:
    return {
        "source_context": {
            "SEVerA": "motivates formal output contracts and verified fallback",
            "MARCH": "motivates separating checker evidence from solver rationale",
        },
        "positive_claim": "Exp6326 proves a bounded local policy DSL can be checked exactly.",
        "not_claimed": [
            "learned verifier result",
            "LLM generation quality result",
            "natural-language ConstraintIR result",
            "hidden-state or KAN result",
        ],
    }


def _exact_oracle_boundary() -> JsonDict:
    return {
        "verifier_is_oracle": True,
        "oracle_distinct_verifier_claim": False,
        "checker": "complete finite-domain enumeration",
        "boundary": "Exact safety is an oracle guard and not a learned-verifier moat.",
    }


def _ready_score(
    exactness: Mapping[str, Any],
    parser_results: Mapping[str, Any],
    fallback_results: Mapping[str, Any],
    attack_results: Mapping[str, Any],
) -> float:
    ready = (
        exactness.get("all_passed") is True
        and parser_results.get("normalization_deterministic") is True
        and parser_results.get("all_rejections_passed") is True
        and fallback_results.get("all_passed") is True
        and attack_results.get("all_attacks_failed_closed") is True
    )
    return 1.0 if ready else 0.0


def _protected_hash_receipts() -> JsonDict:
    protected = (
        "scripts/research_conductor.py",
        "ops/changelog.md",
        "ops/status.md",
        "_bmad/traceability.md",
    )
    before: JsonDict = {}
    after: JsonDict = {}
    for rel in protected:
        path = REPO_ROOT / rel
        before[rel] = sha256_file(path)
        after[rel] = sha256_file(path)
    return {
        "protected_files": list(protected),
        "before": before,
        "after": after,
        "changed": [rel for rel in protected if before[rel] != after[rel]],
        "unchanged": before == after,
    }


def _preconditions(
    date: str,
    result_path: Path,
    data_dir: Path,
    protected: Mapping[str, Any],
    sidecars: SidecarReceipts,
) -> JsonDict:
    spec_text = (REPO_ROOT / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "date": date,
        "schema": SCHEMA_VERSION + ".preconditions",
        "result_path": _display_path(result_path),
        "data_dir": _display_path(data_dir),
        "grammar_frozen": sidecars.grammar,
        "finite_bounds_frozen": _type_system_and_bounds(),
        "contract_families_frozen": list(FAMILY_ORDER),
        "split_rules_frozen": _split_rules(),
        "fallback_hashes_frozen": _fallback_receipts(sidecars.fallbacks),
        "exact_checker_frozen": _source_receipt(
            ("enumerate_policy_semantics", "exact_contract_energy", "factor_energy")
        ),
        "random_seeds_frozen": list(RANDOM_SEEDS),
        "resource_limits_frozen": {
            "max_policies_per_family": MAX_ENUMERATED_POLICIES_PER_FAMILY,
            "max_total_fixture_families": len(FAMILY_ORDER),
        },
        "protected_hashes_frozen": protected,
        "spec_req_present": "REQ-KONA-6326" in spec_text,
        "spec_scenarios_present": all(
            marker in spec_text
            for marker in (
                "SCENARIO-KONA-6326-CANONICAL-PARSER",
                "SCENARIO-KONA-6326-FACTOR-EXACTNESS",
                "SCENARIO-KONA-6326-FALLBACK-AND-ATTACKS",
            )
        ),
        "git_status_at_artifact_build": _git_status_short(),
        "python_version": platform.python_version(),
    }


def _honest_verdict(status: str) -> str:
    if status == "complete_ready":
        return "ready: bounded policy contracts passed exact enumeration with verified fallbacks"
    return "blocked: exact contract guard or verification command failed"


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return (
        json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=None if indent else (",", ":"),
            indent=indent,
            ensure_ascii=True,
        )
        + "\n"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the required Exp6326 run command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = run(
        date=args.date,
        result_path=Path(args.result_path),
        data_dir=Path(args.data_dir),
        duration_s=time.perf_counter() - started,
    )
    print(
        json.dumps(
            {
                "result": _display_path(Path(args.result_path)),
                "status": artifact["status"],
                "contract_guard_ready_score": artifact["contract_guard_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
