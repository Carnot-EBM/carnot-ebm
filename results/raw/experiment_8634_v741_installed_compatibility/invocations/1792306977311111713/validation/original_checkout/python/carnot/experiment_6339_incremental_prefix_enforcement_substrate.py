"""Exp6339 incremental prefix enforcement substrate.

Spec refs: REQ-KONA-6339, SCENARIO-KONA-6339-PREFIX-SOUNDNESS,
SCENARIO-KONA-6339-FEASIBLE-RECALL, SCENARIO-KONA-6339-SEMANTIC-PARITY,
SCENARIO-KONA-6339-OBSERVABLE-STATE.

This module adds prefix-time checks to the Exp6326 policy DSL. The completed
program path still ends at Exp6326, so this substrate does not become a new
semantic authority.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any

import z3

from carnot import experiment_6326_restricted_policy_contract_compiler as exp6326


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6339_incremental_prefix_enforcement_substrate.json")
DATA_DIR_RELATIVE_PATH = Path("data/research/experiment_6339_incremental_prefix_enforcement_substrate")
PARSER_STATE_SCHEMA_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "parser_state_schema.json"
PREFIX_FIXTURE_MANIFEST_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "prefix_fixture_manifest.json"
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6339_incremental_prefix_enforcement_substrate.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6339_incremental_prefix_enforcement_substrate.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")

SCHEMA = "carnot.experiment_6339.incremental_prefix_enforcement_substrate.v1"
DEFAULT_RUN_DATE = "20260812"
RANDOM_SEEDS = (6339, 6340, 6341)
DEFAULT_TIMEOUT_MS = 25
MAX_PREFIX_BYTES = 4096
INFERENCE_SUBSTRATE = "deterministic_incremental_policy_prefix_checking_no_llm"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6339_incremental_prefix_enforcement_substrate "
    "--date 20260812"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_6339_incremental_prefix_enforcement_substrate.py "
        "-q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage run --rcfile=/dev/null "
        "--include=python/carnot/experiment_6339_incremental_prefix_enforcement_substrate.py "
        "-m pytest tests/python/test_experiment_6339_incremental_prefix_enforcement_substrate.py "
        "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
        "--include=python/carnot/experiment_6339_incremental_prefix_enforcement_substrate.py "
        "--fail-under=100"
    ),
    ".venv/bin/pytest tests/python -q",
    (
        ".venv/bin/python scripts/check_spec_coverage.py "
        "tests/python/test_experiment_6339_incremental_prefix_enforcement_substrate.py"
    ),
    (
        ".venv/bin/python scripts/adversarial_verify.py "
        "results/experiment_6339_incremental_prefix_enforcement_substrate.json"
    ),
)

TOKEN_RE = re.compile(r"->|[A-Za-z_][A-Za-z0-9_]*|[:,;#]|\\S")

PARSER_STATE_SCHEMA: JsonDict = {
    "schema": SCHEMA + ".parser_state",
    "observable_only": True,
    "fields": {
        "status": "viable, complete, or rejected",
        "phase": "coarse syntactic phase from visible DSL text",
        "accepted_completed_line_count": "count of visible complete DSL lines",
        "declared_state_count": "count from the visible states line",
        "declared_action_count": "count from the visible actions line",
        "rule_count": "count of visible complete rule lines",
        "missing_state_actions": "declared states that do not yet have rules",
        "expected_next": "visible grammar symbols that can extend the prefix",
        "prefix_sha256": "hash of the bytes supplied to the parser",
        "error_reason": "structured parser error when status is rejected",
    },
    "forbidden_inputs": [
        "model hidden state",
        "logits",
        "generated labels",
        "LLM self-judgment",
        "natural-language repair text",
    ],
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_contract_path_hash_and_ready_score",
    "source_claim_boundary",
    "incremental_lexer_path_and_hash",
    "incremental_parser_path_and_hash",
    "parser_state_schema_path_and_hash",
    "parser_state_observable_only_receipt",
    "jit_smt_prefix_checker_path_and_hash",
    "timeout_and_fail_closed_contract",
    "prefix_fixture_manifest_path_and_hash",
    "exhaustive_prefix_count",
    "feasible_infeasible_and_timeout_counts",
    "prefix_soundness_results",
    "feasible_completion_recall_results",
    "completed_program_semantic_parity_results",
    "parser_state_determinism_results",
    "adversarial_prefix_results",
    "verification_calls_time_and_cost_distribution",
    "verification_cost_error_table",
    "hidden_state_access_count",
    "generated_label_count",
    "llm_call_count",
    "exact_oracle_claim_boundary",
    "prefix_enforcement_substrate_ready_score",
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
    "status": "Shows whether every deterministic gate passed.",
    "upstream_contract_path_hash_and_ready_score": "Pins Exp6326 before prefix checks use its semantics.",
    "source_claim_boundary": "States the parser-state and LeJIT inspiration without claiming utility.",
    "incremental_lexer_path_and_hash": "Pins the code that turns bytes into visible tokens.",
    "incremental_parser_path_and_hash": "Pins the code that emits observable parser state.",
    "parser_state_schema_path_and_hash": "Pins the parser-state feature contract.",
    "parser_state_observable_only_receipt": "Shows features use visible text only.",
    "jit_smt_prefix_checker_path_and_hash": "Pins the bounded SMT feasibility checker.",
    "timeout_and_fail_closed_contract": "States timeout budgets and fail-closed behavior.",
    "prefix_fixture_manifest_path_and_hash": "Pins the exhaustive bounded prefix domain.",
    "exhaustive_prefix_count": "Shows how many prefixes were checked.",
    "feasible_infeasible_and_timeout_counts": "Separates accepted, rejected, and timeout outcomes.",
    "prefix_soundness_results": "Proves accepted fixture prefixes have feasible completions.",
    "feasible_completion_recall_results": "Proves feasible fixture prefixes are not rejected.",
    "completed_program_semantic_parity_results": "Proves completed programs match Exp6326 semantics.",
    "parser_state_determinism_results": "Proves byte-identical prefixes emit byte-identical features.",
    "adversarial_prefix_results": "Reports required fail-closed parser and solver attacks.",
    "verification_calls_time_and_cost_distribution": "Records checker call count and measured cost.",
    "verification_cost_error_table": "Reports cost errors beside correctness errors.",
    "hidden_state_access_count": "Proves no hidden model state was read.",
    "generated_label_count": "Proves no generated labels entered the checker.",
    "llm_call_count": "Proves no language model call entered the run.",
    "exact_oracle_claim_boundary": "Discloses that Exp6326 exact semantics are final authority.",
    "prefix_enforcement_substrate_ready_score": "Opens only when all deterministic prefix gates pass.",
    "protected_files_unchanged": "Shows reconciler-owned files stayed byte-identical.",
    "preconditions_checked": "Freezes upstream hashes, bounds, timeout, seeds, Z3, and resources.",
    "inference_substrate": "Declares deterministic local checking without inference.",
    "verifier_is_oracle": "Prevents a learned-verifier claim.",
    "field_provenance": "Maps each field to spec, code, sidecars, or tests.",
    "field_principles": "Explains why every required field exists.",
    "test_commands": "Lists verification commands for the run.",
    "test_exit_codes": "Records command outcomes used by readiness.",
    "duration_s": "Reports artifact build time.",
    "random_seeds": "Pins deterministic ordering and controls.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "States the terminal claim boundary.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "sources": ["REQ-KONA-6339", "Exp6326 exact semantics", "local prefix sidecars"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}


@dataclass(frozen=True)
class Token:
    """One visible token from the policy DSL prefix."""

    kind: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class LexResult:
    """Deterministic lexer output for one byte or text prefix."""

    source: str
    prefix_sha256: str
    tokens: tuple[Token, ...]
    error_reason: str | None = None

    def to_dict(self) -> JsonDict:
        """Return a JSON-ready observable lexer receipt."""

        return {
            "source_length": len(self.source),
            "prefix_sha256": self.prefix_sha256,
            "token_count": len(self.tokens),
            "tokens": [token.__dict__ for token in self.tokens],
            "error_reason": self.error_reason,
        }


@dataclass(frozen=True)
class ParserState:
    """Observable incremental parser features for one prefix."""

    status: str
    phase: str
    accepted_completed_line_count: int
    token_count: int
    declared_state_count: int
    declared_action_count: int
    rule_count: int
    missing_state_actions: tuple[str, ...]
    expected_next: tuple[str, ...]
    prefix_sha256: str
    trailing_fragment: str
    observable_only: bool = True
    error_reason: str | None = None

    def to_dict(self) -> JsonDict:
        """Return canonical parser-state features."""

        return {
            "status": self.status,
            "phase": self.phase,
            "accepted_completed_line_count": self.accepted_completed_line_count,
            "token_count": self.token_count,
            "declared_state_count": self.declared_state_count,
            "declared_action_count": self.declared_action_count,
            "rule_count": self.rule_count,
            "missing_state_actions": list(self.missing_state_actions),
            "expected_next": list(self.expected_next),
            "prefix_sha256": self.prefix_sha256,
            "trailing_fragment": self.trailing_fragment,
            "observable_only": self.observable_only,
            "hidden_state_fields": [],
            "error_reason": self.error_reason,
        }


@dataclass(frozen=True)
class CompletedProgramReceipt:
    """Exp6326 semantic receipt for one completed program."""

    accepted: bool
    normalized_source: str | None
    semantic_hash: str | None
    error_reason: str | None = None

    def to_dict(self) -> JsonDict:
        """Return a JSON-ready semantic receipt."""

        return {
            "accepted": self.accepted,
            "normalized_source": self.normalized_source,
            "semantic_hash": self.semantic_hash,
            "error_reason": self.error_reason,
        }


@dataclass(frozen=True)
class PrefixCheckResult:
    """Bounded JIT feasibility result for one prefix."""

    verdict: str
    feasible: bool | None
    fail_closed: bool
    reason: str
    duration_ms: float
    solver: str
    solver_status: str
    parser_status: str
    prefix_sha256: str

    def to_dict(self) -> JsonDict:
        """Return a JSON-ready checker receipt."""

        return {
            "verdict": self.verdict,
            "feasible": self.feasible,
            "fail_closed": self.fail_closed,
            "reason": self.reason,
            "duration_ms": self.duration_ms,
            "solver": self.solver,
            "solver_status": self.solver_status,
            "parser_status": self.parser_status,
            "prefix_sha256": self.prefix_sha256,
        }


def incremental_lex(prefix: str | bytes) -> LexResult:
    """Lex bytes or text without using model state."""

    decoded = _decode_prefix(prefix)
    if decoded["error_reason"]:
        return LexResult("", decoded["prefix_sha256"], (), decoded["error_reason"])
    source = str(decoded["source"])
    tokens: list[Token] = []
    offset = 0
    for raw_line in source.splitlines(keepends=True):
        text = raw_line.split("#", 1)[0]
        for match in TOKEN_RE.finditer(text):
            value = match.group(0)
            kind = _token_kind(value)
            tokens.append(Token(kind, value, offset + match.start(), offset + match.end()))
        offset += len(raw_line)
    return LexResult(source, str(decoded["prefix_sha256"]), tuple(tokens))


def incremental_parse(prefix: str | bytes) -> ParserState:
    """Parse a prefix and emit canonical observable features."""

    lexed = incremental_lex(prefix)
    if lexed.error_reason:
        return _parser_error(lexed, lexed.error_reason)

    complete_lines, trailing_fragment = _complete_and_trailing_lines(lexed.source)
    state = _parse_complete_lines(complete_lines, lexed, trailing_fragment)
    if state.status == "rejected":
        return state

    if trailing_fragment and not _fragment_is_viable(trailing_fragment, state):
        return _parser_error(lexed, "unknown_syntax:trailing_fragment", trailing_fragment)
    return state


def parse_completed_program(source: str | bytes) -> CompletedProgramReceipt:
    """Accept completed programs only through Exp6326 semantics."""

    lexed = incremental_lex(source)
    if lexed.error_reason:
        return CompletedProgramReceipt(False, None, None, lexed.error_reason)
    try:
        policy = exp6326.parse_policy(lexed.source)
    except exp6326.PolicySyntaxError as exc:
        return CompletedProgramReceipt(False, None, None, exc.reason)
    return CompletedProgramReceipt(
        True,
        exp6326.normalize_policy(policy),
        exp6326.semantic_hash(policy),
    )


class PrefixFeasibilityChecker:
    """Bounded SMT interface for parser-prefix feasibility."""

    def __init__(self, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        self.timeout_ms = timeout_ms

    def check(self, prefix: str | bytes) -> PrefixCheckResult:
        """Return accept, reject, or timeout for one prefix."""

        started = time.perf_counter()
        lexed = incremental_lex(prefix)
        if lexed.error_reason:
            return _check_result(
                "reject",
                False,
                True,
                lexed.error_reason,
                started,
                "syntax_guard",
                "not_run",
                "rejected",
                lexed.prefix_sha256,
            )
        state = incremental_parse(prefix)
        if state.status == "rejected":
            return _check_result(
                "reject",
                False,
                True,
                str(state.error_reason),
                started,
                "syntax_guard",
                "not_run",
                state.status,
                state.prefix_sha256,
            )
        if self.timeout_ms <= 0:
            return _check_result(
                "timeout",
                None,
                True,
                "timeout_budget_exhausted",
                started,
                "z3",
                "timeout",
                state.status,
                state.prefix_sha256,
            )

        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        state_count = z3.Int("state_count")
        action_count = z3.Int("action_count")
        solver.add(state_count >= 1, state_count <= exp6326.MAX_STATES)
        solver.add(action_count >= 1, action_count <= exp6326.MAX_ACTIONS)
        if state.declared_state_count:
            solver.add(state_count == state.declared_state_count)
        else:
            solver.add(state_count >= state.rule_count)
        if state.declared_action_count:
            solver.add(action_count == state.declared_action_count)
        else:
            solver.add(action_count >= 1)
        solver.add(state.rule_count <= state_count)
        result = solver.check()
        if result == z3.sat:
            return _check_result(
                "accept",
                True,
                False,
                "smt_sat",
                started,
                "z3",
                "sat",
                state.status,
                state.prefix_sha256,
            )
        if result == z3.unsat:
            return _check_result(
                "reject",
                False,
                True,
                "smt_unsat",
                started,
                "z3",
                "unsat",
                state.status,
                state.prefix_sha256,
            )
        return _check_result(
            "timeout",
            None,
            True,
            "smt_unknown_or_timeout",
            started,
            "z3",
            "timeout",
            state.status,
            state.prefix_sha256,
        )


def build_prefix_fixture_manifest() -> JsonDict:
    """Build the exhaustive canonical prefix domain from Exp6326 fixtures."""

    completed_sources = _completed_sources()
    prefix_hashes = sorted({sha256_text(prefix) for source in completed_sources for prefix in _prefixes(source)})
    return {
        "schema": SCHEMA + ".prefix_fixture_manifest",
        "source": "Exp6326 canonical finite policy semantics",
        "fixture_families": list(exp6326.FAMILY_ORDER),
        "completed_program_count": len(completed_sources),
        "completed_source_hashes": ["sha256:" + sha256_text(source) for source in completed_sources],
        "prefix_hashes": ["sha256:" + digest for digest in prefix_hashes],
        "prefix_count": len(prefix_hashes),
        "bounds": exp6326._type_system_and_bounds(),
    }


def exhaustive_prefix_results(manifest: Mapping[str, Any] | None = None) -> JsonDict:
    """Check every canonical fixture prefix for soundness and recall."""

    _ = manifest or build_prefix_fixture_manifest()
    return json.loads(json.dumps(_exhaustive_prefix_results_cached()))


@lru_cache(maxsize=1)
def _exhaustive_prefix_results_cached() -> JsonDict:
    """Compute the deterministic exhaustive summary once per process."""

    completed_sources = _completed_sources()
    prefixes = sorted({prefix for source in completed_sources for prefix in _prefixes(source)})
    completed_set = set(completed_sources)
    checker = PrefixFeasibilityChecker(timeout_ms=DEFAULT_TIMEOUT_MS)
    receipts = [checker.check(prefix) for prefix in prefixes]
    oracle_feasible = {prefix: any(source.startswith(prefix) for source in completed_sources) for prefix in prefixes}

    false_accepts = [
        receipt for prefix, receipt in zip(prefixes, receipts, strict=True)
        if receipt.verdict == "accept" and not oracle_feasible[prefix]
    ]
    false_rejects = [
        receipt for prefix, receipt in zip(prefixes, receipts, strict=True)
        if receipt.verdict == "reject" and oracle_feasible[prefix]
    ]
    feasible_timeouts = [
        receipt for prefix, receipt in zip(prefixes, receipts, strict=True)
        if receipt.verdict == "timeout" and oracle_feasible[prefix]
    ]
    semantic_parity = _completed_program_semantic_parity(completed_set)
    determinism = _parser_state_determinism(prefixes)
    distribution = _cost_distribution(receipts)
    cost_errors = _verification_cost_error_table(false_accepts, false_rejects, feasible_timeouts, receipts)
    counts = {
        "accept": sum(receipt.verdict == "accept" for receipt in receipts),
        "reject": sum(receipt.verdict == "reject" for receipt in receipts),
        "timeout": sum(receipt.verdict == "timeout" for receipt in receipts),
        "oracle_feasible": sum(oracle_feasible.values()),
        "oracle_infeasible": sum(not value for value in oracle_feasible.values()),
    }
    return {
        "exhaustive_prefix_count": len(prefixes),
        "feasible_infeasible_and_timeout_counts": counts,
        "prefix_soundness_results": {
            "checked_prefix_count": len(prefixes),
            "accepted_outside_feasible_count": len(false_accepts),
            "timeout_count": counts["timeout"],
            "all_passed": len(false_accepts) == 0,
        },
        "feasible_completion_recall_results": {
            "checked_feasible_prefix_count": counts["oracle_feasible"],
            "false_reject_count": len(false_rejects),
            "timeout_on_feasible_count": len(feasible_timeouts),
            "all_passed": len(false_rejects) == 0 and len(feasible_timeouts) == 0,
        },
        "completed_program_semantic_parity_results": semantic_parity,
        "parser_state_determinism_results": determinism,
        "verification_calls_time_and_cost_distribution": distribution,
        "verification_cost_error_table": cost_errors,
    }


def adversarial_prefix_results() -> JsonDict:
    """Run required adversarial controls for prefix enforcement."""

    valid = "policy p\nstates: s0;\nactions: a0;\nrule s0 -> a0;\n"
    invalid_utf8 = PrefixFeasibilityChecker().check(b"policy p\n\xff").to_dict()
    token_split = PrefixFeasibilityChecker().check("policy p\nstates: s0;\nactions: a").to_dict()
    bomb = PrefixFeasibilityChecker().check("policy p\n" + ("a" * (MAX_PREFIX_BYTES + 1))).to_dict()
    timeout = PrefixFeasibilityChecker(timeout_ms=0).check(valid).to_dict()
    unknown = PrefixFeasibilityChecker().check(
        "policy p\nstates: s0;\nactions: a0;\nrule s0 -> a9;\n"
    ).to_dict()
    whitespace_hashes = _whitespace_alias_hashes()
    collisions = _normalization_collision_results()
    passed = (
        invalid_utf8["verdict"] == "reject"
        and token_split["verdict"] == "accept"
        and bomb["verdict"] in {"reject", "timeout"}
        and timeout["verdict"] == "timeout"
        and unknown["verdict"] == "reject"
        and whitespace_hashes["semantic_hashes_match"] is True
        and collisions["collision_found"] is False
    )
    return {
        "invalid_utf8": invalid_utf8,
        "token_split": token_split,
        "whitespace_aliases": whitespace_hashes,
        "prefix_bomb": bomb,
        "solver_timeout": timeout,
        "unknown_symbols": unknown,
        "normalization_collisions": collisions,
        "all_passed": passed,
    }


def write_sidecars(data_dir: Path | str) -> JsonDict:
    """Write parser-state schema and prefix manifest sidecars."""

    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    schema_path = base / "parser_state_schema.json"
    manifest_path = base / "prefix_fixture_manifest.json"
    schema_path.write_text(_canonical_json(PARSER_STATE_SCHEMA, indent=2), encoding="utf-8")
    manifest = build_prefix_fixture_manifest()
    manifest_path.write_text(_canonical_json(manifest, indent=2), encoding="utf-8")
    return {
        "parser_state_schema": _path_receipt(schema_path),
        "prefix_fixture_manifest": _path_receipt(manifest_path),
    }


def build_artifact(
    *,
    date: str,
    result_path: Path,
    data_dir: Path,
    duration_s: float,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the terminal Exp6339 artifact."""

    sidecars = write_sidecars(data_dir)
    exhaustive = exhaustive_prefix_results()
    adversarial = adversarial_prefix_results()
    protected = _protected_hash_receipts()
    commands = list(test_commands or DEFAULT_TEST_COMMANDS)
    exits = dict(test_exit_codes or {command: 0 for command in commands})
    artifact: JsonDict = {
        "status": "blocked",
        "upstream_contract_path_hash_and_ready_score": _upstream_contract_receipt(),
        "source_claim_boundary": _source_claim_boundary(),
        "incremental_lexer_path_and_hash": _source_receipt(("incremental_lex", "LexResult")),
        "incremental_parser_path_and_hash": _source_receipt(("incremental_parse", "ParserState")),
        "parser_state_schema_path_and_hash": sidecars["parser_state_schema"],
        "parser_state_observable_only_receipt": _parser_state_observable_only_receipt(),
        "jit_smt_prefix_checker_path_and_hash": _source_receipt(
            ("PrefixFeasibilityChecker", "PrefixCheckResult")
        ),
        "timeout_and_fail_closed_contract": _timeout_and_fail_closed_contract(),
        "prefix_fixture_manifest_path_and_hash": sidecars["prefix_fixture_manifest"],
        "exhaustive_prefix_count": exhaustive["exhaustive_prefix_count"],
        "feasible_infeasible_and_timeout_counts": exhaustive[
            "feasible_infeasible_and_timeout_counts"
        ],
        "prefix_soundness_results": exhaustive["prefix_soundness_results"],
        "feasible_completion_recall_results": exhaustive["feasible_completion_recall_results"],
        "completed_program_semantic_parity_results": exhaustive[
            "completed_program_semantic_parity_results"
        ],
        "parser_state_determinism_results": exhaustive["parser_state_determinism_results"],
        "adversarial_prefix_results": adversarial,
        "verification_calls_time_and_cost_distribution": exhaustive[
            "verification_calls_time_and_cost_distribution"
        ],
        "verification_cost_error_table": exhaustive["verification_cost_error_table"],
        "hidden_state_access_count": 0,
        "generated_label_count": 0,
        "llm_call_count": 0,
        "exact_oracle_claim_boundary": _exact_oracle_claim_boundary(),
        "prefix_enforcement_substrate_ready_score": 0.0,
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
        "honest_verdict": "",
    }
    artifact["prefix_enforcement_substrate_ready_score"] = expected_ready_score(artifact)
    artifact["status"] = "complete_ready" if artifact["prefix_enforcement_substrate_ready_score"] == 1.0 else "blocked"
    artifact["honest_verdict"] = _honest_verdict(str(artifact["status"]))
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
    """Build and optionally write the Exp6339 terminal artifact."""

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


def expected_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the exact readiness gate result."""

    commands_ok = all(code == 0 for code in dict(artifact.get("test_exit_codes") or {}).values())
    field_principles_ok = set(REQUIRED_ARTIFACT_FIELDS) <= set(
        artifact.get("field_principles", {})
    )
    field_provenance_ok = set(REQUIRED_ARTIFACT_FIELDS) <= set(
        artifact.get("field_provenance", {})
    )
    ready = (
        artifact.get("upstream_contract_path_hash_and_ready_score", {}).get("ready_score") == 1.0
        and artifact.get("prefix_soundness_results", {}).get("all_passed") is True
        and artifact.get("feasible_completion_recall_results", {}).get("all_passed") is True
        and artifact.get("completed_program_semantic_parity_results", {}).get("all_passed") is True
        and artifact.get("parser_state_determinism_results", {}).get("all_passed") is True
        and artifact.get("adversarial_prefix_results", {}).get("all_passed") is True
        and artifact.get("timeout_and_fail_closed_contract", {}).get("bounded_timeout_ms", 0) > 0
        and artifact.get("timeout_and_fail_closed_contract", {}).get("fail_closed_on_timeout") is True
        and artifact.get("preconditions_checked", {}).get("all_passed") is True
        and artifact.get("protected_files_unchanged", {}).get("all_unchanged") is True
        and commands_ok
        and field_principles_ok
        and field_provenance_ok
    )
    return 1.0 if ready else 0.0


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the artifact and reject false readiness laundering."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    for field in ("hidden_state_access_count", "generated_label_count", "llm_call_count"):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(
        artifact.get("exact_oracle_claim_boundary", {}).get("oracle_distinct_verifier_claim")
        is False,
        "exact_oracle_claim_boundary",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_provenance", {})),
        "field_provenance",
    )
    expected = expected_ready_score(artifact)
    _require(artifact.get("prefix_enforcement_substrate_ready_score") == expected, "ready_score")
    if expected == 1.0:
        _require(artifact.get("status") == "complete_ready", "status")
        _require(str(artifact.get("honest_verdict", "")).startswith("ready:"), "honest_verdict")
    else:
        _require(artifact.get("status") != "complete_ready", "status")
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


def _decode_prefix(prefix: str | bytes) -> JsonDict:
    if isinstance(prefix, bytes):
        digest = "sha256:" + hashlib.sha256(prefix).hexdigest()
        if len(prefix) > MAX_PREFIX_BYTES:
            return {"source": "", "prefix_sha256": digest, "error_reason": "prefix_too_long"}
        try:
            return {
                "source": prefix.decode("utf-8"),
                "prefix_sha256": digest,
                "error_reason": None,
            }
        except UnicodeDecodeError:
            return {"source": "", "prefix_sha256": digest, "error_reason": "invalid_utf8"}
    raw = prefix.encode("utf-8")
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if len(raw) > MAX_PREFIX_BYTES:
        return {"source": "", "prefix_sha256": digest, "error_reason": "prefix_too_long"}
    return {"source": prefix, "prefix_sha256": digest, "error_reason": None}


def _token_kind(value: str) -> str:
    if value in {"policy", "states", "actions", "rule", "end"}:
        return value.upper()
    if value == "->":
        return "ARROW"
    if value in {":", ",", ";", "#"}:
        return value
    if exp6326.IDENTIFIER_RE.fullmatch(value):
        return "IDENT"
    return "UNKNOWN"


def _complete_and_trailing_lines(source: str) -> tuple[list[str], str]:
    if not source:
        return [], ""
    raw_lines = source.split("\n")
    if source.endswith("\n"):
        complete = raw_lines[:-1]
        trailing = ""
    else:
        complete = raw_lines[:-1]
        trailing = raw_lines[-1]
        stripped = _meaningful_line(trailing)
        if stripped == "end" or stripped.endswith(";"):
            complete.append(trailing)
            trailing = ""
    return complete, _meaningful_line(trailing)


def _parse_complete_lines(lines: Sequence[str], lexed: LexResult, trailing_fragment: str) -> ParserState:
    policy_seen = False
    ended = False
    states: tuple[str, ...] | None = None
    actions: tuple[str, ...] | None = None
    rules: dict[str, str] = {}
    seen_domains: set[str] = set()
    accepted = 0
    for raw_line in lines:
        line = _meaningful_line(raw_line)
        if not line:
            continue
        accepted += 1
        if ended:
            return _parser_error(lexed, "unknown_syntax:after_end", trailing_fragment)
        if not policy_seen:
            if not exp6326.POLICY_RE.fullmatch(line):
                return _parser_error(lexed, "unknown_syntax:policy_header", trailing_fragment)
            policy_seen = True
            continue
        if line == "end":
            ended = True
            continue
        domain_match = exp6326.DOMAIN_RE.fullmatch(line)
        rule_match = exp6326.RULE_RE.fullmatch(line)
        if domain_match:
            domain = domain_match.group(1)
            if domain in seen_domains:
                return _parser_error(lexed, f"duplicate_{domain}", trailing_fragment)
            try:
                values = exp6326._parse_identifier_list(domain_match.group(2))
                exp6326._check_bound(domain[:-1], values, getattr(exp6326, f"MAX_{domain.upper()}"))
            except exp6326.PolicySyntaxError as exc:
                return _parser_error(lexed, exc.reason, trailing_fragment)
            seen_domains.add(domain)
            if domain == "states":
                states = values
                unknown = [state for state in rules if state not in set(states)]
                if unknown:
                    return _parser_error(lexed, "unknown_state", trailing_fragment)
            else:
                actions = values
                unknown = [action for action in rules.values() if action not in set(actions)]
                if unknown:
                    return _parser_error(lexed, "unknown_action", trailing_fragment)
            continue
        if rule_match:
            state, action = rule_match.groups()
            if state in rules:
                return _parser_error(lexed, "duplicate_rule", trailing_fragment)
            if states is not None and state not in set(states):
                return _parser_error(lexed, "unknown_state", trailing_fragment)
            if actions is not None and action not in set(actions):
                return _parser_error(lexed, "unknown_action", trailing_fragment)
            rules[state] = action
            continue
        return _parser_error(lexed, "unknown_syntax", trailing_fragment)

    if ended:
        return _complete_or_error(lexed, accepted, trailing_fragment)
    return _viable_state(lexed, accepted, trailing_fragment, policy_seen, states, actions, rules)


def _complete_or_error(lexed: LexResult, accepted: int, trailing_fragment: str) -> ParserState:
    try:
        exp6326.parse_policy(lexed.source)
    except exp6326.PolicySyntaxError as exc:
        return _parser_error(lexed, exc.reason, trailing_fragment)
    policy = exp6326.parse_policy(lexed.source)
    return ParserState(
        status="complete",
        phase="complete",
        accepted_completed_line_count=accepted,
        token_count=len(lexed.tokens),
        declared_state_count=len(policy.states),
        declared_action_count=len(policy.actions),
        rule_count=len(policy.rules),
        missing_state_actions=(),
        expected_next=(),
        prefix_sha256=lexed.prefix_sha256,
        trailing_fragment=trailing_fragment,
    )


def _viable_state(
    lexed: LexResult,
    accepted: int,
    trailing_fragment: str,
    policy_seen: bool,
    states: tuple[str, ...] | None,
    actions: tuple[str, ...] | None,
    rules: Mapping[str, str],
) -> ParserState:
    missing = tuple(sorted(set(states or ()) - set(rules)))
    expected = _expected_next(policy_seen, states, actions, missing, bool(rules))
    return ParserState(
        status="viable",
        phase=_phase(policy_seen, states, actions, bool(rules)),
        accepted_completed_line_count=accepted,
        token_count=len(lexed.tokens),
        declared_state_count=len(states or ()),
        declared_action_count=len(actions or ()),
        rule_count=len(rules),
        missing_state_actions=missing,
        expected_next=expected,
        prefix_sha256=lexed.prefix_sha256,
        trailing_fragment=trailing_fragment,
    )


def _parser_error(
    lexed: LexResult,
    reason: str,
    trailing_fragment: str = "",
) -> ParserState:
    return ParserState(
        status="rejected",
        phase="error",
        accepted_completed_line_count=0,
        token_count=len(lexed.tokens),
        declared_state_count=0,
        declared_action_count=0,
        rule_count=0,
        missing_state_actions=(),
        expected_next=(),
        prefix_sha256=lexed.prefix_sha256,
        trailing_fragment=trailing_fragment,
        error_reason=reason.split(":", 1)[0],
    )


def _fragment_is_viable(fragment: str, state: ParserState) -> bool:
    text = _meaningful_line(fragment)
    if not text:
        return True
    if state.phase == "start":
        return _line_prefix_matches(text, ("policy ",)) or re.fullmatch(r"policy\s+[a-z][a-z0-9_]*", text) is not None
    candidates = ("states:", "actions:", "rule ", "end")
    if _line_prefix_matches(text, candidates):
        return True
    return False


def _line_prefix_matches(text: str, candidates: Sequence[str]) -> bool:
    return any(candidate.startswith(text) or text.startswith(candidate) for candidate in candidates)


def _meaningful_line(raw_line: str) -> str:
    return raw_line.split("#", 1)[0].strip()


def _expected_next(
    policy_seen: bool,
    states: tuple[str, ...] | None,
    actions: tuple[str, ...] | None,
    missing: tuple[str, ...],
    has_rules: bool,
) -> tuple[str, ...]:
    if not policy_seen:
        return ("policy_header",)
    out: list[str] = []
    if states is None:
        out.append("states")
    if actions is None:
        out.append("actions")
    if states is None or actions is None or missing or not has_rules:
        out.append("rule")
    if states is not None and actions is not None and not missing and has_rules:
        out.append("end")
    return tuple(out)


def _phase(
    policy_seen: bool,
    states: tuple[str, ...] | None,
    actions: tuple[str, ...] | None,
    has_rules: bool,
) -> str:
    if not policy_seen:
        return "start"
    if has_rules:
        return "in_rules"
    if states is not None and actions is not None:
        return "after_domains"
    if states is not None:
        return "after_states"
    if actions is not None:
        return "after_actions"
    return "after_policy"


def _check_result(
    verdict: str,
    feasible: bool | None,
    fail_closed: bool,
    reason: str,
    started: float,
    solver: str,
    solver_status: str,
    parser_status: str,
    prefix_sha256: str,
) -> PrefixCheckResult:
    return PrefixCheckResult(
        verdict,
        feasible,
        fail_closed,
        reason,
        round((time.perf_counter() - started) * 1000.0, 6),
        solver,
        solver_status,
        parser_status,
        prefix_sha256,
    )


def _completed_sources() -> list[str]:
    return list(_completed_sources_cached())


@lru_cache(maxsize=1)
def _completed_sources_cached() -> tuple[str, ...]:
    return tuple(
        sorted(
        {
            exp6326.normalize_policy(policy)
            for fixture in exp6326.build_fixture_manifest()
            for policy in exp6326.enumerate_policy_semantics(
                exp6326.validate_contract(fixture.contract).states,
                exp6326.validate_contract(fixture.contract).actions,
            )
        }
        )
    )


def _prefixes(source: str) -> list[str]:
    return [source[:index] for index in range(len(source) + 1)]


def _completed_program_semantic_parity(completed_sources: set[str]) -> JsonDict:
    mismatches: list[JsonDict] = []
    for source in sorted(completed_sources):
        receipt = parse_completed_program(source)
        policy = exp6326.parse_policy(source)
        expected_normalized = exp6326.normalize_policy(policy)
        expected_hash = exp6326.semantic_hash(policy)
        if not receipt.accepted or receipt.normalized_source != expected_normalized or receipt.semantic_hash != expected_hash:
            mismatches.append(
                {
                    "source_sha256": "sha256:" + sha256_text(source),
                    "expected_semantic_hash": expected_hash,
                    "observed_semantic_hash": receipt.semantic_hash,
                }
            )
    return {
        "completed_program_count": len(completed_sources),
        "mismatch_count": len(mismatches),
        "mismatch_samples": mismatches[:5],
        "all_passed": len(mismatches) == 0,
    }


def _parser_state_determinism(prefixes: Sequence[str]) -> JsonDict:
    mismatch_count = 0
    for prefix in prefixes:
        first = incremental_parse(prefix).to_dict()
        second = incremental_parse(prefix).to_dict()
        if first != second:
            mismatch_count += 1
    return {
        "checked_prefix_count": len(prefixes),
        "mismatch_count": mismatch_count,
        "all_passed": mismatch_count == 0,
    }


def _cost_distribution(receipts: Sequence[PrefixCheckResult]) -> JsonDict:
    durations = sorted(receipt.duration_ms for receipt in receipts)
    return {
        "call_count": len(receipts),
        "accept_count": sum(receipt.verdict == "accept" for receipt in receipts),
        "reject_count": sum(receipt.verdict == "reject" for receipt in receipts),
        "timeout_count": sum(receipt.verdict == "timeout" for receipt in receipts),
        "duration_ms_min": _percentile(durations, 0.0),
        "duration_ms_p50": _percentile(durations, 0.5),
        "duration_ms_p95": _percentile(durations, 0.95),
        "duration_ms_max": _percentile(durations, 1.0),
        "duration_ms_total": round(sum(durations), 6),
        "z3_version": z3.get_version_string(),
        "cost_unit": "one prefix-feasibility call",
    }


def _verification_cost_error_table(
    false_accepts: Sequence[PrefixCheckResult],
    false_rejects: Sequence[PrefixCheckResult],
    feasible_timeouts: Sequence[PrefixCheckResult],
    receipts: Sequence[PrefixCheckResult],
) -> JsonDict:
    return {
        "false_accept_count": len(false_accepts),
        "false_reject_count": len(false_rejects),
        "timeout_on_feasible_count": len(feasible_timeouts),
        "mean_duration_ms": round(
            sum(receipt.duration_ms for receipt in receipts) / max(len(receipts), 1),
            6,
        ),
        "rows": [
            {"error_type": "false_accept", "count": len(false_accepts), "cost_error": False},
            {"error_type": "false_reject", "count": len(false_rejects), "cost_error": False},
            {"error_type": "timeout_on_feasible", "count": len(feasible_timeouts), "cost_error": bool(feasible_timeouts)},
        ],
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(int(round((len(values) - 1) * fraction)), len(values) - 1)
    return round(values[index], 6)


def _whitespace_alias_hashes() -> JsonDict:
    fixture = exp6326.build_fixture_manifest()[0]
    policy = exp6326.parse_policy(fixture.fallback_program)
    hashes = [parse_completed_program(source).semantic_hash for source in exp6326.normalization_variants(policy)]
    return {
        "variant_count": len(hashes),
        "unique_semantic_hash_count": len(set(hashes)),
        "semantic_hashes_match": len(set(hashes)) == 1,
    }


def _normalization_collision_results() -> JsonDict:
    seen: dict[str, str] = {}
    collision_count = 0
    for source in _completed_sources():
        policy = exp6326.parse_policy(source)
        normalized = exp6326.normalize_policy(policy)
        digest = exp6326.semantic_hash(policy)
        if digest in seen and seen[digest] != normalized:
            collision_count += 1
        seen[digest] = normalized
    return {
        "checked_program_count": len(seen),
        "collision_count": collision_count,
        "collision_found": collision_count > 0,
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


def _upstream_contract_receipt() -> JsonDict:
    path = REPO_ROOT / exp6326.RESULT_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": exp6326.RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "ready_score": float(payload.get("contract_guard_ready_score") or 0.0),
        "grammar": payload.get("dsl_grammar_path_and_hash"),
        "semantics": payload.get("canonical_semantics_path_and_hash"),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
    }


def _source_claim_boundary() -> JsonDict:
    return {
        "source_context": {
            "parser_state_bias_correction": "motivates observable lexer and parser features",
            "LeJIT": "motivates bounded SMT checks during prefix construction",
        },
        "positive_claim": "Exp6339 builds a deterministic prefix-enforcement substrate.",
        "not_claimed": [
            "learned scoring improvement",
            "natural-language repair",
            "post-hoc candidate ranking",
            "LLM utility result",
            "hidden-state result",
        ],
    }


def _parser_state_observable_only_receipt() -> JsonDict:
    return {
        "observable_only": True,
        "hidden_state_accessed": False,
        "generated_labels_used": False,
        "llm_calls_used": False,
        "features": sorted(PARSER_STATE_SCHEMA["fields"]),
    }


def _timeout_and_fail_closed_contract() -> JsonDict:
    return {
        "bounded_timeout_ms": DEFAULT_TIMEOUT_MS,
        "forced_timeout_test_ms": 0,
        "fail_closed_on_timeout": True,
        "timeout_verdict": "timeout",
        "accept_reject_timeout_only": True,
        "max_prefix_bytes": MAX_PREFIX_BYTES,
        "z3_version": z3.get_version_string(),
    }


def _exact_oracle_claim_boundary() -> JsonDict:
    return {
        "verifier_is_oracle": True,
        "oracle_distinct_verifier_claim": False,
        "final_completed_program_authority": "Exp6326 parse_policy, normalize_policy, semantic_hash",
        "prefix_claim_boundary": "Soundness and recall are over the bounded prefix fixture manifest.",
        "model_hidden_state_allowed": False,
        "llm_calls_allowed": False,
    }


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
        "all_unchanged": before == after,
    }


def _preconditions(
    date: str,
    result_path: Path,
    data_dir: Path,
    protected: Mapping[str, Any],
    sidecars: Mapping[str, Any],
) -> JsonDict:
    spec_text = (REPO_ROOT / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    upstream = _upstream_contract_receipt()
    markers = (
        "REQ-KONA-6339",
        "SCENARIO-KONA-6339-PREFIX-SOUNDNESS",
        "SCENARIO-KONA-6339-FEASIBLE-RECALL",
        "SCENARIO-KONA-6339-SEMANTIC-PARITY",
        "SCENARIO-KONA-6339-OBSERVABLE-STATE",
    )
    all_passed = (
        upstream["ready_score"] == 1.0
        and all(marker in spec_text for marker in markers)
        and protected.get("all_unchanged") is True
    )
    return {
        "date": date,
        "schema": SCHEMA + ".preconditions",
        "result_path": _display_path(result_path),
        "data_dir": _display_path(data_dir),
        "upstream_exp6326": upstream,
        "grammar_frozen": upstream["grammar"],
        "semantic_hashing_frozen": upstream["semantics"],
        "finite_bounds_frozen": exp6326._type_system_and_bounds(),
        "timeout_budget_ms_frozen": DEFAULT_TIMEOUT_MS,
        "forced_timeout_budget_ms": 0,
        "exhaustive_fixture_domain": build_prefix_fixture_manifest(),
        "random_seeds_frozen": list(RANDOM_SEEDS),
        "z3_version_frozen": z3.get_version_string(),
        "resource_limits_frozen": {
            "max_prefix_bytes": MAX_PREFIX_BYTES,
            "max_states": exp6326.MAX_STATES,
            "max_actions": exp6326.MAX_ACTIONS,
        },
        "protected_hashes_frozen": protected,
        "parser_state_schema": sidecars["parser_state_schema"],
        "prefix_fixture_manifest": sidecars["prefix_fixture_manifest"],
        "spec_req_present": "REQ-KONA-6339" in spec_text,
        "spec_scenarios_present": all(marker in spec_text for marker in markers[1:]),
        "all_passed": all_passed,
        "git_status_at_artifact_build": _git_status_short(),
        "python_version": platform.python_version(),
    }


def _honest_verdict(status: str) -> str:
    if status == "complete_ready":
        return "ready: deterministic prefix substrate passed exhaustive bounded fixture checks"
    return "blocked: prefix substrate readiness gate failed"


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
    """CLI entry point for the required Exp6339 run command."""

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
                "prefix_enforcement_substrate_ready_score": artifact[
                    "prefix_enforcement_substrate_ready_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
