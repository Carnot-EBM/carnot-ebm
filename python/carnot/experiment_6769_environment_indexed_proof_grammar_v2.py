"""Build and audit an environment-indexed runtime grammar for the proof DSL.

The mask reads only the current CNF environment and the emitted prefix. The
independent checker evaluates completed text after generation. This separation
tests construction-time support without turning the checker into a decoder.

Spec refs: REQ-VERIFY-6769 and SCENARIO-VERIFY-6769-*.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_6745_sota_dual_encoding_proposal_corpus as frozen
from carnot.verify import dual_certificate_encoder_a as encoder_a


JsonDict = dict[str, Any]
CommandRunner = Callable[[str, Path], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PANEL_PATH = Path("results/experiment_6768_targetable_proof_panel_expansion.json")
UPSTREAM_STREAM_PATH = Path("results/experiment_6744_hardness_controlled_certificate_stream.json")
RESULT_PATH = Path("results/experiment_6769_environment_indexed_proof_grammar_v2.json")
MODULE_PATH = Path("python/carnot/experiment_6769_environment_indexed_proof_grammar_v2.py")
TEST_PATH = Path("tests/python/test_experiment_6769_environment_indexed_proof_grammar_v2.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6769_environment_indexed_proof_grammar_v2.py")
CODE_PATHS = (MODULE_PATH, TEST_PATH, WRAPPER_PATH)

SCHEMA = "carnot.experiment_6769.environment_indexed_proof_grammar_v2.v1"
ROW_SCHEMA = f"{SCHEMA}.row"
INFERENCE_SUBSTRATE = "deterministic_automaton_no_llm"
RANDOM_SEED = 6_769_000
EOS_TOKEN = "<EOS>"
MINIMUM_PANEL_ROWS = 36
HELD_FAMILIES = ("expander_tseitin", "ladder_tseitin", "pigeonhole_anchor")
TARGET_CLASSES = (
    "undefined_variable",
    "invalid_clause",
    "non_binary_value",
    "duplicate_evidence",
    "missing_evidence",
    "premature_terminal",
)
MODES = ("static_cfg", "environment_indexed", "draft_conditioned")
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
FORBIDDEN_AUTHORITY_FIELDS = frozenset(
    {
        "answer",
        "answer_label",
        "label",
        "exact_outcome",
        "exact_valid",
        "solver_trace",
        "ground_truth_certificate",
    }
)
FORBIDDEN_MECHANISM_NAMES = FORBIDDEN_AUTHORITY_FIELDS | {
    "exact_check_constraints",
    "exact_checker",
}

GAMMA_SCHEMA: JsonDict = {
    "schema": f"{SCHEMA}.gamma",
    "purpose": "Bind token support to one CNF without answer or checker authority.",
    "fields": {
        "variable_symbols": "Exact x1..xN symbols derived from CNF n_vars.",
        "clause_symbols": "Exact c1..cM symbols derived from CNF clause count.",
        "binary_domain": "The only assignment values are ASCII 0 and 1.",
        "claim_branch": "The SAT, UNSAT, or ABSTAIN branch selected by the prefix.",
        "remaining_required_slots": "Unfilled SAT variables or the nonempty UNSAT core slot.",
        "used_symbols": "Symbols already emitted in the active claim branch.",
        "completion_state": "Whether a claim, evidence, terminal, or completion is required.",
    },
    "forbidden_fields": sorted(FORBIDDEN_AUTHORITY_FIELDS),
}

STATIC_GBNF = r"""root ::= ("SAT" " " assignment (" " assignment)* | "UNSAT" " " clause ("," clause)* | "ABSTAIN")
assignment ::= "x" [1-9] [0-9]* "=" [01]
clause ::= "c" [1-9] [0-9]*
"""

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6769_environment_indexed_proof_grammar_v2.py "
    "-q --no-cov -n 0 -o addopts="
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run "
    "--include=*/experiment_6769_environment_indexed_proof_grammar_v2.py "
    "-m pytest tests/python/test_experiment_6769_environment_indexed_proof_grammar_v2.py "
    "-q --no-cov -n 0 -o addopts="
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage report "
    "--include=*/experiment_6769_environment_indexed_proof_grammar_v2.py "
    "--fail-under=100 --show-missing"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6769_environment_indexed_proof_grammar_v2.py"
)
RUFF_COMMAND = f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH} {WRAPPER_PATH}"
FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_PATH} {TEST_PATH} {WRAPPER_PATH}"
VERIFICATION_COMMANDS = (
    ("focused_tests", FOCUSED_COMMAND),
    ("scoped_coverage_run", COVERAGE_RUN_COMMAND),
    ("scoped_coverage", COVERAGE_COMMAND),
    ("spec_coverage", SPEC_COMMAND),
    ("ruff_check", RUFF_COMMAND),
    ("format_check", FORMAT_COMMAND),
)

ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "title",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "reproducibility_receipt",
    "source_artifact_sha256",
    "source_stream_artifact_sha256",
    "gamma_schema",
    "preconditions_checked",
    "static_cfg_compile_receipt",
    "verification_receipts",
    "rows",
    "mode_summaries",
    "runtime_mask_invocation_count",
    "no_ghost_violations",
    "valid_sat_reachable",
    "valid_unsat_reachable",
    "support_contraction_receipts",
    "exact_authority_features_in_grammar",
    "dynamic_proof_grammar_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets readers reject incompatible grammar evidence.",
    "experiment": "The numeric identity binds the result to the planned task.",
    "title": "The title states the narrow runtime grammar purpose.",
    "run_date": "The fixed date identifies the planned evidence window.",
    "status": "The status separates ready, partial, and blocked terminal results.",
    "field_principles": "This map explains the purpose of every artifact field.",
    "inference_substrate": "The value states that a local mask ran without an LLM.",
    "duration_s": "Monotonic wall time records the real task cost.",
    "random_seed": "The fixed seed binds fixture and attack selection.",
    "reproducibility_checksum": "The hash binds code, inputs, seed, and retained rows.",
    "reproducibility_receipt": "Component hashes show every checksum input.",
    "source_artifact_sha256": "The file hash binds the ready Exp6768 panel.",
    "source_stream_artifact_sha256": "The file hash binds archived support witnesses.",
    "gamma_schema": "The contract lists allowed runtime state and forbidden authority.",
    "preconditions_checked": "Observed gates prevent work on missing or weak inputs.",
    "static_cfg_compile_receipt": "The receipt proves the local comparison CFG compiled.",
    "verification_receipts": "Exit codes prevent failed focused checks from becoming readiness.",
    "rows": "Each mode and case keeps its token-mask, parser, and exact evaluation evidence.",
    "mode_summaries": "Row-derived counts compare static, indexed, and draft paths.",
    "runtime_mask_invocation_count": "A nonzero count proves live prefix-policy calls.",
    "no_ghost_violations": "Zero proves dynamic outputs contain no out-of-environment symbol.",
    "valid_sat_reachable": "The boolean requires one exact SAT witness per size bin.",
    "valid_unsat_reachable": "The boolean requires one exact UNSAT witness per size bin.",
    "support_contraction_receipts": "Witness receipts detect valid-support collapse without enumeration.",
    "exact_authority_features_in_grammar": "An empty list proves the mechanism excludes exact authority.",
    "dynamic_proof_grammar_ready": "The Exp6770 gate is true only when every owned check passes.",
    "gate_check_summary": "Failed checks retain expected and observed values.",
    "verifier_is_oracle": "False states that the exact checker does not control generation.",
    "verdict_class": "The closed class makes the terminal outcome machine-readable.",
    "honest_verdict": "A terminal prefix reports readiness without a live-model claim.",
}

_SYMBOL_RE = re.compile(r"[xc][1-9][0-9]*")
_SAT_TERM_RE = re.compile(r"x([1-9][0-9]*)=([01])")
_LEXEME_RE = re.compile(r"(?:UNSAT|ABSTAIN|SAT|x[1-9][0-9]*|c[1-9][0-9]*|[01]|[ =,])")


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so evidence hashes replay exactly."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text in the repository's explicit receipt format."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    """Hash file bytes, while retaining an explicit missing state."""

    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json_object(path: Path) -> JsonDict:
    """Load one JSON object and reject scalar or array substitutes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Build one gate row with its actual observed value."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def first_failed_check(summary: Mapping[str, Any]) -> JsonDict:
    """Return the first failed check or an explicit success row."""

    for row in summary.get("checks", []):
        if row.get("passed") is not True:
            return deepcopy(dict(row))
    return {"check": "all_preconditions", "expected": True, "observed": True, "passed": True}


def backend_import_receipt() -> JsonDict:
    """Import the local GBNF backend used for the static comparison."""

    try:
        module = importlib.import_module("llama_cpp")
        backend = getattr(module, "LlamaGrammar")
    except (ImportError, AttributeError) as error:
        return {"available": False, "backend": "llama_cpp.LlamaGrammar", "error": str(error)}
    return {
        "available": True,
        "backend": f"{backend.__module__}.{backend.__name__}",
        "version": getattr(module, "__version__", "unknown"),
        "error": None,
    }


def checker_import_receipt() -> JsonDict:
    """Import the independent checker without giving it to the mask."""

    checker = getattr(frozen, "exact_check_constraints", None)
    return {
        "available": callable(checker),
        "checker": "carnot.experiment_6745.exact_check_constraints",
        "error": None if callable(checker) else "checker_not_callable",
    }


def evaluate_preconditions(
    panel: Mapping[str, Any],
    *,
    backend_receipt: Mapping[str, Any],
    checker_receipt: Mapping[str, Any],
) -> JsonDict:
    """Check the powered panel and both local runtime imports."""

    family_counts = panel.get("counts_by_family", {})
    class_counts = panel.get("counts_by_error_class", {})
    family_observed = {family: int(family_counts.get(family, 0) or 0) for family in HELD_FAMILIES}
    class_observed = {target: int(class_counts.get(target, 0) or 0) for target in TARGET_CLASSES}
    row_count = int(panel.get("targetable_row_count", 0) or 0)
    checks = [
        _check(
            "exp6768_targetable_panel_ready",
            True,
            panel.get("targetable_panel_ready"),
            panel.get("targetable_panel_ready") is True,
        ),
        _check(
            "exp6768_minimum_rows",
            f">={MINIMUM_PANEL_ROWS}",
            row_count,
            row_count >= MINIMUM_PANEL_ROWS,
        ),
        _check(
            "exp6768_held_family_coverage",
            {family: ">=2" for family in HELD_FAMILIES},
            family_observed,
            all(value >= 2 for value in family_observed.values()),
        ),
        _check(
            "exp6768_target_class_coverage",
            {target: ">=2" for target in TARGET_CLASSES},
            class_observed,
            all(value >= 2 for value in class_observed.values()),
        ),
        _check(
            "local_grammar_backend_import",
            True,
            deepcopy(dict(backend_receipt)),
            backend_receipt.get("available") is True,
        ),
        _check(
            "exact_checker_import",
            True,
            deepcopy(dict(checker_receipt)),
            checker_receipt.get("available") is True,
        ),
    ]
    return {"all_passed": all(row["passed"] for row in checks), "checks": checks}


@dataclass
class RuntimeGamma:
    """Hold only CNF symbols and prefix-derived completion state."""

    variable_symbols: list[str]
    clause_symbols: list[str]
    binary_domain: list[str]
    claim_branch: str | None
    remaining_required_slots: list[str]
    used_symbols: list[str]
    completion_state: str

    @classmethod
    def from_cnf(cls, cnf: Mapping[str, Any]) -> RuntimeGamma:
        """Create an answer-blind environment directly from one CNF shape."""

        return cls(
            variable_symbols=[f"x{index}" for index in range(1, int(cnf["n_vars"]) + 1)],
            clause_symbols=[f"c{index}" for index in range(1, len(cnf["clauses"]) + 1)],
            binary_domain=["0", "1"],
            claim_branch=None,
            remaining_required_slots=[],
            used_symbols=[],
            completion_state="claim_required",
        )

    def snapshot(self) -> JsonDict:
        """Return the complete auditable state without hidden fields."""

        return {
            "variable_symbols": list(self.variable_symbols),
            "clause_symbols": list(self.clause_symbols),
            "binary_domain": list(self.binary_domain),
            "claim_branch": self.claim_branch,
            "remaining_required_slots": list(self.remaining_required_slots),
            "used_symbols": list(self.used_symbols),
            "completion_state": self.completion_state,
        }


def lex_candidate(text: str) -> list[str]:
    """Split proof text into lossless lexical tokens, including invalid text."""

    tokens: list[str] = []
    offset = 0
    while offset < len(text):
        match = _LEXEME_RE.match(text, offset)
        if match is not None:
            tokens.append(match.group(0))
            offset = match.end()
            continue
        end = offset + 1
        while end < len(text) and _LEXEME_RE.match(text, end) is None:
            end += 1
        tokens.append(text[offset:end])
        offset = end
    return tokens


class TokenMaskSession:
    """Invoke a prefix policy before each candidate token and terminal."""

    def __init__(self) -> None:
        self.mask_invocations: list[JsonDict] = []
        self.accepted_tokens: list[str] = []

    def allowed_tokens(self) -> tuple[str, set[str]]:
        """Return the active policy ID and its exact next-token support."""

        raise NotImplementedError

    def advance(self, token: str) -> None:
        """Update prefix state after one allowed token."""

        raise NotImplementedError

    def state_snapshot(self) -> JsonDict | None:
        """Return state exposed in the replay receipt."""

        return None

    def replay(self, candidate: str) -> JsonDict:
        """Try to emit a candidate while applying the live mask each step."""

        candidate_tokens = lex_candidate(candidate)
        blocked_token: str | None = None
        for token in [*candidate_tokens, EOS_TOKEN]:
            policy_id, allowed = self.allowed_tokens()
            invocation = {
                "prefix_token_count": len(self.accepted_tokens),
                "policy_id": policy_id,
                "allowed_tokens": sorted(allowed),
                "attempted_token": token,
                "allowed": token in allowed,
            }
            self.mask_invocations.append(invocation)
            if token not in allowed:
                blocked_token = token
                break
            self.accepted_tokens.append(token)
            self.advance(token)
        terminal = bool(self.accepted_tokens and self.accepted_tokens[-1] == EOS_TOKEN)
        emitted = "".join(token for token in self.accepted_tokens if token != EOS_TOKEN)
        return {
            "candidate": candidate,
            "candidate_tokens": candidate_tokens,
            "accepted_tokens": list(self.accepted_tokens),
            "emitted_output": emitted,
            "blocked_token": blocked_token,
            "candidate_reachable": terminal and blocked_token is None,
            "terminal_reachable": terminal,
            "mask_invocation_count": len(self.mask_invocations),
            "policies_invoked": [row["policy_id"] for row in self.mask_invocations],
            "mask_invocations_sha256": sha256_json(self.mask_invocations),
            "final_gamma": self.state_snapshot(),
        }


class EnvironmentIndexedProofMask(TokenMaskSession):
    """Resolve exact CNF symbols and completion state inside token generation."""

    def __init__(self, gamma: RuntimeGamma) -> None:
        super().__init__()
        self.gamma = gamma
        self.stage = "claim"
        self.pending_symbol: str | None = None

    def allowed_tokens(self) -> tuple[str, set[str]]:
        """Resolve the next environment-indexed policy from prefix state."""

        if self.stage == "claim":
            return "claim_select", {"SAT", "UNSAT", "ABSTAIN"}
        if self.stage == "sat_separator":
            return "sat_open_evidence", {" "}
        if self.stage == "sat_symbol":
            return "sat_select_remaining_variable", set(self.gamma.remaining_required_slots)
        if self.stage == "sat_equals":
            return "sat_bind_operator", {"="}
        if self.stage == "sat_value":
            return "sat_binary_domain", set(self.gamma.binary_domain)
        if self.stage == "sat_separator_or_terminal":
            if self.gamma.remaining_required_slots:
                return "sat_require_remaining_slot", {" "}
            return "sat_complete_terminal", {EOS_TOKEN}
        if self.stage == "unsat_separator":
            return "unsat_open_evidence", {" "}
        if self.stage == "unsat_clause":
            remaining = set(self.gamma.clause_symbols) - set(self.gamma.used_symbols)
            return "unsat_select_unused_clause", remaining
        if self.stage == "unsat_separator_or_terminal":
            remaining = set(self.gamma.clause_symbols) - set(self.gamma.used_symbols)
            return "unsat_extend_or_complete", {EOS_TOKEN, *({","} if remaining else set())}
        if self.stage == "abstain_terminal":
            return "abstain_complete_terminal", {EOS_TOKEN}
        return "complete", set()

    def advance(self, token: str) -> None:
        """Update branch, remaining slots, used symbols, and completion state."""

        if self.stage == "claim":
            self.gamma.claim_branch = token
            if token == "SAT":
                self.gamma.remaining_required_slots = list(self.gamma.variable_symbols)
                self.gamma.completion_state = "sat_evidence_required"
                self.stage = "sat_separator"
            elif token == "UNSAT":
                self.gamma.remaining_required_slots = ["nonempty_clause_core"]
                self.gamma.completion_state = "unsat_evidence_required"
                self.stage = "unsat_separator"
            else:
                self.gamma.completion_state = "terminal_required"
                self.stage = "abstain_terminal"
            return
        if self.stage == "sat_separator":
            self.stage = "sat_symbol"
        elif self.stage == "sat_symbol":
            self.pending_symbol = token
            self.stage = "sat_equals"
        elif self.stage == "sat_equals":
            self.stage = "sat_value"
        elif self.stage == "sat_value":
            assert self.pending_symbol is not None
            self.gamma.used_symbols.append(self.pending_symbol)
            self.gamma.remaining_required_slots.remove(self.pending_symbol)
            self.pending_symbol = None
            self.gamma.completion_state = (
                "terminal_available"
                if not self.gamma.remaining_required_slots
                else "sat_evidence_required"
            )
            self.stage = "sat_separator_or_terminal"
        elif self.stage == "sat_separator_or_terminal":
            if token == EOS_TOKEN:
                self.gamma.completion_state = "complete"
                self.stage = "complete"
            else:
                self.stage = "sat_symbol"
        elif self.stage == "unsat_separator":
            self.stage = "unsat_clause"
        elif self.stage == "unsat_clause":
            self.gamma.used_symbols.append(token)
            self.gamma.remaining_required_slots = []
            self.gamma.completion_state = "terminal_available"
            self.stage = "unsat_separator_or_terminal"
        elif self.stage == "unsat_separator_or_terminal":
            if token == EOS_TOKEN:
                self.gamma.completion_state = "complete"
                self.stage = "complete"
            else:
                self.gamma.completion_state = "unsat_evidence_required"
                self.stage = "unsat_clause"
        elif self.stage == "abstain_terminal":
            self.gamma.completion_state = "complete"
            self.stage = "complete"

    def state_snapshot(self) -> JsonDict:
        """Expose all and only the runtime Gamma fields."""

        return self.gamma.snapshot()


class StaticCFGProofMask(TokenMaskSession):
    """Apply the syntax-only baseline without CNF symbol or uniqueness state."""

    def __init__(self, vocabulary: set[str]) -> None:
        super().__init__()
        self.vocabulary = set(vocabulary)
        self.stage = "claim"

    def allowed_tokens(self) -> tuple[str, set[str]]:
        """Resolve the fixed CFG production state."""

        if self.stage == "claim":
            return "static_claim", {"SAT", "UNSAT", "ABSTAIN"}
        if self.stage in {"sat_separator", "unsat_separator"}:
            return "static_space", {" "}
        if self.stage == "sat_symbol":
            return "static_open_variable", {
                token for token in self.vocabulary if re.fullmatch(r"x[1-9][0-9]*", token)
            }
        if self.stage == "sat_equals":
            return "static_equals", {"="}
        if self.stage == "sat_value":
            return "static_binary", {"0", "1"}
        if self.stage == "sat_more_or_terminal":
            return "static_sat_repeat", {" ", EOS_TOKEN}
        if self.stage == "unsat_clause":
            return "static_open_clause", {
                token for token in self.vocabulary if re.fullmatch(r"c[1-9][0-9]*", token)
            }
        if self.stage == "unsat_more_or_terminal":
            return "static_unsat_repeat", {",", EOS_TOKEN}
        if self.stage == "abstain_terminal":
            return "static_abstain_terminal", {EOS_TOKEN}
        return "static_complete", set()

    def advance(self, token: str) -> None:
        """Advance the context-free production without environment updates."""

        if self.stage == "claim":
            self.stage = {
                "SAT": "sat_separator",
                "UNSAT": "unsat_separator",
                "ABSTAIN": "abstain_terminal",
            }[token]
        elif self.stage == "sat_separator":
            self.stage = "sat_symbol"
        elif self.stage == "sat_symbol":
            self.stage = "sat_equals"
        elif self.stage == "sat_equals":
            self.stage = "sat_value"
        elif self.stage == "sat_value":
            self.stage = "sat_more_or_terminal"
        elif self.stage == "sat_more_or_terminal":
            self.stage = "static_complete" if token == EOS_TOKEN else "sat_symbol"
        elif self.stage == "unsat_separator":
            self.stage = "unsat_clause"
        elif self.stage == "unsat_clause":
            self.stage = "unsat_more_or_terminal"
        elif self.stage == "unsat_more_or_terminal":
            self.stage = "static_complete" if token == EOS_TOKEN else "unsat_clause"
        elif self.stage == "abstain_terminal":
            self.stage = "static_complete"


def compile_static_cfg() -> JsonDict:
    """Compile the syntax-only GBNF with the installed local backend."""

    started = time.monotonic()
    try:
        module = importlib.import_module("llama_cpp")
        grammar_type = getattr(module, "LlamaGrammar")
        compiled = grammar_type.from_string(STATIC_GBNF, verbose=False)
    except (ImportError, AttributeError, ValueError) as error:
        return {
            "backend": "llama_cpp.LlamaGrammar",
            "compiled": False,
            "grammar_sha256": sha256_text(STATIC_GBNF),
            "duration_s": round(time.monotonic() - started, 6),
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "backend": f"{type(compiled).__module__}.{type(compiled).__name__}",
        "compiled": True,
        "grammar_sha256": sha256_text(STATIC_GBNF),
        "duration_s": round(time.monotonic() - started, 6),
        "error": None,
    }


def exact_authority_features_in_mechanism() -> list[str]:
    """Scan the mask and renderer AST for forbidden authority identifiers."""

    found: set[str] = set()
    for value in (RuntimeGamma, EnvironmentIndexedProofMask, render_draft_conditioned):
        tree = ast.parse(inspect.getsource(value))
        for node in ast.walk(tree):
            name = (
                node.id
                if isinstance(node, ast.Name)
                else node.attr
                if isinstance(node, ast.Attribute)
                else None
            )
            if name in FORBIDDEN_MECHANISM_NAMES:
                found.add(str(name))
    return sorted(found)


def render_draft_conditioned(draft: str, gamma: RuntimeGamma) -> JsonDict:
    """Render only complete draft evidence, or emit a safe abstention."""

    parsed = frozen.parse_certificate_dsl(draft)
    rendered = "ABSTAIN"
    intent_preserved = False
    if parsed.get("parser_status") == "abstention":
        intent_preserved = True
    elif parsed.get("parser_status") == "parseable" and parsed.get("claim") == "SAT":
        terms = [str(term) for term in parsed.get("terms", [])]
        symbols = [term.split("=", 1)[0] for term in terms]
        if (
            len(symbols) == len(gamma.variable_symbols)
            and len(symbols) == len(set(symbols))
            and set(symbols) == set(gamma.variable_symbols)
        ):
            rendered = "SAT " + " ".join(terms)
            intent_preserved = True
    elif parsed.get("parser_status") == "parseable" and parsed.get("claim") == "UNSAT":
        terms = [str(term) for term in parsed.get("terms", [])]
        if terms and len(terms) == len(set(terms)) and set(terms) <= set(gamma.clause_symbols):
            rendered = "UNSAT " + ",".join(terms)
            intent_preserved = True
    mask = EnvironmentIndexedProofMask(gamma).replay(rendered)
    return {
        **mask,
        "draft": draft,
        "draft_sha256": sha256_text(draft),
        "rendered_output": mask["emitted_output"],
        "intent_preserved": intent_preserved,
        "candidate_reachable": intent_preserved and mask["candidate_reachable"],
        "exact_checker_calls": 0,
        "synthesized_slots": 0,
    }


def exact_check_output(text: str, cnf: Mapping[str, Any]) -> JsonDict:
    """Evaluate completed output only after generation has returned."""

    parsed = frozen.parse_certificate_dsl(text)
    if parsed.get("parser_status") != "parseable":
        return {
            "attempted": False,
            "authority_available": True,
            "valid": None,
            "reason": parsed.get("parser_status"),
            "checked_assignment_count": 0,
        }
    try:
        constraints = encoder_a.encode_certificate(parsed)["normalized_constraints"]
    except ValueError as error:
        return {
            "attempted": True,
            "authority_available": True,
            "valid": False,
            "reason": f"encoder_rejected:{error}",
            "checked_assignment_count": 0,
        }
    return frozen.exact_check_constraints(cnf, constraints)


def _support_candidate(row: Mapping[str, Any]) -> str:
    """Serialize one archived exact witness without changing its evidence."""

    if row["label"] == "SAT":
        assignment = row["certificate"]["assignment"]
        return "SAT " + " ".join(
            f"x{index}={int(bool(assignment[str(index)]))}"
            for index in range(1, int(row["cnf"]["n_vars"]) + 1)
        )
    return "UNSAT " + ",".join(f"c{index}" for index in range(1, len(row["cnf"]["clauses"]) + 1))


def build_support_fixtures(stream: Mapping[str, Any]) -> list[JsonDict]:
    """Select one archived SAT and UNSAT witness per supported size bin."""

    rows = [row for row in stream.get("rows", []) if isinstance(row, Mapping)]
    if stream.get("hardness_stream_ready") is not True or len(rows) != 72:
        raise ValueError("missing_support_witness:stream_contract")
    bins = sorted({str(row.get("size_bin")) for row in rows})
    fixtures: list[JsonDict] = []
    for size_bin in bins:
        for claim in ("SAT", "UNSAT"):
            matches = sorted(
                (
                    row
                    for row in rows
                    if row.get("size_bin") == size_bin and row.get("label") == claim
                ),
                key=lambda row: str(row.get("row_id")),
            )
            if not matches:
                raise ValueError(f"missing_support_witness:{size_bin}:{claim}")
            row = matches[0]
            fixtures.append(
                {
                    "case_id": f"support_{size_bin}_{claim.lower()}",
                    "case_kind": "fixture",
                    "target_class": f"exact_valid_{claim.lower()}",
                    "claim": claim,
                    "size_bin": size_bin,
                    "cnf": deepcopy(row["cnf"]),
                    "candidate": _support_candidate(row),
                    "source_row_id": row["row_id"],
                    "expected_candidate_reachable": True,
                }
            )
    return fixtures


def build_cases(panel: Mapping[str, Any], stream: Mapping[str, Any]) -> list[JsonDict]:
    """Build exact witnesses and attacks from frozen, retained evidence."""

    fixtures = build_support_fixtures(stream)
    sat_fixture = next(row for row in fixtures if row["claim"] == "SAT")
    cases = [*fixtures]
    cases.append(
        {
            **deepcopy(sat_fixture),
            "case_id": "fixture_abstain",
            "target_class": "valid_abstain",
            "claim": "ABSTAIN",
            "candidate": "ABSTAIN",
            "source_row_id": sat_fixture["source_row_id"],
        }
    )
    panel_rows = [row for row in panel.get("rows", []) if isinstance(row, Mapping)]
    for target_class in TARGET_CLASSES:
        source = next(row for row in panel_rows if row.get("error_class") == target_class)
        candidate = str(source["after_certificate"])
        cases.append(
            {
                "case_id": f"attack_{target_class}",
                "case_kind": "attack",
                "target_class": target_class,
                "claim": candidate.split(" ", 1)[0],
                "size_bin": source.get("size"),
                "cnf": deepcopy(source["cnf"]),
                "candidate": candidate,
                "source_row_id": source["row_id"],
                "expected_candidate_reachable": False,
            }
        )
    valid_sat = str(sat_fixture["candidate"])
    cases.extend(
        [
            {
                **deepcopy(sat_fixture),
                "case_id": "attack_extra_text",
                "case_kind": "attack",
                "target_class": "extra_text",
                "candidate": valid_sat + " EXTRA",
                "expected_candidate_reachable": False,
            },
            {
                **deepcopy(sat_fixture),
                "case_id": "attack_confusable",
                "case_kind": "attack",
                "target_class": "confusable",
                "candidate": valid_sat.replace("SAT", "ＳＡＴ", 1),
                "expected_candidate_reachable": False,
            },
            {
                **deepcopy(sat_fixture),
                "case_id": "attack_adversarial_prefix",
                "case_kind": "attack",
                "target_class": "adversarial_prefix",
                "candidate": "SAT UNSAT c1",
                "expected_candidate_reachable": False,
            },
            {
                **deepcopy(sat_fixture),
                "case_id": "attack_support_collapse",
                "case_kind": "attack",
                "target_class": "support_collapse",
                "candidate": "SAT " + " ".join(reversed(valid_sat.split()[1:])),
                "expected_candidate_reachable": True,
            },
        ]
    )
    return cases


def _ghost_symbols(text: str, cnf: Mapping[str, Any]) -> list[str]:
    """Report emitted x/c symbols that are absent from the current CNF."""

    gamma = RuntimeGamma.from_cnf(cnf)
    allowed = set(gamma.variable_symbols) | set(gamma.clause_symbols)
    return sorted(set(_SYMBOL_RE.findall(text)) - allowed)


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one comparison row without its self-referential field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def _build_row(case: Mapping[str, Any], mode: str, vocabulary: set[str]) -> JsonDict:
    """Run one mode and retain construction plus post-generation evidence."""

    cnf = case["cnf"]
    candidate = str(case["candidate"])
    renderer_calls = 0
    synthesized_slots = 0
    intent_preserved: bool | None = None
    if mode == "static_cfg":
        receipt = StaticCFGProofMask(vocabulary).replay(candidate)
    elif mode == "environment_indexed":
        receipt = EnvironmentIndexedProofMask(RuntimeGamma.from_cnf(cnf)).replay(candidate)
    else:
        receipt = render_draft_conditioned(candidate, RuntimeGamma.from_cnf(cnf))
        renderer_calls = int(receipt["exact_checker_calls"])
        synthesized_slots = int(receipt["synthesized_slots"])
        intent_preserved = bool(receipt["intent_preserved"])
    emitted = str(receipt["emitted_output"])
    parsed = frozen.parse_certificate_dsl(emitted)
    exact = (
        exact_check_output(emitted, cnf)
        if receipt["terminal_reachable"] and parsed.get("parser_status") == "parseable"
        else {
            "attempted": False,
            "authority_available": True,
            "valid": None,
            "reason": "not_applicable",
            "checked_assignment_count": 0,
        }
    )
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_id": f"{mode}|{case['case_id']}",
        "mode": mode,
        "case_id": case["case_id"],
        "case_kind": case["case_kind"],
        "target_class": case["target_class"],
        "claim": case["claim"],
        "size_bin": case["size_bin"],
        "source_row_id": case["source_row_id"],
        "cnf_sha256": sha256_json(cnf),
        "candidate": candidate,
        "candidate_sha256": sha256_text(candidate),
        "expected_candidate_reachable": case["expected_candidate_reachable"],
        "candidate_reachable": receipt["candidate_reachable"],
        "terminal_reachable": receipt["terminal_reachable"],
        "emitted_output": emitted,
        "blocked_token": receipt["blocked_token"],
        "accepted_tokens": receipt["accepted_tokens"],
        "mask_invocation_count": receipt["mask_invocation_count"],
        "mask_invocations_sha256": receipt["mask_invocations_sha256"],
        "policies_invoked": receipt["policies_invoked"],
        "final_gamma": receipt["final_gamma"],
        "ghost_symbols_emitted": _ghost_symbols(emitted, cnf),
        "parser_status": parsed["parser_status"],
        "parse_failure": parsed["parse_failure"],
        "exact_check": exact,
        "renderer_exact_checker_calls": renderer_calls,
        "renderer_synthesized_slots": synthesized_slots,
        "renderer_intent_preserved": intent_preserved,
        "row_sha256": "",
    }
    row["row_sha256"] = row_checksum(row)
    return row


def build_rows(cases: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], JsonDict]:
    """Run every mode against every fixture and attack."""

    compile_receipt = compile_static_cfg()
    vocabulary = {EOS_TOKEN}
    for case in cases:
        vocabulary.update(lex_candidate(str(case["candidate"])))
    rows = [_build_row(case, mode, vocabulary) for case in cases for mode in MODES]
    return rows, compile_receipt


def support_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Derive witness-based support checks without enumerating proof languages."""

    fixtures = [
        row
        for row in rows
        if row.get("case_kind") == "fixture" and row.get("claim") in {"SAT", "UNSAT"}
    ]
    receipts: list[JsonDict] = []
    for case_id in sorted({str(row["case_id"]) for row in fixtures}):
        case_rows = {str(row["mode"]): row for row in fixtures if row["case_id"] == case_id}
        exemplar = next(iter(case_rows.values()))
        per_mode = {
            mode: {
                "candidate_reachable": case_rows.get(mode, {}).get("candidate_reachable") is True,
                "terminal_reachable": case_rows.get(mode, {}).get("terminal_reachable") is True,
                "exact_valid": case_rows.get(mode, {}).get("exact_check", {}).get("valid") is True,
                "mask_invocation_count": int(
                    case_rows.get(mode, {}).get("mask_invocation_count", 0) or 0
                ),
            }
            for mode in MODES
        }
        receipts.append(
            {
                "case_id": case_id,
                "size_bin": exemplar["size_bin"],
                "claim": exemplar["claim"],
                "source_row_id": exemplar["source_row_id"],
                "per_mode": per_mode,
                "support_preserved": all(
                    per_mode[mode]["candidate_reachable"]
                    and per_mode[mode]["terminal_reachable"]
                    and per_mode[mode]["exact_valid"]
                    for mode in MODES
                ),
                "scope": "one exact-valid witness; no proof-language enumeration claimed",
            }
        )
    return receipts


def _mode_summaries(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count reachability and exact outcomes separately for each mode."""

    summaries: JsonDict = {}
    for mode in MODES:
        selected = [row for row in rows if row.get("mode") == mode]
        summaries[mode] = {
            "row_count": len(selected),
            "candidate_reachable": sum(row.get("candidate_reachable") is True for row in selected),
            "terminal_reachable": sum(row.get("terminal_reachable") is True for row in selected),
            "exact_valid": sum(row.get("exact_check", {}).get("valid") is True for row in selected),
            "mask_invocation_count": sum(
                int(row.get("mask_invocation_count", 0) or 0) for row in selected
            ),
        }
    return summaries


def _verification_passes(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Map each declared command to its unmodified pass state."""

    by_id = {str(row.get("check_id")): row for row in receipts}
    return {
        check_id: by_id.get(check_id, {}).get("passed") is True
        for check_id, _command in VERIFICATION_COMMANDS
    }


def derive_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    preconditions: Mapping[str, Any],
    static_cfg_compile_receipt: Mapping[str, Any],
    deterministic_replay: bool,
    verification_receipts: Sequence[Mapping[str, Any]],
    authority_features: Sequence[str],
) -> JsonDict:
    """Recompute all readiness evidence from retained receipts."""

    supports = support_receipts(rows)
    supported_bins = sorted({str(row["size_bin"]) for row in supports})
    sat_by_bin = {
        str(row["size_bin"]): row
        for row in supports
        if row["claim"] == "SAT" and row["support_preserved"] is True
    }
    unsat_by_bin = {
        str(row["size_bin"]): row
        for row in supports
        if row["claim"] == "UNSAT" and row["support_preserved"] is True
    }
    valid_sat = bool(supported_bins) and set(sat_by_bin) == set(supported_bins)
    valid_unsat = bool(supported_bins) and set(unsat_by_bin) == set(supported_bins)
    dynamic_rows = [row for row in rows if row.get("mode") != "static_cfg"]
    invocation_count = sum(int(row.get("mask_invocation_count", 0) or 0) for row in dynamic_rows)
    ghost_violations = sum(bool(row.get("ghost_symbols_emitted")) for row in dynamic_rows)
    parser_passed = bool(dynamic_rows) and all(
        row.get("terminal_reachable") is not True or row.get("parser_status") != "malformed"
        for row in dynamic_rows
    )
    authority_separated = (
        not authority_features
        and all(int(row.get("renderer_exact_checker_calls", 0) or 0) == 0 for row in dynamic_rows)
        and all(int(row.get("renderer_synthesized_slots", 0) or 0) == 0 for row in dynamic_rows)
    )
    verification = _verification_passes(verification_receipts)
    checks = {
        "preconditions": preconditions.get("all_passed") is True,
        "static_cfg_compile": static_cfg_compile_receipt.get("compiled") is True,
        "runtime_invocation": invocation_count > 0,
        "no_ghost": ghost_violations == 0,
        "valid_sat_reachable": valid_sat,
        "valid_unsat_reachable": valid_unsat,
        "parser": parser_passed,
        "exact_authority_separation": authority_separated,
        "determinism": deterministic_replay,
        **verification,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "mode_summaries": _mode_summaries(rows),
        "runtime_mask_invocation_count": invocation_count,
        "no_ghost_violations": ghost_violations,
        "valid_sat_reachable": valid_sat,
        "valid_unsat_reachable": valid_unsat,
        "support_contraction_receipts": supports,
        "dynamic_proof_grammar_ready": not failed,
        "gate_check_summary": {
            "all_passed": not failed,
            "checks": [_check(name, True, passed, passed) for name, passed in checks.items()],
            "failed_checks": failed,
        },
    }


def _code_file_hashes() -> JsonDict:
    """Bind the implementation, focused tests, and executable wrapper."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in CODE_PATHS}


def build_reproducibility_receipt(
    rows: Sequence[Mapping[str, Any]],
    source_artifact_sha256: str,
    source_stream_artifact_sha256: str,
    *,
    code_files: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Hash code, inputs, fixed seed, and rows without wall-time noise."""

    files = deepcopy(dict(code_files)) if code_files is not None else _code_file_hashes()
    input_sha = sha256_json(
        {
            "source_artifact_sha256": source_artifact_sha256,
            "source_stream_artifact_sha256": source_stream_artifact_sha256,
        }
    )
    code_sha = sha256_json(files)
    rows_sha = sha256_json(rows)
    value = sha256_json(
        {
            "input_sha256": input_sha,
            "code_sha256": code_sha,
            "rows_sha256": rows_sha,
            "random_seed": RANDOM_SEED,
        }
    )
    return {
        "algorithm": "sha256",
        "input_sha256": input_sha,
        "code_sha256": code_sha,
        "rows_sha256": rows_sha,
        "code_files": files,
        "value": value,
    }


def _base_artifact(date: str, duration_s: float) -> JsonDict:
    """Build the full field shape shared by ready and blocked results."""

    return {
        "schema": SCHEMA,
        "experiment": 6769,
        "title": "Environment-indexed runtime proof grammar v2",
        "run_date": date,
        "status": "",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "reproducibility_receipt": {},
        "source_artifact_sha256": "missing",
        "source_stream_artifact_sha256": "not_checked",
        "gamma_schema": deepcopy(GAMMA_SCHEMA),
        "preconditions_checked": {},
        "static_cfg_compile_receipt": {},
        "verification_receipts": [],
        "rows": [],
        "mode_summaries": {},
        "runtime_mask_invocation_count": 0,
        "no_ghost_violations": 0,
        "valid_sat_reachable": False,
        "valid_unsat_reachable": False,
        "support_contraction_receipts": [],
        "exact_authority_features_in_grammar": exact_authority_features_in_mechanism(),
        "dynamic_proof_grammar_ready": False,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "",
    }


def _set_reproducibility(artifact: JsonDict, code_files: Mapping[str, Any] | None = None) -> None:
    """Populate the component receipt and top-level checksum."""

    receipt = build_reproducibility_receipt(
        artifact["rows"],
        str(artifact["source_artifact_sha256"]),
        str(artifact["source_stream_artifact_sha256"]),
        code_files=code_files,
    )
    artifact["reproducibility_receipt"] = receipt
    artifact["reproducibility_checksum"] = receipt["value"]


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    failed_check: str,
    expected: Any,
    observed: Any,
    source_artifact_sha256: str,
    source_stream_artifact_sha256: str,
    preconditions: Mapping[str, Any],
    code_files: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a full terminal artifact for one failed precondition."""

    artifact = _base_artifact(date, duration_s)
    artifact.update(
        {
            "status": "complete_blocked_dynamic_grammar_v2",
            "source_artifact_sha256": source_artifact_sha256,
            "source_stream_artifact_sha256": source_stream_artifact_sha256,
            "preconditions_checked": deepcopy(dict(preconditions)),
            "gate_check_summary": {
                "all_passed": False,
                "failed_check": failed_check,
                "expected": expected,
                "observed": observed,
            },
            "verdict_class": "blocked",
            "honest_verdict": (
                f"complete_blocked_dynamic_grammar_v2: {failed_check} expected {expected!r}, "
                f"observed {observed!r}"
            ),
        }
    )
    _set_reproducibility(artifact, code_files)
    return artifact


def build_artifact(
    *,
    date: str,
    duration_s: float,
    rows: Sequence[Mapping[str, Any]],
    source_artifact_sha256: str,
    source_stream_artifact_sha256: str,
    preconditions: Mapping[str, Any],
    static_cfg_compile_receipt: Mapping[str, Any],
    deterministic_replay: bool,
    verification_receipts: Sequence[Mapping[str, Any]],
    code_files: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal artifact from row-derived evidence."""

    authority = exact_authority_features_in_mechanism()
    reduction = derive_evidence(
        rows,
        preconditions=preconditions,
        static_cfg_compile_receipt=static_cfg_compile_receipt,
        deterministic_replay=deterministic_replay,
        verification_receipts=verification_receipts,
        authority_features=authority,
    )
    ready = reduction["dynamic_proof_grammar_ready"] is True
    disqualified = bool(authority) or reduction["runtime_mask_invocation_count"] == 0
    artifact = _base_artifact(date, duration_s)
    artifact.update(
        {
            "status": "complete" if ready else "complete_partial",
            "source_artifact_sha256": source_artifact_sha256,
            "source_stream_artifact_sha256": source_stream_artifact_sha256,
            "preconditions_checked": deepcopy(dict(preconditions)),
            "static_cfg_compile_receipt": deepcopy(dict(static_cfg_compile_receipt)),
            "verification_receipts": [deepcopy(dict(row)) for row in verification_receipts],
            "rows": [deepcopy(dict(row)) for row in rows],
            **reduction,
            "exact_authority_features_in_grammar": authority,
            "verdict_class": "positive" if ready else "disqualified" if disqualified else "partial",
            "honest_verdict": (
                "complete: environment-indexed SAT and UNSAT support remained reachable with "
                "zero ghost violations; no live SOTA comparison was run"
                if ready
                else "complete_partial: one or more dynamic proof grammar gates failed"
            ),
        }
    )
    _set_reproducibility(artifact, code_files)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any], *, code_files: Mapping[str, Any] | None = None
) -> list[str]:
    """Reject schema, row, aggregate, authority, verdict, or checksum drift."""

    missing = sorted(set(ARTIFACT_FIELDS) - set(artifact))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    errors: list[str] = []
    if set(artifact) != set(artifact.get("field_principles", {})):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("gamma_schema") != GAMMA_SCHEMA:
        errors.append("gamma_schema_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    expected_receipt = build_reproducibility_receipt(
        artifact.get("rows", []),
        str(artifact.get("source_artifact_sha256")),
        str(artifact.get("source_stream_artifact_sha256")),
        code_files=code_files,
    )
    if (
        artifact.get("reproducibility_receipt") != expected_receipt
        or artifact.get("reproducibility_checksum") != expected_receipt["value"]
    ):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("status") == "complete_blocked_dynamic_grammar_v2":
        if artifact.get("rows"):
            errors.append("blocked_rows_invalid")
        if artifact.get("dynamic_proof_grammar_ready") is not False:
            errors.append("blocked_readiness_mismatch")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith(
            "complete_blocked_dynamic_grammar_v2"
        ):
            errors.append("blocked_verdict_prefix_mismatch")
        return errors
    if any(row.get("row_sha256") != row_checksum(row) for row in artifact.get("rows", [])):
        errors.append("row_checksum_mismatch")
    reduction = derive_evidence(
        artifact.get("rows", []),
        preconditions=artifact.get("preconditions_checked", {}),
        static_cfg_compile_receipt=artifact.get("static_cfg_compile_receipt", {}),
        deterministic_replay=next(
            (
                row.get("observed") is True
                for row in artifact.get("gate_check_summary", {}).get("checks", [])
                if row.get("check") == "determinism"
            ),
            False,
        ),
        verification_receipts=artifact.get("verification_receipts", []),
        authority_features=artifact.get("exact_authority_features_in_grammar", []),
    )
    aggregate_fields = (
        "mode_summaries",
        "runtime_mask_invocation_count",
        "no_ghost_violations",
        "valid_sat_reachable",
        "valid_unsat_reachable",
        "support_contraction_receipts",
    )
    if any(artifact.get(field) != reduction[field] for field in aggregate_fields):
        errors.append("aggregate_recomputation_mismatch")
    if artifact.get("dynamic_proof_grammar_ready") != reduction["dynamic_proof_grammar_ready"]:
        errors.append("readiness_gate_mismatch")
    if artifact.get("gate_check_summary") != reduction["gate_check_summary"]:
        errors.append("gate_check_summary_mismatch")
    if (
        artifact.get("exact_authority_features_in_grammar")
        != exact_authority_features_in_mechanism()
    ):
        errors.append("exact_authority_separation_mismatch")
    return errors


def write_json_atomic(
    path: Path, artifact: Mapping[str, Any], *, code_files: Mapping[str, Any] | None = None
) -> None:
    """Validate, synchronize, and atomically replace one result artifact."""

    errors = validate_artifact(artifact, code_files=code_files)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".exp6769-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():  # pragma: no cover - OS failure cleanup
            temporary.unlink()


def default_command_runner(command: str, root: Path) -> JsonDict:  # pragma: no cover - CLI path
    """Run one fixed verification command and retain its process receipt."""

    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_s": round(time.monotonic() - started, 6),
    }


def verification_rows(
    commands: Sequence[tuple[str, str]], runner: CommandRunner, root: Path
) -> list[JsonDict]:
    """Run focused tests, coverage, spec coverage, and Python lint checks."""

    rows: list[JsonDict] = []
    for check_id, command in commands:
        receipt = runner(command, root)
        output = str(receipt.get("stdout", "")) + str(receipt.get("stderr", ""))
        coverage_percent: float | None = None
        if check_id == "scoped_coverage":
            match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
            coverage_percent = float(match.group(1)) if match else None
        passed = receipt.get("exit_code") == 0 and (
            check_id != "scoped_coverage" or coverage_percent == 100.0
        )
        rows.append(
            {
                "check_id": check_id,
                "command": command,
                "exit_code": receipt.get("exit_code"),
                "passed": passed,
                "coverage_percent": coverage_percent,
                "summary": output[-2000:],
                "duration_s": receipt.get("duration_s", 0.0),
            }
        )
    return rows


def _blocked_from_load(
    *,
    date: str,
    started: float,
    check: str,
    observed: str,
    panel_hash: str,
    stream_hash: str,
) -> JsonDict:
    """Build one load-failure artifact before runtime imports or evaluation."""

    return build_blocked_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        failed_check=check,
        expected="JSON object",
        observed=observed,
        source_artifact_sha256=panel_hash,
        source_stream_artifact_sha256=stream_hash,
        preconditions={"all_passed": False, "checks": []},
    )


def run(
    date: str,
    root: Path = REPO_ROOT,
    *,
    verification_runner: CommandRunner = default_command_runner,
) -> JsonDict:
    """Run preconditions, live masks, verification, and atomic publication."""

    started = time.monotonic()
    output_path = root / RESULT_PATH
    panel_path = root / UPSTREAM_PANEL_PATH
    stream_path = root / UPSTREAM_STREAM_PATH
    panel_hash = sha256_file(panel_path)
    stream_hash = sha256_file(stream_path)
    try:
        panel = load_json_object(panel_path)
    except (OSError, TypeError, ValueError) as error:
        artifact = _blocked_from_load(
            date=date,
            started=started,
            check="exp6768_json_object",
            observed=f"{type(error).__name__}: {error}",
            panel_hash=panel_hash,
            stream_hash="not_checked",
        )
        write_json_atomic(output_path, artifact)
        return artifact
    preconditions = evaluate_preconditions(
        panel,
        backend_receipt=backend_import_receipt(),
        checker_receipt=checker_import_receipt(),
    )
    if preconditions["all_passed"] is not True:
        failed = first_failed_check(preconditions)
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            failed_check=str(failed["check"]),
            expected=failed["expected"],
            observed=failed["observed"],
            source_artifact_sha256=panel_hash,
            source_stream_artifact_sha256="not_checked",
            preconditions=preconditions,
        )
        write_json_atomic(output_path, artifact)
        return artifact
    try:
        stream = load_json_object(stream_path)
    except (OSError, TypeError, ValueError) as error:
        artifact = _blocked_from_load(
            date=date,
            started=started,
            check="exp6744_json_object",
            observed=f"{type(error).__name__}: {error}",
            panel_hash=panel_hash,
            stream_hash=stream_hash,
        )
        write_json_atomic(output_path, artifact)
        return artifact
    cases = build_cases(panel, stream)
    rows, compile_receipt = build_rows(cases)
    replay_rows, _replay_compile = build_rows(cases)
    verification = verification_rows(VERIFICATION_COMMANDS, verification_runner, root)
    artifact = build_artifact(
        date=date,
        duration_s=max(time.monotonic() - started, 0.000001),
        rows=rows,
        source_artifact_sha256=panel_hash,
        source_stream_artifact_sha256=stream_hash,
        preconditions=preconditions,
        static_cfg_compile_receipt=compile_receipt,
        deterministic_replay=rows == replay_rows,
        verification_receipts=verification,
    )
    write_json_atomic(output_path, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed planning date used by the result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260830")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """Run Exp6769 and return zero for any validated terminal artifact."""

    artifact = run(str(parse_args(argv).date))
    errors = validate_artifact(artifact)
    if errors:
        raise SystemExit(";".join(errors))
    print(json.dumps({"status": artifact["status"], "artifact": str(RESULT_PATH)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
