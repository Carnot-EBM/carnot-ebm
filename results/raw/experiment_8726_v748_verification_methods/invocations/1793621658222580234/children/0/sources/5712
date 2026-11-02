"""Exp 1339 XGrammar2-style dynamic certificate grammar dispatch dry-run.

Spec: REQ-VERIFY-1339, SCENARIO-VERIFY-1339
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Any, Callable, Mapping, Sequence

from carnot.eval import certificate_grammar_backend_bakeoff as backend_bakeoff
from carnot.eval.certificate_grammar_backend_bakeoff import (
    certificate_schema,
    validate_certificate,
)
from carnot.reporting.triggered_certificate_extraction_v5_runtime_fixed_dccd_gbnf import (
    parse_certificate_text_v5,
)


DEFAULT_RUN_DATE = "20260505"
DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1339_xgrammar2_tagdispatch_certificate_grammar_dryrun.json"
)
ARTIFACT_NAME = "experiment_1339_xgrammar2_tagdispatch_certificate_grammar_dryrun"
SCHEMA_VERSION = 1
REQUIRED_STATES = ("UNKNOWN", "SAT", "UNSAT", "REPAIR_HINT")
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "grammar_backend_candidates",
    "dynamic_grammar_compile_ms",
    "mask_generation_ms_per_token_proxy",
    "certificate_states_supported",
    "unknown_state_supported",
    "state_transition_error_count",
    "parse_rate_delta_over_static_gbnf_proxy",
    "dynamic_grammar_ready",
    "honest_verdict",
)


Timer = Callable[[], float]
TimerFactory = Callable[[], Timer]


def default_timer_factory() -> Timer:
    """Return the live monotonic timer used outside deterministic tests."""

    return time.perf_counter


@dataclass(frozen=True)
class SyntheticCertificateCase:
    """One deterministic local certificate string and the branch it should enter."""

    name: str
    expected_state: str
    text: str


@dataclass(frozen=True)
class BranchGrammar:
    """A small regex-backed stand-in for one TagDispatch grammar fragment."""

    state: str
    pattern: Pattern[str]
    parser_kind: str


@dataclass(frozen=True)
class CompiledBranchGrammars:
    """Compiled branch fragments plus the measured compile proxy."""

    branches: tuple[BranchGrammar, ...]
    compile_ms: float


@dataclass(frozen=True)
class DispatchResult:
    """Dynamic-dispatch result for one certificate string."""

    dispatched_state: str
    parseable: bool
    certificate: dict[str, Any]
    errors: list[str]
    parser_kind: str
    existing_parser_parseable: bool
    existing_parser_errors: list[str]


class ConstantStepTimer:
    """Deterministic timer used by tests for compile and mask-proxy timings."""

    def __init__(self, step: float = 0.001) -> None:
        self._current = 0.0
        self._step = float(step)

    def __call__(self) -> float:
        self._current += self._step
        return self._current


def synthetic_certificate_cases() -> list[SyntheticCertificateCase]:
    """Return the tiny UNKNOWN/SAT/UNSAT/repair fixture set for the dry-run."""

    return [
        SyntheticCertificateCase("sat_json", "SAT", _json_certificate("SAT")),
        SyntheticCertificateCase("unsat_json", "UNSAT", _json_certificate("UNSAT")),
        SyntheticCertificateCase(
            "unknown_tail",
            "UNKNOWN",
            "Final label: UNKNOWN. Solver timeout preserves the undecided branch.",
        ),
        SyntheticCertificateCase(
            "repair_hint",
            "REPAIR_HINT",
            "REPAIR_HINT: add the missing upper bound before issuing SAT or UNSAT.",
        ),
    ]


def compile_branch_grammars(*, timer: Timer = time.perf_counter) -> CompiledBranchGrammars:
    """Compile the branch fragments that emulate XGrammar2 TagDispatch locally."""

    start = timer()
    branches = (
        BranchGrammar(
            "REPAIR_HINT",
            re.compile(r"\bREPAIR(?:_HINT)?\b|\"repair_hint\"", re.IGNORECASE),
            "repair_hint_shim",
        ),
        BranchGrammar(
            "UNKNOWN",
            re.compile(r"\b(UNKNOWN|UNDETERMINED|ABSTAIN)\b", re.IGNORECASE),
            "existing_parser",
        ),
        BranchGrammar(
            "UNSAT",
            re.compile(r"\bUNSAT(?:ISFIABLE)?\b", re.IGNORECASE),
            "existing_parser",
        ),
        BranchGrammar(
            "SAT",
            re.compile(r"\bSAT(?:ISFIABLE)?\b", re.IGNORECASE),
            "existing_parser",
        ),
    )
    compile_ms = (timer() - start) * 1000.0
    return CompiledBranchGrammars(branches=branches, compile_ms=round(compile_ms, 6))


def dispatch_certificate_text(
    text: str,
    grammar: CompiledBranchGrammars,
) -> DispatchResult:
    """Route one string to a branch, then parse through the existing certificate path."""

    raw = str(text or "")
    existing = parse_certificate_text_v5(raw)
    branch = _match_branch(raw, grammar)
    if branch is None:
        return DispatchResult(
            dispatched_state="UNSUPPORTED",
            parseable=False,
            certificate={},
            errors=["no_dynamic_branch_match"] + list(existing.errors),
            parser_kind="none",
            existing_parser_parseable=existing.parseable,
            existing_parser_errors=list(existing.errors),
        )

    if branch.state == "REPAIR_HINT":
        certificate = _repair_hint_certificate(raw)
        valid, errors = validate_certificate(certificate, certificate_schema())
        return DispatchResult(
            dispatched_state=branch.state,
            parseable=valid,
            certificate=certificate if valid else {},
            errors=list(errors),
            parser_kind=branch.parser_kind,
            existing_parser_parseable=existing.parseable,
            existing_parser_errors=list(existing.errors),
        )

    return DispatchResult(
        dispatched_state=branch.state,
        parseable=existing.parseable,
        certificate=dict(existing.certificate) if existing.parseable else {},
        errors=list(existing.errors),
        parser_kind=branch.parser_kind,
        existing_parser_parseable=existing.parseable,
        existing_parser_errors=list(existing.errors),
    )


def evaluate_synthetic_cases(
    cases: Sequence[SyntheticCertificateCase],
    *,
    grammar: CompiledBranchGrammars,
    mask_timer: Timer = time.perf_counter,
) -> dict[str, Any]:
    """Run static-parser and dynamic-branch accounting over the synthetic cases."""

    rows: list[dict[str, Any]] = []
    supported: set[str] = set()
    transition_errors = 0
    dynamic_parseable = 0
    static_parseable = 0
    unknown_state_supported = False

    for case in cases:
        result = dispatch_certificate_text(case.text, grammar)
        transition_error = _transition_error(case.expected_state, result)
        if transition_error:
            transition_errors += 1
        else:
            supported.add(case.expected_state)
            if case.expected_state == "UNKNOWN":
                unknown_state_supported = True
        dynamic_parseable += int(result.parseable)
        static_parseable += int(result.existing_parser_parseable)
        rows.append(
            {
                "name": case.name,
                "expected_state": case.expected_state,
                "dispatched_state": result.dispatched_state,
                "parser_kind": result.parser_kind,
                "dynamic_parseable": result.parseable,
                "existing_parser_parseable": result.existing_parser_parseable,
                "existing_parser_errors": result.existing_parser_errors,
                "errors": result.errors,
                "unknown_semantics_preserved": _unknown_semantics_preserved(result),
            }
        )

    dynamic_rate = _rate(dynamic_parseable, len(cases))
    static_rate = _rate(static_parseable, len(cases))
    return {
        "certificate_states_supported": sorted(supported),
        "unknown_state_supported": unknown_state_supported,
        "state_transition_error_count": transition_errors,
        "dynamic_parse_rate": dynamic_rate,
        "static_gbnf_proxy_parse_rate": static_rate,
        "parse_rate_delta_over_static_gbnf_proxy": round(dynamic_rate - static_rate, 6),
        "mask_generation_ms_per_token_proxy": measure_mask_generation_ms_per_token_proxy(
            [case.text for case in cases], grammar=grammar, timer=mask_timer
        ),
        "dispatch_results": rows,
    }


def measure_mask_generation_ms_per_token_proxy(
    texts: Sequence[str],
    *,
    grammar: CompiledBranchGrammars,
    timer: Timer = time.perf_counter,
) -> float:
    """Approximate mask work by testing branch eligibility over every text prefix."""

    start = timer()
    prefix_count = 0
    for text in texts:
        prefix: list[str] = []
        for token in str(text or "").split() or [""]:
            prefix.append(token)
            _match_branch(" ".join(prefix), grammar)
            prefix_count += 1
    elapsed_ms = (timer() - start) * 1000.0
    return round(elapsed_ms / max(prefix_count, 1), 6)


def grammar_backend_candidates(
    *,
    import_checker: Callable[[str], bool] = backend_bakeoff._module_available,
    cli_finder: Callable[[str], str | None] = backend_bakeoff._find_cli,
    help_runner: Callable[[str], str] = backend_bakeoff._help_text,
) -> list[dict[str, Any]]:
    """Summarize local structured-generation candidates for the dynamic dry-run."""

    records = backend_bakeoff.probe_backends(
        import_checker=import_checker,
        cli_finder=cli_finder,
        help_runner=help_runner,
    )
    by_name = {str(record["name"]): record for record in records}
    xgrammar = by_name.get("xgrammar", {})
    xgrammar_import = bool(xgrammar.get("import_available"))
    xgrammar_cli = bool(xgrammar.get("cli_available"))
    llama_cpp = by_name.get("llama_cpp_gbnf", {})
    return [
        {
            "name": "xgrammar2_tagdispatch_native",
            "import_name": "xgrammar",
            "import_available": xgrammar_import,
            "cli_available": xgrammar_cli,
            "available": xgrammar_import,
            "constrained_generation": True,
            "dynamic_dispatch": True,
            "fallback_only": False,
            "failure_reason": None if xgrammar_import else "xgrammar_import_absent",
        },
        {
            "name": "llama_cpp_gbnf_static",
            "import_name": llama_cpp.get("import_name"),
            "import_available": bool(llama_cpp.get("import_available")),
            "cli_available": bool(llama_cpp.get("cli_available")),
            "available": bool(llama_cpp.get("available")),
            "constrained_generation": bool(llama_cpp.get("constrained_generation")),
            "dynamic_dispatch": False,
            "fallback_only": False,
            "failure_reason": llama_cpp.get("failure_reason"),
        },
        {
            "name": "pure_python_tagdispatch_shim",
            "import_name": None,
            "import_available": True,
            "cli_available": False,
            "available": True,
            "constrained_generation": False,
            "dynamic_dispatch": True,
            "fallback_only": True,
            "failure_reason": None,
        },
    ]


def build_dryrun_artifact(
    *,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    import_checker: Callable[[str], bool] = backend_bakeoff._module_available,
    cli_finder: Callable[[str], str | None] = backend_bakeoff._find_cli,
    help_runner: Callable[[str], str] = backend_bakeoff._help_text,
    timer_factory: TimerFactory = default_timer_factory,
) -> dict[str, Any]:
    """Build the completed Exp 1339 artifact without model inference."""

    compile_timer = timer_factory()
    grammar = compile_branch_grammars(timer=compile_timer)
    summary = evaluate_synthetic_cases(
        synthetic_certificate_cases(),
        grammar=grammar,
        mask_timer=timer_factory(),
    )
    candidates = grammar_backend_candidates(
        import_checker=import_checker,
        cli_finder=cli_finder,
        help_runner=help_runner,
    )
    required_supported = set(REQUIRED_STATES).issubset(summary["certificate_states_supported"])
    ready = bool(required_supported and summary["state_transition_error_count"] == 0)
    artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": "complete",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": "REQ-VERIFY-1339",
            "source_experiments": ["1324", "1337"],
        },
        "llm_inference_run": False,
        "sota_model_called": False,
        "grammar_backend_candidates": candidates,
        "dynamic_grammar_compile_ms": grammar.compile_ms,
        "mask_generation_ms_per_token_proxy": summary["mask_generation_ms_per_token_proxy"],
        "certificate_states_supported": summary["certificate_states_supported"],
        "unknown_state_supported": summary["unknown_state_supported"],
        "state_transition_error_count": summary["state_transition_error_count"],
        "parse_rate_delta_over_static_gbnf_proxy": summary[
            "parse_rate_delta_over_static_gbnf_proxy"
        ],
        "dynamic_grammar_ready": ready,
        "honest_verdict": _honest_verdict(ready=ready, candidates=candidates),
        "timing_proxy_method": {
            "compile": "time to compile local regex grammar fragments",
            "mask": "time to check every whitespace-token prefix against branch patterns",
            "limitation": "no token-level XGrammar mask was generated because this dry-run does not load an LLM",
        },
        "synthetic_cases": [
            {"name": case.name, "expected_state": case.expected_state, "text": case.text}
            for case in synthetic_certificate_cases()
        ],
        "dispatch_results": summary["dispatch_results"],
        "dynamic_parse_rate": summary["dynamic_parse_rate"],
        "static_gbnf_proxy_parse_rate": summary["static_gbnf_proxy_parse_rate"],
    }
    return artifact


def write_in_progress_artifact(
    path: Path | str,
    *,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Write the required bootstrap artifact before local probes run."""

    artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": "in_progress",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": "REQ-VERIFY-1339",
        },
    }
    _write_json(Path(path), artifact)
    return artifact


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    import_checker: Callable[[str], bool] = backend_bakeoff._module_available,
    cli_finder: Callable[[str], str | None] = backend_bakeoff._find_cli,
    help_runner: Callable[[str], str] = backend_bakeoff._help_text,
    timer_factory: TimerFactory = default_timer_factory,
) -> dict[str, Any]:
    """Write the in-progress and completed Exp 1339 artifacts."""

    output = Path(output_path)
    write_in_progress_artifact(output, run_date=run_date, project_root=project_root)
    artifact = build_dryrun_artifact(
        run_date=run_date,
        project_root=project_root,
        import_checker=import_checker,
        cli_finder=cli_finder,
        help_runner=help_runner,
        timer_factory=timer_factory,
    )
    _write_json(output, artifact)
    return artifact


def _match_branch(text: str, grammar: CompiledBranchGrammars) -> BranchGrammar | None:
    for branch in grammar.branches:
        if branch.pattern.search(text):
            return branch
    return None


def _transition_error(expected_state: str, result: DispatchResult) -> bool:
    if result.dispatched_state != expected_state or not result.parseable:
        return True
    if expected_state == "UNKNOWN":
        return not _unknown_semantics_preserved(result)
    return False


def _unknown_semantics_preserved(result: DispatchResult) -> bool:
    if result.dispatched_state != "UNKNOWN" or not result.parseable:
        return False
    return _normalised_state(result.certificate.get("final_answer")) == "UNKNOWN"


def _normalised_state(value: Any) -> str:
    text = str(value or "").upper()
    if "UNSATISFIABLE" in text or "UNSAT" in text:
        return "UNSAT"
    if "SATISFIABLE" in text or "SAT" in text:
        return "SAT"
    if "UNKNOWN" in text or "UNDETERMINED" in text or "ABSTAIN" in text:
        return "UNKNOWN"
    return text


def _json_certificate(label: str) -> str:
    return json.dumps(
        {
            "claims": [{"id": "c1", "text": f"synthetic branch predicts {label}"}],
            "equations": [{"lhs": "final_label", "relation": "=", "rhs": label}],
            "final_answer": label,
            "confidence": 0.72,
            "verifier_routes": [{"claim_id": "c1", "verifier": "z3_math"}],
            "proof_numbers": [1.0],
        },
        sort_keys=True,
    )


def _repair_hint_certificate(text: str) -> dict[str, Any]:
    hint = " ".join(str(text or "").split())[:240] or "repair requested"
    return {
        "claims": [{"id": "c1", "text": hint}],
        "equations": [{"lhs": "repair_required", "relation": "=", "rhs": "true"}],
        "final_answer": "ABSTAIN",
        "confidence": 0.25,
        "verifier_routes": [{"claim_id": "c1", "verifier": "z3_math"}],
        "proof_numbers": [0.0],
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _honest_verdict(*, ready: bool, candidates: Sequence[Mapping[str, Any]]) -> str:
    if not ready:
        return "dryrun_not_ready_state_transition_errors"
    native = next(
        candidate
        for candidate in candidates
        if candidate.get("name") == "xgrammar2_tagdispatch_native"
    )
    if native.get("available"):
        return "dryrun_ready_native_xgrammar_importable"
    return "dryrun_ready_pure_python_tagdispatch_xgrammar_absent"


def _module_available(name: str) -> bool:
    return backend_bakeoff._module_available(name)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - thin CLI wrapper covered through run_experiment.
    run_experiment(project_root=Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
