"""Exp 1352 CPU-only certificate completion-budget preflight.

Spec: REQ-VERIFY-1352, SCENARIO-VERIFY-1352
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot.eval import certificate_grammar_backend_bakeoff as backend_bakeoff
from carnot.reporting import xgrammar2_tagdispatch_certificate_grammar_dryrun as tagdispatch


DEFAULT_RUN_DATE = "20260505"
DEFAULT_MAX_TOKENS = 96
DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1352_truncproof_xgrammar_certificate_completion_preflight.json"
)
DEFAULT_EXP1323_PATH = Path(
    "results/experiment_1323_sota_gguf_token_health_prompt_runtime_diagnostic.json"
)
ARTIFACT_NAME = "experiment_1352_truncproof_xgrammar_certificate_completion_preflight"
SCHEMA_VERSION = 1
GRAMMAR_STATES = ("SAT", "UNSAT", "UNKNOWN", "REPAIR_HINT")
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "grammar_states",
    "min_completion_tokens_by_state",
    "max_token_budget_sufficient",
    "structural_tag_supported",
    "xgrammar_backend_available",
    "dynamic_dispatch_preserved",
    "sota_run_allowed",
    "blocker_if_not_allowed",
    "honest_verdict",
)
_STRUCTURAL_TAG_RE = re.compile(
    r"^\s*<CARNOT_CERT_STATE:(SAT|UNSAT|UNKNOWN|REPAIR_HINT|REPAIR)>\s*",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
_MINIMAL_BODIES = {
    "SAT": "SAT",
    "UNSAT": "UNSAT",
    "UNKNOWN": "UNKNOWN",
    "REPAIR_HINT": "REPAIR_HINT: add bound.",
}


@dataclass(frozen=True)
class CompletionCase:
    """One minimal tagged completion used before spending model runtime.

    The body is deliberately tiny because this experiment answers a preflight
    question: can every grammar branch finish inside the active token budget at
    all?  It is not trying to measure real SOTA answer quality.
    """

    state: str
    body: str
    tagged_text: str


def compile_branch_grammars() -> tagdispatch.CompiledBranchGrammars:
    """Reuse the Exp 1339 branch surface so this gate checks the same parser."""

    return tagdispatch.compile_branch_grammars()


def normalise_state(state: str) -> str:
    """Map the human repair alias onto the canonical Exp 1339 branch name."""

    upper = str(state or "").strip().upper()
    if upper == "REPAIR":
        return "REPAIR_HINT"
    return upper


def structural_tag(state: str) -> str:
    """Return the small pre-grammar tag emitted before branch selection."""

    return f"<CARNOT_CERT_STATE:{normalise_state(state)}>"


def parse_structural_tag(text: str) -> tuple[str | None, str]:
    """Split a structural tag from the completion body without model calls."""

    raw = str(text or "")
    match = _STRUCTURAL_TAG_RE.match(raw)
    if match is None:
        return None, raw
    return normalise_state(match.group(1)), raw[match.end() :]


def synthetic_completion_cases() -> list[CompletionCase]:
    """Build one minimal structurally valid completion for every branch."""

    return [
        CompletionCase(
            state=state,
            body=_MINIMAL_BODIES[state],
            tagged_text=f"{structural_tag(state)}\n{_MINIMAL_BODIES[state]}",
        )
        for state in GRAMMAR_STATES
    ]


def estimate_completion_tokens(text: str) -> int:
    """Estimate completion size with a conservative punctuation-aware proxy."""

    return len(_TOKEN_RE.findall(str(text or "")))


def min_completion_tokens_by_state(cases: Sequence[CompletionCase]) -> dict[str, int]:
    """Measure the minimum tagged completion size used by this preflight."""

    return {case.state: estimate_completion_tokens(case.tagged_text) for case in cases}


def evaluate_tagged_dispatch(
    cases: Sequence[CompletionCase],
    *,
    grammar: tagdispatch.CompiledBranchGrammars | None = None,
) -> list[dict[str, Any]]:
    """Check that the tag-selected body still reaches the expected parser branch."""

    active_grammar = grammar or compile_branch_grammars()
    rows: list[dict[str, Any]] = []
    for case in cases:
        tag_state, body = parse_structural_tag(case.tagged_text)
        result = tagdispatch.dispatch_certificate_text(body, active_grammar)
        rows.append(
            {
                "expected_state": case.state,
                "tag_state": tag_state,
                "dispatched_state": result.dispatched_state,
                "dynamic_parseable": result.parseable,
                "parser_kind": result.parser_kind,
                "errors": list(result.errors),
                "existing_parser_parseable": result.existing_parser_parseable,
                "existing_parser_errors": list(result.existing_parser_errors),
                "completion_tokens": estimate_completion_tokens(case.tagged_text),
            }
        )
    return rows


def structural_tag_supported(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when every emitted tag names the expected branch."""

    return bool(rows) and all(row.get("tag_state") == row.get("expected_state") for row in rows)


def dynamic_dispatch_preserved(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when tag, branch dispatch, and parsing all agree."""

    return bool(rows) and all(
        row.get("tag_state") == row.get("expected_state") == row.get("dispatched_state")
        and bool(row.get("dynamic_parseable"))
        for row in rows
    )


def runtime_settings_from_artifact(exp1323_artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract the active certificate max-token setting from the prior diagnostic."""

    settings = dict((exp1323_artifact or {}).get("recommended_certificate_runtime_settings") or {})
    settings.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    return settings


def max_token_budget_check(
    runtime_settings: Mapping[str, Any],
    min_tokens: Mapping[str, int],
) -> tuple[int, int, bool]:
    """Compare active max tokens with the largest required branch completion."""

    max_tokens = int(runtime_settings.get("max_tokens") or runtime_settings.get("max_new_tokens") or 0)
    required = max(min_tokens.values(), default=0)
    return max_tokens, required, max_tokens >= required


def grammar_backend_candidates(
    *,
    import_checker: Callable[[str], bool] = backend_bakeoff._module_available,
    cli_finder: Callable[[str], str | None] = backend_bakeoff._find_cli,
    help_runner: Callable[[str], str] = backend_bakeoff._help_text,
) -> list[dict[str, Any]]:
    """Reuse the Exp 1339 backend probe and expose XGrammar availability."""

    return tagdispatch.grammar_backend_candidates(
        import_checker=import_checker,
        cli_finder=cli_finder,
        help_runner=help_runner,
    )


def xgrammar_backend_available(candidates: Sequence[Mapping[str, Any]]) -> bool:
    """Report whether the native XGrammar backend is actually importable."""

    native = next(
        candidate
        for candidate in candidates
        if candidate.get("name") == "xgrammar2_tagdispatch_native"
    )
    return bool(native.get("available"))


def build_preflight_artifact(
    *,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    runtime_settings: Mapping[str, Any] | None = None,
    exp1323_artifact: Mapping[str, Any] | None = None,
    import_checker: Callable[[str], bool] = backend_bakeoff._module_available,
    cli_finder: Callable[[str], str | None] = backend_bakeoff._find_cli,
    help_runner: Callable[[str], str] = backend_bakeoff._help_text,
    cases: Sequence[CompletionCase] | None = None,
) -> dict[str, Any]:
    """Build the terminal Exp 1352 artifact without loading any SOTA model."""

    case_list = list(cases) if cases is not None else synthetic_completion_cases()
    settings = dict(runtime_settings or runtime_settings_from_artifact(exp1323_artifact))
    settings.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    grammar = compile_branch_grammars()
    rows = evaluate_tagged_dispatch(case_list, grammar=grammar)
    min_tokens = min_completion_tokens_by_state(case_list)
    max_tokens, required_tokens, budget_ok = max_token_budget_check(settings, min_tokens)
    candidates = grammar_backend_candidates(
        import_checker=import_checker,
        cli_finder=cli_finder,
        help_runner=help_runner,
    )
    tag_ok = structural_tag_supported(rows)
    dispatch_ok = dynamic_dispatch_preserved(rows)
    xgrammar_available = xgrammar_backend_available(candidates)
    blocker = _blocker_if_not_allowed(
        max_token_budget_sufficient=budget_ok,
        dynamic_dispatch_preserved=dispatch_ok,
        max_tokens=max_tokens,
        required_tokens=required_tokens,
    )
    sota_allowed = blocker is None
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": "complete",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": "REQ-VERIFY-1352",
            "source_experiments": ["exp1339", "exp1351"],
        },
        "llm_inference_run": False,
        "sota_model_called": False,
        "grammar_states": list(GRAMMAR_STATES),
        "min_completion_tokens_by_state": min_tokens,
        "max_min_completion_tokens": required_tokens,
        "runtime_settings_used": settings,
        "max_token_budget_sufficient": budget_ok,
        "structural_tag_supported": tag_ok,
        "xgrammar_backend_available": xgrammar_available,
        "dynamic_dispatch_preserved": dispatch_ok,
        "sota_run_allowed": sota_allowed,
        "blocker_if_not_allowed": blocker,
        "honest_verdict": _honest_verdict(
            blocker=blocker,
            xgrammar_backend_available=xgrammar_available,
        ),
        "tested_dispatch_backend": "pure_python_tagdispatch_shim",
        "grammar_backend_candidates": candidates,
        "tagged_dispatch_results": rows,
        "synthetic_cases": [
            {"state": case.state, "body": case.body, "tagged_text": case.tagged_text}
            for case in case_list
        ],
        "completion_token_estimator": (
            "regex proxy: contiguous alnum/underscore spans and individual punctuation "
            "characters count as tokens"
        ),
        "sota_run_policy": (
            "allow only when max_token_budget_sufficient and dynamic_dispatch_preserved are true"
        ),
    }


def write_in_progress_artifact(
    path: Path | str,
    *,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Write a bootstrap artifact so a crash cannot leave a missing result."""

    artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": "in_progress",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": "REQ-VERIFY-1352",
        },
    }
    _write_json(Path(path), artifact)
    return artifact


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    exp1323_path: Path | str = DEFAULT_EXP1323_PATH,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    import_checker: Callable[[str], bool] = backend_bakeoff._module_available,
    cli_finder: Callable[[str], str | None] = backend_bakeoff._find_cli,
    help_runner: Callable[[str], str] = backend_bakeoff._help_text,
) -> dict[str, Any]:
    """Write the in-progress artifact, run CPU probes, and write completion."""

    output = Path(output_path)
    write_in_progress_artifact(output, run_date=run_date, project_root=project_root)
    artifact = build_preflight_artifact(
        run_date=run_date,
        project_root=project_root,
        exp1323_artifact=_load_json(Path(exp1323_path)),
        import_checker=import_checker,
        cli_finder=cli_finder,
        help_runner=help_runner,
    )
    _write_json(output, artifact)
    return artifact


def _blocker_if_not_allowed(
    *,
    max_token_budget_sufficient: bool,
    dynamic_dispatch_preserved: bool,
    max_tokens: int,
    required_tokens: int,
) -> str | None:
    if not max_token_budget_sufficient:
        return f"max_token_budget_insufficient: max_tokens={max_tokens} required={required_tokens}"
    if not dynamic_dispatch_preserved:
        return "dynamic_dispatch_not_preserved"
    return None


def _honest_verdict(
    *,
    blocker: str | None,
    xgrammar_backend_available: bool,
) -> str:
    if blocker and blocker.startswith("max_token_budget_insufficient"):
        return "blocked_max_token_budget_insufficient"
    if blocker == "dynamic_dispatch_not_preserved":
        return "blocked_dynamic_dispatch_not_preserved"
    if xgrammar_backend_available:
        return "preflight_allows_exp1353_native_xgrammar_available"
    return "preflight_allows_exp1353_pure_python_fallback_xgrammar_absent"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - exercised through run_experiment in tests.
    run_experiment(project_root=Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
