"""Experiment 4855: hostile audit of the A1 generation-coverage diagnostic.

Spec refs: REQ-ARC-WMTE-4855,
SCENARIO-ARC-WMTE-4855-A1-HOSTILE-AUDIT,
SCENARIO-ARC-WMTE-4855-NON-TEST-CLASSIFICATION.
"""

from __future__ import annotations

import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (REPO_ROOT, PYTHON_ROOT):  # pragma: no cover - direct script guard.
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

EXPERIMENT = "experiment_4855_generation_diagnostic_audit"
EXPERIMENT_ID = 4855
SCHEMA = "carnot.arc.a1_generation_diagnostic_audit_4855.v1"
SOURCE_ARTIFACT_RELATIVE_PATH = "results/experiment_4851_generation_coverage_diagnostic.json"
SOURCE_SCRIPT_RELATIVE_PATH = "python/carnot/experiment_4851_generation_coverage_diagnostic.py"
RESULT_RELATIVE_PATH = "results/experiment_4855_generation_diagnostic_audit.json"
AUDIT_REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DURATION_FLOOR_S = 0.0001
RANDOM_SEED = 4855
TERMINAL_PREFIXES = ("complete_", "blocked_", "success_")
BUCKET_ORDER = ("COVERED", "ENUMERATED_BUT_LOST", "NEVER_ENUMERATED")

SPEC_REFS = [
    "REQ-ARC-WMTE-4855",
    "SCENARIO-ARC-WMTE-4855-A1-HOSTILE-AUDIT",
    "SCENARIO-ARC-WMTE-4855-NON-TEST-CLASSIFICATION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; audit complete is complete_a1_generation_diagnostic_audited."
        )
    },
    "a1_genuinely_diagnostic": {
        "principle": (
            "the load-bearing check -- proposer-blind AND positive-control-covered AND "
            "buckets-match-claim AND live-path-reachable; else A1 is a "
            "tautology/harness non-test and its dominant-bucket finding is void."
        )
    },
    "proposer_blind_confirmed": {
        "principle": (
            "true iff the banked winner was used only to classify, never to seed the "
            "proposer (the tautology trap)."
        )
    },
    "positive_control_confirmed": {
        "principle": (
            "true iff the positive control really came out COVERED on a real adaptered game."
        )
    },
    "buckets_match_claim": {
        "principle": "true iff the per-game buckets support the claimed dominant_bucket."
    },
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts (0.0001s floor)."},
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "source_artifact_path",
    "source_script_path",
    "source_artifact_checksum",
    "source_script_checksum",
    "source_honest_verdict",
    "source_dominant_bucket",
    "live_path_reachable_confirmed",
    "solve_provenance_confirmed",
    "checks",
    "non_diagnostic_reasons",
    "summarizer_result",
    "adversarial_result",
    "live_lint_result",
    "preconditions_checked",
    "audit_report_path",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

JsonDict = dict[str, Any]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _safe_suffix(reasons: list[str]) -> str:
    if not reasons:
        return "audited"
    joined = "_".join(reasons[:3])
    return re.sub(r"[^a-z0-9_]+", "_", joined.lower()).strip("_") or "failed_checks"


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean["reproducibility_checksum"] = ""
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_summarizer(path: Path) -> JsonDict:
    from scripts import summarize_artifact

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        returncode = summarize_artifact.summarize(path)
    return {"returncode": int(returncode), "stdout": buffer.getvalue(), "stderr": ""}


def run_adversarial_verify(path: Path) -> JsonDict:
    from scripts import adversarial_verify

    return dict(adversarial_verify.verify_artifact(path))


def run_arc_orphan_solver_lint(root: Path) -> JsonDict:
    command = [sys.executable, str(root / "scripts" / "arc_orphan_solver_lint.py")]
    proc = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _call_name(callable_node: ast.AST) -> str:
    if isinstance(callable_node, ast.Name):
        return callable_node.id
    if isinstance(callable_node, ast.Attribute):
        return callable_node.attr
    return ""


def _attach_ast_parents(node: ast.AST) -> None:
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            child.__dict__["_parent"] = parent


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _first_parent_call(node: ast.AST) -> ast.Call | None:
    parent = getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, ast.Call):
            return parent
        parent = getattr(parent, "_parent", None)
    return None


def _proposer_blind_check(
    artifact: Mapping[str, Any],
    source_text: str,
) -> tuple[JsonDict, list[str]]:
    reasons: list[str] = []
    if artifact.get("proposer_blind_to_banked_answer") is not True:
        reasons.append("artifact_proposer_blind_flag_false")
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        return (
            {
                "passed": False,
                "artifact_flag": artifact.get("proposer_blind_to_banked_answer"),
                "parse_error": str(exc),
                "winning_prefix_refs": 0,
                "disallowed_refs": [],
            },
            reasons + ["a1_source_not_parseable"],
        )

    _attach_ast_parents(tree)
    function = _find_function(tree, "measure_game_with_stepwise_explorer")
    if function is None:
        return (
            {
                "passed": False,
                "artifact_flag": artifact.get("proposer_blind_to_banked_answer"),
                "function_present": False,
                "winning_prefix_refs": 0,
                "disallowed_refs": [],
            },
            reasons + ["measure_game_with_stepwise_explorer_missing"],
        )

    refs: list[JsonDict] = []
    disallowed: list[JsonDict] = []
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id == "winning_prefix":
                call = _first_parent_call(node)
                call_name = _call_name(call.func) if call is not None else ""
                record = {"line": int(getattr(node, "lineno", 0)), "call": call_name or None}
                refs.append(record)
                if call_name != "classify_game_coverage":
                    disallowed.append(record)

    if not refs:
        reasons.append("winning_prefix_not_used_for_classification")
    if disallowed:
        reasons.append("banked_answer_used_before_classification")

    return (
        {
            "passed": not reasons,
            "artifact_flag": artifact.get("proposer_blind_to_banked_answer"),
            "function_present": True,
            "winning_prefix_refs": len(refs),
            "disallowed_refs": disallowed,
            "allowed_only_in_classify_game_coverage": not disallowed and bool(refs),
        },
        reasons,
    )


def _positive_control_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    coverage = _mapping(artifact.get("positive_control_coverage"))
    reasons: list[str] = []
    if artifact.get("positive_control_covered") is not True or coverage.get("bucket") != "COVERED":
        reasons.append("positive_control_not_covered")
    if coverage.get("adaptered") is not True:
        reasons.append("positive_control_not_adaptered")
    if str(coverage.get("game") or artifact.get("positive_control_game") or "") != str(
        artifact.get("positive_control_game") or ""
    ):
        reasons.append("positive_control_game_mismatch")
    if coverage.get("reached_l1_win") is not True:
        reasons.append("positive_control_did_not_reach_l1_win")
    return (
        {
            "passed": not reasons,
            "positive_control_game": artifact.get("positive_control_game"),
            "artifact_positive_control_covered": artifact.get("positive_control_covered"),
            "coverage_bucket": coverage.get("bucket"),
            "adaptered": coverage.get("adaptered"),
            "reached_l1_win": coverage.get("reached_l1_win"),
            "winning_prefix_len": coverage.get("winning_prefix_len"),
            "matched_winning_prefix_len": coverage.get("matched_winning_prefix_len"),
        },
        reasons,
    )


def _bucket_counts(per_game_coverage: Mapping[str, Any]) -> dict[str, int]:
    counts = Counter(
        str(_mapping(row).get("bucket"))
        for row in per_game_coverage.values()
        if _mapping(row).get("bucket") in BUCKET_ORDER
    )
    return dict(counts)


def _computed_dominant_bucket(per_game_coverage: Mapping[str, Any]) -> str | None:
    counts = _bucket_counts(per_game_coverage)
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], BUCKET_ORDER.index(item[0])))[0][0]


def _bucket_distribution_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    per_game = _mapping(artifact.get("per_game_coverage"))
    counts = _bucket_counts(per_game)
    computed = _computed_dominant_bucket(per_game)
    reasons: list[str] = []
    invalid_games = [
        str(game)
        for game, row in per_game.items()
        if _mapping(row).get("bucket") not in BUCKET_ORDER
    ]
    if invalid_games:
        reasons.append("invalid_per_game_bucket")
    try:
        n_games = int(artifact.get("n_games_measured"))
    except (TypeError, ValueError):
        n_games = -1
        reasons.append("n_games_measured_not_integer")
    if n_games != len(per_game):
        reasons.append("n_games_measured_mismatch")
    if n_games < 3 or len(per_game) < 3:
        reasons.append("n_games_measured_below_3")
    if computed is None:
        reasons.append("dominant_bucket_missing")
    elif artifact.get("dominant_bucket") != computed:
        reasons.append("dominant_bucket_mismatch")
    expected_fragment = f"complete_generation_wall_{str(computed).lower()}_dominant"
    if computed is not None and artifact.get("honest_verdict") != expected_fragment:
        reasons.append("dominant_bucket_verdict_mismatch")
    return (
        {
            "passed": not reasons,
            "bucket_counts": counts,
            "computed_dominant_bucket": computed,
            "claimed_dominant_bucket": artifact.get("dominant_bucket"),
            "honest_verdict": artifact.get("honest_verdict"),
            "n_games_measured": artifact.get("n_games_measured"),
            "per_game_count": len(per_game),
            "invalid_games": invalid_games,
        },
        reasons,
    )


def _live_path_and_provenance_check(
    artifact: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
) -> tuple[JsonDict, list[str], bool, bool]:
    lint_passed = live_lint_result.get("passed") is True
    artifact_live = artifact.get("live_path_reachable") is True
    provenance = artifact.get("solve_provenance")
    live_confirmed = lint_passed and artifact_live
    provenance_confirmed = provenance == "development_proxy"
    reasons: list[str] = []
    if not live_confirmed:
        reasons.append("live_path_unreachable")
    if not provenance_confirmed:
        reasons.append("solve_provenance_not_development_proxy")
    return (
        {
            "passed": not reasons,
            "arc_orphan_solver_lint_passed": lint_passed,
            "artifact_live_path_reachable": artifact_live,
            "solve_provenance": provenance,
            "development_proxy": provenance_confirmed,
        },
        reasons,
        live_confirmed,
        provenance_confirmed,
    )


def _tool_cleanliness_check(
    summarizer_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
) -> JsonDict:
    summarizer_clean = summarizer_result.get("returncode") == 0
    adversarial_clean = (
        adversarial_result.get("loaded") is not False and adversarial_result.get("flag_count") == 0
    )
    return {
        "passed": summarizer_clean and adversarial_clean,
        "summarizer_returncode": summarizer_result.get("returncode"),
        "adversarial_flag_count": adversarial_result.get("flag_count"),
        "adversarial_loaded": adversarial_result.get("loaded"),
    }


def audit_a1_artifact(
    artifact: Mapping[str, Any],
    *,
    source_text: str,
    summarizer_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
) -> JsonDict:
    proposer_check, proposer_reasons = _proposer_blind_check(artifact, source_text)
    positive_check, positive_reasons = _positive_control_check(artifact)
    bucket_check, bucket_reasons = _bucket_distribution_check(artifact)
    live_check, live_reasons, live_confirmed, provenance_confirmed = (
        _live_path_and_provenance_check(artifact, live_lint_result)
    )
    tool_check = _tool_cleanliness_check(summarizer_result, adversarial_result)

    reasons = proposer_reasons + positive_reasons + bucket_reasons + live_reasons
    genuinely_diagnostic = not reasons
    return {
        "honest_verdict": (
            "complete_a1_generation_diagnostic_audited"
            if genuinely_diagnostic
            else f"complete_a1_generation_diagnostic_non_test_{_safe_suffix(reasons)}"
        ),
        "a1_genuinely_diagnostic": genuinely_diagnostic,
        "proposer_blind_confirmed": not proposer_reasons,
        "positive_control_confirmed": not positive_reasons,
        "buckets_match_claim": not bucket_reasons,
        "live_path_reachable_confirmed": live_confirmed,
        "solve_provenance_confirmed": provenance_confirmed,
        "non_diagnostic_reasons": reasons,
        "checks": {
            "proposer_blind_to_banked_answer": proposer_check,
            "positive_control": positive_check,
            "bucket_distribution": bucket_check,
            "live_path_and_provenance": live_check,
            "summarizer_and_adversarial_verify": tool_check,
        },
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    repo = Path(root)
    source = repo / SOURCE_ARTIFACT_RELATIVE_PATH
    script = repo / SOURCE_SCRIPT_RELATIVE_PATH
    spec = repo / SPEC_RELATIVE_PATH
    spec_text = spec.read_text(encoding="utf-8") if spec.exists() else ""
    return {
        "ok": (
            source.exists()
            and script.exists()
            and "REQ-ARC-WMTE-4855" in spec_text
            and (repo / "scripts/summarize_artifact.py").exists()
            and (repo / "scripts/adversarial_verify.py").exists()
            and (repo / "scripts/arc_orphan_solver_lint.py").exists()
        ),
        "source_artifact_present": source.exists(),
        "source_script_present": script.exists(),
        "spec_has_req_4855": "REQ-ARC-WMTE-4855" in spec_text,
        "summarizer_script_present": (repo / "scripts/summarize_artifact.py").exists(),
        "adversarial_verify_script_present": (repo / "scripts/adversarial_verify.py").exists(),
        "arc_orphan_solver_lint_present": (repo / "scripts/arc_orphan_solver_lint.py").exists(),
    }


def _blocked_artifact(checks: Mapping[str, Any]) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "source_artifact_path": SOURCE_ARTIFACT_RELATIVE_PATH,
        "source_script_path": SOURCE_SCRIPT_RELATIVE_PATH,
        "source_artifact_checksum": None,
        "source_script_checksum": None,
        "source_honest_verdict": None,
        "source_dominant_bucket": None,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": "blocked_a1_artifact_missing",
        "a1_genuinely_diagnostic": False,
        "proposer_blind_confirmed": False,
        "positive_control_confirmed": False,
        "buckets_match_claim": False,
        "live_path_reachable_confirmed": False,
        "solve_provenance_confirmed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "checks": {},
        "non_diagnostic_reasons": ["missing_a1_artifact_or_script"],
        "summarizer_result": {},
        "adversarial_result": {},
        "live_lint_result": {},
        "preconditions_checked": dict(checks),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_FLOOR_S,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    source_path: Path,
    source_script_path: Path,
    source_artifact: Mapping[str, Any],
    audit: Mapping[str, Any],
    summarizer_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "source_artifact_path": SOURCE_ARTIFACT_RELATIVE_PATH,
        "source_script_path": SOURCE_SCRIPT_RELATIVE_PATH,
        "source_artifact_checksum": file_checksum(source_path),
        "source_script_checksum": file_checksum(source_script_path),
        "source_honest_verdict": source_artifact.get("honest_verdict"),
        "source_dominant_bucket": source_artifact.get("dominant_bucket"),
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": audit.get("honest_verdict"),
        "a1_genuinely_diagnostic": audit.get("a1_genuinely_diagnostic"),
        "proposer_blind_confirmed": audit.get("proposer_blind_confirmed"),
        "positive_control_confirmed": audit.get("positive_control_confirmed"),
        "buckets_match_claim": audit.get("buckets_match_claim"),
        "live_path_reachable_confirmed": audit.get("live_path_reachable_confirmed"),
        "solve_provenance_confirmed": audit.get("solve_provenance_confirmed"),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "checks": dict(_mapping(audit.get("checks"))),
        "non_diagnostic_reasons": list(audit.get("non_diagnostic_reasons") or []),
        "summarizer_result": dict(summarizer_result),
        "adversarial_result": dict(adversarial_result),
        "live_lint_result": dict(live_lint_result),
        "preconditions_checked": dict(preconditions_checked),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(DURATION_FLOOR_S, duration_s), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    for field in (
        "a1_genuinely_diagnostic",
        "proposer_blind_confirmed",
        "positive_control_confirmed",
        "buckets_match_claim",
        "live_path_reachable_confirmed",
        "solve_provenance_confirmed",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field}_must_be_bool")
    if not isinstance(artifact.get("checks"), dict):
        errors.append("checks_must_be_dict")
    if not isinstance(artifact.get("non_diagnostic_reasons"), list):
        errors.append("non_diagnostic_reasons_must_be_list")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    duration = _finite_float(artifact.get("duration_s"))
    if duration is None or duration < DURATION_FLOOR_S:
        errors.append("duration_below_aggregation_floor")
    expected = "sha256:" + payload_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected:
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_markdown_section(artifact: Mapping[str, Any]) -> str:
    checks = _mapping(artifact.get("checks"))
    rows = [
        "",
        "## Experiment 4855 .447 A1 Generation Diagnostic Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- a1_genuinely_diagnostic: `{artifact.get('a1_genuinely_diagnostic')}`",
        f"- Non-diagnostic reasons: `{', '.join(artifact.get('non_diagnostic_reasons') or []) or '-'}`",
        f"- Inference substrate: `{artifact.get('inference_substrate')}`",
        "",
        "| Check | Passed | Detail |",
        "|---|---:|---|",
    ]
    for name, check in checks.items():
        if not isinstance(check, Mapping):
            continue
        detail = {key: value for key, value in check.items() if key != "passed"}
        rows.append(
            f"| `{name}` | `{check.get('passed')}` | `{json.dumps(detail, sort_keys=True)}` |"
        )
    rows.extend(
        [
            "",
            f"- Source artifact checksum: `{artifact.get('source_artifact_checksum')}`",
            f"- Source script checksum: `{artifact.get('source_script_checksum')}`",
            "",
        ]
    )
    return "\n".join(rows)


def append_markdown_report(
    artifact: Mapping[str, Any],
    *,
    root: Path | str = REPO_ROOT,
) -> Path:
    report_path = Path(root) / AUDIT_REPORT_RELATIVE_PATH
    marker = "## Experiment 4855 .447 A1 Generation Diagnostic Audit"
    if report_path.exists():
        current = report_path.read_text(encoding="utf-8")
        if marker in current:
            return report_path
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        current = "# ARC Null Silent-Bug Audit\n"
    report_path.write_text(current.rstrip() + render_markdown_section(artifact), encoding="utf-8")
    return report_path


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    now: Callable[[], float] | None = None,
) -> JsonDict:
    repo = Path(root)
    clock = now or time.monotonic
    start = clock()
    checks = check_preconditions(repo)
    source_path = repo / SOURCE_ARTIFACT_RELATIVE_PATH
    source_script_path = repo / SOURCE_SCRIPT_RELATIVE_PATH
    if not checks["ok"] or not source_path.exists() or not source_script_path.exists():
        checks = dict(checks)
        checks["ok"] = False
        checks["source_artifact_present"] = source_path.exists()
        checks["source_script_present"] = source_script_path.exists()
        artifact = _blocked_artifact(checks)
        if write:
            write_artifact(artifact, root=repo)
            append_markdown_report(artifact, root=repo)
        return artifact

    source_artifact = _read_json(source_path)
    source_text = source_script_path.read_text(encoding="utf-8")
    summarizer_result = run_summarizer(source_path)
    adversarial_result = run_adversarial_verify(source_path)
    live_lint_result = run_arc_orphan_solver_lint(repo)
    audit = audit_a1_artifact(
        source_artifact,
        source_text=source_text,
        summarizer_result=summarizer_result,
        adversarial_result=adversarial_result,
        live_lint_result=live_lint_result,
    )
    artifact = build_artifact(
        source_path=source_path,
        source_script_path=source_script_path,
        source_artifact=source_artifact,
        audit=audit,
        summarizer_result=summarizer_result,
        adversarial_result=adversarial_result,
        live_lint_result=live_lint_result,
        preconditions_checked=checks,
        duration_s=clock() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=repo)
        append_markdown_report(artifact, root=repo)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "a1_genuinely_diagnostic": artifact["a1_genuinely_diagnostic"],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
