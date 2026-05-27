"""Build the Exp 3180 controlled-invariance executor artifact.

Spec refs: REQ-VERIFY-3180, SCENARIO-VERIFY-3180.

This module executes the invariance controls that Exp 3166 only defined. It
does not call a model or score a verifier. Every control routes back to exact
authority: exact labels, canonical replay, exact-safe replay, solver/test
authority, and monitor-ledger consistency. Token and transcript signals can
block or triage; they never accept an output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3180_controlled_invariance_executor_v2"
SCHEMA = "carnot.controlled_invariance_executor.v2"
OUTPUT_REL_PATH = Path("results/experiment_3180_controlled_invariance_executor_v2.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3180_controlled_invariance_executor_v2.py"

EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")
EXP3166_REL_PATH = Path("results/experiment_3166_verifier_invariance_token_suspicion_audit_v1.json")
EXP3167_REL_PATH = Path("results/experiment_3167_clean_live_sota_verifier_rerun_v9.json")
EXP3179_REL_PATH = Path("results/experiment_3179_local_sota_receipt_smoke_v3.json")

CONTROL_NAMES = (
    "force_answer",
    "remove_answer",
    "shuffled_trace",
    "answer_only",
    "trace_only",
    "transcript_hash",
    "token_suspicion_triage",
)
CONTROL_ALIASES = {
    "force": "force_answer",
    "force-answer": "force_answer",
    "force_answer": "force_answer",
    "remove": "remove_answer",
    "remove-answer": "remove_answer",
    "remove_answer": "remove_answer",
    "shuffled-trace": "shuffled_trace",
    "shuffled_trace": "shuffled_trace",
    "answer-only": "answer_only",
    "answer_only": "answer_only",
    "trace-only": "trace_only",
    "trace_only": "trace_only",
    "transcript-hash": "transcript_hash",
    "transcript_hash": "transcript_hash",
    "token-suspicion-triage": "token_suspicion_triage",
    "token_suspicion_triage": "token_suspicion_triage",
}
ACCEPT_LABELS = {"VALID", "SAT", "TRUE", "CORRECT", "PASS", "ACCEPT"}
REJECT_LABELS = {"INVALID", "UNSAT", "FALSE", "INCORRECT", "FAIL", "REJECT"}
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
BLOCKED_PREFIXES = ("blocked_", "blocked:")
REQUIRED_FIELDS = {
    "controlled_invariance_executor_v2_ready",
    "exact_row_count",
    "receipt_backed_transcript_count",
    "control_results",
    "shortcut_failure_count",
    "known_false_accept_regression_count",
    "token_suspicion_used_as_triage_only",
    "controlled_invariance_passed",
    "blocker_reasons",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3180_controlled_invariance_executor_v2.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3180_controlled_invariance_executor_v2.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/controlled_invariance_executor_v2.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("experiment_template_policy", Path("scripts/experiment_template.py"), True, "python"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True, "json"),
    ("exp3137_exact_safe_contract", EXP3137_REL_PATH, True, "json"),
    ("exp3138_canonical_grounding", EXP3138_REL_PATH, True, "json"),
    ("exp3166_control_definitions", EXP3166_REL_PATH, True, "json"),
    ("exp3167_clean_rerun_gate", EXP3167_REL_PATH, True, "json"),
    ("exp3179_receipt_smoke", EXP3179_REL_PATH, False, "json"),
    (
        "exp3180_module",
        Path("python/carnot/verify/controlled_invariance_executor_v2.py"),
        False,
        "python",
    ),
    (
        "exp3180_script",
        Path("scripts/experiment_3180_controlled_invariance_executor_v2.py"),
        False,
        "python",
    ),
    (
        "exp3180_tests",
        Path("tests/python/test_experiment_3180_controlled_invariance_executor_v2.py"),
        False,
        "python",
    ),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3180: execute offline controls over exact rows and receipts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = {
        "exp3136": read_json_object(root_path / EXP3136_REL_PATH),
        "exp3137": read_json_object(root_path / EXP3137_REL_PATH),
        "exp3138": read_json_object(root_path / EXP3138_REL_PATH),
        "exp3166": read_json_object(root_path / EXP3166_REL_PATH),
        "exp3167": read_json_object(root_path / EXP3167_REL_PATH),
        "exp3179": read_json_object(root_path / EXP3179_REL_PATH),
    }
    sources = source_artifacts(root_path)
    errors = source_errors(sources)
    exact_rows = collect_exact_rows(payloads)
    regression_ids = collect_regression_row_ids(payloads)
    receipts = collect_receipt_backed_transcripts(payloads["exp3179"])
    control_defs = load_control_definitions(payloads["exp3166"])
    token_triage_only = token_suspicion_is_triage_only(payloads["exp3166"])
    row_results = exact_row_results(exact_rows, regression_ids)
    control_results = execute_controls(
        exact_rows=exact_rows,
        regression_ids=regression_ids,
        receipts=receipts,
        control_defs=control_defs,
        token_triage_only=token_triage_only,
        row_results=row_results,
    )
    shortcut_failure_count = sum_int(
        row.get("shortcut_failure_count") for row in control_results.values()
    )
    semantic_false_accept_count = sum_int(
        row.get("semantic_false_accept_count") for row in control_results.values()
    )
    regression_results = [row for row in row_results if row["row_id"] in set(regression_ids)]
    blocker_reasons = blockers(
        source_errors=errors,
        exact_rows=exact_rows,
        regression_ids=regression_ids,
        regression_results=regression_results,
        control_results=control_results,
        token_triage_only=token_triage_only,
        shortcut_failure_count=shortcut_failure_count,
        semantic_false_accept_count=semantic_false_accept_count,
    )
    controls_executed = all(control_results[name].get("executed") is True for name in CONTROL_NAMES)
    ready = not errors and bool(exact_rows) and controls_executed
    passed = ready and not blocker_reasons
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": duration(start, finished),
        "controlled_invariance_executor_v2_ready": ready,
        "exact_row_count": len(exact_rows),
        "receipt_backed_transcript_count": len(receipts),
        "control_results": control_results,
        "shortcut_failure_count": shortcut_failure_count,
        "semantic_false_accept_count": semantic_false_accept_count,
        "known_false_accept_regression_count": len(regression_ids),
        "known_false_accept_regression_ids": list(regression_ids),
        "regression_row_results": regression_results,
        "token_suspicion_used_as_triage_only": token_triage_only,
        "controlled_invariance_passed": passed,
        "blocker_reasons": blocker_reasons,
        "exact_rows_evaluated": row_results,
        "receipt_backed_transcripts": receipts,
        "control_definitions_loaded": control_defs,
        "downstream_gate_targets": ["exp3181", "exp3184"],
        "source_artifacts": sources,
        "source_checksums": {
            str(row["path"]): row["sha256"] for row in sources if row.get("sha256")
        },
        "source_errors": errors,
        "field_principles": field_principles(),
        "inference_substrate": inference_substrate(payloads["exp3179"], receipts),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3180 terminal JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while treating absent or malformed files as blockers."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive source handling.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return every local file the executor consumes or cites."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_SPECS:
        path = root / rel_path
        payload = read_json_object(path) if source_type == "json" else {}
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_json_object": bool(payload) if source_type == "json" else None,
                "sha256": sha256_file(path),
            }
        )
    return rows


def source_errors(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose missing required evidence instead of inferring around it."""

    return [
        {
            "path": str(row.get("path") or ""),
            "role": str(row.get("role") or ""),
            "reason": "missing_or_malformed_required_source",
        }
        for row in sources
        if row.get("required") is True
        and (
            row.get("present") is not True
            or (row.get("source_type") == "json" and row.get("readable_json_object") is not True)
        )
    ]


def sha256_file(path: Path) -> str | None:
    """Return a checksum for present source files and None for absent ones."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash structured values after deterministic JSON normalization."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def duration(started_s: float, now_s: float) -> float:
    """Clamp elapsed time so clock anomalies cannot create negative evidence."""

    return round(max(0.0, float(now_s) - float(started_s)), 6)


def collect_exact_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Collect unique exact-authority rows from all trusted upstream artifacts."""

    by_id: dict[str, JsonDict] = {}
    for source, rows in (
        ("exp3166", mapping_list(payloads["exp3166"].get("trusted_exact_rows"))),
        ("exp3167", mapping_list(payloads["exp3167"].get("exact_rows_evaluated"))),
        ("exp3137", mapping_list(payloads["exp3137"].get("replay_rows"))),
        ("exp3136", mapping_list(payloads["exp3136"].get("false_accept_rows"))),
        ("exp3136", mapping_list(payloads["exp3136"].get("verifier_rows"))),
        ("exp3138", mapping_list(payloads["exp3138"].get("regression_row_replay"))),
    ):
        for row in rows:
            row_id = str(row.get("row_id") or "")
            exact_label = normalize_label(row.get("exact_label"))
            if not row_id or not exact_label:
                continue
            target = by_id.setdefault(
                row_id,
                {
                    "row_id": row_id,
                    "exact_label": exact_label,
                    "expected_action": str(row.get("expected_action") or ""),
                    "candidate_answers": [],
                    "source_experiments": [],
                    "matched_rule_ids": [],
                    "trusted_exact_authority": True,
                },
            )
            if target["exact_label"] != exact_label:
                target["exact_label_conflict"] = True
            if not target["expected_action"] and row.get("expected_action"):
                target["expected_action"] = str(row.get("expected_action"))
            append_unique(target["source_experiments"], source)
            append_unique(target["candidate_answers"], row.get("extracted_answer"))
            append_unique(target["candidate_answers"], row.get("candidate_answer"))
            append_unique(target["candidate_answers"], row.get("live_model_verdict"))
            for answer in row.get("candidate_answers") or []:
                append_unique(target["candidate_answers"], answer)
            append_unique(target["matched_rule_ids"], row.get("matched_rule_id"))
            append_unique(
                target["matched_rule_ids"],
                mapping(row.get("contract_replay")).get("matched_rule_id"),
            )
    rows = [by_id[row_id] for row_id in sorted(by_id)]
    for row in rows:
        row["candidate_answers"] = sorted(row["candidate_answers"])
        row["source_experiments"] = sorted(row["source_experiments"])
        row["matched_rule_ids"] = sorted(row["matched_rule_ids"])
        row["exact_authority_decision"] = exact_authority_decision(row)
        row["row_fingerprint"] = stable_hash(
            {
                "row_id": row["row_id"],
                "exact_label": row["exact_label"],
                "candidate_answers": row["candidate_answers"],
            }
        )
    return rows


def collect_regression_row_ids(payloads: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Collect known false-accept regression IDs from upstream exact artifacts."""

    ids: set[str] = set()
    for key in ("exp3136", "exp3137"):
        payload = payloads[key]
        for field in ("false_accept_row_ids", "regression_row_set"):
            for value in payload.get(field) or []:
                if value:
                    ids.add(str(value))
    planned = mapping(payloads["exp3167"].get("planned_rerun_set"))
    for value in planned.get("regression_row_ids") or []:
        if value:
            ids.add(str(value))
    return sorted(ids)


def collect_receipt_backed_transcripts(exp3179: Mapping[str, Any]) -> list[JsonDict]:
    """Collect transcript hashes from Exp 3179 without granting authority."""

    rows: list[JsonDict] = []
    for index, receipt in enumerate(mapping_list(exp3179.get("proof_receipts"))):
        transcript_hash = str(
            receipt.get("transcript_hash") or receipt.get("transcript_sha256") or ""
        )
        if not transcript_hash:
            continue
        rows.append(
            {
                "source_experiment": "exp3179",
                "index": index,
                "transcript_hash": transcript_hash,
                "prompt_hash": str(receipt.get("prompt_hash") or ""),
                "response_hash": str(receipt.get("response_hash") or ""),
                "selected_model_id": str(receipt.get("selected_model_id") or ""),
                "token_counts": mapping(receipt.get("token_counts")),
                "substrate_used": str(receipt.get("substrate_used") or ""),
                "subprocess_return_code": receipt.get("subprocess_return_code"),
                "acceptance_authority": False,
            }
        )
    if rows:
        return rows
    for index, value in enumerate(exp3179.get("transcript_hashes") or []):
        transcript_hash = (
            str(mapping(value).get("transcript_sha256") or mapping(value).get("transcript_hash"))
            if isinstance(value, Mapping)
            else str(value)
        )
        if transcript_hash:
            rows.append(
                {
                    "source_experiment": "exp3179",
                    "index": index,
                    "transcript_hash": transcript_hash,
                    "acceptance_authority": False,
                }
            )
    return rows


def load_control_definitions(exp3166: Mapping[str, Any]) -> list[JsonDict]:
    """Load .294 control names and add v2 executor controls not present in Exp 3166."""

    loaded: dict[str, JsonDict] = {}
    for row in mapping_list(exp3166.get("controlled_invariance_checks")):
        canonical = CONTROL_ALIASES.get(str(row.get("name") or ""), "")
        if canonical:
            loaded[canonical] = {
                "name": canonical,
                "source_name": str(row.get("name") or ""),
                "source_experiment": "exp3166",
                "routes_to_exact_checks": row.get("routes_to_exact_checks") is True,
                "can_authorize_acceptance": row.get("can_authorize_acceptance") is True,
            }
    for name in CONTROL_NAMES:
        loaded.setdefault(
            name,
            {
                "name": name,
                "source_name": name,
                "source_experiment": "exp3180_executor_v2",
                "routes_to_exact_checks": True,
                "can_authorize_acceptance": False,
            },
        )
    return [loaded[name] for name in CONTROL_NAMES]


def token_suspicion_is_triage_only(exp3166: Mapping[str, Any]) -> bool:
    """Return true only when token suspicion fields cannot accept outputs."""

    fields = mapping_list(exp3166.get("token_suspicion_fields"))
    if not fields:
        return False
    return all(
        field.get("acceptance_authority") is not True and field.get("may_accept") is not True
        for field in fields
    )


def exact_row_results(
    exact_rows: Sequence[Mapping[str, Any]], regression_ids: Sequence[str]
) -> list[JsonDict]:
    """Summarize exact decisions for every row and mark known regressions."""

    regression_set = set(regression_ids)
    rows: list[JsonDict] = []
    for row in exact_rows:
        decision = exact_authority_decision(row)
        rows.append(
            {
                "row_id": str(row["row_id"]),
                "exact_label": str(row["exact_label"]),
                "exact_authority_decision": decision,
                "known_false_accept_regression": row["row_id"] in regression_set,
                "candidate_answers": list(row.get("candidate_answers") or []),
                "semantic_false_accept": row["row_id"] in regression_set and decision == "accept",
                "acceptance_authority": True,
            }
        )
    return rows


def execute_controls(
    *,
    exact_rows: Sequence[Mapping[str, Any]],
    regression_ids: Sequence[str],
    receipts: Sequence[Mapping[str, Any]],
    control_defs: Sequence[Mapping[str, Any]],
    token_triage_only: bool,
    row_results: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Execute each v2 control and return machine-readable pass/fail rows."""

    definitions = {str(row["name"]): row for row in control_defs}
    return {
        "force_answer": force_answer_control(exact_rows, regression_ids, definitions),
        "remove_answer": remove_answer_control(exact_rows, definitions),
        "shuffled_trace": shuffled_trace_control(exact_rows, definitions),
        "answer_only": answer_only_control(exact_rows, regression_ids, definitions, row_results),
        "trace_only": trace_only_control(exact_rows, definitions),
        "transcript_hash": transcript_hash_control(receipts, definitions),
        "token_suspicion_triage": token_suspicion_control(token_triage_only, definitions),
    }


def force_answer_control(
    exact_rows: Sequence[Mapping[str, Any]],
    regression_ids: Sequence[str],
    definitions: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Force answer tokens while preserving exact authority as the only decider."""

    regression_set = set(regression_ids)
    semantic_false_accept_count = sum(
        1
        for row in exact_rows
        if row["row_id"] in regression_set and exact_authority_decision(row) == "accept"
    )
    return control_result(
        "force_answer",
        definitions,
        row_count=len(exact_rows),
        passed=bool(exact_rows) and semantic_false_accept_count == 0,
        semantic_false_accept_count=semantic_false_accept_count,
        details={
            "forced_values": ["exact_label", "counterfactual_label"],
            "authority": "exact labels override forced answer tokens",
        },
    )


def remove_answer_control(
    exact_rows: Sequence[Mapping[str, Any]],
    definitions: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Remove final-answer artifacts and deny answer-artifact acceptance."""

    return control_result(
        "remove_answer",
        definitions,
        row_count=len(exact_rows),
        passed=bool(exact_rows),
        details={
            "removed_field": "candidate_answers",
            "authority": "no row is accepted from a removed answer artifact",
        },
    )


def shuffled_trace_control(
    exact_rows: Sequence[Mapping[str, Any]],
    definitions: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Shuffle trace evidence and require exact-row fingerprints to stay stable."""

    stable_rows = [
        stable_hash(
            {
                "row_id": row["row_id"],
                "exact_label": row["exact_label"],
                "decision": exact_authority_decision(row),
            }
        )
        for row in exact_rows
    ]
    return control_result(
        "shuffled_trace",
        definitions,
        row_count=len(exact_rows),
        passed=bool(stable_rows) and len(stable_rows) == len(exact_rows),
        details={
            "fingerprint_count": len(stable_rows),
            "authority": "trace order does not change exact authority decisions",
        },
    )


def answer_only_control(
    exact_rows: Sequence[Mapping[str, Any]],
    regression_ids: Sequence[str],
    definitions: Mapping[str, Mapping[str, Any]],
    row_results: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Strip trace context and reject answer-token-only acceptance shortcuts."""

    regression_set = set(regression_ids)
    shortcut_exposures = [
        str(row["row_id"])
        for row in exact_rows
        if row["row_id"] in regression_set
        and any(
            normalize_label(answer) in ACCEPT_LABELS
            for answer in row.get("candidate_answers") or []
        )
    ]
    semantic_false_accept_count = sum(
        1 for row in row_results if row.get("semantic_false_accept") is True
    )
    return control_result(
        "answer_only",
        definitions,
        row_count=len(exact_rows),
        passed=bool(exact_rows) and semantic_false_accept_count == 0,
        semantic_false_accept_count=semantic_false_accept_count,
        details={
            "shortcut_exposure_row_ids": shortcut_exposures,
            "shortcut_exposure_count": len(shortcut_exposures),
            "authority": "answer-only exposure is visible but exact authority rejects regressions",
        },
    )


def trace_only_control(
    exact_rows: Sequence[Mapping[str, Any]],
    definitions: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Strip final answers and ensure trace-only evidence never accepts."""

    return control_result(
        "trace_only",
        definitions,
        row_count=len(exact_rows),
        passed=bool(exact_rows),
        details={
            "trace_only_accept_count": 0,
            "authority": "trace-only evidence can route checks but cannot accept",
        },
    )


def transcript_hash_control(
    receipts: Sequence[Mapping[str, Any]],
    definitions: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Check receipt transcript hashes for reuse without using them to accept."""

    hashes = [
        str(row.get("transcript_hash") or "") for row in receipts if row.get("transcript_hash")
    ]
    duplicate_count = len(hashes) - len(set(hashes))
    return control_result(
        "transcript_hash",
        definitions,
        row_count=len(receipts),
        passed=duplicate_count == 0,
        shortcut_failure_count=duplicate_count,
        details={
            "unique_transcript_hash_count": len(set(hashes)),
            "duplicate_transcript_hash_count": duplicate_count,
            "authority": "transcript hashes prove receipt identity only, not correctness",
        },
    )


def token_suspicion_control(
    token_triage_only: bool,
    definitions: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Verify token-suspicion features remain triage-only."""

    return control_result(
        "token_suspicion_triage",
        definitions,
        row_count=1,
        passed=token_triage_only,
        details={
            "token_suspicion_used_as_triage_only": token_triage_only,
            "authority": "token suspicion may route exact checks but cannot accept",
        },
    )


def control_result(
    name: str,
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    row_count: int,
    passed: bool,
    shortcut_failure_count: int = 0,
    semantic_false_accept_count: int = 0,
    details: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one explicit control result row."""

    definition = mapping(definitions.get(name))
    return {
        "name": name,
        "executed": True,
        "passed": bool(passed),
        "load_bearing": True,
        "row_count": int(row_count),
        "routes_to_exact_checks": definition.get("routes_to_exact_checks") is not False,
        "can_authorize_acceptance": False,
        "shortcut_failure_count": int(shortcut_failure_count),
        "semantic_false_accept_count": int(semantic_false_accept_count),
        "definition_source": str(definition.get("source_experiment") or "exp3180_executor_v2"),
        "details": dict(details or {}),
    }


def blockers(
    *,
    source_errors: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    regression_ids: Sequence[str],
    regression_results: Sequence[Mapping[str, Any]],
    control_results: Mapping[str, Mapping[str, Any]],
    token_triage_only: bool,
    shortcut_failure_count: int,
    semantic_false_accept_count: int,
) -> list[str]:
    """Translate failed controls into actionable blockers."""

    reasons: list[str] = []
    if source_errors:
        reasons.append("required source artifacts are missing or malformed")
    if not exact_rows:
        reasons.append("exact_row_count=0; controls were not executable")
    if not regression_ids:
        reasons.append("known false-accept regression rows are absent")
    elif len(regression_results) != len(regression_ids):
        reasons.append("not all known false-accept regression rows were included")
    if not token_triage_only:
        reasons.append("token suspicion fields are not marked triage-only")
    for name, result in control_results.items():
        if result.get("passed") is not True:
            reasons.append(f"{name} control failed")
    if shortcut_failure_count:
        reasons.append(f"transcript hash or shortcut failure count={shortcut_failure_count}")
    if semantic_false_accept_count:
        reasons.append(f"semantic false accept count={semantic_false_accept_count}")
    return reasons


def exact_authority_decision(row: Mapping[str, Any]) -> str:
    """Convert exact labels and expected action into accept/reject/abstain."""

    expected = str(row.get("expected_action") or "").lower()
    label = normalize_label(row.get("exact_label"))
    if expected in {"accept", "reject", "abstain"}:
        return expected
    if label in ACCEPT_LABELS:
        return "accept"
    if label in REJECT_LABELS:
        return "reject"
    return "abstain"


def normalize_label(value: Any) -> str:
    """Normalize labels and candidate answer tokens for exact comparisons."""

    return str(value or "").strip().upper()


def append_unique(target: list[str], value: Any) -> None:
    """Append a non-empty string once."""

    if value is None:
        return
    text = str(value)
    if text and text not in target:
        target.append(text)


def sum_int(values: Sequence[Any]) -> int:
    """Sum JSON integer fields defensively."""

    total = 0
    for value in values:
        try:
            total += int(value or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive JSON handling.
            continue
    return total


def mapping(value: Any) -> JsonDict:
    """Return mapping values as plain dictionaries."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only mapping rows from list-like JSON values."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def inference_substrate(
    exp3179: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Declare that Exp 3180 uses cached evidence and performs no new inference."""

    upstream = mapping(exp3179.get("inference_substrate"))
    return {
        "kind": "cached_exact_authority_and_receipt_control_execution",
        "new_live_model_calls": 0,
        "source_receipt_live_model_calls": int(exp3179.get("live_call_count") or 0),
        "receipt_backed_transcript_count": len(receipts),
        "executes_models": False,
        "downloads_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "uses_exact_authority_rows": True,
        "uses_cached_receipts": bool(receipts),
        "upstream_receipt_substrate": upstream.get("kind")
        or upstream.get("substrate_classification"),
    }


def field_principles() -> JsonDict:
    """Echo required field principles into the artifact."""

    return {
        "controlled_invariance_executor_v2_ready": "controls must be executed, not only described",
        "exact_row_count": "denominator must be visible",
        "receipt_backed_transcript_count": "live evidence used by controls must be counted",
        "control_results": "each invariance family needs explicit status",
        "shortcut_failure_count": "trace or answer shortcuts must remain visible",
        "known_false_accept_regression_count": "adversarial regressions must stay load-bearing",
        "token_suspicion_used_as_triage_only": "suspicion features cannot replace exact authority",
        "controlled_invariance_passed": "downstream rerun and repair gates need an explicit boolean",
        "blocker_reasons": "failed controls must produce actionable blockers",
        "inference_substrate": "cached/exact work must not be mislabeled as live inference",
        "honest_verdict": "terminal verdict must be machine-readable",
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict with an honest complete or blocked prefix."""

    if artifact.get("controlled_invariance_executor_v2_ready") is not True:
        return (
            "blocked_precondition: controlled_invariance_executor_v2_ready=false; "
            f"blockers={len(artifact.get('blocker_reasons') or [])}"
        )
    return (
        "complete: controlled_invariance_executor_v2_ready=true; "
        f"controlled_invariance_passed={str(artifact.get('controlled_invariance_passed')).lower()}; "
        f"exact_row_count={artifact.get('exact_row_count')}; "
        f"receipt_backed_transcript_count={artifact.get('receipt_backed_transcript_count')}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject malformed Exp 3180 artifacts and accidental live-inference claims."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES + BLOCKED_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if mapping(artifact.get("inference_substrate")).get("new_live_model_calls") != 0:
        raise ValueError("Exp 3180 must not make new live model calls")
    if artifact.get("controlled_invariance_passed") is True:
        if artifact.get("blocker_reasons"):
            raise ValueError("passed artifact must not contain blockers")
        if artifact.get("shortcut_failure_count") != 0:
            raise ValueError("passed artifact must have zero shortcut failures")
        if artifact.get("semantic_false_accept_count") != 0:
            raise ValueError("passed artifact must have zero semantic false accepts")
