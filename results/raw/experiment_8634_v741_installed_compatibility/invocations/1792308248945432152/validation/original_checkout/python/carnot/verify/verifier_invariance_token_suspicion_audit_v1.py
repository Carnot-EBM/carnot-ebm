"""Build the Exp 3166 verifier-invariance/token-suspicion audit artifact.

Spec refs: REQ-VERIFY-3166, SCENARIO-VERIFY-3166.

This module is an artifact-only audit for the next clean verifier rerun. It
does not score a verifier, call a model, or unlock repair. Its job is to keep
detector-sensitivity controls, token suspicion signals, and exact acceptance
authority visibly separate so Exp 3167 can route suspicious rows to exact
checks without allowing superficial answer or trace artifacts to accept them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3166_verifier_invariance_token_suspicion_audit_v1"
SCHEMA = "carnot.verifier_invariance_token_suspicion_audit.v1"
OUTPUT_REL_PATH = Path("results/experiment_3166_verifier_invariance_token_suspicion_audit_v1.json")
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / ("experiment_3166_verifier_invariance_token_suspicion_audit_v1.py")
)

EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")
EXP3150_REL_PATH = Path("results/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.json")
EXP3151_REL_PATH = Path("results/experiment_3151_live_inference_authenticity_preflight_v1.json")
EXP3165_REL_PATH = Path("results/experiment_3165_live_sota_authenticity_replay_v2.json")

SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
CONTROL_NAMES = ("force", "remove", "shuffled_trace", "answer_only", "trace_only")
REQUIRED_FIELDS = {
    "verifier_invariance_token_suspicion_audit_ready",
    "controlled_invariance_checks",
    "computed_checks",
    "blocked_checks",
    "token_suspicion_fields",
    "acceptance_authority_fields",
    "diagnostics_allowed_to_gate_repair",
    "diagnostics_not_allowed_to_accept",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3166_verifier_invariance_token_suspicion_audit_v1.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3166_verifier_invariance_token_suspicion_audit_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/verifier_invariance_token_suspicion_audit_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    role: str
    required: bool
    source_type: str = "json"


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("agents_repo_instructions", Path("AGENTS.md"), "repo_instructions", True, "md"),
    SourceSpec("codex_repo_workflow", Path("CODEX.md"), "codex_workflow", True, "md"),
    SourceSpec("claude_authenticity_rules", Path("CLAUDE.md"), "authenticity_rules", True, "md"),
    SourceSpec(
        "research_references", Path("research-references.md"), "research_context", True, "md"
    ),
    SourceSpec(
        "verification_openspec",
        Path("openspec/capabilities/verification/spec.md"),
        "verification_spec",
        True,
        "md",
    ),
    SourceSpec("exp3136", EXP3136_REL_PATH, "false_accept_autopsy_exact_rows", True),
    SourceSpec("exp3137", EXP3137_REL_PATH, "exact_safe_contract_replay", True),
    SourceSpec("exp3138", EXP3138_REL_PATH, "canonical_grounding_replay", True),
    SourceSpec("exp3150", EXP3150_REL_PATH, "adversarial_corrigendum_policy", True),
    SourceSpec("exp3151", EXP3151_REL_PATH, "live_authenticity_preflight_evidence", True),
    SourceSpec("exp3165", EXP3165_REL_PATH, "authenticity_replay_transcript_hashes", True),
    SourceSpec(
        "exp3166_module",
        Path("python/carnot/verify/verifier_invariance_token_suspicion_audit_v1.py"),
        "audit_module",
        False,
        "py",
    ),
    SourceSpec(
        "exp3166_script",
        Path("scripts/experiment_3166_verifier_invariance_token_suspicion_audit_v1.py"),
        "audit_script",
        False,
        "py",
    ),
    SourceSpec(
        "exp3166_tests",
        Path("tests/python/test_experiment_3166_verifier_invariance_token_suspicion_audit_v1.py"),
        "audit_tests",
        False,
        "py",
    ),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3166: build the audit from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = {
        "exp3136": read_json_object(root_path / EXP3136_REL_PATH),
        "exp3137": read_json_object(root_path / EXP3137_REL_PATH),
        "exp3138": read_json_object(root_path / EXP3138_REL_PATH),
        "exp3150": read_json_object(root_path / EXP3150_REL_PATH),
        "exp3151": read_json_object(root_path / EXP3151_REL_PATH),
        "exp3165": read_json_object(root_path / EXP3165_REL_PATH),
    }
    sources = source_artifacts(root_path)
    exact_rows = collect_trusted_exact_rows(
        payloads["exp3136"], payloads["exp3137"], payloads["exp3138"]
    )
    transcript_rows = collect_exp3165_transcript_hashes(payloads["exp3165"])
    authority_fields = acceptance_authority_fields()
    token_fields = token_suspicion_fields(payloads, exact_rows, transcript_rows)
    controls = controlled_invariance_checks(exact_rows, transcript_rows)
    computed = computed_checks(payloads, exact_rows, transcript_rows, token_fields)
    blocked = blocked_checks(payloads, transcript_rows, token_fields)
    downstream_policy = downstream_policy_for_exp3167(authority_fields, token_fields)
    source_problems = source_errors(sources)
    ready = bool(
        not source_problems
        and exact_rows
        and {row["name"] for row in controls} == set(CONTROL_NAMES)
        and blocked
        and all(field.get("acceptance_authority") is False for field in token_fields)
        and all(field.get("source_kind") == "exact_authority" for field in authority_fields)
        and downstream_policy["acceptance_requires_exact_authority"] is True
        and downstream_policy["token_suspicion_may_accept"] is False
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": duration(start, time.perf_counter() if now_s is None else float(now_s)),
        "verifier_invariance_token_suspicion_audit_ready": ready,
        "controlled_invariance_checks": controls,
        "computed_checks": computed,
        "blocked_checks": blocked,
        "token_suspicion_fields": token_fields,
        "acceptance_authority_fields": authority_fields,
        "diagnostics_allowed_to_gate_repair": diagnostics_allowed_to_gate_repair(),
        "diagnostics_not_allowed_to_accept": diagnostics_not_allowed_to_accept(),
        "trusted_exact_rows": exact_rows,
        "exp3165_transcript_hash_inventory": transcript_rows,
        "downstream_policy_for_exp3167": downstream_policy,
        "source_artifacts": sources,
        "source_errors": source_problems,
        "source_checksums": {row["path"]: row["sha256"] for row in sources if row.get("sha256")},
        "inference_substrate": inference_substrate(),
        "field_principles": field_principles(),
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
    """Build, validate, and persist the Exp 3166 terminal artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while treating absent or malformed files as blockers."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return every local file the audit consumes or cites."""

    rows: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        payload = read_json_object(path) if spec.source_type == "json" else {}
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "role": spec.role,
                "required": spec.required,
                "source_type": spec.source_type,
                "present": path.is_file(),
                "readable_json_object": bool(payload) if spec.source_type == "json" else None,
                "sha256": sha256_file(path),
            }
        )
    return rows


def source_errors(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose missing required evidence instead of inferring around it."""

    errors: list[JsonDict] = []
    for row in sources:
        if row.get("required") is not True:
            continue
        if row.get("present") is not True:
            errors.append(
                {
                    "experiment_id": str(row.get("experiment_id") or ""),
                    "path": str(row.get("path") or ""),
                    "reason": "missing_required_source",
                }
            )
        elif row.get("source_type") == "json" and row.get("readable_json_object") is not True:
            errors.append(
                {
                    "experiment_id": str(row.get("experiment_id") or ""),
                    "path": str(row.get("path") or ""),
                    "reason": "malformed_required_json",
                }
            )
    return errors


def sha256_file(path: Path) -> str | None:
    """Return a checksum so downstream audits can trace source evidence."""

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


def collect_trusted_exact_rows(
    exp3136: Mapping[str, Any],
    exp3137: Mapping[str, Any],
    exp3138: Mapping[str, Any],
) -> list[JsonDict]:
    """Collect compact exact-row evidence from the three trusted artifacts."""

    by_id: dict[str, JsonDict] = {}
    for source, rows in (
        ("exp3136", _mapping_list(exp3136.get("false_accept_rows"))),
        ("exp3136", _mapping_list(exp3136.get("verifier_rows"))),
        ("exp3137", _mapping_list(exp3137.get("replay_rows"))),
        ("exp3138", _mapping_list(exp3138.get("regression_row_replay"))),
    ):
        for row in rows:
            row_id = str(row.get("row_id") or "")
            exact_label = str(row.get("exact_label") or "")
            if not row_id or not exact_label:
                continue
            target = by_id.setdefault(
                row_id,
                {
                    "row_id": row_id,
                    "exact_label": exact_label,
                    "expected_action": str(row.get("expected_action") or ""),
                    "source_experiments": [],
                    "candidate_answers": [],
                    "prompt_hashes": [],
                    "raw_output_hashes": [],
                    "answer_extraction_formats": [],
                    "matched_rule_ids": [],
                    "trusted_exact_authority": True,
                    "acceptance_authority": True,
                },
            )
            if exact_label and target["exact_label"] != exact_label:
                target["exact_label_conflict"] = True
            append_unique(target["source_experiments"], source)
            append_unique(target["candidate_answers"], row.get("extracted_answer"))
            append_unique(target["candidate_answers"], row.get("candidate_answer"))
            append_unique(target["candidate_answers"], row.get("live_model_verdict"))
            append_unique(target["prompt_hashes"], row.get("prompt_hash"))
            append_unique(target["prompt_hashes"], row.get("source_prompt_payload_sha256"))
            append_unique(target["raw_output_hashes"], row.get("raw_output_hash"))
            append_unique(
                target["raw_output_hashes"],
                _mapping(row.get("prior_panel_row")).get("raw_output_hash"),
            )
            append_unique(target["answer_extraction_formats"], row.get("answer_extraction_format"))
            append_unique(target["matched_rule_ids"], row.get("matched_rule_id"))
            append_unique(
                target["matched_rule_ids"],
                _mapping(row.get("contract_replay")).get("matched_rule_id"),
            )
            for value in monitor_payload_values(row, "raw_output_hash"):
                append_unique(target["raw_output_hashes"], value)
    normalized: list[JsonDict] = []
    for row_id in sorted(by_id):
        row = by_id[row_id]
        for key in (
            "source_experiments",
            "candidate_answers",
            "prompt_hashes",
            "raw_output_hashes",
            "answer_extraction_formats",
            "matched_rule_ids",
        ):
            row[key] = sorted(row[key])
        row["row_fingerprint"] = stable_hash(
            {
                "row_id": row["row_id"],
                "exact_label": row["exact_label"],
                "sources": row["source_experiments"],
                "candidate_answers": row["candidate_answers"],
            }
        )
        normalized.append(row)
    return normalized


def collect_exp3165_transcript_hashes(exp3165: Mapping[str, Any]) -> list[JsonDict]:
    """Collect transcript hashes without treating them as acceptance evidence."""

    rows: list[JsonDict] = []
    for index, row in enumerate(_mapping_list(exp3165.get("transcript_hashes"))):
        transcript_hash = str(row.get("transcript_sha256") or "")
        if not transcript_hash:
            continue
        rows.append(
            {
                "source_experiment": "exp3165",
                "index": index,
                "transcript_sha256": transcript_hash,
                "prompt_hash": str(row.get("prompt_hash") or ""),
                "response_hash": str(row.get("response_hash") or ""),
                "model_id": str(row.get("model_id") or row.get("selected_model_id") or ""),
                "token_counts": _mapping(row.get("token_counts")),
                "acceptance_authority": False,
            }
        )
    return rows


def controlled_invariance_checks(
    exact_rows: Sequence[Mapping[str, Any]],
    transcript_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Define the five control families as exact-routing checks."""

    exact_count = len(exact_rows)
    transcript_count = len(transcript_rows)
    templates = {
        "force": "Force the candidate answer token to an exact or counterfactual label and require exact replay to decide.",
        "remove": "Remove final-answer artifacts and require abstain/repair routing rather than answer-artifact acceptance.",
        "shuffled_trace": "Shuffle available trace or monitor-event order and require exact labels to remain the gate.",
        "answer_only": "Strip trace context while keeping the answer; this may triage shortcut risk but cannot accept.",
        "trace_only": "Strip the final answer while keeping trace evidence; this may route repair but cannot accept.",
    }
    return [
        {
            "name": name,
            "description": description,
            "principle": "detector shortcuts must be tested",
            "available_exact_row_count": exact_count,
            "available_exp3165_transcript_hash_count": transcript_count,
            "can_define_from_existing_artifacts": exact_count > 0,
            "fully_computable_without_future_rerun": False,
            "routes_to_exact_checks": True,
            "can_authorize_acceptance": False,
            "exp3167_required_action": "run transformed rows through exact-safe/canonical/monitor authority before any acceptance claim",
        }
        for name, description in templates.items()
    ]


def computed_checks(
    payloads: Mapping[str, Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    transcript_rows: Sequence[Mapping[str, Any]],
    token_fields: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """List evidence that was actually computed from checked-in artifacts."""

    exp3137 = payloads["exp3137"]
    exp3138 = payloads["exp3138"]
    return [
        {
            "name": "trusted_exact_row_inventory",
            "principle": "distinguish computed evidence from proposed checks",
            "computed_from_existing_artifacts": True,
            "source_experiments": ["exp3136", "exp3137", "exp3138"],
            "row_count": len(exact_rows),
            "row_ids": [str(row["row_id"]) for row in exact_rows],
            "acceptance_authority": True,
        },
        {
            "name": "exact_safe_replay_gate_inventory",
            "principle": "exact authority must stay separate from suspicion signals",
            "computed_from_existing_artifacts": True,
            "known_false_accept_rows_blocked": bool(exp3137.get("known_false_accept_rows_blocked")),
            "replay_false_accept_rate": exp3137.get("replay_false_accept_rate"),
            "replay_false_reject_rate": exp3137.get("replay_false_reject_rate"),
            "replay_abstention_rate": exp3137.get("replay_abstention_rate"),
        },
        {
            "name": "canonical_grounding_inventory",
            "principle": "exact authority must stay separate from suspicion signals",
            "computed_from_existing_artifacts": True,
            "false_accept_rows_blocked": exp3138.get("false_accept_rows_blocked"),
            "residual_false_accept_rows": list(exp3138.get("residual_false_accept_rows") or []),
            "regression_rows_evaluated": exp3138.get("regression_rows_evaluated"),
        },
        {
            "name": "exp3165_transcript_hash_inventory",
            "principle": "audit must trace to exact rows and preflight evidence",
            "computed_from_existing_artifacts": True,
            "transcript_hash_count": len(transcript_rows),
            "transcript_hashes": [str(row["transcript_sha256"]) for row in transcript_rows],
            "prompt_hashes": [
                str(row["prompt_hash"]) for row in transcript_rows if row.get("prompt_hash")
            ],
            "acceptance_authority": False,
        },
        {
            "name": "token_suspicion_field_availability",
            "principle": "triage signals must be explicitly scoped",
            "computed_from_existing_artifacts": True,
            "available_fields": [
                str(row["name"])
                for row in token_fields
                if row.get("available_in_existing_artifacts")
            ],
            "blocked_fields": [
                str(row["name"])
                for row in token_fields
                if not row.get("available_in_existing_artifacts")
            ],
            "acceptance_authority": False,
        },
    ]


def blocked_checks(
    payloads: Mapping[str, Mapping[str, Any]],
    transcript_rows: Sequence[Mapping[str, Any]],
    token_fields: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Keep missing future telemetry visible to downstream tasks."""

    del payloads
    rows: list[JsonDict] = [
        {
            "name": "future_first_token_logprob_telemetry",
            "principle": "missing token/logprob telemetry must stay visible",
            "blocked_reason": "first-token logprob/entropy telemetry is absent or null in the checked-in exact rows",
            "requires_future_live_logprob_or_token_telemetry": True,
            "acceptance_authority": False,
        },
        {
            "name": "future_token_level_logprob_curve",
            "principle": "missing token/logprob telemetry must stay visible",
            "blocked_reason": "full token-level logprob curves are not present in Exp 3165 transcripts",
            "requires_future_live_logprob_or_token_telemetry": True,
            "acceptance_authority": False,
        },
    ]
    if not transcript_rows:
        rows.append(
            {
                "name": "exp3165_transcript_level_controls",
                "principle": "distinguish computed evidence from proposed checks",
                "blocked_reason": "Exp 3165 contains no replay transcript hashes, so transcript-level shuffled/answer-only/trace-only controls are defined but not executable from Exp 3165.",
                "requires_future_live_logprob_or_token_telemetry": False,
                "requires_future_transcript_text": True,
                "acceptance_authority": False,
            }
        )
    for field in token_fields:
        if field.get("available_in_existing_artifacts") is False:
            rows.append(
                {
                    "name": f"missing_{field['name']}",
                    "principle": "missing token/logprob telemetry must stay visible",
                    "blocked_reason": str(field.get("missing_reason") or "field unavailable"),
                    "requires_future_live_logprob_or_token_telemetry": bool(
                        field.get("requires_future_live_logprob_or_token_telemetry")
                    ),
                    "acceptance_authority": False,
                }
            )
    return rows


def token_suspicion_fields(
    payloads: Mapping[str, Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    transcript_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Define triage fields while explicitly denying acceptance authority."""

    exp3151_counts = _mapping(payloads["exp3151"].get("token_counts"))
    exp3165_counts = _mapping(payloads["exp3165"].get("token_counts"))
    token_count_available = (
        int(exp3151_counts.get("total_tokens") or 0) > 0
        or int(exp3165_counts.get("total_tokens") or 0) > 0
    )
    response_hash_available = any(row.get("response_hash") for row in transcript_rows) or any(
        row.get("raw_output_hashes") for row in exact_rows
    )
    answer_family_available = any(row.get("answer_extraction_formats") for row in exact_rows)
    return [
        triage_field(
            "prompt_token_count",
            token_count_available,
            "existing Exp 3151/3165 token count aggregates",
            "route unusually tiny or huge prompts to exact replay inspection",
        ),
        triage_field(
            "completion_token_count",
            token_count_available,
            "existing Exp 3151/3165 token count aggregates",
            "route empty or implausibly short completions to exact replay inspection",
        ),
        triage_field(
            "first_token_logprob",
            False,
            "future live logprob telemetry",
            "prioritize low-confidence rows for exact checks",
            missing_reason="first-token logprob is absent from existing Exp 3165 evidence",
            future_logprob=True,
        ),
        triage_field(
            "first_token_entropy",
            False,
            "future live first-token entropy telemetry",
            "prioritize high-entropy first-token rows for exact checks",
            missing_reason="first-token entropy is absent or null in existing rows",
            future_logprob=True,
        ),
        triage_field(
            "token_level_logprob_curve",
            False,
            "future live per-token telemetry",
            "prioritize local token-suspicion spans for exact checks",
            missing_reason="token-level logprob curves were not logged",
            future_logprob=True,
        ),
        triage_field(
            "response_hash_reuse",
            response_hash_available,
            "existing raw response or transcript hash evidence",
            "route reused or stale-looking responses to exact replay inspection",
        ),
        triage_field(
            "answer_token_family_mismatch",
            answer_family_available,
            "existing exact row answer-extraction format evidence",
            "route SAT/validity/repairability family mismatches to exact checks",
        ),
        triage_field(
            "trace_presence_or_length",
            bool(exact_rows),
            "existing monitor-event and premise-grounding row evidence",
            "route trace-only or missing-trace rows to canonical/monitor checks",
        ),
    ]


def triage_field(
    name: str,
    available: bool,
    source: str,
    allowed_use: str,
    *,
    missing_reason: str = "",
    future_logprob: bool = False,
) -> JsonDict:
    """Create one token-suspicion descriptor with the policy boundary repeated."""

    return {
        "name": name,
        "principle": "triage signals must be explicitly scoped",
        "available_in_existing_artifacts": bool(available),
        "source": source,
        "allowed_use": allowed_use,
        "acceptance_authority": False,
        "may_route_exact_checks": True,
        "may_accept": False,
        "missing_reason": "" if available else missing_reason,
        "requires_future_live_logprob_or_token_telemetry": bool(future_logprob),
    }


def acceptance_authority_fields() -> list[JsonDict]:
    """Return fields allowed to accept only when exact authority agrees."""

    names = (
        ("exact_label", "trusted exact solver/test/canonical label for the row"),
        ("exact_safe_replay_decision", "Exp 3137 accept/abstain/reject decision"),
        ("canonical_equivalence", "Exp 3138 canonical answer equivalence"),
        ("solver_or_test_authority", "Z3/Python/runtime exact authority label source"),
        ("monitor_ledger_consistency", "monitor ledger and final answer consistency"),
    )
    return [
        {
            "name": name,
            "description": description,
            "principle": "exact authority must stay separate from suspicion signals",
            "source_kind": "exact_authority",
            "acceptance_authority": True,
            "requires_exact_check": True,
        }
        for name, description in names
    ]


def diagnostics_allowed_to_gate_repair() -> list[JsonDict]:
    """Conservative fields that may block or route repair promotion."""

    return [
        {
            "name": "known_false_accept_rows_blocked",
            "principle": "repair gates need conservative fields",
            "source": "exp3137/exp3138",
            "allowed_gate_use": "required true before repair promotion",
            "can_authorize_acceptance": False,
        },
        {
            "name": "residual_false_accept_rows",
            "principle": "repair gates need conservative fields",
            "source": "exp3138",
            "allowed_gate_use": "must be empty before repair promotion",
            "can_authorize_acceptance": False,
        },
        {
            "name": "controlled_invariance_failures",
            "principle": "repair gates need conservative fields",
            "source": "exp3166/exp3167",
            "allowed_gate_use": "may block repair or require exact reroute when a detector shortcut is exposed",
            "can_authorize_acceptance": False,
        },
        {
            "name": "authenticity_transcript_hash_presence",
            "principle": "repair gates need conservative fields",
            "source": "exp3165",
            "allowed_gate_use": "may block live rerun promotion when required replay evidence is absent",
            "can_authorize_acceptance": False,
        },
    ]


def diagnostics_not_allowed_to_accept() -> list[JsonDict]:
    """Suspicion and trace diagnostics that must never become accept gates."""

    return [
        {
            "name": "token_suspicion_fields",
            "principle": "token/trace signals must not become acceptance authority",
            "reason": "token counts, logprobs, entropy, and hash reuse can triage only",
        },
        {
            "name": "answer_only_control_score",
            "principle": "token/trace signals must not become acceptance authority",
            "reason": "answer-only survival can reveal shortcuts but exact labels still decide",
        },
        {
            "name": "trace_only_control_score",
            "principle": "token/trace signals must not become acceptance authority",
            "reason": "trace-only evidence can route repair but cannot prove final-answer correctness",
        },
        {
            "name": "shuffled_trace_stability",
            "principle": "token/trace signals must not become acceptance authority",
            "reason": "invariance under shuffled traces is a diagnostic, not proof of exact correctness",
        },
    ]


def downstream_policy_for_exp3167(
    authority_fields: Sequence[Mapping[str, Any]],
    token_fields: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Expose the exact-only policy the clean rerun must consume."""

    return {
        "target_experiment": "exp3167",
        "acceptance_requires_exact_authority": True,
        "token_suspicion_may_accept": False,
        "trace_or_control_diagnostics_may_accept": False,
        "controlled_invariance_checks_required": list(CONTROL_NAMES),
        "required_acceptance_authority_fields": [str(row["name"]) for row in authority_fields],
        "triage_only_suspicion_fields": [str(row["name"]) for row in token_fields],
        "reroute_policy": "Any suspicious token, answer-artifact, or trace-artifact signal routes to exact-safe replay, canonical grounding, and monitor-ledger checks.",
        "repair_gate_policy": "Repair promotion may be blocked by missing or failed diagnostics, but acceptance remains exact-authority-only.",
    }


def inference_substrate() -> JsonDict:
    """Declare that this audit performs aggregation only."""

    return {
        "kind": "artifact_only_verifier_invariance_token_suspicion_audit",
        "uses_checked_in_artifacts_only": True,
        "declares_no_new_live_model_inference": True,
        "no_new_live_model_inference": True,
        "new_live_model_call_count": 0,
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "downloads_models": False,
        "modifies_research_conductor": False,
    }


def field_principles() -> JsonDict:
    """Mirror the task principles beside their machine-readable fields."""

    return {
        "verifier_invariance_token_suspicion_audit_ready": "rerun evidence needs artifact-sensitivity checks",
        "controlled_invariance_checks": "detector shortcuts must be tested",
        "computed_checks": "distinguish computed evidence from proposed checks",
        "blocked_checks": "missing token/logprob telemetry must stay visible",
        "token_suspicion_fields": "triage signals must be explicitly scoped",
        "acceptance_authority_fields": "exact authority must stay separate from suspicion signals",
        "diagnostics_allowed_to_gate_repair": "repair gates need conservative fields",
        "diagnostics_not_allowed_to_accept": "token/trace signals must not become acceptance authority",
        "source_artifacts": "audit must trace to exact rows and preflight evidence",
        "inference_substrate": "aggregation work must declare no new live model inference",
        "honest_verdict": "terminal verdict must be honest about preconditions",
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict with success only for a complete audit."""

    if artifact.get("verifier_invariance_token_suspicion_audit_ready") is True:
        row_count = len(_mapping_list(artifact.get("trusted_exact_rows")))
        blocked_count = len(_mapping_list(artifact.get("blocked_checks")))
        return (
            "complete: verifier_invariance_token_suspicion_audit_ready=true; "
            f"trusted_exact_rows={row_count}; blocked_future_checks={blocked_count}; "
            "exp3167_policy=exact_authority_only"
        )
    errors = _mapping_list(artifact.get("source_errors"))
    if errors:
        return f"blocked_missing_source: verifier_invariance_token_suspicion_audit_ready=false; errors={len(errors)}"
    return "blocked_precondition: verifier_invariance_token_suspicion_audit_ready=false"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail fast when the audit accidentally promotes suspicion to authority."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3166 artifact missing required fields: {missing}")
    controls = _mapping_list(artifact.get("controlled_invariance_checks"))
    if {str(row.get("name")) for row in controls} != set(CONTROL_NAMES):
        raise ValueError("Exp 3166 artifact must define all five controlled-invariance checks")
    for row in controls:
        if (
            row.get("routes_to_exact_checks") is not True
            or row.get("can_authorize_acceptance") is not False
        ):
            raise ValueError(
                "controlled-invariance checks must route exact checks and cannot accept"
            )
    if any(
        row.get("acceptance_authority") is not False
        for row in _mapping_list(artifact.get("token_suspicion_fields"))
    ):
        raise ValueError("token suspicion fields must not be acceptance authority")
    if any(
        row.get("source_kind") != "exact_authority"
        for row in _mapping_list(artifact.get("acceptance_authority_fields"))
    ):
        raise ValueError("acceptance authority fields must remain exact authority only")
    substrate = _mapping(artifact.get("inference_substrate"))
    if (
        substrate.get("no_new_live_model_inference") is not True
        or substrate.get("executes_models") is not False
    ):
        raise ValueError("Exp 3166 must declare no new live model inference")
    policy = _mapping(artifact.get("downstream_policy_for_exp3167"))
    if (
        policy.get("acceptance_requires_exact_authority") is not True
        or policy.get("token_suspicion_may_accept") is not False
    ):
        raise ValueError("Exp 3167 downstream policy must keep acceptance exact-only")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("verifier_invariance_token_suspicion_audit_ready") is True:
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("ready Exp 3166 artifact must use a terminal success prefix")
    elif not verdict.startswith("blocked_"):
        raise ValueError("blocked Exp 3166 artifact must use a blocked_ verdict")


def append_unique(target: list[str], value: Any) -> None:
    """Append nonempty strings once, keeping summaries compact."""

    if value is None:
        return
    text = str(value)
    if text and text not in target:
        target.append(text)


def monitor_payload_values(row: Mapping[str, Any], key: str) -> list[str]:
    """Extract compact evidence values from monitor-event payloads."""

    values: list[str] = []
    for event in _mapping_list(row.get("monitor_events")):
        payload = _mapping(event.get("payload"))
        value = payload.get(key)
        if value:
            values.append(str(value))
    return values


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _string_values(value: Any) -> list[str]:
    return sorted({str(item) for item in value if isinstance(item, str)})
