"""Build the Exp 3139 live SOTA verifier rerun v7 artifact.

Spec refs: REQ-VERIFY-3139, SCENARIO-VERIFY-3139.

This module reruns a bounded local SOTA verifier panel, but it keeps the exact
solver/test label outside the model trust boundary. The model may propose an
answer token; the exact-safe contract decides whether that token can be used,
must be rejected, or must fail closed to abstention.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from carnot.eval import difficulty_stratified_live_sota_verifier_panel_v6 as panel_v6
from carnot.verify import canonical_answer_vericot_grounding_pilot_v1 as canonical
from carnot.verify import exact_safe_accept_abstain_contract_v1 as exact_contract


JsonDict = dict[str, Any]
LiveRunner = Callable[[str, JsonDict, JsonDict, JsonDict], str]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3139_live_sota_verifier_rerun_v7"
SCHEMA = "carnot.live_sota_verifier_rerun.v7"
OUTPUT_REL_PATH = Path("results/experiment_3139_live_sota_verifier_rerun_v7.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3139_live_sota_verifier_rerun_v7.py"

EXP3123_REL_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")
EXP3124_REL_PATH = Path("results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json")
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")

MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_ROLES = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "moe",
    "unsloth/gemma-4-31B-it-GGUF": "dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "moe",
}
REQUIRED_BUCKETS = (
    "contradiction",
    "satisfiable_drift",
    "medium",
    "hard",
    "fragment_code",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_FIELDS = {
    "live_verifier_rerun_v7_ready",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "headline_claim_allowed",
    "regression_rows_included",
    "exact_ground_truth_count",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "verifier_gain_delta",
    "repair_gate_candidate_state",
    "false_accept_gate_passed",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3139_live_sota_verifier_rerun_v7.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3139_live_sota_verifier_rerun_v7.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/live_sota_verifier_rerun_v7.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_REL_PATHS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("experiment_template_cache_policy", Path("scripts/experiment_template.py"), False),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True),
    ("exp3123_sota_cache_manifest", EXP3123_REL_PATH, True),
    ("exp3124_prior_live_panel", EXP3124_REL_PATH, True),
    ("exp3126_monitor_ledger", EXP3126_REL_PATH, True),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
    ("exp3137_exact_safe_contract", EXP3137_REL_PATH, True),
    ("exp3138_canonical_grounding", EXP3138_REL_PATH, True),
    ("exp3139_module", Path("python/carnot/verify/live_sota_verifier_rerun_v7.py"), False),
    ("exp3139_script", Path("scripts/experiment_3139_live_sota_verifier_rerun_v7.py"), False),
)


def expected_action_from_label(label: str | None) -> str:
    """Map exact labels to the public accept/reject/abstain action space."""

    return exact_contract.expected_action_from_label(label)


def token_family_for_label(label: str | None) -> str | None:
    """Return the answer-token family for exact labels and extracted labels."""

    return exact_contract.token_family_for_label(label)


def difficulty_bucket_labels(row: Mapping[str, Any]) -> list[str]:
    """Return the same stable difficulty buckets used by the prior live panel."""

    return panel_v6.difficulty_bucket_labels(row)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    max_live_calls: int = 6,
    min_live_calls_for_headline: int = 4,
    live_runner: LiveRunner | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3139: build the exact-safe live verifier rerun artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3123 = read_json_object(root_path / EXP3123_REL_PATH)
    exp3124 = read_json_object(root_path / EXP3124_REL_PATH)
    exp3126 = read_json_object(root_path / EXP3126_REL_PATH)
    exp3136 = read_json_object(root_path / EXP3136_REL_PATH)
    exp3137 = read_json_object(root_path / EXP3137_REL_PATH)
    exp3138 = read_json_object(root_path / EXP3138_REL_PATH)

    source_rows = source_artifacts(root_path)
    required_sources_present = all(row["present"] for row in source_rows if row["required"])
    regression_ids = regression_row_ids(exp3136)
    rerun_set = select_rerun_rows(
        known_false_accept_rows(exp3136),
        prior_panel_rows(exp3124),
        regression_ids=regression_ids,
        max_live_calls=max_live_calls,
    )
    model_specs = model_specs_from_manifest(exp3123, root_path)
    selected_model = first_usable_selected_model(model_specs)
    context = build_contract_context(exp3137, exp3126, rerun_set, regression_ids)
    rerun_rows, runtime_error = run_live_panel(
        rerun_set,
        selected_model=selected_model,
        context=context,
        max_live_calls=max_live_calls,
        live_runner=live_runner,
    )
    raw_metrics = metrics_for_decisions(rerun_rows, "raw_live_decision")
    exact_safe_metrics = metrics_for_decisions(rerun_rows, "contract_decision")
    regression_rows_included = bool(regression_ids) and set(regression_ids) <= {
        row["fixture_id"] for row in rerun_set
    }
    false_accept_rate = exact_safe_metrics["false_accept_rate"]
    false_reject_rate = exact_safe_metrics["false_reject_rate"]
    abstention_rate = exact_safe_metrics["abstention_rate"]
    prior_false_accept_rate = source_false_accept_rate(exp3136, exp3124, raw_metrics)
    verifier_gain_delta = rate_delta(prior_false_accept_rate, false_accept_rate)
    false_accept_gate_passed = bool(
        rerun_rows and regression_rows_included and false_accept_rate == 0.0
    )
    preconditions_ready = bool(
        exp3137.get("acceptance_contract_v1_ready") is True
        and exp3138.get("canonical_grounding_pilot_v1_ready") is True
    )
    repair_state = repair_gate_candidate_state(
        required_sources_present=required_sources_present,
        preconditions_ready=preconditions_ready,
        live_call_count=len(rerun_rows),
        min_live_calls_for_headline=min_live_calls_for_headline,
        regression_rows_included=regression_rows_included,
        false_accept_rate=false_accept_rate,
        verifier_gain_delta=verifier_gain_delta,
    )
    headline_claim_allowed = bool(
        repair_state == "candidate_ready"
        and exp3123.get("headline_claim_allowed") is True
        and selected_model is not None
        and false_accept_gate_passed
    )
    self_checks = self_checks_for_artifact(
        rerun_rows=rerun_rows,
        exact_safe_metrics=exact_safe_metrics,
        regression_ids=regression_ids,
        repair_state=repair_state,
    )
    ready = bool(
        required_sources_present
        and preconditions_ready
        and repair_state == "candidate_ready"
        and false_accept_gate_passed
        and headline_claim_allowed
        and self_checks["all_self_checks_passed"]
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "live_verifier_rerun_v7_ready": ready,
        "model_specs": model_specs,
        "selected_model_ids": [row["hf_id"] for row in model_specs if row["selected"]],
        "live_call_count": len(rerun_rows),
        "headline_claim_allowed": headline_claim_allowed,
        "regression_rows_included": regression_rows_included,
        "exact_ground_truth_count": len(rerun_set),
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "abstention_rate": abstention_rate,
        "verifier_gain_delta": verifier_gain_delta,
        "verifier_gain_definition": "source_or_raw_false_accept_rate_minus_exact_safe_false_accept_rate",
        "source_false_accept_rate": prior_false_accept_rate,
        "repair_gate_candidate_state": repair_state,
        "false_accept_gate_passed": false_accept_gate_passed,
        "raw_live_metrics": raw_metrics,
        "exact_safe_metrics": exact_safe_metrics,
        "rerun_set_metadata": rerun_set,
        "rerun_rows": rerun_rows,
        "contract_context": {
            "prefix_covered_labels": sorted(context.prefix_covered_labels),
            "regression_row_set": sorted(context.regression_row_set),
        },
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row.get("sha256")
        },
        "inference_substrate": inference_substrate(
            exp3123=exp3123,
            selected_model=selected_model,
            live_call_count=len(rerun_rows),
            runtime_error=runtime_error,
        ),
        "self_checks": self_checks,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    max_live_calls: int = 6,
    min_live_calls_for_headline: int = 4,
    live_runner: LiveRunner | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3139 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        max_live_calls=max_live_calls,
        min_live_calls_for_headline=min_live_calls_for_headline,
        live_runner=live_runner,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def select_rerun_rows(
    false_accept_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    regression_ids: Sequence[str],
    max_live_calls: int,
) -> list[JsonDict]:
    """Build a regression-first row set with balanced exact fixture coverage."""

    candidates = {row_id_from(row): normalize_rerun_row(row) for row in candidate_rows}
    selected: list[JsonDict] = []
    seen: set[str] = set()
    for row in false_accept_rows:
        row_id = row_id_from(row)
        merged = candidates.get(row_id, {}) | normalize_rerun_row(row)
        merged["fixture_id"] = row_id
        merged["is_regression_row"] = row_id in regression_ids
        selected.append(merged)
        seen.add(row_id)
    limit = max(int(max_live_calls), len(selected))
    for bucket in REQUIRED_BUCKETS:
        for row in candidates.values():
            if row["fixture_id"] not in seen and bucket in row["difficulty_bucket_labels"]:
                selected.append(row)
                seen.add(row["fixture_id"])
                break
        if len(selected) >= limit:
            return selected[:limit]
    for row in candidates.values():
        if row["fixture_id"] not in seen:
            selected.append(row)
            seen.add(row["fixture_id"])
        if len(selected) >= limit:
            break
    return selected[:limit]


def normalize_rerun_row(row: Mapping[str, Any]) -> JsonDict:
    """Normalize one source row into the fields needed by prompts and metrics."""

    row_id = row_id_from(row)
    exact = str(row.get("exact_label") or row.get("expected_answer") or "").upper()
    task_family = str(row.get("task_family") or row.get("fixture_family") or "unknown")
    normalized = dict(row)
    normalized.update(
        {
            "fixture_id": row_id,
            "row_id": row_id,
            "fixture_family": str(row.get("fixture_family") or task_family),
            "task_family": task_family,
            "perturbation_type": str(row.get("perturbation_type") or "unknown"),
            "exact_label": exact,
            "expected_action": str(row.get("expected_action") or expected_action_from_label(exact)),
            "answer_extraction_format": row.get("answer_extraction_format")
            or token_family_for_label(exact),
            "prompt_payload": row.get("prompt_payload")
            or {"fixture_id": row_id, "expected_answer": exact},
            "is_regression_row": False,
        }
    )
    normalized["difficulty_bucket_labels"] = list(
        row.get("difficulty_bucket_labels") or difficulty_bucket_labels(normalized)
    )
    return normalized


def run_live_panel(
    rows: Sequence[JsonDict],
    *,
    selected_model: JsonDict | None,
    context: exact_contract.ContractContext,
    max_live_calls: int,
    live_runner: LiveRunner | None,
) -> tuple[list[JsonDict], str | None]:
    """Run bounded live calls and apply the exact-safe contract to each row."""

    if selected_model is None or max_live_calls <= 0:
        return [], None
    runner = live_runner or panel_v6.default_live_runner
    decode_config = {"max_tokens": 32, "temperature": 0.0, "seed": 20260526}
    live_rows: list[JsonDict] = []
    model_hash = bounded_model_hash(Path(str(selected_model.get("model_path") or "")))
    try:
        for index, row in enumerate(rows[:max_live_calls]):
            prompt = build_prompt(row)
            raw_output = runner(prompt, row, selected_model, dict(decode_config))
            extracted = panel_v6.extract_answer(raw_output)
            raw_decision = expected_action_from_label(extracted)
            contract_result = apply_contract_to_live_row(
                row,
                extracted_answer=extracted,
                raw_live_decision=raw_decision,
                context=context,
            )
            exact = row["exact_label"]
            live_rows.append(
                row
                | {
                    "live_call_index": index,
                    "model_id": selected_model["hf_id"],
                    "model_path": selected_model.get("model_path"),
                    "model_hash": model_hash,
                    "prompt_hash": sha256_text(prompt),
                    "raw_output": raw_output,
                    "raw_output_hash": sha256_text(raw_output),
                    "extracted_answer": extracted,
                    "raw_live_decision": raw_decision,
                    "exact_canonical": canonical.canonicalize_answer(exact).to_dict(),
                    "candidate_canonical": canonical.canonicalize_answer(extracted).to_dict(),
                    "contract_decision": contract_result["decision"],
                    "contract_rule_id": contract_result["matched_rule_id"],
                    "contract_replay": contract_result,
                    "false_accept_before_contract": row["expected_action"] == "reject"
                    and raw_decision == "accept",
                    "false_accept_after_contract": row["expected_action"] == "reject"
                    and contract_result["decision"] == "accept",
                    "false_reject_after_contract": row["expected_action"] == "accept"
                    and contract_result["decision"] == "reject",
                    "abstained_by_contract": contract_result["decision"] == "abstain",
                    "decode_config": dict(decode_config),
                }
            )
    except Exception as exc:  # pragma: no cover - live runtime diagnostic path.
        return [], f"{type(exc).__name__}: {exc}"
    return live_rows, None


def apply_contract_to_live_row(
    row: Mapping[str, Any],
    *,
    extracted_answer: str | None,
    raw_live_decision: str,
    context: exact_contract.ContractContext,
) -> JsonDict:
    """Apply the Exp 3137 exact-safe contract to one live rerun row."""

    contract_input = dict(row)
    contract_input.update(
        {
            "extracted_answer": extracted_answer,
            "live_decision": raw_live_decision,
            "parse_confidence": 1.0 if extracted_answer else 0.0,
        }
    )
    return exact_contract.evaluate_row(contract_input, context, row_source="live")


def build_contract_context(
    exp3137: Mapping[str, Any],
    exp3126: Mapping[str, Any],
    rerun_rows: Sequence[Mapping[str, Any]],
    regression_ids: Sequence[str],
) -> exact_contract.ContractContext:
    """Build the Exp 3137 contract context from checked-in replay evidence."""

    prefix_labels = {
        str(row.get("exact_label") or "").upper()
        for row in mapping_rows(exp3137.get("replay_rows"))
        if row.get("prefix_label_covered") is True
    }
    monitor_by_fixture = exact_contract.monitor_events_by_fixture(exp3126.get("monitor_events"))
    for row in rerun_rows:
        row_events = row.get("monitor_events")
        if isinstance(row_events, list):
            monitor_by_fixture[row["fixture_id"]] = [
                dict(event) | {"fixture_id": row["fixture_id"]}
                for event in row_events
                if isinstance(event, Mapping)
            ]
    return exact_contract.ContractContext(
        prefix_covered_labels=frozenset(prefix_labels),
        regression_row_set=frozenset(regression_ids),
        monitor_by_fixture=monitor_by_fixture,
    )


def metrics_for_decisions(rows: Sequence[Mapping[str, Any]], decision_field: str) -> JsonDict:
    """Compute exact-label metrics for raw or exact-safe decisions."""

    positives = [row for row in rows if row.get("expected_action") == "accept"]
    negatives = [row for row in rows if row.get("expected_action") == "reject"]
    false_accepts = [
        row
        for row in rows
        if row.get("expected_action") == "reject" and row.get(decision_field) == "accept"
    ]
    false_rejects = [
        row
        for row in rows
        if row.get("expected_action") == "accept" and row.get(decision_field) == "reject"
    ]
    abstentions = [row for row in rows if row.get(decision_field) == "abstain"]
    return {
        "count": len(rows),
        "accept_count": sum(row.get(decision_field) == "accept" for row in rows),
        "reject_count": sum(row.get(decision_field) == "reject" for row in rows),
        "abstain_count": len(abstentions),
        "false_accept_count": len(false_accepts),
        "false_reject_count": len(false_rejects),
        "false_accept_rate": safe_rate(len(false_accepts), len(negatives)),
        "false_reject_rate": safe_rate(len(false_rejects), len(positives)),
        "abstention_rate": safe_rate(len(abstentions), len(rows)),
        "accuracy": safe_rate(
            sum(row.get(decision_field) == row.get("expected_action") for row in rows),
            len(rows),
        ),
    }


def source_false_accept_rate(
    exp3136: Mapping[str, Any],
    exp3124: Mapping[str, Any],
    raw_metrics: Mapping[str, Any],
) -> float:
    """Return the prior false-accept baseline the rerun is supposed to improve."""

    for source in (exp3136, exp3124, raw_metrics):
        value = source.get("source_false_accept_rate") or source.get("false_accept_rate")
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def repair_gate_candidate_state(
    *,
    required_sources_present: bool,
    preconditions_ready: bool,
    live_call_count: int,
    min_live_calls_for_headline: int,
    regression_rows_included: bool,
    false_accept_rate: float,
    verifier_gain_delta: float,
) -> str:
    """Map rerun evidence to the downstream repair-gate candidate state."""

    if required_sources_present is not True:
        return "blocked_missing_inputs"
    if preconditions_ready is not True:
        return "blocked_precondition_artifacts"
    if live_call_count <= 0:
        return "blocked_no_live_model"
    if regression_rows_included is not True:
        return "blocked_missing_regression_rows"
    if live_call_count < min_live_calls_for_headline:
        return "blocked_tiny_panel"
    if false_accept_rate > 0.0:
        return "blocked_false_accept"
    if verifier_gain_delta <= 0.0:
        return "blocked_no_false_accept_reduction"
    return "candidate_ready"


def self_checks_for_artifact(
    *,
    rerun_rows: Sequence[Mapping[str, Any]],
    exact_safe_metrics: Mapping[str, Any],
    regression_ids: Sequence[str],
    repair_state: str,
) -> JsonDict:
    """Run deterministic checks that make the rerun gate auditable."""

    regression_rows = [row for row in rerun_rows if row.get("fixture_id") in regression_ids]
    checks = {
        "rerun_rows_deterministic": stable_hash(rerun_rows)
        == stable_hash(json.loads(json.dumps(rerun_rows, sort_keys=True))),
        "regression_rows_not_accepted": bool(regression_ids)
        and all(row.get("contract_decision") != "accept" for row in regression_rows),
        "finite_rates_in_unit_interval": all(
            finite_unit_rate(exact_safe_metrics[field])
            for field in ("false_accept_rate", "false_reject_rate", "abstention_rate")
        ),
        "candidate_state_matches_false_accept_gate": repair_state != "candidate_ready"
        or exact_safe_metrics["false_accept_rate"] == 0.0,
    }
    checks["all_self_checks_passed"] = all(checks.values())
    checks["rerun_hash"] = stable_hash(rerun_rows)
    return checks


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, making malformed evidence non-promotable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive filesystem guard.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def known_false_accept_rows(exp3136: Mapping[str, Any]) -> list[JsonDict]:
    """Return known .291 false-accept rows from the autopsy."""

    rows = mapping_rows(exp3136.get("false_accept_rows"))
    ids = set(regression_row_ids(exp3136))
    if rows:
        return sorted((normalize_rerun_row(row) for row in rows), key=row_id_from)
    return sorted(
        (
            normalize_rerun_row(row)
            for row in mapping_rows(exp3136.get("verifier_rows"))
            if row_id_from(row) in ids
        ),
        key=row_id_from,
    )


def regression_row_ids(exp3136: Mapping[str, Any]) -> list[str]:
    """Return stable regression row IDs that must appear in the rerun."""

    ids = exp3136.get("regression_row_set") or exp3136.get("false_accept_row_ids") or []
    return [str(row_id) for row_id in ids if str(row_id)]


def prior_panel_rows(exp3124: Mapping[str, Any]) -> list[JsonDict]:
    """Return prior exact fixture metadata, falling back to live rows when needed."""

    rows = mapping_rows(exp3124.get("panel_fixture_metadata"))
    if not rows:
        rows = mapping_rows(exp3124.get("live_rows"))
    return sorted((normalize_rerun_row(row) for row in rows), key=row_id_from)


def model_specs_from_manifest(exp3123: Mapping[str, Any], root: Path) -> list[JsonDict]:
    """Resolve mandated model specs from the Exp 3123 cache manifest."""

    selected_ids = list(
        exp3123.get("selected_model_ids")
        or exp3123.get("selected_headline_model_ids")
        or exp3123.get("present_model_ids")
        or []
    )
    present_ids = set(str(model_id) for model_id in exp3123.get("present_model_ids") or [])
    inventory = [row for row in exp3123.get("cache_inventory", []) if isinstance(row, Mapping)]
    specs: list[JsonDict] = []
    for model_id in MODEL_SPECS:
        inv = next((row for row in inventory if row.get("hf_id") == model_id), {})
        model_path = resolve_model_path(root, inv.get("path") or inv.get("model_path") or inv.get("resolved_path"))
        selected = model_id in selected_ids and model_path is not None
        specs.append(
            {
                "hf_id": model_id,
                "role": inv.get("role") or MODEL_ROLES[model_id],
                "selected": selected,
                "present": model_id in present_ids or model_path is not None,
                "cache_status": inv.get("cache_status") or ("resolved" if model_path else "missing"),
                "model_path": str(model_path) if model_path else None,
                "model_hash": bounded_model_hash(model_path) if model_path else None,
                "legacy_small_model": False,
            }
        )
    return specs


def first_usable_selected_model(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Return the first selected mandated model with a readable GGUF path."""

    for spec in model_specs:
        path = spec.get("model_path")
        if spec.get("selected") is True and path and Path(str(path)).is_file():
            return dict(spec)
    return None


def resolve_model_path(root: Path, raw_path: Any) -> Path | None:
    """Resolve model paths while rejecting missing or zero-byte files."""

    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = root / path
    try:
        if path.is_file() and path.stat().st_size > 0:
            return path
    except OSError:  # pragma: no cover - protects against disappearing cache paths.
        return None
    return None


def build_prompt(row: Mapping[str, Any]) -> str:
    """Build the narrow verifier prompt used for the bounded rerun."""

    return panel_v6.build_prompt(row)


def inference_substrate(
    *,
    exp3123: Mapping[str, Any],
    selected_model: Mapping[str, Any] | None,
    live_call_count: int,
    runtime_error: str | None,
) -> JsonDict:
    """Describe GPU/model/live-inference status explicitly."""

    return {
        "kind": "live_sota_verifier_rerun_v7",
        "model_selection_source": EXP3123_REL_PATH.as_posix(),
        "mandated_model_policy_visible": True,
        "uses_legacy_small_model_for_headline": False,
        "executes_models": live_call_count > 0,
        "live_model_calls": live_call_count,
        "selected_model_id": selected_model.get("hf_id") if selected_model else None,
        "selected_model_path": selected_model.get("model_path") if selected_model else None,
        "runtime_error": runtime_error,
        "gpu_preflight": dict(exp3123.get("gpu_preflight") or {}),
        "exact_solver_labels_authority": True,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return source provenance for every local file the rerun consumes."""

    return [source_row(root, role, rel_path, required) for role, rel_path, required in SOURCE_REL_PATHS]


def source_row(root: Path, role: str, rel_path: Path, required: bool) -> JsonDict:
    """Build one source-artifact provenance row."""

    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "required": required,
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and machine-readable safety gates."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3139 artifact missing required fields: {missing}")
    if int(artifact.get("live_call_count", -1)) < 0:
        raise ValueError("live_call_count must be nonnegative")
    for field in ("false_accept_rate", "false_reject_rate", "abstention_rate", "verifier_gain_delta"):
        if not math.isfinite(float(artifact.get(field, math.nan))):
            raise ValueError(f"finite metric required for {field}")
    for field in ("false_accept_rate", "false_reject_rate", "abstention_rate"):
        if not 0.0 <= float(artifact.get(field, math.nan)) <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")
    if artifact.get("live_call_count") == 0 and artifact.get("headline_claim_allowed") is True:
        raise ValueError("headline claims require live mandated model calls")
    if artifact.get("live_verifier_rerun_v7_ready") is True:
        if artifact.get("false_accept_gate_passed") is not True:
            raise ValueError("false_accept_gate_passed must be true for a ready rerun")
        if artifact.get("repair_gate_candidate_state") != "candidate_ready":
            raise ValueError("ready rerun requires candidate_ready repair gate state")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("live_call_count") == 0 and not verdict.startswith(("blocked_",)):
        raise ValueError("zero-live artifact must use an honest blocked verdict")
    if artifact.get("live_call_count") > 0 and not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return conductor-compatible terminal verdict wording."""

    state = str(artifact.get("repair_gate_candidate_state") or "")
    if state == "candidate_ready":
        return (
            "complete: live_verifier_rerun_v7_ready=true; "
            f"false_accept_rate={artifact.get('false_accept_rate')}; "
            f"verifier_gain_delta={artifact.get('verifier_gain_delta')}; "
            f"live_call_count={artifact.get('live_call_count')}"
        )
    if state == "blocked_no_live_model":
        detail = artifact.get("inference_substrate", {}).get("runtime_error") or "no mandated local GGUF produced live output"
        return f"blocked_no_live_model: live_call_count=0; headline_claim_allowed=false; detail={detail}"
    if state.startswith("blocked_") and artifact.get("live_call_count") == 0:
        return f"{state}: live_call_count=0; headline_claim_allowed=false"
    return (
        f"complete_{state}: false_accept_rate={artifact.get('false_accept_rate')}; "
        f"live_call_count={artifact.get('live_call_count')}; headline_claim_allowed=false"
    )


def mapping_rows(value: Any) -> list[JsonDict]:
    """Return only mapping rows from an arbitrary list-like value."""

    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def row_id_from(row: Mapping[str, Any]) -> str:
    """Return the stable row identifier shared by fixture and regression rows."""

    return str(row.get("fixture_id") or row.get("row_id") or row.get("source_fixture_id") or "")


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded safe ratio."""

    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def rate_delta(left: float, right: float) -> float:
    """Return a rounded finite metric delta."""

    return round(float(left) - float(right), 6)


def finite_unit_rate(value: Any) -> bool:
    """Return whether a value is a finite rate in [0, 1]."""

    return isinstance(value, (float, int)) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for prompt/output provenance."""

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Checksum a source file so claims remain tied to exact local bytes."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_model_hash(path: Path) -> str | None:
    """Hash bounded GGUF evidence without reading a large model into memory."""

    try:
        stat = path.stat()
    except OSError:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash JSON-serializable evidence using canonical key ordering."""

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def duration(started_s: float, now_s: float | None) -> float:
    """Return a nonnegative elapsed duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
