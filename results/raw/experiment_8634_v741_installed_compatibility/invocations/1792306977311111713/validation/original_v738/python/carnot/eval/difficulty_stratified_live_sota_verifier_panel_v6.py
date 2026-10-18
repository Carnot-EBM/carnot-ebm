"""Exp 3124 difficulty-stratified live SOTA verifier panel v6.

Spec refs: REQ-VERIFY-3124, SCENARIO-VERIFY-3124.

This module keeps the exact solver or test label as the authority and treats a
local SOTA model as a bounded verifier candidate. The important distinction is
that a model answer can be useful diagnostic evidence, but it cannot replace
the exact fixture label. When no mandated model can run, the artifact still
records the fixture metadata and blocks headline wording instead of filling the
gap with a legacy small model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence


JsonDict = dict[str, Any]
LiveRunner = Callable[[str, JsonDict, JsonDict, JsonDict], str]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6"
SCHEMA = "carnot.difficulty_stratified_live_sota_verifier_panel.v6"
OUTPUT_REL_PATH = Path(
    "results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"
)
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / (
    "experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.py"
)

EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3099_REL_PATH = Path("results/experiment_3099_local_sota_confidence_abstention_panel_v3.json")
EXP3111_REL_PATH = Path("results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json")
EXP3112_REL_PATH = Path("results/experiment_3112_logic_regularized_verifier_pilot_v1.json")
EXP3113_REL_PATH = Path("results/experiment_3113_diagnostic_local_sota_verifier_calibration_v5.json")
EXP3114_REL_PATH = Path(
    "results/experiment_3114_fragment_level_code_constraint_verification_pilot_v1.json"
)
EXP3115_REL_PATH = Path("results/experiment_3115_explicit_repair_gate_micro_panel_v4.json")
EXP3123_REL_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")

MANIFEST_REL_PATH = Path("results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl")
EXP3099_ROWS_REL_PATH = Path("results/local_sota_confidence_abstention_panel_3099/rows.jsonl")
EXP3112_ROWS_REL_PATH = Path("results/logic_regularized_verifier_pilot_3112/rows.jsonl")

MANDATORY_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_DIFFICULTY_BUCKETS = (
    "easy",
    "medium",
    "hard",
    "contradiction",
    "satisfiable_drift",
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
REQUIRED_FIELDS = (
    "difficulty_stratified_live_sota_panel_v6_ready",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "headline_claim_allowed",
    "exact_ground_truth_count",
    "difficulty_buckets",
    "fixture_family_metrics",
    "answer_extraction_metrics",
    "failure_mechanism_counts",
    "false_accept_rate",
    "false_reject_rate",
    "verifier_gain_delta",
    "repair_gate_state",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval -m pytest -o addopts='' tests/python/test_experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/difficulty_stratified_live_sota_verifier_panel_v6.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("experiment_template_cache_policy", Path("scripts/experiment_template.py"), False),
    ("exp3123_sota_cache_preconditions_manifest_v2", EXP3123_REL_PATH, True),
    ("exp3097_exact_protocol", EXP3097_REL_PATH, True),
    ("exp3097_stratified_manifest", MANIFEST_REL_PATH, True),
    ("exp3099_local_sota_panel", EXP3099_REL_PATH, True),
    ("exp3099_panel_rows", EXP3099_ROWS_REL_PATH, True),
    ("exp3111_certified_feedback_v3", EXP3111_REL_PATH, True),
    ("exp3112_logic_pilot", EXP3112_REL_PATH, True),
    ("exp3112_logic_rows", EXP3112_ROWS_REL_PATH, True),
    ("exp3113_diagnostic_calibration_v5", EXP3113_REL_PATH, True),
    ("exp3114_fragment_code_rows", EXP3114_REL_PATH, False),
    ("exp3115_repair_gate", EXP3115_REL_PATH, False),
)
ANSWER_TOKENS = ("UNREPAIRABLE", "REPAIRABLE", "INVALID", "VALID", "UNSAT", "SAT")
_ANSWER_RE = re.compile(r"\b(UNREPAIRABLE|REPAIRABLE|INVALID|VALID|UNSAT|SAT)\b", re.I)
_DEFAULT_LLAMA_CACHE: dict[str, Any] = {}


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows_from_text(text: str) -> list[JsonDict]:
    """Read JSONL object rows while ignoring malformed or non-object lines."""

    rows: list[JsonDict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read JSONL rows from disk, returning no rows when the file is absent."""

    try:
        return read_jsonl_rows_from_text(path.read_text(encoding="utf-8"))
    except OSError:
        return []


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
    """REQ-VERIFY-3124: build the stratified live verifier-panel artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3123 = read_json_object(root_path / EXP3123_REL_PATH)
    exp3097 = read_json_object(root_path / EXP3097_REL_PATH)
    exp3099 = read_json_object(root_path / EXP3099_REL_PATH)
    exp3111 = read_json_object(root_path / EXP3111_REL_PATH)
    exp3112 = read_json_object(root_path / EXP3112_REL_PATH)
    exp3113 = read_json_object(root_path / EXP3113_REL_PATH)
    exp3114 = read_json_object(root_path / EXP3114_REL_PATH)
    exp3115 = read_json_object(root_path / EXP3115_REL_PATH)

    manifest_rel_path = Path(str(exp3097.get("stratified_eval_manifest_path") or MANIFEST_REL_PATH))
    panel_rel_path = Path(str(exp3099.get("panel_rows_path") or EXP3099_ROWS_REL_PATH))
    logic_rel_path = Path(str(exp3112.get("diagnostic_rows_path") or EXP3112_ROWS_REL_PATH))

    manifest_rows = read_jsonl_rows(root_path / manifest_rel_path)
    panel_rows = read_jsonl_rows(root_path / panel_rel_path)
    logic_rows = read_jsonl_rows(root_path / logic_rel_path)
    certificates = [dict(row) for row in exp3111.get("certificates", []) if isinstance(row, Mapping)]
    fragment_checks = [
        dict(row) for row in exp3114.get("fragment_checks", []) if isinstance(row, Mapping)
    ]
    source_rows = source_artifacts(root_path, manifest_rel_path, panel_rel_path, logic_rel_path)
    missing_required_sources = [
        row for row in source_rows if row["required"] is True and row["exists"] is not True
    ]
    panel_fixture_rows = select_panel_rows(manifest_rows, panel_rows, certificates, logic_rows, fragment_checks)
    model_specs = model_specs_from_manifest(exp3123, exp3099, root_path)
    selected_model = first_usable_selected_model(model_specs)
    live_rows, runtime_error = run_live_panel(
        panel_fixture_rows,
        selected_model=selected_model,
        max_live_calls=int(max_live_calls),
        live_runner=live_runner,
    )
    live_metrics = metrics_for_rows(live_rows, "live_decision")
    baseline_metrics = metrics_for_rows(live_rows, "baseline_decision")
    verifier_gain_delta = (
        round(live_metrics["accuracy"] - baseline_metrics["accuracy"], 6) if live_rows else 0.0
    )
    readiness_checks = {
        "exp3123_manifest_ready": exp3123.get("sota_cache_manifest_v2_ready") is True,
        "exp3097_protocol_ready": exp3097.get("eval_protocol_ready") is True,
        "exp3099_panel_ready": exp3099.get("abstention_panel_v3_ready") is True,
        "exp3111_certified_feedback_ready": exp3111.get("certified_coherence_feedback_v3_ready")
        is True,
        "exp3112_logic_pilot_ready": exp3112.get("logic_regularized_verifier_pilot_ready")
        is True,
        "exp3113_diagnostic_ready": exp3113.get("diagnostic_verifier_calibration_v5_ready")
        is True,
        "required_sources_present": not missing_required_sources,
        "exact_rows_present": bool(panel_fixture_rows),
        "selected_mandated_model_usable": selected_model is not None,
        "live_calls_present": bool(live_rows),
        "finite_metrics": finite_metrics(live_metrics, verifier_gain_delta),
    }
    ready = all(readiness_checks.values())
    false_accept_rate = live_metrics["false_accept_rate"]
    false_reject_rate = live_metrics["false_reject_rate"]
    headline_claim_allowed = bool(
        ready
        and exp3123.get("headline_claim_allowed") is True
        and len(live_rows) >= int(min_live_calls_for_headline)
        and false_accept_rate == 0.0
        and verifier_gain_delta > 0.0
    )
    repair_gate = repair_gate_state(
        required_sources_present=readiness_checks["required_sources_present"],
        live_call_count=len(live_rows),
        min_live_calls_for_headline=int(min_live_calls_for_headline),
        false_accept_rate=false_accept_rate,
        verifier_gain_delta=verifier_gain_delta,
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "difficulty_stratified_live_sota_panel_v6_ready": ready,
        "model_specs": model_specs,
        "selected_model_ids": [row["hf_id"] for row in model_specs if row.get("selected") is True],
        "live_call_count": len(live_rows),
        "headline_claim_allowed": headline_claim_allowed,
        "exact_ground_truth_count": len(panel_fixture_rows),
        "difficulty_buckets": difficulty_bucket_metrics(panel_fixture_rows, live_rows),
        "fixture_family_metrics": grouped_metrics(panel_fixture_rows, live_rows, "fixture_family"),
        "generator_family_metrics": grouped_metrics(panel_fixture_rows, live_rows, "generator_family"),
        "answer_extraction_metrics": grouped_metrics(
            panel_fixture_rows,
            live_rows,
            "answer_extraction_format",
        ),
        "failure_mechanism_counts": failure_mechanism_counts(live_rows),
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "precision": live_metrics["precision"],
        "recall": live_metrics["recall"],
        "baseline_metrics": baseline_metrics,
        "live_metrics": live_metrics,
        "verifier_gain_delta": verifier_gain_delta,
        "solver_only_baseline": {
            "authority": "exact_solver_or_test_labels",
            "accuracy": 1.0 if panel_fixture_rows else 0.0,
            "live_lift_over_solver_only": round(live_metrics["accuracy"] - 1.0, 6)
            if live_rows
            else 0.0,
            "lift_claim_meaningful": False,
            "reason": "exact solver/test routing is the authority and cannot be outscored by a verifier",
        },
        "repair_gate_state": repair_gate,
        "readiness_checks": readiness_checks,
        "blocked_reasons": [key for key, ok in readiness_checks.items() if ok is not True],
        "runtime_error": runtime_error,
        "panel_fixture_metadata": panel_fixture_rows,
        "live_rows": live_rows,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row["sha256"] is not None
        },
        "inference_substrate": inference_substrate(
            exp3123=exp3123,
            exp3099=exp3099,
            exp3111=exp3111,
            exp3112=exp3112,
            exp3113=exp3113,
            exp3114=exp3114,
            exp3115=exp3115,
            selected_model=selected_model,
            live_call_count=len(live_rows),
            runtime_error=runtime_error,
        ),
        "duration_s": duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
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
    """Build, validate, and write the Exp 3124 JSON artifact."""

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
    validate_artifact(artifact)
    write_json(out_path, artifact)
    return out_path


def select_panel_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    panel_rows: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
    logic_rows: Sequence[Mapping[str, Any]],
    fragment_checks: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join fixture, cached route, certificate, logic, and fragment metadata."""

    panel_by_id = {str(row.get("source_fixture_id")): dict(row) for row in panel_rows}
    cert_by_id = {str(row.get("fixture_id")): dict(row) for row in certificates}
    logic_by_id = {str(row.get("fixture_id")): dict(row) for row in logic_rows}
    fragments_by_id: dict[str, list[JsonDict]] = {}
    for fragment in fragment_checks:
        fragments_by_id.setdefault(str(fragment.get("fixture_id") or ""), []).append(dict(fragment))
    rows: list[JsonDict] = []
    for manifest in manifest_rows:
        fixture_id = str(manifest.get("source_fixture_id") or "")
        exact_label = str(
            cert_by_id.get(fixture_id, {}).get("exact_label")
            or manifest.get("expected_answer")
            or logic_by_id.get(fixture_id, {}).get("exact_label")
            or ""
        ).upper()
        expected_action = str(
            manifest.get("verifier_target", {}).get("expected_action")
            or panel_by_id.get(fixture_id, {}).get("expected_action")
            or logic_by_id.get(fixture_id, {}).get("expected_action")
            or expected_action_from_answer(exact_label)
        )
        panel = panel_by_id.get(fixture_id, {})
        logic = logic_by_id.get(fixture_id, {})
        certificate = cert_by_id.get(fixture_id, {})
        row: JsonDict = {
            "fixture_id": fixture_id,
            "fixture_family": str(manifest.get("task_family") or "unknown"),
            "task_family": str(manifest.get("task_family") or "unknown"),
            "perturbation_type": str(manifest.get("perturbation_type") or "unknown"),
            "generator_family": generator_family_label(manifest),
            "exact_label": exact_label,
            "solver_label": manifest.get("solver_label") or certificate.get("solver_label"),
            "label_source": manifest.get("label_source") or certificate.get("solver_authority"),
            "expected_action": expected_action,
            "baseline_decision": str(panel.get("route_decision") or logic.get("baseline_decision") or "abstain"),
            "logic_decision": str(logic.get("logic_decision") or "abstain"),
            "certified_decision": certified_decision(certificate, exact_label),
            "answer_extraction_format": extraction_format_for_answer(exact_label),
            "prompt_payload": manifest.get("leakage_safe_prompt_payload") or {},
            "source_prompt_payload_sha256": manifest.get("source_prompt_payload_sha256"),
            "fragment_checks": fragments_by_id.get(fixture_id, []),
            "has_fragment_code_row": bool(fragments_by_id.get(fixture_id)),
            "certificate_present": bool(certificate),
            "cached_route_present": bool(panel),
        }
        row["difficulty_bucket_labels"] = difficulty_bucket_labels(row)
        rows.append(row)
    rows.sort(key=lambda item: item["fixture_id"])
    return rows


def run_live_panel(
    panel_rows: Sequence[JsonDict],
    *,
    selected_model: JsonDict | None,
    max_live_calls: int,
    live_runner: LiveRunner | None,
) -> tuple[list[JsonDict], str | None]:
    """Run bounded verifier calls or return no rows when no model is usable."""

    if selected_model is None or max_live_calls <= 0:
        return [], None
    runner = live_runner or default_live_runner
    selected_rows = select_live_call_rows(panel_rows, max_live_calls)
    live_rows: list[JsonDict] = []
    decode_config = {"max_tokens": 32, "temperature": 0.0, "seed": 20260526}
    model_hash = bounded_model_hash(Path(str(selected_model.get("model_path") or "")))
    try:
        for index, row in enumerate(selected_rows):
            prompt = build_prompt(row)
            raw_output = runner(prompt, row, selected_model, dict(decode_config))
            extracted = extract_answer(raw_output)
            live_decision = decision_from_answer(extracted)
            live_row = dict(row)
            live_row.update(
                {
                    "live_call_index": index,
                    "model_id": selected_model["hf_id"],
                    "model_path": selected_model.get("model_path"),
                    "model_hash": model_hash,
                    "prompt_hash": sha256_text(prompt),
                    "raw_output": raw_output,
                    "raw_output_hash": sha256_text(raw_output),
                    "extracted_answer": extracted,
                    "live_decision": live_decision,
                    "exact_answer_match": extracted == row["exact_label"],
                    "live_correct": live_decision == row["expected_action"],
                    "failure_mechanism": live_failure_mechanism(row, extracted, live_decision),
                    "decode_config": dict(decode_config),
                }
            )
            live_rows.append(live_row)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return live_rows, None


def default_live_runner(  # pragma: no cover - exercised only by live operator runs.
    prompt: str,
    _row: JsonDict,
    model_spec: JsonDict,
    decode_config: JsonDict,
) -> str:
    """Call a local mandated GGUF through llama.cpp for one bounded verifier row."""

    from llama_cpp import Llama

    model_path = str(model_spec["model_path"])
    if model_path not in _DEFAULT_LLAMA_CACHE:
        _DEFAULT_LLAMA_CACHE[model_path] = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_batch=128,
            n_ubatch=128,
            n_gpu_layers=-1,
            main_gpu=0,
            verbose=False,
        )
    output = _DEFAULT_LLAMA_CACHE[model_path](
        prompt,
        max_tokens=int(decode_config["max_tokens"]),
        temperature=float(decode_config["temperature"]),
        seed=int(decode_config["seed"]),
        stop=["\n\n", "</s>"],
    )
    return llama_text(output)


def select_live_call_rows(rows: Sequence[JsonDict], max_live_calls: int) -> list[JsonDict]:
    """Choose a small row set that covers the required difficulty keys first."""

    selected: list[JsonDict] = []
    seen: set[str] = set()
    for bucket in REQUIRED_DIFFICULTY_BUCKETS:
        for row in rows:
            if row["fixture_id"] not in seen and bucket in row["difficulty_bucket_labels"]:
                selected.append(row)
                seen.add(row["fixture_id"])
                break
        if len(selected) >= max_live_calls:
            return selected[:max_live_calls]
    for row in rows:
        if row["fixture_id"] not in seen:
            selected.append(row)
            seen.add(row["fixture_id"])
        if len(selected) >= max_live_calls:
            break
    return selected[:max_live_calls]


def build_prompt(row: Mapping[str, Any]) -> str:
    """Build the deliberately narrow verifier prompt for one exact fixture."""

    payload = json.dumps(row.get("prompt_payload") or {}, sort_keys=True)
    return (
        "You are a verifier. Return exactly one answer token from this set: "
        "VALID, INVALID, SAT, UNSAT, REPAIRABLE, UNREPAIRABLE.\n"
        f"Fixture family: {row.get('fixture_family')}\n"
        f"Perturbation: {row.get('perturbation_type')}\n"
        f"Fixture payload JSON: {payload}\n"
        "Answer token:"
    )


def extract_answer(raw_output: str) -> str | None:
    """Extract the first supported verifier answer token from text or JSON."""

    text = str(raw_output or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        for key in ("answer", "verdict", "label", "result"):
            token = payload.get(key)
            if isinstance(token, str):
                normalized = normalize_answer_token(token)
                if normalized is not None:
                    return normalized
    match = _ANSWER_RE.search(text)
    return normalize_answer_token(match.group(1)) if match else None


def normalize_answer_token(value: str) -> str | None:
    """Normalize a candidate answer token to the exact-label vocabulary."""

    token = re.sub(r"[^A-Za-z_]", "", str(value)).upper()
    return token if token in ANSWER_TOKENS else None


def expected_action_from_answer(answer: str) -> str:
    """Map exact answer labels onto verifier accept/reject/abstain actions."""

    normalized = str(answer).upper()
    if normalized in {"VALID", "SAT"}:
        return "accept"
    if normalized in {"INVALID", "UNSAT", "REPAIRABLE", "UNREPAIRABLE"}:
        return "reject"
    return "abstain"


def decision_from_answer(answer: str | None) -> str:
    """Map a model-extracted answer token onto the verifier action space."""

    return expected_action_from_answer(answer or "")


def extraction_format_for_answer(answer: str) -> str:
    """Classify the expected answer shape so input-structure effects are visible."""

    normalized = str(answer).upper()
    if normalized in {"VALID", "INVALID"}:
        return "validity_token"
    if normalized in {"SAT", "UNSAT"}:
        return "sat_token"
    if normalized in {"REPAIRABLE", "UNREPAIRABLE"}:
        return "repairability_token"
    return "unknown_token"


def difficulty_bucket_labels(row: Mapping[str, Any]) -> list[str]:
    """Return all difficulty and structural buckets that apply to a fixture."""

    family = str(row.get("task_family") or row.get("fixture_family") or "").lower()
    perturbation = str(row.get("perturbation_type") or "").lower()
    exact = str(row.get("exact_label") or "").upper()
    expected = str(row.get("expected_action") or expected_action_from_answer(exact))
    baseline = str(row.get("baseline_decision") or "")
    buckets: set[str] = set()
    if "arithmetic" in family and "repair" not in family:
        buckets.add("easy")
    if "smt" in family:
        buckets.add("medium")
    if "repair" in family or "json" in perturbation or exact in {"REPAIRABLE", "UNREPAIRABLE"}:
        buckets.add("hard")
    if exact in {"INVALID", "UNSAT"} or "false" in perturbation or "unsat" in perturbation:
        buckets.add("contradiction")
    if expected == "accept" and (baseline != "accept" or "drift" in perturbation):
        buckets.add("satisfiable_drift")
    if (
        "code" in family
        or "assertion" in family
        or "repair" in family
        or bool(row.get("has_fragment_code_row"))
    ):
        buckets.add("fragment_code")
    if not buckets.intersection({"easy", "medium", "hard"}):
        buckets.add("hard")
    return [bucket for bucket in REQUIRED_DIFFICULTY_BUCKETS if bucket in buckets]


def live_failure_mechanism(
    row: Mapping[str, Any],
    extracted_answer: str | None,
    live_decision: str,
) -> str:
    """Classify the main row-level failure mechanism for auditability."""

    expected = str(row.get("expected_action"))
    exact = str(row.get("exact_label") or "").upper()
    if live_decision == expected:
        return "no_failure"
    if extracted_answer is None:
        return "data_driven_unparseable"
    if live_decision == "accept" and expected == "reject" and exact in {
        "INVALID",
        "UNSAT",
        "REPAIRABLE",
        "UNREPAIRABLE",
    }:
        return "contradiction"
    if live_decision == "reject" and expected == "accept":
        return "satisfiable_drift"
    return "reasoning_driven_wrong_label"


def metrics_for_rows(rows: Sequence[Mapping[str, Any]], decision_field: str) -> JsonDict:
    """Compute false accepts, false rejects, precision, recall, and accuracy."""

    if not rows:
        return {
            "count": 0,
            "accuracy": 0.0,
            "false_accept_count": 0,
            "false_reject_count": 0,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accept_count": 0,
            "reject_count": 0,
            "abstain_count": 0,
        }
    positives = [row for row in rows if row.get("expected_action") == "accept"]
    negatives = [row for row in rows if row.get("expected_action") == "reject"]
    accept_predictions = [row for row in rows if row.get(decision_field) == "accept"]
    true_accepts = [
        row
        for row in rows
        if row.get(decision_field) == "accept" and row.get("expected_action") == "accept"
    ]
    false_accepts = [
        row
        for row in rows
        if row.get(decision_field) == "accept" and row.get("expected_action") == "reject"
    ]
    false_rejects = [
        row
        for row in rows
        if row.get(decision_field) == "reject" and row.get("expected_action") == "accept"
    ]
    return {
        "count": len(rows),
        "accuracy": rate(
            sum(row.get(decision_field) == row.get("expected_action") for row in rows),
            len(rows),
        ),
        "false_accept_count": len(false_accepts),
        "false_reject_count": len(false_rejects),
        "false_accept_rate": rate(len(false_accepts), len(negatives)),
        "false_reject_rate": rate(len(false_rejects), len(positives)),
        "precision": rate(len(true_accepts), len(accept_predictions)),
        "recall": rate(len(true_accepts), len(positives)),
        "accept_count": len(accept_predictions),
        "reject_count": sum(row.get(decision_field) == "reject" for row in rows),
        "abstain_count": sum(row.get(decision_field) == "abstain" for row in rows),
    }


def difficulty_bucket_metrics(
    panel_rows: Sequence[Mapping[str, Any]],
    live_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Return metrics for each required bucket, preserving zero-live buckets."""

    result: JsonDict = {}
    for bucket in REQUIRED_DIFFICULTY_BUCKETS:
        all_group = [row for row in panel_rows if bucket in row.get("difficulty_bucket_labels", [])]
        live_group = [row for row in live_rows if bucket in row.get("difficulty_bucket_labels", [])]
        metrics = metrics_for_rows(live_group, "live_decision")
        metrics["count"] = len(all_group)
        metrics["live_count"] = len(live_group)
        result[bucket] = metrics
    return result


def grouped_metrics(
    panel_rows: Sequence[Mapping[str, Any]],
    live_rows: Sequence[Mapping[str, Any]],
    group_field: str,
) -> JsonDict:
    """Return per-group live metrics while retaining total fixture counts."""

    keys = sorted({str(row.get(group_field) or "unknown") for row in panel_rows})
    result: JsonDict = {}
    for key in keys:
        all_group = [row for row in panel_rows if str(row.get(group_field) or "unknown") == key]
        live_group = [row for row in live_rows if str(row.get(group_field) or "unknown") == key]
        metrics = metrics_for_rows(live_group, "live_decision")
        metrics["count"] = len(all_group)
        metrics["live_count"] = len(live_group)
        result[key] = metrics
    return result


def failure_mechanism_counts(live_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count row-level and broad failure categories separately."""

    counts: JsonDict = {
        "no_failure": 0,
        "data_driven": 0,
        "reasoning_driven": 0,
        "contradiction": 0,
        "satisfiable_drift": 0,
        "false_accept": 0,
        "false_reject": 0,
        "abstention_or_unparseable": 0,
    }
    for row in live_rows:
        mechanism = str(row.get("failure_mechanism") or "no_failure")
        expected = row.get("expected_action")
        decision = row.get("live_decision")
        if mechanism == "no_failure":
            counts["no_failure"] += 1
        if mechanism.startswith("data_driven"):
            counts["data_driven"] += 1
            counts["abstention_or_unparseable"] += 1
        elif mechanism == "contradiction":
            counts["reasoning_driven"] += 1
            counts["contradiction"] += 1
        elif mechanism == "satisfiable_drift":
            counts["reasoning_driven"] += 1
            counts["satisfiable_drift"] += 1
        elif mechanism != "no_failure":
            counts["reasoning_driven"] += 1
        if expected == "reject" and decision == "accept":
            counts["false_accept"] += 1
        if expected == "accept" and decision == "reject":
            counts["false_reject"] += 1
    return counts


def repair_gate_state(
    *,
    required_sources_present: bool,
    live_call_count: int,
    min_live_calls_for_headline: int,
    false_accept_rate: float,
    verifier_gain_delta: float,
) -> str:
    """Map evidence quality and safety metrics to the downstream repair gate."""

    if required_sources_present is not True:
        return "blocked_missing_inputs"
    if live_call_count <= 0:
        return "blocked_no_live_model"
    if live_call_count < min_live_calls_for_headline:
        return "blocked_tiny_panel"
    if false_accept_rate > 0.0:
        return "blocked_false_accept"
    if verifier_gain_delta <= 0.0:
        return "blocked_no_lift"
    return "unblocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return conductor-compatible terminal verdict wording."""

    gate = str(artifact.get("repair_gate_state") or "")
    if gate == "blocked_missing_inputs":
        return "blocked_missing_inputs: " + ",".join(
            artifact.get("blocked_reasons") or ["required_source_artifacts_unavailable"]
        )
    if gate == "blocked_no_live_model":
        detail = artifact.get("runtime_error") or "no mandated model produced live verifier output"
        return f"blocked_no_live_model: live_call_count=0; headline_claim_allowed=false; detail={detail}"
    if gate == "blocked_tiny_panel":
        return (
            "complete_blocked_tiny_panel: "
            f"live_call_count={artifact.get('live_call_count')}; headline_claim_allowed=false"
        )
    if gate == "blocked_false_accept":
        return (
            "complete_blocked_false_accept: "
            f"false_accept_rate={artifact.get('false_accept_rate')}; "
            f"live_call_count={artifact.get('live_call_count')}; headline_claim_allowed=false"
        )
    if gate == "blocked_no_lift":
        return (
            "complete_blocked_no_lift: "
            f"verifier_gain_delta={artifact.get('verifier_gain_delta')}; "
            "headline_claim_allowed=false"
        )
    return (
        "complete: difficulty_stratified_live_sota_panel_v6_ready=true; "
        f"live_call_count={artifact.get('live_call_count')}; "
        f"verifier_gain_delta={artifact.get('verifier_gain_delta')}; "
        f"repair_gate_state={gate}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3124 terminal artifact violates its schema contract."""

    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if int(artifact.get("live_call_count", -1)) < 0:
        raise ValueError("live_call_count must be nonnegative")
    for field in ("false_accept_rate", "false_reject_rate", "verifier_gain_delta"):
        value = float(artifact.get(field, math.nan))
        if not math.isfinite(value):
            raise ValueError(f"finite metric required for {field}")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("live_call_count") == 0 and artifact.get("headline_claim_allowed") is True:
        raise ValueError("headline claims require live mandated model calls")
    if artifact.get("live_call_count") == 0 and not (
        verdict.startswith("blocked_no_live_model") or verdict.startswith("blocked_missing_inputs")
    ):
        raise ValueError("zero-live artifact must use blocked_no_live_model or missing-input verdict")
    if artifact.get("live_call_count") > 0 and not any(
        verdict.startswith(prefix) for prefix in SUCCESS_PREFIXES
    ):
        raise ValueError("live artifact honest_verdict must start with a success prefix")


def model_specs_from_manifest(
    exp3123: Mapping[str, Any],
    exp3099: Mapping[str, Any],
    root: Path,
) -> list[JsonDict]:
    """Return auditable selected-model evidence from Exp 3123 and Exp 3099."""

    selected_ids = list(
        exp3123.get("selected_model_ids")
        or exp3123.get("selected_headline_model_ids")
        or exp3099.get("selected_model_ids")
        or []
    )
    present_ids = list(exp3123.get("present_model_ids") or [])
    if not selected_ids and present_ids:
        selected_ids = [str(present_ids[0])]
    inventory = [row for row in exp3123.get("cache_inventory", []) if isinstance(row, Mapping)]
    if not inventory and exp3099.get("model_specs"):
        inventory = [row for row in exp3099.get("model_specs", []) if isinstance(row, Mapping)]
    specs: list[JsonDict] = []
    for model_id in MANDATORY_MODEL_IDS:
        inv = next((row for row in inventory if row.get("hf_id") == model_id), {})
        raw_path = inv.get("path") or inv.get("resolved_path") or inv.get("model_path")
        model_path = resolve_model_path(root, raw_path)
        selected = model_id in selected_ids and model_path is not None
        specs.append(
            {
                "hf_id": model_id,
                "selected": selected,
                "present": model_id in present_ids or bool(model_path),
                "cache_status": inv.get("cache_status") or ("resolved" if model_path else "missing"),
                "model_path": str(model_path) if model_path is not None else None,
                "model_hash": bounded_model_hash(model_path) if model_path is not None else None,
                "role": inv.get("role"),
                "legacy_small_model": False,
            }
        )
    return specs


def first_usable_selected_model(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Return the first selected mandated model that has a readable GGUF path."""

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
    except OSError:  # pragma: no cover - protects against files disappearing mid-stat.
        return None
    return None


def generator_family_label(row: Mapping[str, Any]) -> str:
    """Label source generator family from fixture metadata."""

    explicit = row.get("generator_family")
    if explicit:
        return str(explicit)
    fixture_id = str(row.get("source_fixture_id") or row.get("fixture_id") or "")
    family = str(row.get("task_family") or "")
    if "repair" in family:
        return "repair_fixture_generator"
    if fixture_id.startswith("resyn-"):
        return "resynthesized_exact_fixture"
    return "unknown_generator"


def certified_decision(certificate: Mapping[str, Any], exact_label: str) -> str:
    """Read the certified route action or fall back to the exact label action."""

    route = certificate.get("maxsat_route")
    if isinstance(route, Mapping) and route.get("action"):
        return str(route["action"])
    return expected_action_from_answer(exact_label)


def finite_metrics(metrics: Mapping[str, Any], verifier_gain_delta: float) -> bool:
    """Check that the public scalar metrics are finite numbers."""

    return all(
        math.isfinite(float(metrics.get(field, math.nan)))
        for field in ("accuracy", "false_accept_rate", "false_reject_rate", "precision", "recall")
    ) and math.isfinite(float(verifier_gain_delta))


def inference_substrate(
    *,
    exp3123: Mapping[str, Any],
    exp3099: Mapping[str, Any],
    exp3111: Mapping[str, Any],
    exp3112: Mapping[str, Any],
    exp3113: Mapping[str, Any],
    exp3114: Mapping[str, Any],
    exp3115: Mapping[str, Any],
    selected_model: Mapping[str, Any] | None,
    live_call_count: int,
    runtime_error: str | None,
) -> JsonDict:
    """Describe the live/model/solver provenance for the artifact."""

    return {
        "kind": "difficulty_stratified_live_sota_verifier_panel",
        "mandated_model_policy_visible": True,
        "uses_legacy_small_model_for_headline": False,
        "executes_models": live_call_count > 0,
        "live_model_calls": live_call_count,
        "selected_model_id": selected_model.get("hf_id") if selected_model else None,
        "selected_model_path": selected_model.get("model_path") if selected_model else None,
        "runtime_error": runtime_error,
        "exact_solver_labels_authority": True,
        "exp3123_headline_claim_allowed": exp3123.get("headline_claim_allowed"),
        "cached_route_source_executed_models": isinstance(exp3099.get("inference_substrate"), Mapping)
        and exp3099.get("inference_substrate", {}).get("executes_models") is True,
        "certified_feedback_source": EXP3111_REL_PATH.as_posix(),
        "logic_pilot_source": EXP3112_REL_PATH.as_posix(),
        "diagnostic_gate_source": EXP3113_REL_PATH.as_posix(),
        "fragment_source_present": bool(exp3114),
        "repair_gate_source_present": bool(exp3115),
        "source_certified_feedback_executed_solvers": isinstance(
            exp3111.get("inference_substrate"),
            Mapping,
        )
        and exp3111.get("inference_substrate", {}).get("executes_solvers") is True,
        "source_logic_pilot_live_llm_inference": isinstance(
            exp3112.get("inference_substrate"),
            Mapping,
        )
        and exp3112.get("inference_substrate", {}).get("live_llm_inference") is True,
    }


def source_artifacts(
    root: Path,
    manifest_rel_path: Path,
    panel_rel_path: Path,
    logic_rel_path: Path,
) -> list[JsonDict]:
    """Return source provenance, substituting dynamic row paths from artifacts."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_SPECS:
        path = manifest_rel_path if rel_path == MANIFEST_REL_PATH else rel_path
        path = panel_rel_path if rel_path == EXP3099_ROWS_REL_PATH else path
        path = logic_rel_path if rel_path == EXP3112_ROWS_REL_PATH else path
        full_path = root / path
        rows.append(
            {
                "id": source_id,
                "path": path.as_posix(),
                "required": required,
                "exists": full_path.is_file(),
                "sha256": sha256_file(full_path),
            }
        )
    return rows


def llama_text(raw_response: Any) -> str:
    """Extract text from a llama.cpp response object."""

    if isinstance(raw_response, str):
        return raw_response
    if not isinstance(raw_response, Mapping):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return ""


def rate(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded safe ratio."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for prompt/output provenance."""

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum when a source file exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_model_hash(path: Path) -> str | None:
    """Hash bounded file evidence so huge GGUF provenance stays cheap."""

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


def relative_path(root: Path, path: Path) -> str:
    """Return a path relative to the repo root when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON for reproducible artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float, now_s: float | None) -> float:
    """Return a nonnegative elapsed duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
