"""Build the Exp 3181 clean live SOTA verifier rerun v10 artifact.

Spec refs: REQ-VERIFY-3181, SCENARIO-VERIFY-3181.

The v10 rerun is receipt-gated. It may spend new local SOTA calls only after
Exp 3179 proves a full local SOTA receipt and Exp 3180 proves controlled
invariance over exact-authority rows. If either gate is missing or false, this
module writes a complete gated-skip artifact for repair gate v4 instead of
letting downstream jobs infer state from a missing result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairProvider = Callable[[], list[dict[str, Any]] | None]
PanelRunner = Callable[[list[JsonDict], list[JsonDict]], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3181_clean_live_sota_verifier_rerun_v10"
SCHEMA = "carnot.clean_live_sota_verifier_rerun.v10"
OUTPUT_REL_PATH = Path("results/experiment_3181_clean_live_sota_verifier_rerun_v10.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3181_clean_live_sota_verifier_rerun_v10.py"

EXP3178_REL_PATH = Path("results/experiment_3178_receipt_backed_authenticity_contract_v3.json")
EXP3179_REL_PATH = Path("results/experiment_3179_local_sota_receipt_smoke_v3.json")
EXP3180_REL_PATH = Path("results/experiment_3180_controlled_invariance_executor_v2.json")
EXP3167_REL_PATH = Path("results/experiment_3167_clean_live_sota_verifier_rerun_v9.json")

DEFAULT_RANDOM_SEED = 20260527
MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
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
    "clean_live_sota_verifier_rerun_v10_ready",
    "gated_skip",
    "gate_reasons",
    "live_call_count",
    "models_used",
    "proof_receipts_used",
    "exact_row_count",
    "known_false_accept_regression_count",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "controlled_invariance_passed",
    "flagged_adversarial",
    "headline_claim_allowed",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3181_clean_live_sota_verifier_rerun_v10.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3181_clean_live_sota_verifier_rerun_v10.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/clean_live_sota_verifier_rerun_v10.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_REL_PATHS = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("experiment_template_policy", Path("scripts/experiment_template.py"), True, "python"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("exp3178_receipt_contract_v3", EXP3178_REL_PATH, True, "json"),
    ("exp3179_local_sota_receipt_smoke_v3", EXP3179_REL_PATH, True, "json"),
    ("exp3180_controlled_invariance_executor_v2", EXP3180_REL_PATH, True, "json"),
    ("exp3167_clean_rerun_v9", EXP3167_REL_PATH, True, "json"),
    (
        "exp3181_module",
        Path("python/carnot/verify/clean_live_sota_verifier_rerun_v10.py"),
        False,
        "python",
    ),
    (
        "exp3181_script",
        Path("scripts/experiment_3181_clean_live_sota_verifier_rerun_v10.py"),
        False,
        "python",
    ),
    (
        "exp3181_tests",
        Path("tests/python/test_experiment_3181_clean_live_sota_verifier_rerun_v10.py"),
        False,
        "python",
    ),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    cached_pair_provider: CachedPairProvider | None = None,
    panel_runner: PanelRunner | None = None,
    max_live_calls: int = 8,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3181: build a live verifier artifact or explicit gated skip."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3178 = read_json_object(root_path / EXP3178_REL_PATH)
    exp3179 = read_json_object(root_path / EXP3179_REL_PATH)
    exp3180 = read_json_object(root_path / EXP3180_REL_PATH)
    exp3167 = read_json_object(root_path / EXP3167_REL_PATH)
    sources = source_artifacts(root_path)
    source_problems = source_errors(sources)
    exact_rows = collect_exact_rows(exp3180, exp3167)
    regression_ids = collect_regression_ids(exp3180, exact_rows)
    receipts = proof_receipts_used(exp3179)
    receipt_valid = receipt_validity(receipts)
    reasons = [
        *(f"source_error: {row['path']} {row['reason']}" for row in source_problems),
        *gate_reasons(exp3179=exp3179, exp3180=exp3180, receipt_valid=receipt_valid),
    ]
    models = select_models(cached_pair_provider or cached_sota_pair) if not reasons else []
    if not reasons and not models:
        reasons.append("cached_sota_pair returned fewer than two mandated model paths")
    live_rows = select_panel_rows(exact_rows, regression_ids, max_live_calls)
    if not reasons and not live_rows:
        reasons.append("exact authority rows unavailable for live panel")
    if not reasons and panel_runner is None:
        reasons.append("live panel runner unavailable")

    panel_rows: list[JsonDict] = []
    if not reasons and panel_runner is not None:
        panel_rows = normalize_panel_rows(panel_runner(live_rows, models), live_rows)
    live_call_count = len(panel_rows)
    metrics = score_panel(panel_rows, live_rows, regression_ids)
    if panel_rows and metrics["known_false_accepts_accepted"]:
        reasons.append(
            "known_false_accepts_accepted: " + ",".join(metrics["known_false_accepts_accepted"])
        )

    gated_skip = live_call_count == 0
    controlled_passed = exp3180.get("controlled_invariance_passed") is True
    flagged_adversarial = not (
        bool(panel_rows)
        and receipt_valid["valid"] is True
        and controlled_passed
        and not metrics["known_false_accepts_accepted"]
    )
    headline_allowed = bool(panel_rows) and not flagged_adversarial and not reasons
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": duration(start, finished),
        "clean_live_sota_verifier_rerun_v10_ready": True,
        "gated_skip": gated_skip,
        "gate_reasons": reasons,
        "live_call_count": live_call_count,
        "models_used": [] if gated_skip else models,
        "proof_receipts_used": receipts,
        "exact_row_count": exact_row_count(exp3180, exact_rows),
        "known_false_accept_regression_count": known_false_accept_count(exp3180, regression_ids),
        "false_accept_rate": metrics["false_accept_rate"],
        "false_reject_rate": metrics["false_reject_rate"],
        "abstention_rate": metrics["abstention_rate"],
        "controlled_invariance_passed": controlled_passed,
        "flagged_adversarial": flagged_adversarial,
        "headline_claim_allowed": headline_allowed,
        "transcript_receipt_validity": receipt_valid,
        "token_suspicion_used_as_triage_only": (
            exp3180.get("token_suspicion_used_as_triage_only") is True
        ),
        "known_false_accepts_accepted": metrics["known_false_accepts_accepted"],
        "known_false_accepts_rejected": metrics["known_false_accepts_rejected"],
        "prompt_hashes": [row["prompt_hash"] for row in panel_rows],
        "transcript_hashes": [row["transcript_hash"] for row in panel_rows],
        "token_counts": aggregate_token_counts(panel_rows),
        "exact_rows_evaluated": exact_rows,
        "panel_rows": panel_rows,
        "metrics_computed": bool(panel_rows),
        "preconditions_checked": preconditions_checked(exp3179, exp3180, receipt_valid),
        "source_artifacts": sources,
        "source_checksums": {row["path"]: row["sha256"] for row in sources if row.get("sha256")},
        "source_errors": source_problems,
        "previous_v9_summary": previous_v9_summary(exp3167),
        "receipt_contract_ready": exp3178.get("receipt_backed_authenticity_contract_v3_ready")
        is True,
        "field_principles": field_principles(),
        "inference_substrate": inference_substrate(
            gated_skip=gated_skip,
            live_call_count=live_call_count,
            models=models,
            exp3179=exp3179,
            controlled_passed=controlled_passed,
            reasons=reasons,
        ),
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
    cached_pair_provider: CachedPairProvider | None = None,
    panel_runner: PanelRunner | None = None,
    max_live_calls: int = 8,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3181 terminal JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        cached_pair_provider=cached_pair_provider,
        panel_runner=panel_runner,
        max_live_calls=max_live_calls,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to an empty mapping."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return every local source artifact consumed by the v10 gate."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_REL_PATHS:
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

    errors: list[JsonDict] = []
    for row in sources:
        if row.get("required") is not True:
            continue
        if row.get("present") is not True:
            errors.append({"path": str(row.get("path") or ""), "reason": "missing"})
        elif row.get("source_type") == "json" and row.get("readable_json_object") is not True:
            errors.append({"path": str(row.get("path") or ""), "reason": "malformed_json"})
    return errors


def sha256_file(path: Path) -> str | None:
    """Return a source checksum for present files."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash structured values after deterministic JSON normalization."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def duration(started_s: float, now_s: float) -> float:
    """Clamp elapsed time so clock anomalies cannot become evidence."""

    return round(max(0.0, float(now_s) - float(started_s)), 6)


def collect_exact_rows(exp3180: Mapping[str, Any], exp3167: Mapping[str, Any]) -> list[JsonDict]:
    """Collect exact-authority rows from Exp 3180, with a v9 fallback."""

    source_rows = mapping_list(exp3180.get("exact_rows_evaluated"))
    if not source_rows:
        source_rows = mapping_list(exp3167.get("exact_rows_evaluated"))
    rows: list[JsonDict] = []
    for raw in source_rows:
        row_id = str(raw.get("row_id") or raw.get("fixture_id") or "")
        if not row_id:
            continue
        decision = str(raw.get("exact_authority_decision") or "").lower()
        normalized = dict(raw) | {
            "row_id": row_id,
            "exact_label": str(raw.get("exact_label") or ""),
            "exact_authority_decision": decision or decision_from_exact_row(raw),
            "known_false_accept_regression": raw.get("known_false_accept_regression") is True,
        }
        rows.append(normalized)
    return rows


def collect_regression_ids(
    exp3180: Mapping[str, Any], exact_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Return known false-accept regression IDs from Exp 3180 exact evidence."""

    ids = {str(value) for value in exp3180.get("known_false_accept_regression_ids") or [] if value}
    ids.update(
        str(row["row_id"]) for row in exact_rows if row.get("known_false_accept_regression") is True
    )
    return sorted(ids)


def exact_row_count(exp3180: Mapping[str, Any], exact_rows: Sequence[Mapping[str, Any]]) -> int:
    """Prefer the upstream explicit denominator, then the rows actually loaded."""

    return safe_int(exp3180.get("exact_row_count")) or len(exact_rows)


def known_false_accept_count(exp3180: Mapping[str, Any], regression_ids: Sequence[str]) -> int:
    """Prefer the upstream explicit regression denominator."""

    return safe_int(exp3180.get("known_false_accept_regression_count")) or len(regression_ids)


def proof_receipts_used(exp3179: Mapping[str, Any]) -> list[JsonDict]:
    """Normalize Exp 3179 receipts without granting them scoring authority."""

    rows: list[JsonDict] = []
    for index, receipt in enumerate(mapping_list(exp3179.get("proof_receipts"))):
        rows.append(
            {
                "source_experiment": "exp3179",
                "index": index,
                "selected_model_id": str(receipt.get("selected_model_id") or ""),
                "model_path": str(receipt.get("model_path") or ""),
                "model_file_hash": str(receipt.get("model_file_hash") or ""),
                "loader_name": str(receipt.get("loader_name") or ""),
                "substrate_used": str(receipt.get("substrate_used") or ""),
                "prompt_hash": str(receipt.get("prompt_hash") or ""),
                "response_hash": str(receipt.get("response_hash") or ""),
                "transcript_hash": str(receipt.get("transcript_hash") or ""),
                "token_counts": mapping(receipt.get("token_counts")),
                "random_seed": receipt.get("random_seed"),
                "wall_clock_s": receipt.get("wall_clock_s"),
                "command_hash": str(receipt.get("command_hash") or ""),
                "subprocess_return_code": receipt.get("subprocess_return_code"),
                "stderr_tail": str(receipt.get("stderr_tail") or ""),
                "throughput_plausibility": mapping(receipt.get("throughput_plausibility")),
                "replay_count": safe_int(receipt.get("replay_count")),
                "used_for_live_verifier_scoring": False,
            }
        )
    return rows


def receipt_validity(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check receipt identity fields without deciding verifier correctness."""

    reasons: list[str] = []
    if not receipts:
        reasons.append("proof receipts absent")
    hashes = [str(row.get("transcript_hash") or "") for row in receipts]
    if any(not value for value in hashes):
        reasons.append("missing transcript hash")
    if len(set(hashes)) != len(hashes):
        reasons.append("duplicate transcript hashes")
    for row in receipts:
        if row.get("subprocess_return_code") not in (0, "0"):
            reasons.append("nonzero receipt subprocess return code")
            break
    return {"valid": not reasons, "reasons": reasons, "receipt_count": len(receipts)}


def gate_reasons(
    *,
    exp3179: Mapping[str, Any],
    exp3180: Mapping[str, Any],
    receipt_valid: Mapping[str, Any],
) -> list[str]:
    """Translate failed preconditions into exact, repair-actionable blockers."""

    reasons: list[str] = []
    if exp3179.get("clean_rerun_allowed") is not True:
        reasons.append("exp3179.clean_rerun_allowed=false")
        substrate = str(exp3179.get("substrate_classification") or "")
        if substrate:
            reasons.append(f"exp3179.substrate_classification={substrate}")
        blocked = str(exp3179.get("blocked_reason") or "")
        if blocked:
            reasons.append(f"exp3179.blocked_reason: {blocked}")
    if exp3180.get("controlled_invariance_passed") is not True:
        reasons.append("exp3180.controlled_invariance_passed=false")
        for blocker in exp3180.get("blocker_reasons") or []:
            reasons.append(f"exp3180.blocker: {blocker}")
    if receipt_valid.get("valid") is not True:
        for reason in receipt_valid.get("reasons") or []:
            reasons.append(f"receipt_proof_invalid: {reason}")
    return reasons


def select_models(provider: CachedPairProvider) -> list[JsonDict]:
    """Select locally cached mandated SOTA GGUFs from cached_sota_pair()."""

    try:
        pair = provider()
    except Exception:
        return []
    rows = pair if isinstance(pair, list) else []
    selected: list[JsonDict] = []
    for row in rows:
        model_id = str(mapping(row).get("hf_id") or "")
        model_path = str(mapping(row).get("model_path") or "")
        if model_id in MANDATED_MODEL_IDS and model_path:
            selected.append(
                {
                    "hf_id": model_id,
                    "name": str(mapping(row).get("name") or model_id.rsplit("/", 1)[-1]),
                    "model_path": model_path,
                    "gpu": mapping(row).get("gpu"),
                    "legacy_small_model": False,
                    "source": "cached_sota_pair",
                }
            )
    return selected


def select_panel_rows(
    exact_rows: Sequence[Mapping[str, Any]],
    regression_ids: Sequence[str],
    max_live_calls: int,
) -> list[JsonDict]:
    """Choose a bounded set that keeps known regressions load-bearing."""

    regression_set = set(regression_ids)
    ordered = [dict(row) for row in exact_rows if str(row.get("row_id") or "") in regression_set]
    ordered.extend(
        dict(row) for row in exact_rows if str(row.get("row_id") or "") not in regression_set
    )
    seen: set[str] = set()
    unique: list[JsonDict] = []
    for row in ordered:
        row_id = str(row.get("row_id") or "")
        if row_id and row_id not in seen:
            seen.add(row_id)
            unique.append(row)
    return unique[: max(0, int(max_live_calls))]


def normalize_panel_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Attach exact-authority context and replay hashes to live panel rows."""

    exact_by_id = {str(row.get("row_id") or ""): row for row in exact_rows}
    rows: list[JsonDict] = []
    for raw in raw_rows:
        row_id = str(raw.get("row_id") or "")
        exact = mapping(exact_by_id.get(row_id))
        prompt = str(raw.get("prompt") or f"verify {row_id}")
        output_text = str(raw.get("output_text") or raw.get("response") or "")
        prompt_hash = str(raw.get("prompt_hash") or stable_hash(prompt))
        response_hash = str(raw.get("response_hash") or stable_hash(output_text))
        decision = normalize_decision(raw.get("decision"))
        transcript_hash = str(
            raw.get("transcript_hash")
            or stable_hash(
                {
                    "model_id": raw.get("model_id"),
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "decision": decision,
                }
            )
        )
        rows.append(
            {
                "row_id": row_id,
                "model_id": str(raw.get("model_id") or ""),
                "model_path": str(raw.get("model_path") or ""),
                "exact_label": str(exact.get("exact_label") or raw.get("exact_label") or ""),
                "exact_authority_decision": str(
                    exact.get("exact_authority_decision") or decision_from_exact_row(exact or raw)
                ),
                "known_false_accept_regression": exact.get("known_false_accept_regression") is True,
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "transcript_hash": transcript_hash,
                "decision": decision,
                "output_text": output_text,
                "token_counts": mapping(raw.get("token_counts")),
            }
        )
    return rows


def normalize_decision(value: Any) -> str:
    """Normalize verifier decisions into accept/reject/abstain."""

    text = str(value or "").strip().lower()
    if text in {"accept", "reject", "abstain"}:
        return text
    label = str(value or "").strip().upper()
    if label in ACCEPT_LABELS:
        return "accept"
    if label in REJECT_LABELS:
        return "reject"
    return "abstain"


def score_panel(
    panel_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    regression_ids: Sequence[str],
) -> JsonDict:
    """Score decisions against exact authority only."""

    exact_by_id = {str(row.get("row_id") or ""): row for row in exact_rows}
    regression_set = set(regression_ids)
    reject_count = 0
    accept_count = 0
    false_accepts = 0
    false_rejects = 0
    abstentions = 0
    known_accepted: list[str] = []
    known_rejected: list[str] = []
    for row in panel_rows:
        row_id = str(row.get("row_id") or "")
        expected = decision_from_exact_row(exact_by_id.get(row_id) or row)
        decision = normalize_decision(row.get("decision"))
        if expected == "reject":
            reject_count += 1
            if decision == "accept":
                false_accepts += 1
        if expected == "accept":
            accept_count += 1
            if decision == "reject":
                false_rejects += 1
        if decision == "abstain":
            abstentions += 1
        if row_id in regression_set:
            if decision == "accept":
                known_accepted.append(row_id)
            elif decision == "reject":
                known_rejected.append(row_id)
    total = len(panel_rows)
    return {
        "false_accept_rate": rate(false_accepts, reject_count),
        "false_reject_rate": rate(false_rejects, accept_count),
        "abstention_rate": rate(abstentions, total),
        "false_accept_count": false_accepts,
        "false_reject_count": false_rejects,
        "abstention_count": abstentions,
        "known_false_accepts_accepted": sorted(known_accepted),
        "known_false_accepts_rejected": sorted(known_rejected),
    }


def rate(numerator: int, denominator: int) -> float:
    """Return a bounded rate with a zero denominator mapped to 0.0."""

    return 0.0 if denominator <= 0 else round(float(numerator) / float(denominator), 6)


def decision_from_exact_row(row: Mapping[str, Any]) -> str:
    """Convert exact labels and expected actions into the public decision space."""

    expected = str(row.get("expected_action") or "").lower()
    if expected in {"accept", "reject", "abstain"}:
        return expected
    label = str(row.get("exact_label") or "").strip().upper()
    if label in ACCEPT_LABELS:
        return "accept"
    if label in REJECT_LABELS:
        return "reject"
    return "abstain"


def aggregate_token_counts(panel_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate measured token counts from live panel rows."""

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in panel_rows:
        counts = mapping(row.get("token_counts"))
        for key in totals:
            totals[key] += safe_int(counts.get(key))
    totals["source"] = "live_panel_rows" if panel_rows else "gated_skip_no_live_calls"
    return totals


def preconditions_checked(
    exp3179: Mapping[str, Any],
    exp3180: Mapping[str, Any],
    receipt_valid: Mapping[str, Any],
) -> list[JsonDict]:
    """Expose load-bearing gates as machine-readable evidence."""

    return [
        {
            "name": "exp3179_clean_rerun_allowed",
            "passed": exp3179.get("clean_rerun_allowed") is True,
            "detail": f"clean_rerun_allowed={exp3179.get('clean_rerun_allowed')!r}",
        },
        {
            "name": "exp3180_controlled_invariance_passed",
            "passed": exp3180.get("controlled_invariance_passed") is True,
            "detail": (
                f"controlled_invariance_passed={exp3180.get('controlled_invariance_passed')!r}"
            ),
        },
        {
            "name": "transcript_receipt_validity",
            "passed": receipt_valid.get("valid") is True,
            "detail": ",".join(str(reason) for reason in receipt_valid.get("reasons") or []),
        },
    ]


def previous_v9_summary(exp3167: Mapping[str, Any]) -> JsonDict:
    """Carry forward the prior clean-rerun state without reusing its metrics."""

    return {
        "clean_live_verifier_rerun_v9_ready": exp3167.get("clean_live_verifier_rerun_v9_ready")
        is True,
        "gated_skip": exp3167.get("gated_skip") is True,
        "live_call_count": safe_int(exp3167.get("live_call_count")),
        "flagged_adversarial": exp3167.get("flagged_adversarial") is True,
        "headline_claim_allowed": exp3167.get("headline_claim_allowed") is True,
        "honest_verdict": str(exp3167.get("honest_verdict") or ""),
    }


def inference_substrate(
    *,
    gated_skip: bool,
    live_call_count: int,
    models: Sequence[Mapping[str, Any]],
    exp3179: Mapping[str, Any],
    controlled_passed: bool,
    reasons: Sequence[str],
) -> JsonDict:
    """Declare whether Exp 3181 itself executed models."""

    return {
        "kind": "clean_live_sota_verifier_rerun_v10_gated_skip"
        if gated_skip
        else "clean_live_sota_verifier_rerun_v10_live_panel",
        "downloads_models": False,
        "executes_models": not gated_skip,
        "executes_verifiers": not gated_skip,
        "executes_repairs": False,
        "executes_hardware": False,
        "live_model_calls": int(live_call_count),
        "legacy_small_model_used": False,
        "models_selected": [str(row.get("hf_id") or "") for row in models],
        "source_receipt_live_model_calls": safe_int(exp3179.get("live_call_count")),
        "receipt_clean_rerun_allowed": exp3179.get("clean_rerun_allowed") is True,
        "controlled_invariance_passed": controlled_passed,
        "gated_skip_reasons": list(reasons),
    }


def field_principles() -> JsonDict:
    """Echo the required field principles into the artifact."""

    return {
        "clean_live_sota_verifier_rerun_v10_ready": "downstream gates need a materialized artifact",
        "gated_skip": "skipped live work must be explicit",
        "gate_reasons": "blocked repair needs actionable cause",
        "live_call_count": "no-call artifacts must not imply live evidence",
        "models_used": "SOTA policy must be auditable",
        "proof_receipts_used": "live evidence must trace to receipts",
        "exact_row_count": "denominator must be explicit",
        "known_false_accept_regression_count": "adversarial rows must remain load-bearing",
        "false_accept_rate": "headline verifier quality must report failures",
        "false_reject_rate": "verifier rejection cost must be visible",
        "abstention_rate": "abstention can hide failure if unreported",
        "controlled_invariance_passed": "shortcut checks must gate trust",
        "flagged_adversarial": "repair cannot proceed under adversarial evidence",
        "headline_claim_allowed": "smoke or gated-skip evidence must not become headline",
        "inference_substrate": "live or gated-skip substrate must be declared",
        "honest_verdict": "terminal verdict must be explicit",
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that distinguishes live success from gate block."""

    if artifact.get("gated_skip") is not True and artifact.get("flagged_adversarial") is not True:
        return (
            "complete: clean_live_sota_verifier_rerun_v10_ready=true; "
            f"live_call_count={artifact.get('live_call_count')}; "
            f"false_accept_rate={artifact.get('false_accept_rate')}"
        )
    reasons = [str(reason) for reason in artifact.get("gate_reasons") or []]
    prefix = "blocked_precondition:"
    if any(reason.startswith("exp3179.") or reason.startswith("receipt_") for reason in reasons):
        prefix = "blocked_receipt_precondition:"
    elif any(reason.startswith("exp3180.") for reason in reasons):
        prefix = "blocked_invariance_precondition:"
    return (
        f"{prefix} clean_live_sota_verifier_rerun_v10_ready=true; "
        f"gated_skip={artifact.get('gated_skip')}; "
        f"live_call_count={artifact.get('live_call_count')}; "
        f"gate_reasons={len(reasons)}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject malformed Exp 3181 artifacts and accidental evidence promotion."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES + BLOCKED_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    substrate = mapping(artifact.get("inference_substrate"))
    if artifact.get("gated_skip") is True:
        if artifact.get("live_call_count") != 0:
            raise ValueError("gated skip must not claim live calls")
        if artifact.get("models_used"):
            raise ValueError("gated skip must not claim models used")
        if artifact.get("headline_claim_allowed") is not False:
            raise ValueError("gated skip must keep headline claims blocked")
        if substrate.get("executes_models") is not False:
            raise ValueError("gated skip must declare no model execution")
    if artifact.get("gated_skip") is not True and not artifact.get("models_used"):
        raise ValueError("live artifact must record models used")
    if (
        artifact.get("gated_skip") is not True
        and artifact.get("headline_claim_allowed") is True
        and artifact.get("flagged_adversarial") is True
    ):
        raise ValueError("clean live artifact must not stay flagged")


def safe_int(value: Any) -> int:
    """Return a JSON integer, falling back to zero on malformed evidence."""

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def mapping(value: Any) -> JsonDict:
    """Return mapping values as plain dictionaries."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only mapping rows from list-like JSON values."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]
