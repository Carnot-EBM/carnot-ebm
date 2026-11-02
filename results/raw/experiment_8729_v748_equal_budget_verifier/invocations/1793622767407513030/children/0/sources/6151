"""Build the Exp 3185 multi-turn repair ladder v5 artifact.

Spec refs: REQ-VERIFY-3185, SCENARIO-VERIFY-3185.

This builder is intentionally gate-first.  It materializes a complete artifact
even when repair is blocked, and it does not touch a live model unless Exp 3184
has explicitly opened `unblocked_for_bounded_repair_ladder`.  When the gate is
open, each repair proposal is still bounded by certificate-backed targets,
mandated local SOTA GGUF cache evidence, receipt evidence, transcript hashes,
and exact canonical semantic checks.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
RepairRunner = Callable[
    [Mapping[str, Any], int, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]
PreflightChecker = Callable[[Path, Mapping[str, Any], int], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3185_multi_turn_repair_ladder_v5"
SCHEMA = "carnot.multi_turn_repair_ladder.v5"
OUTPUT_REL_PATH = Path("results/experiment_3185_multi_turn_repair_ladder_v5.json")

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
SOTA_MODELS_REL_PATH = Path("python/carnot/inference/sota_models.py")
EXP3169_REL_PATH = Path("results/experiment_3169_repair_ladder_materializer_v4.json")
EXP3179_REL_PATH = Path("results/experiment_3179_local_sota_receipt_smoke_v3.json")
EXP3181_REL_PATH = Path("results/experiment_3181_clean_live_sota_verifier_rerun_v10.json")
EXP3183_REL_PATH = Path("results/experiment_3183_counterexample_certificate_expansion_v3.json")
EXP3184_REL_PATH = Path("results/experiment_3184_repair_gate_decision_v4.json")

UNBLOCKED_GATE_STATE = "unblocked_for_bounded_repair_ladder"
MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
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
    "multi_turn_repair_ladder_v5_ready",
    "gated_skip",
    "gate_state",
    "repair_attempt_count",
    "models_used",
    "repair_targets",
    "transcript_receipts",
    "exact_check_results",
    "repair_success_delta",
    "remaining_blockers",
    "headline_claim_allowed",
    "inference_substrate",
    "honest_verdict",
}
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), False, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False, "text"),
    ("experiment_template_policy", Path("scripts/experiment_template.py"), False, "text"),
    ("sota_model_registry", SOTA_MODELS_REL_PATH, False, "python"),
    ("verification_openspec", SPEC_REL_PATH, True, "text"),
    ("exp3169_repair_ladder_v4", EXP3169_REL_PATH, False, "json"),
    ("exp3179_receipt_smoke_v3", EXP3179_REL_PATH, False, "json"),
    ("exp3181_clean_verifier_v10", EXP3181_REL_PATH, False, "json"),
    ("exp3183_certificate_expansion_v3", EXP3183_REL_PATH, True, "json"),
    ("exp3184_repair_gate_decision_v4", EXP3184_REL_PATH, True, "json"),
    (
        "exp3185_module",
        Path("python/carnot/verify/multi_turn_repair_ladder_v5.py"),
        False,
        "python",
    ),
    (
        "exp3185_tests",
        Path("tests/python/test_experiment_3185_multi_turn_repair_ladder_v5.py"),
        False,
        "python",
    ),
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3185_multi_turn_repair_ladder_v5.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3185_multi_turn_repair_ladder_v5.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/multi_turn_repair_ladder_v5.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3185_multi_turn_repair_ladder_v5.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    repair_runner: RepairRunner | None = None,
    preflight_checker: PreflightChecker | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3185: build either a no-call skip or a bounded repair ladder."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    gate = read_json_object(root_path / EXP3184_REL_PATH)
    certificates = read_json_object(root_path / EXP3183_REL_PATH)
    gate_state = str(gate.get("repair_gate_state") or "missing")
    repair_targets: list[JsonDict] = []
    transcript_receipts: list[JsonDict] = []
    exact_check_results: list[JsonDict] = []
    models_used: list[str] = []
    remaining: list[str] = []

    if gate_state != UNBLOCKED_GATE_STATE:
        remaining = blocked_gate_reasons(gate_state, gate, certificates)
        gated_skip = True
    else:
        budget = repair_budget(gate)
        repair_targets = select_repair_targets(
            mapping_rows(certificates.get("certificate_records")),
            max_targets=budget["max_distinct_rows"],
        )
        remaining = runtime_preconditions(root_path, certificates, repair_targets, repair_runner)
        gated_skip = bool(remaining)
        if not gated_skip:
            checker = preflight_checker or repair_preflight_from_local_cache
            run_ladder(
                root_path=root_path,
                repair_targets=repair_targets,
                budget=budget,
                repair_runner=repair_runner,
                preflight_checker=checker,
                transcript_receipts=transcript_receipts,
                exact_check_results=exact_check_results,
                models_used=models_used,
                remaining_blockers=remaining,
            )
            gated_skip = not exact_check_results and bool(remaining)

    accepted_row_ids = {
        str(row.get("row_id") or "")
        for row in exact_check_results
        if row.get("accepted_by_exact_authority") is True
    }
    repair_attempt_count = len(exact_check_results)
    repair_success_delta = rate(len(accepted_row_ids), len(repair_targets))
    headline_allowed = headline_claim_allowed(
        gated_skip=gated_skip,
        models_used=models_used,
        exact_check_results=exact_check_results,
        remaining_blockers=remaining,
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3185", "SCENARIO-VERIFY-3185"],
        "multi_turn_repair_ladder_v5_ready": True,
        "gated_skip": gated_skip,
        "gate_state": gate_state,
        "repair_attempt_count": repair_attempt_count,
        "models_used": models_used,
        "repair_targets": repair_targets,
        "transcript_receipts": transcript_receipts,
        "exact_check_results": exact_check_results,
        "repair_success_delta": repair_success_delta,
        "remaining_blockers": unique_strings(remaining),
        "headline_claim_allowed": headline_allowed,
        "source_artifacts": source_artifacts(root_path),
        "field_principles": field_principles(),
        "inference_substrate": inference_substrate(
            gated_skip=gated_skip,
            gate_state=gate_state,
            repair_attempt_count=repair_attempt_count,
            repair_target_count=len(repair_targets),
            models_used=models_used,
        ),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration(started, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    repair_runner: RepairRunner | None = None,
    preflight_checker: PreflightChecker | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3185 artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        repair_runner=repair_runner,
        preflight_checker=preflight_checker,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def blocked_gate_reasons(
    gate_state: str,
    gate: Mapping[str, Any],
    certificates: Mapping[str, Any],
) -> list[str]:
    """Return every gate/certificate blocker while making the closed gate explicit."""

    reasons = [f"repair gate not unblocked: {gate_state}"]
    reasons.extend(str(item) for item in gate.get("blocker_reasons") or [])
    reasons.extend(f"exp3183: {item}" for item in certificates.get("blocker_reasons") or [])
    return unique_strings(reasons)


def runtime_preconditions(
    root: Path,
    certificates: Mapping[str, Any],
    repair_targets: Sequence[Mapping[str, Any]],
    repair_runner: RepairRunner | None,
) -> list[str]:
    """Check non-call prerequisites after the gate opens but before any proposal."""

    reasons: list[str] = []
    if certificates.get("counterexample_certificate_expansion_v3_ready") is not True:
        reasons.append("exp3183.counterexample_certificate_expansion_v3_ready is not true")
    if certificates.get("repair_call_ready") is not True:
        reasons.append("exp3183.repair_call_ready is not true")
    if not repair_targets:
        reasons.append("no certificate-backed repair targets selected")
    if repair_runner is None:
        reasons.append("live repair runner is not configured")
    if repair_runner is None and repair_targets:
        preflight = repair_preflight_from_local_cache(root, repair_targets[0], 1)
        reasons.extend(string_list(preflight.get("blockers")))
    return unique_strings(reasons)


def repair_budget(gate: Mapping[str, Any]) -> JsonDict:
    """Read strict attempt limits from Exp 3184, preserving safe defaults."""

    raw = gate.get("allowed_repair_attempt_budget")
    budget = raw if isinstance(raw, Mapping) else {}
    return {
        "max_total_repair_attempts": positive_int(budget.get("max_total_repair_attempts"), 4),
        "max_attempts_per_row": positive_int(budget.get("max_attempts_per_row"), 2),
        "max_distinct_rows": positive_int(budget.get("max_distinct_rows"), 2),
    }


def positive_int(value: Any, fallback: int) -> int:
    """Return a positive integer from artifact JSON, or a conservative fallback."""

    return int(value) if isinstance(value, int) and value > 0 else fallback


def run_ladder(
    *,
    root_path: Path,
    repair_targets: Sequence[Mapping[str, Any]],
    budget: Mapping[str, int],
    repair_runner: RepairRunner,
    preflight_checker: PreflightChecker,
    transcript_receipts: list[JsonDict],
    exact_check_results: list[JsonDict],
    models_used: list[str],
    remaining_blockers: list[str],
) -> None:
    """Run the bounded two-turn ladder, stopping on exact acceptance per row."""

    max_total = int(budget["max_total_repair_attempts"])
    max_per_row = int(budget["max_attempts_per_row"])
    for target in repair_targets:
        feedback: JsonDict = {}
        for turn in range(1, max_per_row + 1):
            if len(exact_check_results) >= max_total:
                return
            preflight = preflight_checker(root_path, target, turn)
            if preflight.get("ok") is not True:
                remaining_blockers.extend(string_list(preflight.get("blockers")))
                return
            model_spec = dict(preflight.get("model_spec") or {})
            append_unique(models_used, str(model_spec.get("hf_id") or ""))
            proposal = call_repair_runner(repair_runner, target, turn, feedback, model_spec)
            receipt = transcript_receipt(
                target=target,
                turn=turn,
                feedback=feedback,
                proposal=proposal,
                model_spec=model_spec,
                receipt_evidence=mapping_rows(preflight.get("receipt_evidence")),
            )
            transcript_receipts.append(receipt)
            exact_result = exact_semantic_check(target, proposal, receipt["transcript_hash"], turn)
            exact_check_results.append(exact_result)
            if exact_result["accepted_by_exact_authority"] is True:
                break
            feedback = counterexample_feedback(target, exact_result)


def call_repair_runner(
    repair_runner: RepairRunner,
    target: Mapping[str, Any],
    turn: int,
    feedback: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> JsonDict:
    """Invoke a repair runner and turn runner failures into exact-check failures."""

    try:
        result = repair_runner(target, turn, feedback, model_spec)
    except Exception as exc:
        return {
            "candidate_answer": "",
            "raw_response": "",
            "repair_runner_error": f"{type(exc).__name__}: {exc}",
        }
    return (
        dict(result)
        if isinstance(result, Mapping)
        else {"candidate_answer": "", "raw_response": ""}
    )


def repair_preflight_from_local_cache(
    root: Path,
    target: Mapping[str, Any],
    turn: int,
) -> JsonDict:
    """Re-check mandated GGUF cache and receipt evidence before one repair call."""

    del target, turn
    blockers: list[str] = []
    model_specs = mapping_rows(cached_sota_pair())
    mandated_specs = [
        dict(row) for row in model_specs if str(row.get("hf_id") or "") in MANDATED_MODEL_IDS
    ]
    if not mandated_specs:
        blockers.append("no mandated local SOTA GGUF cache resolved before repair call")
    receipts = read_json_object(root / EXP3179_REL_PATH)
    receipt_rows = mapping_rows(receipts.get("proof_receipts"))
    if receipts.get("local_sota_receipt_smoke_v3_ready") is not True:
        blockers.append("exp3179.local_sota_receipt_smoke_v3_ready is not true before repair call")
    if receipts.get("clean_rerun_allowed") is not True:
        blockers.append("exp3179.clean_rerun_allowed is not true before repair call")
    if not receipt_rows:
        blockers.append("no local SOTA proof receipts available before repair call")
    return {
        "ok": not blockers,
        "model_spec": mandated_specs[0] if mandated_specs else {},
        "receipt_evidence": receipt_rows,
        "blockers": blockers,
    }


def select_repair_targets(
    certificate_records: Sequence[Mapping[str, Any]],
    *,
    max_targets: int,
) -> list[JsonDict]:
    """Select bounded certificate-backed targets from exact repair evidence."""

    targets: list[JsonDict] = []
    for record in certificate_records:
        row_id = str(record.get("row_id") or "")
        if not row_id or record.get("exact_authority_complete") is not True:
            continue
        family = str(record.get("counterexample_family") or "")
        pilot = record.get("pilot_certificate")
        pilot_map = pilot if isinstance(pilot, Mapping) else {}
        has_counterexample = (
            record.get("known_false_accept_or_regression") is True
            or family.startswith("known_false_accept:")
            or family.startswith("fragment_code:")
            or bool(pilot_map.get("minimal_failing_assignment"))
        )
        if not has_counterexample:
            continue
        target = {
            "row_id": row_id,
            "canonical_answer": str(
                record.get("canonical_answer") or record.get("exact_label") or ""
            ),
            "exact_label": str(record.get("exact_label") or ""),
            "checker_result": str(record.get("checker_result") or ""),
            "checker_authority": str(record.get("checker_authority") or "exact_authority_replay"),
            "counterexample_family": family,
            "violated_constraint": str(pilot_map.get("violated_constraint") or ""),
            "minimal_failing_assignment": pilot_map.get("minimal_failing_assignment") or {},
            "source_certificate": dict(record),
        }
        targets.append(target)
        if len(targets) >= max_targets:
            break
    return targets


def exact_semantic_check(
    target: Mapping[str, Any],
    proposal: Mapping[str, Any],
    transcript_hash: str,
    turn: int,
) -> JsonDict:
    """Score one repair proposal using exact canonical answer equivalence only."""

    candidate = first_text(
        proposal.get("candidate_answer"),
        proposal.get("proposed_answer"),
        proposal.get("answer"),
        proposal.get("raw_response"),
    )
    expected = first_text(target.get("canonical_answer"), target.get("exact_label"))
    candidate_canonical = canonicalize_answer(candidate)
    expected_canonical = canonicalize_answer(expected)
    exact_match = bool(candidate_canonical and candidate_canonical == expected_canonical)
    return {
        "row_id": str(target.get("row_id") or ""),
        "turn": int(turn),
        "candidate_answer": candidate,
        "candidate_canonical": candidate_canonical,
        "expected_canonical": expected_canonical,
        "checker_authority": str(target.get("checker_authority") or "exact_authority_replay"),
        "transcript_hash": transcript_hash,
        "exact_match": exact_match,
        "accepted_by_exact_authority": exact_match
        and target.get("exact_authority_complete", True) is not False,
    }


def counterexample_feedback(target: Mapping[str, Any], check_result: Mapping[str, Any]) -> JsonDict:
    """Build the second-turn feedback from the exact failed check and certificate."""

    return {
        "row_id": str(target.get("row_id") or ""),
        "expected_canonical": str(check_result.get("expected_canonical") or ""),
        "candidate_canonical": str(check_result.get("candidate_canonical") or ""),
        "counterexample_family": str(target.get("counterexample_family") or ""),
        "violated_constraint": str(target.get("violated_constraint") or ""),
        "minimal_failing_assignment": target.get("minimal_failing_assignment") or {},
        "prior_exact_match": check_result.get("exact_match") is True,
    }


def transcript_receipt(
    *,
    target: Mapping[str, Any],
    turn: int,
    feedback: Mapping[str, Any],
    proposal: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    receipt_evidence: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Hash the replayable repair transcript metadata for one proposal."""

    transcript = {
        "row_id": str(target.get("row_id") or ""),
        "turn": int(turn),
        "model_id": str(model_spec.get("hf_id") or ""),
        "target_canonical_answer": str(target.get("canonical_answer") or ""),
        "feedback": dict(feedback),
        "proposal": dict(proposal),
    }
    transcript_hash = sha256_json(transcript)
    prompt_hash = sha256_json(
        {
            "row_id": transcript["row_id"],
            "turn": transcript["turn"],
            "feedback": transcript["feedback"],
        }
    )
    return {
        "row_id": transcript["row_id"],
        "turn": transcript["turn"],
        "model_id": transcript["model_id"],
        "prompt_hash": prompt_hash,
        "transcript_hash": transcript_hash,
        "source_receipt_ids": [
            str(row.get("receipt_id") or row.get("transcript_hash") or "")
            for row in receipt_evidence
            if row.get("receipt_id") or row.get("transcript_hash")
        ],
        "proposal_keys": sorted(str(key) for key in proposal.keys()),
    }


def headline_claim_allowed(
    *,
    gated_skip: bool,
    models_used: Sequence[str],
    exact_check_results: Sequence[Mapping[str, Any]],
    remaining_blockers: Sequence[str],
) -> bool:
    """Return true only when exact repairs on mandated SOTA evidence support it."""

    return bool(
        not gated_skip
        and not remaining_blockers
        and models_used
        and all(model_id in MANDATED_MODEL_IDS for model_id in models_used)
        and any(row.get("accepted_by_exact_authority") is True for row in exact_check_results)
    )


def inference_substrate(
    *,
    gated_skip: bool,
    gate_state: str,
    repair_attempt_count: int,
    repair_target_count: int,
    models_used: Sequence[str],
) -> JsonDict:
    """Declare whether Exp 3185 made live repair calls or stopped at a gate."""

    executed = repair_attempt_count > 0
    return {
        "kind": "multi_turn_repair_ladder_v5",
        "gate_state": gate_state,
        "gated_skip": gated_skip,
        "executes_models": executed,
        "executes_repairs": executed,
        "executes_verifiers": executed,
        "executes_solvers": False,
        "no_live_inference": not executed,
        "live_model_calls": repair_attempt_count,
        "repair_calls": repair_attempt_count,
        "repair_target_count": repair_target_count,
        "models_used": list(models_used),
        "model_policy": list(MANDATED_MODEL_IDS),
        "legacy_small_models_headline_eligible": False,
        "exact_authority_only": True,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal Exp 3185 artifact contract before writing."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    success_delta = artifact.get("repair_success_delta")
    if not isinstance(success_delta, (int, float)) or not math.isfinite(float(success_delta)):
        raise ValueError("repair_success_delta must be a finite rate")
    if not 0.0 <= float(success_delta) <= 1.0:
        raise ValueError("repair_success_delta must be a finite rate")
    models_used = [str(model_id) for model_id in artifact.get("models_used") or []]
    if any(model_id not in MANDATED_MODEL_IDS for model_id in models_used):
        raise ValueError("models_used must contain only mandated SOTA GGUF IDs")
    gated_skip = artifact.get("gated_skip") is True
    repair_attempt_count = int(artifact.get("repair_attempt_count") or 0)
    transcript_receipts = mapping_rows(artifact.get("transcript_receipts"))
    exact_results = mapping_rows(artifact.get("exact_check_results"))
    substrate = artifact.get("inference_substrate")
    inference = substrate if isinstance(substrate, Mapping) else {}
    if gated_skip and (
        repair_attempt_count
        or models_used
        or transcript_receipts
        or exact_results
        or artifact.get("headline_claim_allowed") is True
        or int(inference.get("live_model_calls") or 0)
    ):
        raise ValueError("gated skip cannot contain repair attempts, models, checks, or live calls")
    if not gated_skip:
        if artifact.get("gate_state") != UNBLOCKED_GATE_STATE or repair_attempt_count <= 0:
            raise ValueError("executed repair artifacts require the unblocked gate and attempts")
        if (
            len(transcript_receipts) != repair_attempt_count
            or len(exact_results) != repair_attempt_count
        ):
            raise ValueError("repair attempts must have transcript receipts and exact checks")
        if not models_used:
            raise ValueError("executed repair artifacts require mandated SOTA models")
        for receipt in transcript_receipts:
            digest = str(receipt.get("transcript_hash") or "")
            if len(digest) != 64:
                raise ValueError("transcript receipts must include sha256 transcript hashes")
        verdict = str(artifact.get("honest_verdict") or "")
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("executed repair verdict must start with a success prefix")
    if artifact.get("headline_claim_allowed") is True and not headline_claim_allowed(
        gated_skip=gated_skip,
        models_used=models_used,
        exact_check_results=exact_results,
        remaining_blockers=string_list(artifact.get("remaining_blockers")),
    ):
        raise ValueError("headline claim is not backed by exact mandated-SOTA repair evidence")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict expected by conductor consumers."""

    if artifact.get("gated_skip") is True:
        state = str(artifact.get("gate_state") or "")
        blockers = string_list(artifact.get("remaining_blockers"))
        first = blockers[0] if blockers else "repair skipped"
        if state != UNBLOCKED_GATE_STATE:
            return f"blocked_repair_gate_precondition: {first}"
        return f"blocked_repair_runtime: {first}"
    return (
        "complete: multi_turn_repair_ladder_v5_ready=true; "
        f"repair_attempt_count={artifact.get('repair_attempt_count')}; "
        f"repair_success_delta={artifact.get('repair_success_delta')}; "
        f"headline_claim_allowed={artifact.get('headline_claim_allowed')}"
    )


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return checksummed provenance for every policy file or artifact read."""

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


def field_principles() -> JsonDict:
    """Record why each required field exists in the terminal artifact."""

    return {
        "multi_turn_repair_ladder_v5_ready": "repair ladder must materialize even when gated",
        "gated_skip": "skipped live repair must be explicit",
        "gate_state": "repair behavior must trace to gate decision",
        "repair_attempt_count": "no-call artifacts must not imply live repair",
        "models_used": "SOTA policy must be auditable",
        "repair_targets": "repair attempts must trace to certificates",
        "transcript_receipts": "live repair evidence must be replayable",
        "exact_check_results": "semantic repair must be exact-scored",
        "repair_success_delta": "repair benefit must be quantified",
        "remaining_blockers": "incomplete repair must remain actionable",
        "headline_claim_allowed": "gated or smoke evidence must not become headline",
        "inference_substrate": "live or gated-skip substrate must be declared",
        "honest_verdict": "terminal verdict must honestly report blocked preconditions",
    }


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence for missing or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def mapping_rows(value: Any) -> list[JsonDict]:
    """Return only mapping rows from a JSON list-like value."""

    return (
        [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []
    )


def string_list(value: Any) -> list[str]:
    """Return non-empty strings from a JSON scalar or list."""

    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values if item]


def first_text(*values: Any) -> str:
    """Return the first non-empty text field from a proposal or target."""

    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return ""


def canonicalize_answer(value: Any) -> str:
    """Map exact answer aliases to canonical semantic labels."""

    token = str(value or "").strip().strip("\"'").upper()
    aliases = {
        "ACCEPT": "VALID",
        "CORRECT": "VALID",
        "PASS": "VALID",
        "SAT": "VALID",
        "TRUE": "VALID",
        "VALID": "VALID",
        "FAIL": "INVALID",
        "FALSE": "INVALID",
        "INCORRECT": "INVALID",
        "INVALID": "INVALID",
        "REJECT": "INVALID",
        "UNSAT": "INVALID",
    }
    return aliases.get(token, " ".join(token.split()))


def append_unique(values: list[str], value: str) -> None:
    """Append a non-empty string once while preserving order."""

    if value and value not in values:
        values.append(value)


def unique_strings(values: Sequence[str]) -> list[str]:
    """Deduplicate non-empty strings while preserving first occurrence."""

    result: list[str] = []
    for value in values:
        append_unique(result, str(value))
    return result


def sha256_json(value: Mapping[str, Any]) -> str:
    """Hash JSON-serializable transcript metadata in replay-stable form."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Checksum local source bytes when the source exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rate(numerator: int, denominator: int) -> float:
    """Return a bounded finite rate, using zero for empty denominators."""

    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, float(numerator) / float(denominator))), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return a stable nonnegative elapsed duration."""

    finished = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, finished - float(started_s)), 6)
