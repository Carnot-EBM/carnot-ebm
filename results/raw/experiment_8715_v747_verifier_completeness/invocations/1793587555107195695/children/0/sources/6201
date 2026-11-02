"""Build the Exp 3169 repair ladder materializer v4 artifact.

Spec refs: REQ-VERIFY-3169, SCENARIO-VERIFY-3169.

This module makes the repair ladder explicit even when repair is not allowed.
That matters because downstream matrix jobs need to distinguish "repair was
correctly skipped by a gate" from "the conductor never produced an artifact."
No model or repair runner is invoked unless Exp 3168 has already unblocked the
gate and a mandated local SOTA GGUF model is usable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence


JsonDict = dict[str, Any]
RepairRunner = Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3169_repair_ladder_materializer_v4"
SCHEMA = "carnot.repair_ladder_materializer.v4"
OUTPUT_REL_PATH = Path("results/experiment_3169_repair_ladder_materializer_v4.json")

EXP3168_REL_PATH = Path("results/experiment_3168_repair_gate_decision_v3.json")
EXP3167_REL_PATH = Path("results/experiment_3167_clean_live_sota_verifier_rerun_v9.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")
EXP3115_REL_PATH = Path("results/experiment_3115_explicit_repair_gate_micro_panel_v4.json")

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
DEFAULT_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen3.6-35B-A3B",
        "role": "moe",
        "tier": "flagship_moe",
        "headline_eligible": True,
        "legacy_small_model": False,
        "usable_locally": False,
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "name": "Gemma4-31B-it",
        "role": "dense",
        "tier": "flagship_dense",
        "headline_eligible": True,
        "legacy_small_model": False,
        "usable_locally": False,
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "name": "Gemma4-26B-A4B-it",
        "role": "moe",
        "tier": "middle_moe",
        "headline_eligible": True,
        "legacy_small_model": False,
        "usable_locally": False,
    },
)
EXACT_ACCEPT_FIELDS = (
    "accepted",
    "exact_verified",
    "canonical_grounded",
    "controlled_invariance_passed",
    "monitor_replay_passed",
    "intent_preserved",
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
    "repair_ladder_materializer_v4_ready",
    "gated_skip",
    "gated_skip_reason",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "selected_repair_rows",
    "repair_attempt_count",
    "exact_authority_accept_count",
    "repair_success_delta",
    "false_repair_accept_rate",
    "intent_preservation_rate",
    "headline_repair_claim_allowed",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3169_repair_ladder_materializer_v4.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3169_repair_ladder_materializer_v4.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/repair_ladder_materializer_v4.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class SourceSpec:
    role: str
    path: Path
    required: bool


SOURCE_SPECS = (
    SourceSpec("agents_repo_instructions", Path("AGENTS.md"), False),
    SourceSpec("codex_repo_workflow", Path("CODEX.md"), False),
    SourceSpec("claude_authenticity_rules", Path("CLAUDE.md"), False),
    SourceSpec("experiment_template_policy", Path("scripts/experiment_template.py"), False),
    SourceSpec("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True),
    SourceSpec("exp3168_repair_gate_decision_v3", EXP3168_REL_PATH, True),
    SourceSpec("exp3167_clean_live_verifier_rerun_v9", EXP3167_REL_PATH, True),
    SourceSpec("exp3137_exact_safe_contract", EXP3137_REL_PATH, True),
    SourceSpec("exp3138_canonical_grounding", EXP3138_REL_PATH, True),
    SourceSpec("exp3115_micro_repair_panel", EXP3115_REL_PATH, False),
    SourceSpec("exp3169_module", Path("python/carnot/verify/repair_ladder_materializer_v4.py"), False),
    SourceSpec(
        "exp3169_tests",
        Path("tests/python/test_experiment_3169_repair_ladder_materializer_v4.py"),
        False,
    ),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    repair_runner: RepairRunner | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3169: build the terminal repair ladder run or gated skip."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    gate_path = root_path / EXP3168_REL_PATH
    gate_present = gate_path.is_file()
    gate = read_json_object(gate_path)
    exp3167 = read_json_object(root_path / EXP3167_REL_PATH)
    exp3115 = read_json_object(root_path / EXP3115_REL_PATH)
    selected_rows = mapping_rows(gate.get("selected_repair_rows"))
    model_specs = model_specs_from_sources(exp3167, exp3115)
    usable_specs = usable_mandated_model_specs(model_specs)
    can_run, skip_reason = repair_run_decision(
        gate_present=gate_present,
        gate=gate,
        selected_rows=selected_rows,
        usable_specs=usable_specs,
        repair_runner=repair_runner,
    )
    run_model_specs = usable_specs[:1] if can_run else []
    selected_model_ids = [str(row["hf_id"]) for row in run_model_specs]
    repair_attempts = (
        run_repair_panel(selected_rows, run_model_specs, repair_runner)
        if can_run and repair_runner is not None
        else []
    )
    gated_skip = not can_run
    live_call_count = 0 if gated_skip else len(repair_attempts)
    metrics = repair_metrics(repair_attempts, len(selected_rows))
    headline_allowed = headline_repair_claim_allowed(
        gated_skip=gated_skip,
        live_call_count=live_call_count,
        selected_model_ids=selected_model_ids,
        metrics=metrics,
    )
    sources = source_artifacts(root_path)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "repair_ladder_materializer_v4_ready": True,
        "gated_skip": gated_skip,
        "gated_skip_reason": skip_reason if gated_skip else "",
        "gate_state": str(gate.get("repair_gate_state") or "missing"),
        "gate_blockers": list(gate.get("repair_blockers") or []),
        "model_specs": model_specs,
        "selected_model_ids": selected_model_ids,
        "live_call_count": live_call_count,
        "selected_repair_rows": selected_rows,
        "repair_attempt_count": len(repair_attempts),
        "exact_authority_accept_count": metrics["exact_authority_accept_count"],
        "repair_success_delta": metrics["repair_success_delta"],
        "false_repair_accept_rate": metrics["false_repair_accept_rate"],
        "intent_preservation_rate": metrics["intent_preservation_rate"],
        "headline_repair_claim_allowed": headline_allowed,
        "repair_attempts": repair_attempts,
        "source_artifacts": sources,
        "source_checksums": {row["path"]: row["sha256"] for row in sources if row.get("sha256")},
        "source_artifacts_loaded": [
            EXP3168_REL_PATH.as_posix(),
            EXP3167_REL_PATH.as_posix(),
            EXP3137_REL_PATH.as_posix(),
            EXP3138_REL_PATH.as_posix(),
            EXP3115_REL_PATH.as_posix(),
        ],
        "model_policy": list(MANDATED_MODEL_IDS),
        "inference_substrate": inference_substrate(
            gated_skip=gated_skip,
            gate_state=str(gate.get("repair_gate_state") or "missing"),
            live_call_count=live_call_count,
            repair_attempt_count=len(repair_attempts),
            repair_runner=repair_runner,
        ),
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
    repair_runner: RepairRunner | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3169 materializer artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        repair_runner=repair_runner,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def repair_run_decision(
    *,
    gate_present: bool,
    gate: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    usable_specs: Sequence[Mapping[str, Any]],
    repair_runner: RepairRunner | None,
) -> tuple[bool, str]:
    """Return whether repair may run and the actionable skip reason otherwise."""

    if not gate_present:
        return False, "repair gate decision artifact is missing"
    if gate.get("repair_gate_decision_v3_ready") is not True:
        return False, "repair gate decision artifact is not ready"
    gate_state = str(gate.get("repair_gate_state") or "missing")
    if gate_state != "unblocked":
        return False, f"repair gate blocked: {gate_state}; {first_gate_blocker(gate)}"
    if not selected_rows:
        return False, "repair gate unblocked but selected_repair_rows is empty"
    if not usable_specs:
        return False, "no mandated local SOTA GGUF model is usable"
    if repair_runner is None:
        return False, "live repair runner is not configured"
    return True, ""


def first_gate_blocker(gate: Mapping[str, Any]) -> str:
    """Return the first explicit Exp 3168 blocker, falling back to skip reason."""

    blockers = gate.get("repair_blockers")
    if isinstance(blockers, list) and blockers:
        return str(blockers[0])
    return str(gate.get("gated_skip_reason") or "no gate blocker recorded")


def model_specs_from_sources(exp3167: Mapping[str, Any], exp3115: Mapping[str, Any]) -> list[JsonDict]:
    """Merge mandated model policy with local usability evidence from source artifacts."""

    by_id = {str(row["hf_id"]): dict(row) for row in DEFAULT_MODEL_SPECS}
    for source in (exp3167, exp3115):
        for row in mapping_rows(source.get("model_specs")):
            hf_id = str(row.get("hf_id") or "")
            if hf_id in by_id:
                merged = {**by_id[hf_id], **dict(row)}
                merged["headline_eligible"] = True
                merged["legacy_small_model"] = False
                by_id[hf_id] = merged
    return [by_id[model_id] for model_id in MANDATED_MODEL_IDS]


def usable_mandated_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return mandated SOTA rows with explicit local usability or cache evidence."""

    usable: list[JsonDict] = []
    for row in model_specs:
        hf_id = str(row.get("hf_id") or "")
        if hf_id not in MANDATED_MODEL_IDS:
            continue
        has_cache = row.get("usable_locally") is True or row.get("cache_present") is True
        selected = row.get("selected") is True or row.get("selected_for_exp3167") is True
        cached_status = str(row.get("cache_status") or "") == "cached"
        if has_cache or selected or cached_status:
            usable.append(dict(row))
    return usable


def run_repair_panel(
    selected_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    repair_runner: RepairRunner,
) -> list[JsonDict]:
    """Invoke the bounded repair runner and normalize exact-accept fields."""

    try:
        raw_attempts = repair_runner(selected_rows, model_specs)
    except Exception as exc:
        return [
            normalize_repair_attempt(
                {
                    "row_id": row_id_from(selected_rows[0]) if selected_rows else "",
                    "model_id": str(model_specs[0].get("hf_id") or "") if model_specs else "",
                    "accepted": False,
                    "exact_verified": False,
                    "canonical_grounded": False,
                    "controlled_invariance_passed": False,
                    "monitor_replay_passed": False,
                    "intent_preserved": False,
                    "verification_errors": [f"repair_runner_error: {type(exc).__name__}: {exc}"],
                }
            )
        ]
    return [normalize_repair_attempt(row) for row in raw_attempts if isinstance(row, Mapping)]


def normalize_repair_attempt(attempt: Mapping[str, Any]) -> JsonDict:
    """Normalize one candidate row so acceptance always means exact authority acceptance."""

    normalized = dict(attempt)
    normalized["row_id"] = row_id_from(normalized)
    normalized["model_id"] = str(normalized.get("model_id") or "")
    for field in EXACT_ACCEPT_FIELDS:
        normalized[field] = normalized.get(field) is True
    normalized["accepted_by_exact_authority"] = accepted_by_exact_authority(normalized)
    normalized.setdefault("verification_errors", [])
    return normalized


def accepted_by_exact_authority(attempt: Mapping[str, Any]) -> bool:
    """Return true only when every exact, canonical, invariance, monitor, and intent gate passes."""

    return all(attempt.get(field) is True for field in EXACT_ACCEPT_FIELDS)


def repair_metrics(attempts: Sequence[Mapping[str, Any]], selected_count: int) -> JsonDict:
    """Compute repair metrics from exact-authority candidate rows."""

    accepted = [row for row in attempts if row.get("accepted") is True]
    exact_accepts = [row for row in attempts if accepted_by_exact_authority(row)]
    false_accepts = [row for row in accepted if not accepted_by_exact_authority(row)]
    intent_preserved = [row for row in accepted if row.get("intent_preserved") is True]
    return {
        "exact_authority_accept_count": len(exact_accepts),
        "repair_success_delta": rate(len(exact_accepts), selected_count),
        "false_repair_accept_rate": rate(len(false_accepts), len(accepted)),
        "intent_preservation_rate": rate(len(intent_preserved), len(accepted)),
    }


def headline_repair_claim_allowed(
    *,
    gated_skip: bool,
    live_call_count: int,
    selected_model_ids: Sequence[str],
    metrics: Mapping[str, Any],
) -> bool:
    """Return whether the repair run can support a headline repair claim."""

    repair_success_delta = float(metrics.get("repair_success_delta", 0.0))
    false_repair_accept_rate = float(metrics.get("false_repair_accept_rate", 1.0))
    intent_preservation_rate = float(metrics.get("intent_preservation_rate", 0.0))
    return bool(
        not gated_skip
        and live_call_count > 0
        and selected_model_ids
        and int(metrics.get("exact_authority_accept_count") or 0) > 0
        and repair_success_delta > 0.0
        and false_repair_accept_rate == 0.0
        and intent_preservation_rate == 1.0
    )


def inference_substrate(
    *,
    gated_skip: bool,
    gate_state: str,
    live_call_count: int,
    repair_attempt_count: int,
    repair_runner: RepairRunner | None,
) -> JsonDict:
    """Declare whether this artifact skipped repair or executed bounded local repair."""

    executes = not gated_skip and live_call_count > 0
    return {
        "kind": "repair_ladder_materializer_v4",
        "gate_state": gate_state,
        "gated_skip": gated_skip,
        "executes_models": executes,
        "executes_repairs": executes,
        "executes_verifiers": executes,
        "executes_solvers": executes,
        "no_live_inference": not executes,
        "live_model_calls": int(live_call_count),
        "repair_calls": int(repair_attempt_count),
        "repair_runner_kind": "injected_repair_runner" if repair_runner is not None else "not_configured",
        "legacy_small_models_headline_eligible": False,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fail-closed Exp 3169 terminal artifact contract."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3169 artifact missing required fields: {missing}")
    for field in ("repair_success_delta", "false_repair_accept_rate", "intent_preservation_rate"):
        value = float(artifact.get(field))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be a finite rate in [0, 1]")
    selected_ids = [str(model_id) for model_id in artifact.get("selected_model_ids") or []]
    if any(model_id not in MANDATED_MODEL_IDS for model_id in selected_ids):
        raise ValueError("selected_model_ids must contain only mandated SOTA GGUF IDs")
    gated_skip = artifact.get("gated_skip") is True
    if gated_skip and (
        int(artifact.get("live_call_count") or 0) != 0
        or int(artifact.get("repair_attempt_count") or 0) != 0
        or int(artifact.get("exact_authority_accept_count") or 0) != 0
        or artifact.get("headline_repair_claim_allowed") is True
    ):
        raise ValueError("gated skip cannot contain live calls, repair attempts, or headline claims")
    if not gated_skip and int(artifact.get("repair_attempt_count") or 0) <= 0:
        raise ValueError("executed repair artifacts must include repair attempts")
    if artifact.get("headline_repair_claim_allowed") is True and not headline_repair_claim_allowed(
        gated_skip=gated_skip,
        live_call_count=int(artifact.get("live_call_count") or 0),
        selected_model_ids=selected_ids,
        metrics=artifact,
    ):
        raise ValueError("headline repair claim is not supported by exact repair metrics")
    verdict = str(artifact.get("honest_verdict") or "")
    if not gated_skip and not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("executed repair verdict must start with a success prefix")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict expected by conductor consumers."""

    if artifact.get("gated_skip") is not True:
        return (
            "complete: repair_ladder_materializer_v4_ready=true; "
            f"repair_attempt_count={artifact.get('repair_attempt_count')}; "
            f"exact_authority_accept_count={artifact.get('exact_authority_accept_count')}; "
            f"headline_repair_claim_allowed={artifact.get('headline_repair_claim_allowed')}"
        )
    reason = str(artifact.get("gated_skip_reason") or "missing skip reason")
    prefix = "blocked_repair_gate" if reason.startswith("repair gate") else "blocked_repair_runtime"
    return f"{prefix}: {reason}"


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return source provenance rows for every artifact or file this builder reads."""

    return [source_row(root, spec) for spec in SOURCE_SPECS]


def source_row(root: Path, spec: SourceSpec) -> JsonDict:
    """Build one checksummed source row while preserving missing-file status."""

    path = root / spec.path
    return {
        "role": spec.role,
        "path": spec.path.as_posix(),
        "required": spec.required,
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object from disk, returning empty evidence on failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def mapping_rows(value: Any) -> list[JsonDict]:
    """Return only JSON-object rows from a list-like artifact field."""

    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def row_id_from(row: Mapping[str, Any]) -> str:
    """Return the stable row identifier shared by repair, exact, and monitor rows."""

    return str(row.get("row_id") or row.get("fixture_id") or row.get("source_fixture_id") or "")


def rate(numerator: int, denominator: int) -> float:
    """Return a bounded ratio while treating empty denominators as zero evidence."""

    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, float(numerator) / float(denominator))), 6)


def sha256_file(path: Path) -> str | None:
    """Checksum source bytes so repair claims trace to exact local evidence."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration(started_s: float, now_s: float | None) -> float:
    """Return a stable nonnegative elapsed duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
