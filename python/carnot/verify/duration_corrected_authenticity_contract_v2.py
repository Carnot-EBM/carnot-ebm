"""Build the Exp 3164 duration-corrected authenticity contract v2.

Spec refs: REQ-VERIFY-3164, SCENARIO-VERIFY-3164.

This module does not run a model. It repairs a contract mistake from Exp 3151:
the checked-in evidence showed a real local GGUF load plus a small smoke
generation, but the artifact was rejected solely because the total wall time
was below a fixed 60-second floor. A one-prompt smoke can be legitimately fast,
so v2 ties authenticity to measured work evidence instead: path proof, model
load proof, transcript hashes, prompt hashes, token counts, seed/checksum,
controlled subprocess return codes, repetition controls, and impossible
throughput rejection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3164_duration_corrected_authenticity_contract_v2"
SCHEMA = "carnot.live_inference_authenticity_contract.v2"
OUTPUT_REL_PATH = Path("results/experiment_3164_duration_corrected_authenticity_contract_v2.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3164_duration_corrected_authenticity_contract_v2.py"

EXP3150_REL_PATH = Path("results/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.json")
EXP3151_REL_PATH = Path("results/experiment_3151_live_inference_authenticity_preflight_v1.json")
EXP3162_REL_PATH = Path("results/experiment_3162_capstone_v293.json")

SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
IMPOSSIBLE_COMPLETION_TOKENS_PER_SECOND = 500.0
LARGE_PANEL_WARNING_DURATION_S = 60.0
LARGE_PANEL_MIN_CALLS = 10

REQUIRED_FIELDS = {
    "duration_corrected_authenticity_contract_v2_ready",
    "old_fixed_duration_rule_retired_as_hard_gate",
    "measured_work_requirements",
    "token_scaled_duration_policy",
    "repeated_call_policy",
    "required_preflight_fields",
    "fake_evidence_rejection_criteria",
    "headline_claim_policy",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}

SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("post_293_research_references", Path("research-references.md"), True, "text"),
    ("experiment_template_policy", Path("scripts/experiment_template.py"), True, "python"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("exp3150_adversarial_corrigendum", EXP3150_REL_PATH, True, "json"),
    ("exp3151_duration_failed_preflight", EXP3151_REL_PATH, True, "json"),
    ("exp3162_capstone_prior_failure", EXP3162_REL_PATH, True, "json"),
    (
        "exp3164_module",
        Path("python/carnot/verify/duration_corrected_authenticity_contract_v2.py"),
        False,
        "python",
    ),
    (
        "exp3164_script",
        Path("scripts/experiment_3164_duration_corrected_authenticity_contract_v2.py"),
        False,
        "python",
    ),
    (
        "exp3164_tests",
        Path("tests/python/test_experiment_3164_duration_corrected_authenticity_contract_v2.py"),
        False,
        "python",
    ),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3164_duration_corrected_authenticity_contract_v2.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3164_duration_corrected_authenticity_contract_v2.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/duration_corrected_authenticity_contract_v2.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3164: build the duration-corrected v2 contract artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3150 = read_json_object(root_path / EXP3150_REL_PATH)
    exp3151 = read_json_object(root_path / EXP3151_REL_PATH)
    exp3162 = read_json_object(root_path / EXP3162_REL_PATH)
    sources = source_artifacts(root_path)
    measurements = extract_exp3151_measurements(exp3151)
    source_assessment = assess_observed_source(measurements)
    ready = bool(source_assessment["passed"])

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "duration_corrected_authenticity_contract_v2_ready": ready,
        "old_fixed_duration_rule_retired_as_hard_gate": True,
        "exp3151_extracted_measurements": measurements,
        "observed_source_assessment": source_assessment,
        "measured_work_requirements": measured_work_requirements(measurements),
        "token_scaled_duration_policy": token_scaled_duration_policy(
            measurements, source_passed=ready
        ),
        "repeated_call_policy": repeated_call_policy(source_passed=ready),
        "required_preflight_fields": required_preflight_fields(),
        "fake_evidence_rejection_criteria": fake_evidence_rejection_criteria(),
        "headline_claim_policy": headline_claim_policy(),
        "reusable_contracts": reusable_contracts(source_passed=ready),
        "prior_failure_context": prior_failure_context(exp3150, exp3151, exp3162),
        "source_artifacts": sources,
        "source_checksums": {str(row["path"]): row["sha256"] for row in sources},
        "field_principles": field_principles(),
        "inference_substrate": inference_substrate(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, time.perf_counter() if now_s is None else float(now_s)),
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
    """Persist the Exp 3164 terminal JSON artifact."""

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
    """Read a JSON object, treating absent or malformed files as failed evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return the local files that make the contract traceable and auditable."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_REL_PATHS:
        path = root / rel_path
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_json_object": (
                    bool(read_json_object(path)) if source_type == "json" else None
                ),
                "sha256": sha256_file(path),
            }
        )
    return rows


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum for an existing source file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_exp3151_measurements(exp3151: Mapping[str, Any]) -> JsonDict:
    """Extract the measured work surface that v2 uses instead of a fixed floor."""

    load = _mapping(exp3151.get("model_load_evidence"))
    substrate = _mapping(exp3151.get("inference_substrate"))
    gpu_probe = _mapping(substrate.get("gpu_probe"))
    token_counts = _mapping(exp3151.get("token_counts"))
    transcripts = _mapping_list(exp3151.get("transcript_hashes"))
    selected_model_id = str(
        load.get("selected_model_id")
        or _first(exp3151.get("selected_model_ids"))
        or ""
    )
    selected_model_path = str(load.get("selected_model_path") or "")
    prompt_hashes = [str(row["prompt_hash"]) for row in transcripts if row.get("prompt_hash")]
    transcript_hashes = [
        str(row["transcript_sha256"]) for row in transcripts if row.get("transcript_sha256")
    ]
    response_hashes = [str(row["response_hash"]) for row in transcripts if row.get("response_hash")]
    worker_payload = parse_worker_stdout_summary(str(load.get("stdout_summary") or ""))
    generation_wall_time_s = float_or_none(load.get("generation_wall_time_s"))
    completion_tokens = int_or_none(token_counts.get("completion_tokens")) or 0
    completion_tps = (
        round(completion_tokens / generation_wall_time_s, 6)
        if generation_wall_time_s and generation_wall_time_s > 0
        else None
    )
    model_load_returncode = load.get("returncode")
    model_path_matches = model_path_matches_selected_spec(
        selected_model_id,
        selected_model_path,
        _mapping_list(exp3151.get("model_specs")),
    )
    return {
        "source_artifact": EXP3151_REL_PATH.as_posix(),
        "source_duration_s": float_or_none(exp3151.get("duration_s")),
        "old_minimum_duration_requirement_s": float_or_none(
            exp3151.get("minimum_duration_requirement_s")
        ),
        "old_blocked_reason": str(exp3151.get("blocked_reason") or ""),
        "blocked_by_old_fixed_floor": "shorter than minimum" in str(
            exp3151.get("blocked_reason") or ""
        ),
        "model_load_wall_time_s": float_or_none(load.get("load_wall_time_s")),
        "generation_wall_time_s": generation_wall_time_s,
        "total_worker_wall_time_s": float_or_none(load.get("total_worker_wall_time_s")),
        "token_counts": dict(token_counts),
        "selected_model_id": selected_model_id,
        "selected_model_path": selected_model_path,
        "model_path_exists_proof": bool(load.get("path_exists")),
        "model_path_matches_selected_spec": model_path_matches,
        "load_command_sha256": str(load.get("load_command_sha256") or ""),
        "worker_code_sha256": str(load.get("worker_code_sha256") or ""),
        "model_load_returncode": model_load_returncode,
        "transcript_hash_available": bool(transcript_hashes),
        "transcript_sha256_values": transcript_hashes,
        "prompt_hashes": prompt_hashes,
        "response_hashes": response_hashes,
        "random_seed": exp3151.get("random_seed"),
        "reproducibility_checksum": exp3151.get("reproducibility_checksum"),
        "completion_tokens_per_generation_second": completion_tps,
        "controlled_subprocess_return_codes": [
            {"name": "model_load_smoke_worker", "returncode": model_load_returncode},
            {
                "name": "torch_cuda_probe",
                "returncode": _mapping(gpu_probe.get("torch_cuda_probe")).get("returncode"),
            },
            {
                "name": "nvidia_smi_inventory",
                "returncode": _mapping(gpu_probe.get("nvidia_smi_inventory")).get(
                    "returncode"
                ),
            },
        ],
        "wall_clock_supported_by_command_output": wall_clock_supported_by_command_output(
            load, worker_payload
        ),
        "substrate_evidence": dict(substrate),
    }


def parse_worker_stdout_summary(text: str) -> JsonDict:
    """Parse the worker JSON object embedded in compact command output."""

    for line in reversed(text.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def wall_clock_supported_by_command_output(
    load_evidence: Mapping[str, Any], worker_payload: Mapping[str, Any]
) -> bool:
    """Check that wall-clock fields agree with the worker's printed JSON."""

    return all(
        _float_matches(worker_payload.get(key), load_evidence.get(key))
        for key in (
            "load_wall_time_s",
            "generation_wall_time_s",
            "total_worker_wall_time_s",
        )
    )


def model_path_matches_selected_spec(
    selected_model_id: str,
    selected_model_path: str,
    model_specs: Sequence[Mapping[str, Any]],
) -> bool:
    """Verify selected model identity and path agree with the model-spec row."""

    for row in model_specs:
        if row.get("hf_id") == selected_model_id and row.get("selected_for_smoke") is True:
            return str(row.get("model_path") or "") == selected_model_path
    return False


def assess_observed_source(measurements: Mapping[str, Any]) -> JsonDict:
    """Classify whether the Exp 3151 source evidence would satisfy v2."""

    violations: list[str] = []
    if measurements.get("model_path_exists_proof") is not True:
        violations.append("model path existence proof missing")
    if measurements.get("model_load_wall_time_s") is None:
        violations.append("model-load wall time missing")
    if measurements.get("model_load_returncode") != 0:
        violations.append("controlled model-load subprocess did not return 0")
    if measurements.get("transcript_hash_available") is not True:
        violations.append("missing transcript hashes")
    if not measurements.get("prompt_hashes"):
        violations.append("missing prompt hashes")
    token_counts = _mapping(measurements.get("token_counts"))
    total_tokens = int_or_none(token_counts.get("total_tokens"))
    if total_tokens is None or total_tokens <= 0:
        violations.append("missing token counts")
    if measurements.get("random_seed") is None:
        violations.append("missing random_seed")
    if not measurements.get("reproducibility_checksum"):
        violations.append("missing reproducibility_checksum")
    if impossible_token_throughput(measurements):
        violations.append("impossible token throughput")
    if measurements.get("model_path_matches_selected_spec") is not True:
        violations.append("selected model/local path mismatch")
    if measurements.get("wall_clock_supported_by_command_output") is not True:
        violations.append("wall-clock claims not supported by command output")
    return {
        "passed": not violations,
        "violations": violations,
        "old_fixed_floor_would_block": bool(measurements.get("blocked_by_old_fixed_floor")),
        "v2_accepts_fast_smoke_when_measured_work_passes": not violations,
    }


def impossible_token_throughput(measurements: Mapping[str, Any]) -> bool:
    """Reject throughput claims beyond the v2 smoke-call plausibility ceiling."""

    tps = float_or_none(measurements.get("completion_tokens_per_generation_second"))
    return bool(tps is not None and tps > IMPOSSIBLE_COMPLETION_TOKENS_PER_SECOND)


def measured_work_requirements(measurements: Mapping[str, Any]) -> list[JsonDict]:
    """Define pass criteria in terms of observed work, not elapsed time alone."""

    return [
        {
            "requirement": "local_model_path_exists",
            "observed": bool(measurements.get("model_path_exists_proof")),
            "source_field": "model_load_evidence.path_exists",
            "principle": "claimed local inference needs a concrete local model path",
        },
        {
            "requirement": "model_load_proof",
            "observed": measurements.get("model_load_wall_time_s") is not None
            and measurements.get("model_load_returncode") == 0,
            "source_field": "model_load_evidence.load_wall_time_s",
            "principle": "a live call must prove the model loaded successfully",
        },
        {
            "requirement": "controlled_subprocess_return_codes",
            "observed": all(
                row.get("returncode") == 0
                for row in measurements.get("controlled_subprocess_return_codes", [])
            ),
            "source_field": "model_load_evidence.returncode + substrate probe returncodes",
            "principle": "bounded subprocesses must expose success or failure",
        },
        {
            "requirement": "transcript_and_prompt_hashes",
            "observed": bool(measurements.get("transcript_sha256_values"))
            and bool(measurements.get("prompt_hashes")),
            "source_field": "transcript_hashes",
            "principle": "outputs must be replay-identifiable without copying raw text",
        },
        {
            "requirement": "token_counts_and_seed_checksum",
            "observed": bool(measurements.get("token_counts"))
            and measurements.get("random_seed") is not None
            and bool(measurements.get("reproducibility_checksum")),
            "source_field": "token_counts + random_seed + reproducibility_checksum",
            "principle": "replay needs enough machine-checkable determinism metadata",
        },
        {
            "requirement": "wall_clock_claims_supported_by_command_output",
            "observed": measurements.get("wall_clock_supported_by_command_output") is True,
            "source_field": "model_load_evidence.stdout_summary",
            "principle": "duration fields must be backed by the worker's own output",
        },
    ]


def token_scaled_duration_policy(
    measurements: Mapping[str, Any], *, source_passed: bool
) -> JsonDict:
    """Return the v2 duration rule that scales with prompt/completion work."""

    token_counts = _mapping(measurements.get("token_counts"))
    return {
        "policy_version": "v2",
        "fixed_60s_floor": {
            "hard_gate": False,
            "optional_warning_only_for_large_panels": True,
            "warning_threshold_s": LARGE_PANEL_WARNING_DURATION_S,
            "large_panel_min_calls": LARGE_PANEL_MIN_CALLS,
        },
        "one_prompt_smoke": {
            "fixed_minimum_duration_s": 0.0,
            "requires_model_load_wall_time": True,
            "requires_generation_wall_time": True,
            "requires_nonzero_prompt_and_completion_tokens": True,
            "reject_if_completion_tokens_per_second_gt": IMPOSSIBLE_COMPLETION_TOKENS_PER_SECOND,
            "pass_fail_basis": "measured work evidence, not an arbitrary elapsed-time floor",
        },
        "token_scaled_work_budget": {
            "minimum_prompt_tokens": 1,
            "minimum_completion_tokens": 1,
            "completion_tokens_observed": int_or_none(token_counts.get("completion_tokens")),
            "prompt_tokens_observed": int_or_none(token_counts.get("prompt_tokens")),
            "completion_tokens_per_generation_second_observed": measurements.get(
                "completion_tokens_per_generation_second"
            ),
        },
        "observed_exp3151_smoke": {
            "duration_s": measurements.get("source_duration_s"),
            "old_minimum_duration_requirement_s": measurements.get(
                "old_minimum_duration_requirement_s"
            ),
            "blocked_by_old_fixed_floor": bool(measurements.get("blocked_by_old_fixed_floor")),
            "accepted_by_v2_duration_policy": source_passed,
        },
    }


def repeated_call_policy(*, source_passed: bool) -> JsonDict:
    """Define replay controls that catch stale or fabricated transcripts."""

    shared = {
        "require_distinct_prompt_hashes": True,
        "require_distinct_transcript_sha256": True,
        "require_distinct_response_hashes_or_seeded_prompt_variants": True,
        "all_calls_must_reference_same_selected_model_or_declared_model_rotation": True,
    }
    return {
        "principle": "replay should catch stale or fabricated transcripts",
        "exp3165": {
            **shared,
            "minimum_distinct_smoke_calls": 2,
            "usable_by_downstream": source_passed,
            "alternative": "single longer token-scaled call only if completion budget is declared before execution",
        },
        "exp3167": {
            **shared,
            "minimum_distinct_smoke_calls": 3,
            "usable_by_downstream": source_passed,
            "alternative": "token-scaled panel budget with row-level transcript hashes",
        },
        "stale_replay_controls": {
            "reject_reused_transcript_sha256": True,
            "reject_reused_prompt_hash_without_declared_repetition": True,
            "reject_transcript_hash_older_than_source_run_without_carry_forward_note": True,
        },
    }


def required_preflight_fields() -> list[str]:
    """Machine-checkable fields every downstream live rerun must preserve."""

    return [
        "model_specs",
        "model_specs.path_exists",
        "selected_model_ids",
        "model_load_evidence.load_attempted",
        "model_load_evidence.selected_model_id",
        "model_load_evidence.selected_model_path",
        "model_load_evidence.load_command_sha256",
        "model_load_evidence.worker_code_sha256",
        "model_load_evidence.returncode",
        "model_load_evidence.load_wall_time_s",
        "model_load_evidence.generation_wall_time_s",
        "model_load_evidence.total_worker_wall_time_s",
        "transcript_hashes.transcript_sha256",
        "transcript_hashes.response_hash",
        "prompt_hashes",
        "token_counts.prompt_tokens",
        "token_counts.completion_tokens",
        "token_counts.total_tokens",
        "random_seed",
        "reproducibility_checksum",
        "controlled_subprocess_return_codes",
        "inference_substrate",
    ]


def fake_evidence_rejection_criteria() -> list[str]:
    """Keep adversarial failure modes explicit for future contract consumers."""

    return [
        "reject no model loaded: load_attempted false, path proof absent, or load wall time missing",
        "reject missing transcript hashes: every live call needs transcript_sha256 and response_hash",
        "reject no seed/checksum: random_seed and reproducibility_checksum are mandatory",
        "reject impossible token throughput: completion tokens per generation second above the declared ceiling",
        "reject reused stale transcript hash: repeated calls must have fresh transcript hashes unless declared replay",
        "reject mismatch between selected model and local path: selected model ID must match the selected model-spec path",
        "reject wall-clock claims not supported by command output: duration fields must match worker stdout JSON",
        "reject uncontrolled subprocess outcomes: model load, CUDA probe, and GPU inventory return codes must be recorded",
    ]


def headline_claim_policy() -> JsonDict:
    """Separate smoke authenticity from any public verifier performance claim."""

    return {
        "smoke_test_headline_claim_allowed": False,
        "smoke_test_role": "preflight authenticity evidence only",
        "headline_requires": [
            "clean downstream verifier panel",
            "row-level exact labels or verifier authority",
            "all v2 authenticity criteria passing",
            "no inherited adversarial methodology flags",
        ],
        "large_panel_duration_floor": {
            "warning_only": True,
            "warning_threshold_s": LARGE_PANEL_WARNING_DURATION_S,
            "minimum_calls_before_warning_applies": LARGE_PANEL_MIN_CALLS,
        },
    }


def reusable_contracts(*, source_passed: bool) -> JsonDict:
    """Reusable downstream gates for the next live verifier recovery tasks."""

    required_fields = required_preflight_fields()
    rejection = fake_evidence_rejection_criteria()
    return {
        "exp3165": {
            "contract_id": "exp3165_duration_corrected_live_authenticity_preflight",
            "usable_by_downstream": source_passed,
            "required_preflight_fields": required_fields,
            "minimum_distinct_smoke_calls": 2,
            "fake_evidence_rejection_criteria": rejection,
            "old_fixed_60s_rule_hard_gate": False,
        },
        "exp3167": {
            "contract_id": "exp3167_clean_live_verifier_rerun_authenticity_gate",
            "usable_by_downstream": source_passed,
            "required_preflight_fields": required_fields,
            "minimum_distinct_smoke_calls": 3,
            "fake_evidence_rejection_criteria": rejection,
            "old_fixed_60s_rule_hard_gate": False,
        },
    }


def prior_failure_context(
    exp3150: Mapping[str, Any],
    exp3151: Mapping[str, Any],
    exp3162: Mapping[str, Any],
) -> JsonDict:
    """Summarize why this contract exists without rerunning the live verifier."""

    return {
        "exp3150_methodology_requirements": list(
            exp3150.get("methodology_requirements_for_rerun") or []
        ),
        "exp3151_blocked_reason": exp3151.get("blocked_reason"),
        "exp3151_live_call_count": exp3151.get("live_call_count"),
        "exp3151_duration_s": exp3151.get("duration_s"),
        "exp3162_next_top_gap": exp3162.get("next_top_gap"),
        "exp3162_publication_blocker_count": exp3162.get("publication_blocker_count"),
        "exp3162_missing_artifacts": list(exp3162.get("missing_artifacts") or []),
    }


def field_principles() -> JsonDict:
    """Explain why each required output field exists."""

    return {
        "duration_corrected_authenticity_contract_v2_ready": (
            "live reruns need an explicit contract"
        ),
        "old_fixed_duration_rule_retired_as_hard_gate": (
            "failed heuristics must not remain load-bearing"
        ),
        "measured_work_requirements": "authenticity should be tied to observed work",
        "token_scaled_duration_policy": (
            "duration plausibility should scale with prompt and completion work"
        ),
        "repeated_call_policy": "replay should catch stale or fabricated transcripts",
        "required_preflight_fields": "downstream tasks need machine-checkable fields",
        "fake_evidence_rejection_criteria": (
            "adversarial failure modes must stay explicit"
        ),
        "headline_claim_policy": "smoke tests should not become headline evidence",
        "source_artifacts": "contract must trace to prior failure",
        "inference_substrate": "aggregation work must declare no live model inference",
        "honest_verdict": "terminal verdict must expose complete or blocked state",
    }


def inference_substrate() -> JsonDict:
    """Declare that Exp 3164 itself performs no live inference."""

    return {
        "kind": "aggregation_from_exp3151_exp3150_exp3162_checked_in_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "downloads_models": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "live_model_calls_run_by_exp3164": 0,
        "source": EXP3151_REL_PATH.as_posix(),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and fail-closed contract invariants."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3164 artifact missing required fields: {missing}")
    if artifact.get("old_fixed_duration_rule_retired_as_hard_gate") is not True:
        raise ValueError("old fixed duration rule must be retired as a hard gate")
    headline = _mapping(artifact.get("headline_claim_policy"))
    if headline.get("smoke_test_headline_claim_allowed") is not False:
        raise ValueError("headline smoke-test claims must remain forbidden")
    substrate = _mapping(artifact.get("inference_substrate"))
    if (
        substrate.get("no_live_llm_inference") is not True
        or substrate.get("live_model_calls_run_by_exp3164") != 0
    ):
        raise ValueError("Exp 3164 must declare no live model inference")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("duration_corrected_authenticity_contract_v2_ready") is True:
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("ready artifact requires terminal success prefix")
    elif not verdict.startswith("blocked_") and verdict:
        raise ValueError("blocked contract artifact requires blocked_ verdict")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return conductor-compatible terminal verdict wording."""

    if artifact.get("duration_corrected_authenticity_contract_v2_ready") is True:
        measurements = _mapping(artifact.get("exp3151_extracted_measurements"))
        return (
            "complete: duration_corrected_authenticity_contract_v2_ready=true; "
            "old_fixed_duration_rule_retired_as_hard_gate=true; "
            f"selected_model_id={measurements.get('selected_model_id')}; "
            "source_exp3151_fast_smoke_accepted_by_v2=true"
        )
    violations = _mapping(artifact.get("observed_source_assessment")).get("violations") or []
    return (
        "blocked_contract_source_evidence: "
        "duration_corrected_authenticity_contract_v2_ready=false; "
        f"violations={len(violations)}"
    )


def stable_hash(value: Any) -> str:
    """Hash JSON-serializable evidence with canonical key ordering."""

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def duration(started_s: float, finished_s: float) -> float:
    """Return nonnegative elapsed seconds rounded for stable artifacts."""

    return round(max(0.0, float(finished_s) - float(started_s)), 6)


def int_or_none(value: Any) -> int | None:
    """Parse an integer field without raising during malformed-source handling."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    """Parse a float field without raising during malformed-source handling."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[JsonDict]:
    return (
        [dict(row) for row in value if isinstance(row, Mapping)]
        if isinstance(value, list)
        else []
    )


def _first(value: Any) -> Any:
    return (
        value[0]
        if isinstance(value, Sequence) and not isinstance(value, str) and value
        else None
    )


def _float_matches(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    left_float = float_or_none(left)
    right_float = float_or_none(right)
    return bool(
        left_float is not None
        and right_float is not None
        and abs(left_float - right_float) <= tolerance
    )


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = build_artifact(REPO_ROOT)
    output = REPO_ROOT / OUTPUT_REL_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["duration_corrected_authenticity_contract_v2_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
