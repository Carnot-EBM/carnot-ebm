"""Exp 3100 Z3/test-oracle formal-feedback v2 pilot.

Spec refs: REQ-VERIFY-3100,
           SCENARIO-VERIFY-3100,
           SCENARIO-VERIFY-3100-BLOCKED-HEADLINE.

The pilot consumes the `.289` exact-fixture protocol, then checks a small
repair panel with local Z3 and Python execution oracles. Dafny is recorded as a
preflight fact only; when it is absent, this module continues on the Z3/test-
oracle path instead of claiming Dafny verification. Guided successes are
promoted only when a mandated cached SOTA GGUF pair is available and the live
model repair is validated by the same exact oracle used for baselines.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf

try:  # pragma: no cover - missing dependency is exercised by injected tests.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]
CommandResolver = Callable[[str], str | None]
ResolveGgufFn = Callable[[str, str], str | None]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
LlamaFactory = Callable[..., Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3100_z3_oracle_feedback_v2"
SCHEMA = "carnot.z3_oracle_feedback.v2"
OUTPUT_REL_PATH = Path("results/experiment_3100_z3_oracle_feedback_v2.json")
EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
STRATIFIED_MANIFEST_REL_PATH = Path(
    "results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl"
)
DEFAULT_PANEL_SIZE = 6
DEFAULT_SEED = 3100
MANDATED_MODEL_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = (
    "formal_feedback_v2_ready",
    "model_specs",
    "z3_available",
    "dafny_available",
    "exact_ground_truth_count",
    "formal_feedback_delta",
    "guided_success_count",
    "solver_only_success_count",
    "vacuity_guard_passed",
    "test_oracle_count",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
SOURCE_REL_PATHS: tuple[tuple[str, Path, str], ...] = (
    ("codex", Path("CODEX.md"), "repo spec-first workflow"),
    ("claude", Path("CLAUDE.md"), "artifact authenticity and formal-tool discipline"),
    ("research_references", Path("research-references.md"), "formal-feedback context"),
    ("experiment_template", Path("scripts/experiment_template.py"), "SOTA cache helper context"),
    (
        "exp3071",
        Path("results/experiment_3071_verge_mcs_smt_correction_pilot_v1.json"),
        "prior MCS/Z3 feedback baseline",
    ),
    (
        "exp3086",
        Path("results/experiment_3086_dafny_z3_formal_feedback_pilot_v1.json"),
        "prior Dafny/Z3 formal-feedback pilot",
    ),
    (
        "exp3094",
        Path("results/experiment_3094_capstone_v288.json"),
        ".288 capstone noting absent formal-feedback lift",
    ),
    ("exp3097", EXP3097_REL_PATH, ".289 exact-fixture protocol"),
    ("exp3097_manifest", STRATIFIED_MANIFEST_REL_PATH, ".289 stratified exact manifest"),
)


@dataclass(frozen=True)
class FeedbackConfig:
    """Runtime knobs for Exp 3100.

    Tests pass temporary paths and fake model loaders here. The defaults point
    at the repository paths used by the conductor so `python -m
    carnot.eval.z3_oracle_feedback_v2` writes the requested terminal artifact.
    """

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    panel_size: int = DEFAULT_PANEL_SIZE
    preferred_quant: str = "Q4_K_M"
    decode_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": DEFAULT_SEED,
            "stop": ["\n\n", "</s>"],
        }
    )
    load_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "n_ctx": 2048,
            "n_gpu_layers": -1,
            "verbose": False,
        }
    )
    started_s: float | None = None
    clock: ClockFn = time.perf_counter
    tests_run: Sequence[str] = ()

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_s is None else float(self.started_s)

    def effective_decode_config(self) -> JsonDict:
        return dict(self.decode_config)

    def effective_load_config(self, gpu: int) -> JsonDict:
        config = dict(self.load_config)
        config.setdefault("main_gpu", int(gpu))
        return config


def run_experiment(
    config: FeedbackConfig | None = None,
    *,
    command_resolver: CommandResolver = shutil.which,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    llama_factory: LlamaFactory | None = None,
    z3_module: Any = _z3,
    repo_commit_func: Callable[[Path], str] | None = None,
    python_environment_func: Callable[[], Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run the pilot and write a terminal artifact for every precondition state."""

    active = config or FeedbackConfig()
    started_s = active.start_time()
    dafny_path = command_resolver("dafny")
    z3_path = command_resolver("z3")
    dafny_available = dafny_path is not None
    z3_available = z3_path is not None and z3_module is not None
    exp3097 = safe_load_json(active.repo_root / EXP3097_REL_PATH)
    manifest_path = active.repo_root / str(
        exp3097.get("stratified_eval_manifest_path") or STRATIFIED_MANIFEST_REL_PATH
    )
    manifest_rows = safe_load_jsonl(manifest_path)
    selected_rows = select_repair_panel(manifest_rows, active.panel_size)
    model_specs = model_cache_specs(resolve_gguf_func, active.preferred_quant)
    cached_pair_status = probe_cached_pair(cached_pair_func, active.preferred_quant)
    source_rows = source_artifacts(active.repo_root)
    runtime_blocker = first_runtime_blocker(
        z3_available=z3_available,
        exp3097=exp3097,
        selected_rows=selected_rows,
    )
    commit_probe = repo_commit_func or repo_commit
    env_probe = python_environment_func or python_environment

    fixture_results: list[JsonDict] = []
    headline_blocked_reason: str | None = None
    guided_feasible = False
    live_llm_inference = False
    selected_model: Mapping[str, Any] | None = None
    test_oracle_counter = {"count": 0}

    if runtime_blocker is None:
        selected_pair = cached_pair_status.get("models") or []
        if not selected_pair:
            headline_blocked_reason = "cached_sota_pair_unavailable"
            fixture_results = evaluate_panel_without_guided(
                selected_rows=selected_rows,
                z3_module=z3_module,
                test_oracle_counter=test_oracle_counter,
                guided_failure_reason=headline_blocked_reason,
            )
        else:
            selected_model = dict(selected_pair[0])
            try:
                llm = (llama_factory or default_llama_factory)(
                    model_path=str(selected_model["model_path"]),
                    **active.effective_load_config(int(selected_model.get("gpu", 0))),
                )
                guided_feasible = True
                live_llm_inference = True
                try:
                    fixture_results = evaluate_panel_with_guided(
                        selected_rows=selected_rows,
                        llm=llm,
                        config=active,
                        z3_module=z3_module,
                        test_oracle_counter=test_oracle_counter,
                    )
                finally:
                    close = getattr(llm, "close", None)
                    if callable(close):
                        close()
            except Exception as exc:  # noqa: BLE001 - artifact must preserve any load failure.
                headline_blocked_reason = f"model_load_failed: {type(exc).__name__}: {exc}"
                fixture_results = evaluate_panel_without_guided(
                    selected_rows=selected_rows,
                    z3_module=z3_module,
                    test_oracle_counter=test_oracle_counter,
                    guided_failure_reason=headline_blocked_reason,
                )

    artifact = build_artifact(
        config=active,
        duration_s=active.clock() - started_s,
        dafny_available=dafny_available,
        z3_available=z3_available,
        dafny_path=dafny_path,
        z3_path=z3_path,
        exp3097=exp3097,
        manifest_path=manifest_path,
        selected_rows=selected_rows,
        fixture_results=fixture_results,
        model_specs=model_specs,
        cached_pair_status=cached_pair_status,
        source_rows=source_rows,
        runtime_blocker=runtime_blocker,
        headline_blocked_reason=headline_blocked_reason,
        guided_feasible=guided_feasible,
        live_llm_inference=live_llm_inference,
        selected_model=selected_model,
        test_oracle_count=int(test_oracle_counter["count"]),
        repo_commit=commit_probe(active.repo_root),
        python_env=dict(env_probe()),
    )
    validate_artifact(artifact)
    write_json(active.artifact_path(), artifact)
    return artifact


def evaluate_panel_without_guided(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    z3_module: Any,
    test_oracle_counter: dict[str, int],
    guided_failure_reason: str,
) -> list[JsonDict]:
    """Evaluate exact baselines while marking guided repair as infeasible."""

    results = []
    for row in selected_rows:
        base = evaluate_baselines(row, z3_module, test_oracle_counter)
        base.update(
            {
                "guided_prompt_hash": None,
                "guided_raw_output_hash": None,
                "guided_parse": {"valid_parse": False, "repair": None, "parse_error": ""},
                "guided_candidate": None,
                "guided_validation": {
                    "valid": False,
                    "exact_checked": False,
                    "exact_authority": "not_checked_guided_infeasible",
                    "failure_reason": guided_failure_reason,
                    "diagnostics": {"reason": guided_failure_reason},
                },
                "guided_success": False,
            }
        )
        results.append(base)
    return results


def evaluate_panel_with_guided(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    llm: Any,
    config: FeedbackConfig,
    z3_module: Any,
    test_oracle_counter: dict[str, int],
) -> list[JsonDict]:
    """Evaluate no-feedback, solver-only, and guided conditions for each row."""

    results = []
    for row in selected_rows:
        base = evaluate_baselines(row, z3_module, test_oracle_counter)
        prompt = build_guided_prompt(row, base)
        raw = llm(prompt, **config.effective_decode_config())
        text = extract_text(raw)
        parsed = parse_guided_repair_response(text)
        repair = parsed["repair"] if parsed["valid_parse"] else None
        guided_validation = (
            validate_candidate(row, repair, z3_module=z3_module, counter=test_oracle_counter)
            if parsed["valid_parse"]
            else {
                "valid": False,
                "exact_checked": False,
                "exact_authority": "not_checked_parse_failed",
                "failure_reason": parsed["parse_error"],
                "diagnostics": {},
            }
        )
        base.update(
            {
                "guided_prompt_hash": sha256_text(prompt),
                "guided_raw_output_hash": sha256_text(text),
                "guided_parse": parsed,
                "guided_candidate": repair,
                "guided_validation": guided_validation,
                "guided_success": bool(guided_validation["valid"]),
            }
        )
        results.append(base)
    return results


def evaluate_baselines(
    row: Mapping[str, Any], z3_module: Any, test_oracle_counter: dict[str, int]
) -> JsonDict:
    """Run no-feedback, solver-only, and empty-repair guards for one fixture."""

    original = original_candidate(row)
    no_feedback = validate_candidate(
        row, original, z3_module=z3_module, counter=test_oracle_counter
    )
    solver_candidate = solver_only_repair(row, z3_module=z3_module)
    solver_validation = validate_candidate(
        row,
        solver_candidate or original,
        z3_module=z3_module,
        counter=test_oracle_counter,
    )
    empty_validation = validate_candidate(row, {}, z3_module=z3_module, counter=test_oracle_counter)
    return {
        "fixture_id": str(row["source_fixture_id"]),
        "task_family": str(row.get("task_family", "")),
        "perturbation_type": str(row.get("perturbation_type", "")),
        "expected_answer": str(row.get("expected_answer", "")),
        "solver_label": str(row.get("solver_label", "")),
        "source_prompt_payload_sha256": str(row.get("source_prompt_payload_sha256", "")),
        "no_feedback_candidate": original,
        "no_feedback_validation": no_feedback,
        "no_feedback_success": bool(no_feedback["valid"]),
        "solver_only_candidate": solver_candidate,
        "solver_only_validation": solver_validation,
        "solver_only_success": bool(solver_validation["valid"]),
        "empty_repair_validation": empty_validation,
    }


def select_repair_panel(rows: Sequence[Mapping[str, Any]], panel_size: int) -> list[JsonDict]:
    """Select a small deterministic repair-target panel from the `.289` manifest."""

    buckets: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        if row.get("repair_target", {}).get("applicable") is True:
            buckets[str(row.get("perturbation_type", ""))].append(dict(row))
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda item: str(item.get("source_fixture_id", "")))
    selected: list[JsonDict] = []
    while len(selected) < panel_size and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < panel_size:
                selected.append(buckets[key].pop(0))
    return selected


def first_runtime_blocker(
    *,
    z3_available: bool,
    exp3097: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
) -> str | None:
    """Return a hard precondition blocker before any candidate evaluation."""

    if not z3_available:
        return "z3_or_protocol_precondition_failed"
    if exp3097.get("eval_protocol_ready") is not True:
        return "z3_or_protocol_precondition_failed"
    if not selected_rows:
        return "z3_or_protocol_precondition_failed"
    return None


def original_candidate(row: Mapping[str, Any]) -> JsonDict:
    """Return the unmodified candidate from a manifest row."""

    payload = row["leakage_safe_prompt_payload"]
    if "candidate" in payload:
        return {"candidate": payload["candidate"]}
    if "candidate_assignment" in payload:
        return {"candidate_assignment": dict(payload["candidate_assignment"])}
    return {"candidate_assertion": payload.get("candidate_assertion", "")}


def solver_only_repair(row: Mapping[str, Any], *, z3_module: Any) -> JsonDict | None:
    """Produce the deterministic solver-only repair when local Z3 can solve it."""

    payload = row["leakage_safe_prompt_payload"]
    if "candidate_assignment" not in payload:
        return None
    assignment = dict(payload["candidate_assignment"])
    variables = [str(var) for var in payload.get("variables", [])]
    target = _sum_constraint_target(payload.get("constraints", []), variables)
    if target is None or not variables:
        return None
    repaired = dict(assignment)
    preserved_sum = sum(int(repaired.get(var, 0)) for var in variables[1:])
    repaired[variables[0]] = int(target) - preserved_sum
    validation = validate_candidate(
        row,
        {"candidate_assignment": repaired},
        z3_module=z3_module,
        counter=None,
    )
    return {"candidate_assignment": repaired} if validation["valid"] else None


def validate_candidate(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    *,
    z3_module: Any,
    counter: dict[str, int] | None = None,
) -> JsonDict:
    """Validate a candidate with the exact oracle implied by one fixture row."""

    if counter is not None:
        counter["count"] += 1
    payload = row["leakage_safe_prompt_payload"]
    if "candidate" in payload:
        return _validate_json_candidate(payload, candidate)
    if "candidate_assignment" in payload:
        return _validate_smt_candidate(payload, candidate, z3_module)
    return _validate_python_assertion_candidate(payload, candidate)


def vacuity_guard_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when every selected row rejects the empty repair."""

    if not rows:
        return False
    return all(row.get("empty_repair_validation", {}).get("valid") is False for row in rows)


def build_guided_prompt(row: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    """Build a leakage-safe repair prompt from exact diagnostics, not answers."""

    exposed = {
        "fixture_id": row["source_fixture_id"],
        "perturbation_type": row["perturbation_type"],
        "payload": row["leakage_safe_prompt_payload"],
        "no_feedback_failure": baseline["no_feedback_validation"]["failure_reason"],
        "solver_only_failure": baseline["solver_only_validation"]["failure_reason"],
        "response_schema": {
            "repair": {
                "candidate": "JSON text for json_syntax_repair",
                "candidate_assignment": "integer assignment for numeric_bound_repair",
                "candidate_assertion": "assert statement for python_assertion_repair",
            }
        },
    }
    return (
        "Role: formal-feedback repair assistant\n"
        f"Fixture: {row['source_fixture_id']}\n"
        "Return exactly one JSON object with key repair and no markdown fences.\n"
        f"Payload: {json.dumps(exposed, sort_keys=True, separators=(',', ':'))}\n"
    )


def parse_guided_repair_response(text: str) -> JsonDict:
    """Parse a guided repair without trusting prose or markdown fences."""

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    source = match.group(1) if match else text
    if not match:
        start = source.find("{")
        end = source.rfind("}")
        if start >= 0 and end > start:
            source = source[start : end + 1]
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        return {"valid_parse": False, "repair": None, "parse_error": str(exc)}
    repair = parsed.get("repair", parsed) if isinstance(parsed, Mapping) else None
    return {
        "valid_parse": isinstance(repair, Mapping),
        "repair": dict(repair) if isinstance(repair, Mapping) else None,
        "parse_error": "" if isinstance(repair, Mapping) else "repair is not an object",
    }


def build_artifact(
    *,
    config: FeedbackConfig,
    duration_s: float,
    dafny_available: bool,
    z3_available: bool,
    dafny_path: str | None,
    z3_path: str | None,
    exp3097: Mapping[str, Any],
    manifest_path: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    fixture_results: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    cached_pair_status: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    runtime_blocker: str | None,
    headline_blocked_reason: str | None,
    guided_feasible: bool,
    live_llm_inference: bool,
    selected_model: Mapping[str, Any] | None,
    test_oracle_count: int,
    repo_commit: str,
    python_env: Mapping[str, Any],
) -> JsonDict:
    """Assemble the Exp 3100 terminal artifact."""

    exact_count = len(fixture_results)
    guided = sum(1 for row in fixture_results if row.get("guided_success"))
    solver_only = sum(1 for row in fixture_results if row.get("solver_only_success"))
    no_feedback = sum(1 for row in fixture_results if row.get("no_feedback_success"))
    delta = round((guided - solver_only) / exact_count, 6) if exact_count else 0.0
    vacuity_ok = vacuity_guard_passed(fixture_results)
    ready = (
        runtime_blocker is None
        and headline_blocked_reason is None
        and guided_feasible
        and z3_available
        and exact_count > 0
        and vacuity_ok
        and guided > solver_only
        and live_llm_inference
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "formal_feedback_v2_ready": ready,
        "model_specs": list(model_specs),
        "z3_available": z3_available,
        "dafny_available": dafny_available,
        "exact_ground_truth_count": exact_count,
        "formal_feedback_delta": delta,
        "guided_success_count": guided,
        "solver_only_success_count": solver_only,
        "no_feedback_success_count": no_feedback,
        "vacuity_guard_passed": vacuity_ok,
        "test_oracle_count": int(test_oracle_count),
        "source_artifacts": list(source_rows),
        "inference_substrate": inference_substrate(
            z3_available=z3_available,
            dafny_available=dafny_available,
            cached_pair_status=cached_pair_status,
            model_specs=model_specs,
            live_llm_inference=live_llm_inference,
            selected_model=selected_model,
            repo_commit=repo_commit,
            python_env=python_env,
        ),
        "honest_verdict": honest_verdict(
            ready=ready,
            runtime_blocker=runtime_blocker,
            headline_blocked_reason=headline_blocked_reason,
            z3_available=z3_available,
            dafny_available=dafny_available,
            guided=guided,
            solver_only=solver_only,
            exact_count=exact_count,
        ),
        "dafny_path": dafny_path,
        "z3_path": z3_path,
        "runtime_blocker": runtime_blocker,
        "headline_blocked_reason": headline_blocked_reason,
        "guided_evaluation_feasible": guided_feasible,
        "cached_sota_pair_status": dict(cached_pair_status),
        "exp3097_protocol_ready": exp3097.get("eval_protocol_ready") is True,
        "stratified_eval_manifest_path": _relative_path(config.repo_root, manifest_path),
        "selected_fixture_count": len(selected_rows),
        "selected_fixture_ids": [str(row["source_fixture_id"]) for row in selected_rows],
        "fixture_results": list(fixture_results),
        "tests_or_checks_run": list(config.tests_run),
        "duration_s": round(float(duration_s), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = sha256_json(
        {
            "selected_fixture_ids": artifact["selected_fixture_ids"],
            "guided_success_count": guided,
            "solver_only_success_count": solver_only,
            "formal_feedback_delta": delta,
            "model_cache": [
                {"hf_id": item["hf_id"], "cached": item["cached"]} for item in model_specs
            ],
        }
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if a terminal artifact overstates readiness."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not artifact.get("model_specs"):
        raise ValueError("model_specs must record mandated model cache status")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("runtime_blocker") == "z3_or_protocol_precondition_failed":
        if not verdict.startswith("blocked_z3_or_protocol_precondition_failed:"):
            raise ValueError(
                "blocked precondition artifacts must disclose the blocked precondition"
            )
        return
    if artifact.get("formal_feedback_v2_ready") is not True:
        if artifact.get("headline_blocked_reason") and not verdict.startswith(
            "complete_blocked_headline:"
        ):
            raise ValueError("blocked headline artifacts must use complete_blocked_headline")
        if not artifact.get("headline_blocked_reason") and not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must start with a terminal success prefix")
        return
    if int(artifact.get("exact_ground_truth_count") or 0) <= 0:
        raise ValueError("exact_ground_truth_count must be positive when ready")
    if artifact.get("vacuity_guard_passed") is not True:
        raise ValueError("formal_feedback_v2_ready requires vacuity_guard_passed")
    if float(artifact.get("formal_feedback_delta") or 0.0) <= 0.0:
        raise ValueError("formal_feedback_v2_ready requires positive formal_feedback_delta")
    if int(artifact.get("guided_success_count") or 0) <= int(
        artifact.get("solver_only_success_count") or 0
    ):
        raise ValueError("guided_success_count must exceed solver_only_success_count")
    if artifact.get("guided_evaluation_feasible") is not True:
        raise ValueError("formal_feedback_v2_ready requires live guided evaluation")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def honest_verdict(
    *,
    ready: bool,
    runtime_blocker: str | None,
    headline_blocked_reason: str | None,
    z3_available: bool,
    dafny_available: bool,
    guided: int,
    solver_only: int,
    exact_count: int,
) -> str:
    """Return the terminal verdict string required by downstream calibration."""

    if ready:
        return (
            "complete: formal_feedback_v2_ready=true; "
            f"guided_success_count={guided}; solver_only_success_count={solver_only}; "
            f"exact_ground_truth_count={exact_count}"
        )
    if runtime_blocker is not None:
        return (
            "blocked_z3_or_protocol_precondition_failed: "
            f"z3_available={z3_available}; exact_ground_truth_count={exact_count}"
        )
    if headline_blocked_reason:
        return (
            "complete_blocked_headline: formal_feedback_v2_ready=false; "
            f"reason={headline_blocked_reason}; z3_available={z3_available}; "
            f"dafny_available={dafny_available}; guided_success_count={guided}; "
            f"solver_only_success_count={solver_only}"
        )
    return (
        "complete: formal_feedback_v2_ready=false; "
        f"guided_success_count={guided}; solver_only_success_count={solver_only}; "
        f"exact_ground_truth_count={exact_count}"
    )


def model_cache_specs(resolve_gguf_func: ResolveGgufFn, preferred_quant: str) -> list[JsonDict]:
    """Record cache status for every mandated local SOTA GGUF ID."""

    specs = []
    for model in SOTA_GGUF_MODELS:
        path = resolve_gguf_func(model["hf_id"], preferred_quant)
        evidence = file_evidence(path)
        specs.append(
            {
                "name": model["name"],
                "hf_id": model["hf_id"],
                "role": model["role"],
                "preferred_quant": preferred_quant,
                "cached": path is not None,
                "model_path": path,
                "cache_status": "cached" if path else "missing",
                "model_hash_or_cache_path": evidence["hash"],
                "checksum_feasibility": evidence["checksum_feasibility"],
            }
        )
    return specs


def probe_cached_pair(cached_pair_func: CachedPairFn, preferred_quant: str) -> JsonDict:
    """Call cached_sota_pair or the injected equivalent and preserve its result."""

    try:
        pair = cached_pair_func(gpu_indices=(0, 1), preferred_quant=preferred_quant)
    except TypeError:
        pair = cached_pair_func()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "models": [], "error": f"{type(exc).__name__}: {exc}"}
    return {"available": bool(pair), "models": list(pair or []), "error": None}


def inference_substrate(
    *,
    z3_available: bool,
    dafny_available: bool,
    cached_pair_status: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    live_llm_inference: bool,
    selected_model: Mapping[str, Any] | None,
    repo_commit: str,
    python_env: Mapping[str, Any],
) -> JsonDict:
    """Describe the exact solver, model-cache, and execution substrate."""

    return {
        "kind": "live_llm_inference_plus_z3_test_oracle"
        if live_llm_inference
        else "z3_test_oracle_pilot_without_live_guided_llm",
        "z3_available": z3_available,
        "dafny_available": dafny_available,
        "formal_feedback_backend": "z3_and_python_test_oracle",
        "live_llm_inference": live_llm_inference,
        "cached_sota_pair_available": bool(cached_pair_status.get("available")),
        "cached_sota_pair_models": [
            item.get("hf_id") for item in cached_pair_status.get("models", [])
        ],
        "selected_model": dict(selected_model or {}),
        "model_cache_status": [
            {"hf_id": item["hf_id"], "cached": item["cached"], "model_path": item["model_path"]}
            for item in model_specs
        ],
        "legacy_small_models_promoted": False,
        "repo_commit": repo_commit,
        "python_environment": dict(python_env),
    }


def source_artifacts(repo_root: Path) -> list[JsonDict]:
    """Return existence and checksum evidence for protocol and prior failures."""

    rows = []
    for source_id, rel_path, role in SOURCE_REL_PATHS:
        path = repo_root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "role": role,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def _validate_json_candidate(
    payload: Mapping[str, Any], candidate: Mapping[str, Any] | None
) -> JsonDict:
    required = set(str(field) for field in payload.get("required_fields", []))
    if not candidate:
        return _invalid(
            "python_json_test_oracle", "empty_repair_or_spec", {"required_fields": sorted(required)}
        )
    raw_candidate = candidate.get("candidate", candidate)
    try:
        parsed = (
            json.loads(raw_candidate) if isinstance(raw_candidate, str) else dict(raw_candidate)
        )
    except (TypeError, ValueError) as exc:
        return _invalid("python_json_test_oracle", "json_decode_error", {"error": str(exc)})
    missing = sorted(required - set(parsed))
    if missing:
        return _invalid("python_json_test_oracle", "missing_required_fields", {"missing": missing})
    return _valid("python_json_test_oracle", {"parsed_fields": sorted(parsed)})


def _validate_smt_candidate(
    payload: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    z3_module: Any,
) -> JsonDict:
    if z3_module is None:
        return _invalid("z3_solver", "z3_python_unavailable", {})
    variables = [str(var) for var in payload.get("variables", [])]
    assignment = _assignment_from_candidate(candidate, variables)
    if sorted(assignment) != sorted(variables):
        return _invalid("z3_solver", "empty_or_incomplete_assignment", {"variables": variables})
    solver = z3_module.Solver()
    symbols = {name: z3_module.Int(name) for name in variables}
    for name, value in assignment.items():
        solver.add(symbols[name] == int(value))
    for constraint in payload.get("constraints", []):
        solver.add(_constraint_to_z3(str(constraint), symbols, z3_module))
    if solver.check() == z3_module.sat:
        return _valid("z3_solver", {"assignment": assignment})
    return _invalid("z3_solver", "constraint_violation", {"assignment": assignment})


def _validate_python_assertion_candidate(
    payload: Mapping[str, Any], candidate: Mapping[str, Any] | None
) -> JsonDict:
    text = str((candidate or {}).get("candidate_assertion", ""))
    if not text.strip():
        return _invalid("python_ast_execution_oracle", "empty_assertion", {})
    try:
        module = ast.parse(text)
    except SyntaxError as exc:
        return _invalid("python_ast_execution_oracle", "syntax_error", {"error": str(exc)})
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Assert):
        return _invalid("python_ast_execution_oracle", "missing_assert_statement", {})
    test = module.body[0].test
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Eq)
    ):
        return _invalid("python_ast_execution_oracle", "unsupported_assertion_shape", {})
    expected_left = ast.parse(str(payload["expression"]), mode="eval").body
    if ast.dump(test.left) != ast.dump(expected_left):
        return _invalid("python_ast_execution_oracle", "expression_changed", {})
    left = _safe_eval_ast(test.left)
    right = _safe_eval_ast(test.comparators[0])
    if left != right:
        return _invalid(
            "python_ast_execution_oracle", "assertion_failure", {"left": left, "right": right}
        )
    return _valid("python_ast_execution_oracle", {"left": left, "right": right})


def _assignment_from_candidate(
    candidate: Mapping[str, Any] | None, variables: Sequence[str]
) -> dict[str, int]:
    if not candidate:
        return {}
    raw = candidate.get("candidate_assignment", candidate)
    if not isinstance(raw, Mapping):
        return {}
    output = {}
    for var in variables:
        if var in raw:
            output[var] = int(raw[var])
    return output


def _sum_constraint_target(constraints: Sequence[Any], variables: Sequence[str]) -> int | None:
    pattern = (
        re.compile(
            rf"^\s*{re.escape(variables[0])}\s*\+\s*{re.escape(variables[1])}\s*==\s*(-?\d+)\s*$"
        )
        if len(variables) >= 2
        else None
    )
    if pattern is None:
        return None
    for constraint in constraints:
        match = pattern.match(str(constraint))
        if match:
            return int(match.group(1))
    return None


def _constraint_to_z3(text: str, symbols: Mapping[str, Any], z3_module: Any) -> Any:
    match = re.match(r"^\s*([A-Za-z_]\w*)\s*(>=|<=|==)\s*(-?\d+)\s*$", text)
    if match:
        left = symbols[match.group(1)]
        op = match.group(2)
        right = int(match.group(3))
        if op == ">=":
            return left >= right
        if op == "<=":
            return left <= right
        return left == right
    match = re.match(r"^\s*([A-Za-z_]\w*)\s*\+\s*([A-Za-z_]\w*)\s*==\s*(-?\d+)\s*$", text)
    if match:
        return symbols[match.group(1)] + symbols[match.group(2)] == int(match.group(3))
    return z3_module.BoolVal(False)


def _safe_eval_ast(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def _valid(authority: str, diagnostics: Mapping[str, Any]) -> JsonDict:
    return {
        "valid": True,
        "exact_checked": True,
        "exact_authority": authority,
        "failure_reason": "",
        "diagnostics": dict(diagnostics),
    }


def _invalid(authority: str, reason: str, diagnostics: Mapping[str, Any]) -> JsonDict:
    return {
        "valid": False,
        "exact_checked": True,
        "exact_authority": authority,
        "failure_reason": reason,
        "diagnostics": dict(diagnostics),
    }


def extract_text(raw: Mapping[str, Any]) -> str:
    """Extract llama.cpp-style completion text."""

    choices = raw.get("choices") if isinstance(raw, Mapping) else None
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            return str(first.get("text") or first.get("message", {}).get("content") or "")
    return ""


def file_evidence(path: str | None) -> JsonDict:
    """Return bounded checksum evidence for a model-cache path."""

    if path is None:
        return {"hash": None, "checksum_feasibility": {"method": "missing_file"}}
    p = Path(path)
    if not p.is_file():
        return {"hash": path, "checksum_feasibility": {"method": "missing_file"}}
    data = p.read_bytes()
    return {
        "hash": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "checksum_feasibility": {
            "method": "full_sha256",
            "full_sha256_feasible": True,
            "size_bytes": len(data),
        },
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return sha256_text(encoded)


def safe_load_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def safe_load_jsonl(path: Path) -> list[JsonDict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        if line.strip():
            value = json.loads(line)
            rows.append(dict(value) if isinstance(value, Mapping) else {"non_object_row": value})
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - only used in live runs.
    from llama_cpp import Llama

    return Llama(**kwargs)


def repo_commit(repo_root: Path) -> str:  # pragma: no cover - tests inject this probe.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return f"unknown: {type(exc).__name__}"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def python_environment() -> JsonDict:  # pragma: no cover - tests inject this probe.
    return {
        "executable": sys.executable,
        "version": sys.version,
        "virtual_env": sys.prefix,
    }


def main() -> None:  # pragma: no cover - thin manual entrypoint.
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
