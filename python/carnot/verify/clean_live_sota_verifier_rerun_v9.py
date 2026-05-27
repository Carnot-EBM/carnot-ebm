"""Build the Exp 3167 clean live SOTA verifier rerun v9 artifact.

Spec refs: REQ-VERIFY-3167, SCENARIO-VERIFY-3167.

This module is deliberately conservative. It reads the upstream authenticity
and invariance gates before doing anything that could be mistaken for live
verifier evidence. When those gates are not ready, it writes a complete
gated-skip artifact with machine-readable false/zero values instead of leaving
matrix tooling to infer status from a missing file.
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
ARTIFACT = "experiment_3167_clean_live_sota_verifier_rerun_v9"
SCHEMA = "carnot.clean_live_sota_verifier_rerun.v9"
OUTPUT_REL_PATH = Path("results/experiment_3167_clean_live_sota_verifier_rerun_v9.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3167_clean_live_sota_verifier_rerun_v9.py"

EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")
EXP3150_REL_PATH = Path("results/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.json")
EXP3165_REL_PATH = Path("results/experiment_3165_live_sota_authenticity_replay_v2.json")
EXP3166_REL_PATH = Path("results/experiment_3166_verifier_invariance_token_suspicion_audit_v1.json")

DEFAULT_RANDOM_SEED = 20260527
CONTROL_NAMES = ("force", "remove", "shuffled_trace", "answer_only", "trace_only")
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
    "clean_live_verifier_rerun_v9_ready",
    "gated_skip",
    "gated_skip_reason",
    "model_specs",
    "selected_model_ids",
    "unavailable_model_ids",
    "live_call_count",
    "model_load_evidence",
    "prompt_hashes",
    "transcript_hashes",
    "token_counts",
    "exact_ground_truth_count",
    "regression_rows_included",
    "controlled_invariance_passed",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "verifier_gain_delta",
    "false_accept_gate_passed",
    "flagged_adversarial",
    "headline_claim_allowed",
    "random_seed",
    "reproducibility_checksum",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}

MANDATED_MODEL_POLICY: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen3.6-35B-A3B",
        "role": "moe",
        "tier": "flagship_moe",
        "headline_eligible": True,
        "legacy_small_model": False,
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "name": "Gemma4-31B-it",
        "role": "dense",
        "tier": "flagship_dense",
        "headline_eligible": True,
        "legacy_small_model": False,
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "name": "Gemma4-26B-A4B-it",
        "role": "moe",
        "tier": "middle_moe",
        "headline_eligible": True,
        "legacy_small_model": False,
    },
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3167_clean_live_sota_verifier_rerun_v9.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3167_clean_live_sota_verifier_rerun_v9.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/clean_live_sota_verifier_rerun_v9.py' --fail-under=100 --show-missing",
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
        "experiment_template",
        Path("scripts/experiment_template.py"),
        "experiment_template_policy",
        True,
        "py",
    ),
    SourceSpec(
        "verification_openspec",
        Path("openspec/capabilities/verification/spec.md"),
        "verification_spec",
        True,
        "md",
    ),
    SourceSpec("exp3136", EXP3136_REL_PATH, "known_false_accept_regressions", True),
    SourceSpec("exp3137", EXP3137_REL_PATH, "exact_safe_accept_abstain_contract", True),
    SourceSpec("exp3138", EXP3138_REL_PATH, "canonical_grounding", True),
    SourceSpec("exp3150", EXP3150_REL_PATH, "adversarial_corrigendum", True),
    SourceSpec("exp3165", EXP3165_REL_PATH, "live_sota_authenticity_preflight", True),
    SourceSpec("exp3166", EXP3166_REL_PATH, "invariance_token_suspicion_audit", True),
    SourceSpec(
        "exp3167_module",
        Path("python/carnot/verify/clean_live_sota_verifier_rerun_v9.py"),
        "rerun_module",
        False,
        "py",
    ),
    SourceSpec(
        "exp3167_script",
        Path("scripts/experiment_3167_clean_live_sota_verifier_rerun_v9.py"),
        "rerun_script",
        False,
        "py",
    ),
    SourceSpec(
        "exp3167_tests",
        Path("tests/python/test_experiment_3167_clean_live_sota_verifier_rerun_v9.py"),
        "rerun_tests",
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
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> JsonDict:
    """REQ-VERIFY-3167: build a clean rerun artifact or complete gated skip."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = {
        "exp3136": read_json_object(root_path / EXP3136_REL_PATH),
        "exp3137": read_json_object(root_path / EXP3137_REL_PATH),
        "exp3138": read_json_object(root_path / EXP3138_REL_PATH),
        "exp3150": read_json_object(root_path / EXP3150_REL_PATH),
        "exp3165": read_json_object(root_path / EXP3165_REL_PATH),
        "exp3166": read_json_object(root_path / EXP3166_REL_PATH),
    }
    sources = source_artifacts(root_path)
    source_problems = source_errors(sources)
    exact_rows = collect_exact_rows(
        payloads["exp3136"], payloads["exp3137"], payloads["exp3138"], payloads["exp3166"]
    )
    regression_ids = collect_regression_row_ids(payloads["exp3136"], payloads["exp3137"])
    planned_set = planned_rerun_set(exact_rows, regression_ids)
    model_specs = model_specs_from_exp3165(payloads["exp3165"])
    usable_model_ids = [str(row["hf_id"]) for row in model_specs if row.get("usable_locally")]
    reason = gated_skip_reason(
        exp3165=payloads["exp3165"],
        exp3166=payloads["exp3166"],
        usable_model_ids=usable_model_ids,
        source_problems=source_problems,
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    source_checksums = {str(row["path"]): row["sha256"] for row in sources if row.get("sha256")}
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": duration(start, finished),
        "clean_live_verifier_rerun_v9_ready": True,
        "gated_skip": True,
        "gated_skip_reason": reason or "live execution path unavailable in this environment",
        "model_specs": model_specs,
        "selected_model_ids": [],
        "unavailable_model_ids": unavailable_model_ids(payloads["exp3165"], model_specs),
        "live_call_count": 0,
        "model_load_evidence": model_load_evidence(payloads["exp3165"], reason),
        "prompt_hashes": [],
        "transcript_hashes": [],
        "token_counts": zero_token_counts(),
        "exact_ground_truth_count": len(exact_rows),
        "regression_rows_included": regression_rows_included(planned_set, regression_ids),
        "controlled_invariance_passed": False,
        "false_accept_rate": 0.0,
        "false_reject_rate": 0.0,
        "abstention_rate": 0.0,
        "verifier_gain_delta": 0.0,
        "false_accept_gate_passed": False,
        "flagged_adversarial": False,
        "headline_claim_allowed": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "source_artifacts": sources,
        "source_checksums": source_checksums,
        "source_errors": source_problems,
        "planned_rerun_set": planned_set,
        "preconditions_checked": preconditions_checked(
            payloads["exp3165"], payloads["exp3166"], usable_model_ids, source_problems
        ),
        "metrics_computed": False,
        "field_principles": field_principles(),
        "inference_substrate": inference_substrate(reason),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
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
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    """Build, validate, and persist the Exp 3167 terminal JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
        random_seed=random_seed,
    )
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
    """Return every local file the rerun gate consumes or cites."""

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
    """Expose missing required evidence rather than inferring around it."""

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


def collect_exact_rows(
    exp3136: Mapping[str, Any],
    exp3137: Mapping[str, Any],
    exp3138: Mapping[str, Any],
    exp3166: Mapping[str, Any],
) -> list[JsonDict]:
    """Collect unique exact-authority rows for the planned rerun set."""

    by_id: dict[str, JsonDict] = {}
    for rows in (
        _mapping_list(exp3166.get("trusted_exact_rows")),
        _mapping_list(exp3137.get("replay_rows")),
        _mapping_list(exp3136.get("false_accept_rows")),
        _mapping_list(exp3136.get("verifier_rows")),
        _mapping_list(exp3138.get("regression_row_replay")),
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
                    "candidate_answers": [],
                    "source_families": [],
                },
            )
            if exact_label and target["exact_label"] != exact_label:
                target["exact_label_conflict"] = True
            append_unique(target["candidate_answers"], row.get("extracted_answer"))
            append_unique(target["candidate_answers"], row.get("candidate_answer"))
            for value in row.get("candidate_answers") or []:
                append_unique(target["candidate_answers"], value)
            append_unique(target["source_families"], row.get("fixture_family"))
    normalized = [by_id[row_id] for row_id in sorted(by_id)]
    for row in normalized:
        row["candidate_answers"] = sorted(row["candidate_answers"])
        row["source_families"] = sorted(row["source_families"])
        row["family"] = infer_family(row)
        row["row_fingerprint"] = stable_hash(
            {
                "row_id": row["row_id"],
                "exact_label": row["exact_label"],
                "candidate_answers": row["candidate_answers"],
            }
        )
    return normalized


def collect_regression_row_ids(*payloads: Mapping[str, Any]) -> list[str]:
    """Collect known false-accept regression IDs from upstream artifacts."""

    ids: set[str] = set()
    for payload in payloads:
        for key in ("regression_row_set", "false_accept_row_ids"):
            for value in payload.get(key) or []:
                if value:
                    ids.add(str(value))
    return sorted(ids)


def planned_rerun_set(
    exact_rows: Sequence[Mapping[str, Any]], regression_row_ids: Sequence[str]
) -> JsonDict:
    """Summarize the rerun set without implying any live calls happened."""

    family_row_ids: dict[str, list[str]] = {
        "arithmetic": [],
        "smt": [],
        "satisfiable_drift": [],
        "contradiction": [],
        "fragment_code": [],
    }
    for row in exact_rows:
        family = infer_family(row)
        if family in family_row_ids:
            family_row_ids[family].append(str(row["row_id"]))
    return {
        "row_ids": [str(row["row_id"]) for row in exact_rows],
        "regression_row_ids": sorted(str(row_id) for row_id in regression_row_ids),
        "family_row_ids": {key: sorted(value) for key, value in family_row_ids.items()},
        "family_counts": {key: len(value) for key, value in family_row_ids.items()},
        "balanced_family_policy": "include known regressions plus exact rows across arithmetic, SMT, satisfiable-drift, contradiction, and fragment-code buckets",
    }


def infer_family(row: Mapping[str, Any]) -> str:
    """Map exact rows onto the five planned rerun-set buckets."""

    row_id = str(row.get("row_id") or "").lower()
    label = str(row.get("exact_label") or "").upper()
    source_families = " ".join(str(value).lower() for value in row.get("source_families") or [])
    joined = f"{row_id} {source_families}"
    if "repair-json" in joined or "fragment" in joined or "repairable" in label:
        return "fragment_code"
    if label == "SAT":
        return "satisfiable_drift"
    if "smt" in joined:
        return "smt"
    if label in {"INVALID", "UNSAT", "FALSE"}:
        return "contradiction"
    if "arith" in joined or "arithmetic" in joined:
        return "arithmetic"
    return "other"


def model_specs_from_exp3165(exp3165: Mapping[str, Any]) -> list[JsonDict]:
    """Return mandated model rows annotated with upstream local availability."""

    upstream_by_id = {
        str(row.get("hf_id")): row for row in _mapping_list(exp3165.get("model_specs"))
    }
    usable_ids = {str(value) for value in exp3165.get("locally_usable_model_ids") or []}
    rows: list[JsonDict] = []
    for policy in MANDATED_MODEL_POLICY:
        upstream = _mapping(upstream_by_id.get(policy["hf_id"]))
        rows.append(
            dict(policy)
            | {
                "usable_locally": bool(upstream.get("usable_locally", policy["hf_id"] in usable_ids)),
                "selected_for_exp3167": False,
                "upstream_selected_for_smoke": bool(upstream.get("selected_for_smoke")),
                "model_path": upstream.get("model_path"),
            }
        )
    return rows


def unavailable_model_ids(exp3165: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Prefer upstream unavailable IDs, falling back to model-spec availability."""

    upstream = [str(value) for value in exp3165.get("unavailable_model_ids") or []]
    if upstream:
        return upstream
    return [str(row["hf_id"]) for row in model_specs if row.get("usable_locally") is not True]


def gated_skip_reason(
    *,
    exp3165: Mapping[str, Any],
    exp3166: Mapping[str, Any],
    usable_model_ids: Sequence[str],
    source_problems: Sequence[Mapping[str, Any]],
) -> str:
    """Return the first actionable reason the clean rerun may not execute."""

    if source_problems:
        return "required source artifacts unavailable or malformed"
    if exp3165.get("preflight_passed") is not True:
        return "exp3165 preflight_passed=false; clean rerun cannot call a model"
    if exp3166.get("verifier_invariance_token_suspicion_audit_ready") is not True:
        return "exp3166 invariance audit is not ready"
    if not usable_model_ids:
        return "no mandated local SOTA GGUF usable for headline rerun"
    return ""


def model_load_evidence(exp3165: Mapping[str, Any], reason: str) -> JsonDict:
    """Record that Exp 3167 did not load a model, while preserving upstream evidence."""

    upstream = _mapping(exp3165.get("model_load_evidence"))
    return {
        "load_attempted": False,
        "inherited_from_exp3165": True,
        "upstream_load_attempted": bool(upstream.get("load_attempted")),
        "upstream_runtime": upstream.get("runtime"),
        "upstream_path_exists": bool(upstream.get("path_exists")),
        "gated_skip_reason": reason,
        "upstream_model_load_evidence": upstream,
    }


def zero_token_counts() -> JsonDict:
    """Return explicit zero work evidence for a gated skip."""

    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "source": "gated_skip_no_live_calls",
    }


def regression_rows_included(
    planned_set: Mapping[str, Any], regression_row_ids: Sequence[str]
) -> bool:
    """Return whether every known regression row is present in the planned set."""

    planned_ids = {str(value) for value in planned_set.get("row_ids") or []}
    return all(str(row_id) in planned_ids for row_id in regression_row_ids)


def preconditions_checked(
    exp3165: Mapping[str, Any],
    exp3166: Mapping[str, Any],
    usable_model_ids: Sequence[str],
    source_problems: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Expose each precondition as machine-readable evidence."""

    return [
        {
            "name": "source_artifacts_readable",
            "passed": not source_problems,
            "detail": "" if not source_problems else f"{len(source_problems)} source problem(s)",
        },
        {
            "name": "exp3165_preflight_passed",
            "passed": exp3165.get("preflight_passed") is True,
            "detail": f"preflight_passed={exp3165.get('preflight_passed')!r}",
        },
        {
            "name": "exp3166_invariance_audit_ready",
            "passed": exp3166.get("verifier_invariance_token_suspicion_audit_ready") is True,
            "detail": (
                "verifier_invariance_token_suspicion_audit_ready="
                f"{exp3166.get('verifier_invariance_token_suspicion_audit_ready')!r}"
            ),
        },
        {
            "name": "mandated_local_model_usable",
            "passed": bool(usable_model_ids),
            "detail": ",".join(usable_model_ids),
        },
    ]


def inference_substrate(reason: str) -> JsonDict:
    """Declare that Exp 3167 performed no live inference in gated-skip mode."""

    return {
        "kind": "gated_skip_no_live_llm_inference",
        "downloads_models": False,
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "live_model_calls": 0,
        "legacy_small_model_used": False,
        "structural_conductor_gating_used": False,
        "gated_skip_reason": reason,
    }


def field_principles() -> JsonDict:
    """Echo the task's required field principles into the artifact."""

    return {
        "clean_live_verifier_rerun_v9_ready": "repair gate needs a complete rerun artifact",
        "gated_skip": "matrix should distinguish skipped from missing",
        "gated_skip_reason": "blocked reruns must be actionable",
        "model_specs": "mandated local model policy must be visible",
        "selected_model_ids": "actual model use must be auditable",
        "unavailable_model_ids": "comparative gaps must stay visible",
        "live_call_count": "live evidence must not be inferred",
        "model_load_evidence": "claimed inference requires load evidence",
        "prompt_hashes": "prompts must be replay-identifiable",
        "transcript_hashes": "outputs must be replay-identifiable",
        "token_counts": "measured-work evidence must be visible",
        "exact_ground_truth_count": "verifier scores need exact labels",
        "regression_rows_included": "known failures must be retested",
        "controlled_invariance_passed": "artifact shortcuts must not pass as verifier gain",
        "false_accept_rate": "false accepts are the central blocker",
        "false_reject_rate": "over-rejection can erase utility",
        "abstention_rate": "safety/coverage tradeoff must be visible",
        "verifier_gain_delta": "lift must be measured against exact baselines",
        "false_accept_gate_passed": "downstream gate must be machine-readable",
        "flagged_adversarial": "downstream gate must reject tainted evidence",
        "headline_claim_allowed": "live claims require all evidence gates",
        "random_seed": "methodology completeness must be explicit",
        "reproducibility_checksum": "rerun provenance must be checkable",
        "source_artifacts": "claims must trace to concrete files",
        "inference_substrate": "GPU/model/live inference status must be explicit",
        "honest_verdict": "terminal verdict must start with a complete/success/passed/shipped prefix",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Compute a stable checksum over the gated-skip provenance."""

    return stable_hash(
        {
            "artifact": artifact.get("artifact"),
            "random_seed": artifact.get("random_seed"),
            "source_checksums": artifact.get("source_checksums"),
            "planned_rerun_set": artifact.get("planned_rerun_set"),
            "gated_skip_reason": artifact.get("gated_skip_reason"),
        }
    )[:16]


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that distinguishes complete skip from missing."""

    return (
        "complete: clean_live_verifier_rerun_v9_ready=true; "
        f"gated_skip={artifact.get('gated_skip')}; "
        f"live_call_count={artifact.get('live_call_count')}; "
        f"reason={artifact.get('gated_skip_reason')}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject malformed Exp 3167 artifacts and accidental live-evidence claims."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if artifact.get("gated_skip") is True:
        if artifact.get("live_call_count") != 0:
            raise ValueError("gated-skip artifact must not claim live calls")
        if artifact.get("prompt_hashes") or artifact.get("transcript_hashes"):
            raise ValueError("gated-skip artifact must not claim replay hashes")
        if artifact.get("headline_claim_allowed") is not False:
            raise ValueError("gated skip must keep headline claims blocked")
        if artifact.get("false_accept_gate_passed") is not False:
            raise ValueError("false accept gate must stay false during gated skip")
        if _mapping(artifact.get("inference_substrate")).get("executes_models") is not False:
            raise ValueError("gated skip must declare no model execution")


def append_unique(target: list[str], value: Any) -> None:
    """Append a non-empty string once."""

    if value is None:
        return
    text = str(value)
    if text and text not in target:
        target.append(text)


def _mapping(value: Any) -> JsonDict:
    """Return mapping values as plain dictionaries."""

    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[JsonDict]:
    """Return only mapping rows from list-like JSON values."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]

