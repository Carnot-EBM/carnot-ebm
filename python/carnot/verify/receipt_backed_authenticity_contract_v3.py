"""Build the Exp 3178 receipt-backed authenticity contract v3.

Spec refs: REQ-VERIFY-3178, SCENARIO-VERIFY-3178.

This module is a contract writer, not a model runner. It reads the v2
authenticity contract and the blocked `.294` replay artifacts, then defines
the receipt surface future live tasks must satisfy. The important distinction
is that environmental failures are not interchangeable: a missing HF cache, a
missing loader, unhealthy CUDA, and an admissible CPU smoke each imply different
downstream permissions. Keeping those causes separate prevents a cheap wiring
smoke from becoming headline verifier evidence by accident.
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
ARTIFACT = "experiment_3178_receipt_backed_authenticity_contract_v3"
SCHEMA = "carnot.receipt_backed_authenticity_contract.v3"
OUTPUT_REL_PATH = Path("results/experiment_3178_receipt_backed_authenticity_contract_v3.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3178_receipt_backed_authenticity_contract_v3.py"

EXP3164_REL_PATH = Path("results/experiment_3164_duration_corrected_authenticity_contract_v2.json")
EXP3165_REL_PATH = Path("results/experiment_3165_live_sota_authenticity_replay_v2.json")
EXP3167_REL_PATH = Path("results/experiment_3167_clean_live_sota_verifier_rerun_v9.json")
EXP3176_REL_PATH = Path("results/experiment_3176_capstone_v294.json")

SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)

SUBSTRATE_CLASSES = (
    "model_cache_missing",
    "loader_missing",
    "cuda_unavailable",
    "cuda_available_unhealthy",
    "cpu_fallback_receipt_only",
    "full_local_sota_receipt",
)

REQUIRED_RECEIPT_FIELDS = (
    "selected_model_id",
    "model_path",
    "model_file_hash",
    "loader_name",
    "substrate_used",
    "prompt_hashes",
    "transcript_hashes",
    "token_counts",
    "random_seed",
    "wall_clock_s",
    "command_hash",
    "subprocess_return_code",
    "stderr_tail",
    "throughput_plausibility",
    "replay_count",
)

DEFAULT_INHERITED_V2_FIELDS = (
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
)

REQUIRED_FIELDS = {
    "receipt_backed_authenticity_contract_v3_ready",
    "inherited_v2_contract_fields",
    "substrate_classification_policy",
    "required_receipt_fields",
    "cpu_fallback_policy",
    "fake_evidence_rejection_criteria",
    "clean_rerun_unlock_requirements",
    "headline_claim_policy",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}

SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("experiment_template_policy", Path("scripts/experiment_template.py"), True, "python"),
    ("post_294_research_references", Path("research-references.md"), True, "text"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("exp3164_v2_contract", EXP3164_REL_PATH, True, "json"),
    ("exp3165_live_sota_replay_v2", EXP3165_REL_PATH, True, "json"),
    ("exp3167_clean_live_rerun_v9", EXP3167_REL_PATH, True, "json"),
    ("exp3176_capstone_v294", EXP3176_REL_PATH, True, "json"),
    (
        "exp3178_module",
        Path("python/carnot/verify/receipt_backed_authenticity_contract_v3.py"),
        False,
        "python",
    ),
    (
        "exp3178_script",
        Path("scripts/experiment_3178_receipt_backed_authenticity_contract_v3.py"),
        False,
        "python",
    ),
    (
        "exp3178_tests",
        Path("tests/python/test_experiment_3178_receipt_backed_authenticity_contract_v3.py"),
        False,
        "python",
    ),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3178_receipt_backed_authenticity_contract_v3.py -q --no-cov",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3178_receipt_backed_authenticity_contract_v3.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/receipt_backed_authenticity_contract_v3.py' --fail-under=100 --show-missing",
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
    """REQ-VERIFY-3178: build the no-inference v3 receipt contract artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3164 = read_json_object(root_path / EXP3164_REL_PATH)
    exp3165 = read_json_object(root_path / EXP3165_REL_PATH)
    exp3167 = read_json_object(root_path / EXP3167_REL_PATH)
    exp3176 = read_json_object(root_path / EXP3176_REL_PATH)
    sources = source_artifacts(root_path)
    blockers = contract_blockers(exp3164, exp3165, exp3167, exp3176)
    substrate_observation = exp3165_substrate_observation(exp3165)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "receipt_backed_authenticity_contract_v3_ready": not blockers,
        "inherited_v2_contract_fields": inherited_v2_contract_fields(exp3164),
        "v2_measured_work_requirements": v2_measured_work_requirements(exp3164),
        "exp3165_blocker_reason": exp3165_blocker_reason(exp3165),
        "exp3165_substrate_observation": substrate_observation,
        "substrate_classification_policy": substrate_classification_policy(),
        "required_receipt_fields": list(REQUIRED_RECEIPT_FIELDS),
        "cpu_fallback_policy": cpu_fallback_policy(),
        "fake_evidence_rejection_criteria": fake_evidence_rejection_criteria(exp3164),
        "clean_rerun_unlock_requirements": clean_rerun_unlock_requirements(),
        "headline_claim_policy": headline_claim_policy(),
        "source_artifacts": sources,
        "source_checksums": {row["path"]: row.get("sha256") for row in sources},
        "field_principles": field_principles(),
        "contract_blockers": blockers,
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
    """Build and persist the Exp 3178 terminal JSON artifact."""

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
    """Read one JSON object, treating absent or malformed sources as blockers."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return all files that make the v3 contract traceable."""

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
    """Return a checksum for a present source file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_blockers(
    exp3164: Mapping[str, Any],
    exp3165: Mapping[str, Any],
    exp3167: Mapping[str, Any],
    exp3176: Mapping[str, Any],
) -> list[str]:
    """List missing contract authorities without converting them into evidence."""

    blockers: list[str] = []
    if exp3164.get("duration_corrected_authenticity_contract_v2_ready") is not True:
        blockers.append("missing_exp3164_v2_contract")
    if exp3165.get("live_sota_authenticity_replay_v2_ready") is not True:
        blockers.append("missing_exp3165_replay_artifact")
    if exp3167.get("clean_live_verifier_rerun_v9_ready") is not True:
        blockers.append("missing_exp3167_clean_rerun_artifact")
    if exp3176.get("capstone_v294_ready") is not True:
        blockers.append("missing_exp3176_capstone_artifact")
    return blockers


def inherited_v2_contract_fields(exp3164: Mapping[str, Any]) -> list[str]:
    """Return v2 machine-checkable fields, falling back to the frozen v2 list."""

    fields = [
        str(field)
        for field in exp3164.get("required_preflight_fields", [])
        if isinstance(field, str) and field
    ]
    return unique(fields or DEFAULT_INHERITED_V2_FIELDS)


def v2_measured_work_requirements(exp3164: Mapping[str, Any]) -> list[JsonDict]:
    """Copy the v2 measured-work rows that remain valid in v3."""

    rows = []
    for row in exp3164.get("measured_work_requirements", []):
        if isinstance(row, Mapping):
            rows.append(
                {
                    "requirement": str(row.get("requirement") or ""),
                    "source_field": str(row.get("source_field") or ""),
                    "observed": row.get("observed") is True,
                }
            )
    return rows


def exp3165_blocker_reason(exp3165: Mapping[str, Any]) -> str:
    """Extract the exact `.294` replay blocker string."""

    reason = exp3165.get("blocked_reason")
    if isinstance(reason, str) and reason:
        return reason
    return "missing_exp3165_blocked_reason"


def exp3165_substrate_observation(exp3165: Mapping[str, Any]) -> JsonDict:
    """Classify the prior replay without probing hardware again."""

    substrate = mapping(exp3165.get("inference_substrate"))
    gpu_probe = mapping(substrate.get("gpu_probe"))
    load = mapping(exp3165.get("model_load_evidence"))
    reason = exp3165_blocker_reason(exp3165)
    classified_as = classify_substrate(exp3165, reason, gpu_probe, load)
    return {
        "blocked_reason": reason,
        "classified_as": classified_as,
        "locally_usable_model_ids": [
            str(model_id) for model_id in exp3165.get("locally_usable_model_ids", [])
        ],
        "selected_model_id": load.get("selected_model_id"),
        "selected_model_path": load.get("selected_model_path"),
        "loader_name": load.get("runtime") or substrate.get("runtime"),
        "path_exists": load.get("path_exists") is True,
        "cuda_available": gpu_probe.get("cuda_available") is True,
        "gpu_count": int_or_zero(gpu_probe.get("gpu_count")),
        "torch_cuda_returncode": mapping(gpu_probe.get("torch_cuda_probe")).get("returncode"),
        "nvidia_smi_returncode": mapping(gpu_probe.get("nvidia_smi_inventory")).get("returncode"),
        "live_call_count": int_or_zero(exp3165.get("live_call_count")),
        "preflight_passed": exp3165.get("preflight_passed") is True,
    }


def classify_substrate(
    exp3165: Mapping[str, Any],
    reason: str,
    gpu_probe: Mapping[str, Any],
    load: Mapping[str, Any],
) -> str:
    """Map prior replay evidence into one of the v3 substrate classes."""

    lower_reason = reason.lower()
    usable_models = [
        model_id for model_id in exp3165.get("locally_usable_model_ids", []) if model_id
    ]
    if not usable_models and load.get("path_exists") is not True:
        return "model_cache_missing"
    if "loader" in lower_reason or "llama_cpp" in lower_reason and "missing" in lower_reason:
        return "loader_missing"
    if gpu_probe.get("cuda_available") is not True:
        return "cuda_unavailable"
    if exp3165.get("preflight_passed") is True and int_or_zero(exp3165.get("live_call_count")) >= 2:
        return "full_local_sota_receipt"
    if "cpu" in lower_reason:
        return "cpu_fallback_receipt_only"
    return "cuda_available_unhealthy"


def substrate_classification_policy() -> JsonDict:
    """Define the six v3 substrate classes and their downstream permissions."""

    return {
        "policy_version": "v3",
        "classification_order": list(SUBSTRATE_CLASSES),
        "classes": {
            "model_cache_missing": {
                "definition": "No mandated local SOTA GGUF path is present with nonzero size.",
                "blocks_live_model_call": True,
                "headline_eligible": False,
                "clean_rerun_unlock_allowed": False,
            },
            "loader_missing": {
                "definition": "A model path exists, but the declared loader/import is unavailable.",
                "blocks_live_model_call": True,
                "headline_eligible": False,
                "clean_rerun_unlock_allowed": False,
            },
            "cuda_unavailable": {
                "definition": "A model path and loader may exist, but CUDA is not available to torch/runtime.",
                "blocks_live_model_call": True,
                "cpu_fallback_class": "cpu_fallback_receipt_only",
                "headline_eligible": False,
                "clean_rerun_unlock_allowed": False,
            },
            "cuda_available_unhealthy": {
                "definition": "CUDA is visible, but load or smoke subprocess evidence is unhealthy.",
                "blocks_live_model_call": True,
                "headline_eligible": False,
                "clean_rerun_unlock_allowed": False,
            },
            "cpu_fallback_receipt_only": {
                "definition": "A bounded CPU smoke may prove command and transcript wiring only.",
                "blocks_live_model_call": False,
                "headline_eligible": False,
                "clean_rerun_unlock_allowed": False,
            },
            "full_local_sota_receipt": {
                "definition": "Mandated SOTA GGUF, loader, CUDA, transcripts, tokens, and repeats pass.",
                "blocks_live_model_call": False,
                "headline_eligible": True,
                "clean_rerun_unlock_allowed": True,
            },
        },
    }


def cpu_fallback_policy() -> JsonDict:
    """State exactly what CPU fallback can and cannot prove."""

    return {
        "admissible_for_receipt_wiring": True,
        "allowed_substrate_class": "cpu_fallback_receipt_only",
        "minimum_receipt_fields_required": list(REQUIRED_RECEIPT_FIELDS),
        "headline_verifier_benchmark_allowed": False,
        "clean_rerun_unlock_allowed": False,
        "must_use_loud_label": True,
        "may_inform_next_debug_step": True,
    }


def fake_evidence_rejection_criteria(exp3164: Mapping[str, Any]) -> list[str]:
    """Keep v2 rejection rules and add v3 substrate-specific failure modes."""

    inherited = [
        str(item)
        for item in exp3164.get("fake_evidence_rejection_criteria", [])
        if isinstance(item, str) and item
    ]
    v3 = [
        "reject missing model/cache proof before any live SOTA receipt",
        "reject missing loader/import proof for the declared loader_name",
        "reject missing CUDA health evidence for headline claims",
        "reject CPU-only evidence promoted as headline verifier benchmark",
        "reject missing prompt_hashes or transcript_hashes",
        "reject reused stale transcript hashes unless replay_count declares reuse",
        "reject missing token_counts, random_seed, command_hash, or subprocess_return_code",
        "reject wall-clock claims not supported by command/subprocess output",
        "reject uncontrolled subprocess return codes or missing stderr_tail",
        "reject impossible throughput_plausibility values",
        "reject one-prompt smoke promoted as benchmark evidence",
    ]
    return unique(inherited + v3)


def clean_rerun_unlock_requirements() -> list[str]:
    """List the gates Exp 3181 must satisfy before headline eligibility."""

    return [
        "receipt_backed_authenticity_contract_v3_ready=true",
        "exp3179.local_sota_receipt_smoke_v3_ready=true",
        "exp3179.substrate_classification=full_local_sota_receipt",
        "exp3179.preflight_passed=true",
        "exp3179.replay_count>=2",
        "all_required_receipt_fields_present=true",
        "throughput_plausibility.passed=true",
        "controlled_invariance_passed=true",
        "exact_authority_scoring_passed=true",
        "false_accept_gate_passed=true",
        "headline_claim_policy_passed=true",
    ]


def headline_claim_policy() -> JsonDict:
    """Prevent smoke or fallback receipts from becoming benchmark claims."""

    return {
        "policy_version": "v3",
        "cpu_fallback_headline_allowed": False,
        "one_prompt_smoke_headline_allowed": False,
        "smoke_test_role": "receipt wiring and substrate diagnosis only",
        "headline_requires": [
            "full_local_sota_receipt substrate",
            "at least two deterministic replay receipts or a predeclared panel budget",
            "exact-authority scoring rows",
            "controlled invariance passed",
            "false-accept gate passed",
            "no inherited adversarial methodology flags",
        ],
        "clean_verifier_rerun_requires_exact_authority": True,
        "clean_verifier_rerun_requires_controlled_invariance": True,
    }


def field_principles() -> JsonDict:
    """Explain why the required top-level fields exist."""

    return {
        "receipt_backed_authenticity_contract_v3_ready": (
            "live reruns need an explicit contract"
        ),
        "substrate_classification_policy": (
            "blockers must distinguish cache, loader, CUDA, and CPU causes"
        ),
        "required_receipt_fields": "authenticity should be tied to replayable observed work",
        "headline_claim_policy": "smoke tests should not become headline evidence",
        "inference_substrate": "aggregation work must declare no live model inference",
    }


def inference_substrate() -> JsonDict:
    """Declare that Exp 3178 performs only artifact aggregation and policy writing."""

    return {
        "kind": "aggregation_and_contract_no_live_inference",
        "downloads_models": False,
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "live_model_calls": 0,
        "uses_checked_in_artifacts_only": True,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict compatible with conductor artifact scanning."""

    if artifact.get("receipt_backed_authenticity_contract_v3_ready") is True:
        reason = artifact.get("exp3165_blocker_reason") or "none"
        return (
            "complete: receipt_backed_authenticity_contract_v3_ready=true; "
            f"exp3165_blocker_reason={reason}"
        )
    blockers = ",".join(str(item) for item in artifact.get("contract_blockers", []))
    return f"blocked_missing_v2_contract: receipt_backed_authenticity_contract_v3_ready=false; blockers={blockers}"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail fast when the contract could be misread as live or headline evidence."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    classes = set(mapping(artifact.get("substrate_classification_policy")).get("classes", {}))
    if classes != set(SUBSTRATE_CLASSES):
        raise ValueError(f"substrate classes must be exactly {SUBSTRATE_CLASSES}")
    receipt_fields = set(str(field) for field in artifact.get("required_receipt_fields", []))
    if receipt_fields != set(REQUIRED_RECEIPT_FIELDS):
        raise ValueError("required receipt fields do not match the v3 contract")
    cpu_policy = mapping(artifact.get("cpu_fallback_policy"))
    if cpu_policy.get("headline_verifier_benchmark_allowed") is not False:
        raise ValueError("CPU fallback must not be headline verifier evidence")
    if cpu_policy.get("clean_rerun_unlock_allowed") is not False:
        raise ValueError("CPU fallback must not unlock clean rerun")
    substrate = mapping(artifact.get("inference_substrate"))
    if substrate.get("executes_models") is not False or int_or_zero(substrate.get("live_model_calls")) != 0:
        raise ValueError("Exp 3178 must declare no live model inference")
    if artifact.get("receipt_backed_authenticity_contract_v3_ready") is True and artifact.get(
        "contract_blockers"
    ):
        raise ValueError("ready contract cannot have blockers")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (verdict.startswith(SUCCESS_PREFIXES) or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must start with a terminal success or blocked prefix")


def duration(started_s: float, finished_s: float) -> float:
    """Return a stable positive wall-clock duration."""

    return round(max(0.0, float(finished_s) - float(started_s)), 6)


def mapping(value: Any) -> JsonDict:
    """Normalize arbitrary JSON values into a mutable mapping."""

    return dict(value) if isinstance(value, Mapping) else {}


def int_or_zero(value: Any) -> int:
    """Coerce small JSON counters without raising on missing evidence."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def unique(items: Sequence[str]) -> list[str]:
    """Preserve order while removing duplicates and empty strings."""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
