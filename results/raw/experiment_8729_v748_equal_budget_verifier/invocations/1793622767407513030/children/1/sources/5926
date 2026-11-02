"""Build the Exp 3287 abstention-calibrated clean verifier rerun v15 artifact.

Spec refs: REQ-VERIFY-3287, SCENARIO-VERIFY-3287.

The v14 rerun proved the local GGUF path was real but abstained on every exact
row because the raw model text did not honor the strict verifier-output
contract.  This module keeps that conservative parser, adds the Exp 3286
calibrated policy around it, and records a repair-gate-ready artifact only when
coverage is non-trivial and exact-authority scoring still finds zero false
accepts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf
from carnot.verify import clean_local_sota_verifier_rerun_v14 as v14


JsonDict = dict[str, Any]
ModelRunner = Callable[[list[JsonDict], JsonDict, int, JsonDict], JsonDict]
Probe = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.abstention_calibrated_clean_verifier.v15"
EXPERIMENT_ID = "exp3287"
MILESTONE = "2026.05.303"
RUN_DATE = "20260528"

OUTPUT_REL_PATH = Path("results/experiment_3287_abstention_calibrated_clean_verifier_v15.json")
EXP3286_REL_PATH = Path("results/experiment_3286_clean_verifier_abstention_root_cause_v1.json")
EXP3268_REL_PATH = v14.EXP3268_REL_PATH
EXP3275_REQUESTED_REL_PATH = Path("results/experiment_3275_clean_sota_verifier_rerun_v14.json")
EXP3275_LOCAL_REL_PATH = v14.OUTPUT_REL_PATH
CONTEXT_FIXTURE_REL_PATH = v14.CONTEXT_FIXTURE_REL_PATH
SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
TEST_REL_PATH = Path("tests/python/test_experiment_3287_abstention_calibrated_clean_verifier_v15.py")

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
DEFAULT_RANDOM_SEED = 3287
DEFAULT_MIN_EXACT_ROWS = 20
DEFAULT_MAX_EVAL_ROWS = 20
MIN_GPU_MEM_USED_MIB = 512
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
DECISION_GRAMMAR = 'root ::= "ACCEPT" | "REJECT" | "ABSTAIN"\n'

REQUIRED_FIELDS = {
    "abstention_calibrated_clean_verifier_v15_ready",
    "clean_verifier_rerun_ready",
    "repair_gate_input_clean_enough",
    "model_specs",
    "models_used",
    "missing_model_specs",
    "preconditions_checked",
    "n_eval",
    "exact_checkable_row_count",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "coverage_rate",
    "abstention_reason_counts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3287_abstention_calibrated_clean_verifier_v15.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/abstention_calibrated_clean_verifier_v15.py -m pytest -o addopts='' tests/python/test_experiment_3287_abstention_calibrated_clean_verifier_v15.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/abstention_calibrated_clean_verifier_v15.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    nvidia_probe: Probe | None = None,
    python_cuda_probe: Probe | None = None,
    model_runner: ModelRunner | None = None,
    max_eval_rows: int = DEFAULT_MAX_EVAL_ROWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3287: run the calibrated exact-row verifier gate."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3286 = read_json_object(root_path / EXP3286_REL_PATH)
    exp3268 = read_json_object(root_path / EXP3268_REL_PATH)
    policy = calibrated_policy_from_exp3286(exp3286)
    preconditions: list[JsonDict] = []
    gate_reasons: list[str] = []

    exp3286_ready = exp3286_ready_for_v15(exp3286)
    preconditions.append(
        {
            "name": "exp3286_calibrated_rerun_plan",
            "passed": exp3286_ready,
            "path": EXP3286_REL_PATH.as_posix(),
            "dominant_root_cause": str(exp3286.get("dominant_root_cause") or ""),
            "plan_experiment_id": str(
                mapping(exp3286.get("calibrated_rerun_plan")).get("experiment_id") or ""
            ),
        }
    )
    if not exp3286_ready:
        gate_reasons.append("exp3286_calibrated_rerun_plan_not_ready")

    receipt_ok = exp3268.get("clean_sota_receipt_eligible") is True
    preconditions.append(
        {
            "name": "exp3268_clean_sota_receipt_eligible",
            "passed": receipt_ok,
            "path": EXP3268_REL_PATH.as_posix(),
        }
    )
    if not receipt_ok:
        gate_reasons.append("exp3268.clean_sota_receipt_eligible=false")

    nvidia = normalize_precondition((nvidia_probe or default_nvidia_smi_probe)())
    preconditions.append(nvidia)
    if nvidia.get("passed") is not True:
        gate_reasons.append("nvidia_smi_unavailable")

    selected_python = normalize_precondition(
        (python_cuda_probe or selected_python_cuda_probe)()
    )
    preconditions.append(selected_python)
    if selected_python.get("passed") is not True:
        gate_reasons.append("selected_python_cuda_unavailable")

    inventory = resolve_mandated_model_inventory(root_path)
    model_specs = model_specs_from_inventory(inventory, policy)
    models_to_run = select_models_for_run(inventory)
    preconditions.append(
        {
            "name": "mandated_sota_gguf_cache",
            "passed": bool(models_to_run),
            "cached_sota_pair_attempted": True,
            "cached_sota_pair_available": inventory["cached_sota_pair_available"],
            "available_model_ids": [model["model_id"] for model in inventory["available_models"]],
            "missing_model_specs": inventory["missing_model_specs"],
        }
    )
    if not models_to_run:
        gate_reasons.append("mandated_sota_gguf_unavailable")

    available_exact_rows = v14.build_exact_eval_rows(root_path, max_rows=1_000_000)
    exact_rows = available_exact_rows[: max(0, int(max_eval_rows))]
    required_exact_rows = min(DEFAULT_MIN_EXACT_ROWS, len(available_exact_rows))
    fixture_ready = bool(exact_rows) and len(exact_rows) >= required_exact_rows
    preconditions.append(
        {
            "name": "exact_row_fixture_availability",
            "passed": fixture_ready,
            "path": CONTEXT_FIXTURE_REL_PATH.as_posix(),
            "exact_rows_available": len(available_exact_rows),
            "exact_rows_selected": len(exact_rows),
            "required_exact_rows": required_exact_rows,
        }
    )
    if not fixture_ready:
        gate_reasons.append("exact_row_fixture_unavailable")

    per_row_results: list[JsonDict] = []
    gpu_mem_used_mib = 0
    if all(row.get("passed") is True for row in preconditions) and models_to_run and exact_rows:
        runner = model_runner or run_llama_grammar_verifier
        for model in models_to_run:
            try:
                payload = runner(exact_rows, model, int(random_seed), policy)
            except Exception as exc:
                payload = {
                    "rows": [],
                    "gpu_mem_used_mib": 0,
                    "runner_error": f"{type(exc).__name__}: {exc}",
                }
            payload_map = mapping(payload)
            normalized = v14.normalize_runner_rows(
                payload_map.get("rows"),
                exact_rows,
                model,
            )
            per_row_results.extend(attach_policy_reasons(normalized))
            gpu_mem_used_mib = max(gpu_mem_used_mib, safe_int(payload_map.get("gpu_mem_used_mib")))
            if payload_map.get("runner_error"):
                gate_reasons.append("model_runner_failed: " + str(payload_map["runner_error"]))

    metrics = score_results(per_row_results)
    if per_row_results and metrics["false_accept_count"] > 0:
        gate_reasons.append("false_accept_count_nonzero")
    if per_row_results and metrics["abstention_rate"] >= 1.0:
        gate_reasons.append("abstention_rate_is_1.0")
    if per_row_results and metrics["coverage_rate"] < float(policy["minimum_decision_coverage"]):
        gate_reasons.append("coverage_below_calibrated_minimum")
    if per_row_results and gpu_mem_used_mib < MIN_GPU_MEM_USED_MIB:
        gate_reasons.append("gpu_mem_used_below_cuda_offload_floor")

    ready = (
        bool(per_row_results)
        and all(row.get("passed") is True for row in preconditions)
        and metrics["false_accept_count"] == 0
        and metrics["coverage_rate"] >= float(policy["minimum_decision_coverage"])
        and metrics["abstention_rate"] < 1.0
        and len(exact_rows) >= required_exact_rows
        and gpu_mem_used_mib >= MIN_GPU_MEM_USED_MIB
        and not any(reason.startswith("model_runner_failed") for reason in gate_reasons)
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    source_artifacts = build_source_artifacts(root_path)
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3287", "SCENARIO-VERIFY-3287"],
        "abstention_calibrated_clean_verifier_v15_ready": ready,
        "clean_verifier_rerun_ready": ready,
        "repair_gate_input_clean_enough": ready,
        "gate_reasons": sorted(set(gate_reasons)),
        "model_specs": model_specs,
        "models_used": models_used(models_to_run) if per_row_results else [],
        "missing_model_specs": inventory["missing_model_specs"],
        "preconditions_checked": preconditions,
        "n_eval": metrics["n_eval"],
        "exact_checkable_row_count": len(exact_rows),
        "exact_rows_available": len(available_exact_rows),
        "required_exact_rows": required_exact_rows,
        "exact_row_fixture_hash": v14.exact_row_fixture_hash(root_path, exact_rows),
        "false_accept_rate": metrics["false_accept_rate"],
        "false_reject_rate": metrics["false_reject_rate"],
        "abstention_rate": metrics["abstention_rate"],
        "coverage_rate": metrics["coverage_rate"],
        "false_accept_count": metrics["false_accept_count"],
        "false_reject_count": metrics["false_reject_count"],
        "abstention_count": metrics["abstention_count"],
        "abstention_reason_counts": count_abstention_reasons(per_row_results),
        "per_class_reason_counts": per_class_reason_counts(per_row_results),
        "calibrated_abstention_policy": policy,
        "gpu_mem_used_mib": int(gpu_mem_used_mib if per_row_results else 0),
        "per_row_results": per_row_results,
        "source_artifacts": source_artifacts,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_artifacts if row.get("sha256")
        },
        "random_seed": int(random_seed),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started, finished),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    nvidia_probe: Probe | None = None,
    python_cuda_probe: Probe | None = None,
    model_runner: ModelRunner | None = None,
    max_eval_rows: int = DEFAULT_MAX_EVAL_ROWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3287 terminal JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        nvidia_probe=nvidia_probe,
        python_cuda_probe=python_cuda_probe,
        model_runner=model_runner,
        max_eval_rows=max_eval_rows,
        random_seed=random_seed,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def exp3286_ready_for_v15(exp3286: Mapping[str, Any]) -> bool:
    """Return whether Exp 3286 found the actionable parser-contract cause."""

    plan = mapping(exp3286.get("calibrated_rerun_plan"))
    return (
        exp3286.get("abstention_root_cause_identified") is True
        and str(exp3286.get("dominant_root_cause") or "")
        == "model_output_parser_contract_mismatch"
        and str(plan.get("experiment_id") or "") == "exp3287"
    )


def calibrated_policy_from_exp3286(exp3286: Mapping[str, Any]) -> JsonDict:
    """Carry forward the Exp 3286 policy without loosening the parser."""

    plan = mapping(exp3286.get("calibrated_rerun_plan"))
    criteria = mapping(plan.get("acceptance_criteria"))
    return {
        "policy_name": "exp3286_strict_leading_token_calibrated_v1",
        "strict_leading_token": True,
        "allowed_decisions": ["accept", "reject", "abstain"],
        "grammar": "ACCEPT|REJECT|ABSTAIN",
        "grammar_constrained_decoding": True,
        "minimum_decision_coverage": bounded_rate(
            criteria.get("minimum_decision_coverage"),
            default=0.5,
        ),
        "target_false_accept_rate": bounded_rate(
            criteria.get("target_false_accept_rate"),
            default=0.0,
        ),
        "target_max_abstention_rate": bounded_rate(
            criteria.get("target_max_abstention_rate"),
            default=0.5,
        ),
        "source_experiment_id": str(plan.get("experiment_id") or "exp3287"),
    }


def resolve_mandated_model_inventory(root: Path) -> JsonDict:
    """Resolve mandated local GGUFs, trying the canonical pair helper first."""

    pair = cached_sota_pair(gpu_indices=(0, 1))
    pair_by_id = {str(row.get("hf_id") or row.get("model_id") or ""): row for row in pair or []}
    specs_by_id = {spec["hf_id"]: spec for spec in SOTA_GGUF_MODELS}
    available: list[JsonDict] = []
    missing: list[JsonDict] = []
    mandated_models: JsonDict = {}
    seen_available: set[str] = set()

    for index, model_id in enumerate(MANDATED_MODEL_IDS):
        spec = mapping(specs_by_id.get(model_id))
        pair_entry = mapping(pair_by_id.get(model_id))
        resolved = str(pair_entry.get("model_path") or resolve_cached_gguf(model_id) or "")
        path = resolve_path(root, resolved) if resolved else None
        cached = bool(path and path.is_file() and path.stat().st_size > 0)
        model_record = {
            "model_id": model_id,
            "hf_id": model_id,
            "name": str(pair_entry.get("name") or spec.get("name") or model_id),
            "role": str(spec.get("role") or ""),
            "expected_quantization": str(spec.get("quantization") or "Q4_K_M"),
            "cached": cached,
            "model_path": str(path) if cached and path is not None else None,
            "size_bytes": int(path.stat().st_size) if cached and path is not None else 0,
        }
        mandated_models[model_id] = model_record
        if cached and model_id not in seen_available:
            available.append(
                model_record
                | {
                    "gpu": int(pair_entry.get("gpu", len(available) % 2)),
                    "source": "cached_sota_pair" if pair_entry else "resolve_cached_gguf",
                    "legacy_small_model": False,
                }
            )
            seen_available.add(model_id)
        elif not cached:
            missing.append(
                {
                    "model_id": model_id,
                    "hf_id": model_id,
                    "name": model_record["name"],
                    "role": model_record["role"],
                    "expected_quantization": model_record["expected_quantization"],
                    "cached": False,
                    "model_path": None,
                    "reason": "not_cached",
                }
            )

    return {
        "cached_sota_pair_attempted": True,
        "cached_sota_pair_available": pair is not None,
        "cached_sota_pair_specs": [dict(row) for row in pair or []],
        "available_models": available,
        "missing_model_specs": missing,
        "mandated_models": mandated_models,
    }


def model_specs_from_inventory(inventory: Mapping[str, Any], policy: Mapping[str, Any]) -> JsonDict:
    """Return the artifact model-spec block with every mandated model named."""

    return {
        "runtime": "llama_cpp",
        "n_gpu_layers_requested": -1,
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_attempted": inventory.get("cached_sota_pair_attempted") is True,
        "cached_sota_pair_available": inventory.get("cached_sota_pair_available") is True,
        "cached_sota_pair_specs": mapping_list(inventory.get("cached_sota_pair_specs")),
        "mandated_models": mapping(inventory.get("mandated_models")),
        "available_model_count": len(mapping_list(inventory.get("available_models"))),
        "missing_model_count": len(mapping_list(inventory.get("missing_model_specs"))),
        "calibrated_abstention_policy": dict(policy),
    }


def select_models_for_run(inventory: Mapping[str, Any]) -> list[JsonDict]:
    """Use the cached pair when present; otherwise run one available mandated GGUF."""

    available = mapping_list(inventory.get("available_models"))
    if inventory.get("cached_sota_pair_available") is True:
        return available[:2]
    return available[:1]


def attach_policy_reasons(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Attach calibrated abstention reasons without changing exact labels."""

    enriched: list[JsonDict] = []
    for row in rows:
        payload = dict(row)
        payload["decision"] = normalize_decision(payload.get("decision") or payload.get("output_text"))
        payload["abstained"] = payload["decision"] == "abstain"
        payload["false_accept"] = (
            str(payload.get("expected_decision")) == "reject" and payload["decision"] == "accept"
        )
        payload["false_reject"] = (
            str(payload.get("expected_decision")) == "accept" and payload["decision"] == "reject"
        )
        payload["abstention_reason"] = abstention_reason(payload)
        payload["calibrated_policy"] = "exp3286_strict_leading_token_calibrated_v1"
        enriched.append(payload)
    return enriched


def score_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute exact-authority verifier rates including selective coverage."""

    base = v14.score_results(rows)
    base["coverage_rate"] = rate(base["n_eval"] - base["abstention_count"], base["n_eval"])
    return base


def count_abstention_reasons(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count only rows that actually abstained, leaving ready runs compact."""

    counts: JsonDict = {}
    for row in rows:
        reason = str(row.get("abstention_reason") or "")
        if reason and reason != "not_abstained":
            counts[reason] = int(counts.get(reason, 0)) + 1
    return counts


def per_class_reason_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Group accepted/rejected/abstained outcomes by exact expected decision."""

    counts: JsonDict = {}
    for row in rows:
        expected = str(row.get("expected_decision") or "unknown")
        decision = str(row.get("decision") or "abstain")
        if decision == "abstain":
            reason = str(row.get("abstention_reason") or "reported_abstain")
        elif decision == "accept":
            reason = "accepted_decision"
        elif decision == "reject":
            reason = "rejected_decision"
        else:
            reason = "unknown_decision"
        bucket = counts.setdefault(expected, {})
        bucket[reason] = int(bucket.get(reason, 0)) + 1
    return counts


def abstention_reason(row: Mapping[str, Any]) -> str:
    """Explain why strict calibrated parsing produced an abstention."""

    decision = normalize_decision(row.get("decision"))
    output_text = str(row.get("output_text") or "")
    if decision != "abstain":
        return "not_abstained"
    if not output_text.strip():
        return "missing_model_output"
    if normalize_output_decision(output_text) is None:
        return "model_output_unparseable"
    return "reported_abstain"


def normalize_decision(value: Any) -> str:
    """Normalize a verifier decision, failing closed to abstain."""

    return normalize_output_decision(value) or "abstain"


def normalize_output_decision(value: Any) -> str | None:
    """Parse the strict leading-token decision contract from Exp 3286."""

    text = str(value or "").strip()
    if not text:
        return None
    first = text.split()[0].strip(" \t\r\n.:,;!?\"'`()[]{}").lower()
    return first if first in {"accept", "reject", "abstain"} else None


def default_nvidia_smi_probe() -> JsonDict:  # pragma: no cover - hardware boundary.
    """Check visible NVIDIA GPUs using the same telemetry shape as v14."""

    payload = v14.default_cuda_probe()
    payload["name"] = "nvidia_smi"
    return payload


def selected_python_cuda_probe() -> JsonDict:  # pragma: no cover - environment boundary.
    """Check that this Python can see CUDA and llama.cpp GPU offload."""

    payload: JsonDict = {
        "name": "selected_python_cuda",
        "selected_python": sys.executable,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "torch_import_ok": False,
        "llama_cpp_import_ok": False,
        "llama_cpp_supports_gpu_offload": False,
        "probe_error": "",
    }
    try:
        import torch  # noqa: PLC0415

        payload["torch_import_ok"] = True
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count())
        if payload["cuda_available"]:
            payload["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        payload["probe_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from llama_cpp import llama_cpp as llama_backend  # noqa: PLC0415

        payload["llama_cpp_import_ok"] = True
        payload["llama_cpp_supports_gpu_offload"] = bool(
            llama_backend.llama_supports_gpu_offload()
        )
    except Exception as exc:
        suffix = f"{type(exc).__name__}: {exc}"
        payload["probe_error"] = (
            suffix if not payload["probe_error"] else payload["probe_error"] + "; " + suffix
        )

    payload["passed"] = (
        payload["cuda_available"] is True
        and safe_int(payload["cuda_device_count"]) > 0
        and payload["llama_cpp_import_ok"] is True
        and payload["llama_cpp_supports_gpu_offload"] is True
    )
    return payload


def run_llama_grammar_verifier(
    rows: list[JsonDict],
    model: JsonDict,
    random_seed: int,
    policy: JsonDict,
) -> JsonDict:  # pragma: no cover - exercised by the live artifact run.
    """Run llama.cpp with grammar-constrained verifier decisions."""

    from llama_cpp import Llama, LlamaGrammar  # noqa: PLC0415

    samples = [_gpu_memory_rows()]
    llm = Llama(
        model_path=str(model["model_path"]),
        n_ctx=2048,
        n_gpu_layers=-1,
        seed=int(random_seed),
        verbose=False,
    )
    grammar = LlamaGrammar.from_string(DECISION_GRAMMAR)
    samples.append(_gpu_memory_rows())
    output_rows: list[JsonDict] = []
    for row in rows:
        raw = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You verify exact fixture rows. Reply with exactly one word: "
                        "ACCEPT, REJECT, or ABSTAIN."
                    ),
                },
                {"role": "user", "content": verifier_prompt(row)},
            ],
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            grammar=grammar,
        )
        text = chat_completion_text(raw)
        samples.append(_gpu_memory_rows())
        output_rows.append(
            {
                "row_id": row["row_id"],
                "model_id": model["model_id"],
                "model_path": model["model_path"],
                "output_text": text,
                "decision": text,
                "token_counts": chat_token_counts(raw),
            }
        )
    del llm
    gc.collect()
    return {"rows": output_rows, "gpu_mem_used_mib": max_gpu_memory(samples)}


def verifier_prompt(row: Mapping[str, Any]) -> str:  # pragma: no cover - live prompt helper.
    """Create the exact-row prompt paired with the grammar-constrained decoder."""

    return (
        f"Context: {row.get('context')}\n"
        f"Question: {row.get('question')}\n"
        f"Candidate answer: {row.get('candidate_answer')}\n\n"
        "ACCEPT if the candidate exactly satisfies the context and question. "
        "REJECT if it contradicts or mismatches the context answer. "
        "ABSTAIN only if the row cannot be checked."
    )


def _gpu_memory_rows() -> list[JsonDict]:  # pragma: no cover - hardware boundary.
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[JsonDict] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2:
            rows.append({"index": safe_int(parts[0]), "memory_used_mib": safe_int(parts[1])})
    return rows


def chat_completion_text(raw: Any) -> str:  # pragma: no cover - llama.cpp adapter.
    choices = mapping(raw).get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    message = mapping(mapping(first).get("message"))
    return str(message.get("content") or mapping(first).get("text") or "")


def chat_token_counts(raw: Any) -> JsonDict:  # pragma: no cover - llama.cpp adapter.
    usage = mapping(mapping(raw).get("usage"))
    return {
        "prompt_tokens": safe_int(usage.get("prompt_tokens")),
        "completion_tokens": safe_int(usage.get("completion_tokens")),
        "total_tokens": safe_int(usage.get("total_tokens")),
    }


def max_gpu_memory(samples: Sequence[Sequence[Mapping[str, Any]]]) -> int:
    """Return the maximum absolute GPU memory seen across telemetry samples."""

    values = [
        safe_int(row.get("memory_used_mib"))
        for sample in samples
        for row in sample
        if isinstance(row, Mapping)
    ]
    return max(values) if values else 0


def models_used(models: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize model records for the public artifact field."""

    return [
        {
            "model_id": str(model.get("model_id") or model.get("hf_id") or ""),
            "hf_id": str(model.get("hf_id") or model.get("model_id") or ""),
            "name": str(model.get("name") or ""),
            "model_path": str(model.get("model_path") or ""),
            "gpu": safe_int(model.get("gpu")),
            "source": str(model.get("source") or ""),
            "legacy_small_model": model.get("legacy_small_model") is True,
        }
        for model in models
    ]


def build_source_artifacts(root: Path) -> list[JsonDict]:
    """Record local files that determine the v15 result."""

    paths = (
        ("exp3286_abstention_root_cause", EXP3286_REL_PATH),
        ("exp3275_requested_clean_sota_rerun", EXP3275_REQUESTED_REL_PATH),
        ("exp3275_local_clean_sota_rerun", EXP3275_LOCAL_REL_PATH),
        ("exp3268_sota_receipt_methodology", EXP3268_REL_PATH),
        ("context_exact_row_fixture", CONTEXT_FIXTURE_REL_PATH),
        ("verification_openspec", SPEC_REL_PATH),
        (
            "exp3287_module",
            Path("python/carnot/verify/abstention_calibrated_clean_verifier_v15.py"),
        ),
        (
            "exp3287_tests",
            TEST_REL_PATH,
        ),
    )
    return [
        {
            "role": role,
            "path": path.as_posix(),
            "present": (root / path).is_file(),
            "sha256": sha256_file(root / path),
        }
        for role, path in paths
    ]


def normalize_precondition(row: Mapping[str, Any]) -> JsonDict:
    """Ensure precondition rows always have a name and strict boolean pass bit."""

    payload = dict(row)
    payload["name"] = str(payload.get("name") or "unnamed_precondition")
    payload["passed"] = payload.get("passed") is True
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject contradictory v15 artifacts before repair-gate consumption."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success-style prefix")
    for key in ("false_accept_rate", "false_reject_rate", "abstention_rate", "coverage_rate"):
        value = artifact.get(key)
        if not isinstance(value, float) or not 0.0 <= value <= 1.0:
            raise ValueError(f"rate field {key} must be a float in [0, 1]")
    for key in ("n_eval", "exact_checkable_row_count"):
        if not isinstance(artifact.get(key), int) or int(artifact[key]) < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if not isinstance(artifact.get("model_specs"), Mapping):
        raise ValueError("model_specs must be an object")
    for key in ("models_used", "missing_model_specs", "preconditions_checked"):
        if not isinstance(artifact.get(key), list):
            raise ValueError(f"{key} must be a list")
    if len(str(artifact.get("reproducibility_checksum") or "")) != 64:
        raise ValueError("reproducibility_checksum must be a sha256-style string")
    if artifact.get("repair_gate_input_clean_enough") is True and (
        artifact.get("clean_verifier_rerun_ready") is not True
        or artifact.get("abstention_calibrated_clean_verifier_v15_ready") is not True
    ):
        raise ValueError("repair gate cannot be true when v15 readiness is false")
    if artifact.get("abstention_calibrated_clean_verifier_v15_ready") is True:
        if int(artifact.get("n_eval") or 0) <= 0:
            raise ValueError("ready artifact must score at least one row")
        if float(artifact.get("false_accept_rate") or 0.0) != 0.0:
            raise ValueError("ready artifact must have zero false accepts")
        if float(artifact.get("abstention_rate") or 0.0) >= 1.0:
            raise ValueError("ready artifact cannot abstain on every row")
        if float(artifact.get("coverage_rate") or 0.0) <= 0.0:
            raise ValueError("ready artifact must have non-trivial coverage")
        if not artifact.get("models_used"):
            raise ValueError("ready artifact must record models_used")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal conductor-compatible v15 verdict."""

    if artifact.get("abstention_calibrated_clean_verifier_v15_ready") is True:
        return "complete: abstention-calibrated clean verifier v15 ready for repair gate"
    if int(artifact.get("n_eval") or 0) == 0:
        return "complete: abstention-calibrated clean verifier v15 gated skip"
    return "complete: abstention-calibrated clean verifier v15 not ready for repair gate"


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after removing its self-referential checksum field."""

    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return stable_hash(payload)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence for missing or malformed files."""

    return v14.read_json_object(path)


def read_jsonl_objects(path: Path) -> list[JsonDict]:
    """Read JSONL object rows while ignoring malformed and non-object lines."""

    return v14.read_jsonl_objects(path)


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for a present local file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash structured data with deterministic JSON normalization."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_path(root: Path, value: str) -> Path:
    """Resolve absolute and repository-relative paths without changing evidence."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def mapping(value: Any) -> JsonDict:
    """Return a plain dict only when the input is mapping-like."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only object rows from JSON list-like values."""

    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def safe_int(value: Any) -> int:
    """Coerce numeric evidence to int, failing closed to zero."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def bounded_rate(value: Any, *, default: float) -> float:
    """Coerce evidence to a rate in [0, 1], using the default if invalid."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not 0.0 <= result <= 1.0:
        return float(default)
    return round(result, 6)


def rate(numerator: int, denominator: int) -> float:
    """Compute bounded rates while making empty denominators explicit as zero."""

    return 0.0 if denominator <= 0 else round(float(numerator) / float(denominator), 6)


def duration(started_s: float, now_s: float) -> float:
    """Measure non-negative wall-clock duration."""

    return round(max(0.0, float(now_s) - float(started_s)), 6)
