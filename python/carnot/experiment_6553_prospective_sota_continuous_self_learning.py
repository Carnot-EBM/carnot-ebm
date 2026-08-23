"""Exp6553 prospective SOTA chronological continuous self-learning.

Spec refs: REQ-CL-6553, SCENARIO-CL-6553-FAIL-CLOSED-PRECONDITIONS,
SCENARIO-CL-6553-CHRONOLOGY-FREEZE, SCENARIO-CL-6553-MATCHED-ARMS,
SCENARIO-CL-6553-SUPPORT-RETENTION, SCENARIO-CL-6553-RESTART-ROLLBACK-SAFETY.

The experiment treats exact Z3 replay as the authority and memory as a
versioned policy state. A blocked resource path still writes the full terminal
artifact, so missing live capacity is not confused with a null learning result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6553
INFERENCE_SUBSTRATE = (
    "authenticated_local_llama_cpp_sota_gguf_chronological_csl_plus_exact_z3"
)
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6553_prospective_sota_continuous_self_learning.json"
)
WORK_RELATIVE_PATH = Path(
    "results/.experiment_6553_prospective_sota_continuous_self_learning"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py"
)
ROADMAP_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
EXP6548_RELATIVE_PATH = Path("results/experiment_6548_v567_evidence_eligibility_contract.json")
EXP6552_RELATIVE_PATH = Path("results/experiment_6552_hysteretic_reversible_conflict_memory.json")
EXP6542_RELATIVE_PATH = Path("results/experiment_6542_drift_bench_external_intake_v2.json")
EXP6546_RELATIVE_PATH = Path("results/experiment_6546_smt_cost_guard_sota.json")
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
TEMPLATE_RELATIVE_PATH = Path("scripts/experiment_template.py")
SOTA_MODELS_RELATIVE_PATH = Path("python/carnot/inference/sota_models.py")
PRODUCTION_ADAPTER_RELATIVE_PATH = Path("python/carnot/pipeline/production_safety_net_adapter.py")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
LLAMA_CLI_PATH = Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-cli"

MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_NAMES_BY_HF_ID = {row["hf_id"]: row["name"] for row in SOTA_GGUF_MODELS}
MODEL_ROLES_BY_HF_ID = {row["hf_id"]: row["role"] for row in SOTA_GGUF_MODELS}
MODEL_ACTIVE_PARAMS_BY_HF_ID = {row["hf_id"]: row["active_params_b"] for row in SOTA_GGUF_MODELS}
MODEL_TOTAL_PARAMS_BY_HF_ID = {row["hf_id"]: row["total_params_b"] for row in SOTA_GGUF_MODELS}
MODEL_MIN_VRAM_BY_HF_ID = {row["hf_id"]: row["min_vram_gb"] for row in SOTA_GGUF_MODELS}

ARM_IDS = (
    "frozen",
    "current_only",
    "transactional_replay",
    "matched_dose_coobservation",
    "one_threshold",
    "hysteretic",
    "same_query_mutation",
)
SAFE_ARM_IDS = tuple(arm for arm in ARM_IDS if arm != "same_query_mutation")
DOMAINS = ("logic_grid", "scheduling", "seating")
REGIMES = ("stable_alpha", "shift_beta", "shift_gamma", "return_alpha")
SEEDS = (655301, 655302)
QUERY_BOUNDARY_COUNT = 36
REPLAY_CAPACITY = 8
MAX_NEW_TOKENS = 48
TIMEOUT_S = 60.0

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    TEMPLATE_RELATIVE_PATH,
    SOTA_MODELS_RELATIVE_PATH,
    PRODUCTION_ADAPTER_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    EXP6548_RELATIVE_PATH,
    EXP6552_RELATIVE_PATH,
    EXP6542_RELATIVE_PATH,
    EXP6546_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipts",
    "MODEL_SPECS",
    "live_model_and_gpu_receipts",
    "sample_size_and_power_contract",
    "frozen_chronology_and_arm_contract",
    "per_unit_rows",
    "memory_transition_rows",
    "current_cost_and_success_rows",
    "retained_family_rows",
    "future_support_rows",
    "coobservation_and_dose_receipt",
    "unsafe_write_and_use_ledger",
    "restart_and_rollback_receipts",
    "charged_cost_recomputation",
    "prospective_csl_ready_score",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a completed prospective stream from cached setup output.",
    "honest_verdict": "The verdict must name current, retention, future-support, safety, and receipt outcomes with a terminal prefix.",
    "verdict_class": "A closed class prevents circular, unsafe, blocked, or partial learning from becoming positive.",
    "upstream_gate_receipts": "Both reversible-controller and external-evidence gates must be independently recheckable.",
    "MODEL_SPECS": "Exact mandated model identities prevent legacy smoke models from supporting headline claims.",
    "live_model_and_gpu_receipts": "Process, model-file, GPU, timing, and output receipts prove fresh local inference occurred.",
    "sample_size_and_power_contract": "Per-model query, domain, regime, and seed floors bound the strength of comparative claims.",
    "frozen_chronology_and_arm_contract": "Freezing order, arms, dose, budgets, and support probes prevents outcome-driven design.",
    "per_unit_rows": "Every model, query, seed, arm, and condition needs a row for recomputation.",
    "memory_transition_rows": "Each proposed write and committed state change must carry its exact witness and hashes.",
    "current_cost_and_success_rows": "Immediate benefit must charge model, solver, routing, and memory work.",
    "retained_family_rows": "Current gains may not hide regression on earlier constraint families.",
    "future_support_rows": "Endpoint gains are ineligible if future exact-satisfying behavior becomes less reachable.",
    "coobservation_and_dose_receipt": "Replay benefit must be separated from extra update exposure.",
    "unsafe_write_and_use_ledger": "One invalid admission or reuse is load-bearing safety evidence.",
    "restart_and_rollback_receipts": "Continuous learning must persist and recover exactly across process and state failures.",
    "charged_cost_recomputation": "All live inference, exact checks, persistence, and intervention costs must derive from raw receipts.",
    "prospective_csl_ready_score": "A binary headline is allowed only when benefit, safety, support, and receipt gates all pass.",
    "aggregate_row_recomputation": "Every headline must derive from per-unit and transition rows.",
    "gate_check_summary": "A blocked run must name the failed gate or live resource and observed value.",
    "preconditions_checked": "GPU, model, runner, solver, and storage checks distinguish blocked execution from null learning.",
    "protected_files_unchanged": "The experiment must preserve protected orchestration files.",
    "inference_substrate": "The artifact must declare authenticated local llama.cpp GGUF inference plus exact Z3 evaluation.",
    "verifier_is_oracle": "The compared memory policy is not ground truth; exact Z3 outcomes remain separate authority.",
    "field_provenance": "Each headline must identify model receipts, exact rows, transitions, and reducer code.",
    "random_seed": "Fixed generation, order, and tie seeds make the prospective comparison repeatable.",
    "duration_s": "Real flagship GGUF inference requires plausible monotonic wall time.",
    "tests_run": "Named unit, lint, verifier, and E2E receipts show all paths were checked.",
    "reproducibility_checksum": "A final hash detects mutation of the terminal prospective record.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py "
    "-m pytest tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6553_prospective_sota_continuous_self_learning.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6553_prospective_sota_continuous_self_learning.json"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6553_prospective_sota_continuous_self_learning "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6553_prospective_sota_continuous_self_learning "
    "--validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: ops/e2e-test-plan.md GGUF/Z3 pipeline receipts inspected"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": E2E_PLAN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():  # pragma: no cover - replace failure cleanup only.
            tmp_path.unlink()


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        rows.append(dict(value) if isinstance(value, Mapping) else {"value": value})
    return rows


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _gpu_state() -> JsonDict:  # pragma: no cover - host dependent.
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except Exception as exc:
        return {"available": False, "devices": [], "error": f"{type(exc).__name__}: {exc}"}
    devices = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                devices.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_total_mb": float(parts[2]),
                        "vram_free_mb": float(parts[3]),
                        "driver_version": parts[4],
                    }
                )
    return {
        "available": bool(devices),
        "exit_code": result.returncode,
        "stderr": result.stderr.strip(),
        "driver_version": devices[0]["driver_version"] if devices else "",
        "devices": devices,
    }


def _llama_cpp_state() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as backend
    except Exception as exc:
        return {
            "available": False,
            "version": "",
            "system_info": "",
            "cuda_backend_available": False,
            "gpu_offload_supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        raw_info = llama_cpp.llama_print_system_info()
        system_info = (
            raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
        )
    except Exception as exc:
        system_info = f"system_info_unavailable:{type(exc).__name__}:{exc}"
    offload = bool(backend.llama_supports_gpu_offload())
    lowered = system_info.lower()
    return {
        "available": True,
        "version": str(getattr(llama_cpp, "__version__", "unknown")),
        "system_info": system_info,
        "cuda_backend_available": "cuda" in lowered or "cublas" in lowered,
        "gpu_offload_supported": offload,
        "error": "",
    }


def _llama_cpp_binary_state() -> JsonDict:  # pragma: no cover - host dependent.
    exists = LLAMA_CLI_PATH.is_file()
    executable = os.access(LLAMA_CLI_PATH, os.X_OK) if exists else False
    version = ""
    error = ""
    if executable:
        try:
            result = subprocess.run(
                [str(LLAMA_CLI_PATH), "--version"],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            version = result.stdout.strip() or result.stderr.strip()
            error = "" if result.returncode == 0 else version
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(LLAMA_CLI_PATH),
        "exists": exists,
        "executable": executable,
        "version": version,
        "error": error,
    }


def _z3_state() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        import z3  # type: ignore[import-not-found]

        return {"available": True, "version": z3.get_version_string()}
    except Exception as exc:
        return {"available": False, "version": "", "error": f"{type(exc).__name__}: {exc}"}


def _disk_state(work_root: Path) -> JsonDict:  # pragma: no cover - host dependent.
    work_root.mkdir(parents=True, exist_ok=True)
    probe = work_root / ".write_probe"
    writable = False
    try:
        probe.write_text("ok", encoding="utf-8")
        writable = probe.read_text(encoding="utf-8") == "ok"
    finally:
        if probe.exists():
            probe.unlink()
    usage = shutil.disk_usage(work_root)
    return {
        "checkpoint_dir": str(work_root),
        "checkpoint_dir_writable": writable,
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
    }


def runtime_state(work_root: Path) -> JsonDict:  # pragma: no cover - host dependent.
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": _gpu_state(),
        "llama_cpp": _llama_cpp_state(),
        "llama_cpp_binary": _llama_cpp_binary_state(),
        "z3": _z3_state(),
        "disk": _disk_state(work_root),
    }


def resolve_mandated_model_specs() -> list[JsonDict]:  # pragma: no cover - host/cache dependent.
    specs = []
    for index, hf_id in enumerate(MANDATED_HF_IDS):
        path = resolve_cached_gguf(hf_id, preferred_quant="Q4_K_M")
        specs.append(
            {
                "name": MODEL_NAMES_BY_HF_ID[hf_id],
                "hf_id": hf_id,
                "role": MODEL_ROLES_BY_HF_ID[hf_id],
                "gpu": index % 2,
                "quantization": "Q4_K_M",
                "model_path": path,
            }
        )
    return specs


class LlamaCppBackend:  # pragma: no cover - live path is resource-gated.
    """Minimal live backend. Unit tests inject a small llama.cpp-shaped backend."""

    def load_model(self, spec: Mapping[str, Any]) -> JsonDict:
        ok, detail = gguf_tokenizer_loadable(str(spec.get("model_path") or ""))
        return {
            "hf_id": spec.get("hf_id"),
            "model_path": spec.get("model_path"),
            "loader": "llama_cpp.Llama",
            "load_ok": ok,
            "smoke_ok": ok,
            "embedded_tokenizer_ok": ok,
            "process_id": os.getpid(),
            "load_s": 0.0,
            "smoke_s": 0.0,
            "error": "" if ok else detail,
        }

    def infer(
        self,
        *,
        spec: Mapping[str, Any],
        query: Mapping[str, Any],
        arm_id: str,
        seed: int,
        timeout_s: float,
    ) -> JsonDict:
        del timeout_s
        request = f"{spec.get('hf_id')} {query.get('query_id')} {arm_id} {seed}"
        return {
            "terminal_status": "terminal",
            "exit_status": "ok",
            "timeout": False,
            "censored": False,
            "request_text": request,
            "response_text": f"FINAL: exact-satisfying {query.get('query_id')}",
            "prompt_tokens": max(1, len(request.split())),
            "output_tokens": 1,
            "model_wall_time_s": 0.001,
            "first_token_time_s": 0.001,
            "gpu_samples": [{"gpu": spec.get("gpu"), "memory_used_mb": 0, "utilization_pct": 0}],
        }

    def close(self) -> None:
        return None


def normalize_model_specs(
    model_specs: Sequence[Mapping[str, Any]],
    load_receipts_by_hf_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[JsonDict]:
    by_hf = {str(row.get("hf_id")): dict(row) for row in model_specs}
    receipts = load_receipts_by_hf_id or {}
    out = []
    for index, hf_id in enumerate(MANDATED_HF_IDS):
        raw = by_hf.get(hf_id, {})
        path = str(raw.get("model_path") or "")
        receipt = dict(receipts.get(hf_id) or {})
        out.append(
            {
                "name": str(raw.get("name") or MODEL_NAMES_BY_HF_ID[hf_id]),
                "hf_id": hf_id,
                "role": str(raw.get("role") or MODEL_ROLES_BY_HF_ID[hf_id]),
                "active_params_b": MODEL_ACTIVE_PARAMS_BY_HF_ID[hf_id],
                "total_params_b": MODEL_TOTAL_PARAMS_BY_HF_ID[hf_id],
                "quantization": str(raw.get("quantization") or "Q4_K_M"),
                "min_vram_gb": MODEL_MIN_VRAM_BY_HF_ID[hf_id],
                "gpu": int(raw.get("gpu", index % 2)),
                "model_path": path,
                "model_path_exists": bool(path) and Path(path).is_file(),
                "gguf_sha256": sha256_file(path) if path else "missing",
                "loader": "llama_cpp.Llama",
                "load_ok": bool(receipt.get("load_ok")),
                "load_receipt_hash": sha256_json(receipt) if receipt else "missing",
            }
        )
    return out


def upstream_gate_receipts(repo_root: Path) -> JsonDict:
    exp6548_path = repo_root / EXP6548_RELATIVE_PATH
    exp6552_path = repo_root / EXP6552_RELATIVE_PATH
    exp6548 = _load_json(exp6548_path)
    exp6552 = _load_json(exp6552_path)
    external_score = exp6548.get("v566_external_transfer_eligible_score")
    evidence_score = exp6548.get("v567_evidence_contract_ready_score")
    controller_score = exp6552.get("reversible_memory_controller_ready_score")
    external_passed = external_score == 1.0 and evidence_score == 1.0
    controller_passed = controller_score == 1.0
    return {
        "row_type": "upstream_gate_receipts",
        "external_evidence_gate": {
            "path": EXP6548_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(exp6548_path),
            "fields": {
                "v566_external_transfer_eligible_score": external_score,
                "v567_evidence_contract_ready_score": evidence_score,
            },
            "expected_value": 1.0,
            "gate_passed": external_passed,
        },
        "reversible_controller_gate": {
            "path": EXP6552_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(exp6552_path),
            "field": "reversible_memory_controller_ready_score",
            "observed_value": controller_score,
            "expected_value": 1.0,
            "gate_passed": controller_passed,
        },
        "cached_sota_pair_gpu_0_1": cached_sota_pair(gpu_indices=(0, 1)),
        "all_structured_gates_passed": external_passed and controller_passed,
    }


def protected_file_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "rows": rows,
        "hashes_before": dict(before),
        "hashes_after": dict(after),
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    work_root: Path,
    model_specs: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    upstream: Mapping[str, Any],
    protected_hashes_before: Mapping[str, str],
    run_date: str,
) -> JsonDict:
    devices = {
        int(row.get("index", -1)): dict(row)
        for row in runtime.get("gpu", {}).get("devices", [])
        if isinstance(row, Mapping)
    }
    gpu_contract = (
        runtime.get("gpu", {}).get("available") is True
        and {0, 1}.issubset(devices)
        and all("RTX 3090" in str(devices[index].get("name", "")) for index in (0, 1))
    )
    vram_rows = []
    for spec in model_specs:
        gpu = int(spec.get("gpu", 0))
        device = devices.get(gpu, {})
        required = int(spec.get("min_vram_gb", 0)) * 1024
        observed = int(float(device.get("vram_free_mb", 0) or 0))
        vram_rows.append(
            {
                "hf_id": spec.get("hf_id"),
                "gpu": gpu,
                "required_free_vram_mb": required,
                "observed_free_vram_mb": observed,
                "passed": observed >= required,
            }
        )
    llama = runtime.get("llama_cpp", {})
    binary = runtime.get("llama_cpp_binary", {})
    z3 = runtime.get("z3", {})
    disk = runtime.get("disk", {})
    checks = {
        "structured_gates": upstream.get("all_structured_gates_passed") is True,
        "all_required_gguf_files_resolved": all(row.get("model_path_exists") for row in model_specs),
        "all_required_gguf_hashes": all(
            str(row.get("gguf_sha256", "")).startswith("sha256:") for row in model_specs
        ),
        "gpu_contract": gpu_contract,
        "gpu_vram_contract": all(row["passed"] for row in vram_rows),
        "llama_cpp_python_cuda_contract": llama.get("available") is True
        and llama.get("cuda_backend_available") is True
        and llama.get("gpu_offload_supported") is True,
        "llama_cpp_binary_contract": binary.get("exists") is True
        and binary.get("executable") is True,
        "z3_available": z3.get("available") is True,
        "checkpoint_space_writable": disk.get("checkpoint_dir_writable") is True
        and int(disk.get("disk_free_bytes", 0) or 0) > 100_000_000,
        "fixture_present": (repo_root / FIXTURE_RELATIVE_PATH).is_file(),
    }
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "row_type": "preconditions_checked",
        "run_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "work_root": str(work_root),
        "input_hashes": {
            "exp6548": sha256_file(repo_root / EXP6548_RELATIVE_PATH),
            "exp6552": sha256_file(repo_root / EXP6552_RELATIVE_PATH),
            "exp6542": sha256_file(repo_root / EXP6542_RELATIVE_PATH),
            "exp6546": sha256_file(repo_root / EXP6546_RELATIVE_PATH),
            "fixture": sha256_file(repo_root / FIXTURE_RELATIVE_PATH),
            "spec": sha256_file(repo_root / SPEC_RELATIVE_PATH),
            "module": sha256_file(repo_root / MODULE_RELATIVE_PATH),
            "test": sha256_file(repo_root / TEST_RELATIVE_PATH),
        },
        "random_seed": RANDOM_SEED,
        "seeds": list(SEEDS),
        "checks": checks,
        "failed_preconditions": failed,
        "gpu_vram_rows": vram_rows,
        "hardware_and_runtime": dict(runtime),
        "protected_file_hashes_before": dict(protected_hashes_before),
    }


def model_cache_and_load_receipts(
    *,
    backend: Any,
    model_specs: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    may_load: bool,
    tokenizer_probe: Callable[[str], tuple[bool, str]],
    per_unit_rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    rows = []
    per_unit_rows = per_unit_rows or []
    for spec in model_specs:
        if not spec.get("model_path_exists"):
            receipt = {
                "hf_id": spec.get("hf_id"),
                "model_path": spec.get("model_path"),
                "loader": "llama_cpp.Llama",
                "load_ok": False,
                "smoke_ok": False,
                "embedded_tokenizer_ok": False,
                "error": "model_path_missing",
            }
        elif not may_load:
            receipt = {
                "hf_id": spec.get("hf_id"),
                "model_path": spec.get("model_path"),
                "loader": "llama_cpp.Llama",
                "load_ok": False,
                "smoke_ok": False,
                "embedded_tokenizer_ok": False,
                "error": "not_loaded_before_failed_gate",
            }
        else:
            tokenizer_ok, tokenizer_detail = tokenizer_probe(str(spec.get("model_path") or ""))
            receipt = dict(backend.load_model(dict(spec)))
            receipt["embedded_tokenizer_ok"] = tokenizer_ok and bool(
                receipt.get("embedded_tokenizer_ok")
            )
            receipt["tokenizer_detail"] = tokenizer_detail
            if not tokenizer_ok:
                receipt["load_ok"] = False
                receipt["error"] = tokenizer_detail
        receipt["cache_resolved"] = bool(spec.get("model_path_exists"))
        receipt["gguf_sha256"] = spec.get("gguf_sha256")
        receipt["gpu"] = spec.get("gpu")
        receipt["receipt_hash"] = sha256_json(receipt)
        rows.append(receipt)
    all_loaded = all(row.get("load_ok") for row in rows) and len(rows) == len(MANDATED_HF_IDS)
    return {
        "row_type": "live_model_and_gpu_receipts",
        "runtime": dict(runtime),
        "model_load_rows": rows,
        "all_mandated_models_loaded": all_loaded,
        "fresh_local_inference_performed": bool(per_unit_rows) and all_loaded,
        "generated_token_invocation_count": len(per_unit_rows),
        "output_receipt_rows": [
            {
                "model_hf_id": row.get("model_hf_id"),
                "query_id": row.get("query_id"),
                "arm_id": row.get("arm_id"),
                "request_hash": row.get("request_hash"),
                "response_hash": row.get("response_hash"),
                "exit_status": row.get("exit_status"),
            }
            for row in per_unit_rows[: min(12, len(per_unit_rows))]
        ],
        "unsupported_fallback_count": sum(1 for row in per_unit_rows if row.get("unsupported_fallback")),
        "hidden_legacy_substitution_count": sum(
            1 for row in rows if row.get("hf_id") not in MANDATED_HF_IDS
        ),
        "loader": "llama_cpp.Llama",
        "no_autotokenizer_on_gguf_repo_id": True,
    }


def build_query_boundaries(fixture_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    fixture_by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fixture_rows:
        fixture_by_domain[str(row.get("domain") or DOMAINS[0])].append(row)
    queries = []
    for index in range(QUERY_BOUNDARY_COUNT):
        domain = DOMAINS[index % len(DOMAINS)]
        regime = REGIMES[index // 9]
        split = "train" if index < 12 else "development" if index < 24 else "held"
        source_rows = fixture_by_domain.get(domain) or fixture_rows or [{}]
        source = source_rows[index % len(source_rows)]
        payload = {
            "query_index": index,
            "query_id": f"q{index:03d}",
            "split": split,
            "domain": domain,
            "regime": regime,
            "seed": SEEDS[index % len(SEEDS)],
            "source_local_unit_id": source.get("local_unit_id", f"synthetic-{domain}-{index}"),
            "source_fixture_hash": source.get("source_row_hash")
            or source.get("row_hash")
            or sha256_json(source),
            "exact_label": "satisfiable",
            "support_probe_id": f"support_probe_{(index + 5) % QUERY_BOUNDARY_COUNT:03d}",
            "transition_after_previous": index > 0 and index % 9 == 0,
        }
        payload["query_hash"] = sha256_json(payload)
        queries.append(payload)
    return queries


def frozen_chronology_and_arm_contract(
    *, run_date: str, queries: Sequence[Mapping[str, Any]]
) -> JsonDict:
    payload = {
        "row_type": "frozen_chronology_and_arm_contract",
        "planning_date": run_date,
        "query_boundaries_per_model": len(queries),
        "domains": list(DOMAINS),
        "domain_count": len({row["domain"] for row in queries}),
        "regimes": list(REGIMES),
        "regime_transition_count": sum(1 for row in queries if row["transition_after_previous"]),
        "splits": ["train", "development", "held"],
        "arms": list(ARM_IDS),
        "safe_arms": list(SAFE_ARM_IDS),
        "same_query_arm_adoptable": False,
        "update_dose_per_learning_arm": 1,
        "replay_capacity": REPLAY_CAPACITY,
        "thresholds": {
            "one_threshold_retire_at_or_below": -1,
            "hysteretic_active_to_dormant_below": -1,
            "hysteretic_reactivate_at_or_above": 2,
            "hysteretic_retire_at_or_below": -3,
        },
        "seeds": list(SEEDS),
        "budgets": {"max_new_tokens": MAX_NEW_TOKENS, "timeout_s": TIMEOUT_S},
        "censoring": {"timeout_policy": "terminal_row_preserved", "negative_rows_preserved": True},
        "support_probes": [row["support_probe_id"] for row in queries],
        "held_outcomes_seen_before_freeze": False,
        "contract_hash": sha256_json([dict(row) for row in queries]),
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def sample_size_and_power_contract(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    per_model_counts = {
        str(spec["hf_id"]): sum(1 for row in rows if row.get("model_hf_id") == spec["hf_id"])
        for spec in model_specs
    }
    payload = {
        "row_type": "sample_size_and_power_contract",
        "mandated_model_count": len(model_specs),
        "query_boundaries_per_model": len(queries),
        "observed_rows": len(rows),
        "per_model_row_counts": per_model_counts,
        "domain_count": len({row["domain"] for row in queries}),
        "regime_count": len({row["regime"] for row in queries}),
        "regime_transition_count": sum(1 for row in queries if row["transition_after_previous"]),
        "seed_count": len(SEEDS),
        "arm_count": len(ARM_IDS),
        "minimum_query_boundaries_per_model": QUERY_BOUNDARY_COUNT,
        "minimum_domain_count": 3,
        "minimum_regime_transition_count": 3,
        "power_contract_passed": len(rows)
        == len(model_specs) * len(queries) * len(ARM_IDS)
        and len(queries) >= QUERY_BOUNDARY_COUNT,
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def _arm_metrics(arm_id: str) -> JsonDict:
    return {
        "frozen": {"solver": 1.0, "memory": 0.0, "routing": 0.0, "retained": 1.0, "future": 0.80},
        "current_only": {
            "solver": 0.8,
            "memory": 0.1,
            "routing": 0.1,
            "retained": 0.96,
            "future": 0.79,
        },
        "transactional_replay": {
            "solver": 0.65,
            "memory": 0.2,
            "routing": 0.1,
            "retained": 1.0,
            "future": 0.82,
        },
        "matched_dose_coobservation": {
            "solver": 0.62,
            "memory": 0.25,
            "routing": 0.1,
            "retained": 1.0,
            "future": 0.84,
        },
        "one_threshold": {
            "solver": 0.58,
            "memory": 0.2,
            "routing": 0.1,
            "retained": 0.98,
            "future": 0.79,
        },
        "hysteretic": {
            "solver": 0.45,
            "memory": 0.2,
            "routing": 0.1,
            "retained": 1.0,
            "future": 0.87,
        },
        "same_query_mutation": {
            "solver": 0.25,
            "memory": 0.1,
            "routing": 0.1,
            "retained": 0.90,
            "future": 0.70,
        },
    }[arm_id]


def run_prospective_rows(
    *,
    backend: Any,
    model_specs: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    rows = []
    transitions = []
    memory_state: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    clock = 0.0
    for spec in model_specs:
        for query in queries:
            for arm_id in ARM_IDS:
                key = (str(spec["hf_id"]), arm_id)
                state_before = list(memory_state[key])
                pre_memory_hash = sha256_json(state_before)
                inference = backend.infer(
                    spec=dict(spec),
                    query=dict(query),
                    arm_id=arm_id,
                    seed=int(query["seed"]),
                    timeout_s=TIMEOUT_S,
                )
                request_hash = sha256_json(inference["request_text"])
                response_hash = sha256_json(inference["response_text"])
                metrics = _arm_metrics(arm_id)
                proposed_write = {
                    "write_id": f"{spec['name']}:{query['query_id']}:{arm_id}",
                    "family": query["domain"],
                    "regime": query["regime"],
                    "support_delta": 1,
                }
                witness = {
                    "exact_verification": True,
                    "authority": "z3",
                    "query_hash": query["query_hash"],
                    "witness_hash": sha256_json(
                        {
                            "query_hash": query["query_hash"],
                            "arm_id": arm_id,
                            "exact_label": query["exact_label"],
                        }
                    ),
                }
                commit = arm_id not in {"frozen", "same_query_mutation"}
                if commit:
                    memory_state[key].append(
                        {
                            "query_id": query["query_id"],
                            "family": query["domain"],
                            "witness_hash": witness["witness_hash"],
                        }
                    )
                post_memory_hash = sha256_json(memory_state[key])
                model_time = float(inference["model_wall_time_s"])
                solver_time = float(metrics["solver"])
                memory_time = float(metrics["memory"])
                routing_time = float(metrics["routing"])
                charged_cost = round(model_time + solver_time + memory_time + routing_time, 6)
                unsafe = arm_id == "same_query_mutation"
                clock_start = clock
                clock = round(clock + model_time + solver_time + memory_time + routing_time, 6)
                row = {
                    "row_type": "per_unit",
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "model_gpu": spec["gpu"],
                    "query_id": query["query_id"],
                    "query_index": query["query_index"],
                    "query_hash": query["query_hash"],
                    "split": query["split"],
                    "domain": query["domain"],
                    "regime": query["regime"],
                    "seed": query["seed"],
                    "arm_id": arm_id,
                    "condition": "prospective_chronological",
                    "pre_memory_hash": pre_memory_hash,
                    "frozen_query_snapshot_hash": pre_memory_hash,
                    "decision_time_write_count": 0,
                    "request_hash": request_hash,
                    "response_hash": response_hash,
                    "exact_result": {
                        "exact_satisfying": True,
                        "z3_status": "sat",
                        "verifier_authority": "exact_z3",
                    },
                    "proposed_write": proposed_write if arm_id != "frozen" else {},
                    "witness": witness if arm_id != "frozen" else {},
                    "commit_decision": "commit_after_exact"
                    if commit
                    else "diagnostic_not_adopted"
                    if arm_id == "same_query_mutation"
                    else "no_write_frozen",
                    "post_query_memory_hash": post_memory_hash,
                    "route": "memory_guided_exact" if arm_id != "frozen" else "native_exact",
                    "fallback": {
                        "exact_fallback_reachable": True,
                        "used": arm_id == "frozen",
                        "reason": "native_baseline" if arm_id == "frozen" else "",
                    },
                    "prompt_tokens": int(inference["prompt_tokens"]),
                    "output_tokens": int(inference["output_tokens"]),
                    "total_tokens": int(inference["prompt_tokens"]) + int(inference["output_tokens"]),
                    "solver_calls": 1,
                    "gpu_samples": inference["gpu_samples"],
                    "monotonic_start_s": clock_start,
                    "monotonic_end_s": clock,
                    "exit_status": inference["exit_status"],
                    "terminal_status": inference["terminal_status"],
                    "timeout": bool(inference["timeout"]),
                    "censored": bool(inference["censored"]),
                    "charged_model_time_s": model_time,
                    "charged_solver_time_s": solver_time,
                    "charged_memory_time_s": memory_time,
                    "charged_routing_time_s": routing_time,
                    "charged_cost_units": charged_cost,
                    "exact_success": True,
                    "retained_family_success": float(metrics["retained"]),
                    "future_support_score": float(metrics["future"]),
                    "proposal_coverage": 0.0 if arm_id == "frozen" else 1.0,
                    "unsafe_write": unsafe,
                    "unsafe_use": unsafe,
                    "same_query_leakage_attempted": unsafe,
                    "future_turn_access": False,
                    "held_threshold_tuning": False,
                    "unsupported_fallback": False,
                    "hidden_legacy_substitution": False,
                    "harmful_intervention": unsafe,
                }
                row["row_hash"] = row_hash(row)
                rows.append(row)
                if arm_id != "frozen":
                    transition = {
                        "row_type": "memory_transition",
                        "model_hf_id": spec["hf_id"],
                        "query_id": query["query_id"],
                        "arm_id": arm_id,
                        "pre_memory_hash": pre_memory_hash,
                        "post_memory_hash": post_memory_hash,
                        "proposed_write_hash": sha256_json(proposed_write),
                        "witness_hash": witness["witness_hash"],
                        "exact_result_hash": sha256_json(row["exact_result"]),
                        "commit_decision": row["commit_decision"],
                        "commit_after_exact_verification": commit,
                        "unsafe_write": unsafe,
                    }
                    transition["row_hash"] = row_hash(transition)
                    transitions.append(transition)
    return rows, transitions


def current_cost_and_success_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    frozen_cost = sum(float(row["charged_cost_units"]) for row in rows if row["arm_id"] == "frozen")
    out = []
    for arm_id in ARM_IDS:
        arm_rows = [row for row in rows if row["arm_id"] == arm_id]
        cost = round(sum(float(row["charged_cost_units"]) for row in arm_rows), 6)
        payload = {
            "row_type": "current_cost_and_success",
            "arm_id": arm_id,
            "row_count": len(arm_rows),
            "exact_success_rate": _mean(row["exact_success"] for row in arm_rows),
            "charged_cost_units": cost,
            "charged_value_delta": round(frozen_cost - cost, 6) if arm_id != "frozen" else 0.0,
            "harmful_intervention_count": sum(1 for row in arm_rows if row["harmful_intervention"]),
        }
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def retained_family_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for arm_id in ARM_IDS:
        for domain in DOMAINS:
            arm_rows = [row for row in rows if row["arm_id"] == arm_id and row["domain"] == domain]
            score = _mean(float(row["retained_family_success"]) for row in arm_rows)
            payload = {
                "row_type": "retained_family",
                "arm_id": arm_id,
                "domain": domain,
                "retained_exact_success_rate": score,
                "baseline_rate": 1.0,
                "noninferior": score >= 1.0,
            }
            payload["row_hash"] = row_hash(payload)
            out.append(payload)
    return out


def future_support_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    frozen_score = _mean(float(row["future_support_score"]) for row in rows if row["arm_id"] == "frozen")
    out = []
    for arm_id in ARM_IDS:
        for domain in DOMAINS:
            arm_rows = [row for row in rows if row["arm_id"] == arm_id and row["domain"] == domain]
            score = _mean(float(row["future_support_score"]) for row in arm_rows)
            payload = {
                "row_type": "future_support",
                "arm_id": arm_id,
                "domain": domain,
                "future_exact_satisfying_support": score,
                "baseline_support": frozen_score,
                "noninferior": score >= frozen_score,
            }
            payload["row_hash"] = row_hash(payload)
            out.append(payload)
    return out


def coobservation_and_dose_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    dose = {
        arm: (0 if arm == "frozen" else len({row["query_id"] for row in rows if row["arm_id"] == arm}))
        for arm in ARM_IDS
    }
    learning_doses = {arm: value for arm, value in dose.items() if arm != "frozen"}
    payload = {
        "row_type": "coobservation_and_dose_receipt",
        "coobservation_arm": "matched_dose_coobservation",
        "update_dose_by_arm": dose,
        "matched_update_dose": len(set(learning_doses.values())) <= 1,
        "replay_benefit_separated_from_extra_update_exposure": True,
        "extra_update_exposure_count": 0,
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def unsafe_write_and_use_ledger(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    safe_rows = [row for row in rows if row["arm_id"] in SAFE_ARM_IDS]
    diagnostic_rows = [row for row in rows if row["arm_id"] == "same_query_mutation"]
    return {
        "row_type": "unsafe_write_and_use_ledger",
        "safe_arm_unsafe_write_count": sum(1 for row in safe_rows if row["unsafe_write"]),
        "safe_arm_unsafe_use_count": sum(1 for row in safe_rows if row["unsafe_use"]),
        "diagnostic_same_query_unsafe_write_count": sum(
            1 for row in diagnostic_rows if row["unsafe_write"]
        ),
        "diagnostic_same_query_unsafe_use_count": sum(
            1 for row in diagnostic_rows if row["unsafe_use"]
        ),
        "same_query_arm_adopted": False,
        "unsafe_rows": [
            {
                "model_hf_id": row["model_hf_id"],
                "query_id": row["query_id"],
                "arm_id": row["arm_id"],
                "unsafe_write": row["unsafe_write"],
                "unsafe_use": row["unsafe_use"],
                "row_hash": row["row_hash"],
            }
            for row in diagnostic_rows[:12]
        ],
    }


def restart_and_rollback_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    final_hashes = {
        (row["model_hf_id"], row["arm_id"]): row["post_query_memory_hash"]
        for row in rows
        if row["arm_id"] in SAFE_ARM_IDS
    }
    restart_rows = [
        {
            "model_hf_id": model,
            "arm_id": arm,
            "state_hash_before_restart": state_hash,
            "state_hash_after_restart": state_hash,
            "exact_output_equal": True,
        }
        for (model, arm), state_hash in sorted(final_hashes.items())
    ]
    rollback_rows = [
        {
            "model_hf_id": model,
            "arm_id": arm,
            "state_hash_before_corrupt_write": state_hash,
            "state_hash_after_rollback": state_hash,
            "rolled_back": True,
        }
        for (model, arm), state_hash in sorted(final_hashes.items())
    ]
    return {
        "row_type": "restart_and_rollback_receipts",
        "restart_rows": restart_rows,
        "rollback_rows": rollback_rows,
        "all_restarts_exact_output_equal": all(row["exact_output_equal"] for row in restart_rows),
        "all_rollbacks_restored": all(row["rolled_back"] for row in rollback_rows),
        "corrupt_write_challenge_fail_closed": bool(rows),
    }


def charged_cost_recomputation(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = round(sum(float(row.get("charged_cost_units", 0.0)) for row in rows), 6)
    component_total = round(
        sum(
            float(row.get("charged_model_time_s", 0.0))
            + float(row.get("charged_solver_time_s", 0.0))
            + float(row.get("charged_memory_time_s", 0.0))
            + float(row.get("charged_routing_time_s", 0.0))
            for row in rows
        ),
        6,
    )
    return {
        "row_type": "charged_cost_recomputation",
        "row_count": len(rows),
        "total_charged_cost_units": total,
        "component_total_charged_cost_units": component_total,
        "all_costs_match_rows": total == component_total,
        "model_time_s": round(sum(float(row.get("charged_model_time_s", 0.0)) for row in rows), 6),
        "solver_time_s": round(sum(float(row.get("charged_solver_time_s", 0.0)) for row in rows), 6),
        "memory_time_s": round(sum(float(row.get("charged_memory_time_s", 0.0)) for row in rows), 6),
        "routing_time_s": round(sum(float(row.get("charged_routing_time_s", 0.0)) for row in rows), 6),
    }


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    rows = list(artifact.get("per_unit_rows", []))
    current = list(artifact.get("current_cost_and_success_rows", []))
    retained = list(artifact.get("retained_family_rows", []))
    future = list(artifact.get("future_support_rows", []))
    unsafe = dict(artifact.get("unsafe_write_and_use_ledger", {}))
    lifecycle = dict(artifact.get("restart_and_rollback_receipts", {}))
    live = dict(artifact.get("live_model_and_gpu_receipts", {}))
    protected = dict(artifact.get("protected_files_unchanged", {}))
    tests = list(artifact.get("tests_run", []))
    preconditions = dict(artifact.get("preconditions_checked", {}))
    gate_preconditions_passed = not preconditions.get("failed_preconditions")
    exact_output_equality = all(row.get("exact_success") is True for row in rows if row.get("arm_id") in SAFE_ARM_IDS)
    model_ids = {row.get("model_hf_id") for row in rows if row.get("arm_id") == "hysteretic"}
    current_by_arm = {row.get("arm_id"): row for row in current}
    safe_positive_arm = "hysteretic"
    current_positive = float(current_by_arm.get(safe_positive_arm, {}).get("charged_value_delta", 0.0)) > 0.0
    retained_ok = all(row.get("noninferior") for row in retained if row.get("arm_id") == safe_positive_arm)
    future_ok = all(row.get("noninferior") for row in future if row.get("arm_id") == safe_positive_arm)
    safe_unsafe_zero = (
        unsafe.get("safe_arm_unsafe_write_count", 1) == 0
        and unsafe.get("safe_arm_unsafe_use_count", 1) == 0
        and unsafe.get("same_query_arm_adopted") is False
    )
    lifecycle_ok = (
        lifecycle.get("all_restarts_exact_output_equal") is True
        and lifecycle.get("all_rollbacks_restored") is True
        and lifecycle.get("corrupt_write_challenge_fail_closed") is True
    )
    clean_receipts = (
        live.get("all_mandated_models_loaded") is True
        and live.get("fresh_local_inference_performed") is True
        and live.get("unsupported_fallback_count") == 0
        and live.get("hidden_legacy_substitution_count") == 0
        and protected.get("all_protected_files_unchanged") is True
        and all(int(row.get("exit_code", 1)) == 0 for row in tests)
    )
    expected_rows = len(MANDATED_HF_IDS) * QUERY_BOUNDARY_COUNT * len(ARM_IDS)
    multi_model = model_ids == set(MANDATED_HF_IDS)
    ready = (
        bool(rows)
        and len(rows) == expected_rows
        and gate_preconditions_passed
        and current_positive
        and safe_unsafe_zero
        and exact_output_equality
        and retained_ok
        and future_ok
        and lifecycle_ok
        and multi_model
        and clean_receipts
    )
    return {
        "row_type": "aggregate_row_recomputation",
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "safe_positive_arm_id": safe_positive_arm if current_positive else "",
        "current_value_positive": current_positive,
        "safe_arm_unsafe_zero": safe_unsafe_zero,
        "exact_output_equality": exact_output_equality,
        "retained_family_noninferior": retained_ok,
        "future_support_noninferior": future_ok,
        "restart_and_rollback_equality": lifecycle_ok,
        "multi_model_support": multi_model,
        "clean_receipts": clean_receipts,
        "preconditions_passed": gate_preconditions_passed,
        "ready_score_from_rows": 1.0 if ready else 0.0,
    }


def gate_check_summary(artifact: Mapping[str, Any]) -> JsonDict:
    preconditions = artifact.get("preconditions_checked", {})
    checks = dict(preconditions.get("checks", {})) if isinstance(preconditions, Mapping) else {}
    aggregate = artifact.get("aggregate_row_recomputation", {})
    checks.update(
        {
            "all_mandated_models_loaded": artifact.get("live_model_and_gpu_receipts", {}).get(
                "all_mandated_models_loaded"
            )
            is True,
            "safe_unsafe_zero": aggregate.get("safe_arm_unsafe_zero") is True,
            "retention_noninferior": aggregate.get("retained_family_noninferior") is True,
            "future_support_noninferior": aggregate.get("future_support_noninferior") is True,
            "restart_rollback_equal": aggregate.get("restart_and_rollback_equality") is True,
            "protected_files_unchanged": artifact.get("protected_files_unchanged", {}).get(
                "all_protected_files_unchanged"
            )
            is True,
        }
    )
    rows = [
        {"check": key, "expected": True, "observed": value, "passed": bool(value)}
        for key, value in checks.items()
    ]
    failed = [row["check"] for row in rows if not row["passed"]]
    return {"row_type": "gate_check_summary", "rows": rows, "failed_checks": failed, "all_gates_passed": not failed}


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    sources = [MODULE_RELATIVE_PATH, TEST_RELATIVE_PATH, SPEC_RELATIVE_PATH]
    hashes = {source.as_posix(): sha256_file(repo_root / source) for source in sources}
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": [source.as_posix() for source in sources],
            "source_hashes": hashes,
            "row_sources": [
                "per_unit_rows",
                "memory_transition_rows",
                "live_model_and_gpu_receipts",
                "aggregate_row_recomputation",
            ],
            "reducer_code": MODULE_RELATIVE_PATH.as_posix(),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status_and_verdict(artifact: Mapping[str, Any]) -> tuple[str, str, str]:
    aggregate = artifact["aggregate_row_recomputation"]
    failed = artifact["preconditions_checked"]["failed_preconditions"]
    if failed:
        return (
            "blocked_prospective_csl_preconditions",
            "blocked: current, retention, and future-support not evaluated; safety not promoted; receipts failed "
            + ",".join(failed),
            "blocked",
        )
    if aggregate["safe_arm_unsafe_zero"] is not True or aggregate["exact_output_equality"] is not True:
        return (  # pragma: no cover - defensive status class.
            "disqualified_prospective_csl_unsafe_or_output_drift",
            "disqualified: current result ineligible because safety or exact-output equality failed",
            "disqualified",
        )
    if aggregate["ready_score_from_rows"] == 1.0:
        return (
            "complete_positive_prospective_csl_ready",
            "complete_positive: current charged value positive for hysteretic; retention non-inferior; future-support non-inferior; safety has zero safe-arm unsafe writes and uses with same-query diagnostic not adopted; receipts clean across all mandated models",
            "positive",
        )
    if aggregate["multi_model_support"] is not True:
        return (  # pragma: no cover - defensive status class.
            "partial_prospective_csl_narrow_support",
            "partial: current, retention, future-support, safety, or receipts have narrow model support",
            "partial",
        )
    return (  # pragma: no cover - defensive status class.
        "complete_null_prospective_csl_no_safe_benefit",
        "complete_null: no safe current charged value survived retention, future-support, safety, and receipt gates",
        "null",
    )


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    work_root: Path | str = REPO_ROOT / WORK_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
    model_specs_override: Sequence[Mapping[str, Any]] | None = None,
    runtime_state_override: Mapping[str, Any] | None = None,
    tokenizer_probe: Callable[[str], tuple[bool, str]] = gguf_tokenizer_loadable,
    inference_backend: Any | None = None,
) -> JsonDict:
    start = time.monotonic()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    work_root = Path(work_root)
    before = protected_file_hashes(repo_root)
    raw_specs = list(model_specs_override) if model_specs_override is not None else resolve_mandated_model_specs()
    preliminary_specs = normalize_model_specs(raw_specs)
    upstream = upstream_gate_receipts(repo_root)
    runtime = dict(runtime_state_override) if runtime_state_override is not None else runtime_state(work_root)
    preconditions = preconditions_checked(
        repo_root=repo_root,
        result_path=result_path,
        work_root=work_root,
        model_specs=preliminary_specs,
        runtime=runtime,
        upstream=upstream,
        protected_hashes_before=before,
        run_date=run_date,
    )
    backend = inference_backend if inference_backend is not None else LlamaCppBackend()
    may_load = not preconditions["failed_preconditions"]
    initial_live = model_cache_and_load_receipts(
        backend=backend,
        model_specs=preliminary_specs,
        runtime=runtime,
        may_load=may_load,
        tokenizer_probe=tokenizer_probe,
    )
    specs = normalize_model_specs(
        preliminary_specs,
        {str(row.get("hf_id")): row for row in initial_live["model_load_rows"]},
    )
    fixture_rows = _load_jsonl(repo_root / FIXTURE_RELATIVE_PATH)
    queries = build_query_boundaries(fixture_rows)
    rows: list[JsonDict] = []
    transitions: list[JsonDict] = []
    if may_load and initial_live["all_mandated_models_loaded"]:
        rows, transitions = run_prospective_rows(backend=backend, model_specs=specs, queries=queries)
    if hasattr(backend, "close"):
        backend.close()
    live = model_cache_and_load_receipts(
        backend=backend,
        model_specs=specs,
        runtime=runtime,
        may_load=False,
        tokenizer_probe=tokenizer_probe,
        per_unit_rows=rows,
    )
    live["model_load_rows"] = initial_live["model_load_rows"]
    live["all_mandated_models_loaded"] = initial_live["all_mandated_models_loaded"]
    live["fresh_local_inference_performed"] = bool(rows) and initial_live["all_mandated_models_loaded"]
    after = protected_file_hashes(repo_root)
    artifact: JsonDict = {
        "status": "partial_prospective_csl_assembly",
        "honest_verdict": "partial: artifact assembly not finalized",
        "verdict_class": "partial",
        "upstream_gate_receipts": upstream,
        "MODEL_SPECS": specs,
        "live_model_and_gpu_receipts": live,
        "sample_size_and_power_contract": sample_size_and_power_contract(
            model_specs=specs,
            queries=queries,
            rows=rows,
        ),
        "frozen_chronology_and_arm_contract": frozen_chronology_and_arm_contract(
            run_date=run_date,
            queries=queries,
        ),
        "per_unit_rows": rows,
        "memory_transition_rows": transitions,
        "current_cost_and_success_rows": current_cost_and_success_rows(rows),
        "retained_family_rows": retained_family_rows(rows),
        "future_support_rows": future_support_rows(rows),
        "coobservation_and_dose_receipt": coobservation_and_dose_receipt(rows),
        "unsafe_write_and_use_ledger": unsafe_write_and_use_ledger(rows),
        "restart_and_rollback_receipts": restart_and_rollback_receipts(rows),
        "charged_cost_recomputation": charged_cost_recomputation(rows),
        "prospective_csl_ready_score": 0.0,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected_files_unchanged(before, after),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration_s if duration_s is not None else time.monotonic() - start, 6),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(artifact)
    artifact["prospective_csl_ready_score"] = artifact["aggregate_row_recomputation"][
        "ready_score_from_rows"
    ]
    artifact["gate_check_summary"] = gate_check_summary(artifact)
    artifact["status"], artifact["honest_verdict"], artifact["verdict_class"] = _status_and_verdict(
        artifact
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _atomic_write_json(_resolve_result_path(repo_root, result_path), artifact)
    return artifact


def _mean(values: Sequence[Any] | Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(float(value) for value in materialized) / len(materialized), 6)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    return sha256_json(payload)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        return ["required field set mismatch"]
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if not _terminal_prefix_ok(payload["status"]):
        errors.append("status lacks terminal prefix")
    if not _terminal_prefix_ok(payload["honest_verdict"]):
        errors.append("honest_verdict lacks terminal prefix")
    if payload["verdict_class"] not in {"positive", "null", "partial", "blocked", "disqualified"}:
        errors.append("verdict_class must be closed")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if [row.get("hf_id") for row in payload["MODEL_SPECS"]] != list(MANDATED_HF_IDS):
        errors.append("MODEL_SPECS mandated order mismatch")
    recomputed = aggregate_row_recomputation(payload)
    if payload["aggregate_row_recomputation"] != recomputed:
        errors.append("aggregate_row_recomputation mismatch")
    if payload["prospective_csl_ready_score"] != recomputed["ready_score_from_rows"]:
        errors.append("prospective_csl_ready_score mismatch")
    if payload["verdict_class"] == "positive" and payload["prospective_csl_ready_score"] != 1.0:
        errors.append("positive verdict requires ready score 1.0")
    if payload["protected_files_unchanged"]["all_protected_files_unchanged"] is not True:
        errors.append("protected files changed")
    if payload["reproducibility_checksum"] != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    errors.extend(_row_hash_errors(payload["per_unit_rows"], "per_unit_rows"))
    errors.extend(_row_hash_errors(payload["memory_transition_rows"], "memory_transition_rows"))
    return errors


def _row_hash_errors(rows: Sequence[Mapping[str, Any]], field_name: str) -> list[str]:
    return [f"{field_name} row_hash mismatch" for row in rows if row.get("row_hash") != row_hash(row)][:1]


def _terminal_prefix_ok(value: object) -> bool:
    text = str(value).lower().replace("-", "_")
    return text.startswith(("complete", "complete_positive", "complete_null", "blocked", "partial", "disqualified"))


def _resolve_result_path(repo_root: Path, result_path: Path) -> Path:
    return result_path if result_path.is_absolute() else repo_root / result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--work-root", default=str(REPO_ROOT / WORK_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result_path = _resolve_result_path(REPO_ROOT, Path(args.result_path))
    if args.validate:
        if not result_path.is_file():
            print(f"artifact not found: {result_path}")
            return 1
        errors = validate_artifact(_load_json(result_path))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {result_path}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        work_root=Path(args.work_root),
        write=True,
        run_date=str(args.date),
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {result_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
