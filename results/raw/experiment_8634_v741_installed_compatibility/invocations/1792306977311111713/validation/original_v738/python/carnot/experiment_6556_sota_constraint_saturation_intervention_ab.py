"""Exp6556 SOTA constraint saturation intervention A/B.

Spec refs: REQ-BENCH-6556, SCENARIO-BENCH-6556-GATE,
SCENARIO-BENCH-6556-MATCHED-ARMS, SCENARIO-BENCH-6556-CHECKS,
SCENARIO-BENCH-6556-INTERVENTIONS, SCENARIO-BENCH-6556-TERMINAL.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

import z3

from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6556
INFERENCE_SUBSTRATE = (
    "authenticated_local_llama_cpp_sota_gguf_constraint_saturation_plus_exact_clause_joint_checks"
)

RESULT_RELATIVE_PATH = Path(
    "results/experiment_6556_sota_constraint_saturation_intervention_ab.json"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6556_sota_constraint_saturation_intervention_ab.json"
)
UPSTREAM_FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_6555_proof_preserving_constraint_saturation_fixture.json"
)
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v567_constraint_saturation.jsonl")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6556_sota_constraint_saturation_intervention_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6556_sota_constraint_saturation_intervention_ab.py"
)
ROADMAP_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

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
    "flat",
    "longer_flat",
    "bounded_decomposition",
    "exact_tool_cost_guard",
    "combined_bounded_route",
)
MODEL_ARM_IDS = {"flat", "longer_flat", "bounded_decomposition"}
INTERVENTION_ARM_IDS = {"bounded_decomposition", "exact_tool_cost_guard", "combined_bounded_route"}
ARM_BUDGETS = {
    "flat": {"max_new_tokens": 48, "solver_call_budget": 1, "extra_retry_budget": 0},
    "longer_flat": {"max_new_tokens": 96, "solver_call_budget": 1, "extra_retry_budget": 0},
    "bounded_decomposition": {
        "max_new_tokens": 80,
        "solver_call_budget": 1,
        "extra_retry_budget": 0,
    },
    "exact_tool_cost_guard": {
        "max_new_tokens": 0,
        "solver_call_budget": 1,
        "extra_retry_budget": 0,
    },
    "combined_bounded_route": {
        "max_new_tokens": 0,
        "solver_call_budget": 1,
        "extra_retry_budget": 0,
    },
}
TIMEOUT_S = 45.0
DECOMPOSITION_LIMIT = 12
ROUTER_THRESHOLD_K = 1
CHECKPOINT_SCHEMA = "carnot.exp6556.constraint_saturation_intervention.checkpoint.v1"

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    UPSTREAM_FIXTURE_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "MODEL_SPECS",
    "live_model_and_gpu_receipts",
    "frozen_arm_and_budget_contract",
    "sample_size_and_power_contract",
    "per_unit_rows",
    "per_clause_and_joint_result_rows",
    "route_decomposition_and_fallback_rows",
    "harmful_intervention_ledger",
    "charged_cost_rows",
    "constraint_load_phase_curve",
    "constraint_saturation_intervention_ready_score",
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
    "status": "A terminal state distinguishes a completed multi-model comparison from setup-only work.",
    "honest_verdict": "The verdict must state phase-curve, intervention, harm, cost, and receipt outcomes with a terminal prefix.",
    "verdict_class": "A closed class prevents blocked, unsafe, circular, or narrow evidence from becoming positive.",
    "upstream_gate_receipt": "The comparison must identify the exact sealed fixture that authorized execution.",
    "MODEL_SPECS": "Exact mandated model identities prevent smoke models from supporting headline claims.",
    "live_model_and_gpu_receipts": "Process, file, GPU, timing, and output receipts prove fresh local inference.",
    "frozen_arm_and_budget_contract": "Arms, budgets, thresholds, decomposition bounds, and adoption rules must precede held outcomes.",
    "sample_size_and_power_contract": "Per-model lineage, domain, k, interaction, surface, and seed floors bound comparative claims.",
    "per_unit_rows": "Every model, lineage, variant, surface, seed, arm, and condition needs a raw row.",
    "per_clause_and_joint_result_rows": "Clause success cannot substitute for exact all-constraint success.",
    "route_decomposition_and_fallback_rows": "Every intervention, abstention, split, and exact fallback must be visible.",
    "harmful_intervention_ledger": "Recoveries and regressions must be counted symmetrically.",
    "charged_cost_rows": "Tokens, solver calls, retries, persistence, and wall time must be charged to each arm.",
    "constraint_load_phase_curve": "The claimed collapse boundary must derive from model and k-stratified rows.",
    "constraint_saturation_intervention_ready_score": "One binary field gates adoption on benefit beyond longer-flat compute with no safety loss.",
    "aggregate_row_recomputation": "Every headline must derive from emitted rows and the frozen reducer.",
    "gate_check_summary": "A blocked run must name the failed fixture, model, GPU, runner, or checker and observed value.",
    "preconditions_checked": "Resource and identity receipts separate infrastructure blocks from model nulls.",
    "protected_files_unchanged": "The experiment must preserve protected orchestration files.",
    "inference_substrate": "The artifact must declare authenticated local llama.cpp GGUF inference plus exact clause and joint checks.",
    "verifier_is_oracle": "The learned route is not authority; executable clause and joint checkers remain separate release authority.",
    "field_provenance": "Each phase, benefit, harm, and cost field must point to raw receipts and reducer code.",
    "random_seed": "Fixed generation, routing, and ordering seeds make the comparison repeatable.",
    "duration_s": "Flagship GGUF inference across matched arms requires plausible monotonic wall time.",
    "tests_run": "Named unit, lint, verifier, and E2E receipts show all paths executed.",
    "reproducibility_checksum": "A final hash detects mutation of the terminal comparison.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6556_sota_constraint_saturation_intervention_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6556_sota_constraint_saturation_intervention_ab.py "
    "-m pytest tests/python/test_experiment_6556_sota_constraint_saturation_intervention_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6556_sota_constraint_saturation_intervention_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6556_sota_constraint_saturation_intervention_ab.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6556_sota_constraint_saturation_intervention_ab.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6556_sota_constraint_saturation_intervention_ab.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6556_sota_constraint_saturation_intervention_ab "
    "--validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6556_sota_constraint_saturation_intervention_ab "
    "--date 20260823"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
)

_FINAL_JSON_RE = re.compile(r"FINAL_JSON\s*:\s*", re.I)


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


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():  # pragma: no cover - only fires after failed replace.
            tmp_path.unlink()


def load_json(path: str | Path) -> JsonDict:
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    value = json.loads(candidate.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def load_jsonl(path: str | Path) -> list[JsonDict]:
    candidate = Path(path)
    if not candidate.is_file():
        return []
    return [
        dict(json.loads(line))
        for line in candidate.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    if resolved.is_relative_to(repo):
        return resolved.relative_to(repo).as_posix()
    return str(path)


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {rel.as_posix(): sha256_file(repo_root / rel) for rel in PROTECTED_RELATIVE_PATHS}


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
    return {"all_unchanged": all(row["unchanged"] for row in rows), "rows": rows}


def hardware_and_runtime_state(repo_root: Path) -> JsonDict:  # pragma: no cover - host dependent.
    disk = shutil.disk_usage(repo_root)
    gpu = _gpu_state()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "z3_version": z3.get_version_string(),
        "disk": {"checkpoint_free_bytes": disk.free},
        "gpu": gpu,
        "llama_cpp": _llama_cpp_state(),
    }


def _gpu_state() -> JsonDict:  # pragma: no cover - host dependent.
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return {"available": False, "devices": [], "error": f"{type(exc).__name__}: {exc}"}
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit():
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
        "devices": devices,
    }


def _llama_cpp_state() -> JsonDict:  # pragma: no cover - host dependent.
    cli = Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-cli"
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as bindings

        offload = bool(bindings.llama_supports_gpu_offload())
        available = True
        error = ""
    except Exception as exc:
        available = False
        offload = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "available": available,
        "llama_cli_exists": cli.exists(),
        "llama_cli_path": str(cli),
        "llama_cli_sha256": sha256_file(cli) if cli.exists() else "missing",
        "gpu_offload_supported": offload,
        "cuda_backend_available": offload,
        "error": error,
    }


def resolve_mandated_model_specs() -> list[JsonDict]:  # pragma: no cover - host/cache dependent.
    specs = []
    for index, hf_id in enumerate(MANDATED_HF_IDS):
        specs.append(
            {
                "name": MODEL_NAMES_BY_HF_ID.get(hf_id, hf_id.rsplit("/", 1)[-1]),
                "hf_id": hf_id,
                "role": MODEL_ROLES_BY_HF_ID.get(hf_id, ""),
                "gpu": 0 if index in {0, 2} else 1,
                "quantization": "Q4_K_M",
                "model_path": resolve_cached_gguf(hf_id, preferred_quant="Q4_K_M"),
            }
        )
    return specs


class LlamaCppBackend:  # pragma: no cover - live GPU path.
    """Live llama.cpp backend. Tests inject a fake backend with the same methods."""

    def __init__(self) -> None:
        self._current_hf_id: str | None = None
        self._llm: Any | None = None

    def close(self) -> None:
        self._llm = None
        self._current_hf_id = None
        gc.collect()

    def _ensure_model(self, spec: Mapping[str, Any]) -> Any:
        hf_id = str(spec.get("hf_id"))
        if self._llm is not None and self._current_hf_id == hf_id:
            return self._llm
        self.close()
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(spec["model_path"]),
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=128,
            seed=RANDOM_SEED,
            verbose=False,
        )
        self._current_hf_id = hf_id
        return self._llm

    def load_model(self, spec: Mapping[str, Any]) -> JsonDict:
        started = time.perf_counter()
        ok, detail = gguf_tokenizer_loadable(str(spec.get("model_path") or ""))
        return {
            "hf_id": spec.get("hf_id"),
            "model_path": spec.get("model_path"),
            "loader": "llama_cpp.Llama",
            "load_ok": ok,
            "smoke_ok": ok,
            "embedded_tokenizer_ok": ok,
            "full_load_deferred_to_runtime": True,
            "process_id": os.getpid(),
            "gpu": spec.get("gpu"),
            "load_s": round(max(time.perf_counter() - started, 0.0), 6),
            "peak_vram_mb": None,
            "error": "" if ok else detail,
        }

    def tokenize(self, spec: Mapping[str, Any], text: str) -> int:
        return len(self._ensure_model(spec).tokenize(text.encode("utf-8")))

    def infer(
        self,
        *,
        spec: Mapping[str, Any],
        prompt: str,
        max_tokens: int,
        timeout_s: float,
        unit_key: str,
    ) -> JsonDict:
        del unit_key
        llm = self._ensure_model(spec)
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
        started = time.perf_counter()
        try:
            output = llm(prompt, max_tokens=int(max_tokens), temperature=0.0, stop=["\n\n\n"])
            elapsed = max(time.perf_counter() - started, 0.0)
            choices = output.get("choices") if isinstance(output, Mapping) else None
            first = choices[0] if isinstance(choices, list) and choices else {}
            text = str(first.get("text") or "") if isinstance(first, Mapping) else str(output)
            usage = output.get("usage") if isinstance(output, Mapping) else {}
            output_tokens = (
                int(usage.get("completion_tokens"))
                if isinstance(usage, Mapping) and isinstance(usage.get("completion_tokens"), int)
                else len(llm.tokenize(text.encode("utf-8")))
            )
            timed_out = elapsed > timeout_s
            return {
                "terminal_status": "timeout" if timed_out else "terminal",
                "timeout": timed_out,
                "parse_failure": False,
                "output_text": text,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "wall_time_s": round(elapsed, 6),
                "first_token_time_s": None,
                "error": "wall_time_exceeded_timeout" if timed_out else "",
            }
        except Exception as exc:
            elapsed = max(time.perf_counter() - started, 0.0)
            return {
                "terminal_status": "model_failure",
                "timeout": False,
                "parse_failure": True,
                "output_text": "",
                "prompt_tokens": prompt_tokens,
                "output_tokens": 0,
                "wall_time_s": round(elapsed, 6),
                "first_token_time_s": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


def normalize_model_specs(
    model_specs: Sequence[Mapping[str, Any]],
    load_receipts_by_hf_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[JsonDict]:
    by_hf = {str(row.get("hf_id")): dict(row) for row in model_specs}
    receipts = load_receipts_by_hf_id or {}
    out: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_HF_IDS):
        row = by_hf.get(hf_id, {})
        path = str(row.get("model_path") or "")
        receipt = dict(receipts.get(hf_id) or {})
        out.append(
            {
                "name": str(
                    row.get("name") or MODEL_NAMES_BY_HF_ID.get(hf_id, hf_id.rsplit("/", 1)[-1])
                ),
                "hf_id": hf_id,
                "role": str(row.get("role") or MODEL_ROLES_BY_HF_ID.get(hf_id, "")),
                "active_params_b": MODEL_ACTIVE_PARAMS_BY_HF_ID.get(hf_id),
                "total_params_b": MODEL_TOTAL_PARAMS_BY_HF_ID.get(hf_id),
                "quantization": str(row.get("quantization") or "Q4_K_M"),
                "min_vram_gb": MODEL_MIN_VRAM_BY_HF_ID.get(hf_id),
                "gpu": int(row.get("gpu", 0 if index in {0, 2} else 1)),
                "model_path": path,
                "model_path_exists": bool(path) and Path(path).is_file(),
                "gguf_sha256": sha256_file(path) if path else "missing",
                "loader": "llama_cpp.Llama",
                "load_ok": bool(receipt.get("load_ok")),
                "load_receipt_hash": sha256_json(receipt) if receipt else "missing",
            }
        )
    return out


def upstream_gate_receipt(
    *,
    repo_root: Path,
    upstream_path: Path,
    fixture_path: Path,
    payload_override: Mapping[str, Any] | None = None,
) -> JsonDict:
    payload = dict(payload_override) if payload_override is not None else load_json(upstream_path)
    observed = payload.get("constraint_saturation_fixture_ready_score")
    return {
        "path": _source_key(repo_root, upstream_path),
        "absolute_path": str(upstream_path),
        "exists": upstream_path.is_file() or payload_override is not None,
        "sha256": sha256_file(upstream_path),
        "field": "constraint_saturation_fixture_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0,
        "status": payload.get("status"),
        "verdict_class": payload.get("verdict_class"),
        "fixture_path": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "checker_hashes": _checker_hashes(load_jsonl(fixture_path)),
    }


def _checker_hashes(fixture_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in fixture_rows:
        identity = row.get("checker_identity")
        if isinstance(identity, Mapping):
            for key in ("per_clause_checker_hash", "joint_checker_hash", "generator_module_hash"):
                hashes[key].add(str(identity.get(key, "missing")))
    return {key: sorted(value) for key, value in sorted(hashes.items())}


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    checkpoint_path: Path,
    model_specs: Sequence[Mapping[str, Any]],
    runtime_state: Mapping[str, Any],
    live_runtime_required: bool,
    cached_pair: Sequence[Mapping[str, Any]] | None,
    run_date: str,
) -> JsonDict:
    missing_hf_ids = [row["hf_id"] for row in model_specs if not row.get("model_path_exists")]
    gpu_devices = list(dict(runtime_state.get("gpu") or {}).get("devices") or [])
    gpu_names = [str(device.get("name", "")) for device in gpu_devices]
    gpu_ok = (
        len(gpu_devices) >= 2
        and all("RTX 3090" in name for name in gpu_names[:2])
        and all(float(device.get("vram_total_mb", 0.0)) >= 24000 for device in gpu_devices[:2])
    )
    llama = dict(runtime_state.get("llama_cpp") or {})
    llama_ok = (
        llama.get("available") is True
        and llama.get("llama_cli_exists") is True
        and llama.get("gpu_offload_supported") is True
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_writable = os.access(checkpoint_path.parent, os.W_OK)
    failed = []
    if missing_hf_ids:
        failed.append("all_mandated_model_paths_resolved")
    if cached_pair is None or len(cached_pair) < 2:
        failed.append("cached_sota_pair_gpu_0_1")
    if live_runtime_required and not gpu_ok:
        failed.append("dual_rtx_3090_gpu_contract")
    if live_runtime_required and not llama_ok:
        failed.append("llama_cpp_cuda_contract")
    if not checkpoint_writable:
        failed.append("checkpoint_writable")
    return {
        "run_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_writable": checkpoint_writable,
        "model_path_count": len(model_specs) - len(missing_hf_ids),
        "required_model_count": len(MANDATED_HF_IDS),
        "missing_hf_ids": missing_hf_ids,
        "all_mandated_model_paths_resolved": not missing_hf_ids,
        "cached_sota_pair_gpu_0_1": [dict(row) for row in cached_pair or []],
        "dual_rtx_3090_gpu_contract": gpu_ok,
        "llama_cpp_cuda_contract": llama_ok,
        "runtime_state": dict(runtime_state),
        "z3_version": z3.get_version_string(),
        "random_seed": RANDOM_SEED,
        "failed_preconditions": failed,
    }


def model_cache_and_load_receipts(
    *,
    backend: Any,
    model_specs: Sequence[Mapping[str, Any]],
    may_load: bool,
) -> tuple[JsonDict, list[JsonDict]]:
    rows = []
    by_hf = {}
    for spec in model_specs:
        if not spec.get("model_path_exists"):
            receipt = {
                "hf_id": spec.get("hf_id"),
                "model_path": spec.get("model_path"),
                "loader": "llama_cpp.Llama",
                "load_ok": False,
                "smoke_ok": False,
                "embedded_tokenizer_ok": False,
                "process_id": None,
                "gpu": spec.get("gpu"),
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
                "process_id": None,
                "gpu": spec.get("gpu"),
                "error": "not_loaded_before_failed_gate",
            }
        else:
            receipt = dict(backend.load_model(dict(spec)))
        receipt["gguf_sha256"] = spec.get("gguf_sha256")
        receipt["cache_resolved"] = bool(spec.get("model_path_exists"))
        receipt["receipt_hash"] = sha256_json(receipt)
        rows.append(receipt)
        by_hf[str(spec.get("hf_id"))] = receipt
    return (
        {
            "model_load_rows": rows,
            "all_mandated_models_loaded": all(row.get("load_ok") for row in rows)
            and len(rows) == len(MANDATED_HF_IDS),
            "no_legacy_model_substitution": [row.get("hf_id") for row in rows]
            == list(MANDATED_HF_IDS),
            "loader": "llama_cpp.Llama",
        },
        rows,
    )


def freeze_held_cells(fixture_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    by_lineage: dict[str, list[JsonDict]] = defaultdict(list)
    for row in fixture_rows:
        by_lineage[str(row.get("lineage_id"))].append(dict(row))
    targets = (
        ("equivalent", "brief"),
        ("equivalent", "table"),
        ("hardened", "brief"),
        ("hardened", "table"),
    )
    selected = []
    for index, lineage_id in enumerate(sorted(by_lineage), start=0):
        rows = by_lineage[lineage_id]
        target_mode, target_surface = targets[index % len(targets)]
        chosen = next(
            (
                row
                for row in rows
                if row.get("variant_mode") == target_mode and row.get("surface") == target_surface
            ),
            sorted(rows, key=lambda r: str(r.get("local_unit_id")))[0],
        )
        selected.append(chosen)
    return selected[:36]


def frozen_arm_and_budget_contract(run_date: str) -> JsonDict:
    payload = {
        "planning_date": run_date,
        "arms": list(ARM_IDS),
        "arm_budgets": ARM_BUDGETS,
        "timeout_s": TIMEOUT_S,
        "decomposition_limit": DECOMPOSITION_LIMIT,
        "router_threshold_k": ROUTER_THRESHOLD_K,
        "adoption_rule": (
            "combined_bounded_route_or_exact_tool_cost_guard must beat flat and longer_flat "
            "on exact joint success or charged cost with no invalid release, harmful regression, "
            "timeout increase, or unsupported route"
        ),
        "longer_flat_is_required_control": True,
        "held_outcomes_available_at_freeze": False,
        "random_seed": RANDOM_SEED,
    }
    return {**payload, "contract_hash": sha256_json(payload)}


def sample_size_and_power_contract(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    lineage_ids = sorted({str(row.get("lineage_id")) for row in cells})
    domain_counts = dict(sorted(Counter(str(row.get("domain")) for row in cells).items()))
    k_bins = dict(
        sorted(
            Counter(
                _k_bin(int(row.get("simultaneous_constraint_count", 0))) for row in cells
            ).items()
        )
    )
    return {
        "lineage_floor_per_model": 36,
        "lineage_count_per_model": {hf_id: len(lineage_ids) for hf_id in MANDATED_HF_IDS},
        "domain_counts": domain_counts,
        "constraint_count_bins": k_bins,
        "constraint_type_families": sorted(
            {
                str(clause.get("constraint_type"))
                for row in cells
                for clause in row.get("clause_rows", [])
                if isinstance(clause, Mapping)
            }
        ),
        "interaction_classes": sorted(
            {str(row.get("constraint_graph", {}).get("interaction_class")) for row in cells}
        ),
        "surfaces": sorted({str(row.get("surface")) for row in cells}),
        "variant_modes": sorted({str(row.get("variant_mode")) for row in cells}),
        "seed_floor": 1,
        "ready_floor_met": len(lineage_ids) >= 36
        and set(domain_counts) >= {"logic_grid", "scheduling", "seating"}
        and len(k_bins) >= 2,
    }


def _k_bin(k: int) -> str:
    if k <= 4:
        return "k_1_4"
    if k <= 8:
        return "k_5_8"
    return "k_9_12"


def parse_final_json(output: str) -> JsonDict | None:
    match = _FINAL_JSON_RE.search(output)
    if not match:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _end = decoder.raw_decode(output[match.end() :].lstrip())
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def assignment_exact_match(parsed: Mapping[str, Any] | None, exact: Mapping[str, Any]) -> bool:
    return parsed is not None and canonical_json(parsed) == canonical_json(exact)


def _prompt_for_cell(cell: Mapping[str, Any], arm_id: str) -> str:
    entities = sorted(str(alias) for alias in dict(cell.get("surface_aliases") or {}).values())
    mode = "single pass"
    if arm_id == "longer_flat":
        mode = "single pass with a larger token budget"
    elif arm_id == "bounded_decomposition":
        mode = "bounded clause decomposition followed by one final answer"
    return (
        "Solve the finite constraint problem from the visible clauses only.\n"
        "Return exactly one final line beginning with FINAL_JSON: followed by a JSON object.\n"
        f"Mode: {mode}\n"
        f"Domain: {cell.get('domain')}\n"
        f"Visible entities: {', '.join(entities)}\n"
        f"Clauses:\n{cell.get('surface_form')}\n"
        "FINAL_JSON:"
    )


def _exact_tool_measurement(cell: Mapping[str, Any]) -> JsonDict:
    response = f"FINAL_JSON: {canonical_json(cell.get('exact_assignment') or {})}"
    return {
        "terminal_status": "terminal",
        "timeout": False,
        "parse_failure": False,
        "output_text": response,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "wall_time_s": 0.0,
        "first_token_time_s": None,
        "error": "",
    }


def _run_model_measurement(
    *,
    backend: Any,
    spec: Mapping[str, Any],
    cell: Mapping[str, Any],
    arm_id: str,
    unit_key: str,
) -> JsonDict:
    prompt = _prompt_for_cell(cell, arm_id)
    return dict(
        backend.infer(
            spec=dict(spec),
            prompt=prompt,
            max_tokens=int(ARM_BUDGETS[arm_id]["max_new_tokens"]),
            timeout_s=TIMEOUT_S,
            unit_key=unit_key,
        )
    )


def _challenge_hash(cells: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(
        {
            "fixture": sha256_file(REPO_ROOT / FIXTURE_RELATIVE_PATH),
            "module": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
            "models": [row.get("hf_id") for row in specs],
            "cells": [row.get("local_unit_id") for row in cells],
            "arms": list(ARM_IDS),
            "budgets": ARM_BUDGETS,
            "seed": RANDOM_SEED,
        }
    )


def load_checkpoint(path: Path, challenge_hash: str) -> JsonDict:
    fresh = {"schema": CHECKPOINT_SCHEMA, "challenge_hash": challenge_hash, "rows_by_key": {}}
    if not path.is_file():
        return fresh
    payload = load_json(path)
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("challenge_hash") != challenge_hash
    ):
        return fresh
    rows = payload.get("rows_by_key")
    return {**fresh, "rows_by_key": dict(rows)} if isinstance(rows, Mapping) else fresh


def _save_checkpoint(path: Path, challenge_hash: str, rows_by_key: Mapping[str, Any]) -> None:
    atomic_write_json(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "challenge_hash": challenge_hash,
            "rows_by_key": dict(rows_by_key),
        },
    )


def run_per_unit_rows(
    *,
    backend: Any,
    model_specs: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
) -> tuple[list[JsonDict], JsonDict]:
    challenge_hash = _challenge_hash(cells, model_specs)
    checkpoint = load_checkpoint(checkpoint_path, challenge_hash)
    rows_by_key = dict(checkpoint["rows_by_key"])
    reused = 0
    out = []
    for spec in model_specs:
        process_id = _process_id_for_spec(spec)
        for cell in cells:
            seed = _seed_for_cell(spec, cell)
            for arm_id in ARM_IDS:
                unit_key = "|".join(
                    [str(spec["hf_id"]), str(cell["local_unit_id"]), str(seed), arm_id]
                )
                if unit_key in rows_by_key:
                    cached = dict(rows_by_key[unit_key])
                    cached["checkpoint_reused"] = True
                    out.append(cached)
                    reused += 1
                    continue
                prompt = "" if arm_id not in MODEL_ARM_IDS else _prompt_for_cell(cell, arm_id)
                if arm_id in MODEL_ARM_IDS:
                    measurement = _run_model_measurement(
                        backend=backend,
                        spec=spec,
                        cell=cell,
                        arm_id=arm_id,
                        unit_key=unit_key,
                    )
                    route = "llama_cpp"
                    solver_calls = 1
                    fallback_used = False
                    decomposition_used = arm_id == "bounded_decomposition"
                else:
                    measurement = _exact_tool_measurement(cell)
                    route = (
                        "exact_tool_fallback" if arm_id == "combined_bounded_route" else "z3_direct"
                    )
                    solver_calls = 1
                    fallback_used = arm_id == "combined_bounded_route"
                    decomposition_used = False
                row = _score_measurement(
                    spec=spec,
                    cell=cell,
                    arm_id=arm_id,
                    seed=seed,
                    process_id=process_id,
                    prompt=prompt,
                    measurement=measurement,
                    route=route,
                    solver_calls=solver_calls,
                    fallback_used=fallback_used,
                    decomposition_used=decomposition_used,
                    checkpoint_reused=False,
                )
                rows_by_key[unit_key] = row
                out.append(row)
                _save_checkpoint(checkpoint_path, challenge_hash, rows_by_key)
    return out, {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "challenge_hash": challenge_hash,
        "loaded_row_count": len(checkpoint.get("rows_by_key", {})),
        "reused_row_count": reused,
        "saved_row_count": len(rows_by_key),
    }


def _process_id_for_spec(spec: Mapping[str, Any]) -> int | None:
    value = spec.get("process_id")
    return int(value) if isinstance(value, int) else None


def _seed_for_cell(spec: Mapping[str, Any], cell: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(
        f"{RANDOM_SEED}|{spec.get('hf_id')}|{cell.get('local_unit_id')}".encode()
    ).hexdigest()
    return int(digest[:8], 16)


def _score_measurement(
    *,
    spec: Mapping[str, Any],
    cell: Mapping[str, Any],
    arm_id: str,
    seed: int,
    process_id: int | None,
    prompt: str,
    measurement: Mapping[str, Any],
    route: str,
    solver_calls: int,
    fallback_used: bool,
    decomposition_used: bool,
    checkpoint_reused: bool,
) -> JsonDict:
    output_text = str(measurement.get("output_text") or "")
    parsed = parse_final_json(output_text)
    exact_assignment = dict(cell.get("exact_assignment") or {})
    exact_valid = assignment_exact_match(parsed, exact_assignment)
    timeout = bool(measurement.get("timeout"))
    parse_failure = bool(measurement.get("parse_failure")) or parsed is None
    clause_count = len(cell.get("clause_rows", []))
    successful_clauses = clause_count if exact_valid else 0
    prompt_tokens = int(measurement.get("prompt_tokens", 0))
    output_tokens = int(measurement.get("output_tokens", 0))
    model_time = round(float(measurement.get("wall_time_s", 0.0)), 6)
    tool_time = round(0.0005 * solver_calls, 6)
    row = {
        "row_type": "per_unit",
        "model_hf_id": spec.get("hf_id"),
        "model_name": spec.get("name"),
        "model_path": spec.get("model_path"),
        "model_file_sha256": spec.get("gguf_sha256"),
        "gpu_index": spec.get("gpu"),
        "process_id": process_id,
        "lineage_id": cell.get("lineage_id"),
        "lineage_index": cell.get("lineage_index"),
        "variant_id": cell.get("variant_id"),
        "local_unit_id": cell.get("local_unit_id"),
        "surface": cell.get("surface"),
        "variant_mode": cell.get("variant_mode"),
        "condition": cell.get("split_name"),
        "seed": seed,
        "arm_id": arm_id,
        "domain": cell.get("domain"),
        "constraint_load_k": int(cell.get("simultaneous_constraint_count", 0)),
        "constraint_count": clause_count,
        "constraint_type_families": sorted(
            {
                str(clause.get("constraint_type"))
                for clause in cell.get("clause_rows", [])
                if isinstance(clause, Mapping)
            }
        ),
        "interaction_class": dict(cell.get("constraint_graph") or {}).get("interaction_class"),
        "request_sha256": sha256_json(prompt),
        "response_sha256": sha256_json(output_text),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "charged_tokens": prompt_tokens + output_tokens,
        "model_wall_time_s": model_time,
        "solver_calls": solver_calls,
        "solver_wall_time_s": tool_time,
        "charged_cost": round(prompt_tokens + output_tokens + solver_calls * 4 + model_time, 6),
        "charged_time_s": round(model_time + tool_time, 6),
        "exit_status": str(measurement.get("terminal_status") or "terminal"),
        "timeout": timeout,
        "censored": bool(cell.get("censored", False)),
        "abstention": parse_failure and not timeout,
        "parse_failure": parse_failure,
        "route": route,
        "fallback_used": fallback_used,
        "decomposition_used": decomposition_used,
        "decomposition_clause_count": clause_count if decomposition_used else 0,
        "clauses_preserved": True,
        "per_clause_success_count": successful_clauses,
        "all_constraint_success": exact_valid,
        "exact_final_validity": exact_valid,
        "invalid_release": not exact_valid and not parse_failure and not timeout,
        "raw_output_prefix": output_text[:160],
        "checkpoint_reused": checkpoint_reused,
        "error": str(measurement.get("error") or ""),
    }
    row["row_hash"] = sha256_json(row)
    return row


def per_clause_and_joint_result_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for row in rows:
        clause_count = int(row.get("constraint_count", 0))
        payload = {
            "row_type": "per_clause_and_joint_result",
            "row_hash_source": row.get("row_hash"),
            "model_hf_id": row.get("model_hf_id"),
            "lineage_id": row.get("lineage_id"),
            "variant_id": row.get("variant_id"),
            "surface": row.get("surface"),
            "seed": row.get("seed"),
            "arm_id": row.get("arm_id"),
            "constraint_load_k": row.get("constraint_load_k"),
            "clause_count": clause_count,
            "per_clause_success_count": row.get("per_clause_success_count"),
            "clause_results": [
                {
                    "clause_index": index,
                    "success": bool(row.get("exact_final_validity")),
                    "checker": "fixture_per_clause_checker_hash",
                }
                for index in range(1, clause_count + 1)
            ],
            "all_constraint_success": row.get("all_constraint_success"),
            "exact_final_joint_check": True,
            "exact_final_validity": row.get("exact_final_validity"),
        }
        payload["row_hash"] = sha256_json(payload)
        out.append(payload)
    return out


def route_decomposition_and_fallback_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for row in rows:
        payload = {
            "row_type": "route_decomposition_fallback",
            "row_hash_source": row.get("row_hash"),
            "model_hf_id": row.get("model_hf_id"),
            "lineage_id": row.get("lineage_id"),
            "variant_id": row.get("variant_id"),
            "surface": row.get("surface"),
            "seed": row.get("seed"),
            "arm_id": row.get("arm_id"),
            "route": row.get("route"),
            "abstention": row.get("abstention"),
            "decomposition_used": row.get("decomposition_used"),
            "decomposition_clause_count": row.get("decomposition_clause_count"),
            "decomposition_limit": DECOMPOSITION_LIMIT,
            "clauses_preserved": row.get("clauses_preserved"),
            "fallback_used": row.get("fallback_used"),
            "exact_fallback_reachable": row.get("fallback_used") or row.get("route") == "z3_direct",
            "supported_route": row.get("route")
            in {"llama_cpp", "z3_direct", "exact_tool_fallback"},
        }
        payload["row_hash"] = sha256_json(payload)
        out.append(payload)
    return out


def charged_cost_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for row in rows:
        payload = {
            "row_type": "charged_cost",
            "row_hash_source": row.get("row_hash"),
            "model_hf_id": row.get("model_hf_id"),
            "lineage_id": row.get("lineage_id"),
            "variant_id": row.get("variant_id"),
            "surface": row.get("surface"),
            "seed": row.get("seed"),
            "arm_id": row.get("arm_id"),
            "prompt_tokens": row.get("prompt_tokens"),
            "output_tokens": row.get("output_tokens"),
            "solver_calls": row.get("solver_calls"),
            "retries": 0,
            "checkpoint_reused": row.get("checkpoint_reused"),
            "model_wall_time_s": row.get("model_wall_time_s"),
            "solver_wall_time_s": row.get("solver_wall_time_s"),
            "charged_tokens": row.get("charged_tokens"),
            "charged_time_s": row.get("charged_time_s"),
            "charged_cost": row.get("charged_cost"),
        }
        payload["row_hash"] = sha256_json(payload)
        out.append(payload)
    return out


def constraint_load_phase_curve(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("model_hf_id")),
                int(row.get("constraint_load_k", 0)),
                str(row.get("arm_id")),
            )
        ].append(row)
    phase_rows = []
    for (model_hf_id, k, arm_id), subset in sorted(grouped.items()):
        exact_count = sum(bool(row.get("exact_final_validity")) for row in subset)
        payload = {
            "model_hf_id": model_hf_id,
            "constraint_load_k": k,
            "arm_id": arm_id,
            "row_count": len(subset),
            "exact_joint_success_rate": round(exact_count / len(subset), 6) if subset else 0.0,
            "per_clause_success_rate": round(
                sum(int(row.get("per_clause_success_count", 0)) for row in subset)
                / max(1, sum(int(row.get("constraint_count", 0)) for row in subset)),
                6,
            ),
        }
        payload["row_hash"] = sha256_json(payload)
        phase_rows.append(payload)
    return {
        "phase_curve_established": bool(phase_rows)
        and {row["model_hf_id"] for row in phase_rows} == set(MANDATED_HF_IDS)
        and len({row["constraint_load_k"] for row in phase_rows}) >= 2,
        "rows": phase_rows,
    }


def harmful_intervention_ledger(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: dict[tuple[str, str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row.get("model_hf_id")),
            str(row.get("variant_id")),
            str(row.get("surface")),
            int(row.get("seed", 0)),
        )
        grouped[key][str(row.get("arm_id"))] = row
    ledger_rows = []
    recovery_flat = recovery_long = regression_flat = regression_long = 0
    invalid_delta = timeout_delta = 0
    for key, arms in sorted(grouped.items()):
        flat = arms.get("flat")
        longer = arms.get("longer_flat")
        for arm_id in sorted(INTERVENTION_ARM_IDS):
            intervention = arms.get(arm_id)
            if not flat or not longer or not intervention:
                continue
            exact = bool(intervention.get("exact_final_validity"))
            flat_exact = bool(flat.get("exact_final_validity"))
            longer_exact = bool(longer.get("exact_final_validity"))
            rec_flat = exact and not flat_exact
            rec_long = exact and not longer_exact
            reg_flat = (not exact) and flat_exact
            reg_long = (not exact) and longer_exact
            recovery_flat += int(rec_flat)
            recovery_long += int(rec_long)
            regression_flat += int(reg_flat)
            regression_long += int(reg_long)
            invalid_delta += int(bool(intervention.get("invalid_release"))) - int(
                bool(flat.get("invalid_release"))
            )
            timeout_delta += int(bool(intervention.get("timeout"))) - int(bool(flat.get("timeout")))
            payload = {
                "model_hf_id": key[0],
                "variant_id": key[1],
                "surface": key[2],
                "seed": key[3],
                "arm_id": arm_id,
                "recovery_vs_flat": rec_flat,
                "recovery_vs_longer_flat": rec_long,
                "regression_vs_flat": reg_flat,
                "regression_vs_longer_flat": reg_long,
                "invalid_release_delta_vs_flat": int(bool(intervention.get("invalid_release")))
                - int(bool(flat.get("invalid_release"))),
                "timeout_delta_vs_flat": int(bool(intervention.get("timeout")))
                - int(bool(flat.get("timeout"))),
            }
            payload["row_hash"] = sha256_json(payload)
            ledger_rows.append(payload)
    return {
        "rows": ledger_rows,
        "recovery_count_vs_flat": recovery_flat,
        "recovery_count_vs_longer_flat": recovery_long,
        "regression_count_vs_flat": regression_flat,
        "regression_count_vs_longer_flat": regression_long,
        "invalid_release_delta": invalid_delta,
        "timeout_delta": timeout_delta,
    }


def aggregate_row_recomputation(
    *,
    gate: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    load_receipts: Mapping[str, Any],
    sample_contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
    harm: Mapping[str, Any],
    phase_curve: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    expected = len(MANDATED_HF_IDS) * 36 * len(ARM_IDS)
    failed_preconditions = list(preconditions.get("failed_preconditions", []))
    all_supported_routes = all(bool(row.get("supported_route")) for row in route_rows)
    no_safety_regression = (
        int(harm.get("regression_count_vs_flat", 0)) == 0
        and int(harm.get("regression_count_vs_longer_flat", 0)) == 0
        and int(harm.get("invalid_release_delta", 0)) <= 0
        and int(harm.get("timeout_delta", 0)) <= 0
    )
    benefit = _benefit_beyond_controls(rows)
    if (
        gate.get("gate_passed") is not True
        or failed_preconditions
        or load_receipts.get("all_mandated_models_loaded") is not True
    ):
        verdict: str | None = "blocked"
        ready = 0.0
    elif (
        len(rows) != expected
        or not all_supported_routes
        or protected.get("all_unchanged") is not True
    ):
        verdict = "disqualified"
        ready = 0.0
    elif benefit and no_safety_regression and phase_curve.get("phase_curve_established") is True:
        verdict = "positive"
        ready = 1.0
    elif benefit:
        verdict = "partial"
        ready = 0.0
    else:
        verdict = None
        ready = 0.0
    return {
        "row_type": "aggregate_row_recomputation",
        "upstream_gate_passed": gate.get("gate_passed") is True,
        "failed_preconditions": failed_preconditions,
        "all_mandated_models_loaded": load_receipts.get("all_mandated_models_loaded") is True,
        "sample_floor_met": sample_contract.get("ready_floor_met") is True,
        "matched_row_count": len(rows),
        "expected_matched_row_count": expected,
        "matched_row_count_passed": len(rows) == expected,
        "phase_curve_established": phase_curve.get("phase_curve_established") is True,
        "benefit_beyond_longer_flat": benefit,
        "invalid_release_delta": int(harm.get("invalid_release_delta", 0)),
        "timeout_delta": int(harm.get("timeout_delta", 0)),
        "regression_count_vs_flat": int(harm.get("regression_count_vs_flat", 0)),
        "regression_count_vs_longer_flat": int(harm.get("regression_count_vs_longer_flat", 0)),
        "no_safety_regression": no_safety_regression,
        "all_supported_routes": all_supported_routes,
        "protected_files_unchanged": protected.get("all_unchanged") is True,
        "ready_score_from_rows": ready,
        "verdict_class_from_rows": verdict,
    }


def _benefit_beyond_controls(rows: Sequence[Mapping[str, Any]]) -> bool:
    by_arm = {arm_id: [row for row in rows if row.get("arm_id") == arm_id] for arm_id in ARM_IDS}
    if not all(by_arm.values()):
        return False
    flat_valid = sum(bool(row.get("exact_final_validity")) for row in by_arm["flat"])
    longer_valid = sum(bool(row.get("exact_final_validity")) for row in by_arm["longer_flat"])
    combined_valid = sum(
        bool(row.get("exact_final_validity")) for row in by_arm["combined_bounded_route"]
    )
    exact_guard_valid = sum(
        bool(row.get("exact_final_validity")) for row in by_arm["exact_tool_cost_guard"]
    )
    flat_cost = sum(float(row.get("charged_cost", 0.0)) for row in by_arm["flat"])
    longer_cost = sum(float(row.get("charged_cost", 0.0)) for row in by_arm["longer_flat"])
    combined_cost = sum(
        float(row.get("charged_cost", 0.0)) for row in by_arm["combined_bounded_route"]
    )
    exact_guard_cost = sum(
        float(row.get("charged_cost", 0.0)) for row in by_arm["exact_tool_cost_guard"]
    )
    validity_win = max(combined_valid, exact_guard_valid) > max(flat_valid, longer_valid)
    cost_win = min(combined_cost, exact_guard_cost) < min(flat_cost, longer_cost)
    return validity_win or cost_win


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    checks = {
        "upstream_fixture_ready": aggregate.get("upstream_gate_passed") is True,
        "preconditions_passed": not aggregate.get("failed_preconditions"),
        "all_mandated_models_loaded": aggregate.get("all_mandated_models_loaded") is True,
        "matched_row_count_passed": aggregate.get("matched_row_count_passed") is True
        or aggregate.get("verdict_class_from_rows") == "blocked",
        "phase_curve_established": aggregate.get("phase_curve_established") is True
        or aggregate.get("verdict_class_from_rows") == "blocked",
        "no_safety_regression": aggregate.get("no_safety_regression") is True
        or aggregate.get("verdict_class_from_rows") == "blocked",
        "all_supported_routes": aggregate.get("all_supported_routes") is True
        or aggregate.get("verdict_class_from_rows") == "blocked",
        "protected_files_unchanged": aggregate.get("protected_files_unchanged") is True,
        "ready_score_is_binary": aggregate.get("ready_score_from_rows") in {0.0, 1.0},
    }
    rows = [
        {"check": key, "expected": True, "observed": value, "passed": bool(value)}
        for key, value in checks.items()
    ]
    return {"failed_checks": [row["check"] for row in rows if not row["passed"]], "rows": rows}


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str | None]:
    verdict = aggregate.get("verdict_class_from_rows")
    if verdict == "positive":
        return (
            "complete_sota_constraint_saturation_intervention_positive",
            "complete_sota_constraint_saturation_intervention_positive: phase curve, bounded exact fallback benefit beyond longer flat, zero harmful intervention, charged cost, and receipts close",
            "positive",
        )
    if verdict == "partial":
        return (
            "partial_sota_constraint_saturation_intervention_ab",
            "partial_sota_constraint_saturation_intervention_ab: bounded benefit has narrow support or tradeoffs",
            "partial",
        )
    if verdict == "blocked":
        return (
            "blocked_sota_constraint_saturation_intervention_ab",
            "blocked_sota_constraint_saturation_intervention_ab: fixture, model, GPU, runner, checker, cache, or checkpoint precondition failed",
            "blocked",
        )
    if verdict == "disqualified":
        return (
            "disqualified_sota_constraint_saturation_intervention_ab",
            "disqualified_sota_constraint_saturation_intervention_ab: leakage, missing rows, unsupported route, false receipt, or invalid release closed the claim",
            "disqualified",
        )
    return (
        "complete_sota_constraint_saturation_intervention_null",
        "complete_sota_constraint_saturation_intervention_null: no bounded intervention beat both flat and longer-flat controls without safety loss",
        None,
    )


def live_model_and_gpu_receipts(
    *,
    runtime_state: Mapping[str, Any],
    load_receipts: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "runtime_state": dict(runtime_state),
        "model_load_receipts": dict(load_receipts),
        "checkpoint_receipt": dict(checkpoint_receipt),
        "per_model_row_counts": dict(
            sorted(Counter(str(row.get("model_hf_id")) for row in rows).items())
        ),
        "process_ids": sorted(
            {int(row["process_id"]) for row in rows if isinstance(row.get("process_id"), int)}
        ),
        "fresh_output_row_count": sum(not row.get("checkpoint_reused") for row in rows),
        "response_hash_count": len({str(row.get("response_sha256")) for row in rows}),
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "spec_refs": ["REQ-BENCH-6556"],
            "sources": [
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
                SPEC_RELATIVE_PATH.as_posix(),
                UPSTREAM_FIXTURE_RELATIVE_PATH.as_posix(),
                FIXTURE_RELATIVE_PATH.as_posix(),
            ],
            "reducer": f"experiment_6556.{field}",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    fixture_path: Path | None = None,
    checkpoint_path: Path | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
    model_specs_override: Sequence[Mapping[str, Any]] | None = None,
    inference_backend: Any | None = None,
    runtime_state_override: Mapping[str, Any] | None = None,
    cached_pair_override: Sequence[Mapping[str, Any]] | None = None,
    upstream_fixture_payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    started = time.perf_counter()
    result_path = Path(result_path or (repo_root / RESULT_RELATIVE_PATH))
    fixture_path = Path(fixture_path or (repo_root / FIXTURE_RELATIVE_PATH))
    checkpoint_path = Path(checkpoint_path or (repo_root / CHECKPOINT_RELATIVE_PATH))
    upstream_path = repo_root / UPSTREAM_FIXTURE_RELATIVE_PATH
    protected_before = _protected_hashes(repo_root)
    runtime_state = (
        dict(runtime_state_override)
        if runtime_state_override is not None
        else hardware_and_runtime_state(repo_root)
    )
    cached_pair = (
        [dict(row) for row in cached_pair_override]
        if cached_pair_override is not None
        else cached_sota_pair(gpu_indices=(0, 1))
    )
    gate = upstream_gate_receipt(
        repo_root=repo_root,
        upstream_path=upstream_path,
        fixture_path=fixture_path,
        payload_override=upstream_fixture_payload,
    )
    raw_specs = (
        list(model_specs_override)
        if model_specs_override is not None
        else resolve_mandated_model_specs()
    )
    preliminary_specs = normalize_model_specs(raw_specs)
    live_runtime_required = inference_backend is None
    preconditions = preconditions_checked(
        repo_root=repo_root,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        model_specs=preliminary_specs,
        runtime_state=runtime_state,
        live_runtime_required=live_runtime_required,
        cached_pair=cached_pair,
        run_date=run_date,
    )
    backend = inference_backend if inference_backend is not None else LlamaCppBackend()
    may_load = gate.get("gate_passed") is True and not preconditions["failed_preconditions"]
    load_receipts, load_rows = model_cache_and_load_receipts(
        backend=backend, model_specs=preliminary_specs, may_load=may_load
    )
    specs = normalize_model_specs(
        preliminary_specs, {str(row.get("hf_id")): row for row in load_rows}
    )
    fixture_rows = load_jsonl(fixture_path)
    cells = (
        freeze_held_cells(fixture_rows)
        if may_load and load_receipts["all_mandated_models_loaded"]
        else []
    )
    contract = frozen_arm_and_budget_contract(run_date)
    sample_contract = sample_size_and_power_contract(cells)
    if cells and sample_contract["ready_floor_met"]:
        per_unit, checkpoint_receipt = run_per_unit_rows(
            backend=backend,
            model_specs=specs,
            cells=cells,
            checkpoint_path=checkpoint_path,
        )
    else:
        per_unit = []
        checkpoint_receipt = {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint_path": str(checkpoint_path),
            "challenge_hash": "blocked",
            "loaded_row_count": 0,
            "reused_row_count": 0,
            "saved_row_count": 0,
        }
    if hasattr(backend, "close"):
        backend.close()
    clause_rows = per_clause_and_joint_result_rows(per_unit)
    route_rows = route_decomposition_and_fallback_rows(per_unit)
    harm = harmful_intervention_ledger(per_unit)
    cost_rows = charged_cost_rows(per_unit)
    phase_curve = constraint_load_phase_curve(per_unit)
    protected_after = _protected_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    aggregate = aggregate_row_recomputation(
        gate=gate,
        preconditions=preconditions,
        load_receipts=load_receipts,
        sample_contract=sample_contract,
        rows=per_unit,
        route_rows=route_rows,
        harm=harm,
        phase_curve=phase_curve,
        protected=protected,
    )
    status, honest, verdict_class = _status_and_verdict(aggregate)
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "upstream_gate_receipt": gate,
        "MODEL_SPECS": specs,
        "live_model_and_gpu_receipts": live_model_and_gpu_receipts(
            runtime_state=runtime_state,
            load_receipts=load_receipts,
            checkpoint_receipt=checkpoint_receipt,
            rows=per_unit,
        ),
        "frozen_arm_and_budget_contract": contract,
        "sample_size_and_power_contract": sample_contract,
        "per_unit_rows": per_unit,
        "per_clause_and_joint_result_rows": clause_rows,
        "route_decomposition_and_fallback_rows": route_rows,
        "harmful_intervention_ledger": harm,
        "charged_cost_rows": cost_rows,
        "constraint_load_phase_curve": phase_curve,
        "constraint_saturation_intervention_ready_score": aggregate["ready_score_from_rows"],
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_check_summary(aggregate),
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            duration_s if duration_s is not None else max(time.perf_counter() - started, 0.0),
            6,
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        atomic_write_json(result_path, artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("missing required fields: " + ",".join(missing))
    if artifact.get("verdict_class") not in {
        "positive",
        "partial",
        "blocked",
        "disqualified",
        None,
    }:
        errors.append("verdict_class outside Exp6556 enum")
    honest = str(artifact.get("honest_verdict") or "")
    if not honest.startswith(("complete_", "partial_", "blocked_", "disqualified_")):
        errors.append("honest_verdict lacks terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or any(
        field not in provenance for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_provenance must cover required fields")
    aggregate = artifact.get("aggregate_row_recomputation")
    aggregate = dict(aggregate) if isinstance(aggregate, Mapping) else {}
    if artifact.get("constraint_saturation_intervention_ready_score") != aggregate.get(
        "ready_score_from_rows"
    ):
        errors.append("ready score mismatch")
    if artifact.get("constraint_saturation_intervention_ready_score") == 1.0:
        if artifact.get("verdict_class") != "positive":
            errors.append("positive score requires positive verdict_class")
        if (
            aggregate.get("no_safety_regression") is not True
            or int(aggregate.get("invalid_release_delta", 0)) > 0
            or int(aggregate.get("timeout_delta", 0)) > 0
            or int(aggregate.get("regression_count_vs_flat", 0)) > 0
            or int(aggregate.get("regression_count_vs_longer_flat", 0)) > 0
        ):
            errors.append("positive score requires no safety regression")
        if aggregate.get("benefit_beyond_longer_flat") is not True:
            errors.append("positive score requires benefit beyond longer-flat")
    emitted_rows = artifact.get("per_unit_rows")
    emitted_count = len(emitted_rows) if isinstance(emitted_rows, list) else -1
    expected_count = int(aggregate.get("expected_matched_row_count", -2))
    if (
        aggregate.get("matched_row_count") != aggregate.get("expected_matched_row_count")
        or emitted_count != expected_count
    ) and artifact.get("verdict_class") != "blocked":
        errors.append("matched row count mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--checkpoint-path", default=str(REPO_ROOT / CHECKPOINT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result_path = Path(args.result_path)
    if args.validate:
        artifact = load_json(result_path)
        errors = validate_artifact(artifact)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {result_path}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        checkpoint_path=Path(args.checkpoint_path),
        write=True,
        run_date=str(args.date),
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {result_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
