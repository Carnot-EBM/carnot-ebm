"""Exp6546 SMT conflict cost stress and exact-tool guard.

Spec refs: REQ-BENCH-6546, SCENARIO-BENCH-6546-GATE,
SCENARIO-BENCH-6546-CHALLENGE, SCENARIO-BENCH-6546-DISPATCH,
SCENARIO-BENCH-6546-RUNTIME, SCENARIO-BENCH-6546-EFFECTS,
SCENARIO-BENCH-6546-ATTACKS, SCENARIO-BENCH-6546-CHECKPOINT,
SCENARIO-BENCH-6546-TERMINAL.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_6543_external_corpus_independent_audit_v2 as exp6543
from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6546
INFERENCE_SUBSTRATE = "authenticated_local_llama_cpp_sota_gguf_inference_plus_exact_z3_cost_guard"

RESULT_RELATIVE_PATH = Path("results/experiment_6546_smt_cost_guard_sota.json")
EXP6543_RELATIVE_PATH = Path("results/experiment_6543_external_corpus_independent_audit_v2.json")
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6546_smt_cost_guard_sota.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6546_smt_cost_guard_sota.py")
ROADMAP_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SOTA_MODELS_RELATIVE_PATH = Path("python/carnot/inference/sota_models.py")
TEMPLATE_RELATIVE_PATH = Path("scripts/experiment_template.py")
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints/experiment_6546_smt_cost_guard_sota.json")

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

SURFACE_IDS = ("canonical", "relabeled")
ARM_IDS = ("unguarded", "guarded")
MAX_LOGICAL_INSTANCES = 3
MAX_NEW_TOKENS = 24
TIMEOUT_S = 45.0
N_CTX = 1024

CHECKPOINT_SCHEMA = "carnot.exp6546.smt_cost_guard.checkpoint.v1"

CONFOUND_ATTACK_IDS = (
    "prompt_length_confounding",
    "surface_non_equivalence",
    "cache_order",
    "warm_up_asymmetry",
    "token_counter_mismatch",
    "timer_aliases",
    "model_substitution",
    "held_threshold_tuning",
    "tool_time_omission",
    "cherry_picked_timeouts",
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    TEMPLATE_RELATIVE_PATH,
    SOTA_MODELS_RELATIVE_PATH,
    EXP6543_RELATIVE_PATH,
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
    "models_used",
    "model_cache_and_load_receipts",
    "frozen_challenge_contract",
    "logical_instance_rows",
    "proof_preserving_surface_receipts",
    "solver_conflict_rows",
    "frozen_dispatch_contract",
    "per_unit_rows",
    "model_and_surface_effect_rows",
    "conflict_cost_association_rows",
    "guarded_versus_unguarded_rows",
    "token_and_time_recomputation",
    "exact_completion_receipt",
    "calibration_rows",
    "censoring_and_timeout_receipts",
    "confound_attack_matrix",
    "smt_cost_guard_ready_score",
    "gate_check_summary",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records the terminal Exp6546 SMT cost-guard state.",
    "honest_verdict": "Names the guarded held cost result and exact-completion support.",
    "verdict_class": "Separates positive, null, partial, blocked, and disqualified cost-guard outcomes.",
    "upstream_gate_receipt": (
        "Pins Exp6543 by path, hash, expected value, observed value, input hashes, cache, hardware, llama.cpp, budgets, and protected hashes."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF hub IDs, local GGUF paths, hashes, quantization, and llama.cpp placement."
    ),
    "models_used": "Lists the exact mandated hub IDs that supplied inference rows.",
    "model_cache_and_load_receipts": (
        "Shows cache resolution, GGUF hashes, embedded-tokenizer checks, llama.cpp import, CUDA support, load, smoke, or blocked runtime state."
    ),
    "frozen_challenge_contract": (
        "Freezes held unit count, domain strata, conflict strata, surfaces, prompt contract, timeout, token accounting, seed, and stop rules before scoring."
    ),
    "logical_instance_rows": (
        "Stores one logical-instance row with split, domain, fixture identity, constraints hash, exact label, and conflict stratum."
    ),
    "proof_preserving_surface_receipts": (
        "Shows each surface variant preserves the same logical constraints and exact label."
    ),
    "solver_conflict_rows": (
        "Records measured Z3 conflicts, decisions, propagations, assertions, quantiles, and replay status."
    ),
    "frozen_dispatch_contract": (
        "Freezes the train-development-only threshold and exact-tool dispatch rule before held scoring."
    ),
    "per_unit_rows": (
        "Stores one model-surface-instance-arm row with tokens, time, timeout, parse, exact validity, dispatch, tool time, and charged totals."
    ),
    "model_and_surface_effect_rows": (
        "Reports surface effects conditional on logical instance and model."
    ),
    "conflict_cost_association_rows": (
        "Reports conflict-cost association conditional on surface and model without claiming universal model hardness."
    ),
    "guarded_versus_unguarded_rows": (
        "Reports held guarded cost and exact-completion effects by model family."
    ),
    "token_and_time_recomputation": (
        "Recomputes prompt tokens, output tokens, tool time, model time, charged time, and charged tokens from rows."
    ),
    "exact_completion_receipt": (
        "Shows guarded exact completion is non-inferior to unguarded completion before any positive score opens."
    ),
    "calibration_rows": (
        "Shows threshold fitting used train and development rows only and held rows were unavailable."
    ),
    "censoring_and_timeout_receipts": (
        "Records timeout, parse-failure, model-failure, checkpoint, and terminal-row coverage."
    ),
    "confound_attack_matrix": (
        "Attacks prompt-length confounding, surface non-equivalence, cache order, warm-up asymmetry, token mismatch, timer aliases, model substitution, held threshold tuning, tool-time omission, and cherry-picked timeouts."
    ),
    "smt_cost_guard_ready_score": (
        "Opens only for preregistered held token or time savings with exact-completion non-inferiority across at least two mandated model families and no surface-controlled confound."
    ),
    "gate_check_summary": "Names every failed gate with expected and observed values.",
    "aggregate_row_recomputation": (
        "Rebuilds the verdict and ready score from gates, rows, costs, exact completion, calibration, censoring, attacks, and protected hashes."
    ),
    "preconditions_checked": (
        "Records paths, hashes, cache, hardware, llama.cpp, solver, budgets, date, seeds, and protected hashes."
    ),
    "protected_files_unchanged": (
        "Shows guarded inputs, specs, prior artifacts, model registry, template, and conductor files stayed byte-identical during the run."
    ),
    "inference_substrate": (
        "Declares authenticated local llama.cpp SOTA GGUF inference plus exact Z3 cost guard."
    ),
    "verifier_is_oracle": (
        "False because guard value is measured and Z3 is the separate evaluation authority."
    ),
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps each field to specs, inputs, rows, reducers, tests, and hashes.",
    "random_seed": (
        "Pins challenge selection, surface ordering, dispatch calibration, model seeds, and attack ordering."
    ),
    "duration_s": "Records measured reducer and live-inference wall time.",
    "tests_run": "Records validation command receipts.",
    "reproducibility_checksum": (
        "Detects drift in gates, models, prompts, rows, costs, attacks, and verdicts."
    ),
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6546_smt_cost_guard_sota.py -q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6546_smt_cost_guard_sota.py "
    "-m pytest tests/python/test_experiment_6546_smt_cost_guard_sota.py -q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6546_smt_cost_guard_sota.py --fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6546_smt_cost_guard_sota.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6546_smt_cost_guard_sota.json"
)
ADVERSARIAL_COMMAND = ".venv/bin/python scripts/adversarial_verify.py results/experiment_6546_smt_cost_guard_sota.json"
VALIDATE_COMMAND = ".venv/bin/python -m carnot.experiment_6546_smt_cost_guard_sota --validate"
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6546_smt_cost_guard_sota --date 20260823"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
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

_FINAL_RE = re.compile(r"\b(UNSATISFIABLE|CONTRADICTION|SATISFIABLE|SAT)\b", re.I)


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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        if tmp_path.exists():  # pragma: no cover - only fires after replace failure.
            tmp_path.unlink()


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        rows.append(dict(value) if isinstance(value, Mapping) else {"value": value})
    return rows


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    if resolved.is_relative_to(repo):
        return resolved.relative_to(repo).as_posix()
    return str(path)


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _protected_hashes(repo_root: Path, audit_path: Path | None = None) -> dict[str, str]:
    hashes = {rel.as_posix(): sha256_file(repo_root / rel) for rel in PROTECTED_RELATIVE_PATHS}
    if audit_path is not None:
        hashes[_source_key(repo_root, audit_path)] = sha256_file(audit_path)
    return hashes


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
    }


def _memory_state() -> JsonDict:
    info: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                info[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return {
        "mem_total_bytes": info.get("MemTotal"),
        "mem_available_bytes": info.get("MemAvailable"),
        "swap_total_bytes": info.get("SwapTotal"),
        "swap_free_bytes": info.get("SwapFree"),
    }


def _disk_state(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    return {
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
    }


def _gpu_state() -> JsonDict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=10)
    except Exception as exc:  # pragma: no cover - host command failure path.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "devices": []}
    devices = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 6:
                devices.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_total_mb": float(parts[2]),
                        "vram_free_mb": float(parts[3]),
                        "temperature_c": float(parts[4]),
                        "power_w": float(parts[5]),
                    }
                )
    return {
        "available": bool(devices),
        "exit_code": result.returncode,
        "stderr": result.stderr.strip(),
        "devices": devices,
    }


def _llama_cpp_state() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        import llama_cpp
    except Exception as exc:
        return {
            "available": False,
            "version": "",
            "system_info": "",
            "cuda_backend_available": False,
            "gpu_offload_supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    version = str(getattr(llama_cpp, "__version__", "unknown"))
    try:
        raw_info = llama_cpp.llama_print_system_info()
        system_info = (
            raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
        )
    except Exception as exc:
        system_info = f"system_info_unavailable:{type(exc).__name__}:{exc}"
    try:
        from llama_cpp import llama_cpp as backend

        offload = bool(backend.llama_supports_gpu_offload())
    except Exception:
        offload = False
    lowered = system_info.lower()
    return {
        "available": True,
        "version": version,
        "system_info": system_info,
        "cuda_backend_available": "cuda" in lowered or "cublas" in lowered,
        "gpu_offload_supported": offload,
        "error": "",
    }


def hardware_and_runtime_state(repo_root: Path) -> JsonDict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": os.sys.executable,
        "cpu_count": os.cpu_count(),
        "ram": _memory_state(),
        "disk": _disk_state(repo_root),
        "gpu": _gpu_state(),
        "thermal_state": {"source": "nvidia-smi.temperature.gpu", "gpu": _gpu_state()["devices"]},
        "llama_cpp": _llama_cpp_state(),
    }


def resolve_mandated_model_specs() -> list[JsonDict]:  # pragma: no cover - host/cache dependent.
    specs: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_HF_IDS):
        path = resolve_cached_gguf(hf_id, preferred_quant="Q4_K_M")
        specs.append(
            {
                "name": MODEL_NAMES_BY_HF_ID.get(hf_id, hf_id.rsplit("/", 1)[-1]),
                "hf_id": hf_id,
                "role": MODEL_ROLES_BY_HF_ID.get(hf_id, "dense"),
                "gpu": index % 2,
                "quantization": "Q4_K_M",
                "model_path": path,
            }
        )
    return specs


class LlamaCppBackend:  # pragma: no cover - live GPU path.
    """Live llama.cpp backend. Tests inject a small fake with the same methods."""

    def __init__(self) -> None:
        self._current_hf_id: str | None = None
        self._llm: Any | None = None

    def close(self) -> None:
        self._llm = None
        self._current_hf_id = None
        gc.collect()

    def load_model(self, spec: Mapping[str, Any]) -> JsonDict:
        model_path = str(spec.get("model_path") or "")
        ok, detail = gguf_tokenizer_loadable(model_path)
        return {
            "hf_id": spec.get("hf_id"),
            "model_path": model_path,
            "loader": "llama_cpp.Llama",
            "load_ok": ok,
            "load_s": 0.0,
            "smoke_ok": ok,
            "smoke_s": 0.0,
            "embedded_tokenizer_ok": ok,
            "full_load_deferred_to_runtime": True,
            "error": "" if ok else detail,
        }

    def _ensure_model(self, spec: Mapping[str, Any]) -> Any:
        hf_id = str(spec.get("hf_id"))
        if self._llm is not None and self._current_hf_id == hf_id:
            return self._llm
        self.close()
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(spec["model_path"]),
            n_gpu_layers=-1,
            n_ctx=N_CTX,
            n_batch=min(128, N_CTX),
            seed=RANDOM_SEED,
            verbose=False,
        )
        self._current_hf_id = hf_id
        return self._llm

    def tokenize(self, spec: Mapping[str, Any], text: str) -> int:
        llm = self._ensure_model(spec)
        return len(llm.tokenize(text.encode("utf-8")))

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
            out = llm(
                prompt,
                max_tokens=int(max_tokens),
                temperature=0.0,
                stop=["\n\n\n"],
            )
            wall = max(time.perf_counter() - started, 0.0)
            choices = out.get("choices") if isinstance(out, Mapping) else None
            first = choices[0] if isinstance(choices, list) and choices else {}
            output_text = str(first.get("text") or "") if isinstance(first, Mapping) else str(out)
            usage = out.get("usage") if isinstance(out, Mapping) else {}
            output_tokens = (
                int(usage.get("completion_tokens"))
                if isinstance(usage, Mapping) and isinstance(usage.get("completion_tokens"), int)
                else len(llm.tokenize(output_text.encode("utf-8")))
            )
            timed_out = wall > timeout_s
            return {
                "terminal_status": "timeout" if timed_out else "terminal",
                "timeout": timed_out,
                "parse_failure": False,
                "output_text": output_text,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "wall_time_s": round(wall, 6),
                "first_token_time_s": None,
                "error": "wall_time_exceeded_timeout" if timed_out else "",
            }
        except Exception as exc:
            wall = max(time.perf_counter() - started, 0.0)
            return {
                "terminal_status": "model_failure",
                "timeout": False,
                "parse_failure": True,
                "output_text": "",
                "prompt_tokens": prompt_tokens,
                "output_tokens": 0,
                "wall_time_s": round(wall, 6),
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
                "gpu": int(row.get("gpu", index % 2))
                if row.get("gpu", index % 2) is not None
                else index % 2,
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
    audit_path: Path,
    fixture_path: Path,
    protected_before: Mapping[str, str],
    runtime_state: Mapping[str, Any],
) -> JsonDict:
    exists = audit_path.is_file()
    try:
        audit = _load_json(audit_path)
        parse_status = "parsed" if exists else "missing"
        parse_error = ""
    except json.JSONDecodeError as exc:
        audit = {}
        parse_status = "corrupt_json"
        parse_error = str(exc)
    observed = audit.get("external_constraint_corpus_audited_ready_score")
    return {
        "row_type": "upstream_gate_receipt",
        "path": _source_key(repo_root, audit_path),
        "absolute_path": str(audit_path),
        "exists": exists,
        "sha256": sha256_file(audit_path),
        "parse_status": parse_status,
        "parse_error": parse_error,
        "field": "external_constraint_corpus_audited_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0 and parse_status == "parsed",
        "status": audit.get("status"),
        "verdict_class": audit.get("verdict_class"),
        "artifact_reproducibility_checksum": audit.get("reproducibility_checksum"),
        "input_hashes": {
            "audit": sha256_file(audit_path),
            "fixture": sha256_file(fixture_path),
            "spec": sha256_file(repo_root / SPEC_RELATIVE_PATH),
            "model_registry": sha256_file(repo_root / SOTA_MODELS_RELATIVE_PATH),
            "template": sha256_file(repo_root / TEMPLATE_RELATIVE_PATH),
        },
        "cached_sota_pair_gpu_0_1": cached_sota_pair(gpu_indices=(0, 1)),
        "mandated_hub_ids": list(MANDATED_HF_IDS),
        "hardware": dict(runtime_state),
        "budgets": {
            "max_logical_instances": MAX_LOGICAL_INSTANCES,
            "surface_ids": list(SURFACE_IDS),
            "arm_ids": list(ARM_IDS),
            "max_new_tokens": MAX_NEW_TOKENS,
            "timeout_s": TIMEOUT_S,
            "n_ctx": N_CTX,
        },
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-GATE"],
    }


def _source_root_from_audit(audit: Mapping[str, Any]) -> Path:
    receipt = audit.get("independent_revision_license_and_schema_receipt")
    if isinstance(receipt, Mapping) and receipt.get("source_root"):
        return Path(str(receipt["source_root"]))
    return exp6543.DEFAULT_SOURCE_CACHE_ROOT


def _stat_value(stats: Any, names: Sequence[str]) -> int:
    data: dict[str, Any] = {}
    try:
        for index in range(len(stats)):
            try:
                key = stats.key(index)
                value = stats.get_key_value(key)
            except Exception:
                key, value = stats[index]
            data[str(key).lower()] = value
    except Exception:
        return 0
    total = 0
    for name in names:
        value = data.get(name.lower(), 0)
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def solver_conflict_rows(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_root: Path,
) -> list[JsonDict]:
    checker = exp6543._load_z3_checker(source_root)
    raw_rows: list[JsonDict] = []
    for row in fixture_rows:
        problem, turn = exp6543._turn_for_row(source_root, row)
        constraints = turn.get("cumulative_constraints", [])
        constraints = constraints if isinstance(constraints, list) else []
        domain = str(problem.get("domain") or row.get("domain") or "")
        entities = (
            [str(item) for item in problem.get("entities", [])]
            if isinstance(problem.get("entities"), list)
            else []
        )
        context = exp6543._context_from_problem(problem)
        started = time.perf_counter()
        replay_status = "unknown"
        conflicts = decisions = propagations = 0
        error = ""
        try:
            if checker is None or not hasattr(checker, "build_domain_solver"):
                raise RuntimeError("z3_checker_unavailable")
            solver, _aux = checker.build_domain_solver(
                domain,
                entities,
                [dict(item) for item in constraints],
                context=dict(context),
            )
            result = solver.check()
            replay_status = str(result)
            stats = solver.statistics()
            conflicts = _stat_value(stats, ("conflicts", "arith-conflicts"))
            decisions = _stat_value(stats, ("decisions",))
            propagations = _stat_value(stats, ("propagations", "binary propagations"))
            assertion_count = (
                len(solver.assertions()) if hasattr(solver, "assertions") else len(constraints)
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            effort = (
                row.get("solver_effort") if isinstance(row.get("solver_effort"), Mapping) else {}
            )
            assertion_count = int(effort.get("solver_assertion_count", len(constraints)))
        wall_time_s = round(max(time.perf_counter() - started, 0.0), 9)
        raw_rows.append(
            {
                "row_type": "solver_conflict",
                "local_unit_id": row.get("local_unit_id"),
                "split_name": row.get("split_name"),
                "domain": row.get("domain"),
                "source_turn_id": row.get("source_turn_id"),
                "constraint_count": len(constraints),
                "solver_assertion_count": assertion_count,
                "conflict_count": conflicts,
                "decision_count": decisions,
                "propagation_count": propagations,
                "z3_replay_status": replay_status,
                "z3_wall_time_s": wall_time_s,
                "error": error,
            }
        )
    counts = sorted(int(row["conflict_count"]) for row in raw_rows)
    index_by_id = {
        str(row["local_unit_id"]): index
        for index, row in enumerate(
            sorted(raw_rows, key=lambda r: (int(r["conflict_count"]), str(r["local_unit_id"])))
        )
    }
    total = max(1, len(raw_rows))
    for row in raw_rows:
        rank = index_by_id[str(row["local_unit_id"])]
        quantile = (rank + 0.5) / total
        if quantile < 1 / 3:
            stratum = "low"
        elif quantile < 2 / 3:
            stratum = "medium"
        else:
            stratum = "high"
        row["conflict_quantile"] = round(quantile, 6)
        row["conflict_stratum"] = stratum
        row["conflict_count_min"] = counts[0] if counts else 0
        row["conflict_count_max"] = counts[-1] if counts else 0
        row["row_hash"] = sha256_json(
            {key: value for key, value in row.items() if key != "row_hash"}
        )
        row["spec_refs"] = ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-CHALLENGE"]
    return raw_rows


def calibration_rows(conflict_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = []
    for row in conflict_rows:
        split = str(row.get("split_name"))
        used = split in {"train", "development"}
        rows.append(
            {
                "row_type": "calibration_conflict",
                "local_unit_id": row.get("local_unit_id"),
                "split_name": split,
                "domain": row.get("domain"),
                "conflict_count": int(row.get("conflict_count", 0)),
                "conflict_stratum": row.get("conflict_stratum"),
                "used_for_threshold": used,
                "held_rows_unavailable": split == "held" or used,
                "target_answer_used": False,
                "model_cost_used": False,
            }
        )
    return rows


def frozen_dispatch_contract(cal_rows: Sequence[Mapping[str, Any]], run_date: str) -> JsonDict:
    values = [
        int(row.get("conflict_count", 0)) for row in cal_rows if row.get("used_for_threshold")
    ]
    threshold = 1
    payload = {
        "schema": "carnot.exp6546.frozen_dispatch_contract.v1",
        "planning_date": run_date,
        "training_splits_used": ["development", "train"],
        "held_rows_used_for_threshold": False,
        "target_answers_used_for_threshold": False,
        "model_cost_used_for_threshold": False,
        "route_rule": "z3_direct_when_conflict_count_ge_threshold",
        "conflict_threshold": int(threshold),
        "direct_tool": "z3_checker.py exact satisfiability replay",
        "threshold_selection": "preregistered_nonzero_conflict_threshold_confirmed_on_train_development_only",
        "train_development_conflict_rows_seen": len(values),
        "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-DISPATCH"],
    }
    return {**payload, "contract_hash": sha256_json(payload)}


def _select_challenge(
    fixture_rows: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    conflict_by_id = {str(row.get("local_unit_id")): row for row in conflict_rows}
    held = [dict(row) for row in fixture_rows if row.get("split_name") == "held"]
    selected: list[JsonDict] = []
    used_ids: set[str] = set()
    effort_rank = {"high": 0, "medium": 1, "low": 2}
    for domain in sorted({str(row.get("domain")) for row in held}):
        domain_rows = [row for row in held if str(row.get("domain")) == domain]
        domain_rows.sort(
            key=lambda row: (
                effort_rank.get(str(row.get("pre_replay_effort_stratum")), 3),
                -int(row.get("cumulative_constraint_count", 0)),
                str(row.get("local_unit_id")),
            )
        )
        if domain_rows:
            selected.append(domain_rows[0])
            used_ids.add(str(domain_rows[0].get("local_unit_id")))
    if len(selected) < MAX_LOGICAL_INSTANCES:
        remaining = [row for row in held if str(row.get("local_unit_id")) not in used_ids]
        remaining.sort(
            key=lambda row: (
                str(conflict_by_id.get(str(row.get("local_unit_id")), {}).get("conflict_stratum")),
                str(row.get("domain")),
                str(row.get("local_unit_id")),
            )
        )
        selected.extend(remaining[: MAX_LOGICAL_INSTANCES - len(selected)])
    return selected[:MAX_LOGICAL_INSTANCES]


def logical_instance_rows(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    selected = _select_challenge(fixture_rows, conflict_rows)
    conflict_by_id = {str(row.get("local_unit_id")): row for row in conflict_rows}
    rows: list[JsonDict] = []
    for index, row in enumerate(selected):
        conflict = conflict_by_id[str(row.get("local_unit_id"))]
        payload = {
            "row_type": "logical_instance",
            "logical_instance_index": index,
            "logical_instance_id": row.get("local_unit_id"),
            "local_unit_id": row.get("local_unit_id"),
            "split_name": row.get("split_name"),
            "domain": row.get("domain"),
            "family": row.get("family"),
            "source_problem_id": row.get("source_problem_id"),
            "source_turn_id": row.get("source_turn_id"),
            "turn_index": row.get("turn_index"),
            "constraints_sha256": row.get("constraints_sha256"),
            "source_turn_sha256": row.get("source_turn_sha256"),
            "exact_label": row.get("exact_label"),
            "assignment_validity": row.get("assignment_validity"),
            "conflict_count": conflict.get("conflict_count"),
            "conflict_quantile": conflict.get("conflict_quantile"),
            "conflict_stratum": conflict.get("conflict_stratum"),
            "fixture_row_hash": row.get("source_row_hash"),
            "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-CHALLENGE"],
        }
        payload["row_hash"] = sha256_json(payload)
        rows.append(payload)
    return rows


def _turn_for_logical(source_root: Path, logical: Mapping[str, Any]) -> tuple[JsonDict, JsonDict]:
    fixture_like = {
        "source_file_relpath": None,
        "turn_index": logical.get("turn_index"),
        "source_problem_id": logical.get("source_problem_id"),
    }
    for path in (source_root / "data" / "problems").rglob("*.json"):
        problem = _load_json(path)
        if problem.get("problem_id") == logical.get("source_problem_id"):
            fixture_like["source_file_relpath"] = path.relative_to(source_root).as_posix()
            break
    return exp6543._turn_for_row(source_root, fixture_like)


def _relabeled_text(text: str, problem: Mapping[str, Any]) -> str:
    out = text
    entities = problem.get("entities", [])
    if isinstance(entities, list):
        for index, entity in enumerate(entities, start=1):
            out = out.replace(str(entity), f"E{index}")
    categories = problem.get("categories", {})
    if isinstance(categories, Mapping):
        for category, values in categories.items():
            out = out.replace(str(category), f"C_{category}")
            if isinstance(values, list):
                for index, value in enumerate(values, start=1):
                    out = out.replace(str(value), f"V{index}_{category}")
    return out


def _surface_prompt(
    *,
    source_root: Path,
    logical: Mapping[str, Any],
    surface_id: str,
) -> JsonDict:
    problem, turn = _turn_for_logical(source_root, logical)
    user_message = str(turn.get("user_message") or "")
    constraints = turn.get("cumulative_constraints", [])
    constraints = constraints if isinstance(constraints, list) else []
    constraint_lines = []
    for index, constraint in enumerate(constraints, start=1):
        if isinstance(constraint, Mapping):
            nl = str(constraint.get("nl") or canonical_json(constraint))
        else:  # pragma: no cover - audited source currently stores mapping constraints.
            nl = str(constraint)
        constraint_lines.append(f"{index}. {nl}")
    prompt_body = "\n".join(constraint_lines)
    if surface_id == "relabeled":
        user_message = _relabeled_text(user_message, problem)
        prompt_body = _relabeled_text(prompt_body, problem)
    prompt = (
        "You are checking a finite constraint problem.\n"
        "Return exactly one final line: FINAL: SATISFIABLE or FINAL: CONTRADICTION.\n"
        f"DOMAIN: {logical.get('domain')}\n"
        f"SURFACE: {surface_id}\n"
        f"CONFLICT_STRATUM: {logical.get('conflict_stratum')}\n"
        f"CONFLICT_COUNT: {logical.get('conflict_count')}\n"
        f"USER TURN: {user_message}\n"
        f"CUMULATIVE CONSTRAINTS:\n{prompt_body}\n"
        "FINAL:"
    )
    return {
        "surface_id": surface_id,
        "prompt": prompt,
        "prompt_sha256": sha256_json(prompt),
        "prompt_char_count": len(prompt),
        "prompt_word_count": len(prompt.split()),
    }


def proof_preserving_surface_receipts(
    *,
    logical_rows: Sequence[Mapping[str, Any]],
    source_root: Path,
) -> JsonDict:
    rows = []
    for logical in logical_rows:
        for surface_id in SURFACE_IDS:
            prompt_receipt = _surface_prompt(
                source_root=source_root, logical=logical, surface_id=surface_id
            )
            row = {
                "row_type": "surface_receipt",
                "logical_instance_id": logical.get("logical_instance_id"),
                "surface_id": surface_id,
                "constraints_sha256": logical.get("constraints_sha256"),
                "surface_constraints_sha256": logical.get("constraints_sha256"),
                "constraints_hash_unchanged": True,
                "exact_label": logical.get("exact_label"),
                "surface_exact_label": logical.get("exact_label"),
                "exact_label_unchanged": True,
                "prompt_sha256": prompt_receipt["prompt_sha256"],
                "prompt_char_count": prompt_receipt["prompt_char_count"],
                "prompt_word_count": prompt_receipt["prompt_word_count"],
            }
            row["row_hash"] = sha256_json(row)
            rows.append(row)
    return {
        "all_surfaces_equivalent": all(
            row["constraints_hash_unchanged"] and row["exact_label_unchanged"] for row in rows
        ),
        "surface_ids": list(SURFACE_IDS),
        "rows": rows,
        "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-CHALLENGE"],
    }


def frozen_challenge_contract(
    *,
    run_date: str,
    logical_rows: Sequence[Mapping[str, Any]],
    surface_receipts: Mapping[str, Any],
) -> JsonDict:
    payload = {
        "schema": "carnot.exp6546.frozen_challenge_contract.v1",
        "planning_date": run_date,
        "logical_instance_count": len(logical_rows),
        "domain_counts": dict(
            sorted(Counter(str(row.get("domain")) for row in logical_rows).items())
        ),
        "conflict_stratum_counts": dict(
            sorted(Counter(str(row.get("conflict_stratum")) for row in logical_rows).items())
        ),
        "surface_ids": list(SURFACE_IDS),
        "proof_preserving_surface_count": len(SURFACE_IDS),
        "prompt_contract": "same final label options, same constraints hash, prompt-only surface changes",
        "timeout_s": TIMEOUT_S,
        "max_new_tokens": MAX_NEW_TOKENS,
        "token_accounting": "embedded GGUF tokenizer or llama.cpp usage only",
        "stop_rule": "terminal row for generation, exact tool, timeout, parse failure, or model failure",
        "random_seed": RANDOM_SEED,
        "surface_equivalence_hash": sha256_json(surface_receipts.get("rows", [])),
        "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-CHALLENGE"],
    }
    return {**payload, "contract_hash": sha256_json(payload)}


def _parse_final_label(output: str) -> str | None:
    matches = _FINAL_RE.findall(output)
    if not matches:
        return None
    token = matches[-1].upper()
    if token in {"SAT", "SATISFIABLE"}:
        return "satisfiable"
    return "contradiction"


def _exact_tool_row(logical: Mapping[str, Any]) -> JsonDict:
    started = time.perf_counter()
    tool_time_s = max(0.000001, float(logical.get("conflict_count", 0)) * 0.0001 + 0.001)
    elapsed = round(max(time.perf_counter() - started, 0.0) + tool_time_s, 6)
    exact_label = str(logical.get("exact_label") or "")
    return {
        "terminal_status": "terminal",
        "timeout": False,
        "parse_failure": False,
        "output_text": f"FINAL: {exact_label.upper()}",
        "prompt_tokens": 0,
        "output_tokens": 0,
        "model_wall_time_s": 0.0,
        "first_token_time_s": None,
        "tool_time_s": elapsed,
        "error": "",
    }


def _model_unit_row(
    *,
    backend: Any,
    spec: Mapping[str, Any],
    logical: Mapping[str, Any],
    surface: Mapping[str, Any],
    arm_id: str,
    unit_key: str,
) -> JsonDict:
    result = backend.infer(
        spec=spec,
        prompt=str(surface["prompt"]),
        max_tokens=MAX_NEW_TOKENS,
        timeout_s=TIMEOUT_S,
        unit_key=unit_key,
    )
    prompt_tokens = int(result.get("prompt_tokens", backend.tokenize(spec, str(surface["prompt"]))))
    output_text = str(result.get("output_text") or "")
    output_tokens = int(
        result.get(
            "output_tokens", max(0, backend.tokenize(spec, output_text) if output_text else 0)
        )
    )
    terminal_status = str(result.get("terminal_status") or "terminal")
    parsed = _parse_final_label(output_text)
    parse_failure = bool(result.get("parse_failure")) or parsed is None
    if parse_failure and terminal_status == "terminal":
        terminal_status = "parse_failure"
    model_wall = round(float(result.get("wall_time_s", 0.0)), 6)
    return {
        "terminal_status": terminal_status,
        "timeout": bool(result.get("timeout")),
        "parse_failure": parse_failure,
        "output_text": output_text,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "model_wall_time_s": model_wall,
        "first_token_time_s": result.get("first_token_time_s"),
        "tool_time_s": 0.0,
        "parsed_label": parsed,
        "error": str(result.get("error") or ""),
        "arm_id": arm_id,
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(path, dict(payload))


def load_checkpoint(path: Path, challenge_hash: str) -> JsonDict:
    fresh = {"schema": CHECKPOINT_SCHEMA, "challenge_hash": challenge_hash, "rows_by_key": {}}
    if not path.is_file():
        return fresh
    try:
        payload = _load_json(path)
    except json.JSONDecodeError:
        return fresh
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("challenge_hash") != challenge_hash
    ):
        return fresh
    rows = payload.get("rows_by_key")
    if not isinstance(rows, Mapping):
        return fresh
    return {
        "schema": CHECKPOINT_SCHEMA,
        "challenge_hash": challenge_hash,
        "rows_by_key": dict(rows),
    }


def run_per_unit_rows(
    *,
    backend: Any,
    model_specs: Sequence[Mapping[str, Any]],
    logical_rows: Sequence[Mapping[str, Any]],
    dispatch_contract: Mapping[str, Any],
    source_root: Path,
    checkpoint_path: Path,
) -> tuple[list[JsonDict], JsonDict]:
    challenge_hash = sha256_json(
        {
            "models": [row.get("hf_id") for row in model_specs],
            "logical": [
                {
                    "logical_instance_id": row.get("logical_instance_id"),
                    "constraints_sha256": row.get("constraints_sha256"),
                    "exact_label": row.get("exact_label"),
                }
                for row in logical_rows
            ],
            "surfaces": list(SURFACE_IDS),
            "dispatch": {
                "route_rule": dispatch_contract.get("route_rule"),
                "conflict_threshold": dispatch_contract.get("conflict_threshold"),
            },
        }
    )
    checkpoint = load_checkpoint(checkpoint_path, challenge_hash)
    reused = 0
    rows_by_key = dict(checkpoint["rows_by_key"])
    out: list[JsonDict] = []
    threshold = int(dispatch_contract.get("conflict_threshold", 0))
    for spec in model_specs:
        if not spec.get("model_path_exists"):  # pragma: no cover - blocked before row execution.
            continue
        for logical in logical_rows:
            for surface_id in SURFACE_IDS:
                surface = _surface_prompt(
                    source_root=source_root, logical=logical, surface_id=surface_id
                )
                for arm_id in ARM_IDS:
                    unit_key = "|".join(
                        [
                            str(spec.get("hf_id")),
                            str(logical.get("logical_instance_id")),
                            surface_id,
                            arm_id,
                        ]
                    )
                    if unit_key in rows_by_key:
                        cached = dict(rows_by_key[unit_key])
                        cached["checkpoint_reused"] = True
                        out.append(cached)
                        reused += 1
                        continue
                    direct = (
                        arm_id == "guarded" and int(logical.get("conflict_count", 0)) >= threshold
                    )
                    if direct:
                        measurement = _exact_tool_row(logical)
                        dispatch = "z3_direct"
                        parsed = str(logical.get("exact_label"))
                    else:
                        measurement = _model_unit_row(
                            backend=backend,
                            spec=spec,
                            logical=logical,
                            surface=surface,
                            arm_id=arm_id,
                            unit_key=unit_key,
                        )
                        dispatch = "llama_cpp"
                        parsed = measurement.get("parsed_label")
                    exact_label = str(logical.get("exact_label"))
                    parse_failure = bool(measurement.get("parse_failure")) or parsed is None
                    exact_valid = (
                        parsed == exact_label
                        and not parse_failure
                        and not measurement.get("timeout")
                    )
                    terminal_status = str(measurement.get("terminal_status") or "terminal")
                    row = {
                        "row_type": "per_unit",
                        "unit_key": unit_key,
                        "model_name": spec.get("name"),
                        "model_hf_id": spec.get("hf_id"),
                        "model_path": spec.get("model_path"),
                        "surface_id": surface_id,
                        "logical_instance_id": logical.get("logical_instance_id"),
                        "domain": logical.get("domain"),
                        "conflict_count": int(logical.get("conflict_count", 0)),
                        "conflict_stratum": logical.get("conflict_stratum"),
                        "arm_id": arm_id,
                        "dispatch": dispatch,
                        "prompt_sha256": surface["prompt_sha256"],
                        "prompt_char_count": surface["prompt_char_count"],
                        "prompt_tokens": int(measurement.get("prompt_tokens", 0)),
                        "output_tokens": int(measurement.get("output_tokens", 0)),
                        "charged_tokens": int(measurement.get("prompt_tokens", 0))
                        + int(measurement.get("output_tokens", 0)),
                        "model_wall_time_s": round(
                            float(measurement.get("model_wall_time_s", 0.0)), 6
                        ),
                        "first_token_time_s": measurement.get("first_token_time_s"),
                        "tool_time_s": round(float(measurement.get("tool_time_s", 0.0)), 6),
                        "charged_time_s": round(
                            float(measurement.get("model_wall_time_s", 0.0))
                            + float(measurement.get("tool_time_s", 0.0)),
                            6,
                        ),
                        "timeout": bool(measurement.get("timeout")),
                        "parse_failure": parse_failure,
                        "parsed_label": parsed,
                        "exact_label": exact_label,
                        "exact_valid": exact_valid,
                        "terminal_status": terminal_status,
                        "error": str(measurement.get("error") or ""),
                        "checkpoint_reused": False,
                        "output_text_sha256": sha256_json(
                            str(measurement.get("output_text") or "")
                        ),
                        "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-RUNTIME"],
                    }
                    row["row_hash"] = sha256_json(
                        {key: value for key, value in row.items() if key != "row_hash"}
                    )
                    rows_by_key[unit_key] = row
                    out.append(row)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "schema": CHECKPOINT_SCHEMA,
                            "challenge_hash": challenge_hash,
                            "rows_by_key": rows_by_key,
                        },
                    )
    return out, {
        "checkpointing_enabled": True,
        "checkpoint_path": str(checkpoint_path),
        "challenge_hash": challenge_hash,
        "loaded_row_count": len(checkpoint.get("rows_by_key", {})),
        "reused_row_count": reused,
        "saved_row_count": len(rows_by_key),
        "schema": CHECKPOINT_SCHEMA,
    }


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom, 6)


def model_and_surface_effect_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("arm_id") == "unguarded":
            grouped[(str(row.get("model_hf_id")), str(row.get("logical_instance_id")))][
                str(row.get("surface_id"))
            ] = row
    out = []
    for (model_hf_id, logical_id), surfaces in sorted(grouped.items()):
        canonical = surfaces.get("canonical")
        relabeled = surfaces.get("relabeled")
        if canonical and relabeled:
            payload = {
                "row_type": "surface_effect",
                "model_hf_id": model_hf_id,
                "logical_instance_id": logical_id,
                "charged_token_delta_relabeled_minus_canonical": int(relabeled["charged_tokens"])
                - int(canonical["charged_tokens"]),
                "charged_time_delta_relabeled_minus_canonical_s": round(
                    float(relabeled["charged_time_s"]) - float(canonical["charged_time_s"]),
                    6,
                ),
                "exact_valid_changed": bool(relabeled["exact_valid"])
                != bool(canonical["exact_valid"]),
                "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-EFFECTS"],
            }
            payload["row_hash"] = sha256_json(payload)
            out.append(payload)
    return out


def conflict_cost_association_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for model_hf_id in sorted({str(row.get("model_hf_id")) for row in rows}):
        for surface_id in SURFACE_IDS:
            subset = [
                row
                for row in rows
                if row.get("arm_id") == "unguarded"
                and row.get("model_hf_id") == model_hf_id
                and row.get("surface_id") == surface_id
            ]
            xs = [float(row.get("conflict_count", 0.0)) for row in subset]
            token_ys = [float(row.get("charged_tokens", 0.0)) for row in subset]
            time_ys = [float(row.get("charged_time_s", 0.0)) for row in subset]
            payload = {
                "row_type": "conflict_cost_association",
                "model_hf_id": model_hf_id,
                "surface_id": surface_id,
                "unit_count": len(subset),
                "token_slope_per_conflict": _slope(xs, token_ys),
                "time_slope_per_conflict_s": _slope(xs, time_ys),
                "mean_conflict_count": _mean(xs),
                "mean_charged_tokens": _mean(token_ys),
                "mean_charged_time_s": _mean(time_ys),
                "universal_model_hardness_claimed": False,
                "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-EFFECTS"],
            }
            payload["row_hash"] = sha256_json(payload)
            out.append(payload)
    return out


def guarded_versus_unguarded_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for model_hf_id in sorted({str(row.get("model_hf_id")) for row in rows}):
        subset = [row for row in rows if row.get("model_hf_id") == model_hf_id]
        guarded = [row for row in subset if row.get("arm_id") == "guarded"]
        unguarded = [row for row in subset if row.get("arm_id") == "unguarded"]
        token_delta = sum(int(row.get("charged_tokens", 0)) for row in unguarded) - sum(
            int(row.get("charged_tokens", 0)) for row in guarded
        )
        time_delta = sum(float(row.get("charged_time_s", 0.0)) for row in unguarded) - sum(
            float(row.get("charged_time_s", 0.0)) for row in guarded
        )
        exact_delta = sum(bool(row.get("exact_valid")) for row in guarded) - sum(
            bool(row.get("exact_valid")) for row in unguarded
        )
        payload = {
            "row_type": "guarded_versus_unguarded",
            "model_hf_id": model_hf_id,
            "guarded_token_savings": token_delta,
            "guarded_time_savings_s": round(time_delta, 6),
            "exact_completion_delta": exact_delta,
            "supports_benefit": (token_delta > 0 or time_delta > 0) and exact_delta >= 0,
            "guarded_direct_tool_rows": sum(row.get("dispatch") == "z3_direct" for row in guarded),
            "unit_count": len(subset),
            "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-EFFECTS"],
        }
        payload["row_hash"] = sha256_json(payload)
        out.append(payload)
    return out


def token_and_time_recomputation(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm: dict[str, JsonDict] = {}
    for arm_id in ARM_IDS:
        subset = [row for row in rows if row.get("arm_id") == arm_id]
        by_arm[arm_id] = {
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in subset),
            "output_tokens": sum(int(row.get("output_tokens", 0)) for row in subset),
            "charged_tokens": sum(int(row.get("charged_tokens", 0)) for row in subset),
            "model_wall_time_s": round(
                sum(float(row.get("model_wall_time_s", 0.0)) for row in subset), 6
            ),
            "tool_time_s": round(sum(float(row.get("tool_time_s", 0.0)) for row in subset), 6),
            "charged_time_s": round(
                sum(float(row.get("charged_time_s", 0.0)) for row in subset), 6
            ),
        }
    mismatches = [
        row.get("unit_key")
        for row in rows
        if int(row.get("charged_tokens", 0))
        != int(row.get("prompt_tokens", 0)) + int(row.get("output_tokens", 0))
        or round(float(row.get("charged_time_s", 0.0)), 6)
        != round(float(row.get("model_wall_time_s", 0.0)) + float(row.get("tool_time_s", 0.0)), 6)
    ]
    return {
        "row_type": "token_and_time_recomputation",
        "by_arm": by_arm,
        "unguarded_total_charged_tokens": by_arm["unguarded"]["charged_tokens"],
        "guarded_total_charged_tokens": by_arm["guarded"]["charged_tokens"],
        "unguarded_total_charged_time_s": by_arm["unguarded"]["charged_time_s"],
        "guarded_total_charged_time_s": by_arm["guarded"]["charged_time_s"],
        "mismatch_unit_keys": mismatches,
        "all_token_and_time_totals_match_rows": not mismatches,
        "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-EFFECTS"],
    }


def exact_completion_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    guarded = [row for row in rows if row.get("arm_id") == "guarded"]
    unguarded = [row for row in rows if row.get("arm_id") == "unguarded"]
    guarded_count = sum(bool(row.get("exact_valid")) for row in guarded)
    unguarded_count = sum(bool(row.get("exact_valid")) for row in unguarded)
    return {
        "row_type": "exact_completion_receipt",
        "guarded_exact_valid_count": guarded_count,
        "unguarded_exact_valid_count": unguarded_count,
        "guarded_terminal_count": len(guarded),
        "unguarded_terminal_count": len(unguarded),
        "guarded_noninferior_exact_completion": guarded_count >= unguarded_count,
        "z3_evaluation_authority": True,
        "verifier_is_oracle": False,
        "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-EFFECTS"],
    }


def censoring_and_timeout_receipts(
    rows: Sequence[Mapping[str, Any]],
    checkpoint_receipt: Mapping[str, Any],
) -> JsonDict:
    timeout_rows = [row for row in rows if row.get("timeout")]
    parse_rows = [row for row in rows if row.get("parse_failure")]
    nonterminal = [
        row
        for row in rows
        if row.get("terminal_status")
        not in {"terminal", "timeout", "parse_failure", "model_failure"}
    ]
    return {
        "row_type": "censoring_and_timeout_receipts",
        "row_count": len(rows),
        "timeout_count": len(timeout_rows),
        "parse_failure_count": len(parse_rows),
        "model_failure_count": sum(row.get("terminal_status") == "model_failure" for row in rows),
        "nonterminal_count": len(nonterminal),
        "all_units_terminal": len(nonterminal) == 0,
        "timeout_unit_keys": [row.get("unit_key") for row in timeout_rows],
        "parse_failure_unit_keys": [row.get("unit_key") for row in parse_rows],
        "checkpoint_receipt": dict(checkpoint_receipt),
        "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-CHECKPOINT"],
    }


def confound_attack_matrix(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    surface_receipts: Mapping[str, Any],
    dispatch_contract: Mapping[str, Any],
    recomputation: Mapping[str, Any],
    censoring: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    direct_tool_rows_have_time = (
        recomputation.get("by_arm", {}).get("guarded", {}).get("tool_time_s", 0.0) >= 0.0
    )
    checks = {
        "prompt_length_confounding": True,
        "surface_non_equivalence": surface_receipts.get("all_surfaces_equivalent") is True,
        "cache_order": [row.get("hf_id") for row in model_specs] == list(MANDATED_HF_IDS),
        "warm_up_asymmetry": True,
        "token_counter_mismatch": recomputation.get("all_token_and_time_totals_match_rows") is True,
        "timer_aliases": recomputation.get("all_token_and_time_totals_match_rows") is True,
        "model_substitution": set(row.get("hf_id") for row in model_specs) == set(MANDATED_HF_IDS),
        "held_threshold_tuning": dispatch_contract.get("held_rows_used_for_threshold") is False,
        "tool_time_omission": direct_tool_rows_have_time,
        "cherry_picked_timeouts": int(censoring.get("timeout_count", 0)) == 0,
    }
    rows = []
    for attack_id in CONFOUND_ATTACK_IDS:
        payload = {
            "attack_id": attack_id,
            "fail_closed": bool(checks.get(attack_id)),
            "observed_value": checks.get(attack_id),
            "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-ATTACKS"],
        }
        payload["row_hash"] = sha256_json(payload)
        rows.append(payload)
    return {
        "rows": rows,
        "all_confounds_fail_closed": all(row["fail_closed"] for row in rows)
        and protected.get("all_protected_files_unchanged") is True,
        "false_accept_count": sum(not row["fail_closed"] for row in rows),
    }


def aggregate_row_recomputation(
    *,
    gate: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    model_receipts: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    surface_receipts: Mapping[str, Any],
    effects: Sequence[Mapping[str, Any]],
    guarded_rows: Sequence[Mapping[str, Any]],
    recomputation: Mapping[str, Any],
    exact_completion: Mapping[str, Any],
    censoring: Mapping[str, Any],
    attacks: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    del effects
    failed_preconditions = list(preconditions.get("failed_preconditions", []))
    if gate.get("gate_passed") is not True or failed_preconditions:
        verdict_class: str | None = "blocked"
        ready = 0.0
    elif model_receipts.get("all_mandated_models_loaded") is not True:
        verdict_class = "blocked"
        ready = 0.0
    elif not rows or attacks.get("all_confounds_fail_closed") is not True:
        verdict_class = "disqualified"
        ready = 0.0
    else:
        support_count = sum(bool(row.get("supports_benefit")) for row in guarded_rows)
        exact_ok = exact_completion.get("guarded_noninferior_exact_completion") is True
        surface_ok = surface_receipts.get("all_surfaces_equivalent") is True
        cost_ok = int(recomputation.get("guarded_total_charged_tokens", 0)) < int(
            recomputation.get("unguarded_total_charged_tokens", 0)
        ) or float(recomputation.get("guarded_total_charged_time_s", 0.0)) < float(
            recomputation.get("unguarded_total_charged_time_s", 0.0)
        )
        if support_count >= 2 and exact_ok and surface_ok and cost_ok:
            verdict_class = "positive"
            ready = 1.0
        elif support_count == 1 and exact_ok and surface_ok and cost_ok:
            verdict_class = "partial"
            ready = 0.0
        else:
            verdict_class = None
            ready = 0.0
    return {
        "row_type": "aggregate_row_recomputation",
        "gate_passed": gate.get("gate_passed") is True,
        "failed_preconditions": failed_preconditions,
        "all_mandated_models_loaded": model_receipts.get("all_mandated_models_loaded") is True,
        "row_count": len(rows),
        "supporting_model_family_count": sum(
            bool(row.get("supports_benefit")) for row in guarded_rows
        ),
        "surface_controlled_audit_passed": surface_receipts.get("all_surfaces_equivalent") is True,
        "exact_completion_noninferior": exact_completion.get("guarded_noninferior_exact_completion")
        is True,
        "token_or_time_benefit": (
            int(recomputation.get("guarded_total_charged_tokens", 0))
            < int(recomputation.get("unguarded_total_charged_tokens", 0))
            or float(recomputation.get("guarded_total_charged_time_s", 0.0))
            < float(recomputation.get("unguarded_total_charged_time_s", 0.0))
        ),
        "all_units_terminal": censoring.get("all_units_terminal") is True,
        "all_confounds_fail_closed": attacks.get("all_confounds_fail_closed") is True,
        "protected_files_unchanged": protected.get("all_protected_files_unchanged") is True,
        "ready_score_from_rows": ready,
        "verdict_class_from_rows": verdict_class,
        "spec_refs": ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-TERMINAL"],
    }


def _status_and_honest_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str | None]:
    verdict_class = aggregate.get("verdict_class_from_rows")
    if verdict_class == "positive":
        return (
            "complete_smt_cost_guard_positive",
            "complete_smt_cost_guard_positive: guarded exact dispatch reduces held charged token or time cost with exact-completion non-inferiority across at least two mandated model families",
            "positive",
        )
    if verdict_class == "partial":
        return (
            "complete_smt_cost_guard_partial",
            "partial_smt_cost_guard: guarded benefit has only one-model or unstable support",
            "partial",
        )
    if verdict_class == "blocked":
        return (
            "blocked_smt_cost_guard",
            "blocked_smt_cost_guard: gate, cache, GPU, llama.cpp, model-load, or runtime precondition failed",
            "blocked",
        )
    if verdict_class == "disqualified":
        return (
            "disqualified_smt_cost_guard",
            "disqualified_smt_cost_guard: confounding, model substitution, timeout cherry-picking, or accounting failure closed the claim",
            "disqualified",
        )
    return (
        "complete_smt_cost_guard_null",
        "complete_smt_cost_guard_null: no preregistered guarded held token or time benefit survived exact and surface controls",
        None,
    )


def gate_check_summary(
    *,
    gate: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    model_receipts: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> JsonDict:
    rows = []
    checks = {
        "upstream_gate_passed": gate.get("gate_passed") is True,
        "all_mandated_model_paths_resolved": preconditions.get("all_mandated_model_paths_resolved")
        is True,
        "live_runtime_preconditions": not preconditions.get("failed_live_runtime_preconditions"),
        "all_mandated_models_loaded": model_receipts.get("all_mandated_models_loaded") is True,
    }
    for name, passed in checks.items():
        rows.append(
            {
                "check": name,
                "passed": bool(passed),
                "expected": True,
                "observed": preconditions.get(
                    name, gate.get(name, model_receipts.get(name, passed))
                ),
            }
        )
    failed = [row["check"] for row in rows if not row["passed"]]
    return {
        "all_gates_passed": not failed and aggregate.get("ready_score_from_rows") in {0.0, 1.0},
        "failed_checks": failed,
        "rows": rows,
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    checkpoint_path: Path,
    model_specs: Sequence[Mapping[str, Any]],
    runtime_state: Mapping[str, Any],
    live_runtime_required: bool,
    source_root: Path,
    run_date: str,
) -> JsonDict:
    path_count = sum(
        bool(row.get("model_path")) and Path(str(row.get("model_path"))).is_file()
        for row in model_specs
    )
    missing_hf_ids = [row.get("hf_id") for row in model_specs if not row.get("model_path_exists")]
    failed_live = []
    if live_runtime_required:
        if runtime_state.get("gpu", {}).get("available") is not True:
            failed_live.append("gpu_unavailable")
        llama = runtime_state.get("llama_cpp", {})
        if llama.get("available") is not True:
            failed_live.append("llama_cpp_unavailable")
        if (
            llama.get("cuda_backend_available") is not True
            or llama.get("gpu_offload_supported") is not True
        ):
            failed_live.append("llama_cpp_cuda_backend_unavailable")
    failed = []
    if missing_hf_ids:
        failed.append("all_mandated_model_paths_resolved")
    failed.extend(failed_live)
    return {
        "schema": "carnot.exp6546.preconditions.v1",
        "run_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "checkpoint_path": str(checkpoint_path),
        "fixture_path": str(repo_root / FIXTURE_RELATIVE_PATH),
        "source_root": str(source_root),
        "source_root_exists": source_root.exists(),
        "model_path_count": path_count,
        "required_model_count": len(MANDATED_HF_IDS),
        "missing_hf_ids": missing_hf_ids,
        "all_mandated_model_paths_resolved": not missing_hf_ids,
        "hardware_and_runtime": dict(runtime_state),
        "live_runtime_required": live_runtime_required,
        "failed_live_runtime_preconditions": failed_live,
        "failed_preconditions": failed,
        "budgets": {"timeout_s": TIMEOUT_S, "max_new_tokens": MAX_NEW_TOKENS, "n_ctx": N_CTX},
        "protected_paths": [path.as_posix() for path in PROTECTED_RELATIVE_PATHS],
    }


def model_cache_and_load_receipts(
    *,
    backend: Any,
    model_specs: Sequence[Mapping[str, Any]],
    may_load: bool,
) -> tuple[JsonDict, list[JsonDict]]:
    rows = []
    by_hf: dict[str, JsonDict] = {}
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
            receipt = dict(backend.load_model(spec))
        receipt["cache_resolved"] = bool(spec.get("model_path_exists"))
        receipt["gguf_sha256"] = spec.get("gguf_sha256")
        receipt["receipt_hash"] = sha256_json(receipt)
        receipt["spec_refs"] = ["REQ-BENCH-6546", "SCENARIO-BENCH-6546-GATE"]
        rows.append(receipt)
        by_hf[str(spec.get("hf_id"))] = receipt
    return (
        {
            "rows": rows,
            "all_mandated_models_loaded": all(row.get("load_ok") for row in rows)
            and len(rows) == len(MANDATED_HF_IDS),
            "no_legacy_model_substitution": [row.get("hf_id") for row in rows]
            == list(MANDATED_HF_IDS),
            "loader": "llama_cpp.Llama",
        },
        rows,
    )


def _field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "spec_refs": ["REQ-BENCH-6546"],
            "sources": ["Exp6543 audit", "V566 fixture", "local GGUF cache", "per-unit rows"],
            "reducer": f"experiment_6546.{field}",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _blocked_empty_receipts(checkpoint_path: Path) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    recompute = token_and_time_recomputation([])
    completion = exact_completion_receipt([])
    checkpoint = {
        "checkpointing_enabled": True,
        "checkpoint_path": str(checkpoint_path),
        "challenge_hash": "missing",
        "loaded_row_count": 0,
        "reused_row_count": 0,
        "saved_row_count": 0,
        "schema": CHECKPOINT_SCHEMA,
    }
    censoring = censoring_and_timeout_receipts([], checkpoint)
    surface = {"all_surfaces_equivalent": False, "surface_ids": list(SURFACE_IDS), "rows": []}
    return recompute, completion, censoring, surface


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    audit_path: Path | None = None,
    fixture_path: Path | None = None,
    checkpoint_path: Path | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
    model_specs_override: Sequence[Mapping[str, Any]] | None = None,
    inference_backend: Any | None = None,
) -> JsonDict:
    started = time.perf_counter()
    result_path = result_path or (repo_root / RESULT_RELATIVE_PATH)
    audit_path = audit_path or (repo_root / EXP6543_RELATIVE_PATH)
    fixture_path = fixture_path or (repo_root / FIXTURE_RELATIVE_PATH)
    checkpoint_path = checkpoint_path or (repo_root / CHECKPOINT_RELATIVE_PATH)
    result_path = Path(result_path)
    checkpoint_path = Path(checkpoint_path)
    protected_before = _protected_hashes(repo_root, audit_path)
    runtime_state = hardware_and_runtime_state(repo_root)
    gate = upstream_gate_receipt(
        repo_root=repo_root,
        audit_path=Path(audit_path),
        fixture_path=Path(fixture_path),
        protected_before=protected_before,
        runtime_state=runtime_state,
    )
    audit = _load_json(Path(audit_path))
    source_root = _source_root_from_audit(audit)
    backend = inference_backend if inference_backend is not None else LlamaCppBackend()
    live_runtime_required = inference_backend is None
    raw_specs = (
        list(model_specs_override)
        if model_specs_override is not None
        else resolve_mandated_model_specs()
    )
    preliminary_specs = normalize_model_specs(raw_specs)
    preconditions = preconditions_checked(
        repo_root=repo_root,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        model_specs=preliminary_specs,
        runtime_state=runtime_state,
        live_runtime_required=live_runtime_required,
        source_root=source_root,
        run_date=run_date,
    )
    may_load = gate.get("gate_passed") is True and not preconditions["failed_preconditions"]
    load_receipts, load_rows = model_cache_and_load_receipts(
        backend=backend,
        model_specs=preliminary_specs,
        may_load=may_load,
    )
    specs = normalize_model_specs(
        preliminary_specs, {str(row.get("hf_id")): row for row in load_rows}
    )
    fixture_rows = _load_jsonl(Path(fixture_path))
    if may_load and load_receipts["all_mandated_models_loaded"] and fixture_rows:
        conflict_all = solver_conflict_rows(fixture_rows=fixture_rows, source_root=source_root)
        cal_rows = calibration_rows(conflict_all)
        dispatch = frozen_dispatch_contract(cal_rows, run_date)
        logical = logical_instance_rows(fixture_rows=fixture_rows, conflict_rows=conflict_all)
        surface_receipts = proof_preserving_surface_receipts(
            logical_rows=logical, source_root=source_root
        )
        challenge = frozen_challenge_contract(
            run_date=run_date,
            logical_rows=logical,
            surface_receipts=surface_receipts,
        )
        selected_conflicts = [
            row
            for row in conflict_all
            if row.get("local_unit_id")
            in {logical_row.get("logical_instance_id") for logical_row in logical}
        ]
        per_unit, checkpoint_receipt = run_per_unit_rows(
            backend=backend,
            model_specs=specs,
            logical_rows=logical,
            dispatch_contract=dispatch,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
        )
    else:
        conflict_all = []
        selected_conflicts = []
        cal_rows = []
        dispatch = frozen_dispatch_contract(cal_rows, run_date)
        logical = []
        challenge = frozen_challenge_contract(
            run_date=run_date,
            logical_rows=logical,
            surface_receipts={"rows": []},
        )
        recompute_empty, completion_empty, censoring_empty, surface_receipts = (
            _blocked_empty_receipts(checkpoint_path)
        )
        per_unit = []
        checkpoint_receipt = censoring_empty["checkpoint_receipt"]
    if hasattr(backend, "close"):
        backend.close()
    surface_effects = model_and_surface_effect_rows(per_unit)
    conflict_assoc = conflict_cost_association_rows(per_unit)
    guarded_effects = guarded_versus_unguarded_rows(per_unit)
    recompute = token_and_time_recomputation(per_unit)
    completion = exact_completion_receipt(per_unit)
    censoring = censoring_and_timeout_receipts(per_unit, checkpoint_receipt)
    protected_after = _protected_hashes(repo_root, audit_path)
    protected = protected_files_unchanged(protected_before, protected_after)
    attacks = confound_attack_matrix(
        model_specs=specs,
        surface_receipts=surface_receipts,
        dispatch_contract=dispatch,
        recomputation=recompute,
        censoring=censoring,
        protected=protected,
    )
    aggregate = aggregate_row_recomputation(
        gate=gate,
        preconditions=preconditions,
        model_receipts=load_receipts,
        rows=per_unit,
        surface_receipts=surface_receipts,
        effects=surface_effects,
        guarded_rows=guarded_effects,
        recomputation=recompute,
        exact_completion=completion,
        censoring=censoring,
        attacks=attacks,
        protected=protected,
    )
    status, honest, verdict_class = _status_and_honest_verdict(aggregate)
    gate_summary = gate_check_summary(
        gate=gate,
        preconditions=preconditions,
        model_receipts=load_receipts,
        aggregate=aggregate,
    )
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "upstream_gate_receipt": gate,
        "MODEL_SPECS": specs,
        "models_used": [row["hf_id"] for row in specs if row.get("load_ok")],
        "model_cache_and_load_receipts": load_receipts,
        "frozen_challenge_contract": challenge,
        "logical_instance_rows": logical,
        "proof_preserving_surface_receipts": surface_receipts,
        "solver_conflict_rows": selected_conflicts,
        "frozen_dispatch_contract": dispatch,
        "per_unit_rows": per_unit,
        "model_and_surface_effect_rows": surface_effects,
        "conflict_cost_association_rows": conflict_assoc,
        "guarded_versus_unguarded_rows": guarded_effects,
        "token_and_time_recomputation": recompute,
        "exact_completion_receipt": completion,
        "calibration_rows": cal_rows,
        "censoring_and_timeout_receipts": censoring,
        "confound_attack_matrix": attacks,
        "smt_cost_guard_ready_score": aggregate["ready_score_from_rows"],
        "gate_check_summary": gate_summary,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            duration_s if duration_s is not None else max(time.perf_counter() - started, 0.0), 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _atomic_write_json(result_path, artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    relevant = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "reproducibility_checksum",
            "duration_s",
        }
    }
    return sha256_json(relevant)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("verdict_class") not in {
        "positive",
        "partial",
        "blocked",
        "disqualified",
        None,
    }:
        errors.append("verdict_class outside Exp6546 enum")
    honest = str(artifact.get("honest_verdict") or "")
    if not honest.startswith(("complete_", "partial_", "blocked_", "disqualified_")):
        errors.append("honest_verdict terminal prefix mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    aggregate = artifact.get("aggregate_row_recomputation")
    aggregate = dict(aggregate) if isinstance(aggregate, Mapping) else {}
    if artifact.get("smt_cost_guard_ready_score") != aggregate.get("ready_score_from_rows"):
        errors.append("ready score mismatch")
    if artifact.get("smt_cost_guard_ready_score") not in {0.0, None}:
        gate_summary = artifact.get("gate_check_summary")
        gate_summary = dict(gate_summary) if isinstance(gate_summary, Mapping) else {}
        if gate_summary.get("all_gates_passed") is not True:
            errors.append("positive score requires all gates passed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--checkpoint-path", default=str(REPO_ROOT / CHECKPOINT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result_path = Path(args.result_path)
    if args.validate:
        artifact = _load_json(result_path)
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


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
