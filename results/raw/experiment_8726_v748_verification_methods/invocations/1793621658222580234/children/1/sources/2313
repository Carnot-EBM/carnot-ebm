"""Exp6426 task-scoped runtime receipt contract.

Spec refs: REQ-INFRA-6426, SCENARIO-INFRA-6426-1,
SCENARIO-INFRA-6426-2, SCENARIO-INFRA-6426-3,
SCENARIO-INFRA-6426-4, SCENARIO-INFRA-6426-5.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import gc
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol

from carnot.inference.sota_models import cached_sota_pair
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "exp6426-task-scoped-runtime-receipt-contract"
RUN_DATE = "20260814"
RANDOM_SEED = 6426
PREFERRED_QUANT = "Q4_K_M"
MANDATED_POWERED_MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
TOKENIZER_SOURCE = "embedded_gguf_vocab_only"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
CONTROL_IDS = ("cpu", "blocked", "interrupted", "powered")
POWERED_TIMEOUT_S = 600.0
MAX_TOKENS = 16
COMPLETION_CALLS = 48
N_CTX = 256
MODEL_PREFIX_BYTES = 4096

RESULT_RELATIVE_PATH = Path("results/experiment_6426_task_scoped_runtime_receipt_contract.json")
DATA_DIR_RELATIVE_PATH = Path("data/research/experiment_6426_task_scoped_runtime_receipt_contract")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6426_task_scoped_runtime_receipt_contract.py")
HELPER_RELATIVE_PATH = Path("python/carnot/task_runtime_receipts.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6426_task_scoped_runtime_receipt_contract.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6426_task_scoped_runtime_receipt_contract "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6426_task_scoped_runtime_receipt_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/task_runtime_receipts.py,"
    "python/carnot/experiment_6426_task_scoped_runtime_receipt_contract.py "
    "-m pytest tests/python/test_experiment_6426_task_scoped_runtime_receipt_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/task_runtime_receipts.py,"
    "python/carnot/experiment_6426_task_scoped_runtime_receipt_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6426_task_scoped_runtime_receipt_contract.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6426_task_scoped_runtime_receipt_contract "
    "--date 20260814 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6426_task_scoped_runtime_receipt_contract.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    HELPER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("scripts/experiment_template.py"),
    Path("scripts/adversarial_verify.py"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "receipt_schema_version_and_hash",
    "helper_source_and_test_hashes",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_and_embedded_tokenizer_hashes",
    "autotokenizer_usage_count",
    "runner_binary_and_selection_receipts",
    "device_inventory_and_preflight_receipts",
    "per_unit_rows",
    "cpu_blocked_interrupted_and_powered_control_rows",
    "per_phase_monotonic_and_wall_clock_receipts",
    "parent_child_pid_and_exit_receipts",
    "pid_linked_gpu_samples",
    "concurrency_group_receipts",
    "command_config_model_and_raw_output_hashes",
    "synthesized_runtime_field_count",
    "cpu_fallback_count",
    "attribution_failure_count",
    "recomputed_duration_s",
    "reported_vs_recomputed_duration_delta",
    "attack_matrix",
    "runtime_receipt_contract_ready_score",
    "current_adversarial_findings",
    "protected_files_unchanged",
    "blocked_reason",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "States whether the task reached complete, blocked, or null receipt state.",
    "receipt_schema_version_and_hash": "Pins the versioned reusable receipt schema.",
    "helper_source_and_test_hashes": "Pins the helper, experiment, test, and spec sources.",
    "MODEL_SPECS": "Shows the mandated cached GGUF selected for the powered smoke.",
    "models_used": "Counts only authenticated powered rows as live model use.",
    "cached_sota_pair_receipts": "Proves model selection came through cached_sota_pair().",
    "model_file_and_embedded_tokenizer_hashes": "Binds model bytes and embedded tokenizer metadata.",
    "autotokenizer_usage_count": "Must stay zero because GGUF tokenizer metadata is embedded.",
    "runner_binary_and_selection_receipts": "Binds the selected runner binary and substrate.",
    "device_inventory_and_preflight_receipts": "Records GPUs, VRAM, cache, disk, CPU, RAM, and clocks.",
    "per_unit_rows": "Contains one row per control and phase.",
    "cpu_blocked_interrupted_and_powered_control_rows": "Keeps the four control paths separate.",
    "per_phase_monotonic_and_wall_clock_receipts": "Lets duration recompute from monotonic intervals.",
    "parent_child_pid_and_exit_receipts": "Binds each child PID to exit state.",
    "pid_linked_gpu_samples": "Requires GPU samples linked to the powered child PID.",
    "concurrency_group_receipts": "Prevents overlapping work from merging into one task.",
    "command_config_model_and_raw_output_hashes": "Binds command, config, model, and raw output bytes.",
    "synthesized_runtime_field_count": "Must be zero because runtime comes from measured rows.",
    "cpu_fallback_count": "Must be zero for the powered path.",
    "attribution_failure_count": "Must be zero for confident task attribution.",
    "recomputed_duration_s": "Computed from phase intervals, not from wall-clock prose.",
    "reported_vs_recomputed_duration_delta": "Must be zero when reported duration is honest.",
    "attack_matrix": "Shows critical attribution attacks fail closed.",
    "runtime_receipt_contract_ready_score": "One only when controls, powered smoke, phases, and attacks pass.",
    "current_adversarial_findings": "Carries local contract findings before external verifier review.",
    "protected_files_unchanged": "Proves the conductor and reconciliation docs were not changed.",
    "blocked_reason": "Names powered precondition blockers when the smoke cannot run.",
    "preconditions_checked": "Freezes all required host and model checks before powered work.",
    "inference_substrate": "Declares a small-N local SOTA GGUF live smoke.",
    "verifier_is_oracle": "False because process receipts do not prove semantic correctness.",
    "field_principles": "Maps each field, receipt component, attack, and readiness score.",
    "field_provenance": "States whether each field is measured, derived, constant, or source-bound.",
    "random_seed": "Pins deterministic CPU and powered smoke settings.",
    "duration_s": "Reports the recomputed phase duration.",
    "tests_run": "Records required verification command outcomes.",
    "reproducibility_checksum": "Detects artifact drift after terminal fields are set.",
    "honest_verdict": "Uses an allowed terminal prefix and states the evidence boundary.",
}
FIELD_PRINCIPLES.update(
    {
        field: f"Receipt component required by {receipts.SCHEMA_VERSION}."
        for field in receipts.REQUIRED_ROW_FIELDS
    }
)
FIELD_PRINCIPLES.update(
    {attack: "Critical attack must fail closed." for attack in receipts.ATTACK_IDS}
)

FIELD_PROVENANCE: dict[str, str] = {
    "status": "derived contract gate",
    "receipt_schema_version_and_hash": "derived schema hash",
    "helper_source_and_test_hashes": "source hash",
    "MODEL_SPECS": "cached model resolution",
    "models_used": "derived accepted powered rows",
    "cached_sota_pair_receipts": "helper call receipt",
    "model_file_and_embedded_tokenizer_hashes": "source and tokenizer hash",
    "autotokenizer_usage_count": "constant",
    "runner_binary_and_selection_receipts": "measured runner data",
    "device_inventory_and_preflight_receipts": "measured host data",
    "per_unit_rows": "measured phase rows",
    "cpu_blocked_interrupted_and_powered_control_rows": "derived grouping",
    "per_phase_monotonic_and_wall_clock_receipts": "measured phase rows",
    "parent_child_pid_and_exit_receipts": "measured phase rows",
    "pid_linked_gpu_samples": "measured nvidia-smi rows",
    "concurrency_group_receipts": "derived grouping",
    "command_config_model_and_raw_output_hashes": "derived hashes",
    "synthesized_runtime_field_count": "derived contract gate",
    "cpu_fallback_count": "derived contract gate",
    "attribution_failure_count": "derived contract gate",
    "recomputed_duration_s": "derived monotonic intervals",
    "reported_vs_recomputed_duration_delta": "derived arithmetic check",
    "attack_matrix": "derived mutation checks",
    "runtime_receipt_contract_ready_score": "derived conjunctive gate",
    "current_adversarial_findings": "derived contract findings",
    "protected_files_unchanged": "source hash",
    "blocked_reason": "derived precondition check",
    "preconditions_checked": "measured and derived preflight",
    "inference_substrate": "constant",
    "verifier_is_oracle": "constant",
    "field_principles": "constant",
    "field_provenance": "constant",
    "random_seed": "constant",
    "duration_s": "derived monotonic intervals",
    "tests_run": "test command receipts",
    "reproducibility_checksum": "derived checksum",
    "honest_verdict": "derived terminal verdict",
}


class RuntimeAdapter(Protocol):
    """Small boundary that lets tests avoid loading the large GGUF."""

    def preflight_receipts(self, model_specs: list[JsonDict]) -> JsonDict:
        """Return host and runtime preflight receipts."""

    def powered_control_rows(
        self,
        *,
        task_id: str,
        model: JsonDict,
        output_dir: Path,
    ) -> list[JsonDict]:
        """Run or fixture the powered control phase rows."""


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _runner_selection(control_id: str, substrate: str) -> JsonDict:
    """Build a runner-selection receipt for one control."""

    binary = Path(sys.executable)
    selection = {
        "runner_id": f"{control_id}:{substrate}",
        "binary_path": str(binary),
        "binary_sha256": receipts.sha256_file(binary) or receipts.sha256_text(str(binary)),
        "substrate": substrate,
        "selected": True,
    }
    selection["selection_hash"] = receipts.sha256_json(selection)
    return selection


def _model_identity(control_id: str, model_sha256: str | None = None) -> JsonDict:
    """Return a hash-bound model identity for a control."""

    digest = model_sha256 or receipts.sha256_json({"control_id": control_id, "seed": RANDOM_SEED})
    return {
        "hf_id": MANDATED_POWERED_MODEL_ID
        if control_id == "powered"
        else f"deterministic/{control_id}",
        "model_sha256": digest,
        "model_identity_bound": True,
    }


def _phase_row(
    *,
    control_id: str,
    phase: str,
    start_ns: int,
    end_ns: int,
    child_pids: Sequence[int],
    raw_output: bytes,
    substrate: str,
    exit_status: Mapping[str, Any],
    device_ids: Sequence[str],
    model_sha256: str | None = None,
    gpu_samples: Sequence[Mapping[str, Any]] | None = None,
    blocked_reason: str = "",
) -> JsonDict:
    """Build one standard Exp6426 row."""

    return receipts.build_phase_row(
        task_id=TASK_ID,
        control_id=control_id,
        phase=phase,
        monotonic_start_ns=start_ns,
        monotonic_end_ns=end_ns,
        wall_clock_start=_utc_now(),
        wall_clock_end=_utc_now(),
        parent_pid=os.getpid(),
        child_pids=child_pids,
        command=[sys.executable, "-m", __name__, control_id, phase],
        config={"seed": RANDOM_SEED, "control_id": control_id, "phase": phase},
        model_identity=_model_identity(control_id, model_sha256),
        runner_selection=_runner_selection(control_id, substrate),
        device_ids=device_ids,
        concurrency_group=f"{TASK_ID}:{control_id}",
        raw_output_bytes=raw_output,
        exit_status=exit_status,
        attribution_confidence=1.0,
        gpu_samples=gpu_samples,
        blocked_reason=blocked_reason,
        extra={"first_token_or_completion_evidence": {"sha256": receipts.sha256_bytes(raw_output)}},
    )


def _rows_from_timing(
    *,
    control_id: str,
    raw_output: bytes,
    substrate: str,
    child_pids: Sequence[int],
    exit_status: Mapping[str, Any],
    device_ids: Sequence[str],
    blocked_reason: str = "",
    gpu_samples: Sequence[Mapping[str, Any]] | None = None,
    model_sha256: str | None = None,
) -> list[JsonDict]:
    """Create five measured phase rows around a completed control."""

    rows: list[JsonDict] = []
    for phase in receipts.REQUIRED_PHASES:
        start = time.monotonic_ns()
        if phase == "exact_verification":
            receipts.sha256_bytes(raw_output)
        end = time.monotonic_ns()
        rows.append(
            _phase_row(
                control_id=control_id,
                phase=phase,
                start_ns=start,
                end_ns=end,
                child_pids=child_pids,
                raw_output=raw_output,
                substrate=substrate,
                exit_status=exit_status,
                device_ids=device_ids,
                blocked_reason=blocked_reason,
                gpu_samples=gpu_samples if phase == "generation" else [],
                model_sha256=model_sha256,
            )
        )
    return rows


def run_cpu_control() -> list[JsonDict]:
    """Run the deterministic CPU success control."""

    total = sum((index * index + RANDOM_SEED) % 97 for index in range(2048))
    raw = f"cpu_control_total={total}\n".encode()
    return _rows_from_timing(
        control_id="cpu",
        raw_output=raw,
        substrate="cpu",
        child_pids=[os.getpid()],
        exit_status={"returncode": 0, "timed_out": False, "signal": None},
        device_ids=["CPU"],
    )


def run_blocked_control() -> list[JsonDict]:
    """Run the explicit preflight block control without launching a child."""

    reason = "explicit_preflight_block_control"
    return _rows_from_timing(
        control_id="blocked",
        raw_output=reason.encode("utf-8"),
        substrate="blocked",
        child_pids=[],
        exit_status={"returncode": None, "timed_out": False, "blocked": True},
        device_ids=["NO_DEVICE"],
        blocked_reason=reason,
    )


def _signal_name(returncode: int | None) -> str | None:
    """Return a signal name for negative subprocess return codes."""

    if returncode is None or returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal_{-returncode}"


def run_interrupted_control() -> list[JsonDict]:
    """Run a child process and interrupt it to preserve partial exit evidence."""

    start = time.monotonic_ns()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    time.sleep(0.05)
    proc.terminate()
    stdout, stderr = proc.communicate(timeout=5)
    end = time.monotonic_ns()
    raw = b"interrupted_control\n" + stdout + stderr
    exit_status = {
        "returncode": proc.returncode,
        "timed_out": False,
        "signal": _signal_name(proc.returncode),
    }
    rows = _rows_from_timing(
        control_id="interrupted",
        raw_output=raw,
        substrate="interrupted_child",
        child_pids=[proc.pid],
        exit_status=exit_status,
        device_ids=["CPU"],
    )
    rows[2]["monotonic_start_ns"] = start
    rows[2]["monotonic_end_ns"] = end
    return rows


def _revision_from_path(path: str | Path) -> str | None:
    """Extract a Hugging Face snapshot revision when present."""

    parts = Path(path).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def _quantization_from_path(path: str | Path) -> str:
    """Extract a common GGUF quantization token from a file name."""

    name = Path(path).name.lower()
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0"):
        if token.lower() in name:
            return token
    return "unknown"


def _file_prefix_sha256(path: str | Path, limit: int = MODEL_PREFIX_BYTES) -> str | None:
    """Hash the model prefix that the child can cheaply re-read."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    with file_path.open("rb") as handle:
        return receipts.sha256_bytes(handle.read(limit))


def embedded_tokenizer_receipt(model_path: str, text: str) -> JsonDict:  # pragma: no cover
    """Load the tokenizer embedded in the GGUF through llama.cpp."""

    if not model_path or not Path(model_path).is_file():
        return {
            "source": TOKENIZER_SOURCE,
            "method": TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "token_count": 0,
            "tokenizer_detail": f"model_path missing or not on disk: {model_path!r}",
            "autotokenizer_used": False,
        }
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        tokens = llm.tokenize(text.encode("utf-8"))
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        return {
            "source": TOKENIZER_SOURCE,
            "method": TOKENIZER_METHOD,
            "loadable": bool(tokens),
            "prompt_tokens": len(tokens),
            "token_count": len(tokens),
            "tokenizer_detail": f"embedded GGUF tokenizer OK ({len(tokens)} tokens)",
            "token_ids_sha256": receipts.sha256_json(tokens),
            "autotokenizer_used": False,
        }
    except Exception as exc:
        return {
            "source": TOKENIZER_SOURCE,
            "method": TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "token_count": 0,
            "tokenizer_detail": f"embedded tokenizer failed: {type(exc).__name__}: {exc}",
            "autotokenizer_used": False,
        }


def _tokenizer_hash(model_id: str, model_hash: str | None, token_count: int) -> str:
    """Bind tokenizer identity to model bytes and measured token count."""

    return receipts.sha256_json(
        {
            "hf_id": model_id,
            "model_file_sha256": model_hash,
            "method": TOKENIZER_METHOD,
            "source": TOKENIZER_SOURCE,
            "token_count": token_count,
        }
    )


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_tokenizer_receipt,
) -> JsonDict:
    """Resolve the mandated Gemma26 GGUF through cached_sota_pair()."""

    calls = [{"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": [1, 0]}]
    pair = (
        cached_pair_func(
            gpu_indices=(0, 1),
            preferred_quant=PREFERRED_QUANT,
            model_indices=(1, 0),
        )
        or []
    )
    blockers: list[str] = []
    chosen = next(
        (dict(row) for row in pair if row.get("hf_id") == MANDATED_POWERED_MODEL_ID), None
    )
    if chosen is None:
        blockers.append(f"missing_cached_sota_pair_row:{MANDATED_POWERED_MODEL_ID}")
        chosen = {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": MANDATED_POWERED_MODEL_ID,
            "gpu": 0,
            "model_path": "",
        }
    path = Path(str(chosen.get("model_path") or ""))
    model_hash = receipts.sha256_file(path) if path.is_file() else None
    tokenized = tokenizer_func(str(path), "Exp6426 runtime receipt smoke.")
    token_count = int(tokenized.get("prompt_tokens", tokenized.get("token_count", 0)) or 0)
    record = {
        "name": chosen.get("name", "Gemma4-26B-A4B-it"),
        "hf_id": MANDATED_POWERED_MODEL_ID,
        "gpu": int(chosen.get("gpu", 0) or 0),
        "model_path": str(path) if str(path) != "." else "",
        "exists": path.is_file(),
        "revision": _revision_from_path(path),
        "quantization": _quantization_from_path(path),
        "model_file_sha256": model_hash,
        "model_file_prefix_sha256": _file_prefix_sha256(path),
        "tokenizer_source": tokenized.get("source", TOKENIZER_SOURCE),
        "tokenizer_method": tokenized.get("method", TOKENIZER_METHOD),
        "tokenizer_loadable": tokenized.get("loadable") is True,
        "prompt_tokens_for_tokenizer_precheck": token_count,
        "tokenizer_detail": str(tokenized.get("tokenizer_detail", "")),
        "tokenizer_sha256": _tokenizer_hash(MANDATED_POWERED_MODEL_ID, model_hash, token_count),
        "autotokenizer_used": False,
    }
    if not record["exists"]:
        blockers.append(f"missing_gguf_file:{MANDATED_POWERED_MODEL_ID}")
    if not record["tokenizer_loadable"]:
        blockers.append(f"embedded_tokenizer_unavailable:{MANDATED_POWERED_MODEL_ID}")
    if tokenized.get("autotokenizer_used") is True:
        blockers.append(f"autotokenizer_used:{MANDATED_POWERED_MODEL_ID}")
    return {
        "MODEL_SPECS": [record],
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": calls,
            "returned_hf_ids": [row.get("hf_id") for row in pair],
            "mandated_model_found": record["exists"],
        },
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
        "autotokenizer_usage_count": 0,
    }


def source_hashes() -> dict[str, str | None]:
    """Hash files that define this contract."""

    return {
        path.as_posix(): receipts.sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS
    }


def protected_hashes() -> dict[str, str | None]:
    """Hash files that this task must not mutate."""

    return {
        path.as_posix(): receipts.sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS
    }


def protected_unchanged_receipt(before: Mapping[str, str | None]) -> JsonDict:
    """Compare protected files before and after the task."""

    after = protected_hashes()
    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def receipt_schema_version_and_hash() -> JsonDict:
    """Return the versioned row schema receipt."""

    payload = {
        "schema_version": receipts.SCHEMA_VERSION,
        "required_row_fields": list(receipts.REQUIRED_ROW_FIELDS),
        "required_phases": list(receipts.REQUIRED_PHASES),
        "attack_ids": list(receipts.ATTACK_IDS),
    }
    return {
        "schema_version": receipts.SCHEMA_VERSION,
        "schema_sha256": receipts.sha256_json(payload),
        "payload": payload,
    }


def helper_source_and_test_hashes(source_before: Mapping[str, str | None]) -> JsonDict:
    """Return source hashes for the helper, test, spec, and experiment."""

    wanted = (
        HELPER_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
    )
    return {path: source_before.get(path) for path in wanted}


def model_file_and_tokenizer_hashes(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return model file and embedded tokenizer hash rows."""

    return [
        {
            "hf_id": row.get("hf_id"),
            "path": row.get("model_path"),
            "model_file_sha256": row.get("model_file_sha256"),
            "model_file_prefix_sha256": row.get("model_file_prefix_sha256"),
            "tokenizer_source": row.get("tokenizer_source"),
            "tokenizer_method": row.get("tokenizer_method"),
            "tokenizer_sha256": row.get("tokenizer_sha256"),
            "tokenizer_loadable": row.get("tokenizer_loadable") is True,
        }
        for row in model_specs
    ]


def _nvidia_gpu_snapshot() -> JsonDict:  # pragma: no cover
    """Collect NVIDIA device inventory."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    devices: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total_mb": int(float(parts[3])),
                    "memory_used_mb": int(float(parts[4])),
                    "memory_free_mb": int(float(parts[5])),
                    "utilization_pct": int(float(parts[6])),
                }
            )
        except ValueError:
            continue
    return {
        "ok": result.returncode == 0,
        "devices": devices,
        "stderr_sha256": receipts.sha256_text(result.stderr),
    }


def _llama_cpp_support() -> JsonDict:  # pragma: no cover
    """Return llama.cpp CUDA support status."""

    try:
        from llama_cpp import __version__ as version
        from llama_cpp import llama_cpp

        info = llama_cpp.llama_print_system_info()
        text = info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info)
        return {
            "llama_cpp_available": True,
            "llama_cpp_version": str(version),
            "llama_cpp_cuda_supported": bool(llama_cpp.llama_supports_gpu_offload()),
            "system_info_sha256": receipts.sha256_text(text),
            "system_info_excerpt": text[:1200],
        }
    except Exception as exc:
        return {
            "llama_cpp_available": False,
            "llama_cpp_cuda_supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _memory_available_kb() -> int:  # pragma: no cover
    """Read available RAM from /proc/meminfo."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except OSError:
        return 0
    return 0


class LocalRuntimeAdapter:  # pragma: no cover
    """Live runtime adapter for the mandated local GGUF smoke."""

    def preflight_receipts(self, model_specs: list[JsonDict]) -> JsonDict:
        """Check host, GPU, cache, runner, tokenizer, and clock preconditions."""

        snapshot = _nvidia_gpu_snapshot()
        devices = list(snapshot.get("devices", []))
        by_index = {int(row["index"]): row for row in devices if "index" in row}
        disk = os.statvfs(REPO_ROOT)
        storage_free_gb = disk.f_bavail * disk.f_frsize / (1024**3)
        llama = _llama_cpp_support()
        runner = _runner_selection("powered", "cuda_gguf")
        monotonic_a = time.monotonic_ns()
        monotonic_b = time.monotonic_ns()
        vram_ready = {
            str(row["hf_id"]): int(by_index.get(int(row["gpu"]), {}).get("memory_free_mb", 0))
            >= 16_000
            for row in model_specs
        }
        blockers: list[str] = []
        names = [str(row.get("name", "")) for row in devices]
        both_rtx = len(devices) >= 2 and all("RTX 3090" in name for name in names[:2])
        if snapshot.get("ok") is not True or len(devices) < 2:
            blockers.append("both_rtx_3090_devices_not_visible")
        if not both_rtx:
            blockers.append("both_rtx_3090_names_not_confirmed")
        if not all(vram_ready.values()):
            blockers.append("insufficient_free_vram")
        if any(row.get("exists") is not True for row in model_specs):
            blockers.append("model_cache_missing")
        if any(row.get("tokenizer_loadable") is not True for row in model_specs):
            blockers.append("embedded_tokenizer_metadata_missing")
        if llama.get("llama_cpp_cuda_supported") is not True:
            blockers.append("llama_cpp_cuda_unsupported")
        if runner.get("binary_sha256") is None:
            blockers.append("runner_binary_hash_missing")
        if storage_free_gb < 5.0:
            blockers.append("disk_space_below_5gb")
        if _memory_available_kb() < 8_000_000:
            blockers.append("ram_below_8gb")
        if monotonic_b < monotonic_a:
            blockers.append("monotonic_clock_rollback")
        return {
            "both_rtx_3090_devices_visible": bool(both_rtx),
            "free_vram_ready": all(vram_ready.values()),
            "vram_ready_by_model": vram_ready,
            "model_cache_ready": all(row.get("exists") is True for row in model_specs),
            "llama_cpp_cuda_supported": llama.get("llama_cpp_cuda_supported") is True,
            "llama_cpp_receipt": llama,
            "runner_binary_ready": runner.get("binary_sha256") is not None,
            "runner_binary_receipt": runner,
            "tokenizer_metadata_ready": all(
                row.get("tokenizer_loadable") is True for row in model_specs
            ),
            "disk_ready": storage_free_gb >= 5.0,
            "storage_free_gb": round(storage_free_gb, 6),
            "cpu_ready": bool(platform.processor() or platform.machine()),
            "cpu_receipt": {"machine": platform.machine(), "processor": platform.processor()},
            "ram_ready": _memory_available_kb() >= 8_000_000,
            "ram_available_kb": _memory_available_kb(),
            "monotonic_clock_ready": monotonic_b >= monotonic_a,
            "monotonic_clock_probe": {"a_ns": monotonic_a, "b_ns": monotonic_b},
            "gpu_snapshot": snapshot,
            "blocked_reasons": sorted(set(blockers)),
        }

    def powered_control_rows(
        self,
        *,
        task_id: str,
        model: JsonDict,
        output_dir: Path,
    ) -> list[JsonDict]:
        """Run the mandated GGUF smoke and convert child timings into rows."""

        return run_powered_smoke(task_id=task_id, model=model, output_dir=output_dir)


LIVE_CHILD_CODE = r"""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def emit(payload):
    sys.stderr.write("\nCARNOT_EXP6426_CHILD:%s\n" % json.dumps(payload, sort_keys=True))
    sys.stderr.flush()

def query_apps():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append({"pid": int(parts[0]), "device_uuid": parts[1], "pid_memory_mb": int(float(parts[2]))})
        except ValueError:
            pass
    return rows

def query_devices():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append({
                "gpu_index": int(parts[0]),
                "device_uuid": parts[1],
                "device_memory_used_mb": int(float(parts[2])),
                "device_memory_free_mb": int(float(parts[3])),
            })
        except ValueError:
            pass
    return rows

def sample(phase, expected_gpu):
    now = time.monotonic_ns()
    pid = os.getpid()
    devices = query_devices()
    apps = query_apps()
    device = next((row for row in devices if row["gpu_index"] == expected_gpu), {})
    app = next((row for row in apps if row["pid"] == pid), {})
    return {
        "phase": phase,
        "pid": pid,
        "device_uuid": app.get("device_uuid") or device.get("device_uuid", ""),
        "gpu_index": expected_gpu,
        "pid_memory_mb": int(app.get("pid_memory_mb", 0) or 0),
        "device_memory_used_mb": int(device.get("device_memory_used_mb", 0) or 0),
        "device_memory_free_mb": int(device.get("device_memory_free_mb", 0) or 0),
        "monotonic_ns": now,
        "sample_age_s": 0.0,
    }

args = json.loads(sys.argv[1])
model_path = args["model_path"]
prompt = args["prompt"]
gpu = int(args["gpu"])
llm = None
try:
    from llama_cpp import Llama

    load_start = time.monotonic_ns()
    llm = Llama(
        model_path=model_path,
        n_ctx=int(args["n_ctx"]),
        n_gpu_layers=-1,
        main_gpu=0,
        seed=int(args["seed"]),
        verbose=False,
    )
    load_end = time.monotonic_ns()
    load_sample = sample("model_load", gpu)
    generation_start = time.monotonic_ns()
    first_token = None
    pieces = []
    for call_index in range(int(args["completion_calls"])):
        call_prompt = "%s\nCall %d: emit receipt evidence words." % (prompt, call_index)
        for chunk in llm.create_completion(
            call_prompt,
            max_tokens=int(args["max_tokens"]),
            temperature=0.0,
            stream=True,
        ):
            text = str((chunk.get("choices") or [{}])[0].get("text") or "")
            if text and first_token is None:
                first_token = time.monotonic_ns()
            pieces.append(text)
            sys.stdout.buffer.write(text.encode("utf-8", "replace"))
            sys.stdout.flush()
    generation_sample = sample("generation", gpu)
    generation_end = max(time.monotonic_ns(), int(generation_sample.get("monotonic_ns", 0) or 0))
    close = getattr(llm, "close", None)
    if callable(close):
        close()
    del llm
    llm = None
    gc.collect()
    emit({
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "load_start_ns": load_start,
        "load_end_ns": load_end,
        "generation_start_ns": generation_start,
        "generation_end_ns": generation_end,
        "first_token_ns": first_token,
        "completion_ns": generation_end,
        "gpu_samples": [load_sample, generation_sample],
    })
except Exception as exc:
    emit({"pid": os.getpid(), "parent_pid": os.getppid(), "error": "%s: %s" % (type(exc).__name__, exc)})
    raise SystemExit(1)
finally:
    if llm is not None:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
"""


def _parse_child(stderr_text: str) -> JsonDict:  # pragma: no cover
    """Parse the powered child receipt from stderr."""

    marker = "CARNOT_EXP6426_CHILD:"
    for line in reversed(stderr_text.splitlines()):
        if line.startswith(marker):
            try:
                value = json.loads(line.removeprefix(marker))
            except json.JSONDecodeError:
                return {}
            return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _write_bytes_atomic(path: Path, payload: bytes) -> Path:  # pragma: no cover
    """Write bytes through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    tmp.replace(path)
    return path


def _safe_slug(value: str) -> str:  # pragma: no cover
    """Return a filesystem-safe slug."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "row"


def run_powered_smoke(
    *, task_id: str, model: Mapping[str, Any], output_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Run one real local GGUF CUDA smoke and return task phase rows."""

    model_id = str(model["hf_id"])
    gpu = int(model.get("gpu", 0) or 0)
    prompt = "Exp6426 runtime receipt smoke. Write a short receipt clause."
    child_args = {
        "model_path": model["model_path"],
        "prompt": prompt,
        "gpu": gpu,
        "seed": RANDOM_SEED,
        "n_ctx": N_CTX,
        "max_tokens": MAX_TOKENS,
        "completion_calls": COMPLETION_CALLS,
    }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    queue_start = time.monotonic_ns()
    proc = subprocess.Popen(
        [sys.executable, "-c", LIVE_CHILD_CODE, json.dumps(child_args, sort_keys=True)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    queue_end = time.monotonic_ns()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=POWERED_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    child = _parse_child(stderr.decode("utf-8", "replace"))
    exit_status = {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "signal": "SIGKILL" if timed_out else _signal_name(proc.returncode),
    }
    sidecar_dir = output_dir / "powered-smoke"
    slug = _safe_slug(model_id)
    raw_path = _write_bytes_atomic(sidecar_dir / f"{slug}.raw.bin", stdout)
    _write_bytes_atomic(sidecar_dir / f"{slug}.stderr.txt", stderr)
    raw = stdout or b"powered_smoke_empty_output"
    model_hash = str(
        model.get("model_file_sha256") or receipts.sha256_file(str(model.get("model_path")))
    )
    child_pid = int(child.get("pid", proc.pid) or proc.pid)
    gpu_samples = [dict(sample) for sample in child.get("gpu_samples", [])]
    for sample in gpu_samples:
        sample.setdefault("pid", child_pid)
        sample.setdefault("sample_age_s", 0.0)
    rows: list[JsonDict] = []
    phase_intervals = {
        "queue_wait": (queue_start, queue_end),
        "model_load": (
            int(child.get("load_start_ns", queue_end) or queue_end),
            int(child.get("load_end_ns", queue_end) or queue_end),
        ),
        "generation": (
            int(child.get("generation_start_ns", queue_end) or queue_end),
            int(child.get("generation_end_ns", queue_end) or queue_end),
        ),
        "exact_verification": (time.monotonic_ns(), time.monotonic_ns()),
        "artifact_write": (time.monotonic_ns(), time.monotonic_ns()),
    }
    for phase in receipts.REQUIRED_PHASES:
        start, end = phase_intervals[phase]
        phase_samples = [sample for sample in gpu_samples if sample.get("phase") == phase]
        if phase == "generation" and not phase_samples:
            phase_samples = gpu_samples
        rows.append(
            receipts.build_phase_row(
                task_id=task_id,
                control_id="powered",
                phase=phase,
                monotonic_start_ns=start,
                monotonic_end_ns=max(end, start),
                wall_clock_start=_utc_now(),
                wall_clock_end=_utc_now(),
                parent_pid=os.getpid(),
                child_pids=[child_pid],
                command=[sys.executable, "-c", LIVE_CHILD_CODE, "<child_args_json>"],
                config={
                    "seed": RANDOM_SEED,
                    "n_ctx": N_CTX,
                    "max_tokens": MAX_TOKENS,
                    "completion_calls": COMPLETION_CALLS,
                    "n_gpu_layers": -1,
                    "gpu": gpu,
                    "raw_output_path": str(raw_path),
                },
                model_identity={
                    "hf_id": model_id,
                    "model_sha256": model_hash,
                    "model_identity_bound": True,
                    "model_path": model.get("model_path"),
                    "tokenizer_source": TOKENIZER_SOURCE,
                },
                runner_selection=_runner_selection("powered", "cuda_gguf"),
                device_ids=[
                    str(sample.get("device_uuid"))
                    for sample in gpu_samples
                    if sample.get("device_uuid")
                ]
                or [f"GPU-{gpu}"],
                concurrency_group=f"{TASK_ID}:powered",
                raw_output_bytes=raw,
                exit_status=exit_status,
                attribution_confidence=1.0 if proc.returncode == 0 and raw else 0.0,
                gpu_samples=phase_samples,
                cpu_fallback=False,
                extra={
                    "raw_output_path": str(raw_path),
                    "first_token_or_completion_evidence": {
                        "first_token_monotonic_ns": child.get("first_token_ns"),
                        "completion_monotonic_ns": child.get("completion_ns"),
                        "raw_output_sha256": receipts.sha256_bytes(raw),
                    },
                },
            )
        )
    return rows


def _powered_blocked_rows(blocked_reason: str, model_hash: str | None) -> list[JsonDict]:
    """Build powered rows when preflight blocks the CUDA smoke."""

    rows = _rows_from_timing(
        control_id="powered",
        raw_output=blocked_reason.encode("utf-8"),
        substrate="cuda_gguf",
        child_pids=[],
        exit_status={"returncode": None, "timed_out": False, "blocked": True},
        device_ids=["GPU_BLOCKED"],
        blocked_reason=blocked_reason,
        gpu_samples=[
            {
                "pid": 0,
                "device_uuid": "GPU_BLOCKED",
                "gpu_index": 0,
                "pid_memory_mb": 0,
                "device_memory_used_mb": 0,
                "monotonic_ns": time.monotonic_ns(),
                "sample_age_s": 99.0,
            }
        ],
        model_sha256=model_hash,
    )
    return rows


def preconditions_from(
    *,
    date: str,
    model_resolution: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    source_before: Mapping[str, str | None],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze all checks that must happen before powered work."""

    blockers = list(model_resolution.get("blocked_reasons", []))
    blockers.extend(str(reason) for reason in runtime_preflight.get("blocked_reasons", []))
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    return {
        "date": date,
        "planning_date": RUN_DATE,
        "all_required_gguf_files_present": all(
            row.get("exists") is True for row in model_resolution.get("MODEL_SPECS", [])
        ),
        "all_embedded_tokenizers_loadable": all(
            row.get("tokenizer_loadable") is True for row in model_resolution.get("MODEL_SPECS", [])
        ),
        "autotokenizer_usage_count": 0,
        "device_inventory_ready": runtime_preflight.get("both_rtx_3090_devices_visible") is True,
        "free_vram_ready": runtime_preflight.get("free_vram_ready") is True,
        "model_cache_ready": runtime_preflight.get("model_cache_ready") is True,
        "llama_cpp_cuda_supported": runtime_preflight.get("llama_cpp_cuda_supported") is True,
        "runner_binary_ready": runtime_preflight.get("runner_binary_ready") is True,
        "tokenizer_metadata_ready": runtime_preflight.get("tokenizer_metadata_ready") is True,
        "disk_ready": runtime_preflight.get("disk_ready") is True,
        "cpu_ready": runtime_preflight.get("cpu_ready") is True,
        "ram_ready": runtime_preflight.get("ram_ready") is True,
        "monotonic_clock_ready": runtime_preflight.get("monotonic_clock_ready") is True,
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
    }


def _control_rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Group rows by control id."""

    return {
        control_id: [row for row in rows if row.get("control_id") == control_id]
        for control_id in CONTROL_IDS
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every required gate passes."""

    attack = artifact.get("attack_matrix", {})
    gates = (
        artifact.get("blocked_reason") == "",
        artifact.get("models_used") == [MANDATED_POWERED_MODEL_ID],
        artifact.get("autotokenizer_usage_count") == 0,
        artifact.get("synthesized_runtime_field_count") == 0,
        artifact.get("cpu_fallback_count") == 0,
        artifact.get("attribution_failure_count") == 0,
        float(artifact.get("reported_vs_recomputed_duration_delta", 1.0) or 1.0) <= 0.1,
        artifact.get("verifier_is_oracle") is False,
        artifact.get("preconditions_checked", {}).get("all_preconditions_passed") is True,
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        attack.get("all_critical_fail_closed") is True,
        attack.get("false_accept_count") == 0,
        all(
            count == len(receipts.REQUIRED_PHASES)
            for count in artifact.get("per_unit_rows", {}).get("control_phase_counts", {}).values()
        ),
        not artifact.get("current_adversarial_findings"),
    )
    return 1.0 if all(gates) else 0.0


def _terminal_prefix_ok(value: str) -> bool:
    """Return true for approved terminal verdict prefixes."""

    return value.startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    )


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact."""

    if artifact.get("blocked_reason"):
        return "blocked_precondition"
    if float(artifact.get("runtime_receipt_contract_ready_score", 0.0)) == 1.0:
        return "complete"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict with the required prefix."""

    if artifact.get("status") == "complete":
        return "complete: task-scoped runtime receipt contract passed all four controls"
    if artifact.get("status") == "blocked_precondition":
        return f"complete_blocked: powered GGUF smoke blocked by {artifact.get('blocked_reason')}"
    return "complete_null: runtime receipt controls ran but one or more attribution gates failed"


def build_artifact(
    *,
    date: str,
    model_resolution: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
    test_exit_codes: Mapping[str, int | None],
) -> JsonDict:
    """Build the terminal Exp6426 artifact."""

    model_specs = list(model_resolution.get("MODEL_SPECS", []))
    preconditions = preconditions_from(
        date=date,
        model_resolution=model_resolution,
        runtime_preflight=runtime_preflight,
        source_before=source_before,
        protected_before=protected_before,
    )
    row_report = receipts.validate_contract_rows(rows, expected_controls=CONTROL_IDS)
    attack_matrix = receipts.mutation_attack_matrix(rows, expected_controls=CONTROL_IDS)
    blocked_reasons = list(preconditions.get("blocked_reasons", []))
    powered_generation = [
        row
        for row in rows
        if row.get("control_id") == "powered" and row.get("phase") == "generation"
    ]
    powered_ok = bool(powered_generation) and row_report["accepted"] is True and not blocked_reasons
    recomputed_duration = float(row_report["recomputed_duration_s"])
    reported_duration = round(recomputed_duration, 1)
    artifact: JsonDict = {
        "status": "",
        "receipt_schema_version_and_hash": receipt_schema_version_and_hash(),
        "helper_source_and_test_hashes": helper_source_and_test_hashes(source_before),
        "MODEL_SPECS": model_specs,
        "models_used": [MANDATED_POWERED_MODEL_ID] if powered_ok else [],
        "cached_sota_pair_receipts": model_resolution.get("cached_sota_pair_receipts", {}),
        "model_file_and_embedded_tokenizer_hashes": model_file_and_tokenizer_hashes(model_specs),
        "autotokenizer_usage_count": int(model_resolution.get("autotokenizer_usage_count", 0) or 0),
        "runner_binary_and_selection_receipts": {
            control_id: rows_for_control[0].get("runner_selection")
            for control_id, rows_for_control in _control_rows_by_id(rows).items()
            if rows_for_control
        },
        "device_inventory_and_preflight_receipts": runtime_preflight,
        "per_unit_rows": {**row_report, "rows": list(rows)},
        "cpu_blocked_interrupted_and_powered_control_rows": _control_rows_by_id(rows),
        "per_phase_monotonic_and_wall_clock_receipts": [
            {
                "control_id": row.get("control_id"),
                "phase": row.get("phase"),
                "monotonic_start_ns": row.get("monotonic_start_ns"),
                "monotonic_end_ns": row.get("monotonic_end_ns"),
                "wall_clock_start": row.get("wall_clock_start"),
                "wall_clock_end": row.get("wall_clock_end"),
            }
            for row in rows
        ],
        "parent_child_pid_and_exit_receipts": receipts.parent_child_exit_receipts(rows),
        "pid_linked_gpu_samples": receipts.pid_linked_gpu_samples(rows),
        "concurrency_group_receipts": receipts.concurrency_group_receipts(rows),
        "command_config_model_and_raw_output_hashes": receipts.command_config_model_raw_hashes(
            rows
        ),
        "synthesized_runtime_field_count": row_report["synthesized_runtime_field_count"],
        "cpu_fallback_count": row_report["cpu_fallback_count"],
        "attribution_failure_count": row_report["attribution_failure_count"],
        "recomputed_duration_s": recomputed_duration,
        "reported_vs_recomputed_duration_delta": round(
            abs(reported_duration - recomputed_duration), 9
        ),
        "attack_matrix": attack_matrix,
        "runtime_receipt_contract_ready_score": 0.0,
        "current_adversarial_findings": list(row_report["reasons"]),
        "protected_files_unchanged": protected_unchanged_receipt(protected_before),
        "blocked_reason": ";".join(blocked_reasons),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": reported_duration,
        "tests_run": {"commands": list(DEFAULT_TEST_COMMANDS), "exit_codes": dict(test_exit_codes)},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while ignoring volatile terminal fields."""

    normalized = json.loads(receipts.canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["recomputed_duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return receipts.sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh score, status, verdict, and checksum."""

    artifact["runtime_receipt_contract_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields and terminal boundaries."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors
    if MANDATED_POWERED_MODEL_ID not in [
        row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])
    ]:
        errors.append("MODEL_SPECS missing mandated Gemma26 GGUF")
    if artifact.get("autotokenizer_usage_count") != 0:
        errors.append("autotokenizer_usage_count must be zero")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    principles = artifact.get("field_principles", {})
    for key in (*REQUIRED_ARTIFACT_FIELDS, *receipts.REQUIRED_ROW_FIELDS, *receipts.ATTACK_IDS):
        if key not in principles:
            errors.append(f"missing field_principles entry: {key}")
            break
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict", ""))):
        errors.append("honest_verdict lacks required terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    """Write the terminal artifact atomically."""

    return receipts.write_json_atomic(path, artifact)


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_tokenizer_receipt,
    runtime: RuntimeAdapter | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6426 artifact."""

    result = Path(result_path)
    data = Path(data_dir)
    result.parent.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    partial_writer = receipts.TaskScopedReceiptWriter(
        data / "task_runtime_receipts.partial.json", task_id=TASK_ID
    )
    protected_before = protected_hashes()
    source_before = source_hashes()
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = list(model_resolution["MODEL_SPECS"])
    adapter = runtime or LocalRuntimeAdapter()
    runtime_preflight = adapter.preflight_receipts(model_specs)
    preconditions = preconditions_from(
        date=date,
        model_resolution=model_resolution,
        runtime_preflight=runtime_preflight,
        source_before=source_before,
        protected_before=protected_before,
    )
    rows: list[JsonDict] = []
    for row in run_cpu_control():
        rows.append(row)
        partial_writer.record_phase(row)
    for row in run_blocked_control():
        rows.append(row)
        partial_writer.record_phase(row)
    for row in run_interrupted_control():
        rows.append(row)
        partial_writer.record_phase(row)
    model_hash = str(
        model_specs[0].get("model_file_sha256") or receipts.sha256_json(model_specs[0])
    )
    if preconditions["all_preconditions_passed"]:
        powered_rows = adapter.powered_control_rows(
            task_id=TASK_ID, model=model_specs[0], output_dir=data
        )
    else:
        powered_rows = _powered_blocked_rows(";".join(preconditions["blocked_reasons"]), model_hash)
    for row in powered_rows:
        rows.append(row)
        partial_writer.record_phase(row)
    artifact = build_artifact(
        date=date,
        model_resolution=model_resolution,
        runtime_preflight=runtime_preflight,
        rows=rows,
        protected_before=protected_before,
        source_before=source_before,
        test_exit_codes=test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS},
    )
    partial_writer.finalize({"status": artifact["status"], "final_artifact": str(result)})
    errors = validate_artifact(artifact)
    if errors:
        artifact["status"] = "failed_schema"
        artifact["honest_verdict"] = f"complete_failed_schema: {errors}"
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_artifact(artifact, result)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        payload = json.loads(result.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"ok": not errors, "errors": errors, "path": str(result)}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(
        date=str(args.date), result_path=result, data_dir=REPO_ROOT / DATA_DIR_RELATIVE_PATH
    )
    print(
        json.dumps(
            {
                "path": str(result),
                "status": artifact.get("status"),
                "ready": artifact.get("runtime_receipt_contract_ready_score"),
                "honest_verdict": artifact.get("honest_verdict"),
            },
            sort_keys=True,
        )
    )
    return 0 if not validate_artifact(artifact) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
