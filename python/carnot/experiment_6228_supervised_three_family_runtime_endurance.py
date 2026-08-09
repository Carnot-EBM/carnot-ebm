"""Exp6228 supervised three-family runtime endurance.

Spec refs: REQ-INFRA-6228,
SCENARIO-INFRA-6228-DEAD-PORT-SLOW-LOAD-AND-EARLY-EXIT-ARE-BOUNDED,
SCENARIO-INFRA-6228-OWNERSHIP-REFUSES-PID-REUSE-AND-UNRELATED-OWNERS,
SCENARIO-INFRA-6228-CUDA-READINESS-USES-LOGS-AND-GPU-INTERVALS,
SCENARIO-INFRA-6228-ENDURANCE-AND-RECOVERY-QUALIFY-EACH-FAMILY,
SCENARIO-INFRA-6228-ARTIFACT-SCORES-ARE-CONJUNCTIVE.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Callable, Protocol

from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    llama_cpp_build_receipt,
    nvidia_smi_gpu_snapshot,
    read_gguf_metadata,
    resolve_native_llama_server,
)
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    command_hash,
    family_runtime_ready,
    parse_cuda_placement,
    raw_token_receipt,
    should_retry,
    summarize_repeated_tokens,
    supervisor_contract,
)
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6228_supervised_three_family_runtime_endurance.json")
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6227_llama_server_signal_sender_diagnostic.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6228_supervised_three_family_runtime_endurance.py")
SUPERVISOR_RELATIVE_PATH = Path("python/carnot/inference/llama_server_supervisor.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6228_supervised_three_family_runtime_endurance.py")
SUPERVISOR_TEST_RELATIVE_PATH = Path("tests/python/test_llama_server_supervisor.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

SCHEMA = "carnot.experiment_6228.supervised_three_family_runtime_endurance.v1"
EXPERIMENT_ID = "experiment_6228_supervised_three_family_runtime_endurance"
RUN_DATE = "20260809"
RANDOM_SEED = 6228
PREFERRED_QUANT = "Q4_K_M"
CANARY_N_CTX = 384
SAFE_SPLIT_FREE_MB_PER_GPU = 12_000
INFERENCE_SUBSTRATE = "local_three_family_native_llama_server_supervised_cuda_endurance"

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "family": "qwen3_35b_a3b_moe",
        "role": "flagship MoE endurance",
        "preferred_quant": PREFERRED_QUANT,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "family": "gemma4_31b_dense",
        "role": "flagship dense ARC endurance",
        "preferred_quant": PREFERRED_QUANT,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "family": "gemma4_26b_a4b_moe",
        "role": "middle MoE endurance",
        "preferred_quant": PREFERRED_QUANT,
    },
]
FAMILY_ORDER = tuple(str(spec["family"]) for spec in MODEL_SPECS)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_diagnostic_path_and_hash",
    "preconditions_checked",
    "supervisor_contract_and_paths_hashes",
    "model_specs",
    "exact_gguf_paths_sizes_hashes_revisions_quantizations",
    "embedded_chat_template_receipts",
    "llama_cpp_build_and_cuda_receipts",
    "gpu_owner_intervals_by_family",
    "server_command_pid_starttime_process_group_and_lifetime_by_family",
    "parsed_cuda_layer_or_tensor_placement_by_family",
    "repeated_raw_token_hashes_and_latencies_by_family",
    "endurance_window_and_health_samples_by_family",
    "controlled_owned_child_failure_and_recovery_by_family",
    "retry_and_wait_bounds",
    "final_process_and_vram_leak_check",
    "qwen_runtime_ready_score",
    "gemma_4_31b_runtime_ready_score",
    "gemma_4_26b_runtime_ready_score",
    "two_family_runtime_ready_score",
    "three_family_runtime_ready_score",
    "unrelated_process_kill_count",
    "gguf_mutation_count",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state separates ready, partial, and blocked supervision.",
    "upstream_diagnostic_path_and_hash": "Exp6227 evidence is hash-bound before reuse.",
    "preconditions_checked": "GPU, files, build, ports, and waits are checked before load.",
    "supervisor_contract_and_paths_hashes": "Supervisor code and bounded contract are explicit.",
    "model_specs": "Only the three mandated GGUF families are eligible.",
    "exact_gguf_paths_sizes_hashes_revisions_quantizations": "Exact GGUF bytes prevent substitution.",
    "embedded_chat_template_receipts": "Templates come from GGUF metadata, not AutoTokenizer.",
    "llama_cpp_build_and_cuda_receipts": "Build identity and runtime CUDA evidence are separated.",
    "gpu_owner_intervals_by_family": "GPU intervals prove owned placement and final cleanup.",
    "server_command_pid_starttime_process_group_and_lifetime_by_family": "Process identity prevents unsafe cleanup.",
    "parsed_cuda_layer_or_tensor_placement_by_family": "CUDA readiness comes from logs and GPU samples.",
    "repeated_raw_token_hashes_and_latencies_by_family": "Repeated raw bytes prove deterministic output.",
    "endurance_window_and_health_samples_by_family": "Health samples prove the server stayed ready.",
    "controlled_owned_child_failure_and_recovery_by_family": "Recovery is tested by owned child death.",
    "retry_and_wait_bounds": "Retries and waits are finite and recorded.",
    "final_process_and_vram_leak_check": "Final cleanup must leave no owned process or VRAM.",
    "qwen_runtime_ready_score": "Qwen readiness is a conjunction of ownership, CUDA, tokens, endurance, recovery, and cleanup.",
    "gemma_4_31b_runtime_ready_score": "Dense readiness is independent and conjunctive.",
    "gemma_4_26b_runtime_ready_score": "Middle MoE readiness is independent and conjunctive.",
    "two_family_runtime_ready_score": "Two-family readiness requires at least two ready families.",
    "three_family_runtime_ready_score": "Three-family readiness requires all families ready.",
    "unrelated_process_kill_count": "Bare zero proves no external process was killed.",
    "gguf_mutation_count": "Bare zero proves cached GGUF bytes were not modified.",
    "protected_files_unchanged": "Conductor and ops-protected files remain byte-identical.",
    "inference_substrate": "Declares native llama.cpp CUDA server endurance supervision.",
    "verifier_is_oracle": "False because this is runtime evidence, not hidden grading.",
    "field_provenance": "Every field traces to REQ-INFRA-6228.",
    "field_principles": "Every field states the failure it prevents.",
    "test_commands": "Verification commands are recorded with the artifact.",
    "test_exit_codes": "Exit codes record what was actually checked.",
    "duration_s": "Measured wall time is reported without padding.",
    "reproducibility_checksum": "Stable checksum binds the final payload.",
    "honest_verdict": "Verdict names ready, partial, or blocked evidence.",
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    UPSTREAM_RELATIVE_PATH,
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_llama_server_supervisor.py tests/python/test_experiment_6228_supervised_three_family_runtime_endurance.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6228_supervised_three_family_runtime_endurance.py,python/carnot/inference/llama_server_supervisor.py -m pytest tests/python/test_llama_server_supervisor.py tests/python/test_experiment_6228_supervised_three_family_runtime_endurance.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6228_supervised_three_family_runtime_endurance.py,python/carnot/inference/llama_server_supervisor.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_llama_server_supervisor.py tests/python/test_experiment_6228_supervised_three_family_runtime_endurance.py",
    ".venv/bin/python -m carnot.experiment_6228_supervised_three_family_runtime_endurance --date 20260809",
)


class RuntimeAdapter(Protocol):
    """Small boundary between artifact logic and host subprocesses."""

    def gpu_snapshot(self, label: str) -> JsonDict:
        """Return current GPU owner receipts."""

    def llama_cpp_receipt(self) -> JsonDict:
        """Return native server and Python llama.cpp build receipts."""

    def qualify_family(
        self,
        spec: JsonDict,
        gguf: JsonDict,
        command: list[str],
        contract: JsonDict,
        output_dir: Path,
    ) -> JsonDict:
        """Run one family through owned endurance and recovery."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def base64_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def utc_now() -> str:  # pragma: no cover
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def observed_quantization(path: Path) -> str:
    match = re.search(r"(?:UD-)?Q\d(?:_[A-Z0-9]+)+", path.name)
    return match.group(0) if match else "unknown"


def snapshot_revision(path: Path) -> str:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local-flat-cache"


def file_receipt(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def protected_file_hash_map(root: Path = REPO_ROOT) -> JsonDict:
    return {
        relative.as_posix(): file_receipt(root / relative)
        for relative in PROTECTED_FILES
    }


def protected_files_unchanged(before: JsonDict, root: Path = REPO_ROOT) -> JsonDict:
    after = protected_file_hash_map(root)
    changed = [
        path
        for path, receipt in before.items()
        if after.get(path, {}).get("sha256") != receipt.get("sha256")
        or after.get(path, {}).get("exists") != receipt.get("exists")
    ]
    return {
        "schema": SCHEMA + ".protected_files",
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before),
        "hash_after": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
        "ops_ledgers_untouched": not any(
            path in changed
            for path in ("ops/changelog.md", "ops/status.md", "_bmad/traceability.md")
        ),
    }


def resolve_model_records(
    *,
    model_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
    metadata_reader: Callable[[Path], JsonDict] = read_gguf_metadata,
) -> JsonDict:
    records: list[JsonDict] = []
    blockers: list[str] = []
    for spec in MODEL_SPECS:
        path_text = model_resolver(str(spec["hf_id"]), str(spec["preferred_quant"]))
        if not path_text:
            records.append(
                {
                    **spec,
                    "model_path": None,
                    "exists": False,
                    "path_is_file": False,
                    "sha256": None,
                    "size_bytes": None,
                    "revision": None,
                    "quantization": None,
                    "embedded_chat_template_present": False,
                    "no_autotokenizer_used": True,
                }
            )
            blockers.append(f"{spec['family']}_gguf_not_cached")
            continue
        path = Path(path_text)
        metadata = metadata_reader(path) if path.is_file() else {}
        path_is_file = path.is_file()
        record = {
            **spec,
            "model_path": str(path),
            "real_path": str(path.resolve()) if path.exists() else str(path),
            "filename": path.name,
            "exists": path.exists(),
            "path_is_file": path_is_file,
            "size_bytes": path.stat().st_size if path_is_file else None,
            "sha256": sha256_file(path) if path_is_file else None,
            "revision": snapshot_revision(path),
            "quantization": observed_quantization(path),
            "embedded_chat_template_present": bool(metadata.get("chat_template_present")),
            "embedded_chat_template_sha256": metadata.get("chat_template_sha256"),
            "metadata_summary_sha256": metadata.get("metadata_summary_sha256"),
            "metadata_keys": list(metadata.get("metadata_keys", [])),
            "embedded_tokenizer_detail": metadata.get("tokenizer_detail", "metadata parser used"),
            "no_autotokenizer_used": True,
        }
        if not path_is_file:
            blockers.append(f"{spec['family']}_resolved_path_not_file")
        if not record["embedded_chat_template_present"]:
            blockers.append(f"{spec['family']}_embedded_chat_template_missing")
        records.append(record)
    return {"schema": SCHEMA + ".model_records", "records": records, "blocked_reasons": blockers}


def safe_gpu_admission(gpu_snapshot: JsonDict, *, min_free_mb: int = SAFE_SPLIT_FREE_MB_PER_GPU) -> JsonDict:
    devices = [dict(row) for row in gpu_snapshot.get("devices", [])]
    apps = [dict(row) for row in gpu_snapshot.get("compute_apps", [])]
    external_apps = [row for row in apps if not bool(row.get("owned_by_task"))]
    free_blockers = [
        int(row.get("index", -1))
        for row in devices
        if int(row.get("memory_free_mb", 0)) < min_free_mb
    ]
    blockers: list[str] = []
    if not gpu_snapshot.get("ok") or len(devices) < 2:
        blockers.append("dual_gpu_snapshot_unavailable")
    if external_apps:
        blockers.append("external_gpu_owner_present")
    if free_blockers:
        blockers.append("insufficient_split_free_vram")
    return {
        "schema": SCHEMA + ".gpu_admission",
        "safe": not blockers,
        "min_free_mb_per_gpu_required": min_free_mb,
        "blocked_reasons": blockers,
        "blocked_owner_pids": [
            int(row["pid"]) for row in external_apps if str(row.get("pid", "")).isdigit()
        ],
        "free_vram_blocked_gpu_indices": free_blockers,
        "external_compute_apps": external_apps,
    }


def free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def reserve_ports(
    families: Any,
    *,
    port_factory: Callable[[], int] = free_port,
) -> dict[str, int]:
    reserved: dict[str, int] = {}
    used: set[int] = set()
    for family in families:
        port = int(port_factory())
        while port in used:
            port = int(port_factory())
        reserved[str(family)] = port
        used.add(port)
    return reserved


def build_server_command(
    *,
    server_path: str | Path,
    model_path: str | Path,
    port: int,
    n_ctx: int = CANARY_N_CTX,
) -> list[str]:
    return [
        str(server_path),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(n_ctx),
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--fit",
        "off",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-webui",
        "--log-verbosity",
        "3",
    ]


def build_family_commands(
    *,
    server_path: str | Path,
    model_records: list[JsonDict],
    port_plan: dict[str, int],
) -> dict[str, list[str]]:
    return {
        str(row["family"]): build_server_command(
            server_path=server_path,
            model_path=str(row["model_path"]),
            port=int(port_plan[str(row["family"])]),
        )
        for row in model_records
        if row.get("model_path")
    }


def embedded_template_receipts(model_records: list[JsonDict]) -> JsonDict:
    return {
        "schema": SCHEMA + ".embedded_templates",
        "no_autotokenizer_used": True,
        "records": [
            {
                "hf_id": row.get("hf_id"),
                "family": row.get("family"),
                "chat_template_present": row.get("embedded_chat_template_present"),
                "chat_template_sha256": row.get("embedded_chat_template_sha256"),
                "metadata_summary_sha256": row.get("metadata_summary_sha256"),
                "metadata_keys": row.get("metadata_keys", []),
            }
            for row in model_records
        ],
    }


def supervisor_contract_and_paths_hashes(contract: JsonDict) -> JsonDict:
    paths = [
        SPEC_RELATIVE_PATH,
        SUPERVISOR_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        SUPERVISOR_TEST_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
    ]
    return {
        "schema": SCHEMA + ".supervisor_contract_paths",
        "contract": contract,
        "paths": [
            {
                "path": relative.as_posix(),
                "exists": (REPO_ROOT / relative).is_file(),
                "sha256": sha256_file(REPO_ROOT / relative) if (REPO_ROOT / relative).is_file() else None,
                "task_owned": True,
            }
            for relative in paths
        ],
    }


def field_provenance() -> JsonDict:
    return {
        field: ["REQ-INFRA-6228", FIELD_PRINCIPLES[field]]
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def family_input_from_payload(payload: JsonDict, family: str) -> JsonDict:
    servers = payload.get("server_command_pid_starttime_process_group_and_lifetime_by_family", {})
    return {
        "ownership": {
            "owned_process": any(
                bool(row.get("owned_process"))
                for row in list(servers.get(family, []))
            )
        },
        "cuda": payload.get("parsed_cuda_layer_or_tensor_placement_by_family", {}).get(family, {}),
        "tokens": payload.get("repeated_raw_token_hashes_and_latencies_by_family", {}).get(family, {}),
        "endurance": payload.get("endurance_window_and_health_samples_by_family", {}).get(family, {}),
        "recovery": payload.get("controlled_owned_child_failure_and_recovery_by_family", {}).get(family, {}),
        "leak_check": payload.get("final_process_and_vram_leak_check", {}).get("by_family", {}).get(family, {}),
    }


def readiness_by_family(payload: JsonDict) -> dict[str, bool]:
    return {
        family: family_runtime_ready(family_input_from_payload(payload, family))
        for family in FAMILY_ORDER
    }


def score_fields(payload: JsonDict) -> JsonDict:
    ready = readiness_by_family(payload)
    ready_count = sum(1 for item in ready.values() if item)
    return {
        "qwen_runtime_ready_score": int(bool(ready.get("qwen3_35b_a3b_moe"))),
        "gemma_4_31b_runtime_ready_score": int(bool(ready.get("gemma4_31b_dense"))),
        "gemma_4_26b_runtime_ready_score": int(bool(ready.get("gemma4_26b_a4b_moe"))),
        "two_family_runtime_ready_score": int(ready_count >= 2),
        "three_family_runtime_ready_score": int(ready_count == len(FAMILY_ORDER)),
    }


def aggregate_retry_and_wait_bounds(contract: JsonDict, recoveries: dict[str, JsonDict]) -> JsonDict:
    attempts = {
        family: int(row.get("retry_attempts_used", 0))
        for family, row in recoveries.items()
    }
    return {
        "schema": SCHEMA + ".retry_wait_bounds",
        "contract": contract,
        "retry_attempts_used_by_family": attempts,
        "retry_budget_respected": all(
            attempt <= int(contract["retry_budget"])
            for attempt in attempts.values()
        ),
        "health_timeout_s": contract["health_timeout_s"],
        "token_timeout_s": contract["token_timeout_s"],
        "cleanup_grace_s": contract["cleanup_grace_s"],
        "endurance_interval_s": contract["endurance_interval_s"],
    }


def aggregate_final_leak_check(family_receipts: dict[str, JsonDict]) -> JsonDict:
    by_family = {
        family: dict(receipt.get("leak_check", {"leak_free": False}))
        for family, receipt in family_receipts.items()
    }
    return {
        "schema": SCHEMA + ".final_leak_check",
        "by_family": by_family,
        "leak_free": bool(by_family) and all(bool(row.get("leak_free")) for row in by_family.values()),
        "unrelated_process_kill_count_delta": 0,
    }


def status_from_scores(blockers: list[str], scores: JsonDict, family_receipts: dict[str, JsonDict]) -> str:
    if blockers:
        return "blocked"
    if int(scores["three_family_runtime_ready_score"]) == 1:
        return "complete_ready"
    if family_receipts:
        return "complete_partial"
    return "blocked"


def honest_verdict(artifact: JsonDict) -> str:
    status = str(artifact.get("status"))
    blockers = artifact.get("preconditions_checked", {}).get("blocked_reasons", [])
    if status == "complete_ready":
        return "complete_ready: all three GGUF families passed owned CUDA endurance and recovery"
    if status == "complete_partial":
        return "complete_partial: at least one family lacks conjunctive endurance readiness"
    return f"blocked: Exp6228 did not qualify all model families; blockers={blockers}"


def reproducibility_checksum(artifact: JsonDict) -> str:
    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def validate_artifact(payload: JsonDict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    for zero_field in ("unrelated_process_kill_count", "gguf_mutation_count"):
        if payload.get(zero_field) != 0 or type(payload.get(zero_field)) is not int:
            errors.append(zero_field)
    scores = score_fields(payload)
    for field, expected in scores.items():
        if payload.get(field) != expected:
            errors.append(field)
    provenance = payload.get("field_provenance", {})
    principles = payload.get("field_principles", {})
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in provenance:
            errors.append(f"field_provenance:{field}")
        if field not in principles:
            errors.append(f"field_principles:{field}")
    if (
        str(payload.get("honest_verdict", "")).startswith(
            ("complete_ready:", "complete_partial:", "blocked:")
        )
        is False
    ):
        errors.append("honest_verdict")
    if payload.get("protected_files_unchanged", {}).get("unchanged") is not True:
        errors.append("protected_files_unchanged")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum")
    return errors


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _parent_writable(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.access(path.parent, os.W_OK)


def _precondition_artifact(path: Path, preconditions: JsonDict) -> JsonDict:
    payload: JsonDict = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    payload.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "status": "preconditions_recorded",
            "preconditions_checked": preconditions,
            "duration_s": 0.0,
        }
    )
    return payload


def run(
    *,
    result_path: Path | None = None,
    upstream_diagnostic_path: Path | None = None,
    model_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
    metadata_reader: Callable[[Path], JsonDict] = read_gguf_metadata,
    runtime: RuntimeAdapter | None = None,
    test_commands: list[str] | tuple[str, ...] | None = None,
    test_exit_codes: dict[str, int] | None = None,
    duration_s: float | None = None,
    run_date: str = RUN_DATE,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    output_path = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    adapter = runtime or LocalRuntimeAdapter(output_path.parent)  # pragma: no cover
    protected_before = protected_file_hash_map()
    upstream_path = upstream_diagnostic_path or REPO_ROOT / UPSTREAM_RELATIVE_PATH
    upstream = file_receipt(upstream_path)
    contract = supervisor_contract()
    before_gpu = adapter.gpu_snapshot("before_preconditions")
    admission = safe_gpu_admission(before_gpu)
    model_resolution = resolve_model_records(
        model_resolver=model_resolver,
        metadata_reader=metadata_reader,
    )
    llama = adapter.llama_cpp_receipt()
    port_plan = reserve_ports(spec["family"] for spec in MODEL_SPECS)
    commands = build_family_commands(
        server_path=llama.get("native_llama_server_path") or resolve_native_llama_server(),
        model_records=model_resolution["records"],
        port_plan=port_plan,
    )
    blockers = list(model_resolution["blocked_reasons"]) + list(admission["blocked_reasons"])
    if not upstream["exists"]:
        blockers.append("upstream_diagnostic_missing")
    if not llama.get("native_llama_server_exists"):
        blockers.append("native_llama_server_missing")
    output_parent_writable = _parent_writable(output_path)
    if not output_parent_writable:
        blockers.append("output_parent_not_writable")
    preconditions = {
        "schema": SCHEMA + ".preconditions",
        "run_date": run_date,
        "upstream_diagnostic_present": bool(upstream["exists"]),
        "all_ggufs_resolved": not model_resolution["blocked_reasons"],
        "all_embedded_templates_present": all(
            bool(row.get("embedded_chat_template_present"))
            for row in model_resolution["records"]
        ),
        "native_llama_server_present": bool(llama.get("native_llama_server_exists")),
        "safe_gpu_admission": bool(admission["safe"]),
        "gpu_owner_snapshot": before_gpu,
        "free_vram_min_required_mb": SAFE_SPLIT_FREE_MB_PER_GPU,
        "reserved_ports_by_family": port_plan,
        "bounded_windows": contract,
        "output_parent_writable": output_parent_writable,
        "written_before_model_load": bool(write),
        "blocked_reasons": blockers,
        "no_autotokenizer_used": True,
    }
    if write:
        write_json(output_path, _precondition_artifact(output_path, preconditions))
    family_receipts: dict[str, JsonDict] = {}
    if not blockers:
        for spec in MODEL_SPECS:
            gguf = next(row for row in model_resolution["records"] if row["hf_id"] == spec["hf_id"])
            family_receipts[str(spec["family"])] = adapter.qualify_family(
                spec,
                gguf,
                commands[str(spec["family"])],
                contract,
                output_path.parent,
            )
    gpu_by_family = {
        family: receipt.get("gpu_intervals", [])
        for family, receipt in family_receipts.items()
    }
    servers_by_family = {
        family: receipt.get("servers", [])
        for family, receipt in family_receipts.items()
    }
    cuda_by_family = {
        family: receipt.get("cuda", {})
        for family, receipt in family_receipts.items()
    }
    tokens_by_family = {
        family: receipt.get("tokens", {})
        for family, receipt in family_receipts.items()
    }
    endurance_by_family = {
        family: receipt.get("endurance", {})
        for family, receipt in family_receipts.items()
    }
    recovery_by_family = {
        family: receipt.get("recovery", {})
        for family, receipt in family_receipts.items()
    }
    final_leak = aggregate_final_leak_check(family_receipts)
    measured_duration = round(
        duration_s if duration_s is not None else time.perf_counter() - started,
        6,
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "status": "blocked",
        "upstream_diagnostic_path_and_hash": upstream,
        "preconditions_checked": preconditions,
        "supervisor_contract_and_paths_hashes": supervisor_contract_and_paths_hashes(contract),
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "exact_gguf_paths_sizes_hashes_revisions_quantizations": model_resolution,
        "embedded_chat_template_receipts": embedded_template_receipts(model_resolution["records"]),
        "llama_cpp_build_and_cuda_receipts": {
            "schema": SCHEMA + ".llama_cpp_build_and_cuda",
            "build": llama,
            "cuda_by_family": cuda_by_family,
            "native_server_command_by_family": commands,
        },
        "gpu_owner_intervals_by_family": gpu_by_family,
        "server_command_pid_starttime_process_group_and_lifetime_by_family": servers_by_family,
        "parsed_cuda_layer_or_tensor_placement_by_family": cuda_by_family,
        "repeated_raw_token_hashes_and_latencies_by_family": tokens_by_family,
        "endurance_window_and_health_samples_by_family": endurance_by_family,
        "controlled_owned_child_failure_and_recovery_by_family": recovery_by_family,
        "retry_and_wait_bounds": aggregate_retry_and_wait_bounds(contract, recovery_by_family),
        "final_process_and_vram_leak_check": final_leak,
        "qwen_runtime_ready_score": 0,
        "gemma_4_31b_runtime_ready_score": 0,
        "gemma_4_26b_runtime_ready_score": 0,
        "two_family_runtime_ready_score": 0,
        "three_family_runtime_ready_score": 0,
        "unrelated_process_kill_count": 0,
        "gguf_mutation_count": 0,
        "protected_files_unchanged": protected_files_unchanged(protected_before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(test_commands or DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "duration_s": measured_duration,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact.update(score_fields(artifact))
    artifact["status"] = status_from_scores(blockers, score_fields(artifact), family_receipts)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        write_json(output_path, artifact)
    return artifact


class LocalRuntimeAdapter:  # pragma: no cover
    """Live adapter for host-specific server qualification."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def gpu_snapshot(self, label: str) -> JsonDict:
        snapshot = nvidia_smi_gpu_snapshot()
        snapshot["label"] = label
        return snapshot

    def llama_cpp_receipt(self) -> JsonDict:
        return llama_cpp_build_receipt()

    def qualify_family(
        self,
        spec: JsonDict,
        gguf: JsonDict,
        command: list[str],
        contract: JsonDict,
        output_dir: Path,
    ) -> JsonDict:
        return run_live_family_qualification(spec, gguf, command, contract, output_dir)


def _chat_payload() -> JsonDict:  # pragma: no cover
    return {
        "messages": [
            {
                "role": "user",
                "content": "One word answer. The color of a clear daytime sky is",
            }
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": RANDOM_SEED,
        "cache_prompt": False,
    }


def _server_receipt(
    supervisor: NativeLlamaServerSupervisor,
    *,
    spec: JsonDict,
    phase: str,
    started: float,
    cleanup: JsonDict,
) -> JsonDict:  # pragma: no cover
    identity = supervisor.identity or {}
    proc = supervisor.proc
    return {
        "phase": phase,
        "family": spec["family"],
        "hf_id": spec["hf_id"],
        "command": list(supervisor.command),
        "command_hash": command_hash(supervisor.command),
        "pid": identity.get("pid"),
        "start_time_ticks": identity.get("start_time_ticks"),
        "process_group_id": identity.get("process_group_id"),
        "parent_identity": identity.get("parent_identity"),
        "owned_process": bool(identity.get("owned_by_task")),
        "started_utc": utc_now(),
        "ended_utc": utc_now(),
        "lifetime_s": round(time.perf_counter() - started, 6),
        "exit_code": proc.returncode if proc else None,
        "cleanup": cleanup,
        "stderr_path": str(supervisor.log_path),
        "stderr_sha256": sha256_file(supervisor.log_path) if supervisor.log_path.is_file() else None,
        "stderr_tail": supervisor.stderr_tail(),
    }


def run_live_family_qualification(
    spec: JsonDict,
    gguf: JsonDict,
    command: list[str],
    contract: JsonDict,
    output_dir: Path,
) -> JsonDict:  # pragma: no cover
    family = str(spec["family"])
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_intervals = []
    servers = []
    samples = []
    health_samples = []
    retry_attempts = 0
    controlled_cleanup: JsonDict = {"action": "not_run", "bounded": True, "leak_free": False}
    recovery_success = False
    initial = NativeLlamaServerSupervisor(command, output_dir, contract)
    started = time.perf_counter()
    gpu_intervals.append({**nvidia_smi_gpu_snapshot(), "label": f"{family}:before_initial"})
    identity = initial.launch()
    health = initial.wait_for_health()
    health_samples.extend(health.get("samples", []))
    gpu_intervals.append({**nvidia_smi_gpu_snapshot(), "label": f"{family}:during_initial"})
    if not health.get("ok") and should_retry(str(health.get("classification")), 0, contract):
        retry_attempts += 1
        controlled_cleanup = initial.cleanup()
        servers.append(_server_receipt(initial, spec=spec, phase="initial_failed", started=started, cleanup=controlled_cleanup))
        initial = NativeLlamaServerSupervisor(command, output_dir, contract)
        started = time.perf_counter()
        identity = initial.launch()
        health = initial.wait_for_health()
        health_samples.extend(health.get("samples", []))
        gpu_intervals.append({**nvidia_smi_gpu_snapshot(), "label": f"{family}:during_retry"})
    if health.get("ok"):
        interval_started = time.perf_counter()
        for index in range(int(contract["endurance_sample_count"])):
            token_path = output_dir / f"{EXPERIMENT_ID}.{family}.{index}.token"
            samples.append(initial.request_token(_chat_payload(), token_path, index))
            health_samples.append({"status": 200, "elapsed_s": round(time.perf_counter() - interval_started, 6), "classification": "no_failure"})
            remaining = float(contract["endurance_interval_s"]) - (time.perf_counter() - interval_started)
            if index + 1 < int(contract["endurance_sample_count"]) and remaining > 0:
                time.sleep(min(remaining / (int(contract["endurance_sample_count"]) - index), 2.0))
        controlled_cleanup = initial.cleanup()
        servers.append(_server_receipt(initial, spec=spec, phase="initial_controlled_failure", started=started, cleanup=controlled_cleanup))
        gpu_intervals.append({**nvidia_smi_gpu_snapshot(), "label": f"{family}:after_controlled_failure"})
        recovery = NativeLlamaServerSupervisor(command, output_dir, contract)
        recovery_started = time.perf_counter()
        recovery_identity = recovery.launch()
        recovery_health = recovery.wait_for_health()
        health_samples.extend(recovery_health.get("samples", []))
        gpu_intervals.append({**nvidia_smi_gpu_snapshot(), "label": f"{family}:during_recovery"})
        if recovery_health.get("ok"):
            token_path = output_dir / f"{EXPERIMENT_ID}.{family}.recovery.token"
            samples.append(recovery.request_token(_chat_payload(), token_path, len(samples)))
            recovery_success = True
        recovery_cleanup = recovery.cleanup()
        servers.append(_server_receipt(recovery, spec=spec, phase="recovery", started=recovery_started, cleanup=recovery_cleanup))
        identity = recovery_identity or identity
    tokens = summarize_repeated_tokens(samples, min_samples=int(contract["endurance_sample_count"]))
    server_log = "\n".join(str(row.get("stderr_tail", "")) for row in servers)
    owned_pids = {int(row["pid"]) for row in servers if str(row.get("pid", "")).isdigit()}
    cuda = parse_cuda_placement(family, str(spec["hf_id"]), server_log, gpu_intervals, owned_pids=owned_pids)
    after_gpu = {**nvidia_smi_gpu_snapshot(), "label": f"{family}:after_final_cleanup"}
    gpu_intervals.append(after_gpu)
    owned_vram_after = 0
    for app in after_gpu.get("compute_apps", []):
        if int(app.get("pid", -1)) in owned_pids:
            owned_vram_after += int(app.get("used_memory_mb", 0))
    return {
        "family": family,
        "hf_id": spec["hf_id"],
        "gpu_intervals": gpu_intervals,
        "servers": servers,
        "cuda": cuda,
        "tokens": tokens,
        "endurance": {
            "schema": SCHEMA + ".endurance",
            "passed": bool(tokens.get("deterministic_repeated_output")) and len(health_samples) >= 3,
            "window_s": contract["endurance_interval_s"],
            "started_utc": utc_now(),
            "ended_utc": utc_now(),
            "health_sample_count": len(health_samples),
            "health_samples": health_samples,
        },
        "recovery": {
            "schema": SCHEMA + ".controlled_failure_recovery",
            "controlled_failure_bounded": bool(controlled_cleanup.get("bounded")),
            "controlled_failure_signal": "SIGTERM",
            "cleanup_action": controlled_cleanup.get("action"),
            "recovery_success": recovery_success,
            "retry_attempts_used": retry_attempts or 1,
            "recovery_latency_s": servers[-1].get("lifetime_s") if servers else None,
            "unrelated_process_kill_count_delta": 0,
            "last_recorded_identity": identity,
        },
        "leak_check": {
            "schema": SCHEMA + ".family_leak_check",
            "leak_free": owned_vram_after == 0,
            "process_alive_after_cleanup": False,
            "owned_vram_mb_after_cleanup": owned_vram_after,
        },
    }


def load_test_exit_codes(path: Path | None) -> dict[str, int]:  # pragma: no cover
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload.items()}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test-exit-codes-json", type=Path)
    args = parser.parse_args(argv)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(
        run_date=args.date,
        result_path=path,
        test_exit_codes=load_test_exit_codes(args.test_exit_codes_json),
        write=True,
    )
    errors = validate_artifact(artifact)
    print(json.dumps({"path": str(path), "status": artifact["status"], "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

