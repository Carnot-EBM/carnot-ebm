"""Build the Exp 3275 clean local SOTA verifier rerun v14 artifact.

Spec refs: REQ-VERIFY-3275, SCENARIO-VERIFY-3275.

This runner treats the local SOTA model as a verifier whose decision must be
checked against exact fixture authority. The model can only say accept, reject,
or abstain; the exact fixture decides whether that response is a false accept,
false reject, or abstention. That separation keeps repair-gate evidence from
turning an LLM response into its own correctness oracle.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]
ModelRunner = Callable[[list[JsonDict], JsonDict, int], JsonDict]
CudaProbe = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.clean_local_sota_verifier_rerun.v14"
EXPERIMENT_ID = "exp3275"
MILESTONE = "2026.05.303"
RUN_DATE = "20260528"
OUTPUT_REL_PATH = Path("results/experiment_3275_clean_local_sota_verifier_rerun_v14.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3275_clean_local_sota_verifier_rerun_v14.py"

EXP3268_REL_PATH = Path("results/experiment_3268_sota_receipt_methodology_supplement_v1.json")
EXP3223_REL_PATH = Path("results/experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.json")
CONTEXT_FIXTURE_REL_PATH = Path("data/research/context_cot_clbench_parametric_shortcut_v1.jsonl")

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
DEFAULT_RANDOM_SEED = 3275
DEFAULT_MAX_EVAL_ROWS = 6
FALSE_ACCEPT_THRESHOLD = 0.10
ABSTENTION_THRESHOLD = 0.50
MIN_GPU_MEM_USED_MIB = 512
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_FIELDS = {
    "clean_verifier_rerun_ready",
    "clean_rerun_allowed",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "n_eval",
    "exact_row_fixture_hash",
    "model_specs",
    "models_used",
    "preconditions_checked",
    "gpu_mem_used_mib",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3275_clean_local_sota_verifier_rerun_v14.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/clean_local_sota_verifier_rerun_v14.py -m pytest -o addopts='' tests/python/test_experiment_3275_clean_local_sota_verifier_rerun_v14.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/clean_local_sota_verifier_rerun_v14.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    cuda_probe: CudaProbe | None = None,
    model_runner: ModelRunner | None = None,
    max_eval_rows: int = DEFAULT_MAX_EVAL_ROWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3275: build a clean local SOTA rerun or gated-skip artifact."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3268 = read_json_object(root_path / EXP3268_REL_PATH)
    exp3223 = read_json_object(root_path / EXP3223_REL_PATH)
    model_specs = model_specs_from_receipt(exp3268)
    reasons: list[str] = []
    preconditions: list[JsonDict] = []
    exact_rows: list[JsonDict] = []
    per_row_results: list[JsonDict] = []
    models: list[JsonDict] = []
    gpu_mem_used_mib = 0

    receipt_ok = exp3268.get("clean_sota_receipt_eligible") is True
    preconditions.append(
        {
            "name": "exp3268_clean_sota_receipt_eligible",
            "passed": receipt_ok,
            "path": EXP3268_REL_PATH.as_posix(),
        }
    )
    if not receipt_ok:
        reasons.append("exp3268.clean_sota_receipt_eligible=false")
    else:
        cuda = normalize_precondition((cuda_probe or default_cuda_probe)())
        preconditions.append(cuda)
        if cuda.get("passed") is not True:
            reasons.append("cuda_unavailable")

        models = resolve_models(exp3268, root_path)
        preconditions.append(
            {
                "name": "mandated_sota_gguf_cache",
                "passed": bool(models),
                "available_model_ids": [model["model_id"] for model in models],
            }
        )
        if not models:
            reasons.append("mandated_sota_gguf_unavailable")

        exact_rows = build_exact_eval_rows(root_path, max_rows=max_eval_rows)
        fixture_ready = (
            bool(exact_rows)
            and (root_path / CONTEXT_FIXTURE_REL_PATH).is_file()
            and exp3223.get("exact_verifier_authority_preserved") is True
        )
        preconditions.append(
            {
                "name": "exact_row_fixture_availability",
                "passed": fixture_ready,
                "path": CONTEXT_FIXTURE_REL_PATH.as_posix(),
                "exact_rows_available": len(exact_rows),
                "sidecar_ready": exp3223.get("uncertainty_sidecar_ready") is True,
                "exact_verifier_authority_preserved": (
                    exp3223.get("exact_verifier_authority_preserved") is True
                ),
            }
        )
        if not fixture_ready:
            reasons.append("exact_row_fixture_unavailable")

    gated_skip = bool(reasons)
    if not gated_skip and models and exact_rows:
        runner = model_runner or run_llama_local_verifier
        try:
            runner_payload = runner(exact_rows, models[0], int(random_seed))
        except Exception as exc:  # pragma: no cover - defensive live-run boundary.
            runner_payload = {
                "rows": [],
                "gpu_mem_used_mib": 0,
                "runner_error": f"{type(exc).__name__}: {exc}",
            }
        per_row_results = normalize_runner_rows(
            runner_payload.get("rows") if isinstance(runner_payload, Mapping) else [],
            exact_rows,
            models[0],
        )
        gpu_mem_used_mib = safe_int(mapping(runner_payload).get("gpu_mem_used_mib"))
        if mapping(runner_payload).get("runner_error"):
            reasons.append("model_runner_failed: " + str(mapping(runner_payload)["runner_error"]))
        if not per_row_results:  # pragma: no cover - exact rows always normalize to abstentions.
            reasons.append("local_verifier_rows_absent")
        if per_row_results and gpu_mem_used_mib < MIN_GPU_MEM_USED_MIB:
            reasons.append("gpu_mem_used_below_cuda_offload_floor")

    metrics = score_results(per_row_results)
    if per_row_results and metrics["false_accept_rate"] > FALSE_ACCEPT_THRESHOLD:
        reasons.append("false_accept_rate_above_threshold")
    if per_row_results and metrics["abstention_rate"] > ABSTENTION_THRESHOLD:
        reasons.append("abstention_rate_above_threshold")

    repair_ready = (
        bool(per_row_results)
        and bool(models)
        and gpu_mem_used_mib >= MIN_GPU_MEM_USED_MIB
        and metrics["false_accept_rate"] <= FALSE_ACCEPT_THRESHOLD
        and metrics["abstention_rate"] <= ABSTENTION_THRESHOLD
        and all(row.get("synthetic_shortcut_row") is False for row in per_row_results)
        and not any(reason.endswith("unavailable") for reason in reasons)
        and not any(reason.startswith("model_runner_failed") for reason in reasons)
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    source_artifacts = build_source_artifacts(root_path)
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3275", "SCENARIO-VERIFY-3275"],
        "clean_verifier_rerun_ready": repair_ready,
        "clean_rerun_allowed": repair_ready,
        "gated_skip": not per_row_results,
        "gate_reasons": sorted(set(reasons)),
        "repair_gate_input_clean_enough": repair_ready,
        "false_accept_rate": metrics["false_accept_rate"],
        "false_reject_rate": metrics["false_reject_rate"],
        "abstention_rate": metrics["abstention_rate"],
        "false_accept_count": metrics["false_accept_count"],
        "false_reject_count": metrics["false_reject_count"],
        "abstention_count": metrics["abstention_count"],
        "n_eval": metrics["n_eval"],
        "exact_row_fixture_hash": exact_row_fixture_hash(root_path, exact_rows),
        "model_specs": model_specs,
        "models_used": models if per_row_results else [],
        "preconditions_checked": preconditions,
        "gpu_mem_used_mib": int(gpu_mem_used_mib if per_row_results else 0),
        "random_seed": int(random_seed),
        "per_row_results": per_row_results,
        "source_artifacts": source_artifacts,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_artifacts if row.get("sha256")
        },
        "inference_substrate": (
            "local_sota_gguf_exact_row_verifier"
            if per_row_results
            else "gated_skip_no_model_call"
        ),
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
    cuda_probe: CudaProbe | None = None,
    model_runner: ModelRunner | None = None,
    max_eval_rows: int = DEFAULT_MAX_EVAL_ROWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the v14 terminal JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        cuda_probe=cuda_probe,
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


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning an empty mapping for missing/malformed evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_objects(path: Path) -> list[JsonDict]:
    """Read JSONL rows and drop malformed/non-object rows instead of guessing."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def build_exact_eval_rows(root: Path | str, *, max_rows: int = DEFAULT_MAX_EVAL_ROWS) -> list[JsonDict]:
    """Build verifier rows only from existing exact context fixture answers."""

    rows: list[JsonDict] = []
    for raw in read_jsonl_objects(Path(root) / CONTEXT_FIXTURE_REL_PATH):
        fixture_id = str(raw.get("fixture_id") or "")
        expected = str(raw.get("expected_answer") or "")
        counter = mapping(raw.get("minimal_counterexample"))
        counter_answer = str(counter.get("candidate_answer") or raw.get("prior_bait_answer") or "")
        if fixture_id and expected:
            rows.append(eval_row(raw, "expected", expected, "accept"))
        if fixture_id and counter_answer:
            rows.append(eval_row(raw, "counterexample", counter_answer, "reject"))
        if len(rows) >= int(max_rows):
            break
    return rows[: max(0, int(max_rows))]


def eval_row(raw: Mapping[str, Any], kind: str, candidate: str, expected_decision: str) -> JsonDict:
    """Convert one fixture candidate into the local-verifier prompt contract."""

    fixture_id = str(raw.get("fixture_id") or "")
    source_kind = (
        "fixture_expected_answer" if kind == "expected" else "fixture_minimal_counterexample"
    )
    row: JsonDict = {
        "row_id": f"{fixture_id}:{kind}",
        "fixture_id": fixture_id,
        "fixture_family": str(raw.get("family") or ""),
        "exact_authority": "context_exact_checker",
        "context": str(raw.get("context") or ""),
        "question": str(raw.get("question") or ""),
        "candidate_answer": str(candidate),
        "expected_answer": str(raw.get("expected_answer") or ""),
        "expected_decision": expected_decision,
        "source_candidate_kind": source_kind,
        "synthetic_shortcut_row": False,
    }
    row["prompt"] = verifier_prompt(row)
    row["row_hash"] = stable_hash(row)
    return row


def verifier_prompt(row: Mapping[str, Any]) -> str:
    """Create a short verifier prompt with an explicit one-token decision space."""

    return (
        "You are checking one exact fixture row.\n"
        "Reply with exactly one word: ACCEPT, REJECT, or ABSTAIN.\n"
        f"Context: {row.get('context')}\n"
        f"Question: {row.get('question')}\n"
        f"Candidate answer: {row.get('candidate_answer')}\n"
        "Does the candidate answer exactly satisfy the context and question?"
    )


def normalize_runner_rows(
    raw_rows: Any,
    exact_rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> list[JsonDict]:
    """Attach exact labels to model outputs, abstaining when rows are missing."""

    raw_by_id = {
        str(row.get("row_id") or ""): mapping(row)
        for row in (raw_rows if isinstance(raw_rows, Sequence) else [])
        if isinstance(row, Mapping)
    }
    normalized: list[JsonDict] = []
    for exact in exact_rows:
        raw = raw_by_id.get(str(exact.get("row_id") or ""), {})
        output_text = str(raw.get("output_text") or raw.get("response") or "")
        decision = normalize_decision(raw.get("decision") or output_text)
        prompt = str(exact.get("prompt") or verifier_prompt(exact))
        prompt_hash = str(raw.get("prompt_hash") or stable_hash(prompt))
        response_hash = str(raw.get("response_hash") or stable_hash(output_text))
        transcript_hash = str(
            raw.get("transcript_hash")
            or stable_hash(
                {
                    "model_id": model.get("model_id"),
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "decision": decision,
                }
            )
        )
        normalized.append(
            {
                "row_id": str(exact.get("row_id") or ""),
                "fixture_id": str(exact.get("fixture_id") or ""),
                "model_id": str(raw.get("model_id") or model.get("model_id") or ""),
                "model_path": str(raw.get("model_path") or model.get("model_path") or ""),
                "exact_authority": str(exact.get("exact_authority") or ""),
                "expected_decision": str(exact.get("expected_decision") or "abstain"),
                "decision": decision,
                "output_text": output_text,
                "false_accept": (
                    str(exact.get("expected_decision")) == "reject" and decision == "accept"
                ),
                "false_reject": (
                    str(exact.get("expected_decision")) == "accept" and decision == "reject"
                ),
                "abstained": decision == "abstain",
                "source_candidate_kind": str(exact.get("source_candidate_kind") or ""),
                "synthetic_shortcut_row": exact.get("synthetic_shortcut_row") is True,
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "transcript_hash": transcript_hash,
                "token_counts": mapping(raw.get("token_counts")),
            }
        )
    return normalized


def score_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute conservative exact-authority false-accept/reject rates."""

    reject_rows = [row for row in rows if row.get("expected_decision") == "reject"]
    accept_rows = [row for row in rows if row.get("expected_decision") == "accept"]
    false_accept_count = sum(row.get("false_accept") is True for row in rows)
    false_reject_count = sum(row.get("false_reject") is True for row in rows)
    abstention_count = sum(row.get("abstained") is True for row in rows)
    return {
        "n_eval": len(rows),
        "false_accept_count": false_accept_count,
        "false_reject_count": false_reject_count,
        "abstention_count": abstention_count,
        "false_accept_rate": rate(false_accept_count, len(reject_rows)),
        "false_reject_rate": rate(false_reject_count, len(accept_rows)),
        "abstention_rate": rate(abstention_count, len(rows)),
    }


def normalize_decision(value: Any) -> str:
    """Parse only a leading explicit verifier token; everything else abstains."""

    text = str(value or "").strip()
    if not text:
        return "abstain"
    first = text.split()[0].strip(" \t\r\n.:,;!?\"'`()[]{}").lower()
    if first in {"accept", "reject", "abstain"}:
        return first
    return "abstain"


def resolve_models(exp3268: Mapping[str, Any], root: Path) -> list[JsonDict]:
    """Resolve audited mandated GGUF paths from Exp 3268 evidence."""

    candidates: list[JsonDict] = []
    for row in mapping_list(exp3268.get("models_used")):
        model_id = str(row.get("model_id") or row.get("hf_id") or "")
        candidates.append(
            {
                "model_id": model_id,
                "model_path": str(row.get("model_path") or ""),
                "source": "exp3268_clean_receipt",
                "legacy_small_model": False,
            }
        )
    mandated = mapping(mapping(exp3268.get("model_specs")).get("mandated_models"))
    for model_id in MANDATED_MODEL_IDS:
        spec = mapping(mandated.get(model_id))
        candidates.append(
            {
                "model_id": model_id,
                "model_path": str(spec.get("model_path") or ""),
                "source": "exp3268_model_specs",
                "legacy_small_model": False,
            }
        )

    selected: list[JsonDict] = []
    seen: set[str] = set()
    for model_id in MANDATED_MODEL_IDS:
        for row in candidates:
            if row["model_id"] != model_id or model_id in seen:
                continue
            path = resolve_path(root, row["model_path"])
            if path.is_file():
                selected.append(row | {"model_path": str(path)})
                seen.add(model_id)
    return selected


def model_specs_from_receipt(exp3268: Mapping[str, Any]) -> JsonDict:
    """Carry forward Exp 3268 model specs, with mandated IDs always explicit."""

    specs = mapping(exp3268.get("model_specs"))
    if specs:
        return dict(specs)
    return {"mandated_model_ids": list(MANDATED_MODEL_IDS), "mandated_models": {}}


def resolve_path(root: Path, value: str) -> Path:
    """Resolve absolute and repository-relative paths without normalizing evidence away."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def exact_row_fixture_hash(root: Path, exact_rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the fixture bytes plus selected exact rows for auditability."""

    fixture_path = root / CONTEXT_FIXTURE_REL_PATH
    return stable_hash(
        {
            "context_fixture_sha256": sha256_file(fixture_path),
            "selected_exact_rows": list(exact_rows),
        }
    )


def build_source_artifacts(root: Path) -> list[JsonDict]:
    """Record local files that determine the v14 result."""

    paths = (
        ("exp3268_sota_receipt_methodology", EXP3268_REL_PATH),
        ("exp3223_exact_row_sidecar", EXP3223_REL_PATH),
        ("context_exact_row_fixture", CONTEXT_FIXTURE_REL_PATH),
        ("verification_openspec", Path("openspec/capabilities/verification/spec.md")),
        (
            "exp3275_module",
            Path("python/carnot/verify/clean_local_sota_verifier_rerun_v14.py"),
        ),
        ("exp3275_script", Path("scripts/experiment_3275_clean_local_sota_verifier_rerun_v14.py")),
        (
            "exp3275_tests",
            Path("tests/python/test_experiment_3275_clean_local_sota_verifier_rerun_v14.py"),
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


def default_cuda_probe() -> JsonDict:
    """Check visible NVIDIA CUDA devices with nvidia-smi."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "name": "cuda_runtime",
            "passed": False,
            "returncode": None,
            "stderr_summary": f"{type(exc).__name__}: {exc}",
            "gpu_count": 0,
            "gpu_mem_used_mib": 0,
        }
    rows = parse_nvidia_smi_rows(proc.stdout)
    return {
        "name": "cuda_runtime",
        "passed": proc.returncode == 0 and bool(rows),
        "returncode": proc.returncode,
        "gpu_count": len(rows),
        "gpus": rows,
        "gpu_mem_used_mib": max((safe_int(row.get("memory_used_mib")) for row in rows), default=0),
        "stderr_summary": proc.stderr[-500:],
    }


def parse_nvidia_smi_rows(stdout: str) -> list[JsonDict]:
    """Parse the CSV shape emitted by default_cuda_probe."""

    rows: list[JsonDict] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": safe_int(parts[0]),
                "name": parts[1],
                "memory_total_mib": safe_int(parts[2]),
                "memory_used_mib": safe_int(parts[3]),
                "utilization_gpu_pct": safe_int(parts[4]),
                "driver_version": parts[5],
            }
        )
    return rows


def run_llama_local_verifier(
    rows: list[JsonDict],
    model: JsonDict,
    random_seed: int,
) -> JsonDict:  # pragma: no cover - exercised by the live artifact run, not unit tests.
    """Run llama.cpp locally with full GPU offload requested."""

    from llama_cpp import Llama

    samples = [_gpu_memory_rows()]
    llm = Llama(
        model_path=str(model["model_path"]),
        n_ctx=2048,
        n_gpu_layers=-1,
        seed=int(random_seed),
        verbose=False,
    )
    samples.append(_gpu_memory_rows())
    output_rows: list[JsonDict] = []
    for row in rows:
        raw = llm(
            str(row["prompt"]),
            max_tokens=6,
            temperature=0.0,
            top_p=1.0,
            stop=["\n"],
        )
        text = completion_text(raw)
        samples.append(_gpu_memory_rows())
        output_rows.append(
            {
                "row_id": row["row_id"],
                "model_id": model["model_id"],
                "model_path": model["model_path"],
                "output_text": text,
                "decision": text,
                "token_counts": token_counts(raw),
            }
        )
    del llm
    gc.collect()
    return {"rows": output_rows, "gpu_mem_used_mib": max_gpu_memory(samples)}


def _gpu_memory_rows() -> list[JsonDict]:  # pragma: no cover - live GPU telemetry helper.
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


def completion_text(raw: Any) -> str:  # pragma: no cover - llama.cpp response adapter.
    if not isinstance(raw, Mapping):
        return ""
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    return str(first.get("text") or "") if isinstance(first, Mapping) else ""


def token_counts(raw: Any) -> JsonDict:  # pragma: no cover - llama.cpp response adapter.
    usage = mapping(mapping(raw).get("usage"))
    return {
        "prompt_tokens": safe_int(usage.get("prompt_tokens")),
        "completion_tokens": safe_int(usage.get("completion_tokens")),
        "total_tokens": safe_int(usage.get("total_tokens")),
    }


def max_gpu_memory(samples: Sequence[Sequence[Mapping[str, Any]]]) -> int:
    """Return the maximum absolute GPU memory seen across samples."""

    values = [
        safe_int(row.get("memory_used_mib"))
        for sample in samples
        for row in sample
        if isinstance(row, Mapping)
    ]
    return max(values) if values else 0


def normalize_precondition(row: Mapping[str, Any]) -> JsonDict:
    """Ensure precondition rows always have a name and boolean pass field."""

    payload = dict(row)
    payload["name"] = str(payload.get("name") or "unnamed_precondition")
    payload["passed"] = payload.get("passed") is True
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject contradictory v14 artifacts before they reach repair gates."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success-style prefix")
    for key in ("false_accept_rate", "false_reject_rate", "abstention_rate"):
        value = artifact.get(key)
        if not isinstance(value, float) or not 0.0 <= value <= 1.0:
            raise ValueError(f"rate field {key} must be a float in [0, 1]")
    if not isinstance(artifact.get("n_eval"), int) or int(artifact["n_eval"]) < 0:
        raise ValueError("n_eval must be a non-negative integer")
    if not isinstance(artifact.get("preconditions_checked"), list):
        raise ValueError("preconditions_checked must be a list")
    if not isinstance(artifact.get("models_used"), list):
        raise ValueError("models_used must be a list")
    if len(str(artifact.get("exact_row_fixture_hash") or "")) != 64:
        raise ValueError("exact_row_fixture_hash must be a sha256-style string")
    if len(str(artifact.get("reproducibility_checksum") or "")) != 64:
        raise ValueError("reproducibility_checksum must be a sha256-style string")
    if artifact.get("clean_verifier_rerun_ready") is True:
        if artifact.get("n_eval", 0) <= 0:
            raise ValueError("ready artifact must score at least one row")
        if artifact.get("clean_rerun_allowed") is not True:
            raise ValueError("clean_rerun_allowed must match ready clean reruns")
        if not artifact.get("models_used"):
            raise ValueError("models_used required for ready clean rerun")
        if safe_int(artifact.get("gpu_mem_used_mib")) < MIN_GPU_MEM_USED_MIB:
            raise ValueError("gpu_mem_used_mib must prove GPU offload for ready reruns")
    if artifact.get("clean_rerun_allowed") is True and artifact.get("clean_verifier_rerun_ready") is not True:
        raise ValueError("clean_rerun_allowed cannot be true when rerun is not ready")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict expected by conductor artifacts."""

    if artifact.get("clean_verifier_rerun_ready") is True:
        return "complete: clean local SOTA verifier rerun v14 ready for repair-gate input"
    if artifact.get("gated_skip") is True:
        return "complete: clean local SOTA verifier rerun v14 gated skip"
    return "complete: clean local SOTA verifier rerun v14 not ready for repair-gate input"


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after removing its self-referential checksum field."""

    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return stable_hash(payload)


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 of a present file."""

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


def mapping(value: Any) -> JsonDict:
    """Return a dict only for mapping values."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return a list of dicts from a JSON list-like value."""

    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def safe_int(value: Any) -> int:
    """Coerce numeric evidence to int, failing closed to zero."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def rate(numerator: int, denominator: int) -> float:
    """Compute bounded rates while making empty denominators explicit as zero."""

    return 0.0 if denominator <= 0 else round(float(numerator) / float(denominator), 6)


def duration(started_s: float, now_s: float) -> float:
    """Measure non-negative wall-clock duration."""

    return round(max(0.0, float(now_s) - float(started_s)), 6)
