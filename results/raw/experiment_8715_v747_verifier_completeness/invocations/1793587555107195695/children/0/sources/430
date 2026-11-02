"""Exp 3085 I-CALM task-abstention panel over exact ReSyn fixtures.

Spec refs: REQ-VERIFY-3085,
           SCENARIO-VERIFY-3085,
           SCENARIO-VERIFY-3085-BLOCKED.

The panel keeps two boundaries explicit. First, the model never sees exact
labels or authority payloads; prompts are built only from the fixture bank's
leakage-safe payloads. Second, exact labels are used only after generation to
score whether the model accepted a good case, rejected a bad case, or abstained.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf


JsonDict = dict[str, Any]
ResolveGgufFn = Callable[[str, str], str | None]
LlamaFactory = Callable[..., Any]
ClockFn = Callable[[], float]
RepoCommitFn = Callable[[Path], str]
ProbeFn = Callable[[], Mapping[str, Any]]

ARTIFACT = "experiment_3085_icalm_task_abstention_sota_panel_v2"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCRIPT_FILENAME = f"{ARTIFACT}.py"
SCHEMA = "carnot.icalm_task_abstention_sota_panel.v2"
RUN_DATE = "20260525"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
PANEL_ROWS_REL_PATH = Path("results") / "icalm_task_abstention_sota_panel_3085" / "rows.jsonl"
FIXTURE_MANIFEST_REL_PATH = Path("results/resyn_exact_fixture_bank_3084/fixture_manifest.jsonl")
EXP3070_REL_PATH = Path("results/experiment_3070_first_token_abstention_sota_panel_v1.json")
DEFAULT_SEED = 308500
DEFAULT_SAMPLE_PER_FAMILY = 6
DEFAULT_LOGPROBS = 5
DEFAULT_ABSTENTION_THRESHOLD = 0.70
DEFAULT_DECODE_CONFIG: JsonDict = {
    "max_tokens": 16,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 40,
    "repeat_penalty": 1.0,
    "stop": ["\n"],
}
DEFAULT_LOAD_CONFIG: JsonDict = {
    "n_ctx": 2048,
    "n_batch": 64,
    "n_ubatch": 64,
    "n_gpu_layers": -1,
    "main_gpu": 0,
    "logits_all": True,
    "verbose": False,
}
MANDATED_MODEL_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
PROMPT_POLICIES = ("baseline", "task_abstention")
REQUIRED_ARTIFACT_FIELDS = (
    "abstention_panel_v2_ready",
    "first_token_panel_ready",
    "abstention_precision",
    "rejection_recall",
    "abstention_coverage",
    "overacceptance_rate",
    "exact_ground_truth_count",
    "fixture_manifest_path",
    "models_used",
    "model_specs",
    "legacy_smoke_only_used",
    "preconditions_checked",
    "prompt_hashes",
    "inference_substrate",
    "honest_verdict",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and knobs for Exp 3085."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    rows_path: Path | None = None
    fixture_manifest_path: Path | None = None
    sample_per_family: int = DEFAULT_SAMPLE_PER_FAMILY
    seed: int = DEFAULT_SEED
    preferred_quant: str = "Q4_K_M"
    logprobs: int = DEFAULT_LOGPROBS
    abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD
    decode_config: Mapping[str, Any] | None = None
    load_config: Mapping[str, Any] | None = None
    tests_run: Sequence[str] = ()

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def panel_rows_path(self) -> Path:
        return self.rows_path or self.repo_root / PANEL_ROWS_REL_PATH

    def manifest_path(self) -> Path:
        return self.fixture_manifest_path or self.repo_root / FIXTURE_MANIFEST_REL_PATH

    def effective_decode_config(self) -> JsonDict:
        config = dict(DEFAULT_DECODE_CONFIG)
        if self.decode_config:
            config.update(dict(self.decode_config))
        return config

    def effective_load_config(self, gpu: int = 0) -> JsonDict:
        config = dict(DEFAULT_LOAD_CONFIG)
        if self.load_config:
            config.update(dict(self.load_config))
        config["main_gpu"] = int(gpu)
        return config


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    llama_factory: LlamaFactory | None = None,
    monotonic: ClockFn = time.monotonic,
    repo_commit_func: RepoCommitFn | None = None,
    cuda_probe_func: ProbeFn | None = None,
    gpu_inventory_func: ProbeFn | None = None,
    python_environment_func: ProbeFn | None = None,
) -> JsonDict:
    """Run the abstention panel and write the terminal artifact."""

    active = config or ExperimentConfig()
    commit_fn = repo_commit_func or _repo_commit
    cuda_fn = cuda_probe_func or _cuda_probe
    gpu_fn = gpu_inventory_func or _gpu_inventory
    python_env_fn = python_environment_func or _python_environment
    started = monotonic()
    fixture_status = _fixture_manifest_status(active.manifest_path())
    cuda_status = dict(cuda_fn())
    cache_resolution = _resolve_cache(resolve_gguf_func, active)
    selected = _selected_model(cache_resolution)
    preconditions = _preconditions(
        fixture_status=fixture_status,
        cuda_status=cuda_status,
        cache_resolution=cache_resolution,
        selected_model=selected,
        load_ok=False,
        load_detail="not_attempted",
    )
    runtime_blocker = _first_precondition_failure(preconditions)
    if runtime_blocker is not None:
        artifact = _blocked_artifact(
            config=active,
            duration_s=round(monotonic() - started, 6),
            runtime_blocker=runtime_blocker,
            preconditions_checked=preconditions,
            cache_resolution=cache_resolution,
            repo_commit_func=commit_fn,
            cuda_status=cuda_status,
            gpu_inventory_func=gpu_fn,
            python_environment_func=python_env_fn,
        )
        _write_json(active.artifact_path(), artifact)
        return artifact

    try:
        fixture_rows = sample_balanced_fixtures(
            load_fixture_manifest(active.manifest_path()),
            per_family=active.sample_per_family,
            seed=active.seed,
        )
        panel_rows = _run_live_panel(
            config=active,
            selected_model=selected or {},
            fixture_rows=fixture_rows,
            llama_factory=llama_factory or _default_llama_factory,
        )
        preconditions = _preconditions(
            fixture_status=fixture_status,
            cuda_status=cuda_status,
            cache_resolution=cache_resolution,
            selected_model=selected,
            load_ok=True,
            load_detail=str((selected or {}).get("model_path", "loaded")),
        )
        runtime_blocker = None
    except Exception as exc:
        panel_rows = []
        preconditions = _preconditions(
            fixture_status=fixture_status,
            cuda_status=cuda_status,
            cache_resolution=cache_resolution,
            selected_model=selected,
            load_ok=False,
            load_detail=f"{type(exc).__name__}: {exc}",
        )
        runtime_blocker = f"model_load_failed: {type(exc).__name__}: {exc}"

    duration_s = round(monotonic() - started, 6)
    if panel_rows:
        _write_jsonl(active.panel_rows_path(), panel_rows)
    artifact = _build_artifact(
        config=active,
        rows=panel_rows,
        selected_models=[selected] if selected and runtime_blocker is None else [],
        cache_resolution=cache_resolution,
        duration_s=duration_s,
        runtime_blocker=runtime_blocker,
        preconditions_checked=preconditions,
        repo_commit_func=commit_fn,
        cuda_status=cuda_status,
        gpu_inventory_func=gpu_fn,
        python_environment_func=python_env_fn,
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_fixture_manifest(path: Path) -> list[JsonDict]:
    """Load the Exp 3084 fixture manifest as JSONL."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load panel transcript rows."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_balanced_fixtures(
    rows: Sequence[Mapping[str, Any]], *, per_family: int, seed: int
) -> list[JsonDict]:
    """Return a deterministic balanced sample across fixture families."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    selected: list[Mapping[str, Any]] = []
    for family in sorted(grouped):
        family_rows = sorted(grouped[family], key=lambda row: str(row["fixture_id"]))
        limit = min(max(int(per_family), 0), len(family_rows))
        selected.extend(family_rows[:limit])
    return [dict(row) for row in sorted(selected, key=lambda row: str(row["fixture_id"]))]


def build_prompt(row: Mapping[str, Any], *, policy: str) -> str:
    """Build a leakage-safe prompt for one fixture row and policy."""

    payload = row["leakage_safe_prompt_payload"]
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    answers = ", ".join(_allowed_answers(row))
    if policy == "baseline":
        instruction = (
            "Reply with the first token as the answer. Use format: "
            "<ANSWER> | confidence=<0.00-1.00>."
        )
    else:
        instruction = (
            "Use an I-CALM abstention policy. Reply with first token exactly "
            "ACCEPT, REJECT, or ABSTAIN. ACCEPT only when the candidate or claim "
            "should be accepted. REJECT when the candidate, claim, or constraints "
            "should be rejected or repaired. ABSTAIN when confidence is below 0.70. "
            "Use format: <DECISION> | answer=<ANSWER> | confidence=<0.00-1.00>."
        )
    return (
        "Role: exact-fixture analyst\n"
        f"Policy: {policy}\n"
        f"Case Hash: {row['prompt_payload_sha256']}\n"
        f"Family: {row['family']}\n"
        f"Allowed Answers: {answers}\n"
        f"Task Payload: {payload_text}\n"
        f"{instruction}\n"
    )


def expected_answer_and_action(row: Mapping[str, Any]) -> JsonDict:
    """Return the exact post-generation answer/action target for scoring."""

    label = row["exact_label"]
    family = str(row["family"])
    if family == "smt_constraints":
        answer = "SAT" if bool(label.get("is_satisfiable")) else "UNSAT"
        action = "accept" if answer == "SAT" else "reject"
    elif family == "arithmetic_code_assertions":
        answer = "VALID" if bool(label.get("assertion_passes")) else "INVALID"
        action = "accept" if answer == "VALID" else "reject"
    elif family == "repairable_invalid_candidates":
        if bool(label.get("candidate_valid")):
            answer = "VALID"
            action = "accept"
        elif bool(label.get("repairable")):
            answer = "REPAIRABLE"
            action = "reject"
        else:
            answer = "UNREPAIRABLE"
            action = "reject"
    else:
        raise ValueError(f"unknown fixture family: {family}")
    return {"expected_answer": answer, "expected_action": action}


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that overstate Exp 3085 readiness."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("legacy_smoke_only_used") is not False:
        raise ValueError("legacy smoke evidence cannot satisfy REQ-VERIFY-3085")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("abstention_panel_v2_ready") is not True:
        if not verdict.startswith("blocked_sota_or_fixture_precondition_failed"):
            raise ValueError("honest_verdict must disclose blocked_sota_or_fixture_precondition_failed")
        return
    if not artifact.get("model_specs"):
        raise ValueError("model_specs must be present when the panel is ready")
    if int(artifact.get("exact_ground_truth_count") or 0) < 6:
        raise ValueError("exact_ground_truth_count must be at least 6 when ready")
    if not artifact.get("prompt_hashes"):
        raise ValueError("prompt_hashes must be non-empty when ready")
    if int(artifact.get("baseline_row_count") or 0) <= 0:
        raise ValueError("baseline_row_count must be non-zero when ready")
    if int(artifact.get("task_abstention_row_count") or 0) <= 0:
        raise ValueError("task_abstention_row_count must be non-zero when ready")
    if artifact.get("first_token_panel_ready") is True and _float(
        artifact.get("first_token_confidence_coverage")
    ) <= 0.0:
        raise ValueError("first_token_panel_ready requires first-token confidence coverage")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_live_panel(
    *,
    config: ExperimentConfig,
    selected_model: Mapping[str, Any],
    fixture_rows: Sequence[Mapping[str, Any]],
    llama_factory: LlamaFactory,
) -> list[JsonDict]:
    load_config = config.effective_load_config(int(selected_model.get("gpu", 0)))
    llm = llama_factory(model_path=str(selected_model["model_path"]), **load_config)
    rows: list[JsonDict] = []
    try:
        for fixture in fixture_rows:
            expected = expected_answer_and_action(fixture)
            for policy in PROMPT_POLICIES:
                prompt = build_prompt(fixture, policy=policy)
                raw = llm(
                    prompt,
                    **config.effective_decode_config(),
                    seed=config.seed,
                    logprobs=config.logprobs,
                )
                text = _normalize_output(_extract_text(raw))
                confidence = _confidence_from_output(raw)
                parsed = _parse_policy_response(text)
                first_token_confidence = (
                    _float(confidence.get("confidence_score"))
                    if confidence.get("confidence_available")
                    else None
                )
                if policy == "baseline":
                    decision = _baseline_decision(parsed["answer"])
                else:
                    decision = _derive_decision(
                        parsed["raw_action"],
                        parsed["answer"],
                        parsed["verbal_confidence"],
                        first_token_confidence,
                        config.abstention_threshold,
                    )
                answer_correct = _answer_matches(parsed["answer"], expected["expected_answer"])
                rows.append(
                    {
                        "fixture_id": fixture["fixture_id"],
                        "case_hash": fixture["prompt_payload_sha256"],
                        "family": fixture["family"],
                        "task_axis": fixture["task_axis"],
                        "perturbation_family": fixture["perturbation_family"],
                        "policy": policy,
                        "model_id": selected_model["hf_id"],
                        "model_name": selected_model["name"],
                        "prompt_hash": _sha256_text(prompt),
                        "raw_output_hash": _sha256_text(text),
                        "decision_text": text[:120],
                        "raw_action": parsed["raw_action"],
                        "answer": parsed["answer"],
                        "verbal_confidence": parsed["verbal_confidence"],
                        "decision": decision,
                        "expected_answer": expected["expected_answer"],
                        "expected_action": expected["expected_action"],
                        "expected_reject": expected["expected_action"] == "reject",
                        "answer_correct": answer_correct,
                        "decision_correct": _decision_matches_expected(
                            decision, expected["expected_action"]
                        ),
                        **confidence,
                    }
                )
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    return rows


def _build_artifact(
    *,
    config: ExperimentConfig,
    rows: Sequence[Mapping[str, Any]],
    selected_models: Sequence[Mapping[str, Any]],
    cache_resolution: Mapping[str, str | None],
    duration_s: float,
    runtime_blocker: str | None,
    preconditions_checked: Mapping[str, Any],
    repo_commit_func: RepoCommitFn,
    cuda_status: Mapping[str, Any],
    gpu_inventory_func: ProbeFn,
    python_environment_func: ProbeFn,
) -> JsonDict:
    metrics = _metrics(rows)
    model_specs = [_model_spec(model) for model in selected_models]
    models_used = [str(model["hf_id"]) for model in selected_models]
    first_token_ready = metrics["first_token_confidence_coverage"] > 0.0
    ready = (
        runtime_blocker is None
        and bool(model_specs)
        and metrics["exact_ground_truth_count"] >= 6
        and metrics["baseline_row_count"] > 0
        and metrics["task_abstention_row_count"] > 0
    )
    panel_path = config.panel_rows_path()
    prompt_hashes = [str(row["prompt_hash"]) for row in rows]
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "abstention_panel_v2_ready": ready,
        "first_token_panel_ready": ready and first_token_ready,
        "abstention_precision": metrics["abstention_precision"],
        "rejection_recall": metrics["rejection_recall"],
        "abstention_coverage": metrics["abstention_coverage"],
        "overacceptance_rate": metrics["overacceptance_rate"],
        "exact_ground_truth_count": metrics["exact_ground_truth_count"] if ready else 0,
        "fixture_manifest_path": _relative_path(config.repo_root, config.manifest_path()),
        "fixture_manifest_sha256": sha256_file(config.manifest_path())
        if config.manifest_path().is_file()
        else None,
        "models_used": models_used,
        "model_specs": model_specs,
        "legacy_smoke_only_used": False,
        "preconditions_checked": dict(preconditions_checked),
        "prompt_hashes": prompt_hashes if ready else [],
        "prompt_hash_count": len(prompt_hashes) if ready else 0,
        "inference_substrate": _substrate(
            config=config,
            cache_resolution=cache_resolution,
            selected_models=selected_models,
            duration_s=duration_s,
            cuda_status=cuda_status,
            rows=rows,
            repo_commit_func=repo_commit_func,
            gpu_inventory_func=gpu_inventory_func,
            python_environment_func=python_environment_func,
        ),
        "honest_verdict": _honest_verdict(ready, metrics, runtime_blocker),
        "accepted_count": metrics["accepted_count"],
        "rejected_count": metrics["rejected_count"],
        "abstained_count": metrics["abstained_count"],
        "baseline_row_count": metrics["baseline_row_count"],
        "task_abstention_row_count": metrics["task_abstention_row_count"],
        "baseline_answer_accuracy": metrics["baseline_answer_accuracy"],
        "task_answer_accuracy": metrics["task_answer_accuracy"],
        "baseline_overacceptance_rate": metrics["baseline_overacceptance_rate"],
        "verbal_confidence_coverage": metrics["verbal_confidence_coverage"],
        "first_token_confidence_coverage": metrics["first_token_confidence_coverage"],
        "abstention_precision_reaches_0_7": metrics["abstention_precision"] >= 0.7,
        "exp3070_comparison": _exp3070_comparison(config.repo_root, metrics["abstention_precision"]),
        "panel_rows_path": _relative_path(config.repo_root, panel_path),
        "panel_row_count": len(rows),
        "panel_rows_sha256": sha256_file(panel_path) if rows and panel_path.is_file() else "",
        "prompt_policies": list(PROMPT_POLICIES),
        "decode_config": config.effective_decode_config(),
        "load_config": config.effective_load_config(),
        "seed": config.seed,
        "duration_s": duration_s,
        "runtime_blocker": runtime_blocker,
        "tests_or_checks_run": list(config.tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "models_used": artifact["models_used"],
            "prompt_hashes": artifact["prompt_hashes"],
            "metrics": {
                "abstention_precision": artifact["abstention_precision"],
                "rejection_recall": artifact["rejection_recall"],
                "abstention_coverage": artifact["abstention_coverage"],
                "overacceptance_rate": artifact["overacceptance_rate"],
            },
        }
    )
    return artifact


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    runtime_blocker: str,
    preconditions_checked: Mapping[str, Any],
    cache_resolution: Mapping[str, str | None],
    repo_commit_func: RepoCommitFn,
    cuda_status: Mapping[str, Any],
    gpu_inventory_func: ProbeFn,
    python_environment_func: ProbeFn,
) -> JsonDict:
    artifact = _build_artifact(
        config=config,
        rows=[],
        selected_models=[],
        cache_resolution=cache_resolution,
        duration_s=duration_s,
        runtime_blocker=runtime_blocker,
        preconditions_checked=preconditions_checked,
        repo_commit_func=repo_commit_func,
        cuda_status=cuda_status,
        gpu_inventory_func=gpu_inventory_func,
        python_environment_func=python_environment_func,
    )
    validate_artifact(artifact)
    return artifact


def _metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    baseline = [row for row in rows if row.get("policy") == "baseline"]
    task = [row for row in rows if row.get("policy") == "task_abstention"]
    accepted = [row for row in task if row.get("decision") == "accept"]
    rejected = [row for row in task if row.get("decision") == "reject"]
    abstained = [row for row in task if row.get("decision") == "abstain"]
    expected_reject = [row for row in task if row.get("expected_reject")]
    accepted_correct = [
        row
        for row in accepted
        if row.get("answer_correct") is True and row.get("expected_action") == "accept"
    ]
    rejected_bad = [row for row in rejected if row.get("expected_reject")]
    overaccepted = [row for row in accepted if row.get("expected_reject")]
    baseline_overaccepted = [
        row
        for row in baseline
        if row.get("decision") == "accept" and row.get("expected_action") == "reject"
    ]
    confidence_rows = [row for row in rows if row.get("confidence_available")]
    verbal_rows = [row for row in rows if row.get("verbal_confidence") is not None]
    return {
        "exact_ground_truth_count": len({row["fixture_id"] for row in task}),
        "baseline_row_count": len(baseline),
        "task_abstention_row_count": len(task),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "abstained_count": len(abstained),
        "abstention_precision": round(len(accepted_correct) / len(accepted), 6)
        if accepted
        else 0.0,
        "rejection_recall": round(len(rejected_bad) / len(expected_reject), 6)
        if expected_reject
        else 0.0,
        "abstention_coverage": round(len(abstained) / len(task), 6) if task else 0.0,
        "overacceptance_rate": round(len(overaccepted) / len(expected_reject), 6)
        if expected_reject
        else 0.0,
        "baseline_overacceptance_rate": round(
            len(baseline_overaccepted)
            / len([row for row in baseline if row.get("expected_action") == "reject"]),
            6,
        )
        if [row for row in baseline if row.get("expected_action") == "reject"]
        else 0.0,
        "baseline_answer_accuracy": round(
            len([row for row in baseline if row.get("answer_correct")]) / len(baseline), 6
        )
        if baseline
        else 0.0,
        "task_answer_accuracy": round(
            len([row for row in task if row.get("answer_correct")]) / len(task), 6
        )
        if task
        else 0.0,
        "first_token_confidence_coverage": round(len(confidence_rows) / len(rows), 6)
        if rows
        else 0.0,
        "verbal_confidence_coverage": round(len(verbal_rows) / len(rows), 6) if rows else 0.0,
    }


def _empty_metrics() -> JsonDict:
    return _metrics([])


def _parse_policy_response(text: str) -> JsonDict:
    upper = text.upper()
    raw_action = None
    action_match = re.search(r"\b(ACCEPT|REJECT|ABSTAIN)\b", upper)
    if action_match:
        raw_action = action_match.group(1).lower()
    answer = None
    answer_match = re.search(
        r"\bANSWER\s*[:=]\s*(UNREPAIRABLE|REPAIRABLE|INVALID|VALID|UNSAT|SAT|UNKNOWN)\b",
        upper,
    )
    if answer_match:
        answer = answer_match.group(1)
    else:
        for candidate in ("UNREPAIRABLE", "REPAIRABLE", "INVALID", "VALID", "UNSAT", "SAT"):
            if re.search(rf"\b{candidate}\b", upper):
                answer = candidate
                break
    return {
        "raw_action": raw_action,
        "answer": answer,
        "verbal_confidence": _verbal_confidence(text),
    }


def _verbal_confidence(text: str) -> float | None:
    lowered = text.lower()
    numeric = re.search(r"confidence\s*[:=]?\s*(-?\d+(?:\.\d+)?)", lowered)
    if numeric:
        return max(0.0, min(1.0, _float(numeric.group(1))))
    if "high confidence" in lowered or "confidence high" in lowered:
        return 0.9
    if "medium confidence" in lowered or "confidence medium" in lowered:
        return 0.5
    if "low confidence" in lowered or "confidence low" in lowered:
        return 0.2
    return None


def _derive_decision(
    raw_action: str | None,
    answer: str | None,
    verbal_confidence: float | None,
    first_token_confidence: float | None,
    threshold: float,
) -> str:
    decision = raw_action or _baseline_decision(answer)
    confidence_values = [
        value for value in (verbal_confidence, first_token_confidence) if value is not None
    ]
    if confidence_values and min(confidence_values) < threshold:
        return "abstain"
    return decision


def _baseline_decision(answer: str | None) -> str:
    if answer in {"SAT", "VALID"}:
        return "accept"
    if answer in {"UNSAT", "INVALID", "REPAIRABLE", "UNREPAIRABLE"}:
        return "reject"
    return "abstain"


def _answer_matches(answer: str | None, expected: str) -> bool:
    return str(answer or "").strip().upper().rstrip(".") == expected


def _decision_matches_expected(decision: str, expected_action: str) -> bool:
    return decision == expected_action


def _allowed_answers(row: Mapping[str, Any]) -> tuple[str, ...]:
    family = str(row["family"])
    if family == "smt_constraints":
        return ("SAT", "UNSAT")
    if family == "arithmetic_code_assertions":
        return ("VALID", "INVALID")
    if family == "repairable_invalid_candidates":
        return ("VALID", "REPAIRABLE", "UNREPAIRABLE")
    return ("UNKNOWN",)


def _fixture_manifest_status(path: Path) -> JsonDict:
    if not path.is_file():
        return {"ok": False, "path": str(path), "row_count": 0, "detail": "manifest_missing"}
    try:
        rows = load_fixture_manifest(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "path": str(path), "row_count": 0, "detail": str(exc)}
    return {
        "ok": len(rows) > 0,
        "path": str(path),
        "row_count": len(rows),
        "detail": "readable",
    }


def _preconditions(
    *,
    fixture_status: Mapping[str, Any],
    cuda_status: Mapping[str, Any],
    cache_resolution: Mapping[str, str | None],
    selected_model: Mapping[str, Any] | None,
    load_ok: bool,
    load_detail: str,
) -> JsonDict:
    cuda_ok = bool(cuda_status.get("cuda_available")) and int(cuda_status.get("gpu_count") or 0) > 0
    return {
        "fixture_manifest": dict(fixture_status),
        "cuda_gpu": {"ok": cuda_ok, "detail": dict(cuda_status)},
        "gguf_cache": {
            "ok": selected_model is not None,
            "detail": dict(cache_resolution),
        },
        "selected_model_load": {
            "ok": bool(load_ok),
            "detail": load_detail,
            "hf_id": selected_model.get("hf_id") if selected_model else None,
        },
    }


def _first_precondition_failure(preconditions: Mapping[str, Any]) -> str | None:
    if not preconditions["fixture_manifest"]["ok"]:
        return "fixture_manifest_missing_or_unreadable"
    if not preconditions["cuda_gpu"]["ok"]:
        return "cuda_gpu_unavailable"
    if not preconditions["gguf_cache"]["ok"]:
        return "no_mandated_gguf_resolved"
    return None


def _exp3070_comparison(repo_root: Path, abstention_precision: float) -> JsonDict:
    path = repo_root / EXP3070_REL_PATH
    if not path.is_file():
        path = REPO_ROOT / EXP3070_REL_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    prior = _float(payload.get("abstention_precision"))
    return {
        "source_path": _relative_path(repo_root, path),
        "exp3070_first_token_panel_ready": bool(payload.get("first_token_panel_ready")),
        "exp3070_abstention_precision": prior,
        "exp3070_rejection_recall": _float(payload.get("rejection_recall")),
        "exp3070_abstention_coverage": _float(payload.get("abstention_coverage")),
        "abstention_precision_delta": round(abstention_precision - prior, 6),
        "reaches_0_7_gate": abstention_precision >= 0.7,
    }


def _substrate(
    *,
    config: ExperimentConfig,
    cache_resolution: Mapping[str, str | None],
    selected_models: Sequence[Mapping[str, Any]],
    duration_s: float,
    cuda_status: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    repo_commit_func: RepoCommitFn,
    gpu_inventory_func: ProbeFn,
    python_environment_func: ProbeFn,
) -> JsonDict:
    return {
        "cuda_probe": dict(cuda_status),
        "gpu_inventory": dict(gpu_inventory_func()),
        "python_environment": dict(python_environment_func()),
        "repo_commit": repo_commit_func(config.repo_root),
        "runtime": "llama_cpp",
        "gguf_cache_resolution": dict(cache_resolution),
        "model_ids": list(MANDATED_MODEL_IDS),
        "selected_model_paths": [str(model["model_path"]) for model in selected_models],
        "quantization": config.preferred_quant,
        "seed": config.seed,
        "decode_config": config.effective_decode_config(),
        "load_config": config.effective_load_config(),
        "logprobs_requested": config.logprobs,
        "prompt_policies": list(PROMPT_POLICIES),
        "fixture_manifest_path": _relative_path(config.repo_root, config.manifest_path()),
        "confidence_support": {
            "first_token_available": any(row.get("confidence_available") for row in rows),
            "top_logprobs": any(
                row.get("confidence_signal") == "first_token_topk_entropy" for row in rows
            ),
            "chosen_token_logprob": any(row.get("first_token_logprob") is not None for row in rows),
            "verbal_confidence": any(row.get("verbal_confidence") is not None for row in rows),
        },
        "wall_clock_duration_s": duration_s,
    }


def _honest_verdict(
    ready: bool,
    metrics: Mapping[str, Any],
    runtime_blocker: str | None,
) -> str:
    if ready:
        prefix = "complete:" if metrics["abstention_precision"] >= 0.7 else "complete_below_gate:"
        return (
            f"{prefix} abstention_panel_v2_ready=true; "
            f"abstention_precision={metrics['abstention_precision']}; "
            f"rejection_recall={metrics['rejection_recall']}; "
            f"abstention_coverage={metrics['abstention_coverage']}; "
            f"overacceptance_rate={metrics['overacceptance_rate']}"
        )
    return f"blocked_sota_or_fixture_precondition_failed: {runtime_blocker or 'panel_metrics_vacuous'}"


def _resolve_cache(
    resolve_gguf_func: ResolveGgufFn, config: ExperimentConfig
) -> dict[str, str | None]:
    return {hf_id: resolve_gguf_func(hf_id, config.preferred_quant) for hf_id in MANDATED_MODEL_IDS}


def _selected_model(cache_resolution: Mapping[str, str | None]) -> JsonDict | None:
    for index, model in enumerate(SOTA_GGUF_MODELS):
        path = cache_resolution.get(model["hf_id"])
        if path:
            return {
                "name": model["name"],
                "hf_id": model["hf_id"],
                "model_path": path,
                "gpu": min(index, 1),
                "role": model["role"],
                "quantization": model["quantization"],
                "family": _model_family(model["hf_id"]),
            }
    return None


def _model_spec(model: Mapping[str, Any]) -> JsonDict:
    evidence = _file_evidence(str(model["model_path"]), full_limit_bytes=512 * 1024 * 1024)
    return {
        "name": model["name"],
        "hf_id": model["hf_id"],
        "model_path": model["model_path"],
        "gpu": model["gpu"],
        "role": model["role"],
        "family": model["family"],
        "quantization": model["quantization"],
        "model_hash_or_cache_path": evidence.get("model_hash_or_cache_path"),
        "checksum_feasibility": {
            "method": evidence.get("method"),
            "full_sha256_feasible": bool(evidence.get("full_sha256_feasible")),
            "size_bytes": evidence.get("size_bytes"),
        },
    }


def _model_family(hf_id: str) -> str:
    lowered = hf_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return hf_id.split("/", 1)[0].lower()


def _confidence_from_output(output: Mapping[str, Any]) -> JsonDict:
    choice = _first_choice(output)
    logprobs = choice.get("logprobs") if isinstance(choice, Mapping) else None
    if not isinstance(logprobs, Mapping):
        return _missing_confidence()
    token_logprobs = _float_list(logprobs.get("token_logprobs"))
    top_logprobs = logprobs.get("top_logprobs")
    tokens = [str(token) for token in logprobs.get("tokens", [])]
    index = _first_content_index(tokens, token_logprobs)
    topk = top_logprobs[index] if isinstance(top_logprobs, list) and index < len(top_logprobs) else None
    topk_confidence = _topk_entropy_confidence(topk if isinstance(topk, Mapping) else {})
    token = tokens[index].strip() if tokens and index < len(tokens) else ""
    if topk_confidence["confidence_available"]:
        return topk_confidence | {
            "first_token": token or topk_confidence["first_token"],
            "first_token_logprob": token_logprobs[index] if index < len(token_logprobs) else None,
        }
    if index < len(token_logprobs):
        probability = max(0.0, min(1.0, math.exp(token_logprobs[index])))
        return {
            "confidence_available": True,
            "confidence_signal": "first_token_logprob_proxy",
            "confidence_score": round(probability, 6),
            "first_token_entropy": None,
            "first_token": token,
            "first_token_logprob": token_logprobs[index],
            "first_token_top_logprobs": {},
            "confidence_limitation": "top_logprobs_unavailable; using chosen-token probability proxy",
        }
    return _missing_confidence()


def _first_choice(output: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = output.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return choices[0]
    return {}


def _first_content_index(tokens: Sequence[str], token_logprobs: Sequence[float]) -> int:
    for index, token in enumerate(tokens):
        if token.strip():
            return index
    return 0 if token_logprobs or tokens else 0


def _topk_entropy_confidence(top_logprobs: Mapping[str, Any]) -> JsonDict:
    values: list[tuple[str, float]] = []
    for token, raw in top_logprobs.items():
        try:
            values.append((str(token), float(raw)))
        except (TypeError, ValueError):
            continue
    if not values:
        return _missing_confidence()
    max_logprob = max(value for _token, value in values)
    weights = [math.exp(value - max_logprob) for _token, value in values]
    total = sum(weights)
    probs = [weight / total for weight in weights]
    entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0.0)
    normalizer = math.log(len(probs)) if len(probs) > 1 else 1.0
    confidence = max(0.0, min(1.0, 1.0 - (entropy / normalizer if normalizer else 0.0)))
    first_token = max(values, key=lambda item: item[1])[0].strip()
    return {
        "confidence_available": True,
        "confidence_signal": "first_token_topk_entropy",
        "confidence_score": round(confidence, 6),
        "first_token_entropy": round(1.0 - confidence, 6),
        "first_token": first_token,
        "first_token_top_logprobs": {token: value for token, value in values},
        "confidence_limitation": None,
    }


def _missing_confidence() -> JsonDict:
    return {
        "confidence_available": False,
        "confidence_signal": "unavailable",
        "confidence_score": 0.0,
        "first_token_entropy": None,
        "first_token": "",
        "first_token_logprob": None,
        "first_token_top_logprobs": {},
        "confidence_limitation": "no_first_token_logprob_or_topk_logprobs",
    }


def _extract_text(output: Mapping[str, Any]) -> str:
    choice = _first_choice(output)
    return str(choice.get("text", "")) if isinstance(choice, Mapping) else ""


def _normalize_output(text: str) -> str:
    return " ".join(text.strip().split())


def _file_evidence(path_text: str, *, full_limit_bytes: int) -> JsonDict:
    path = Path(path_text)
    size = path.stat().st_size
    if size <= full_limit_bytes:
        return {
            "model_hash_or_cache_path": f"sha256:{sha256_file(path)}",
            "method": "full_sha256",
            "full_sha256_feasible": True,
            "size_bytes": size,
        }
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(1024 * 1024)
        handle.seek(max(0, size - 1024 * 1024))
        tail = handle.read(1024 * 1024)
    digest.update(head)
    digest.update(tail)
    return {
        "model_hash_or_cache_path": f"bounded_sha256:{digest.hexdigest()}",
        "method": "bounded_head_tail_sha256",
        "full_sha256_feasible": False,
        "size_bytes": size,
    }


def _cuda_probe() -> JsonDict:  # pragma: no cover - live hardware probe.
    try:
        import torch  # noqa: PLC0415

        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_count": int(torch.cuda.device_count()),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
        }
    except Exception as exc:
        return {"cuda_available": False, "gpu_count": 0, "detail": f"{type(exc).__name__}: {exc}"}


def _gpu_inventory() -> JsonDict:  # pragma: no cover - live hardware probe.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "gpus": [], "detail": f"{type(exc).__name__}: {exc}"}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]),
                    "driver_version": parts[5],
                }
            )
    return {"available": result.returncode == 0, "gpus": gpus}


def _python_environment() -> JsonDict:  # pragma: no cover - environment metadata.
    return {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "virtual_env": sys.prefix,
    }


def _repo_commit(repo_root: Path) -> str:  # pragma: no cover - git metadata.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - live hardware path.
    from llama_cpp import Llama  # noqa: PLC0415

    return Llama(**kwargs)


def _prompt_payload_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    output: list[float] = []
    for raw in value:
        try:
            output.append(float(raw))
        except (TypeError, ValueError):
            continue
    return output
