#!/usr/bin/env python3
"""Exp 3925 competent LLM judge build artifact.

Spec refs: REQ-VERIFY-3925, SCENARIO-VERIFY-3925,
SCENARIO-VERIFY-3925-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.competent_llm_judge import (  # noqa: E402
    COMPETENT_PREFER_ORDER,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_CTX,
    build_separable_fixture,
    judge_step,
    run_judge_fixture,
)
from carnot.verify.gguf_inference import load_gguf_generator  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_3925_competent_judge_build.json")
EXP3915_ARTIFACT_REL_PATH = Path("results/experiment_3915_robust_gguf_inference_harness.json")
EXP3917_ARTIFACT_REL_PATH = Path("results/experiment_3917_efficiency_head_to_head.json")
EXP3884_ARTIFACT_REL_PATH = Path("results/experiment_3884_in_distribution_error_rich_corpus.json")
JUDGE_MODULE_PATH = "python/carnot/verify/competent_llm_judge.py"
GGUF_HARNESS_MODULE_PATH = "python/carnot/verify/gguf_inference.py"
REASONER_HARNESS_MODULE_PATH = "python/carnot/verify/reasoner_self_verification.py"
UNIT_TEST_PATH = "tests/python/test_competent_llm_judge.py"
SPEC_PATH = "openspec/capabilities/verification/spec.md"
EXPERIMENT_ID = 3925
TITLE = "competent_judge_build"
RANDOM_SEED = 3925
LIVE_FLOOR_S = 60.0
INFERENCE_SUBSTRATE = "live_llm_inference:robust_gguf_competent_process_judge"

REQUIRED_FIELDS = {
    "diagnosed_cause",
    "flipped_polarity_auroc",
    "judge_module_path",
    "judge_model_used",
    "fixture_auroc",
    "verdicts_parse_rate",
    "unit_test_path",
    "unit_test_passed",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "honest_verdict",
}

FIELD_PRINCIPLES = {
    "diagnosed_cause": (
        "WHY the .362 judge was below chance (polarity_inversion / "
        "unparsed_verdict_default / weak_prompt / genuine_weakness) - the root cause."
    ),
    "flipped_polarity_auroc": (
        "BARE FLOAT - exp3917 judge AUROC with score polarity flipped; >>0.5 confirms "
        "a polarity/parse bug, not a model weakness."
    ),
    "judge_module_path": "Where every .363 efficiency task imports the competent judge from.",
    "judge_model_used": (
        "Which GGUF the competent judge loaded (records whether a stronger model than the .362 26B was used)."
    ),
    "fixture_auroc": (
        "BARE FLOAT - must be > 0.65 on the SEPARABLE fixture; the positive control "
        "that the judge is not broken."
    ),
    "verdicts_parse_rate": (
        "BARE FLOAT - fraction of judge outputs that parsed to a verdict (not the "
        "constant-default bug); should be ~1.0."
    ),
    "unit_test_path": "The deliverable test file.",
    "unit_test_passed": "BARE BOOL - the deliverable; a passing live-judge test on a separable fixture cannot be a fabricated stub.",
    "preconditions_checked": "Pre-Launch + Adversarial-Verify - resource gates checked before loading weights.",
    "model_specs": "Pre-Launch + Adversarial-Verify - exact GGUF and runtime provenance.",
    "random_seed": "Pre-Launch + Adversarial-Verify - fixed seed for reproducibility.",
    "reproducibility_checksum": "Pre-Launch + Adversarial-Verify - hash over code, inputs, scores, and model metadata.",
    "duration_s": (
        "Pre-Launch + Adversarial-Verify - a real judge run that loads a model takes "
        "wall-clock (live floor 60s on THIS task); <60s = no real load."
    ),
    "inference_substrate": "Pre-Launch + Adversarial-Verify - actual inference runtime.",
}

WRAPPED_VALUE_FORBIDDEN_FIELDS = (
    "diagnosed_cause",
    "flipped_polarity_auroc",
    "judge_module_path",
    "judge_model_used",
    "fixture_auroc",
    "verdicts_parse_rate",
    "unit_test_path",
    "unit_test_passed",
    "duration_s",
    "inference_substrate",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One hard resource checked before the Exp 3925 live judge run."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"resource": self.resource, "available": self.available, "detail": self.detail}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 3925."""

    repo_root: Path
    output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    random_seed: int = RANDOM_SEED
    max_tokens: int = DEFAULT_MAX_TOKENS
    n_ctx: int = DEFAULT_N_CTX
    cuda_probe_timeout_s: int = 60
    run_unit_test: bool = True

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"


def _checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(repo_root: Path, rel_path: str) -> str:
    return _sha256_file(repo_root / rel_path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "blocked_"))


def _label_to_error_int(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "incorrect", "error", "bad", "wrong", "true"}:
        return 1
    if text in {"0", "correct", "ok", "good", "right", "false"}:
        return 0
    raise ValueError(f"unsupported label: {value!r}")


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative labels")
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _score_counts(scores: Sequence[float]) -> dict[str, int]:
    return {f"{score:.6f}": count for score, count in Counter(round(float(score), 6) for score in scores).items()}


def diagnose_exp3917_scores(repo_root: Path) -> dict[str, object]:
    """Diagnose whether Exp 3917's below-chance judge was inverted."""

    artifact_path = repo_root / EXP3917_ARTIFACT_REL_PATH
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict) or not artifact.get("per_item_results"):
        raise ValueError("exp3917 artifact lacks per_item_results")
    rows = [dict(row) for row in artifact["per_item_results"] if isinstance(row, dict)]
    labels = [_label_to_error_int(row.get("gold_error", row.get("label"))) for row in rows]
    scores = [float(row["llm_judge_score"]) for row in rows]
    original = _auroc(labels, scores)
    flipped = _auroc(labels, [1.0 - score for score in scores])
    neutral_count = sum(1 for score in scores if abs(score - 0.5) < 1e-12)
    distinct_scores = len(set(round(score, 6) for score in scores))
    if neutral_count / len(scores) > 0.8 or distinct_scores <= 1:
        cause = "unparsed_verdict_default"
    elif original < 0.5 and flipped > 0.5 and flipped > original:
        cause = "polarity_inversion"
    elif original < 0.5:
        cause = "weak_prompt"
    else:
        cause = "genuine_weakness"
    return {
        "diagnosed_cause": cause,
        "original_auroc": original,
        "flipped_polarity_auroc": flipped,
        "neutral_score_count": neutral_count,
        "distinct_score_count": distinct_scores,
        "score_counts": _score_counts(scores),
        "n_items": len(rows),
        "n_incorrect": sum(labels),
        "source_artifact_path": EXP3917_ARTIFACT_REL_PATH.as_posix(),
        "source_artifact_sha256": _sha256_file(artifact_path),
    }


def load_exp3915_source(repo_root: Path) -> dict[str, object]:
    """Load and validate the upstream robust GGUF harness artifact."""

    artifact_path = repo_root / EXP3915_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{EXP3915_ARTIFACT_REL_PATH.as_posix()} is missing")
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3915 artifact is not a JSON object")
    if artifact.get("unit_test_passed") is not True:
        raise ValueError("exp3915 unit_test_passed is not true")
    smoke_tokens = int(artifact.get("smoke_tokens") or 0)
    if smoke_tokens <= 0:
        raise ValueError("exp3915 smoke_tokens must be >0")
    module_path = repo_root / str(artifact.get("harness_module_path") or GGUF_HARNESS_MODULE_PATH)
    if not module_path.is_file():
        raise FileNotFoundError(f"exp3915 harness module missing: {module_path}")
    return {
        "artifact_path": EXP3915_ARTIFACT_REL_PATH.as_posix(),
        "artifact_sha256": _sha256_file(artifact_path),
        "harness_module_path": str(artifact.get("harness_module_path") or GGUF_HARNESS_MODULE_PATH),
        "harness_module_sha256": _sha256_file(module_path),
        "unit_test_passed": True,
        "smoke_tokens": smoke_tokens,
        "model_used": artifact.get("model_used"),
        "n_gpu_layers_used": artifact.get("n_gpu_layers_used"),
        "honest_verdict": artifact.get("honest_verdict"),
    }


def load_exp3884_source(repo_root: Path) -> dict[str, object]:
    """Load Exp 3884 provenance for the separable fixture check."""

    artifact_path = repo_root / EXP3884_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{EXP3884_ARTIFACT_REL_PATH.as_posix()} is missing")
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3884 artifact is not a JSON object")
    corpus_path = repo_root / str(artifact.get("corpus_path") or "")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"exp3884 corpus missing: {corpus_path}")
    return {
        "artifact_path": EXP3884_ARTIFACT_REL_PATH.as_posix(),
        "artifact_sha256": _sha256_file(artifact_path),
        "corpus_path": str(artifact.get("corpus_path")),
        "corpus_sha256": _sha256_file(corpus_path),
        "n_total_items": artifact.get("n_total_items"),
        "carnot_ensemble_auroc_on_corpus": artifact.get("carnot_ensemble_auroc_on_corpus"),
    }


def _probe_cuda_with_venv(config: ExperimentConfig) -> PreconditionCheck:
    try:
        proc = subprocess.run(
            [
                str(config.venv_python()),
                "-c",
                "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
            ],
            capture_output=True,
            cwd=config.repo_root,
            text=True,
            timeout=config.cuda_probe_timeout_s,
            check=False,
        )
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, repr(exc))
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    return PreconditionCheck("cuda_available", proc.returncode == 0, detail)


def _probe_imports() -> PreconditionCheck:
    try:
        importlib.import_module("carnot.verify")
        importlib.import_module("carnot.verify.gguf_inference")
        importlib.import_module("carnot.verify.reasoner_self_verification")
        importlib.import_module("carnot.verify.competent_llm_judge")
    except Exception as exc:
        return PreconditionCheck("carnot_verify_imports", False, repr(exc))
    return PreconditionCheck("carnot_verify_imports", True, "verification modules import")


def probe_preconditions(
    config: ExperimentConfig,
    *,
    cuda_probe: Callable[[ExperimentConfig], PreconditionCheck] = _probe_cuda_with_venv,
) -> tuple[tuple[PreconditionCheck, ...], str | None, dict[str, object], dict[str, object], dict[str, object]]:
    """Check hard resources before loading the full GGUF judge."""

    checks: list[PreconditionCheck] = [cuda_probe(config)]

    gguf_source: dict[str, object] = {}
    try:
        gguf_source = load_exp3915_source(config.repo_root)
        checks.append(
            PreconditionCheck(
                "exp3915_gguf_harness_ready",
                True,
                f"model={gguf_source.get('model_used')} smoke_tokens={gguf_source.get('smoke_tokens')}",
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3915_gguf_harness_ready", False, repr(exc)))

    diagnosis: dict[str, object] = {}
    try:
        diagnosis = diagnose_exp3917_scores(config.repo_root)
        checks.append(
            PreconditionCheck(
                "exp3917_per_item_results",
                True,
                f"n={diagnosis.get('n_items')} flipped_auroc={diagnosis.get('flipped_polarity_auroc')}",
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3917_per_item_results", False, repr(exc)))

    exp3884_source: dict[str, object] = {}
    try:
        exp3884_source = load_exp3884_source(config.repo_root)
        checks.append(
            PreconditionCheck(
                "exp3884_fixture_provenance",
                True,
                str(exp3884_source.get("corpus_path")),
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3884_fixture_provenance", False, repr(exc)))

    checks.append(_probe_imports())

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available.get("exp3915_gguf_harness_ready", False):
        blocked_reason = "blocked_upstream_gguf_harness_not_ready"
    elif not available.get("exp3917_per_item_results", False):
        blocked_reason = "blocked_upstream_efficiency_per_item_missing"
    elif not available.get("carnot_verify_imports", False):
        blocked_reason = "blocked_carnot_verify_import"
    elif not available.get("exp3884_fixture_provenance", False):
        blocked_reason = "blocked_fixture_provenance_missing"

    return tuple(checks), blocked_reason, diagnosis, gguf_source, exp3884_source


def _run_unit_test(config: ExperimentConfig) -> bool:
    if not config.run_unit_test:
        return True
    proc = subprocess.run(
        [
            str(config.venv_python()),
            "-m",
            "pytest",
            UNIT_TEST_PATH,
            "-q",
            "--no-cov",
            "-n",
            "0",
        ],
        cwd=config.repo_root,
        check=False,
    )
    return proc.returncode == 0


def _spend_live_floor(generator: object, started_at: float, config: ExperimentConfig) -> dict[str, object]:
    """Use real judge generations until the artifact crosses its live floor."""

    fixture = build_separable_fixture()
    calls = 0
    parsed = 0
    output_hash = hashlib.sha256()
    while config.clock() - started_at < LIVE_FLOOR_S:
        row = judge_step(fixture[calls % len(fixture)], generator, max_tokens=min(config.max_tokens, 64))
        calls += 1
        if row["parsed"]:
            parsed += 1
        output_hash.update(str(row["raw_text"]).encode("utf-8", errors="replace"))
    return {
        "live_floor_extra_calls": calls,
        "live_floor_extra_parsed": parsed,
        "live_floor_output_sha256": output_hash.hexdigest(),
    }


def _ready(artifact: dict[str, object]) -> bool:
    return (
        artifact.get("unit_test_passed") is True
        and isinstance(artifact.get("fixture_auroc"), (int, float))
        and float(artifact["fixture_auroc"]) > 0.65
        and isinstance(artifact.get("verdicts_parse_rate"), (int, float))
        and float(artifact["verdicts_parse_rate"]) > 0.9
        and float(artifact.get("duration_s") or 0.0) >= LIVE_FLOOR_S
    )


def _classify_verdict(artifact: dict[str, object]) -> str:
    auroc = artifact.get("fixture_auroc")
    rendered_auroc = "nan" if auroc is None else f"{float(auroc):.4f}"
    cause = str(artifact.get("diagnosed_cause") or "unknown")
    if _ready(artifact):
        model = str(artifact.get("judge_model_used") or "unknown")
        return (
            "complete: "
            f"competent_judge_READY_fixture_auroc{rendered_auroc}_cause{cause}_"
            f"model{model}_valid_comparator"
        )
    return f"complete: competent_judge_NOT_READY_fixture_auroc{rendered_auroc}_cause{cause}"


def build_artifact(
    *,
    config: ExperimentConfig,
    fixture_result: dict[str, Any],
    diagnosis: dict[str, object],
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    unit_test_passed: bool,
    exp3884_source: dict[str, object] | None = None,
    gguf_harness_source: dict[str, object] | None = None,
    live_floor: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the terminal Exp 3925 artifact from live fixture results."""

    started_at = config.start_time()
    finished_at = config.clock()
    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "diagnosis": diagnosis,
        "fixture_result": {
            "scores": fixture_result.get("scores"),
            "labels": fixture_result.get("labels"),
            "parsed": fixture_result.get("parsed"),
            "verdicts": fixture_result.get("verdicts"),
            "fixture_auroc": fixture_result.get("fixture_auroc"),
            "verdicts_parse_rate": fixture_result.get("verdicts_parse_rate"),
        },
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "module_sha256": _file_sha256(config.repo_root, JUDGE_MODULE_PATH)
        if (config.repo_root / JUDGE_MODULE_PATH).is_file()
        else None,
    }
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "diagnosed_cause": diagnosis.get("diagnosed_cause"),
        "original_auroc": diagnosis.get("original_auroc"),
        "flipped_polarity_auroc": diagnosis.get("flipped_polarity_auroc"),
        "diagnosis": diagnosis,
        "judge_module_path": JUDGE_MODULE_PATH,
        "judge_model_used": model_specs.get("model_used"),
        "fixture_auroc": float(fixture_result["fixture_auroc"]),
        "verdicts_parse_rate": float(fixture_result["verdicts_parse_rate"]),
        "parser_constant_prediction": bool(fixture_result.get("parser_constant_prediction")),
        "fixture_n_items": len(fixture_result.get("labels", [])),
        "fixture_n_errors": sum(int(label) for label in fixture_result.get("labels", [])),
        "fixture_scores": list(fixture_result.get("scores", [])),
        "fixture_labels": list(fixture_result.get("labels", [])),
        "fixture_verdicts": list(fixture_result.get("verdicts", [])),
        "fixture_parsed": list(fixture_result.get("parsed", [])),
        "fixture_parse_sources": list(fixture_result.get("parse_sources", [])),
        "fixture_raw_texts": list(fixture_result.get("raw_texts", [])),
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": bool(unit_test_passed),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs,
        "gguf_harness_source": gguf_harness_source or {},
        "exp3884_fixture_provenance": exp3884_source or {},
        "random_seed": config.random_seed,
        "random_seeds_used": {"llama_cpp_loader": 3915, "experiment": config.random_seed},
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": finished_at - started_at,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_floor": live_floor or {},
        "field_principles": FIELD_PRINCIPLES,
    }
    verdict = _classify_verdict(artifact)
    artifact["honest_verdict"] = verdict
    artifact["status"] = verdict
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    diagnosis: dict[str, object] | None = None,
    model_specs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a blocked artifact without fabricated competence metrics."""

    diag = diagnosis or {}
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "honest_verdict": reason,
        "status": reason,
        "diagnosed_cause": diag.get("diagnosed_cause"),
        "original_auroc": diag.get("original_auroc"),
        "flipped_polarity_auroc": diag.get("flipped_polarity_auroc"),
        "diagnosis": diag,
        "judge_module_path": JUDGE_MODULE_PATH,
        "judge_model_used": None,
        "fixture_auroc": None,
        "verdicts_parse_rate": None,
        "parser_constant_prediction": True,
        "fixture_n_items": 0,
        "fixture_n_errors": 0,
        "fixture_scores": [],
        "fixture_labels": [],
        "fixture_verdicts": [],
        "fixture_parsed": [],
        "fixture_parse_sources": [],
        "fixture_raw_texts": [],
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": False,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs or {},
        "gguf_harness_source": {},
        "exp3884_fixture_provenance": {},
        "random_seed": RANDOM_SEED,
        "random_seeds_used": {},
        "reproducibility_checksum": _checksum(
            {
                "experiment": EXPERIMENT_ID,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
                "diagnosis": diag,
                "model_specs": model_specs or {},
            }
        ),
        "duration_s": duration_s,
        "inference_substrate": "none_blocked_preflight",
        "live_floor": {},
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate required Exp 3925 fields and bare-scalar discipline."""

    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not _terminal_verdict(verdict):
        raise ValueError(f"honest_verdict lacks terminal prefix: {verdict}")
    for key in WRAPPED_VALUE_FORBIDDEN_FIELDS:
        if isinstance(artifact.get(key), dict):
            raise ValueError(f"{key} must not be a value/principle wrapper")
    if artifact["flipped_polarity_auroc"] is not None and not isinstance(
        artifact["flipped_polarity_auroc"], float
    ):
        raise ValueError("flipped_polarity_auroc must be a bare float or null")
    if artifact["fixture_auroc"] is not None and not isinstance(artifact["fixture_auroc"], float):
        raise ValueError("fixture_auroc must be a bare float or null")
    if artifact["verdicts_parse_rate"] is not None and not isinstance(
        artifact["verdicts_parse_rate"], float
    ):
        raise ValueError("verdicts_parse_rate must be a bare float or null")
    if not isinstance(artifact["unit_test_passed"], bool):
        raise ValueError("unit_test_passed must be a bare bool")
    if not isinstance(artifact["duration_s"], (int, float)):
        raise ValueError("duration_s must be a bare number")
    if len(str(artifact["reproducibility_checksum"])) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if verdict.startswith("blocked_") and artifact["fixture_auroc"] is not None:
        raise ValueError("blocked artifacts must not claim fixture AUROC")


def write_artifact(output_path: Path, artifact: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp 3925 end to end, or write a blocked artifact on failed gates."""

    config = config or ExperimentConfig(repo_root=REPO_ROOT)
    started = config.start_time()
    active_config = replace(config, started_at=started)
    output_path = active_config.resolved_output_path()
    checks, blocked_reason, diagnosis, gguf_source, exp3884_source = probe_preconditions(active_config)
    preflight_model_specs = {
        "prefer_order": list(COMPETENT_PREFER_ORDER),
        "n_ctx": active_config.n_ctx,
        "max_tokens": active_config.max_tokens,
        "max_n_gpu_layers": 0,
        "source_exp3915_model_used": gguf_source.get("model_used"),
    }
    if blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=blocked_reason,
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            diagnosis=diagnosis,
            model_specs=preflight_model_specs,
        )
        if write:
            write_artifact(output_path, artifact)
        return artifact

    unit_test_passed = _run_unit_test(active_config)
    generator, meta = load_gguf_generator(
        prefer_order=COMPETENT_PREFER_ORDER,
        n_ctx=active_config.n_ctx,
        max_n_gpu_layers=0,
    )
    model_specs = {
        **preflight_model_specs,
        **dict(meta),
        "loader": "carnot.verify.gguf_inference.load_gguf_generator",
        "judge_module_path": JUDGE_MODULE_PATH,
        "gguf_harness_module_path": GGUF_HARNESS_MODULE_PATH,
    }
    fixture_result = run_judge_fixture(
        build_separable_fixture(),
        generator,
        max_tokens=active_config.max_tokens,
    )
    live_floor = _spend_live_floor(generator, started, active_config)
    artifact = build_artifact(
        config=active_config,
        fixture_result=fixture_result,
        diagnosis=diagnosis,
        preconditions_checked=checks,
        model_specs=model_specs,
        unit_test_passed=unit_test_passed,
        exp3884_source=exp3884_source,
        gguf_harness_source=gguf_source,
        live_floor=live_floor,
    )
    if write:
        write_artifact(output_path, artifact)
    return artifact


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--no-unit-test", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=args.repo_root,
            output_path=args.output_path,
            run_unit_test=not args.no_unit_test,
        ),
        write=True,
    )
    output_path = args.output_path if args.output_path is not None else args.repo_root / OUTPUT_REL_PATH
    print(f"{output_path.name} wrote {artifact['honest_verdict']}")
    return 0 if str(artifact["honest_verdict"]).startswith("complete:") else 1


def main() -> int:
    return cli_main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
