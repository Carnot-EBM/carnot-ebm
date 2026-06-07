#!/usr/bin/env python3
"""Exp 3920 facts graph-grounding last retry.

Spec refs: REQ-VERIFY-3920, SCENARIO-VERIFY-3920,
SCENARIO-VERIFY-3920-BLOCKED.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.graph_grounding_fact_verifier_defabricated import (  # noqa: E402
    RobustGeneratorGraphExtractor,
    build_nonseparable_graph_grounding_fixture,
    compute_hallugraph_score,
    load_facts_rows,
    load_robust_graph_grounding_generator,
    reproducibility_checksum,
    resolve_facts_corpus,
    score_nonseparable_graph_grounding_fixture,
    sha256_file,
    sha256_text,
)


OUTPUT_REL_PATH = Path("results/experiment_3920_facts_graph_grounding_last_retry.json")
PER_ITEM_REL_PATH = Path("results/experiment_3920_facts_graph_grounding_last_retry_scores.jsonl")
EXP3915_REL_PATH = Path("results/experiment_3915_robust_gguf_inference_harness.json")
VERIFIER_MODULE_PATH = "python/carnot/verify/graph_grounding_fact_verifier_defabricated.py"
UNIT_TEST_PATH = "tests/python/test_graph_grounding_fact_verifier.py"
RANDOM_SEED = 3920
MIN_CORPUS_ITEMS = 60
MIN_READY_DURATION_S = 60.0
INFERENCE_SUBSTRATE = "live_gguf_inference:exp3915_robust_generator_graph_grounding"

FIELD_PRINCIPLES = {
    "verifier_module_path": "Where a future facts run would import the tested verifier from.",
    "gguf_harness_model_used": "Which GGUF the robust harness loaded.",
    "fixture_auroc": (
        "BARE FLOAT - must be in [0.6,0.95]; 1.0 is the exp3896 separability tell."
    ),
    "model_invoked": (
        "BARE BOOL - true only when fixture and corpus rows record model token evidence."
    ),
    "corpus_run_token_count": (
        "BARE INT - total tokens consumed on the corpus slice; >0 proves invocation."
    ),
    "unit_test_path": "The deliverable test file.",
    "unit_test_passed": "BARE BOOL - passing test on the non-separable fixture.",
    "preconditions_checked": "Pre-Launch + Adversarial-Verify resource checks.",
    "model_specs": "The robust GGUF harness metadata and runtime settings.",
    "random_seed": "Fixed seed for reproducible fixture and corpus selection.",
    "reproducibility_checksum": "Hash over fixture, corpus, model, and score evidence.",
    "duration_s": "Measured wall-clock; READY requires a real corpus run >=60s.",
    "inference_substrate": "Declares the actual live inference runtime.",
}
REQUIRED_FIELDS = {
    "verifier_module_path",
    "gguf_harness_model_used",
    "fixture_auroc",
    "model_invoked",
    "corpus_run_token_count",
    "unit_test_path",
    "unit_test_passed",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "honest_verdict",
    "field_principles",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One Exp 3920 live-resource precondition check."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the Exp 3920 runner."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    per_item_scores_path: Path | None = None
    sample_size: int = MIN_CORPUS_ITEMS
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    random_seed: int = RANDOM_SEED
    max_tokens: int = 64
    n_ctx: int = 1024
    run_unit_test: bool = True
    cuda_probe_timeout_s: int = 60

    def resolved_output_path(self) -> Path:
        path = self.output_path or OUTPUT_REL_PATH
        return path if path.is_absolute() else self.repo_root / path

    def resolved_per_item_scores_path(self) -> Path:
        path = self.per_item_scores_path or PER_ITEM_REL_PATH
        return path if path.is_absolute() else self.repo_root / path

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _run_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y%m%d")


def _checks_to_dicts(checks: Sequence[PreconditionCheck | Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for check in checks:
        if isinstance(check, PreconditionCheck):
            output.append(check.as_dict())
        else:
            output.append(dict(check))
    return output


def _probe_cuda(config: ExperimentConfig) -> PreconditionCheck:
    try:
        proc = subprocess.run(
            [
                str(config.venv_python()),
                "-c",
                "import torch; assert torch.cuda.is_available()",
            ],
            capture_output=True,
            text=True,
            timeout=config.cuda_probe_timeout_s,
            check=False,
        )
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, repr(exc))
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    return PreconditionCheck("cuda_available", proc.returncode == 0, detail)


def _probe_exp3915(config: ExperimentConfig) -> tuple[PreconditionCheck, dict[str, Any]]:
    path = config.repo_root / EXP3915_REL_PATH
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PreconditionCheck("exp3915_robust_harness_ready", False, repr(exc)), {}
    ready = bool(artifact.get("unit_test_passed")) and int(artifact.get("smoke_tokens") or 0) > 0
    detail = json.dumps(
        {
            "path": str(EXP3915_REL_PATH),
            "unit_test_passed": artifact.get("unit_test_passed"),
            "smoke_tokens": artifact.get("smoke_tokens"),
            "harness_module_path": artifact.get("harness_module_path"),
            "model_used": artifact.get("model_used"),
        },
        sort_keys=True,
    )
    return PreconditionCheck("exp3915_robust_harness_ready", ready, detail), artifact


def _facts_corpus_status(config: ExperimentConfig) -> tuple[Path | None, int, PreconditionCheck]:
    corpus_path = resolve_facts_corpus(config.repo_root)
    if corpus_path is None:
        return None, 0, PreconditionCheck("facts_corpus_with_gold_labels", False, "missing")
    rows = load_facts_rows(corpus_path, max(MIN_CORPUS_ITEMS, config.sample_size))
    labels = {int(bool(row.get("is_hallucination"))) for row in rows}
    available = len(rows) >= MIN_CORPUS_ITEMS and labels == {0, 1}
    return (
        corpus_path,
        len(rows) if available else 0,
        PreconditionCheck(
            "facts_corpus_with_gold_labels",
            available,
            f"{corpus_path} n_items={len(rows)} labels={sorted(labels)}",
        ),
    )


def probe_preconditions(
    config: ExperimentConfig,
) -> tuple[str | None, list[PreconditionCheck], dict[str, Any], Path | None, int]:
    """Check hard Exp 3920 resources before loading the robust generator."""

    checks = [_probe_cuda(config)]
    exp3915_check, exp3915_artifact = _probe_exp3915(config)
    checks.append(exp3915_check)
    corpus_path, corpus_n, corpus_check = _facts_corpus_status(config)
    checks.append(corpus_check)

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available.get("exp3915_robust_harness_ready", False):
        blocked_reason = "blocked_upstream_gguf_harness_not_ready"
    elif not available.get("facts_corpus_with_gold_labels", False):
        blocked_reason = "blocked_facts_corpus_missing"

    return blocked_reason, checks, exp3915_artifact, corpus_path, corpus_n


def score_facts_corpus_slice(
    rows: Sequence[Mapping[str, Any]],
    *,
    generator: Any,
    per_item_scores_path: Path,
    model_specs: Mapping[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    """Score a labeled facts-corpus slice with the robust graph extractor."""

    clean_rows = [
        dict(row)
        for row in rows
        if str(row.get("answer") or "").strip()
        and str(row.get("evidence_passage") or "").strip()
        and "is_hallucination" in row
    ][:MIN_CORPUS_ITEMS]
    extractor = RobustGeneratorGraphExtractor(generator, model_specs, max_tokens=max_tokens)
    per_item: list[dict[str, Any]] = []
    labels: list[int] = []
    graph_scores: list[float] = []
    for index, row in enumerate(clean_rows):
        extraction = extractor.extract_pair(
            str(row.get("answer") or ""),
            str(row.get("evidence_passage") or ""),
            index,
        )
        score = compute_hallugraph_score(extraction)
        label = int(bool(row.get("is_hallucination")))
        labels.append(label)
        graph_scores.append(score.hallucination_score)
        per_item.append(
            {
                "index": index,
                "item_id": str(row.get("id") or row.get("question_id") or f"facts-{index}"),
                "gold_ungrounded": bool(label),
                "graph_score": score.hallucination_score,
                "eg": score.entity_grounding,
                "rp": score.relation_preservation,
                "cfi": score.composite_fidelity_index,
                "completion_tokens": extraction.completion_tokens,
                "prompt_sha256": extraction.prompt_sha256,
                "parse_fallback_used": extraction.parse_fallback_used,
                "missing_entities": list(score.missing_entities),
                "unsupported_relations": list(score.unsupported_relations),
                "answer_sha256": sha256_text(str(row.get("answer") or "")),
                "evidence_sha256": sha256_text(str(row.get("evidence_passage") or "")),
            }
        )

    per_item_scores_path.parent.mkdir(parents=True, exist_ok=True)
    per_item_scores_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_item),
        encoding="utf-8",
    )
    token_count = int(sum(int(row["completion_tokens"]) for row in per_item))
    return {
        "n_items": len(per_item),
        "labels": labels,
        "graph_scores": [round(float(score), 6) for score in graph_scores],
        "model_invoked": extractor.invocation_count >= len(per_item) and len(per_item) >= MIN_CORPUS_ITEMS,
        "model_call_count": extractor.invocation_count,
        "corpus_run_token_count": token_count,
        "per_item_scores_path": _artifact_path(per_item_scores_path),
        "per_item_scores_sha256": sha256_file(per_item_scores_path),
        "parse_fallback_count": sum(1 for row in per_item if row["parse_fallback_used"]),
    }


def _artifact_path(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        return str(Path(*parts[parts.index("results") :]))
    return str(path)


def _fixture_auroc_value(fixture_result: Mapping[str, Any]) -> float | None:
    value = fixture_result.get("fixture_auroc")
    return float(value) if isinstance(value, (int, float)) else None


def _ready(
    *,
    fixture_auroc: float | None,
    model_invoked: bool,
    unit_test_passed: bool,
    corpus_run_token_count: int,
    duration_s: float,
) -> bool:
    return (
        model_invoked
        and unit_test_passed
        and fixture_auroc is not None
        and 0.6 <= fixture_auroc <= 0.95
        and duration_s >= MIN_READY_DURATION_S
        and corpus_run_token_count > 0
    )


def build_artifact(
    *,
    fixture_result: Mapping[str, Any],
    corpus_result: Mapping[str, Any],
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    model_specs: Mapping[str, Any],
    unit_test_passed: bool,
) -> dict[str, Any]:
    """Build the terminal Exp 3920 artifact from live fixture and corpus results."""

    started_at = config.start_time()
    finished_at = config.clock()
    duration_s = round(max(0.0, finished_at - started_at), 6)
    fixture_auroc = _fixture_auroc_value(fixture_result)
    rendered_auroc = "nan" if fixture_auroc is None else f"{fixture_auroc:.4f}"
    corpus_run_token_count = int(corpus_result.get("corpus_run_token_count") or 0)
    model_invoked = bool(fixture_result.get("model_invoked")) and bool(
        corpus_result.get("model_invoked")
    )
    if _ready(
        fixture_auroc=fixture_auroc,
        model_invoked=model_invoked,
        unit_test_passed=unit_test_passed,
        corpus_run_token_count=corpus_run_token_count,
        duration_s=duration_s,
    ):
        verdict = (
            "complete: "
            f"facts_graph_verifier_READY_fixture_auroc{rendered_auroc}_"
            f"model_invoked_tokens{corpus_run_token_count}"
        )
    else:
        verdict = (
            "complete: "
            f"facts_graph_verifier_NOT_READY_fixture_auroc{rendered_auroc}_"
            "facts_route_retires_to_future_work"
        )

    checksum_payload = {
        "fixture": [
            {
                "id": item["id"],
                "claim": item["claim"],
                "source": item["source"],
                "gold_hallucinated": item["gold_hallucinated"],
            }
            for item in build_nonseparable_graph_grounding_fixture()
        ],
        "fixture_scores": fixture_result.get("graph_scores"),
        "corpus_scores": corpus_result.get("graph_scores"),
        "corpus_labels": corpus_result.get("labels"),
        "model_specs": dict(model_specs),
        "random_seed": config.random_seed,
    }
    artifact = {
        "experiment": 3920,
        "title": "facts_graph_grounding_last_retry",
        "run_date": _run_date(finished_at),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "honest_verdict": verdict,
        "status": verdict,
        "verifier_module_path": VERIFIER_MODULE_PATH,
        "gguf_harness_model_used": str(model_specs.get("model_used") or ""),
        "fixture_auroc": fixture_auroc,
        "fixture_n_items": int(fixture_result.get("fixture_n_items") or 0),
        "fixture_token_count": int(fixture_result.get("fixture_token_count") or 0),
        "model_invoked": model_invoked,
        "model_call_count": int(fixture_result.get("model_call_count") or 0)
        + int(corpus_result.get("model_call_count") or 0),
        "corpus_n_items": int(corpus_result.get("n_items") or 0),
        "corpus_run_token_count": corpus_run_token_count,
        "corpus_per_item_scores_path": str(corpus_result.get("per_item_scores_path") or ""),
        "corpus_per_item_scores_sha256": str(corpus_result.get("per_item_scores_sha256") or ""),
        "fixture_result": dict(fixture_result),
        "corpus_result": dict(corpus_result),
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": bool(unit_test_passed),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "model_specs": dict(model_specs),
        "random_seed": config.random_seed,
        "reproducibility_checksum": reproducibility_checksum(checksum_payload),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    model_specs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a terminal blocked artifact without fabricated readiness metrics."""

    artifact = {
        "experiment": 3920,
        "title": "facts_graph_grounding_last_retry",
        "honest_verdict": reason,
        "status": reason,
        "verifier_module_path": VERIFIER_MODULE_PATH,
        "gguf_harness_model_used": "",
        "fixture_auroc": None,
        "fixture_n_items": 0,
        "fixture_token_count": 0,
        "model_invoked": False,
        "model_call_count": 0,
        "corpus_n_items": 0,
        "corpus_run_token_count": 0,
        "corpus_per_item_scores_path": "",
        "corpus_per_item_scores_sha256": "",
        "fixture_result": {},
        "corpus_result": {},
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": False,
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "model_specs": dict(model_specs),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "blocked_reason": reason,
                "preconditions_checked": _checks_to_dicts(preconditions_checked),
                "model_specs": dict(model_specs),
            }
        ),
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "inference_substrate": "none_blocked_preflight",
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3920 terminal artifact schema."""

    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if not isinstance(artifact.get("verifier_module_path"), str):
        raise ValueError("verifier_module_path must be a bare string")
    if not isinstance(artifact.get("gguf_harness_model_used"), str):
        raise ValueError("gguf_harness_model_used must be a bare string")
    fixture_auroc = artifact.get("fixture_auroc")
    if fixture_auroc is not None and not isinstance(fixture_auroc, float):
        raise ValueError("fixture_auroc must be a bare float or null")
    if type(artifact.get("model_invoked")) is not bool:
        raise ValueError("model_invoked must be a bare bool")
    if not isinstance(artifact.get("corpus_run_token_count"), int):
        raise ValueError("corpus_run_token_count must be a bare int")
    if not isinstance(artifact.get("unit_test_path"), str):
        raise ValueError("unit_test_path must be a bare string")
    if type(artifact.get("unit_test_passed")) is not bool:
        raise ValueError("unit_test_passed must be a bare bool")
    if not isinstance(artifact.get("duration_s"), (int, float)):
        raise ValueError("duration_s must be a bare number")
    if len(str(artifact.get("reproducibility_checksum") or "")) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    uncovered = set(FIELD_PRINCIPLES) - set(principles)
    if uncovered:
        raise ValueError(f"field_principles missing required fields: {sorted(uncovered)}")
    if verdict.startswith("blocked_") and bool(artifact.get("model_invoked")):
        raise ValueError("blocked artifacts cannot claim model_invoked=true")
    if verdict.startswith("complete: facts_graph_verifier_READY"):
        if not bool(artifact.get("model_invoked")):
            raise ValueError("READY requires model_invoked=true")
        if not bool(artifact.get("unit_test_passed")):
            raise ValueError("READY requires unit_test_passed=true")
        if fixture_auroc is None or not 0.6 <= float(fixture_auroc) <= 0.95:
            raise ValueError("READY requires fixture_auroc in [0.6,0.95]")
        if float(artifact.get("duration_s") or 0.0) < MIN_READY_DURATION_S:
            raise ValueError("READY requires duration_s>=60")
        if int(artifact.get("corpus_run_token_count") or 0) <= 0:
            raise ValueError("READY requires corpus_run_token_count>0")


def write_artifact(output_path: Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_unit_test(config: ExperimentConfig, artifact_path: Path) -> bool:
    if not config.run_unit_test:
        return False
    child_env = os.environ.copy()
    child_env["CARNOT_3920_ARTIFACT_UNDER_TEST"] = str(artifact_path)
    for key in list(child_env):
        if key.startswith(("PYTEST_", "COV_CORE")):
            child_env.pop(key, None)
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
        env=child_env,
        check=False,
    )
    return proc.returncode == 0


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Run Exp 3920 end to end and write a terminal artifact."""

    config = config or ExperimentConfig(repo_root=REPO_ROOT)
    started = config.start_time()
    active_config = ExperimentConfig(
        repo_root=config.repo_root,
        output_path=config.output_path,
        per_item_scores_path=config.per_item_scores_path,
        sample_size=max(MIN_CORPUS_ITEMS, config.sample_size),
        started_at=started,
        clock=config.clock,
        random_seed=config.random_seed,
        max_tokens=config.max_tokens,
        n_ctx=config.n_ctx,
        run_unit_test=config.run_unit_test,
        cuda_probe_timeout_s=config.cuda_probe_timeout_s,
    )
    output_path = active_config.resolved_output_path()
    blocked_reason, checks, exp3915_artifact, corpus_path, _corpus_n = probe_preconditions(
        active_config
    )
    if blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=blocked_reason,
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=exp3915_artifact,
        )
        if write:
            write_artifact(output_path, artifact)
        return artifact

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        generator, generator_meta = load_robust_graph_grounding_generator(n_ctx=active_config.n_ctx)
        model_specs = {
            **dict(exp3915_artifact),
            **dict(generator_meta),
            "harness_module_path": "python/carnot/verify/gguf_inference.py",
            "max_tokens": active_config.max_tokens,
            "n_ctx": active_config.n_ctx,
            "llama_cpp_cuda_visible_devices": "",
        }
        fixture_result = score_nonseparable_graph_grounding_fixture(
            build_nonseparable_graph_grounding_fixture(),
            generator=generator,
            max_tokens=active_config.max_tokens,
        )
        assert corpus_path is not None
        rows = load_facts_rows(corpus_path, active_config.sample_size)
        corpus_result = score_facts_corpus_slice(
            rows,
            generator=generator,
            per_item_scores_path=active_config.resolved_per_item_scores_path(),
            model_specs=model_specs,
            max_tokens=active_config.max_tokens,
        )
    except Exception as exc:
        artifact = build_blocked_artifact(
            reason="blocked_llama_cpp_inference_failed",
            preconditions_checked=[*checks, PreconditionCheck("live_inference", False, repr(exc))],
            duration_s=active_config.clock() - started,
            model_specs=exp3915_artifact,
        )
        if write:
            write_artifact(output_path, artifact)
        return artifact

    provisional_unit_test_passed = bool(active_config.run_unit_test)
    artifact = build_artifact(
        fixture_result=fixture_result,
        corpus_result=corpus_result,
        config=active_config,
        preconditions_checked=checks,
        model_specs=model_specs,
        unit_test_passed=provisional_unit_test_passed,
    )
    if write:
        write_artifact(output_path, artifact)

    unit_test_passed = _run_unit_test(active_config, output_path)
    if unit_test_passed != provisional_unit_test_passed:
        artifact = build_artifact(
            fixture_result=fixture_result,
            corpus_result=corpus_result,
            config=active_config,
            preconditions_checked=checks,
            model_specs=model_specs,
            unit_test_passed=unit_test_passed,
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
    print(f"{output_path} wrote {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
