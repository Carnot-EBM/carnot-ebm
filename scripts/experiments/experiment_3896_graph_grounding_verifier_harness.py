#!/usr/bin/env python3
"""Exp 3896 graph-grounding verifier harness positive control.

Spec refs: REQ-VERIFY-3896, SCENARIO-VERIFY-3896,
SCENARIO-VERIFY-3896-BLOCKED.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
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

from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.graph_grounding_fact_verifier_defabricated import (
    build_graph_grounding_fixture,
    load_facts_rows,
    reproducibility_checksum,
    resolve_facts_corpus,
    score_graph_grounding_fixture,
)


OUTPUT_REL_PATH = Path("results/experiment_3896_graph_grounding_verifier_harness.json")
VERIFIER_MODULE_PATH = "python/carnot/verify/graph_grounding_fact_verifier_defabricated.py"
UNIT_TEST_PATH = "tests/python/test_graph_grounding_fact_verifier.py"
RANDOM_SEED = 3896
MIN_READY_DURATION_S = 60.0
INFERENCE_SUBSTRATE = "live_llama_cpp_sota_gguf_graph_grounding_fixture"
MODEL_CANDIDATES = (
    ("unsloth/gemma-4-26B-A4B-it-GGUF", ("UD-Q4_K_M", "Q4_K_M", "IQ4", "Q5")),
    ("unsloth/Qwen3.6-35B-A3B-GGUF", ("UD-Q4_K_M", "Q4_K_M", "IQ4", "Q5")),
)
REQUIRED_FIELDS = {
    "verifier_module_path",
    "fixture_auroc",
    "model_invoked",
    "unit_test_path",
    "unit_test_passed",
    "facts_corpus_path",
    "facts_corpus_n_items",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "honest_verdict",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One hard resource checked before the Exp 3896 live fixture run."""

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
    """Runtime configuration for the Exp 3896 runner."""

    repo_root: Path
    output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    random_seed: int = RANDOM_SEED
    max_tokens: int = 220
    cuda_probe_timeout_s: int = 60
    run_unit_test: bool = True
    n_gpu_layers: int = -1
    n_ctx: int = 2048
    n_batch: int = 128

    def resolved_output_path(self) -> Path:
        if self.output_path is None:
            return self.repo_root / OUTPUT_REL_PATH
        return self.output_path if self.output_path.is_absolute() else self.repo_root / self.output_path

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "blocked_"))


def _probe_cuda(config: ExperimentConfig) -> PreconditionCheck:
    command = [
        str(config.venv_python()),
        "-c",
        "import torch; assert torch.cuda.is_available()",
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.cuda_probe_timeout_s,
            check=False,
        )
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, repr(exc))
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    return PreconditionCheck("cuda_available", proc.returncode == 0, detail)


def _resolve_model(config: ExperimentConfig) -> tuple[dict[str, object], list[PreconditionCheck]]:
    checks: list[PreconditionCheck] = []
    selected_hf_id: str | None = None
    selected_path: str | None = None
    for hf_id, _preferred_tokens in MODEL_CANDIDATES:
        path = resolve_cached_gguf(hf_id)
        available = path is not None and Path(path).is_file() and Path(path).stat().st_size > 0
        checks.append(
            PreconditionCheck(
                f"{hf_id.split('/')[-1].lower().replace('-', '_')}_cached",
                available,
                str(path) if path else "missing",
            )
        )
        if available and selected_path is None:
            selected_hf_id = hf_id
            selected_path = str(path)

    model_specs: dict[str, object] = {
        "hf_id": selected_hf_id,
        "model_path": selected_path,
        "candidate_hf_ids": [hf_id for hf_id, _tokens in MODEL_CANDIDATES],
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": config.n_gpu_layers,
        "n_ctx": config.n_ctx,
        "n_batch": config.n_batch,
        "max_tokens": config.max_tokens,
    }
    return model_specs, checks


def _facts_corpus_status(config: ExperimentConfig) -> tuple[Path | None, int, PreconditionCheck]:
    corpus_path = resolve_facts_corpus(config.repo_root)
    if corpus_path is None:
        return None, 0, PreconditionCheck("facts_corpus_with_gold_labels", False, "missing")
    rows = load_facts_rows(corpus_path, 120)
    labels = {int(bool(row.get("is_hallucination"))) for row in rows}
    available = len(rows) > 0 and labels == {0, 1}
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
) -> tuple[str | None, list[PreconditionCheck], dict[str, object], Path | None, int]:
    """Check live resources before loading the graph-grounding GGUF."""

    checks = [_probe_cuda(config)]
    model_specs, model_checks = _resolve_model(config)
    checks.extend(model_checks)
    try:
        __import__("llama_cpp")
        checks.append(PreconditionCheck("llama_cpp_import", True, "import llama_cpp OK"))
    except Exception as exc:
        checks.append(PreconditionCheck("llama_cpp_import", False, repr(exc)))
    corpus_path, facts_n_items, corpus_check = _facts_corpus_status(config)
    checks.append(corpus_check)

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not model_specs.get("model_path"):
        blocked_reason = "blocked_model_not_cached"
    elif not available.get("llama_cpp_import", False):
        blocked_reason = "blocked_llama_cpp_not_installed"
    elif not available.get("facts_corpus_with_gold_labels", False):
        blocked_reason = "blocked_facts_corpus_missing"
    return blocked_reason, checks, model_specs, corpus_path, facts_n_items


def _ready(
    *,
    fixture_result: dict[str, object],
    unit_test_passed: bool,
    duration_s: float,
) -> bool:
    auroc = fixture_result.get("fixture_auroc")
    return (
        bool(fixture_result.get("model_invoked"))
        and unit_test_passed
        and isinstance(auroc, (int, float))
        and float(auroc) > 0.6
        and bool(fixture_result.get("planted_hallucinated_relation_flagged"))
        and bool(fixture_result.get("stub_rejected"))
        and duration_s >= MIN_READY_DURATION_S
    )


def build_artifact(
    *,
    fixture_result: dict[str, object],
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    unit_test_passed: bool,
    facts_corpus_path: Path | None,
    facts_corpus_n_items: int,
) -> dict[str, object]:
    """Build the terminal Exp 3896 artifact from a completed fixture run."""

    started_at = config.start_time()
    finished_at = config.clock()
    duration_s = finished_at - started_at
    auroc_value = fixture_result.get("fixture_auroc")
    fixture_auroc = float(auroc_value) if isinstance(auroc_value, (int, float)) else None
    rendered_auroc = "nan" if fixture_auroc is None else f"{fixture_auroc:.4f}"
    model_invoked = bool(fixture_result.get("model_invoked"))
    if _ready(
        fixture_result=fixture_result,
        unit_test_passed=unit_test_passed,
        duration_s=duration_s,
    ):
        verdict = (
            "complete: "
            f"graph_grounding_verifier_READY_fixture_auroc{rendered_auroc}_"
            "facts_run_can_proceed"
        )
    else:
        verdict = (
            "complete: "
            f"graph_grounding_verifier_NOT_READY_fixture_auroc{rendered_auroc}_"
            f"model_invoked{str(model_invoked).lower()}"
        )

    fixture = build_graph_grounding_fixture()
    checksum_payload = {
        "experiment": 3896,
        "fixture": [
            {
                "id": item["id"],
                "claim": item["claim"],
                "source": item["source"],
                "gold_hallucinated": item["gold_hallucinated"],
            }
            for item in fixture
        ],
        "labels": fixture_result.get("labels"),
        "graph_scores": fixture_result.get("graph_scores"),
        "consistency_passes": fixture_result.get("consistency_passes"),
        "consistency_scores": fixture_result.get("consistency_scores"),
        "model_specs": model_specs,
        "facts_corpus_path": str(facts_corpus_path) if facts_corpus_path else "",
        "facts_corpus_n_items": facts_corpus_n_items,
        "random_seed": config.random_seed,
    }
    artifact: dict[str, object] = {
        "experiment": 3896,
        "title": "graph_grounding_verifier_harness",
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "honest_verdict": verdict,
        "status": verdict,
        "verifier_module_path": VERIFIER_MODULE_PATH,
        "fixture_auroc": fixture_auroc,
        "fixture_n_items": int(fixture_result.get("fixture_n_items") or 0),
        "fixture_n_hallucinated": int(fixture_result.get("fixture_n_hallucinated") or 0),
        "model_invoked": model_invoked,
        "model_call_count": int(fixture_result.get("model_call_count") or 0),
        "planted_hallucinated_relation_flagged": bool(
            fixture_result.get("planted_hallucinated_relation_flagged")
        ),
        "stub_rejected": bool(fixture_result.get("stub_rejected")),
        "parse_fallback_count": int(fixture_result.get("parse_fallback_count") or 0),
        "consistency_passes": int(fixture_result.get("consistency_passes") or 0),
        "consistency_scores": list(fixture_result.get("consistency_scores") or []),
        "per_item_scores": list(fixture_result.get("per_item_scores") or []),
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": bool(unit_test_passed),
        "facts_corpus_path": str(facts_corpus_path) if facts_corpus_path else "",
        "facts_corpus_n_items": int(facts_corpus_n_items),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "reproducibility_checksum": reproducibility_checksum(checksum_payload),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    model_specs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a blocked artifact without fabricated graph-grounding metrics."""

    artifact: dict[str, object] = {
        "experiment": 3896,
        "title": "graph_grounding_verifier_harness",
        "honest_verdict": reason,
        "status": reason,
        "verifier_module_path": VERIFIER_MODULE_PATH,
        "fixture_auroc": None,
        "fixture_n_items": 0,
        "fixture_n_hallucinated": 0,
        "model_invoked": False,
        "model_call_count": 0,
        "planted_hallucinated_relation_flagged": False,
        "stub_rejected": False,
        "parse_fallback_count": 0,
        "consistency_passes": 0,
        "consistency_scores": [],
        "per_item_scores": [],
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": False,
        "facts_corpus_path": "",
        "facts_corpus_n_items": 0,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs or {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "experiment": 3896,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
                "model_specs": model_specs or {},
            }
        ),
        "duration_s": duration_s,
        "inference_substrate": "none_blocked_preflight",
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate required Exp 3896 fields and terminal-prefix discipline."""

    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not _terminal_verdict(verdict):
        raise ValueError(f"honest_verdict lacks terminal prefix: {verdict}")
    if not isinstance(artifact["verifier_module_path"], str):
        raise ValueError("verifier_module_path must be a bare string")
    if artifact["fixture_auroc"] is not None and not isinstance(artifact["fixture_auroc"], float):
        raise ValueError("fixture_auroc must be a bare float or null")
    if not isinstance(artifact["model_invoked"], bool):
        raise ValueError("model_invoked must be a bare bool")
    if not isinstance(artifact["unit_test_path"], str):
        raise ValueError("unit_test_path must be a bare string")
    if not isinstance(artifact["unit_test_passed"], bool):
        raise ValueError("unit_test_passed must be a bare bool")
    if not isinstance(artifact["facts_corpus_path"], str):
        raise ValueError("facts_corpus_path must be a bare string")
    if not isinstance(artifact["facts_corpus_n_items"], int):
        raise ValueError("facts_corpus_n_items must be a bare int")
    if not isinstance(artifact["duration_s"], (int, float)):
        raise ValueError("duration_s must be a bare number")
    if len(str(artifact["reproducibility_checksum"])) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if verdict.startswith("blocked_") and bool(artifact["model_invoked"]):
        raise ValueError("blocked artifacts cannot claim model_invoked=true")
    if verdict.startswith("complete: graph_grounding_verifier_READY"):
        if not bool(artifact["model_invoked"]):
            raise ValueError("READY artifacts require model_invoked=true")
        if not bool(artifact["unit_test_passed"]):
            raise ValueError("READY artifacts require unit_test_passed=true")
        if float(artifact["duration_s"]) < MIN_READY_DURATION_S:
            raise ValueError("READY artifacts require duration_s>=60")


def write_artifact(output_path: Path, artifact: dict[str, object]) -> None:
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_unit_test(config: ExperimentConfig, artifact_path: Path) -> bool:
    if not config.run_unit_test:
        return False
    child_env = os.environ.copy()
    child_env["CARNOT_3896_ARTIFACT_UNDER_TEST"] = str(artifact_path)
    for key in list(child_env):
        if key.startswith(("PYTEST_", "COV_CORE")):
            child_env.pop(key, None)
    command = [
        str(config.venv_python()),
        "-m",
        "pytest",
        UNIT_TEST_PATH,
        "-q",
        "--no-cov",
        "-n",
        "0",
    ]
    proc = subprocess.run(command, cwd=config.repo_root, env=child_env, check=False)
    return proc.returncode == 0


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp 3896 end to end, or write a blocked artifact on failed gates."""

    config = config or ExperimentConfig(repo_root=REPO_ROOT)
    started = config.start_time()
    active_config = ExperimentConfig(
        repo_root=config.repo_root,
        output_path=config.output_path,
        started_at=started,
        clock=config.clock,
        random_seed=config.random_seed,
        max_tokens=config.max_tokens,
        cuda_probe_timeout_s=config.cuda_probe_timeout_s,
        run_unit_test=config.run_unit_test,
        n_gpu_layers=config.n_gpu_layers,
        n_ctx=config.n_ctx,
        n_batch=config.n_batch,
    )
    output_path = active_config.resolved_output_path()
    blocked_reason, checks, model_specs, corpus_path, facts_n_items = probe_preconditions(
        active_config
    )
    if blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=blocked_reason,
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=model_specs,
        )
        if write:
            write_artifact(output_path, artifact)
        return artifact

    try:
        fixture_result = score_graph_grounding_fixture(
            build_graph_grounding_fixture(),
            model_path=str(model_specs["model_path"]),
            max_tokens=active_config.max_tokens,
            n_gpu_layers=active_config.n_gpu_layers,
            n_ctx=active_config.n_ctx,
            n_batch=active_config.n_batch,
            consistency_passes=2,
        )
    except Exception as exc:
        artifact = build_blocked_artifact(
            reason="blocked_llama_cpp_inference_failed",
            preconditions_checked=[
                *checks,
                PreconditionCheck("llama_cpp_inference", False, repr(exc)),
            ],
            duration_s=active_config.clock() - started,
            model_specs=model_specs,
        )
        if write:
            write_artifact(output_path, artifact)
        return artifact

    provisional_unit_test_passed = bool(active_config.run_unit_test)
    artifact = build_artifact(
        fixture_result=fixture_result,
        config=active_config,
        preconditions_checked=checks,
        model_specs=model_specs,
        unit_test_passed=provisional_unit_test_passed,
        facts_corpus_path=corpus_path,
        facts_corpus_n_items=facts_n_items,
    )
    if write:
        write_artifact(output_path, artifact)

    unit_test_passed = _run_unit_test(active_config, output_path)
    if unit_test_passed != provisional_unit_test_passed:
        artifact = build_artifact(
            fixture_result=fixture_result,
            config=active_config,
            preconditions_checked=checks,
            model_specs=model_specs,
            unit_test_passed=unit_test_passed,
            facts_corpus_path=corpus_path,
            facts_corpus_n_items=facts_n_items,
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
    verdict = str(artifact["honest_verdict"])
    return 0 if verdict.startswith("complete: graph_grounding_verifier_READY") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
