"""Exp 2879 manifest-only MBPP/HumanEval execution pilot.

This module fills the code-corpus row gap in the cross-corpus matrix without
pretending to run a full live-generation benchmark. It reads the checked-in
manifest contract, selects canonical/reference code rows that already include
tests, and executes only those deterministic reference programs through the
existing gVisor sandbox wrapper. If sandbox isolation is unavailable, the pilot
blocks instead of falling back to unsafe in-process execution.

Spec: REQ-CODE-2879, SCENARIO-CODE-2879.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.verify.sandbox import get_sandbox_status, sandboxed_exec_function


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
OUTPUT_FILENAME = "experiment_2879_code_corpus_manifest_execution_pilot_v1.json"
MANIFEST_CONTRACT_REL_PATH = Path("results/experiment_2863_eval_manifest_contract_v2.json")
CROSS_CORPUS_MATRIX_REL_PATH = Path("results/experiment_2865_cross_corpus_matrix_v5.json")
CODE_CORPORA = ("mbpp", "humaneval")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "code_manifest_pilot_ready",
    "source_artifacts",
    "manifest_paths",
    "selection_rule",
    "n_mbpp_rows",
    "n_humaneval_rows",
    "deterministic_execution_used",
    "sandbox_status",
    "pilot_rows",
    "headline_metric_claim_made",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict; complete only for sandboxed deterministic rows.",
    "code_manifest_pilot_ready": "True only when selected MBPP and HumanEval rows pass.",
    "source_artifacts": "Contract, matrix, and manifest files used as local evidence.",
    "manifest_paths": "MBPP/HumanEval paths resolved from the manifest contract.",
    "selection_rule": "Human-readable deterministic sample selection policy.",
    "n_mbpp_rows": "Number of MBPP rows actually executed in the pilot.",
    "n_humaneval_rows": "Number of HumanEval rows actually executed in the pilot.",
    "deterministic_execution_used": "True only after sandbox execution is attempted.",
    "sandbox_status": "Blocks when runsc/gVisor isolation is unavailable.",
    "pilot_rows": "Per-row pass/fail/test metadata and verifier feature coverage.",
    "headline_metric_claim_made": "Always false; no generated-code labels or AUROC.",
    "tests_run": "Commands used to validate the pilot code and artifact.",
    "duration_s": "Measured wall-clock runtime; no padding.",
}

SELECTION_RULE = (
    "Select the first eligible row in manifest order for each of MBPP and HumanEval. "
    "Eligibility requires canonical/reference code plus local tests; no LLM generation."
)


@dataclass(frozen=True)
class ManifestResolution:
    """Verified manifest contract data for one code corpus."""

    corpus: str
    path: Path
    declared_sha256: str
    actual_sha256: str
    count: int
    ready: bool


@dataclass(frozen=True)
class ExecutionOutcome:
    """Result returned by the deterministic sandbox execution wrapper."""

    passed: bool
    error_type: str | None = None
    error_message: str = ""
    timed_out: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2879 manifest-only pilot."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    manifest_contract_path: Path = MANIFEST_CONTRACT_REL_PATH
    cross_corpus_matrix_path: Path = CROSS_CORPUS_MATRIX_REL_PATH
    tests_run: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    timeout_s: float = 10.0

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


SandboxFunction = Callable[..., tuple[Any, Exception | None]]
Executor = Callable[[str, float], ExecutionOutcome]


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _source_name(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stable_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_code_manifests(config: ExperimentConfig) -> tuple[dict[str, ManifestResolution], bool]:
    contract_path = _repo_path(config.repo_root, config.manifest_contract_path)
    contract = _read_json(contract_path)
    paths = dict(contract.get("resolved_manifest_paths") or {})
    declared = dict(contract.get("resolved_manifest_sha256") or {})
    counts = dict(contract.get("resolved_manifest_counts") or {})
    resolved: dict[str, ManifestResolution] = {}
    for corpus in CODE_CORPORA:
        manifest_path = _repo_path(config.repo_root, Path(str(paths.get(corpus) or "")))
        actual_sha = _sha256(manifest_path) if manifest_path.is_file() else ""
        declared_sha = str(declared.get(corpus) or "")
        ready = bool(contract.get(f"{corpus}_ready") and declared_sha and actual_sha == declared_sha)
        resolved[corpus] = ManifestResolution(
            corpus=corpus,
            path=manifest_path,
            declared_sha256=declared_sha,
            actual_sha256=actual_sha,
            count=int(counts.get(corpus) or 0),
            ready=ready,
        )
    return resolved, bool(contract.get("manifest_contract_ready") and all(r.ready for r in resolved.values()))


def _eligible_mbpp(row: dict[str, Any]) -> bool:
    tests = row.get("tests")
    return bool(row.get("stable_id") and row.get("canonical_code") and isinstance(tests, list) and tests)


def _eligible_humaneval(row: dict[str, Any]) -> bool:
    return bool(
        row.get("stable_id")
        and row.get("prompt")
        and row.get("canonical_solution")
        and row.get("entry_point")
        and row.get("tests")
    )


def _select_rows(resolved: dict[str, ManifestResolution]) -> dict[str, dict[str, Any]]:
    mbpp_rows = _read_jsonl(resolved["mbpp"].path)
    humaneval_rows = _read_jsonl(resolved["humaneval"].path)
    return {
        "mbpp": next((row for row in mbpp_rows if _eligible_mbpp(row)), {}),
        "humaneval": next((row for row in humaneval_rows if _eligible_humaneval(row)), {}),
    }


def _mbpp_script(row: dict[str, Any]) -> tuple[str, int, dict[str, bool]]:
    tests = [str(test) for test in row["tests"]]
    imports = "\n".join(str(item) for item in row.get("test_imports") or [])
    script = (
        f"{imports}\n{row['canonical_code']}\n"
        "\ndef __carnot_pilot__():\n    "
        + "\n    ".join(tests)
        + "\n    return True\n"
    )
    return script, len(tests), {
        "manifest_contract_checksum": True,
        "canonical_reference_solution": True,
        "mbpp_assert_tests": True,
        "official_check": False,
        "gvisor_runsc_sandbox": True,
        "timeout_enforced": True,
        "no_llm_generation": True,
    }


def _humaneval_script(row: dict[str, Any]) -> tuple[str, int, dict[str, bool]]:
    tests = str(row["tests"])
    entry_point = str(row["entry_point"])
    script = (
        f"{row['prompt']}{row['canonical_solution']}\n{tests}\n"
        "\ndef __carnot_pilot__():\n"
        f"    check({entry_point})\n"
        "    return True\n"
    )
    return script, tests.count("assert "), {
        "manifest_contract_checksum": True,
        "canonical_reference_solution": True,
        "mbpp_assert_tests": False,
        "official_check": True,
        "gvisor_runsc_sandbox": True,
        "timeout_enforced": True,
        "no_llm_generation": True,
    }


def _script_for_row(corpus: str, row: dict[str, Any]) -> tuple[str, int, dict[str, bool]]:
    return _mbpp_script(row) if corpus == "mbpp" else _humaneval_script(row)


def execute_script_in_sandbox(
    script: str,
    timeout_s: float,
    sandbox_function: SandboxFunction = sandboxed_exec_function,
) -> ExecutionOutcome:
    """Execute the pilot harness through the existing sandbox without fallback."""

    result, error = sandbox_function(
        script,
        "__carnot_pilot__",
        (),
        timeout=timeout_s,
        allow_fallback=False,
    )
    if error is None and result is True:
        return ExecutionOutcome(passed=True)
    if error is None:
        return ExecutionOutcome(
            passed=False,
            error_type="AssertionError",
            error_message=f"pilot harness returned {result!r}",
        )
    return ExecutionOutcome(
        passed=False,
        error_type=type(error).__name__,
        error_message=str(error),
        timed_out=isinstance(error, TimeoutError),
    )


def _source_artifacts(
    config: ExperimentConfig,
    resolved: dict[str, ManifestResolution],
) -> tuple[list[str], dict[str, str]]:
    paths = [
        _repo_path(config.repo_root, config.manifest_contract_path),
        _repo_path(config.repo_root, config.cross_corpus_matrix_path),
        resolved["mbpp"].path,
        resolved["humaneval"].path,
    ]
    names = [_source_name(config.repo_root, path) for path in paths]
    return names, {name: _sha256(path) for name, path in zip(names, paths, strict=True)}


def _base_artifact(
    config: ExperimentConfig,
    started: float,
    resolved: dict[str, ManifestResolution],
) -> dict[str, Any]:
    source_artifacts, source_sha = _source_artifacts(config, resolved)
    return {
        "artifact": "experiment_2879_code_corpus_manifest_execution_pilot_v1",
        "schema": "carnot.code_corpus_manifest_execution_pilot.v1",
        "source_artifacts": source_artifacts,
        "source_artifact_sha256": source_sha,
        "manifest_paths": {corpus: str(resolved[corpus].path) for corpus in CODE_CORPORA},
        "manifest_declared_sha256": {
            corpus: resolved[corpus].declared_sha256 for corpus in CODE_CORPORA
        },
        "manifest_actual_sha256": {corpus: resolved[corpus].actual_sha256 for corpus in CODE_CORPORA},
        "manifest_checksum_verified": {corpus: resolved[corpus].ready for corpus in CODE_CORPORA},
        "manifest_counts": {corpus: resolved[corpus].count for corpus in CODE_CORPORA},
        "selection_rule": SELECTION_RULE,
        "n_mbpp_rows": 0,
        "n_humaneval_rows": 0,
        "deterministic_execution_used": False,
        "sandbox_status": "",
        "pilot_rows": [],
        "selection_checksums": {},
        "headline_metric_claim_made": False,
        "tests_run": list(config.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": max(0.0, config.clock() - started),
    }


def _pilot_row(
    *,
    corpus: str,
    row: dict[str, Any],
    script: str,
    n_tests: int,
    feature_coverage: dict[str, bool],
    outcome: ExecutionOutcome,
    manifest: ManifestResolution,
) -> dict[str, Any]:
    return {
        "corpus": "MBPP" if corpus == "mbpp" else "HumanEval",
        "stable_id": str(row["stable_id"]),
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.declared_sha256,
        "row_sha256": _stable_json_sha256(row),
        "execution_payload_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
        "reference_source": "canonical_code" if corpus == "mbpp" else "prompt+canonical_solution",
        "tests_source": "MBPP assert tests" if corpus == "mbpp" else "HumanEval check(candidate)",
        "n_tests": n_tests,
        "passed": outcome.passed,
        "error_type": outcome.error_type,
        "error_message": outcome.error_message,
        "timed_out": outcome.timed_out,
        "verifier_feature_coverage": dict(feature_coverage),
    }


def build_experiment_artifact(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    sandbox_status_provider: Callable[[], dict[str, Any]] = get_sandbox_status,
    executor: Executor = execute_script_in_sandbox,
) -> dict[str, Any]:
    """Build the Exp 2879 artifact without generating any candidate code."""

    started = config.start_time()
    resolved, manifest_contract_ready = _resolve_code_manifests(config)
    artifact = _base_artifact(config, started, resolved)
    artifact["manifest_contract_ready"] = manifest_contract_ready
    if not manifest_contract_ready:
        artifact["honest_verdict"] = "blocked_manifest_contract"
        artifact["code_manifest_pilot_ready"] = False
        artifact["sandbox_status"] = "not_checked_manifest_contract_blocked"
        return artifact

    sandbox_status = sandbox_status_provider()
    sandbox_ready = bool(sandbox_status.get("available") and sandbox_status.get("runtime") == "runsc")
    artifact["sandbox_status"] = (
        "available: runsc" if sandbox_ready else "blocked_sandbox: runsc unavailable"
    )
    if not sandbox_ready:
        artifact["honest_verdict"] = "blocked_sandbox"
        artifact["code_manifest_pilot_ready"] = False
        return artifact

    selected = _select_rows(resolved)
    artifact["selection_checksums"] = {
        str(row["stable_id"]): _stable_json_sha256(row) for row in selected.values() if row
    }
    if not all(selected.values()):
        artifact["honest_verdict"] = "blocked_no_eligible_code_rows"
        artifact["code_manifest_pilot_ready"] = False
        return artifact

    pilot_rows = []
    for corpus in CODE_CORPORA:
        script, n_tests, feature_coverage = _script_for_row(corpus, selected[corpus])
        outcome = executor(script, config.timeout_s)
        pilot_rows.append(
            _pilot_row(
                corpus=corpus,
                row=selected[corpus],
                script=script,
                n_tests=n_tests,
                feature_coverage=feature_coverage,
                outcome=outcome,
                manifest=resolved[corpus],
            )
        )

    ready = all(row["passed"] for row in pilot_rows)
    artifact.update(
        {
            "honest_verdict": (
                "complete: MBPP/HumanEval manifest-only execution pilot ready"
                if ready
                else "blocked_execution_failure"
            ),
            "code_manifest_pilot_ready": ready,
            "n_mbpp_rows": sum(1 for row in pilot_rows if row["corpus"] == "MBPP"),
            "n_humaneval_rows": sum(1 for row in pilot_rows if row["corpus"] == "HumanEval"),
            "deterministic_execution_used": True,
            "pilot_rows": pilot_rows,
        }
    )
    return artifact


def write_experiment_artifact(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    sandbox_status_provider: Callable[[], dict[str, Any]] = get_sandbox_status,
    executor: Executor = execute_script_in_sandbox,
) -> dict[str, Any]:
    """Write the Exp 2879 manifest-only pilot artifact to ``results/``."""

    artifact = build_experiment_artifact(
        config,
        sandbox_status_provider=sandbox_status_provider,
        executor=executor,
    )
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
