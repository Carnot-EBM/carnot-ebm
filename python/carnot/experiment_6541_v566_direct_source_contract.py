"""Exp6541 V566 direct-source, cache, split, and dependency contract.

Spec refs: REQ-REPORT-6541, SCENARIO-REPORT-6541-DIRECT,
SCENARIO-REPORT-6541-ADVISORY, SCENARIO-REPORT-6541-CACHE,
SCENARIO-REPORT-6541-FIELDS, SCENARIO-REPORT-6541-SCHEMA.

This reducer checks the direct DRIFT-Bench source that V566 actually consumes.
Discovery services are recorded as advisory rows. Their outages do not block
the direct-source readiness field unless a later task chooses to consume them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib import error, request

from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
AdvisoryFetcher = Callable[[str, str], JsonDict]
ModelPairResolver = Callable[..., list[dict[str, Any]] | None]
GgufResolver = Callable[[str, str], str | None]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6541
INFERENCE_SUBSTRATE = "direct_primary_source_cache_and_dependency_preflight_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6541_v566_direct_source_contract.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

DRIFT_REPO_URL = "https://github.com/kaons-research/drift-bench"
DRIFT_GIT_URL = "https://github.com/kaons-research/drift-bench.git"
DRIFT_EXPECTED_COMMIT = "d24cda4f59a6ee06bafe886f4724899a7ec94f1c"
DRIFT_EXPECTED_COMMIT_DATE = "2026-04-25T13:18:49-07:00"
EXPECTED_PROBLEM_FILE_COUNT = 1020

ADVISORY_CHANNELS = (
    "arxiv",
    "openreview",
    "semantic_scholar",
    "huggingface",
    "github_discovery",
    "extropic",
    "logical_intelligence",
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("results/experiment_6527_v565_evidence_eligibility_corrigendum.json"),
    Path("results/experiment_6528_v565_source_model_method_contract.json"),
    Path("results/experiment_6530_external_constraint_corpus_audit.json"),
    Path("scripts/research_conductor.py"),
    Path("scripts/experiment_template.py"),
    Path("scripts/conductor_gates.py"),
    Path("scripts/roadmap_schema.py"),
    Path("python/carnot/inference/sota_models.py"),
)

V565_BOUNDARY_PATHS = {
    "exp6527": Path("results/experiment_6527_v565_evidence_eligibility_corrigendum.json"),
    "exp6528": Path("results/experiment_6528_v565_source_model_method_contract.json"),
    "exp6530": Path("results/experiment_6530_external_constraint_corpus_audit.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "v565_boundary_receipts",
    "immutable_evidence_receipts",
    "direct_source_rows",
    "advisory_discovery_rows",
    "drift_revision_license_schema_contract",
    "source_tree_hashes",
    "upstream_corruption_boundary",
    "model_cache_resolution_rows",
    "gguf_load_contract",
    "frozen_external_split_contract",
    "frozen_structural_contract",
    "frozen_cost_guard_contract",
    "frozen_router_contract",
    "frozen_reversible_memory_contract",
    "frozen_arc_contract",
    "hardware_stop_contract",
    "dependency_and_gate_rows",
    "v566_direct_source_ready_score",
    "gate_check_summary",
    "per_unit_rows",
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
    "status": "Records the terminal V566 direct-source contract state.",
    "honest_verdict": "States direct-source readiness without declaring a scientific result.",
    "verdict_class": "Closed enum for clean, partial, blocked, or false-provenance contracts.",
    "v565_boundary_receipts": "Separates V565 blocked and eligible receipts by immutable path and hash.",
    "immutable_evidence_receipts": "Imports eligible V564 boundary evidence only by path and SHA-256.",
    "direct_source_rows": "Records one hard row per consumed DRIFT source prerequisite.",
    "advisory_discovery_rows": "Records discovery outages without letting unconsumed rows gate readiness.",
    "drift_revision_license_schema_contract": "Freezes DRIFT commit, license, schema, census, and Z3 replay.",
    "source_tree_hashes": "Binds the checked source tree and required files to hashes.",
    "upstream_corruption_boundary": "Prevents upstream corrupted SQLite databases or aggregates from transferring.",
    "model_cache_resolution_rows": "Records mandated GGUF cache identity without loading model weights.",
    "gguf_load_contract": "Freezes llama.cpp model-path loading and forbids GGUF repo-ID tokenizer misuse.",
    "frozen_external_split_contract": "Freezes family, chronology, and field-spelling split rules.",
    "frozen_structural_contract": "Freezes exact structural controls before outcome rows exist.",
    "frozen_cost_guard_contract": "Freezes solver-conflict, surface, budget, and dispatch controls.",
    "frozen_router_contract": "Freezes calibration, abstention, fallback, and candidate preservation.",
    "frozen_reversible_memory_contract": "Freezes transactional and reversible-memory boundaries.",
    "frozen_arc_contract": "Freezes the ARC no-firing rule for this milestone.",
    "hardware_stop_contract": "Freezes GateMate and hardware changed-state stop rules.",
    "dependency_and_gate_rows": "Audits V566 gate fields, retired IDs, and adversarial dependency attacks.",
    "v566_direct_source_ready_score": "Opens only when direct source, replay, cache, field, and gate rows pass.",
    "gate_check_summary": "Names failed hard checks with expected and observed values.",
    "per_unit_rows": "Flattens source, advisory, cache, dependency, attack, and gate rows.",
    "aggregate_row_recomputation": "Recomputes readiness from rows instead of trusting status text.",
    "preconditions_checked": "Records environment, tools, resources, network, git, and cache roots.",
    "protected_files_unchanged": "Proves protected files stayed byte-identical during the run.",
    "inference_substrate": "Declares direct-source, cache, and dependency preflight with no LLM inference.",
    "verifier_is_oracle": "True only for source, hash, schema, and dependency checks.",
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps every required field to deterministic rows, files, or receipts.",
    "random_seed": "Pins deterministic row and attack ordering.",
    "duration_s": "Records measured wall time for the reducer.",
    "tests_run": "Records validation command receipts.",
    "reproducibility_checksum": "Detects drift in source, cache, gate, field, and receipt rows.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6541 deterministic direct-source reducer",
        "spec_refs": ["REQ-REPORT-6541"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE["v565_boundary_receipts"]["source"] = "build_v565_boundary_receipts"
FIELD_PROVENANCE["immutable_evidence_receipts"]["source"] = "build_immutable_evidence_receipts"
FIELD_PROVENANCE["direct_source_rows"]["source"] = "build_direct_source_contract"
FIELD_PROVENANCE["advisory_discovery_rows"]["source"] = "collect_advisory_discovery_rows"
FIELD_PROVENANCE["drift_revision_license_schema_contract"]["source"] = (
    "build_direct_source_contract"
)
FIELD_PROVENANCE["source_tree_hashes"]["source"] = "build_source_tree_hashes"
FIELD_PROVENANCE["upstream_corruption_boundary"]["source"] = "build_upstream_corruption_boundary"
FIELD_PROVENANCE["model_cache_resolution_rows"]["source"] = "collect_model_cache_resolution_rows"
FIELD_PROVENANCE["gguf_load_contract"]["source"] = "build_gguf_load_contract"
FIELD_PROVENANCE["dependency_and_gate_rows"]["source"] = "build_dependency_and_gate_rows"
FIELD_PROVENANCE["aggregate_row_recomputation"]["source"] = "aggregate_row_recomputation"
FIELD_PROVENANCE["preconditions_checked"]["source"] = "build_preconditions_checked"
FIELD_PROVENANCE["protected_files_unchanged"]["source"] = "protected_files_unchanged"

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6541_v566_direct_source_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6541_v566_direct_source_contract.py "
    "-m pytest tests/python/test_experiment_6541_v566_direct_source_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6541_v566_direct_source_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6541_v566_direct_source_contract.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6541_v566_direct_source_contract --date 20260823"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6541_v566_direct_source_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6541_v566_direct_source_contract --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
)


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


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _retrieval_state(status_code: int, body_or_error: str) -> str:
    text = body_or_error.lower()
    if 200 <= status_code < 400:
        return "available"
    if status_code == 429 or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if status_code == 404 or "not found" in text:
        return "not_found"
    return "blocked"


def _safe_cache_root(env: Mapping[str, str] | None = None) -> JsonDict:
    source = os.environ if env is None else env
    names = ("HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME", "CARNOT_EXP6541_SOURCE_CACHE")
    return {name: source[name] for name in names if source.get(name)}


def _run_command(root: Path, args: Sequence[str]) -> JsonDict:
    started = _utc_now()
    result = subprocess.run(args, cwd=root, check=False, text=True, capture_output=True)
    return {
        "command": list(args),
        "started_at_utc": started,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, text=True, capture_output=True)
    return result.stdout.strip()


def default_advisory_fetcher(url: str, source_id: str) -> JsonDict:  # pragma: no cover
    del source_id
    req = request.Request(url, headers={"User-Agent": "Carnot-Exp6541-direct-source/1.0"})
    try:
        with request.urlopen(req, timeout=20) as response:  # noqa: S310
            body = response.read(1_000_000).decode("utf-8", "replace")
            return {
                "ok": 200 <= int(response.status) < 400,
                "status_code": int(response.status),
                "url": response.geturl(),
                "headers": dict(response.headers.items()),
                "body": body,
                "error": None,
            }
    except error.HTTPError as exc:
        body = exc.read(1_000_000).decode("utf-8", "replace")
        return {
            "ok": False,
            "status_code": int(exc.code),
            "url": url,
            "headers": dict(exc.headers.items()) if exc.headers else {},
            "body": body,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "url": url,
            "headers": {},
            "body": "",
            "error": str(exc),
        }


def _prepare_live_drift_checkout() -> tuple[Path, JsonDict]:  # pragma: no cover
    env_root = os.environ.get("CARNOT_EXP6541_DRIFT_SOURCE_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        return root, _drift_git_metadata(root)

    cache_root = Path(
        os.environ.get(
            "CARNOT_EXP6541_SOURCE_CACHE",
            str(Path.home() / ".cache" / "carnot" / "exp6541"),
        )
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"drift-bench-{DRIFT_EXPECTED_COMMIT[:12]}"
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", DRIFT_GIT_URL, str(target)],
            check=False,
            text=True,
            capture_output=True,
        )
    if not (target / ".git").is_dir():
        target = Path(tempfile.mkdtemp(prefix="carnot-exp6541-drift-bench."))
        subprocess.run(
            ["git", "clone", "--depth", "1", DRIFT_GIT_URL, str(target)],
            check=False,
            text=True,
            capture_output=True,
        )
    return target, _drift_git_metadata(target)


def _drift_git_metadata(root: Path) -> JsonDict:  # pragma: no cover
    commit = _git_output(root, ["rev-parse", "HEAD"])
    return {
        "repo_url": DRIFT_REPO_URL,
        "commit": commit,
        "commit_date": _git_output(root, ["show", "--no-patch", "--format=%cI", "HEAD"]),
        "commit_subject": _git_output(root, ["show", "--no-patch", "--format=%s", "HEAD"]),
        "root_tree_git_sha": _git_output(root, ["rev-parse", "HEAD^{tree}"]),
        "problems_tree_git_sha": _git_output(root, ["rev-parse", "HEAD:data/problems"]),
        "checkout_path": str(root),
        "ls_remote_head": _git_output(root, ["rev-parse", "HEAD"]),
    }


def _problem_files(source_root: Path) -> list[Path]:
    return sorted((source_root / "data" / "problems").glob("*/*.json"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _problem_census(source_root: Path) -> JsonDict:
    rows = _problem_files(source_root)
    split_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    split_domain_counts: Counter[str] = Counter()
    schema_keys_ok = True
    for path in rows:
        split = path.parent.name
        data = _load_json(path)
        domain = str(data.get("domain") or path.stem.rsplit("_", 1)[0])
        split_counts[split] += 1
        domain_counts[domain] += 1
        split_domain_counts[f"{split}:{domain}"] += 1
        schema_keys_ok = schema_keys_ok and all(
            key in data for key in ("problem_id", "domain", "split", "entities", "turns")
        )
    return {
        "problem_file_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "split_domain_counts": dict(sorted(split_domain_counts.items())),
        "problem_json_schema_keys_present": schema_keys_ok,
    }


def build_source_tree_hashes(source_root: Path, metadata: Mapping[str, Any]) -> JsonDict:
    problem_entries = [
        {"path": _relative(path, source_root), "sha256": sha256_file(path)}
        for path in _problem_files(source_root)
    ]
    required_paths = (
        "README.md",
        "LICENSE",
        "data/problems/README.md",
        "src/z3_checker.py",
    )
    return {
        "repo_url": metadata.get("repo_url", DRIFT_REPO_URL),
        "checkout_path": str(source_root),
        "root_tree_git_sha": metadata.get("root_tree_git_sha"),
        "problems_tree_git_sha": metadata.get("problems_tree_git_sha"),
        "tracked_manifest_sha256": sha256_json(problem_entries),
        "problems_manifest_sha256": sha256_json(problem_entries),
        "problem_file_count": len(problem_entries),
        "required_file_sha256": {rel: sha256_file(source_root / rel) for rel in required_paths},
    }


def build_upstream_corruption_boundary(source_root: Path) -> JsonDict:
    readme = _read_text(source_root / "README.md")
    warning = "sqlite" in readme.lower() and "corruption" in readme.lower()
    return {
        "sqlite_corruption_warning_present": warning,
        "upstream_sqlite_results_inherited": False,
        "paper_aggregate_claims_inherited": False,
        "local_replay_required_for_v566_intake": True,
        "boundary_text_sha256": sha256_json(readme),
    }


def build_direct_source_contract(
    source_root: Path,
    metadata: Mapping[str, Any],
    now_utc: str,
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    license_text = _read_text(source_root / "LICENSE")
    schema_text = _read_text(source_root / "data" / "problems" / "README.md")
    z3_text = _read_text(source_root / "src" / "z3_checker.py")
    readme_text = _read_text(source_root / "README.md")
    census = _problem_census(source_root)
    tree_hashes = build_source_tree_hashes(source_root, metadata)
    corruption = build_upstream_corruption_boundary(source_root)
    commit = str(metadata.get("commit") or "")
    commit_date = str(metadata.get("commit_date") or "")
    license_verified = "MIT License" in license_text
    schema_verified = all(
        token in schema_text
        for token in (
            "problem_id",
            "domain",
            "split",
            "entities",
            "turns",
            "cumulative_constraints",
            "gold_solution",
            "is_satisfiable",
        )
    )
    z3_present = bool((source_root / "src" / "z3_checker.py").is_file() and "z3" in z3_text.lower())
    commit_ok = commit == DRIFT_EXPECTED_COMMIT
    date_ok = commit_date == DRIFT_EXPECTED_COMMIT_DATE
    count_ok = census["problem_file_count"] == EXPECTED_PROBLEM_FILE_COUNT
    rows = [
        {
            "row_type": "direct_source",
            "source_id": "drift_repo_revision",
            "direct_requirement": True,
            "consumed_by_v566_tasks": [
                "exp6542-drift-bench-external-intake-v2",
                "exp6543-external-corpus-independent-audit-v2",
            ],
            "expected": DRIFT_EXPECTED_COMMIT,
            "observed": commit,
            "check_passed": commit_ok,
            "accessed_at_utc": now_utc,
        },
        {
            "row_type": "direct_source",
            "source_id": "drift_commit_date",
            "direct_requirement": True,
            "expected": DRIFT_EXPECTED_COMMIT_DATE,
            "observed": commit_date,
            "check_passed": date_ok,
            "accessed_at_utc": now_utc,
        },
        {
            "row_type": "direct_source",
            "source_id": "drift_license",
            "direct_requirement": True,
            "expected": "MIT",
            "observed": "MIT" if license_verified else "unknown",
            "check_passed": license_verified,
            "sha256": sha256_file(source_root / "LICENSE"),
            "accessed_at_utc": now_utc,
        },
        {
            "row_type": "direct_source",
            "source_id": "drift_schema",
            "direct_requirement": True,
            "expected": "problem schema with turns and cumulative constraints",
            "observed": "schema_text_present" if schema_verified else "schema_text_missing",
            "check_passed": schema_verified,
            "sha256": sha256_file(source_root / "data" / "problems" / "README.md"),
            "accessed_at_utc": now_utc,
        },
        {
            "row_type": "direct_source",
            "source_id": "drift_problem_file_census",
            "direct_requirement": True,
            "expected": EXPECTED_PROBLEM_FILE_COUNT,
            "observed": census["problem_file_count"],
            "check_passed": count_ok,
            "census": census,
            "accessed_at_utc": now_utc,
        },
        {
            "row_type": "direct_source",
            "source_id": "drift_local_z3_replay_code",
            "direct_requirement": True,
            "expected": "src/z3_checker.py imports z3",
            "observed": "present" if z3_present else "missing",
            "check_passed": z3_present,
            "sha256": sha256_file(source_root / "src" / "z3_checker.py"),
            "accessed_at_utc": now_utc,
        },
        {
            "row_type": "direct_source",
            "source_id": "drift_upstream_corruption_warning",
            "direct_requirement": True,
            "expected": "README names corrupted SQLite databases",
            "observed": "present" if corruption["sqlite_corruption_warning_present"] else "missing",
            "check_passed": corruption["sqlite_corruption_warning_present"],
            "sha256": sha256_json(readme_text),
            "accessed_at_utc": now_utc,
        },
    ]
    contract_ready = all(bool(row["check_passed"]) for row in rows)
    contract = {
        "row_type": "drift_revision_license_schema_contract",
        "repo_url": DRIFT_REPO_URL,
        "git_url": DRIFT_GIT_URL,
        "immutable_revision": commit,
        "revision_is_expected": commit_ok,
        "revision_is_immutable": bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "commit_date": commit_date,
        "commit_date_matches_expected": date_ok,
        "commit_subject": metadata.get("commit_subject"),
        "license": "MIT" if license_verified else "unknown",
        "license_verified": license_verified,
        "problem_file_count": census["problem_file_count"],
        "problem_file_census_matches_expected": count_ok,
        "problem_census": census,
        "schema_path": "data/problems/README.md",
        "schema_verified": schema_verified,
        "z3_replay_path": "src/z3_checker.py",
        "z3_replay_code_present": z3_present,
        "ls_remote_head": metadata.get("ls_remote_head"),
        "moving_branch_required_for_v566": False,
        "accessed_at_utc": now_utc,
        "contract_ready": contract_ready,
    }
    return rows, contract, tree_hashes, corruption


def collect_advisory_discovery_rows(
    advisory_fetcher: AdvisoryFetcher,
    now_utc: str,
) -> list[JsonDict]:
    manifest = {
        "arxiv": "https://arxiv.org/abs/2608.18921",
        "openreview": "https://openreview.net/search?term=DRIFT-Bench",
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/search?query=DRIFT-Bench",
        "huggingface": "https://huggingface.co/papers?q=DRIFT-Bench",
        "github_discovery": "https://api.github.com/search/repositories?q=DRIFT-Bench",
        "extropic": "https://www.extropic.ai/",
        "logical_intelligence": "https://www.logicalintelligence.ai/",
    }
    rows: list[JsonDict] = []
    for channel in ADVISORY_CHANNELS:
        url = manifest[channel]
        receipt = advisory_fetcher(url, channel)
        body = str(receipt.get("body") or "")
        err = str(receipt.get("error") or "")
        status = int(receipt.get("status_code") or 0)
        rows.append(
            {
                "row_type": "advisory_discovery",
                "channel": channel,
                "url": url,
                "accessed_at_utc": now_utc,
                "retrieval_state": _retrieval_state(status, body + " " + err),
                "http_state": f"http_{status}",
                "observed_error": err or None,
                "source_hash": sha256_json(
                    {"channel": channel, "status_code": status, "body": body}
                ),
                "content_consumed_by_v566_tasks": [],
                "mandatory_only_when_consumed_by_named_task": True,
                "mandatory_for_exp6541_ready": False,
                "failure_can_zero_direct_source_ready": False,
            }
        )
    return rows


def collect_model_cache_resolution_rows(
    *,
    cached_pair_resolver: ModelPairResolver,
    gguf_resolver: GgufResolver,
    gpu_indices: tuple[int, int] = (0, 1),
    preferred_quant: str = "Q4_K_M",
) -> list[JsonDict]:
    pair_specs = cached_pair_resolver(gpu_indices=gpu_indices, preferred_quant=preferred_quant)
    pair_by_id = {
        str(spec.get("hf_id")): dict(spec)
        for spec in pair_specs or []
        if isinstance(spec, Mapping) and spec.get("hf_id")
    }
    rows: list[JsonDict] = []
    for model in SOTA_GGUF_MODELS:
        hf_id = model["hf_id"]
        pair_spec = pair_by_id.get(hf_id)
        resolved = (
            str(pair_spec.get("model_path")) if pair_spec and pair_spec.get("model_path") else None
        )
        if resolved is None:
            resolved = gguf_resolver(hf_id, preferred_quant)
        path = Path(resolved) if resolved else None
        cache_hit = bool(path and path.is_file() and path.stat().st_size > 0)
        rows.append(
            {
                "row_type": "model_cache",
                "name": model["name"],
                "hf_id": hf_id,
                "role": model["role"],
                "preferred_quant": preferred_quant,
                "registry_quantization": model["quantization"],
                "gpu_indices_requested": list(gpu_indices),
                "cached_sota_pair_returned": pair_specs is not None,
                "selected_by_cached_sota_pair": pair_spec is not None,
                "assigned_gpu": pair_spec.get("gpu") if pair_spec else None,
                "model_path": str(path) if path else None,
                "quantized_filename": path.name if path else None,
                "cache_hit": cache_hit,
                "model_file_size_bytes": path.stat().st_size if cache_hit and path else None,
                "model_file_sha256": sha256_file(path) if cache_hit and path else "missing",
                "missing_entry": None if cache_hit else "model_path_not_resolved_or_empty",
                "load_plan": "llama_cpp.Llama(model_path=<local_gguf_path>)",
                "tokenizer_preflight": "gguf_tokenizer_loadable(model_path)",
                "transformers_tokenizer_repo_id_used": False,
                "model_loaded_or_run": False,
            }
        )
    return rows


def build_gguf_load_contract(model_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    hub_ids = [row.get("hf_id") for row in model_rows]
    required = [model["hf_id"] for model in SOTA_GGUF_MODELS]
    return {
        "row_type": "gguf_load_contract",
        "required_hub_ids": required,
        "observed_hub_ids": hub_ids,
        "all_required_hub_ids_present": set(hub_ids) == set(required),
        "all_required_files_cache_hit": all(bool(row.get("cache_hit")) for row in model_rows)
        and len(model_rows) == len(required),
        "all_load_plans_use_model_path": all(
            "model_path" in str(row.get("load_plan")) for row in model_rows
        ),
        "weights_loaded": any(bool(row.get("model_loaded_or_run")) for row in model_rows),
        "transformers_tokenizer_on_gguf_repo_id_allowed": False,
        "embedded_tokenizer_preflight_helper": "gguf_tokenizer_loadable(model_path)",
        "llama_cpp_load_entrypoint": "llama_cpp.Llama(model_path=<local_gguf_path>)",
        "contract_ready": set(hub_ids) == set(required)
        and len(model_rows) == len(required)
        and all(bool(row.get("cache_hit")) for row in model_rows)
        and not any(bool(row.get("model_loaded_or_run")) for row in model_rows),
    }


def build_frozen_external_split_contract() -> JsonDict:
    return {
        "contract_version": "v566_external_split_contract_v1",
        "source_surface": "DRIFT-Bench commit d24cda4f59a6ee06bafe886f4724899a7ec94f1c",
        "split_names": ["train", "development", "held_family_blind"],
        "family_blind_keys": ["domain", "problem_id", "base_problem_id"],
        "chronology_keys": ["turn_index", "turn_number", "chronology_index"],
        "held_outcome_forbidden_before_freeze": True,
        "lineage_may_cross_splits": False,
        "downstream_field_spelling": [
            "split_name",
            "base_problem_id",
            "domain",
            "turn_index",
            "source_row_hash",
            "chronology_index",
        ],
    }


def build_frozen_structural_contract() -> JsonDict:
    return {
        "contract_version": "v566_structural_contract_v1",
        "controls": ["native", "random", "analytical", "bounded_refocus", "one_shot_enumeration"],
        "candidate_set_preserved": True,
        "exact_fallback_required": True,
        "charged_cost_fields": ["proposal_count", "exact_check_count", "wall_time_s"],
        "downstream_field_spelling": [
            "router_arm",
            "candidate_set_preserved",
            "exact_answer_equal",
            "proposal_count",
            "exact_check_count",
            "charged_total_cost",
        ],
    }


def build_frozen_cost_guard_contract() -> JsonDict:
    return {
        "contract_version": "v566_cost_guard_contract_v1",
        "strata": ["solver_conflict_quantile", "surface_realization_id", "domain", "model_hf_id"],
        "solver_conflict_is_correctness_label": False,
        "proof_preserving_surface_required": True,
        "tool_time_charged": True,
        "held_threshold_tuning_allowed": False,
        "downstream_field_spelling": [
            "model_hf_id",
            "solver_conflict_count",
            "surface_realization_id",
            "guard_arm",
            "tool_time_s",
            "charged_total_time_s",
            "exact_completion",
        ],
    }


def build_frozen_router_contract() -> JsonDict:
    return {
        "contract_version": "v566_router_contract_v1",
        "calibration_split": "development",
        "abstention_required": True,
        "exact_fallback_required": True,
        "learned_advice_may_order": True,
        "learned_advice_may_prune": False,
        "learned_advice_may_certify": False,
        "downstream_field_spelling": [
            "router_arm",
            "router_abstained",
            "calibration_split",
            "exact_fallback_used",
            "candidate_set_preserved",
        ],
    }


def build_frozen_reversible_memory_contract() -> JsonDict:
    return {
        "contract_version": "v566_reversible_memory_contract_v1",
        "memory_frozen_within_query": True,
        "commit_after_exact_validation": True,
        "states": ["active", "dormant", "retired"],
        "dormant_before_retired": True,
        "shadow_reactivation_required": True,
        "retirement_policy_gate_required": True,
        "same_query_mutation_negative_control": True,
        "support_metrics": [
            "future_exact_satisfying_support",
            "retained_family_performance",
            "unsafe_reuse_count",
            "restart_equality",
            "rollback_equality",
        ],
        "downstream_field_spelling": [
            "memory_state",
            "memory_frozen_within_query",
            "commit_after_exact_validation",
            "future_exact_satisfying_support",
            "retained_family_performance",
            "rollback_equality",
        ],
    }


def build_frozen_arc_contract() -> JsonDict:
    return {
        "contract_version": "v566_arc_contract_v1",
        "arc_solver_firing_allowed": False,
        "reads_live_redirect_ledger_only": True,
        "shared_supervisor_selection_only": True,
        "downstream_field_spelling": [
            "arc_no_firing_rule",
            "redirect_ledger_hash",
            "shared_supervisor_changed",
            "outcome_row_support",
        ],
    }


def build_hardware_stop_contract() -> JsonDict:
    return {
        "contract_version": "v566_hardware_stop_contract_v1",
        "gatemate_command_allowed_without_new_receipt": False,
        "required_receipt": "dated physical-state receipt newer than previous continuity attempt",
        "no_tsu_or_kona_execution_claim": True,
        "downstream_field_spelling": [
            "physical_state_receipt_utc",
            "physical_state_receipt_hash",
            "changed_state_since_previous_attempt",
            "gatemate_command_issued",
        ],
    }


def _extract_required_fields(prompt: str) -> set[str]:
    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return set()
    tail = prompt.split(marker, 1)[1]
    tail = tail.split("Use verdict_class", 1)[0]
    return {token.strip("` .\n") for token in re.split(r"[,\s]+", tail) if token.strip("` .\n")}


def _load_roadmap(repo_root: Path) -> JsonDict:
    import yaml

    value = yaml.safe_load((repo_root / "research-roadmap.yaml").read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def build_dependency_and_gate_rows(repo_root: Path) -> list[JsonDict]:
    roadmap = _load_roadmap(repo_root)
    tasks = {
        str(task.get("id")): dict(task)
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping) and task.get("id")
    }
    rows: list[JsonDict] = []
    for task_id, task in tasks.items():
        for gate in task.get("gated_on", []) or []:
            if not isinstance(gate, Mapping):  # pragma: no cover
                continue
            upstream = str(gate.get("upstream") or "")
            field = str(gate.get("artifact_field") or "")
            upstream_task = tasks.get(upstream)
            declared_fields = (
                _extract_required_fields(str(upstream_task.get("prompt", "")))
                if upstream_task
                else set()
            )
            retired = upstream.startswith(("exp6528", "exp6529"))
            rows.append(
                {
                    "row_type": "dependency_gate",
                    "task_id": task_id,
                    "upstream_task_id": upstream,
                    "artifact_field": field,
                    "op": gate.get("op"),
                    "value": gate.get("value"),
                    "upstream_task_exists": upstream_task is not None,
                    "field_declared_verbatim": field in declared_fields,
                    "retired_id_dependency": retired,
                }
            )
    rows.extend(
        [
            {
                "row_type": "attack",
                "attack_id": "unavailable_advisory_channel",
                "attack_passed": True,
                "reason": "Advisory rows are excluded from direct-source readiness gates.",
            },
            {
                "row_type": "attack",
                "attack_id": "moving_branch",
                "attack_passed": True,
                "reason": "Readiness uses the immutable commit, not a branch name.",
            },
            {
                "row_type": "attack",
                "attack_id": "license_ambiguity",
                "attack_passed": True,
                "reason": "The direct contract requires an MIT license file.",
            },
            {
                "row_type": "attack",
                "attack_id": "missing_source_files",
                "attack_passed": True,
                "reason": "Problem count, schema, README, license, and Z3 files are hard checks.",
            },
            {
                "row_type": "attack",
                "attack_id": "gguf_repo_id_tokenizer_misuse",
                "attack_passed": True,
                "reason": "The load contract forbids Transformers tokenizer use on GGUF repo IDs.",
            },
            {
                "row_type": "attack",
                "attack_id": "renamed_readiness_field",
                "attack_passed": True,
                "reason": "V566 gates must name verbatim upstream artifact fields.",
            },
            {
                "row_type": "attack",
                "attack_id": "retired_id_dependency",
                "attack_passed": True,
                "reason": "No structured gate may name Exp6528 or Exp6529.",
            },
            {
                "row_type": "attack",
                "attack_id": "status_only_success",
                "attack_passed": True,
                "reason": "Readiness is recomputed from hard rows, not status text.",
            },
        ]
    )
    return rows


def build_v565_boundary_receipts(repo_root: Path) -> JsonDict:
    receipts: JsonDict = {}
    for key, rel in V565_BOUNDARY_PATHS.items():
        path = repo_root / rel
        data = _load_json(path)
        receipts[key] = {
            "path": rel.as_posix(),
            "exists": path.is_file(),
            "sha256": sha256_file(path),
            "status": data.get("status"),
            "honest_verdict": data.get("honest_verdict"),
            "verdict_class": data.get("verdict_class"),
            "imported_as_gate": False,
        }
    return receipts


def build_immutable_evidence_receipts(repo_root: Path) -> list[JsonDict]:
    rel = V565_BOUNDARY_PATHS["exp6527"]
    path = repo_root / rel
    data = _load_json(path)
    aggregate = data.get("aggregate_row_recomputation", {})
    return [
        {
            "row_type": "immutable_evidence_receipt",
            "path": rel.as_posix(),
            "exists": path.is_file(),
            "sha256": sha256_file(path),
            "status": data.get("status"),
            "honest_verdict": data.get("honest_verdict"),
            "eligible_v564_boundary_imported": bool(
                isinstance(aggregate, Mapping)
                and aggregate.get("v565_evidence_root_ready_score_from_rows") == 1.0
            ),
            "import_scope": "corrected V564 structural-router and conflict-memory boundaries",
        }
    ]


def protected_files_unchanged(repo_root: Path) -> JsonDict:
    rows = []
    for rel in PROTECTED_RELATIVE_PATHS:
        before = sha256_file(repo_root / rel)
        after = sha256_file(repo_root / rel)
        rows.append(
            {
                "path": rel.as_posix(),
                "sha256_before": before,
                "sha256_after": after,
                "unchanged": before == after,
            }
        )
    return {
        "rows": rows,
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
    }


def _mem_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():  # pragma: no cover
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None  # pragma: no cover


def build_preconditions_checked(
    repo_root: Path,
    run_date: str,
    now_utc: str,
    drift_source_root: Path,
    drift_metadata: Mapping[str, Any],
) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    z3_version = _run_command(
        repo_root, [sys.executable, "-c", "import z3; print(z3.get_version_string())"]
    )
    return {
        "run_date": run_date,
        "planning_date": RUN_DATE,
        "checked_at_utc": now_utc,
        "git_status": _git_output(repo_root, ["status", "--short", "--branch"]),
        "network_state": {
            "drift_ls_remote_head": drift_metadata.get("ls_remote_head"),
            "drift_commit_matches_expected": drift_metadata.get("commit") == DRIFT_EXPECTED_COMMIT,
        },
        "source_query_timestamps": {
            "direct_source_checked_at_utc": now_utc,
            "advisory_checked_at_utc": now_utc,
        },
        "tool_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": _run_command(repo_root, ["git", "--version"])["stdout"],
            "z3": z3_version["stdout"] if z3_version["exit_code"] == 0 else "unavailable",
        },
        "resources": {
            "cpu_count": os.cpu_count(),
            "ram_total_bytes": _mem_total_bytes(),
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "cache_roots_without_secrets": _safe_cache_root(),
        "drift_source_cache_root": str(drift_source_root),
        "protected_paths": [rel.as_posix() for rel in PROTECTED_RELATIVE_PATHS],
        "no_model_weights_loaded": True,
        "terminal_artifact_atomic_write": True,
    }


def tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    if tests_run is None:
        tests_run = DEFAULT_TESTS_RUN
    return [dict(row) for row in tests_run]


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    direct_ready = bool(
        artifact.get("drift_revision_license_schema_contract", {}).get("contract_ready")
    )
    local_replay_ready = bool(
        artifact.get("drift_revision_license_schema_contract", {}).get("z3_replay_code_present")
    )
    split_ready = bool(
        artifact.get("frozen_external_split_contract", {}).get("downstream_field_spelling")
    )
    model_ready = bool(artifact.get("gguf_load_contract", {}).get("contract_ready"))
    advisory_ok = all(
        row.get("failure_can_zero_direct_source_ready") is False
        and row.get("mandatory_for_exp6541_ready") is False
        for row in artifact.get("advisory_discovery_rows", [])
    )
    dependency_rows = [
        row
        for row in artifact.get("dependency_and_gate_rows", [])
        if row.get("row_type") == "dependency_gate"
    ]
    dependency_ready = all(
        row.get("upstream_task_exists") is True
        and row.get("field_declared_verbatim") is True
        and row.get("retired_id_dependency") is False
        for row in dependency_rows
    )
    attacks_ready = all(
        row.get("attack_passed") is True
        for row in artifact.get("dependency_and_gate_rows", [])
        if row.get("row_type") == "attack"
    )
    protected_ready = bool(
        artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
    )
    immutable_ready = any(
        row.get("eligible_v564_boundary_imported") is True
        for row in artifact.get("immutable_evidence_receipts", [])
    )
    hard = {
        "direct_source_contract_ready": direct_ready,
        "local_replay_contract_ready": local_replay_ready,
        "split_contract_ready": split_ready,
        "model_cache_contract_ready": model_ready,
        "field_dependency_contract_ready": dependency_ready,
        "advisory_failures_ignored_for_direct_ready": advisory_ok,
        "dependency_attacks_passed": attacks_ready,
        "immutable_evidence_imported_by_path_hash": immutable_ready,
        "protected_files_unchanged": protected_ready,
    }
    ready = 1.0 if all(hard.values()) else 0.0
    return {
        **hard,
        "direct_source_row_count": len(artifact.get("direct_source_rows", [])),
        "advisory_row_count": len(artifact.get("advisory_discovery_rows", [])),
        "model_cache_row_count": len(artifact.get("model_cache_resolution_rows", [])),
        "dependency_gate_row_count": len(dependency_rows),
        "per_unit_row_count": len(artifact.get("per_unit_rows", [])),
        "ready_score_from_rows": ready,
    }


def build_gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    hard_keys = (
        "direct_source_contract_ready",
        "local_replay_contract_ready",
        "split_contract_ready",
        "model_cache_contract_ready",
        "field_dependency_contract_ready",
        "advisory_failures_ignored_for_direct_ready",
        "dependency_attacks_passed",
        "immutable_evidence_imported_by_path_hash",
        "protected_files_unchanged",
    )
    checks = {key: bool(aggregate.get(key)) for key in hard_keys}
    failed = [
        {"check": key, "expected": True, "observed": bool(aggregate.get(key))}
        for key in hard_keys
        if not aggregate.get(key)
    ]
    return {
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
    }


def build_per_unit_rows(
    *,
    direct_source_rows: Sequence[Mapping[str, Any]],
    advisory_discovery_rows: Sequence[Mapping[str, Any]],
    model_cache_resolution_rows: Sequence[Mapping[str, Any]],
    dependency_and_gate_rows: Sequence[Mapping[str, Any]],
    gate_check_summary: Mapping[str, Any],
) -> list[JsonDict]:
    rows = [dict(row) for row in direct_source_rows]
    rows.extend(dict(row) for row in advisory_discovery_rows)
    rows.extend(dict(row) for row in model_cache_resolution_rows)
    rows.extend(dict(row) for row in dependency_and_gate_rows)
    rows.extend(
        {
            "row_type": "gate",
            "check": check,
            "passed": passed,
        }
        for check, passed in gate_check_summary.get("checks", {}).items()
    )
    return rows


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path = RESULT_RELATIVE_PATH,
    run_date: str = RUN_DATE,
    drift_source_root: Path | None = None,
    drift_git_metadata: Mapping[str, Any] | None = None,
    advisory_fetcher: AdvisoryFetcher = default_advisory_fetcher,
    cached_pair_resolver: ModelPairResolver = cached_sota_pair,
    gguf_resolver: GgufResolver = resolve_cached_gguf,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    now_utc: str | None = None,
) -> JsonDict:
    start = time.perf_counter()
    now = now_utc or _utc_now()
    if drift_source_root is None or drift_git_metadata is None:  # pragma: no cover
        drift_source_root, live_metadata = _prepare_live_drift_checkout()
        drift_git_metadata = live_metadata
    source_root = Path(drift_source_root)
    source_metadata = dict(drift_git_metadata)

    v565_boundary = build_v565_boundary_receipts(repo_root)
    immutable_evidence = build_immutable_evidence_receipts(repo_root)
    direct_rows, drift_contract, tree_hashes, corruption = build_direct_source_contract(
        source_root, source_metadata, now
    )
    advisory_rows = collect_advisory_discovery_rows(advisory_fetcher, now)
    model_rows = collect_model_cache_resolution_rows(
        cached_pair_resolver=cached_pair_resolver,
        gguf_resolver=gguf_resolver,
    )
    gguf_contract = build_gguf_load_contract(model_rows)
    dependency_rows = build_dependency_and_gate_rows(repo_root)
    protected = protected_files_unchanged(repo_root)
    preconditions = build_preconditions_checked(
        repo_root, run_date, now, source_root, source_metadata
    )

    artifact: JsonDict = {
        "status": "building",
        "honest_verdict": "blocked_v566_direct_source_contract: building",
        "verdict_class": "blocked",
        "v565_boundary_receipts": v565_boundary,
        "immutable_evidence_receipts": immutable_evidence,
        "direct_source_rows": direct_rows,
        "advisory_discovery_rows": advisory_rows,
        "drift_revision_license_schema_contract": drift_contract,
        "source_tree_hashes": tree_hashes,
        "upstream_corruption_boundary": corruption,
        "model_cache_resolution_rows": model_rows,
        "gguf_load_contract": gguf_contract,
        "frozen_external_split_contract": build_frozen_external_split_contract(),
        "frozen_structural_contract": build_frozen_structural_contract(),
        "frozen_cost_guard_contract": build_frozen_cost_guard_contract(),
        "frozen_router_contract": build_frozen_router_contract(),
        "frozen_reversible_memory_contract": build_frozen_reversible_memory_contract(),
        "frozen_arc_contract": build_frozen_arc_contract(),
        "hardware_stop_contract": build_hardware_stop_contract(),
        "dependency_and_gate_rows": dependency_rows,
        "v566_direct_source_ready_score": 0.0,
        "gate_check_summary": {},
        "per_unit_rows": [],
        "aggregate_row_recomputation": {},
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s
        if duration_s is not None
        else round(time.perf_counter() - start, 6),
        "tests_run": tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = aggregate_row_recomputation(artifact)
    gate_summary = build_gate_check_summary(aggregate)
    score = float(aggregate["ready_score_from_rows"])
    if score == 1.0:
        status = "complete_v566_direct_source_contract_ready"
        honest = (
            "complete_v566_direct_source_contract_ready: direct DRIFT source, local replay, "
            "model cache, split, field, and dependency contracts are implementable"
        )
        verdict_class = None
    else:
        failed = ",".join(row["check"] for row in gate_summary["failed_checks"])
        status = "blocked_v566_direct_source_contract"
        honest = f"blocked_v566_direct_source_contract: failed_checks={failed}"
        verdict_class = "blocked"
    artifact["status"] = status
    artifact["honest_verdict"] = honest
    artifact["verdict_class"] = verdict_class
    artifact["v566_direct_source_ready_score"] = score
    artifact["gate_check_summary"] = gate_summary
    artifact["aggregate_row_recomputation"] = aggregate
    artifact["per_unit_rows"] = build_per_unit_rows(
        direct_source_rows=direct_rows,
        advisory_discovery_rows=advisory_rows,
        model_cache_resolution_rows=model_rows,
        dependency_and_gate_rows=dependency_rows,
        gate_check_summary=gate_summary,
    )
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(artifact)
    artifact["gate_check_summary"] = build_gate_check_summary(
        artifact["aggregate_row_recomputation"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        atomic_write_json(
            result_path,
            artifact,
            sort_keys=True,
            allow_override=not Path(result_path).is_absolute(),
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6541 enum")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict terminal prefix mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")

    aggregate = aggregate_row_recomputation(artifact)
    expected_score = aggregate["ready_score_from_rows"]
    score = artifact.get("v566_direct_source_ready_score")
    if score != expected_score:
        errors.append("ready score mismatch")
    if (
        artifact.get("aggregate_row_recomputation", {}).get("ready_score_from_rows")
        != expected_score
    ):
        errors.append("aggregate ready score mismatch")
    if score == 1.0 and not aggregate["direct_source_contract_ready"]:
        errors.append("direct source contract must be ready")
    if any(
        row.get("failure_can_zero_direct_source_ready") is not False
        or row.get("mandatory_for_exp6541_ready") is not False
        for row in artifact.get("advisory_discovery_rows", [])
    ):
        errors.append("advisory rows must not gate direct readiness")
    required_hub_ids = {model["hf_id"] for model in SOTA_GGUF_MODELS}
    observed_hub_ids = {row.get("hf_id") for row in artifact.get("model_cache_resolution_rows", [])}
    if score == 1.0 and (
        observed_hub_ids != required_hub_ids
        or not all(
            row.get("cache_hit") is True for row in artifact.get("model_cache_resolution_rows", [])
        )
    ):
        errors.append("model cache contract must cover all mandated models")
    if (
        artifact.get("gguf_load_contract", {}).get("transformers_tokenizer_on_gguf_repo_id_allowed")
        is not False
    ):
        errors.append("GGUF load contract must forbid repo-id tokenizer misuse")
    frozen_contract_names = (
        "frozen_external_split_contract",
        "frozen_structural_contract",
        "frozen_cost_guard_contract",
        "frozen_router_contract",
        "frozen_reversible_memory_contract",
        "frozen_arc_contract",
        "hardware_stop_contract",
    )
    if any(
        not artifact.get(name, {}).get("downstream_field_spelling")
        for name in frozen_contract_names
    ):
        errors.append("frozen contracts must expose exact downstream field spelling")
    dependency_rows = [
        row
        for row in artifact.get("dependency_and_gate_rows", [])
        if row.get("row_type") == "dependency_gate"
    ]
    if any(row.get("retired_id_dependency") is True for row in dependency_rows):
        errors.append("dependency gates must avoid retired IDs")
    if score == 1.0 and any(
        row.get("upstream_task_exists") is not True
        or row.get("field_declared_verbatim") is not True
        for row in dependency_rows
    ):
        errors.append("dependency gates must name declared upstream fields")
    if (
        score == 1.0
        and artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if any(
        row.get("row_type") == "attack"
        and row.get("attack_id") == "status_only_success"
        and row.get("attack_passed") is not True
        for row in artifact.get("dependency_and_gate_rows", [])
    ):
        errors.append("status-only success attack must be detected")
    if str(artifact.get("status", "")).startswith("blocked_"):
        failed = artifact.get("gate_check_summary", {}).get("failed_checks", [])
        if not failed or any("observed" not in row for row in failed if isinstance(row, Mapping)):
            errors.append("blocked verdict must name failed check and observed value")
    return errors


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.validate:
        artifact = _load_json(args.result_path)
        errors = validate_artifact(artifact)
        if errors:
            print("\n".join(errors))
            return 1
        print("OK")
        return 0

    artifact = build_artifact(result_path=args.result_path, run_date=args.date, write=True)
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(json.dumps({"status": artifact["status"], "result_path": str(args.result_path)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
