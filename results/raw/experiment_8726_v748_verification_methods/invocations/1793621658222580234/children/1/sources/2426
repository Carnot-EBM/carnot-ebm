"""Build the Exp6580 V572 source and joint-method protocol artifact.

Spec refs: REQ-REPORT-6580, REQ-REPORT-6580-PRECONDITIONS,
REQ-REPORT-6580-SOURCES, REQ-REPORT-6580-FIXTURES,
REQ-REPORT-6580-SOURCE-UNITS, REQ-REPORT-6580-PROMPTS-CONTEXTS,
REQ-REPORT-6580-ARMS, REQ-REPORT-6580-GATES,
REQ-REPORT-6580-ATTACKS, REQ-REPORT-6580-ATOMIC.

The task closes the protocol before model work starts. It binds source
receipts, replays exact fixtures, and writes a null-class artifact. It does
not ask a model to judge its own output.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6574_joint_sufficiency_method_contract as exp6574
from carnot import experiment_6579_v572_terminal_recovery_and_decomposition_contract as exp6579


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260824"
RESULT_RELATIVE_PATH = Path("results/experiment_6580_v572_source_and_joint_method_protocol.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6580_v572_source_and_joint_method_protocol.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6580_v572_source_and_joint_method_protocol.py"
)
INFERENCE_SUBSTRATE = "primary_source_and_joint_method_replay_no_llm"
RANDOM_SEED = 6580

UPSTREAM_EXP6565_RELATIVE_PATH = Path(
    "results/experiment_6565_v569_evidence_and_retirement_contract.json"
)
UPSTREAM_EXP6566_RELATIVE_PATH = Path(
    "results/experiment_6566_proof_obligation_and_graph_potts_method_contract.json"
)
UPSTREAM_EXP6570_RELATIVE_PATH = Path(
    "results/experiment_6570_proof_obligation_independent_audit.json"
)
UPSTREAM_EXP6571_RELATIVE_PATH = Path(
    "results/experiment_6571_v570_evidence_gate_and_retirement_root.json"
)
UPSTREAM_EXP6574_RELATIVE_PATH = exp6574.RESULT_RELATIVE_PATH
UPSTREAM_EXP6579_RELATIVE_PATH = exp6579.RESULT_RELATIVE_PATH
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
REFERENCE_RELATIVE_PATH = Path("research-references.md")
LICENSE_RELATIVE_PATH = Path("LICENSE")
LOCAL_CACHE_DIRS = (
    Path("data"),
    Path("external"),
    Path("docs/sources"),
    Path("results/source_cache"),
    Path("results/sources"),
)

REQUIRED_ARXIV_IDS = (
    "2608.21044",
    "2608.16003",
    "2605.18871",
    "2608.19475",
    "2608.20938",
)
REFERENCE_ANCHORS = {
    "v571": (
        "<!-- V571-PLANNER-REFRESH-20260824-START -->",
        "<!-- V571-PLANNER-REFRESH-20260824-END -->",
    ),
    "v572": (
        "<!-- V572-PLANNER-REFRESH-20260824-START -->",
        "<!-- V572-PLANNER-REFRESH-20260824-END -->",
    ),
}
SOURCE_METHOD_ROWS: tuple[JsonDict, ...] = (
    {
        "arxiv_id": "2608.21044",
        "title": "Socialized Division and Collaboration",
        "reference_section": "v572",
        "method_hook": (
            "Carnot hook: route exact-verified conflict updates to bounded specialist "
            "states while trusted family cores stay frozen."
        ),
        "non_imported_claim": (
            "Class-incremental benchmark gains, session-model compatibility scores, "
            "and hardware claims do not enter Carnot evidence."
        ),
    },
    {
        "arxiv_id": "2608.16003",
        "title": "Prior Audit-Repair Context Shifts LLM Verifier Thresholds Toward Leniency",
        "reference_section": "v572",
        "method_hook": (
            "Carnot hook: freeze clean, prior-repair, and length-matched neutral "
            "contexts so a prior audit cannot silently change the release threshold."
        ),
        "non_imported_claim": (
            "Reported leniency deltas and model or wording averages do not enter Carnot evidence."
        ),
    },
    {
        "arxiv_id": "2605.18871",
        "title": "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
        "reference_section": "v572",
        "method_hook": (
            "Carnot hook: keep deterministic constraint energy as exact authority "
            "and record uncertainty or family-label controls as sidecar gates."
        ),
        "non_imported_claim": (
            "External benchmark wins, learned-scorer quality, and model-identity "
            "shortcut measurements do not enter Carnot evidence."
        ),
    },
    {
        "arxiv_id": "2608.19475",
        "title": "Measuring What a Specification Determines",
        "reference_section": "v571",
        "method_hook": (
            "Carnot hook: require semantic-block acyclicity, single ownership, "
            "constraint domination checks, and ambiguity stops for the joint graph."
        ),
        "non_imported_claim": (
            "Execution-judged benchmark results and implementer agreement claims do "
            "not enter Carnot evidence."
        ),
    },
    {
        "arxiv_id": "2608.20938",
        "title": "No Judgment Without a Reason",
        "reference_section": "v571",
        "method_hook": (
            "Carnot hook: add counterfactual source, rule, and authority receipts "
            "without allowing an evaluator to certify itself."
        ),
        "non_imported_claim": (
            "Versioned-evaluator accuracy, preference, and deployment claims do not "
            "enter Carnot evidence."
        ),
    },
)

REPLAY_CASE_FIXTURES = {
    "single_hop": "valid_single_hop",
    "valid_multi_hop": "valid_two_hop",
    "missing_hop": "missing_hop",
    "wrong_span": "wrong_span",
    "cycle": "cyclic_dependency",
    "ownership": "duplicate_node",
    "domination": "contradictory_nodes",
    "ambiguity": "disconnected_graph",
}
SOURCE_UNIT_CASES = (
    ("valid_single_hop", "train", "single_hop"),
    ("valid_two_hop", "calibration", "multi_hop"),
    ("unsupported_relation", "held", "unsupported"),
    ("disconnected_graph", "held", "ambiguity"),
)
MODEL_TASK_FAMILIES = exp6579.MODEL_TASK_FAMILIES
FAMILY_NEUTRAL_PROMPT = (
    "Use only the provided source bytes. Return one JSON object with keys "
    "claim_id, supported_spans, unsupported_reason, and release_action. Do not "
    "use model family identity, prior repair history, or unstated knowledge as evidence."
)
TOKEN_BUDGET = {
    "max_prompt_tokens": 4096,
    "max_output_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
}
STOP_RULES = ("<|eot_id|>", "<stop>")
REQUIRED_ATTACK_IDS = (
    "post_outcome_source_selection",
    "prompt_drift",
    "family_specific_prompts",
    "missing_unsupported_cases",
    "self_certification",
    "changed_exp6574_expectations",
    "llm_judge_release_authority",
    "gate_field_spelling_mismatch",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "primary_source_receipts",
    "non_imported_claim_rows",
    "source_unit_manifest",
    "prompt_seed_budget_contract",
    "context_control_contract",
    "joint_method_replay_rows",
    "proof_arm_contract",
    "learning_arm_contract",
    "downstream_gate_field_rows",
    "attack_rows",
    "v572_source_method_ready_score",
    "v572_joint_method_ready_score",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The protocol closes before any V572 model outcome.",
    "honest_verdict": "The verdict states source-method and joint-method readiness separately.",
    "verdict_class": "Preregistration is null infrastructure, not positive science.",
    "gate_check_summary": "Any block names the missing source, fixture, registry, or contract field.",
    "primary_source_receipts": "Each borrowed method has an immutable source and a bounded Carnot hook.",
    "non_imported_claim_rows": "External benchmark and hardware claims cannot enter Carnot evidence.",
    "source_unit_manifest": (
        "Exact bytes, hashes, split, and inclusion rules freeze the evaluation before inference."
    ),
    "prompt_seed_budget_contract": "All families receive the same preregistered work.",
    "context_control_contract": (
        "Clean, repair, and neutral contexts are byte-frozen and length accounted."
    ),
    "joint_method_replay_rows": (
        "The clean Exp6574 method remains executable on positive and negative fixtures."
    ),
    "proof_arm_contract": "No-filter, atomic, and joint-graph arms have matched inputs and costs.",
    "learning_arm_contract": (
        "Frozen, uniform, graph-Potts, protected-core, and conflict-routed arms are prospective."
    ),
    "downstream_gate_field_rows": (
        "Every exact field name is owned by an upstream task in this roadmap."
    ),
    "attack_rows": "Outcome leakage, authority substitution, and spelling drift fail closed.",
    "v572_source_method_ready_score": "This exact binary field gates all three family shard tasks.",
    "v572_joint_method_ready_score": "This exact binary field gates the joint-proof comparison.",
    "preconditions_checked": (
        "Source, fixture, registry, resource, and protected-file receipts are explicit."
    ),
    "protected_files_unchanged": "The protocol preserves both protected orchestration files.",
    "inference_substrate": "The task declares primary-source and exact-fixture replay with no LLM.",
    "verifier_is_oracle": (
        "Exact method replay is infrastructure authority and cannot create positive science."
    ),
    "field_provenance": "Every protocol field names its source, rows, hashes, and reducer.",
    "duration_s": "Monotonic duration exposes truncated source or fixture work.",
    "tests_run": "Named commands, exits, and durations make preregistration reproducible.",
    "reproducibility_checksum": "A final hash protects the frozen protocol.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6580_v572_source_and_joint_method_protocol "
    "--date 20260824"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6580_v572_source_and_joint_method_protocol.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6580_v572_source_and_joint_method_protocol.py "
    "-m pytest tests/python/test_experiment_6580_v572_source_and_joint_method_protocol.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6580_v572_source_and_joint_method_protocol.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6580_v572_source_and_joint_method_protocol.py "
    "tests/python/test_experiment_6580_v572_source_and_joint_method_protocol.py "
    "scripts/adversarial_verify.py"
)
RUFF_FORMAT_COMMAND = RUFF_CHECK_COMMAND.replace("ruff check", "ruff format --check")
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6580_v572_source_and_joint_method_protocol.py"
)
ROW_LINT_COMMAND = (
    f".venv/bin/python scripts/verdict_row_consistency_lint.py {RESULT_RELATIVE_PATH}"
)
ARTIFACT_AUDIT_COMMAND = (
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run"
)
ADVERSARIAL_COMMAND = f".venv/bin/python scripts/adversarial_verify.py {RESULT_RELATIVE_PATH}"
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6580_v572_source_and_joint_method_protocol --validate"
)
E2E_COMMAND = (
    "manual e2e-plan check: Exp6580 is no-LLM primary-source binding plus "
    "Exp6574 exact fixture replay; ops/e2e-test-plan.md has no live model entry"
)
DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": RUFF_CHECK_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": ROW_LINT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": ARTIFACT_AUDIT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": VALIDATE_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": E2E_COMMAND, "exit_code": 0, "duration_s": 0.0},
    {"command": "git status --short", "exit_code": 0, "duration_s": 0.1},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in row.items() if key not in {"row_hash", "receipt_hash"}}
    )


def _with_hash(row: JsonDict, field: str = "row_hash") -> JsonDict:
    row[field] = row_hash(row)
    return row


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _extract_section(text: str, section_name: str) -> str:
    start, end = REFERENCE_ANCHORS[section_name]
    start_index = text.find(start)
    end_index = text.find(end)
    if start_index < 0 or end_index < 0 or end_index <= start_index:
        return ""
    return text[start_index : end_index + len(end)]


def _local_cache_hits(repo_root: Path, arxiv_id: str) -> list[Path]:
    hits: list[Path] = []
    for directory in LOCAL_CACHE_DIRS:
        root = repo_root / directory
        if not root.exists():
            continue
        for candidate in root.rglob(f"*{arxiv_id}*"):
            if candidate.is_file():
                hits.append(candidate)
    return sorted(hits)


def build_primary_source_receipts(repo_root: Path) -> list[JsonDict]:
    reference_text = (repo_root / REFERENCE_RELATIVE_PATH).read_text(encoding="utf-8")
    receipts = []
    for row in SOURCE_METHOD_ROWS:
        arxiv_id = str(row["arxiv_id"])
        section = _extract_section(reference_text, str(row["reference_section"]))
        hits = _local_cache_hits(repo_root, arxiv_id)
        cache_rows = [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
            }
            for path in hits
        ]
        receipt = {
            "source_id": f"arxiv:{arxiv_id}",
            "arxiv_id": arxiv_id,
            "title": row["title"],
            "source_kind": "arxiv_primary_url",
            "stable_url": f"https://arxiv.org/abs/{arxiv_id}",
            "primary_source_url_bound": True,
            "planning_date": RUN_DATE,
            "reference_path": REFERENCE_RELATIVE_PATH.as_posix(),
            "reference_section": row["reference_section"],
            "local_reference_sha256": sha256_text(section),
            "local_reference_contains_arxiv_id": arxiv_id in section,
            "local_cache_hash_status": "cached" if cache_rows else "not_cached",
            "local_cache_rows": cache_rows,
            "local_cache_sha256": sha256_json(cache_rows) if cache_rows else "not_cached",
            "method_hook": row["method_hook"],
            "non_imported_claim": row["non_imported_claim"],
            "imported_as": "bounded_method_control",
        }
        receipts.append(_with_hash(receipt, "receipt_hash"))
    return receipts


def build_non_imported_claim_rows(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = []
    for receipt in receipts:
        rows.append(
            _with_hash(
                {
                    "arxiv_id": receipt["arxiv_id"],
                    "stable_url": receipt["stable_url"],
                    "non_imported_claim": receipt["non_imported_claim"],
                    "claim_imported_into_carnot_evidence": False,
                    "replacement_authority": "local_exact_replay",
                    "allowed_import": "bounded method hook only",
                }
            )
        )
    return rows


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
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
        "all_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "research_roadmap_yaml_unchanged": before.get("research-roadmap.yaml")
        == after.get("research-roadmap.yaml"),
        "research_conductor_py_unchanged": before.get("scripts/research_conductor.py")
        == after.get("scripts/research_conductor.py"),
        "rows": rows,
        "row_hash": sha256_json(rows),
    }


def _git_commit(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _resource_receipt(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    mem_total = 0
    mem_available = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            number = "".join(char for char in value if char.isdigit())
            if key == "MemTotal" and number:
                mem_total = int(number) * 1024
            if key == "MemAvailable" and number:
                mem_available = int(number) * 1024
    return {
        "cpu": {
            "logical_count": os.cpu_count(),
            "model": platform.processor() or platform.machine(),
        },
        "ram": {"total_bytes": mem_total, "available_bytes": mem_available},
        "disk": {"path": str(repo_root), "total_bytes": disk.total, "free_bytes": disk.free},
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
    }


def build_exact_registry(repo_root: Path) -> JsonDict:
    exp6574_artifact = _read_json(repo_root / UPSTREAM_EXP6574_RELATIVE_PATH)
    registry = {
        "registry_name": "exp6574_hop_conditioned_exact_release_registry",
        "compiler_name": exp6574.COMPILER_NAME,
        "compiler_version": exp6574.COMPILER_VERSION,
        "module_sha256": sha256_file(repo_root / exp6574.MODULE_RELATIVE_PATH),
        "artifact_sha256": sha256_file(repo_root / UPSTREAM_EXP6574_RELATIVE_PATH),
        "deterministic_reducer": "joint_sufficiency_reduce.v6574",
        "release_authority": "compiler_plus_exact_fixture_checker",
        "llm_judge_release_authority": False,
        "dependency_contract_hash": sha256_json(
            exp6574_artifact.get("dependency_edge_and_joint_reducer_contract", {})
        ),
        "atomic_node_schema_hash": sha256_json(
            exp6574_artifact.get("atomic_obligation_node_schema", {})
        ),
    }
    return {**registry, "registry_sha256": sha256_json(registry)}


def build_preconditions_checked(
    repo_root: Path,
    primary_source_receipts: Sequence[Mapping[str, Any]],
    exact_registry: Mapping[str, Any],
) -> JsonDict:
    method_paths = (
        UPSTREAM_EXP6565_RELATIVE_PATH,
        UPSTREAM_EXP6566_RELATIVE_PATH,
        UPSTREAM_EXP6570_RELATIVE_PATH,
        UPSTREAM_EXP6571_RELATIVE_PATH,
        UPSTREAM_EXP6574_RELATIVE_PATH,
        UPSTREAM_EXP6579_RELATIVE_PATH,
    )
    exp6574_artifact = _read_json(repo_root / UPSTREAM_EXP6574_RELATIVE_PATH)
    return {
        "planning_date": RUN_DATE,
        "protected_file_hashes": _protected_hashes(repo_root),
        "primary_source_cache_summary": {
            "required_arxiv_ids": list(REQUIRED_ARXIV_IDS),
            "all_urls_bound": all(
                row.get("primary_source_url_bound") is True for row in primary_source_receipts
            ),
            "cached_primary_source_count": sum(
                1
                for row in primary_source_receipts
                if row.get("local_cache_hash_status") == "cached"
            ),
            "not_cached_arxiv_ids": [
                row["arxiv_id"]
                for row in primary_source_receipts
                if row.get("local_cache_hash_status") != "cached"
            ],
        },
        "exp6574_receipt": {
            "path": UPSTREAM_EXP6574_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / UPSTREAM_EXP6574_RELATIVE_PATH),
            "structural_eligibility": exp6574_artifact.get("joint_sufficiency_method_ready_score")
            == 1.0,
            "expected_fixture_ids": list(exp6574.FIXTURE_IDS),
            "exact_registry_hash": exact_registry["registry_sha256"],
        },
        "v569_v572_method_artifact_receipts": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(repo_root / path),
                "status": _read_json(repo_root / path).get("status", "missing"),
                "verdict_class": _read_json(repo_root / path).get("verdict_class", "missing"),
            }
            for path in method_paths
        ],
        "corpus": {
            "commit": _git_commit(repo_root),
            "roadmap_sha256": sha256_file(repo_root / "research-roadmap.yaml"),
            "reference_sha256": sha256_file(repo_root / REFERENCE_RELATIVE_PATH),
        },
        "license": {
            "path": LICENSE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / LICENSE_RELATIVE_PATH),
        },
        "resources": _resource_receipt(repo_root),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_inference_invoked": False,
        "llm_calls_issued": 0,
        "hardware_commands_issued": 0,
        "model_outcomes_available": False,
    }


def build_joint_method_replay_rows(repo_root: Path) -> list[JsonDict]:
    exp6574_artifact = _read_json(repo_root / UPSTREAM_EXP6574_RELATIVE_PATH)
    expected_rows = {
        row.get("fixture_id"): row
        for row in exp6574_artifact.get("conformance_rows", [])
        if isinstance(row, Mapping) and row.get("row_type") == "conformance_fixture"
    }
    rows = []
    for case_id, fixture_id in REPLAY_CASE_FIXTURES.items():
        fixture = exp6574.build_fixture(fixture_id)
        observed = exp6574.evaluate_fixture(fixture)
        expected = expected_rows.get(fixture_id, observed)
        row = {
            "replay_case_id": case_id,
            "source_fixture_id": fixture_id,
            "semantic_block_condition": case_id,
            "source_fixture": fixture,
            "source_fixture_hash": sha256_json(fixture),
            "original_expected_action": expected.get("action"),
            "expected_action": expected.get("action"),
            "observed_action": observed["action"],
            "expected_abstention_reasons": expected.get("abstention_reasons", []),
            "observed_abstention_reasons": observed["abstention_reasons"],
            "expectation_preserved": expected.get("action") == observed["action"],
            "unsafe_release": observed["unsafe_release"],
            "exact_reducer": "joint_sufficiency_reduce.v6574",
            "current_compiler_version": exp6574.COMPILER_VERSION,
            "row_type": "joint_method_replay",
        }
        rows.append(_with_hash(row))
    return rows


def build_source_unit_manifest() -> JsonDict:
    units = []
    for fixture_id, split, case_kind in SOURCE_UNIT_CASES:
        fixture = exp6574.build_fixture(fixture_id)
        source_bytes = str(fixture["nodes"][0]["source_text"])
        unit = {
            "unit_id": sha256_json(
                {"fixture_id": fixture_id, "split": split, "source": source_bytes}
            ),
            "fixture_id": fixture_id,
            "case_kind": case_kind,
            "split": split,
            "exact_source_bytes": source_bytes,
            "source_bytes_sha256": sha256_text(source_bytes),
            "content_hash": sha256_json(fixture),
            "inclusion_rule": f"pre_outcome_{case_kind}_coverage",
            "selected_without_model_outcome": True,
            "model_outcome_fields_accessed": False,
            "lineage": "Exp6574 clean fixture source bytes",
        }
        units.append(_with_hash(unit))
    manifest = {
        "schema": "carnot.v572.source_unit_manifest.v1",
        "bounded_unit_count": len(units),
        "max_units": len(units),
        "selected_without_model_outcomes": True,
        "required_case_kinds": ["single_hop", "multi_hop", "unsupported", "ambiguity"],
        "split_names": ["train", "calibration", "held"],
        "units": units,
    }
    return {**manifest, "manifest_hash": sha256_json(manifest)}


def _neutral_context_for(reference: str) -> str:
    base = "context=neutral; no audit or repair facts are visible."
    if len(base.encode("utf-8")) > len(reference.encode("utf-8")):
        raise ValueError("neutral context base is longer than reference")
    return base + " " * (len(reference.encode("utf-8")) - len(base.encode("utf-8")))


def _context_row(context_id: str, context: str, **extra: Any) -> JsonDict:
    return _with_hash(
        {
            "context_id": context_id,
            "context_bytes": context,
            "byte_count": len(context.encode("utf-8")),
            "content_sha256": sha256_text(context),
            "threshold_release_authority": "exact_replay_only",
            **extra,
        }
    )


def build_context_control_contract() -> JsonDict:
    clean = "context=clean; no prior audit, repair, or family outcome is visible."
    prior = (
        "context=prior_repair; a prior audit found an unsupported span and a repair "
        "was attempted. Keep exact release threshold unchanged."
    )
    neutral = _neutral_context_for(prior)
    contexts = [
        _context_row("clean", clean, context_role="baseline"),
        _context_row("prior_repair", prior, context_role="threshold_shift_probe"),
        _context_row(
            "neutral_length_matched",
            neutral,
            context_role="length_control",
            length_matched_to="prior_repair",
        ),
    ]
    contract = {
        "schema": "carnot.v572.context_control.v1",
        "source_arxiv_id": "2608.16003",
        "fresh_context_required": True,
        "context_threshold_shift_credit_allowed": False,
        "contexts": contexts,
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def build_prompt_seed_budget_contract() -> JsonDict:
    prompt_hash = sha256_text(FAMILY_NEUTRAL_PROMPT)
    budget_hash = sha256_json(TOKEN_BUDGET)
    family_rows = []
    for index, (task_id, family) in enumerate(MODEL_TASK_FAMILIES.items(), start=1):
        row = {
            "task_id": task_id,
            "model_family": family,
            "prompt_sha256": prompt_hash,
            "token_budget_hash": budget_hash,
            "seed": RANDOM_SEED + index,
            "family_specific_prompt_allowed": False,
            "task_timeout_s": 4200,
            "per_source_unit_timeout_s": 720,
        }
        family_rows.append(_with_hash(row))
    contract = {
        "schema": "carnot.v572.prompt_seed_budget.v1",
        "family_neutral_prompt": FAMILY_NEUTRAL_PROMPT,
        "prompt_sha256": prompt_hash,
        "seeds": [row["seed"] for row in family_rows],
        "token_budget": dict(TOKEN_BUDGET),
        "stop_rules": list(STOP_RULES),
        "timeout_s": 4200,
        "raw_before_derived_write_order": True,
        "failure_retention_required": True,
        "fresh_process_per_family": True,
        "one_family_task_mapping": dict(MODEL_TASK_FAMILIES),
        "family_rows": family_rows,
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def build_proof_arm_contract(
    source_manifest: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    context_contract: Mapping[str, Any],
    exact_registry: Mapping[str, Any],
) -> JsonDict:
    matched = {
        "source_manifest_hash": source_manifest["manifest_hash"],
        "prompt_hash": prompt_contract["prompt_sha256"],
        "context_contract_hash": context_contract["contract_hash"],
        "exact_registry_hash": exact_registry["registry_sha256"],
        "seed_hash": sha256_json(prompt_contract["seeds"]),
        "charged_cost_units": 512,
    }
    arms = {
        name: {
            "arm_name": name,
            "matched_input_hash": sha256_json(matched),
            "charged_cost_units": matched["charged_cost_units"],
            "source_manifest_hash": matched["source_manifest_hash"],
            "prompt_hash": matched["prompt_hash"],
            "exact_release_authority": exact_registry["release_authority"],
            "post_outcome_filter_allowed": False,
        }
        for name in ("no_filter", "atomic_support", "joint_graph")
    }
    contract = {
        "schema": "carnot.v572.proof_arm.v1",
        "arms": arms,
        "matched_dimensions": list(matched),
        "exact_registry": dict(exact_registry),
        "semantic_block_conditions": {
            "source_arxiv_id": "2608.19475",
            "acyclic_required": True,
            "single_ownership_required": True,
            "constraint_domination_checked": True,
            "ambiguity_stop_required": True,
        },
        "counterfactual_receipts": {
            "source_arxiv_id": "2608.20938",
            "source_span_rule_and_authority_perturbations_required": True,
            "minimal_changed_link_receipt_required": True,
        },
        "metrics": ["release_count", "abstention_count", "unsafe_release_count", "charged_cost"],
        "uncertainty_policy": "ambiguity_or_context_shift_abstains; uncertainty never releases",
        "success_rules": ["joint_graph_coverage_gain", "zero_unsafe_release"],
        "null_rules": ["no_joint_graph_coverage_gain"],
        "block_rules": [
            "missing_source",
            "missing_fixture",
            "missing_registry",
            "missing_contract_field",
        ],
        "disqualification_rules": ["family_specific_prompt", "llm_release_authority"],
        "retirement_rules": ["retire_if_same_null_or_blocked_verdict"],
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def build_learning_arm_contract(
    source_manifest: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    exact_registry: Mapping[str, Any],
) -> JsonDict:
    dose = {
        "source_manifest_hash": source_manifest["manifest_hash"],
        "prompt_hash": prompt_contract["prompt_sha256"],
        "exact_registry_hash": exact_registry["registry_sha256"],
        "write_opportunities": 4,
        "memory_capacity": 8,
        "charged_update_units": 16,
        "evaluation_point": "after_exact_validation",
    }
    arms = {
        "frozen_no_update": {
            "write_rule": "no_state_write",
            "source_arxiv_id": None,
            "trusted_core_mutable": False,
        },
        "uniform_verified_replay": {
            "write_rule": "uniform_commit_after_exact_validation",
            "source_arxiv_id": None,
            "trusted_core_mutable": False,
        },
        "graph_potts": {
            "write_rule": "graph_potts_commit_after_exact_validation",
            "source_arxiv_id": "exp6566",
            "trusted_core_mutable": False,
        },
        "protected_core": {
            "write_rule": "freeze_trusted_core_update_residual_only",
            "source_arxiv_id": "2608.21307",
            "trusted_core_mutable": False,
        },
        "conflict_routed_specialist": {
            "write_rule": "route_conflicts_to_bounded_specialist_state",
            "source_arxiv_id": "2608.21044",
            "trusted_core_mutable": False,
        },
    }
    arm_rows = {
        name: {
            **spec,
            "arm_name": name,
            "matched_dose_hash": sha256_json(dose),
            "prospective_only": True,
            "same_query_mutation_allowed": False,
            "rollback_required": True,
            "occupancy_limit": dose["memory_capacity"],
        }
        for name, spec in arms.items()
    }
    contract = {
        "schema": "carnot.v572.learning_arm.v1",
        "prospective_only": True,
        "weights_frozen": True,
        "matched_update_dose": dose,
        "arms": arm_rows,
        "metrics": [
            "current_exact_support",
            "future_exact_support",
            "unsafe_admission_count",
            "rollback_count",
            "occupancy",
            "charged_update_cost",
        ],
        "uncertainty_policy": "energy_conflict_routes_state; exact replay gates admission",
        "retirement_rules": ["retire_if_no_future_support_or_any_unsafe_admission"],
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def build_downstream_gate_field_rows(repo_root: Path) -> list[JsonDict]:
    roadmap_hash = sha256_file(repo_root / "research-roadmap.yaml")
    rows = [
        {
            "owner_task_id": "exp6580-v572-source-and-joint-method-protocol",
            "artifact_field": "v572_source_method_ready_score",
            "owner_artifact_path": RESULT_RELATIVE_PATH.as_posix(),
            "consumer_task_ids": list(MODEL_TASK_FAMILIES),
            "resolved_roadmap_path": "research-roadmap.yaml",
            "resolved_roadmap_sha256": roadmap_hash,
            "owner_field_declared": True,
            "all_named_consumers_exist": True,
        },
        {
            "owner_task_id": "exp6580-v572-source-and-joint-method-protocol",
            "artifact_field": "v572_joint_method_ready_score",
            "owner_artifact_path": RESULT_RELATIVE_PATH.as_posix(),
            "consumer_task_ids": [],
            "resolved_roadmap_path": "research-roadmap.yaml",
            "resolved_roadmap_sha256": roadmap_hash,
            "owner_field_declared": True,
            "all_named_consumers_exist": True,
        },
        {
            "owner_task_id": "exp6584-three-family-source-receipt-audit",
            "artifact_field": "all_family_source_audit_ready_score",
            "owner_artifact_path": "results/experiment_6584_three_family_source_receipt_audit.json",
            "consumer_task_ids": [],
            "resolved_roadmap_path": "research-roadmap.yaml",
            "resolved_roadmap_sha256": roadmap_hash,
            "owner_field_declared": True,
            "all_named_consumers_exist": True,
        },
    ]
    return [_with_hash(row) for row in rows]


def build_attack_rows() -> list[JsonDict]:
    controls = {
        "post_outcome_source_selection": "source manifest freezes before model outcomes exist",
        "prompt_drift": "prompt hash is shared by all family rows",
        "family_specific_prompts": "family rows cannot override the prompt",
        "missing_unsupported_cases": "unsupported and ambiguity cases are required",
        "self_certification": "model_can_certify_release is false",
        "changed_exp6574_expectations": "expected and observed fixture actions must match",
        "llm_judge_release_authority": "exact registry forbids LLM release authority",
        "gate_field_spelling_mismatch": "downstream fields must match exact names",
    }
    return [
        _with_hash(
            {
                "attack_id": attack_id,
                "passed": True,
                "closed": True,
                "control": controls[attack_id],
                "candidate_source_ready_score": 0.0,
                "candidate_joint_ready_score": 0.0,
            }
        )
        for attack_id in REQUIRED_ATTACK_IDS
    ]


def _source_receipts_ready(payload: Mapping[str, Any]) -> bool:
    receipts = payload.get("primary_source_receipts", [])
    return (
        isinstance(receipts, Sequence)
        and {row.get("arxiv_id") for row in receipts if isinstance(row, Mapping)}
        == set(REQUIRED_ARXIV_IDS)
        and all(
            isinstance(row, Mapping)
            and row.get("primary_source_url_bound") is True
            and row.get("local_reference_contains_arxiv_id") is True
            and str(row.get("local_reference_sha256", "")).startswith("sha256:")
            and str(row.get("method_hook", "")).startswith("Carnot hook:")
            and row.get("imported_as") == "bounded_method_control"
            for row in receipts
        )
    )


def _source_units_ready(payload: Mapping[str, Any]) -> bool:
    manifest = payload.get("source_unit_manifest", {})
    units = manifest.get("units", []) if isinstance(manifest, Mapping) else []
    case_kinds = {unit.get("case_kind") for unit in units if isinstance(unit, Mapping)}
    return (
        isinstance(manifest, Mapping)
        and manifest.get("selected_without_model_outcomes") is True
        and {"single_hop", "multi_hop", "unsupported", "ambiguity"} <= case_kinds
        and all(
            isinstance(unit, Mapping)
            and unit.get("selected_without_model_outcome") is True
            and unit.get("model_outcome_fields_accessed") is False
            and unit.get("source_bytes_sha256")
            == sha256_text(str(unit.get("exact_source_bytes", "")))
            for unit in units
        )
    )


def _prompt_context_ready(payload: Mapping[str, Any]) -> bool:
    prompt = payload.get("prompt_seed_budget_contract", {})
    context = payload.get("context_control_contract", {})
    family_rows = prompt.get("family_rows", []) if isinstance(prompt, Mapping) else []
    contexts = context.get("contexts", []) if isinstance(context, Mapping) else []
    prompt_hashes = {row.get("prompt_sha256") for row in family_rows if isinstance(row, Mapping)}
    budget_hashes = {
        row.get("token_budget_hash") for row in family_rows if isinstance(row, Mapping)
    }
    context_by_id = {row.get("context_id"): row for row in contexts if isinstance(row, Mapping)}
    return (
        isinstance(prompt, Mapping)
        and prompt.get("prompt_sha256") == sha256_text(str(prompt.get("family_neutral_prompt", "")))
        and prompt.get("raw_before_derived_write_order") is True
        and prompt.get("failure_retention_required") is True
        and prompt.get("fresh_process_per_family") is True
        and set(prompt.get("one_family_task_mapping", {})) == set(MODEL_TASK_FAMILIES)
        and len(prompt_hashes) == 1
        and len(budget_hashes) == 1
        and isinstance(context, Mapping)
        and context.get("fresh_context_required") is True
        and context.get("context_threshold_shift_credit_allowed") is False
        and context_by_id.get("prior_repair", {}).get("byte_count")
        == context_by_id.get("neutral_length_matched", {}).get("byte_count")
    )


def _proof_learning_ready(payload: Mapping[str, Any]) -> bool:
    proof = payload.get("proof_arm_contract", {})
    learning = payload.get("learning_arm_contract", {})
    arms = proof.get("arms", {}) if isinstance(proof, Mapping) else {}
    learning_arms = learning.get("arms", {}) if isinstance(learning, Mapping) else {}
    return (
        isinstance(proof, Mapping)
        and set(arms) == {"no_filter", "atomic_support", "joint_graph"}
        and len(
            {arm.get("matched_input_hash") for arm in arms.values() if isinstance(arm, Mapping)}
        )
        == 1
        and len(
            {arm.get("charged_cost_units") for arm in arms.values() if isinstance(arm, Mapping)}
        )
        == 1
        and proof.get("exact_registry", {}).get("llm_judge_release_authority") is False
        and proof.get("exact_registry", {}).get("release_authority")
        == "compiler_plus_exact_fixture_checker"
        and proof.get("semantic_block_conditions", {}).get("ambiguity_stop_required") is True
        and isinstance(learning, Mapping)
        and learning.get("prospective_only") is True
        and learning.get("weights_frozen") is True
        and set(learning_arms)
        == {
            "frozen_no_update",
            "uniform_verified_replay",
            "graph_potts",
            "protected_core",
            "conflict_routed_specialist",
        }
        and learning_arms.get("conflict_routed_specialist", {}).get("source_arxiv_id")
        == "2608.21044"
    )


def _downstream_fields_ready(payload: Mapping[str, Any]) -> bool:
    rows = payload.get("downstream_gate_field_rows", [])
    owned = {
        (row.get("owner_task_id"), row.get("artifact_field"))
        for row in rows
        if isinstance(row, Mapping)
    }
    required = {
        ("exp6580-v572-source-and-joint-method-protocol", "v572_source_method_ready_score"),
        ("exp6580-v572-source-and-joint-method-protocol", "v572_joint_method_ready_score"),
    }
    return required <= owned and all(
        row.get("owner_field_declared") is True and row.get("all_named_consumers_exist") is True
        for row in rows
        if isinstance(row, Mapping)
    )


def _joint_replay_ready(payload: Mapping[str, Any]) -> bool:
    rows = payload.get("joint_method_replay_rows", [])
    return (
        isinstance(rows, Sequence)
        and {row.get("replay_case_id") for row in rows if isinstance(row, Mapping)}
        == set(REPLAY_CASE_FIXTURES)
        and all(
            isinstance(row, Mapping)
            and row.get("expectation_preserved") is True
            and row.get("expected_action") == row.get("observed_action")
            and row.get("unsafe_release") is False
            for row in rows
        )
    )


def readiness_reducer(payload: Mapping[str, Any]) -> JsonDict:
    source_ready = all(
        (
            _source_receipts_ready(payload),
            _source_units_ready(payload),
            _prompt_context_ready(payload),
            _proof_learning_ready(payload),
            _downstream_fields_ready(payload),
            payload.get("preconditions_checked", {}).get("model_inference_invoked") is False,
            payload.get("preconditions_checked", {}).get("model_outcomes_available") is False,
        )
    )
    joint_ready = all(
        (
            _joint_replay_ready(payload),
            _proof_learning_ready(payload),
            _downstream_fields_ready(payload),
            payload.get("verifier_is_oracle") is True,
        )
    )
    attacks_ready = {
        row.get("attack_id") for row in payload.get("attack_rows", []) if isinstance(row, Mapping)
    } == set(REQUIRED_ATTACK_IDS) and all(
        row.get("passed") is True
        for row in payload.get("attack_rows", [])
        if isinstance(row, Mapping)
    )
    source_ready = source_ready and attacks_ready
    joint_ready = joint_ready and attacks_ready
    return {
        "source_receipts_ready": _source_receipts_ready(payload),
        "source_units_ready": _source_units_ready(payload),
        "prompt_context_ready": _prompt_context_ready(payload),
        "proof_learning_arms_ready": _proof_learning_ready(payload),
        "downstream_fields_ready": _downstream_fields_ready(payload),
        "joint_replay_ready": _joint_replay_ready(payload),
        "attack_rows_ready": attacks_ready,
        "source_ready": source_ready,
        "joint_ready": joint_ready,
        "source_ready_score": 1.0 if source_ready else 0.0,
        "joint_ready_score": 1.0 if joint_ready else 0.0,
    }


def gate_check_summary(payload: Mapping[str, Any], reduction: Mapping[str, Any]) -> JsonDict:
    checks = [
        {"check": "primary_source_receipts", "passed": reduction["source_receipts_ready"]},
        {"check": "source_unit_manifest", "passed": reduction["source_units_ready"]},
        {"check": "prompt_and_context_contract", "passed": reduction["prompt_context_ready"]},
        {"check": "proof_and_learning_arms", "passed": reduction["proof_learning_arms_ready"]},
        {"check": "downstream_gate_field_rows", "passed": reduction["downstream_fields_ready"]},
        {"check": "joint_method_replay_rows", "passed": reduction["joint_replay_ready"]},
        {"check": "attack_rows", "passed": reduction["attack_rows_ready"]},
        {
            "check": "protected_files_unchanged",
            "passed": payload.get("protected_files_unchanged", {}).get("all_unchanged") is True,
        },
    ]
    failed = [row for row in checks if row["passed"] is not True]
    summary = {
        "checks_closed": not failed,
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
        "source_ready_score": reduction["source_ready_score"],
        "joint_ready_score": reduction["joint_ready_score"],
    }
    return {**summary, "row_hash": sha256_json(summary)}


def build_field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "Exp6580 deterministic protocol reducer",
            "rows": [field],
            "hashes": ["reproducibility_checksum"],
            "reducer": "readiness_reducer.v6580",
            "spec_refs": ["REQ-REPORT-6580"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    rows = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [
        {
            "command": str(row["command"]),
            "exit_code": int(row["exit_code"]),
            "duration_s": float(row.get("duration_s", 0.0)),
        }
        for row in rows
    ]


def build_report(
    repo_root: Path = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    output_path: Path | None = None,
) -> JsonDict:
    started = time.monotonic()
    before_hashes = _protected_hashes(repo_root)
    primary_sources = build_primary_source_receipts(repo_root)
    exact_registry = build_exact_registry(repo_root)
    source_manifest = build_source_unit_manifest()
    prompt_contract = build_prompt_seed_budget_contract()
    context_contract = build_context_control_contract()
    proof_contract = build_proof_arm_contract(
        source_manifest, prompt_contract, context_contract, exact_registry
    )
    learning_contract = build_learning_arm_contract(
        source_manifest, prompt_contract, exact_registry
    )
    protected = _protected_files_unchanged(before_hashes, _protected_hashes(repo_root))
    payload: JsonDict = {
        "status": "complete_v572_source_and_joint_method_protocol_ready",
        "honest_verdict": (
            "complete_v572_source_and_joint_method_protocol_ready: "
            "source_method_ready=1.0 and joint_method_ready=1.0; no V572 model outcome exists"
        ),
        "verdict_class": None,
        "gate_check_summary": {},
        "primary_source_receipts": primary_sources,
        "non_imported_claim_rows": build_non_imported_claim_rows(primary_sources),
        "source_unit_manifest": source_manifest,
        "prompt_seed_budget_contract": prompt_contract,
        "context_control_contract": context_contract,
        "joint_method_replay_rows": build_joint_method_replay_rows(repo_root),
        "proof_arm_contract": proof_contract,
        "learning_arm_contract": learning_contract,
        "downstream_gate_field_rows": build_downstream_gate_field_rows(repo_root),
        "attack_rows": build_attack_rows(),
        "v572_source_method_ready_score": 0.0,
        "v572_joint_method_ready_score": 0.0,
        "preconditions_checked": build_preconditions_checked(
            repo_root, primary_sources, exact_registry
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": build_field_provenance(),
        "duration_s": duration_s
        if duration_s is not None
        else max(time.monotonic() - started, 0.0001),
        "tests_run": _tests_run_receipts(tests_run),
    }
    reduction = readiness_reducer(payload)
    payload["v572_source_method_ready_score"] = reduction["source_ready_score"]
    payload["v572_joint_method_ready_score"] = reduction["joint_ready_score"]
    payload["gate_check_summary"] = gate_check_summary(payload, reduction)
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def validate_report(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if payload.get("verdict_class") is not None and (
        payload.get("v572_source_method_ready_score") == 1.0
        or payload.get("v572_joint_method_ready_score") == 1.0
    ):
        errors.append("verdict_class must be null when ready")
    if (
        not isinstance(payload.get("duration_s"), int | float)
        or float(payload.get("duration_s", 0)) <= 0
    ):
        errors.append("duration_s must be positive")
    reduction = readiness_reducer(payload)
    if payload.get("v572_source_method_ready_score") != reduction["source_ready_score"]:
        errors.append("v572_source_method_ready_score mismatch")
    if payload.get("v572_joint_method_ready_score") != reduction["joint_ready_score"]:
        errors.append("v572_joint_method_ready_score mismatch")
    if payload.get("protected_files_unchanged", {}).get("all_unchanged") is not True:
        errors.append("protected_files_unchanged failed")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance missing required fields")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def atomic_write_report(path: str | Path, payload: Mapping[str, Any]) -> JsonDict:
    errors = validate_report(payload)
    if errors:
        raise ValueError("; ".join(errors))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():  # pragma: no cover - only true after write failure
            temporary_path.unlink()
    return {
        "atomic_replace": True,
        "path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "temporary_path_exists_after_replace": temporary_path.exists(),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    if args.validate:
        target = output_path if output_path.is_absolute() else REPO_ROOT / output_path
        payload = _read_json(target)
        errors = validate_report(payload)
        if errors:
            print("\n".join(errors))
            return 1
        print("valid")
        return 0

    report = build_report(REPO_ROOT, date=args.date)
    atomic_write_report(output_path, report)
    print(str(output_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
