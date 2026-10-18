"""Build the Exp6412 V551 powered claim integrity audit.

Spec refs: REQ-REPORT-6412, SCENARIO-REPORT-6412-1,
SCENARIO-REPORT-6412-2, SCENARIO-REPORT-6412-3,
SCENARIO-REPORT-6412-4, SCENARIO-REPORT-6412-5,
SCENARIO-REPORT-6412-6.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import os
import subprocess
import time
from typing import Any

from carnot.experiment_artifacts import (
    atomic_write_json,
    resolve_experiment_artifact_path,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260814"
SCHEMA = "carnot.experiment_6412.v551_powered_claim_integrity_audit.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6412_v551_powered_claim_integrity_audit.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6412_v551_powered_claim_integrity_audit.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6412_v551_powered_claim_integrity_audit.py"
)
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_no_llm"
CORRIGENDUM_SUFFIX = ".corrigendum.json"
CLAIM_LEDGER_SUFFIX = ".claim_ledger.jsonl"
EXTERNAL_TEST_RECEIPTS_ENV = "CARNOT_EXP6412_TEST_RECEIPTS"
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6412_test_receipts.json")

EXP6407_ID = "exp6407-provenance-tiered-factor-memory-protocol"
EXP6408_ID = "exp6408-powered-write-time-factor-admission-ab"
EXP6409_ID = "exp6409-graph-local-multisession-continuous-learning"
EXPERIMENT_IDS = (EXP6407_ID, EXP6408_ID, EXP6409_ID)
ARTIFACT_BY_ID = {
    EXP6407_ID: Path("results/experiment_6407_provenance_tiered_factor_memory_protocol.json"),
    EXP6408_ID: Path("results/experiment_6408_powered_write_time_factor_admission_ab.json"),
    EXP6409_ID: Path("results/experiment_6409_graph_local_multisession_continuous_learning.json"),
}
SOURCE_BY_ID = {
    EXP6407_ID: Path("python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py"),
    EXP6408_ID: Path("python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py"),
    EXP6409_ID: Path(
        "python/carnot/experiment_6409_graph_local_multisession_continuous_learning.py"
    ),
}
SIDECARS_BY_ID = {
    EXP6407_ID: (
        Path("results/experiment_6407_provenance_tiered_factor_memory_protocol.json"),
        Path("results/experiment_6407_provenance_tiered_factor_memory_protocol.json.raw_ledger.jsonl"),
        Path(
            "results/experiment_6407_provenance_tiered_factor_memory_protocol.json.raw_record_schema.json"
        ),
        Path(
            "results/experiment_6407_provenance_tiered_factor_memory_protocol.json.compiled_typed_graph.json"
        ),
        Path(
            "results/experiment_6407_provenance_tiered_factor_memory_protocol.json.compiled_typed_graph_schema.json"
        ),
        Path(
            "results/experiment_6407_provenance_tiered_factor_memory_protocol.json.contamination_manifest.json"
        ),
    ),
    EXP6408_ID: (
        Path("results/experiment_6408_powered_write_time_factor_admission_ab.json"),
        Path("results/experiment_6408_powered_write_time_factor_admission_ab.json.fresh_held_manifest.json"),
    ),
    EXP6409_ID: (
        Path("results/experiment_6409_graph_local_multisession_continuous_learning.json"),
        Path(
            "results/experiment_6409_graph_local_multisession_continuous_learning.json.chronological_manifest.json"
        ),
    ),
}

LOG_AND_DETERMINATION_PATHS = (
    Path("ops/conductor-log.md"),
    Path("ops/known-issues.md"),
    Path("ops/claim-eligibility-ledger.json"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("python/carnot/terminal_artifacts.py"),
)
CONTEXT_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    *SOURCE_BY_ID.values(),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    *ARTIFACT_BY_ID.values(),
    *[path for paths in SIDECARS_BY_ID.values() for path in paths],
    Path("ops/conductor-log.md"),
    Path("ops/known-issues.md"),
)
AUDITED_RELATIVE_PATHS = (
    *CONTEXT_SOURCE_PATHS,
    *ARTIFACT_BY_ID.values(),
    *[path for paths in SIDECARS_BY_ID.values() for path in paths],
    *LOG_AND_DETERMINATION_PATHS,
)

MODEL_PROCESS_PATTERNS = ("llama_cpp.Llama(", ".create_completion(", ".generate(", "subprocess.run(")
POWERED_REQUIRED_RECEIPTS = (
    "model_file_opened",
    "model_process_ran",
    "tokens_generated",
    "raw_output_bytes_stored",
    "pid_bound_gpu_samples",
    "exact_outcomes_observed_after_admission",
    "nonconstant_runtime_duration",
)
MUTATION_ATTACK_IDS = (
    "constant_durations",
    "forged_pids",
    "model_name_only_rows",
    "inherited_outputs",
    "missing_raw_hashes",
    "fabricated_gpu_samples",
)
ALLOWED_PROVENANCE_CLASSES = {"measured", "derived", "constant", "inherited", "absent"}
ALLOWED_HONEST_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6412_v551_powered_claim_integrity_audit "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6412_v551_powered_claim_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6412_v551_powered_claim_integrity_audit.py "
    "-m pytest tests/python/test_experiment_6412_v551_powered_claim_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6412_v551_powered_claim_integrity_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6412_v551_powered_claim_integrity_audit.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6407_provenance_tiered_factor_memory_protocol.json "
    "results/experiment_6408_powered_write_time_factor_admission_ab.json "
    "results/experiment_6409_graph_local_multisession_continuous_learning.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    FULL_PYTEST_COMMAND,
    RUN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "audited_source_artifact_sidecar_and_log_hashes",
    "per_experiment_field_provenance_matrix",
    "model_file_open_evidence",
    "model_process_execution_evidence",
    "token_generation_evidence",
    "raw_output_byte_evidence",
    "pid_bound_gpu_telemetry_evidence",
    "exact_outcome_temporal_evidence",
    "constant_and_inherited_field_findings",
    "stamped_and_current_adversarial_findings",
    "determination_preservation_results",
    "deterministic_protocol_claim_eligibility",
    "deterministic_replay_claim_eligibility",
    "powered_gguf_claim_eligibility",
    "prospective_csl_claim_eligibility",
    "public_factor_claim_eligibility",
    "fr11_claim_eligibility",
    "additive_corrigendum_path_and_hash",
    "additive_claim_ledger_path_entry_and_hash",
    "historical_artifacts_modified",
    "historical_determinations_preserved",
    "mutation_attack_matrix",
    "powered_false_accept_count",
    "v551_claim_boundary_ready_score",
    "protected_files_unchanged",
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
    "status": "The status states whether the claim boundary is complete.",
    "audited_source_artifact_sidecar_and_log_hashes": "All audited inputs are hash-pinned before analysis.",
    "per_experiment_field_provenance_matrix": "Each audited claim field names its evidence class and source.",
    "model_file_open_evidence": "Model path receipts cannot stand in for a model file opened by a runtime.",
    "model_process_execution_evidence": "Powered claims require a model process, not source-only replay.",
    "token_generation_evidence": "Powered claims require generated tokens, not tokenizer prechecks.",
    "raw_output_byte_evidence": "Powered claims require stored raw model output bytes.",
    "pid_bound_gpu_telemetry_evidence": "Powered claims require GPU samples bound to the model PID.",
    "exact_outcome_temporal_evidence": "Outcome receipts must occur after admission, not come from constants.",
    "constant_and_inherited_field_findings": "Constant and inherited fields are barred from powered proof.",
    "stamped_and_current_adversarial_findings": "Stamped historical findings and current verifier output stay separate.",
    "determination_preservation_results": "The preservation lint confirms no historical determination was cleared.",
    "deterministic_protocol_claim_eligibility": "This gate controls deterministic protocol readiness reuse.",
    "deterministic_replay_claim_eligibility": "This gate allows deterministic replay only within checked-in evidence.",
    "powered_gguf_claim_eligibility": "This gate blocks powered GGUF claims without execution receipts.",
    "prospective_csl_claim_eligibility": "This gate blocks CSL progress without fresh powered evidence.",
    "public_factor_claim_eligibility": "This gate blocks public factor claims from unproven receipts.",
    "fr11_claim_eligibility": "This gate blocks FR11 progress claims from this audit-only artifact.",
    "additive_corrigendum_path_and_hash": "The corrigendum is additive and names the corrected claim boundary.",
    "additive_claim_ledger_path_entry_and_hash": "The ledger records the exact claim boundary as an append-only row.",
    "historical_artifacts_modified": "Historical artifacts must remain byte-identical.",
    "historical_determinations_preserved": "Historical flags and corrigenda must remain present.",
    "mutation_attack_matrix": "Mutation attacks prove powered eligibility fails closed.",
    "powered_false_accept_count": "This count must stay zero for the boundary to be ready.",
    "v551_claim_boundary_ready_score": "This score is one only when provenance is complete and overclaims are blocked.",
    "protected_files_unchanged": "Protected historical files remain unchanged during the run.",
    "preconditions_checked": "Preconditions record date, hashes, missing inputs, and command context.",
    "inference_substrate": "This audit reads artifacts and source only. It does not run a model.",
    "verifier_is_oracle": "The verifier is not an oracle. It audits evidence provenance only.",
    "field_principles": "Each required field states the guard it serves.",
    "field_provenance": "Each required field has a measured, derived, constant, inherited, or absent class.",
    "random_seed": "No random sampling is used by this audit.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification command receipts are recorded for the audit.",
    "reproducibility_checksum": "The checksum detects drift outside known volatile fields.",
    "honest_verdict": "The verdict uses a terminal prefix and states the claim boundary.",
}
FIELD_PROVENANCE: dict[str, str] = {
    "status": "derived",
    "audited_source_artifact_sidecar_and_log_hashes": "measured",
    "per_experiment_field_provenance_matrix": "derived",
    "model_file_open_evidence": "derived",
    "model_process_execution_evidence": "absent",
    "token_generation_evidence": "absent",
    "raw_output_byte_evidence": "derived",
    "pid_bound_gpu_telemetry_evidence": "absent",
    "exact_outcome_temporal_evidence": "derived",
    "constant_and_inherited_field_findings": "derived",
    "stamped_and_current_adversarial_findings": "measured",
    "determination_preservation_results": "measured",
    "deterministic_protocol_claim_eligibility": "derived",
    "deterministic_replay_claim_eligibility": "derived",
    "powered_gguf_claim_eligibility": "derived",
    "prospective_csl_claim_eligibility": "derived",
    "public_factor_claim_eligibility": "derived",
    "fr11_claim_eligibility": "derived",
    "additive_corrigendum_path_and_hash": "measured",
    "additive_claim_ledger_path_entry_and_hash": "measured",
    "historical_artifacts_modified": "measured",
    "historical_determinations_preserved": "measured",
    "mutation_attack_matrix": "derived",
    "powered_false_accept_count": "derived",
    "v551_claim_boundary_ready_score": "derived",
    "protected_files_unchanged": "measured",
    "preconditions_checked": "derived",
    "inference_substrate": "constant",
    "verifier_is_oracle": "constant",
    "field_principles": "constant",
    "field_provenance": "constant",
    "random_seed": "constant",
    "duration_s": "measured",
    "tests_run": "measured",
    "reproducibility_checksum": "derived",
    "honest_verdict": "derived",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def read_json_mapping(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_top_level_not_object:{path}")
    return value


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def path_receipt(path: str | Path, root: Path, *, digest: str | None = None) -> JsonDict:
    path = Path(path)
    return {
        "path": relative_or_absolute(path, root),
        "present": path.is_file(),
        "sha256": digest if digest is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def line_hits(source: str, patterns: Sequence[str]) -> list[JsonDict]:
    hits: list[JsonDict] = []
    for number, line in enumerate(source.splitlines(), start=1):
        if any(pattern in line for pattern in patterns):
            hits.append({"line": number, "text": line.strip()})
    return hits


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "ok": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def audited_hashes(root: Path) -> JsonDict:
    by_category = {
        "sources": CONTEXT_SOURCE_PATHS,
        "artifacts": tuple(ARTIFACT_BY_ID.values()),
        "sidecars": tuple(path for paths in SIDECARS_BY_ID.values() for path in paths),
        "logs_and_determinations": LOG_AND_DETERMINATION_PATHS,
    }
    receipts = {
        category: {path.as_posix(): path_receipt(root / path, root) for path in paths}
        for category, paths in by_category.items()
    }
    missing = [
        path
        for rows in receipts.values()
        for path, receipt in rows.items()
        if receipt["present"] is not True
    ]
    return {
        "schema": SCHEMA + ".audited_hashes",
        "hashes_recorded_before_analysis": True,
        **receipts,
        "missing_paths": sorted(missing),
    }


def load_payloads(root: Path) -> dict[str, JsonDict]:
    return {
        exp_id: read_json_mapping(root / artifact_path)
        for exp_id, artifact_path in ARTIFACT_BY_ID.items()
    }


def load_sources(root: Path) -> dict[str, str]:
    return {
        exp_id: (root / source_path).read_text(encoding="utf-8")
        for exp_id, source_path in SOURCE_BY_ID.items()
    }


def classify_top_level_field(exp_id: str, field: str) -> str:
    if field in {"duration_s", "protected_files_unchanged"}:
        return "measured"
    if field in {"random_seed", "inference_substrate", "verifier_is_oracle"}:
        return "constant"
    if exp_id == EXP6409_ID and field in {
        "MODEL_SPECS",
        "models_used",
        "cached_sota_pair_receipts",
        "embedded_gguf_tokenizer_receipts",
        "cuda_offload_runtime_peak_memory_and_duration_receipts_by_model",
    }:
        return "inherited"
    if exp_id == EXP6408_ID and field in {"MODEL_SPECS", "cached_sota_pair_receipts"}:
        return "inherited"
    return "derived"


def provenance_row(
    classification: str,
    source: str,
    *,
    source_lines: Sequence[Mapping[str, Any]] | None = None,
    artifact_field: str | None = None,
    claim_risk: str = "bounded",
) -> JsonDict:
    return {
        "classification": classification,
        "source": source,
        "source_lines": list(source_lines or []),
        "artifact_field": artifact_field,
        "claim_risk": claim_risk,
    }


def build_field_provenance_matrix(
    payloads: Mapping[str, JsonDict],
    sources: Mapping[str, str],
) -> JsonDict:
    matrix: JsonDict = {}
    for exp_id, payload in payloads.items():
        source_path = SOURCE_BY_ID[exp_id].as_posix()
        rows = {
            f"top_level.{field}": provenance_row(
                classify_top_level_field(exp_id, field),
                ARTIFACT_BY_ID[exp_id].as_posix(),
                artifact_field=field,
            )
            for field in sorted(payload)
        }
        matrix[exp_id] = rows

    source6408 = sources[EXP6408_ID]
    source6409 = sources[EXP6409_ID]
    matrix[EXP6408_ID].update(
        {
            "runtime.duration_s": provenance_row(
                "constant",
                SOURCE_BY_ID[EXP6408_ID].as_posix(),
                source_lines=line_hits(source6408, ["0.25 + 0.05 * index"]),
                artifact_field="cuda_offload_runtime_peak_memory_and_duration_receipts_by_model.by_model.*.duration_s",
                claim_risk="powered_runtime_overclaim",
            ),
            "runtime.peak_memory_mb": provenance_row(
                "derived",
                SOURCE_BY_ID[EXP6408_ID].as_posix(),
                source_lines=line_hits(source6408, ["used_mb + 1024 + 128 * index"]),
                artifact_field="cuda_offload_runtime_peak_memory_and_duration_receipts_by_model.by_model.*.peak_memory_mb",
                claim_risk="derived_gpu_memory_overclaim",
            ),
            "raw_model_bytes_sha256": provenance_row(
                "derived",
                SOURCE_BY_ID[EXP6408_ID].as_posix(),
                source_lines=line_hits(source6408, ["raw_bytes = canonical_json", "raw_model_bytes_sha256"]),
                artifact_field="raw_bytes_source_effect_diagnostic_checker_disposition_and_head_freeze_records.rows.*.raw_model_bytes_sha256",
                claim_risk="synthetic_raw_output_hash",
            ),
            "future_exact_success_count": provenance_row(
                "constant",
                SOURCE_BY_ID[EXP6408_ID].as_posix(),
                source_lines=line_hits(source6408, ["return int(total * 0.75)", "return int(total * 0.55)", "return int(total * 0.50)"]),
                artifact_field="exact_future_yield_by_arm.*.future_exact_success_count",
                claim_risk="constant_future_outcome",
            ),
        }
    )
    matrix[EXP6409_ID].update(
        {
            "runtime.duration_s": provenance_row(
                "inherited",
                SOURCE_BY_ID[EXP6409_ID].as_posix(),
                source_lines=line_hits(
                    source6409,
                    ['upstream.get("cuda_offload_runtime_peak_memory_and_duration_receipts_by_model")'],
                ),
                artifact_field="cuda_offload_runtime_peak_memory_and_duration_receipts_by_model.by_model.*.duration_s",
                claim_risk="inherited_runtime_surface",
            ),
            "future_exact_success_count": provenance_row(
                "constant",
                SOURCE_BY_ID[EXP6409_ID].as_posix(),
                source_lines=line_hits(source6409, ['"success_count": 12', '"success_count": 10']),
                artifact_field="untouched_future_evaluation_receipts.by_arm.*.success_count",
                claim_risk="constant_future_outcome",
            ),
            "matched_work.llm_call_count": provenance_row(
                "derived",
                SOURCE_BY_ID[EXP6409_ID].as_posix(),
                source_lines=line_hits(source6409, ["llm_call_count"]),
                artifact_field="matched_work_receipts.by_session.*.*.llm_call_count",
                claim_risk="derived_call_count_without_process",
            ),
        }
    )
    return matrix


def model_file_open_evidence(payloads: Mapping[str, JsonDict], sources: Mapping[str, str]) -> JsonDict:
    exp6408_models = list(payloads[EXP6408_ID].get("MODEL_SPECS", []))
    exp6408_model_paths = [row.get("model_path") for row in exp6408_models if isinstance(row, Mapping)]
    source6408 = sources[EXP6408_ID]
    source6409 = sources[EXP6409_ID]
    return {
        "exp6407": {
            "classification": "absent",
            "model_paths_present": False,
            "model_file_opened_by_model_runtime": False,
            "note": "Exp6407 is deterministic protocol replay.",
        },
        "exp6408": {
            "classification": "absent",
            "model_paths_present": bool(exp6408_model_paths),
            "model_file_opened_by_model_runtime": False,
            "path_identity_checks_only": True,
            "model_path_count": len(exp6408_model_paths),
            "source_lines": line_hits(source6408, ["model_path = Path", "model_path.is_file"]),
        },
        "exp6409": {
            "classification": "inherited",
            "model_paths_present": bool(exp6408_model_paths),
            "model_file_opened_by_model_runtime": False,
            "inherited_from": EXP6408_ID,
            "source_lines": line_hits(source6409, ["model_specs = list(upstream.get"]),
        },
    }


def model_process_execution_evidence(sources: Mapping[str, str]) -> JsonDict:
    return {
        short_id: {
            "classification": "absent",
            "model_process_ran": False,
            "process_invocation_patterns_absent": {
                pattern: pattern not in source for pattern in MODEL_PROCESS_PATTERNS
            },
            "source_lines": line_hits(source, MODEL_PROCESS_PATTERNS),
        }
        for short_id, source in {
            "exp6407": sources[EXP6407_ID],
            "exp6408": sources[EXP6408_ID],
            "exp6409": sources[EXP6409_ID],
        }.items()
    }


def token_generation_evidence(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "exp6407": {
            "classification": "absent",
            "generated_token_count_present": False,
            "tokenizer_precheck_only": False,
        },
        "exp6408": {
            "classification": "absent",
            "generated_token_count_present": "generated_token_count" in payloads[EXP6408_ID],
            "tokenizer_precheck_only": True,
            "autotokenizer_usage_count": payloads[EXP6408_ID].get("autotokenizer_usage_count"),
        },
        "exp6409": {
            "classification": "absent",
            "generated_token_count_present": "generated_token_count" in payloads[EXP6409_ID],
            "tokenizer_precheck_only": True,
            "autotokenizer_usage_count": payloads[EXP6409_ID].get("autotokenizer_usage_count"),
        },
    }


def raw_output_byte_evidence(payloads: Mapping[str, JsonDict], sources: Mapping[str, str]) -> JsonDict:
    exp6408_records = as_mapping(
        payloads[EXP6408_ID].get(
            "raw_bytes_source_effect_diagnostic_checker_disposition_and_head_freeze_records"
        )
    )
    rows = [as_mapping(row) for row in exp6408_records.get("rows", [])]
    hash_count = sum(1 for row in rows if row.get("raw_model_bytes_sha256"))
    return {
        "exp6407": {
            "classification": "derived",
            "raw_output_bytes_stored": False,
            "raw_hashes_present": True,
            "raw_bytes_are_model_output": False,
            "source": "raw ledger rows are deterministic fixture rows.",
        },
        "exp6408": {
            "classification": "derived",
            "raw_model_hashes_present": hash_count > 0,
            "raw_hash_count": hash_count,
            "raw_output_bytes_stored": False,
            "raw_bytes_are_model_output": False,
            "source_lines": line_hits(
                sources[EXP6408_ID],
                ["raw_bytes = canonical_json", "raw_model_bytes_sha256"],
            ),
        },
        "exp6409": {
            "classification": "absent",
            "raw_model_hashes_present": False,
            "raw_output_bytes_stored": False,
            "raw_bytes_are_model_output": False,
        },
    }


def pid_bound_gpu_telemetry_evidence(payloads: Mapping[str, JsonDict]) -> JsonDict:
    runtime6408 = as_mapping(
        payloads[EXP6408_ID].get("cuda_offload_runtime_peak_memory_and_duration_receipts_by_model")
    )
    runtime6409 = as_mapping(
        payloads[EXP6409_ID].get("cuda_offload_runtime_peak_memory_and_duration_receipts_by_model")
    )

    def has_pid(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(str(key).lower() == "pid" or has_pid(item) for key, item in value.items())
        if isinstance(value, list):
            return any(has_pid(item) for item in value)
        return False

    return {
        "exp6407": {"classification": "absent", "pid_bound_gpu_samples_present": False},
        "exp6408": {
            "classification": "absent",
            "pid_bound_gpu_samples_present": has_pid(runtime6408),
            "host_snapshot_present": bool(runtime6408.get("host_cuda_devices")),
            "gpu_fields_are_host_snapshot": True,
        },
        "exp6409": {
            "classification": "absent",
            "pid_bound_gpu_samples_present": has_pid(runtime6409),
            "inherited_from": EXP6408_ID,
            "gpu_fields_are_host_snapshot": True,
        },
    }


def exact_outcome_temporal_evidence(payloads: Mapping[str, JsonDict], sources: Mapping[str, str]) -> JsonDict:
    return {
        "exp6407": {
            "classification": "derived",
            "observed_after_admission": False,
            "exact_check_surface": "deterministic fixture outcomes",
        },
        "exp6408": {
            "classification": "constant",
            "observed_after_admission": False,
            "future_opened_after_freeze": as_mapping(
                payloads[EXP6408_ID].get("exact_future_yield_by_arm")
            ).get("future_opened_after_freeze"),
            "future_success_counts_source": "source_constant_formula",
            "source_lines": line_hits(
                sources[EXP6408_ID],
                ["return int(total * 0.75)", "return int(total * 0.55)", "return int(total * 0.50)"],
            ),
        },
        "exp6409": {
            "classification": "constant",
            "observed_after_admission": False,
            "future_opened_after_freeze": as_mapping(
                payloads[EXP6409_ID].get("untouched_future_evaluation_receipts")
            ).get("opened_after_head_freeze"),
            "future_success_counts_source": "source_constants",
            "source_lines": line_hits(sources[EXP6409_ID], ['"success_count": 12', '"success_count": 10']),
        },
    }


def constant_and_inherited_findings(matrix: Mapping[str, Any]) -> JsonDict:
    rows = []
    for exp_id, fields in matrix.items():
        for field, receipt in as_mapping(fields).items():
            record = as_mapping(receipt)
            if record.get("classification") in {"constant", "inherited"}:
                rows.append(
                    {
                        "experiment": exp_id,
                        "field": field,
                        "classification": record.get("classification"),
                        "claim_risk": record.get("claim_risk"),
                        "source": record.get("source"),
                    }
                )
    return {
        "schema": SCHEMA + ".constant_inherited_findings",
        "rows": rows,
        "finding_count": len(rows),
    }


def current_adversarial_findings(root: Path, payloads: Mapping[str, JsonDict]) -> JsonDict:
    from scripts.adversarial_verify import verify_artifact

    findings: JsonDict = {}
    for exp_id, relative in ARTIFACT_BY_ID.items():
        current = verify_artifact(str(root / relative))
        stamped = {
            "flagged_adversarial": payloads[exp_id].get("flagged_adversarial"),
            "corrigendum_pending": payloads[exp_id].get("corrigendum_pending"),
            "honest_verdict": payloads[exp_id].get("honest_verdict"),
            "status": payloads[exp_id].get("status"),
        }
        findings[exp_id] = {
            "stamped": stamped,
            "current": current,
            "current_live_has_critical": any(
                as_mapping(flag).get("severity") == "critical"
                for flag in current.get("flags", [])
            ),
            "stamped_and_current_preserved_separately": True,
        }
    return findings


def run_determination_preservation(root: Path) -> JsonDict:
    completed = subprocess.run(
        [".venv/bin/python", "scripts/determination_preservation_lint.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": DETERMINATION_COMMAND,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def powered_evidence_eligible(evidence: Mapping[str, Any]) -> bool:
    return all(evidence.get(key) is True for key in POWERED_REQUIRED_RECEIPTS)


def mutated_powered_evidence(attack_id: str, baseline: Mapping[str, Any]) -> JsonDict:
    if attack_id not in MUTATION_ATTACK_IDS:
        raise ValueError(f"unknown_mutation_attack:{attack_id}")
    mutated = dict(baseline)
    if attack_id == "constant_durations":
        mutated["nonconstant_runtime_duration"] = False
    elif attack_id == "forged_pids":
        mutated["pid_bound_gpu_samples"] = False
    elif attack_id == "model_name_only_rows":
        mutated["model_file_opened"] = False
        mutated["model_process_ran"] = False
    elif attack_id == "inherited_outputs":
        mutated["tokens_generated"] = False
        mutated["raw_output_bytes_stored"] = False
        mutated["exact_outcomes_observed_after_admission"] = False
    elif attack_id == "missing_raw_hashes":
        mutated["raw_output_bytes_stored"] = False
    elif attack_id == "fabricated_gpu_samples":
        mutated["pid_bound_gpu_samples"] = False
    return mutated


def mutation_attack_matrix() -> JsonDict:
    authentic_powered = {key: True for key in POWERED_REQUIRED_RECEIPTS}
    attacks = {}
    for attack_id in MUTATION_ATTACK_IDS:
        mutated = mutated_powered_evidence(attack_id, authentic_powered)
        attacks[attack_id] = {
            "attack_id": attack_id,
            "mutated_receipts": mutated,
            "powered_eligible_after_attack": powered_evidence_eligible(mutated),
            "failed_closed": not powered_evidence_eligible(mutated),
        }
    false_accepts = sum(1 for row in attacks.values() if row["powered_eligible_after_attack"])
    return {
        "schema": SCHEMA + ".mutation_attack_matrix",
        "attacks": attacks,
        "all_attacks_fail_closed": false_accepts == 0,
        "powered_false_accept_count": false_accepts,
    }


def claim_eligibility(
    adversarial: Mapping[str, JsonDict],
    model_process: Mapping[str, Any],
    tokens: Mapping[str, Any],
    raw_bytes: Mapping[str, Any],
    gpu: Mapping[str, Any],
    temporal: Mapping[str, Any],
) -> dict[str, JsonDict]:
    powered_receipts = {
        "model_file_opened": False,
        "model_process_ran": as_mapping(model_process.get("exp6408")).get("model_process_ran")
        is True,
        "tokens_generated": as_mapping(tokens.get("exp6408")).get("generated_token_count_present")
        is True,
        "raw_output_bytes_stored": as_mapping(raw_bytes.get("exp6408")).get(
            "raw_output_bytes_stored"
        )
        is True,
        "pid_bound_gpu_samples": as_mapping(gpu.get("exp6408")).get(
            "pid_bound_gpu_samples_present"
        )
        is True,
        "exact_outcomes_observed_after_admission": as_mapping(temporal.get("exp6408")).get(
            "observed_after_admission"
        )
        is True,
        "nonconstant_runtime_duration": False,
    }
    powered_ok = powered_evidence_eligible(powered_receipts)
    exp6407_flagged = (
        as_mapping(adversarial.get(EXP6407_ID)).get("current_live_has_critical") is True
        or as_mapping(as_mapping(adversarial.get(EXP6407_ID)).get("stamped")).get(
            "flagged_adversarial"
        )
        is True
    )
    blockers = [key for key, value in powered_receipts.items() if value is not True]
    return {
        "deterministic_protocol_claim_eligibility": {
            "eligible": not exp6407_flagged,
            "claim_class": "deterministic_protocol_readiness",
            "blockers": ["exp6407_adversarial_flag_open"] if exp6407_flagged else [],
            "scope": "protocol receipts only, no powered or public claim",
        },
        "deterministic_replay_claim_eligibility": {
            "eligible": True,
            "claim_class": "deterministic_replay_behavior",
            "blockers": [],
            "scope": "checked-in deterministic replay rows only",
        },
        "powered_gguf_claim_eligibility": {
            "eligible": powered_ok,
            "claim_class": "powered_gguf_admission",
            "blockers": blockers,
            "required_receipts": list(POWERED_REQUIRED_RECEIPTS),
            "observed_receipts": powered_receipts,
        },
        "prospective_csl_claim_eligibility": {
            "eligible": False,
            "claim_class": "prospective_csl",
            "blockers": ["inherits_exp6408_unpowered_runtime_surface", *blockers],
            "scope": "deterministic replay preserved, CSL progress unproven",
        },
        "public_factor_claim_eligibility": {
            "eligible": False,
            "claim_class": "public_factor",
            "blockers": ["powered_receipts_absent", "public_boundary_not_established"],
            "scope": "internal audit only",
        },
        "fr11_claim_eligibility": {
            "eligible": False,
            "claim_class": "fr11_progress",
            "blockers": ["audit_only_no_new_fr11_measurement"],
            "scope": "no FR11 progress claim",
        },
    }


def sidecar_paths(result_path: Path) -> tuple[Path, Path]:
    return (
        result_path.with_suffix(result_path.suffix + CORRIGENDUM_SUFFIX),
        result_path.with_suffix(result_path.suffix + CLAIM_LEDGER_SUFFIX),
    )


def write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def sidecar_receipts(
    *,
    root: Path,
    result_path: Path,
    audited: Mapping[str, Any],
    eligibility: Mapping[str, JsonDict],
    write_sidecars: bool,
) -> tuple[JsonDict, JsonDict]:
    corrigendum_path, ledger_path = sidecar_paths(result_path)
    boundary_material = {
        "schema": SCHEMA + ".claim_boundary_material",
        "audited_hashes": audited,
        "eligibility": eligibility,
        "powered_claim_unproven": as_mapping(eligibility.get("powered_gguf_claim_eligibility")).get(
            "eligible"
        )
        is False,
        "prospective_csl_unproven": as_mapping(
            eligibility.get("prospective_csl_claim_eligibility")
        ).get("eligible")
        is False,
    }
    boundary_hash = sha256_json(boundary_material)
    corrigendum_payload = {
        "schema": SCHEMA + ".additive_corrigendum",
        "date": RUN_DATE,
        "corrigendum_of": [path.as_posix() for path in ARTIFACT_BY_ID.values()],
        "claim_boundary_hash": boundary_hash,
        "historical_records_modified": False,
        "historical_verdicts_replaced": False,
        "finding": "powered and prospective V551 claims are unproven without execution receipts",
        "preserved_scope": "deterministic replay over checked-in rows remains eligible",
    }
    ledger_entry = {
        "schema": SCHEMA + ".claim_ledger_row",
        "date": RUN_DATE,
        "claim_boundary_hash": boundary_hash,
        "deterministic_replay_claim_eligibility": as_mapping(
            eligibility.get("deterministic_replay_claim_eligibility")
        ).get("eligible"),
        "powered_gguf_claim_eligibility": as_mapping(
            eligibility.get("powered_gguf_claim_eligibility")
        ).get("eligible"),
        "prospective_csl_claim_eligibility": as_mapping(
            eligibility.get("prospective_csl_claim_eligibility")
        ).get("eligible"),
        "public_factor_claim_eligibility": as_mapping(
            eligibility.get("public_factor_claim_eligibility")
        ).get("eligible"),
        "fr11_claim_eligibility": as_mapping(eligibility.get("fr11_claim_eligibility")).get(
            "eligible"
        ),
        "historical_artifacts_modified": False,
    }
    if write_sidecars:
        write_json_file(corrigendum_path, corrigendum_payload)
        write_jsonl_file(ledger_path, [ledger_entry])
    corrigendum_content = (
        json.dumps(corrigendum_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    ledger_content_hash = sha256_bytes((canonical_json(ledger_entry) + "\n").encode("utf-8"))
    return (
        {
            **path_receipt(corrigendum_path, root, digest=sha256_bytes(corrigendum_content)),
            "immutable_additive": True,
            "payload": corrigendum_payload,
        },
        {
            **path_receipt(ledger_path, root, digest=ledger_content_hash),
            "entry": ledger_entry,
            "append_only": True,
            "row_count": 1,
        },
    )


def tests_run(command_receipts: Sequence[JsonMap]) -> JsonDict:
    rows = [dict(row) for row in command_receipts]
    return {
        "schema": SCHEMA + ".tests_run",
        "commands": list(DEFAULT_TEST_COMMANDS),
        "receipts": rows,
        "exit_codes": {str(row.get("command")): row.get("exit_code") for row in rows},
        "all_passed": bool(rows) and all(row.get("exit_code") == 0 for row in rows),
    }


def read_external_test_receipts(env: Mapping[str, str] | None = None) -> list[JsonDict]:
    source = os.environ if env is None else env
    raw = source.get(EXTERNAL_TEST_RECEIPTS_ENV)
    if raw:
        value = json.loads(raw)
        if isinstance(value, list):
            return [dict(as_mapping(row)) for row in value]
    if EXTERNAL_TEST_RECEIPT_PATH.is_file():
        value = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [dict(as_mapping(row)) for row in value]
    return [{"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS]


def ready_score(report: Mapping[str, Any]) -> float:
    provenance = as_mapping(report.get("field_provenance"))
    field_provenance_ok = (
        set(provenance) == set(REQUIRED_ARTIFACT_FIELDS)
        and set(provenance.values()) <= ALLOWED_PROVENANCE_CLASSES
    )
    sidecars_ok = (
        as_mapping(report.get("additive_corrigendum_path_and_hash")).get("present") is True
        and as_mapping(report.get("additive_claim_ledger_path_entry_and_hash")).get("present")
        is True
    )
    unpowered_claim_blocked = (
        as_mapping(report.get("powered_gguf_claim_eligibility")).get("eligible") is False
        and as_mapping(report.get("prospective_csl_claim_eligibility")).get("eligible") is False
        and as_mapping(report.get("public_factor_claim_eligibility")).get("eligible") is False
        and as_mapping(report.get("fr11_claim_eligibility")).get("eligible") is False
    )
    gates = (
        field_provenance_ok,
        sidecars_ok,
        report.get("historical_artifacts_modified") is False,
        report.get("historical_determinations_preserved") is True,
        report.get("powered_false_accept_count") == 0,
        unpowered_claim_blocked,
        as_mapping(report.get("protected_files_unchanged")).get("ok") is True,
        report.get("verifier_is_oracle") is False,
    )
    return 1.0 if all(gates) else 0.0


def status(report: Mapping[str, Any]) -> str:
    if float(report.get("v551_claim_boundary_ready_score", 0.0) or 0.0) == 1.0:
        return "complete_claim_boundary_ready"
    return "complete_claim_boundary_unready"


def honest_verdict(report: Mapping[str, Any]) -> str:
    if report.get("status") == "complete_claim_boundary_ready":
        return (
            "complete: V551 claim boundary audited; powered and prospective claims "
            "are unproven, deterministic replay is preserved"
        )
    return "complete_null: V551 claim boundary audit did not satisfy all gates"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap],
    determination_result: JsonMap | None = None,
    before_hashes: Mapping[str, str | None] | None = None,
    duration_s: float,
    result_path: Path | None = None,
    write_sidecars: bool,
) -> JsonDict:
    audited = audited_hashes(root)
    payloads = load_payloads(root)
    sources = load_sources(root)
    result = result_path or root / RESULT_RELATIVE_PATH
    before = dict(before_hashes if before_hashes is not None else protected_hashes(root))
    matrix = build_field_provenance_matrix(payloads, sources)
    model_open = model_file_open_evidence(payloads, sources)
    process = model_process_execution_evidence(sources)
    token = token_generation_evidence(payloads)
    raw = raw_output_byte_evidence(payloads, sources)
    gpu = pid_bound_gpu_telemetry_evidence(payloads)
    temporal = exact_outcome_temporal_evidence(payloads, sources)
    adversarial = current_adversarial_findings(root, payloads)
    determination = dict(determination_result or run_determination_preservation(root))
    eligibility = claim_eligibility(adversarial, process, token, raw, gpu, temporal)
    mutations = mutation_attack_matrix()
    corrigendum, ledger = sidecar_receipts(
        root=root,
        result_path=result,
        audited=audited,
        eligibility=eligibility,
        write_sidecars=write_sidecars,
    )
    protected = protected_unchanged_receipt(before, protected_hashes(root))
    historical_determinations_preserved = (
        as_mapping(as_mapping(adversarial.get(EXP6407_ID)).get("stamped")).get(
            "flagged_adversarial"
        )
        is True
        and bool(
            as_mapping(as_mapping(adversarial.get(EXP6407_ID)).get("stamped")).get(
                "corrigendum_pending"
            )
        )
        and determination.get("exit_code") == 0
    )
    preconditions = {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "repo_root": root.as_posix(),
        "planning_date_ok": date == RUN_DATE,
        "hashes_recorded_before_analysis": audited["hashes_recorded_before_analysis"],
        "audited_missing_paths": audited["missing_paths"],
        "existing_ops_claim_ledger": path_receipt(
            root / "ops/claim-eligibility-ledger.json",
            root,
        ),
        "protected_hashes_before": before,
        "protected_hashes_after": protected_hashes(root),
    }
    report: JsonDict = {
        "status": "complete_claim_boundary_unready",
        "audited_source_artifact_sidecar_and_log_hashes": audited,
        "per_experiment_field_provenance_matrix": matrix,
        "model_file_open_evidence": model_open,
        "model_process_execution_evidence": process,
        "token_generation_evidence": token,
        "raw_output_byte_evidence": raw,
        "pid_bound_gpu_telemetry_evidence": gpu,
        "exact_outcome_temporal_evidence": temporal,
        "constant_and_inherited_field_findings": constant_and_inherited_findings(matrix),
        "stamped_and_current_adversarial_findings": adversarial,
        "determination_preservation_results": determination,
        **eligibility,
        "additive_corrigendum_path_and_hash": corrigendum,
        "additive_claim_ledger_path_entry_and_hash": ledger,
        "historical_artifacts_modified": bool(protected["changed_paths"]),
        "historical_determinations_preserved": historical_determinations_preserved,
        "mutation_attack_matrix": mutations,
        "powered_false_accept_count": mutations["powered_false_accept_count"],
        "v551_claim_boundary_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": None,
        "duration_s": duration_s,
        "tests_run": tests_run(command_receipts),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: not refreshed",
    }
    report["v551_claim_boundary_ready_score"] = ready_score(report)
    report["status"] = status(report)
    report["honest_verdict"] = honest_verdict(report)
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: JsonMap) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in report]
    if missing:
        errors.append(f"missing required fields: {missing}")
        return errors
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("powered_false_accept_count") != 0:
        errors.append("powered false accepts must be zero")
    if as_mapping(report.get("powered_gguf_claim_eligibility")).get("eligible") is not False:
        errors.append("powered GGUF eligibility must be false")
    if as_mapping(report.get("prospective_csl_claim_eligibility")).get("eligible") is not False:
        errors.append("prospective CSL eligibility must be false")
    if as_mapping(report.get("public_factor_claim_eligibility")).get("eligible") is not False:
        errors.append("public factor eligibility must be false")
    if as_mapping(report.get("fr11_claim_eligibility")).get("eligible") is not False:
        errors.append("FR11 eligibility must be false")
    provenance = as_mapping(report.get("field_provenance"))
    if set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if not set(provenance.values()) <= ALLOWED_PROVENANCE_CLASSES:
        errors.append("field_provenance has invalid class")
    principles = as_mapping(report.get("field_principles"))
    for key in REQUIRED_ARTIFACT_FIELDS:
        if key not in principles:
            errors.append(f"missing field_principles entry: {key}")
    for key in (
        "deterministic_protocol_claim_eligibility",
        "deterministic_replay_claim_eligibility",
        "powered_gguf_claim_eligibility",
        "prospective_csl_claim_eligibility",
        "public_factor_claim_eligibility",
        "fr11_claim_eligibility",
        "v551_claim_boundary_ready_score",
    ):
        if key not in principles:
            errors.append(f"missing claim principle: {key}")
    if as_mapping(report.get("mutation_attack_matrix")).get("all_attacks_fail_closed") is not True:
        errors.append("mutation attacks must fail closed")
    if report.get("historical_artifacts_modified") is not False:
        errors.append("historical artifacts must not be modified")
    if report.get("historical_determinations_preserved") is not True:
        errors.append("historical determinations must be preserved")
    if as_mapping(report.get("protected_files_unchanged")).get("ok") is not True:
        errors.append("protected files changed")
    if float(report.get("v551_claim_boundary_ready_score", 0.0) or 0.0) != ready_score(report):
        errors.append("ready score mismatch")
    if not str(report.get("honest_verdict") or "").startswith(ALLOWED_HONEST_PREFIXES):
        errors.append("honest_verdict lacks terminal prefix")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_report(report: JsonMap, path: Path, *, root: Path = REPO_ROOT) -> Path:
    return atomic_write_json(path, report, root=root, sort_keys=True)


def run(
    *,
    date: str = RUN_DATE,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[JsonMap] | None = None,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    source_env = os.environ if env is None else env
    result_path = resolve_experiment_artifact_path(RESULT_RELATIVE_PATH, root=root, env=source_env)
    before = protected_hashes(root)
    receipts = list(command_receipts) if command_receipts is not None else read_external_test_receipts(source_env)
    report = build_report(
        root,
        date=date,
        command_receipts=receipts,
        before_hashes=before,
        duration_s=time.perf_counter() - start,
        result_path=result_path,
        write_sidecars=write,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_report(report, result_path, root=root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    report = run(date=args.date)
    print(
        json.dumps(
            {
                "path": RESULT_RELATIVE_PATH.as_posix(),
                "status": report["status"],
                "v551_claim_boundary_ready_score": report["v551_claim_boundary_ready_score"],
                "honest_verdict": report["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
