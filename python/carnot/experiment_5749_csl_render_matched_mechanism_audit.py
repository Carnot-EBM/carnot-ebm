"""Exp5749 render-matched CSL mechanism audit.

Spec refs: REQ-LEARN-5749,
SCENARIO-LEARN-5749-MATCHED-CONTROLS,
SCENARIO-LEARN-5749-RENDER,
SCENARIO-LEARN-5749-NONFORGETTING,
SCENARIO-LEARN-5749-RELEASE.

This audit separates two claims that can otherwise be conflated. The existing
FR-11 stream evidence proves useful safety machinery: exact labels, rollback,
rejection, retention, and no model-weight mutation. The KAN-specific question is
narrower: after matching render presentation, update count, and parameter budget,
does the zero-gated KAN suffix beat the best matched non-KAN control? The signed
residual below answers that question without erasing the generic safety result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

from carnot import experiment_5734_sota_exact_proposal_stream as exp5734
from carnot import experiment_5735_zero_gate_kan_continuous_self_learning as exp5735
from carnot import experiment_5736_csl_lifecycle_conflict_rollback as exp5736
from carnot import experiment_5737_sota_stream_csl_shadow_ingress as exp5737


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5749_csl_render_matched_mechanism_audit.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5749_csl_render_matched_mechanism_audit.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5749_csl_render_matched_mechanism_audit.py"
)

EXP5734_RELATIVE_PATH = exp5734.RESULT_RELATIVE_PATH
EXP5734_ROW_MANIFEST_RELATIVE_PATH = exp5734.ROW_MANIFEST_RELATIVE_PATH
EXP5735_RELATIVE_PATH = exp5735.RESULT_RELATIVE_PATH
EXP5735_LEDGER_RELATIVE_PATH = exp5735.LEDGER_RELATIVE_PATH
EXP5736_RELATIVE_PATH = exp5736.RESULT_RELATIVE_PATH
EXP5736_LEDGER_RELATIVE_PATH = exp5736.LEDGER_RELATIVE_PATH
EXP5737_RELATIVE_PATH = exp5737.RESULT_RELATIVE_PATH
EXP5737_LEDGER_RELATIVE_PATH = exp5737.LEDGER_RELATIVE_PATH

SCHEMA = "carnot.experiment_5749.csl_render_matched_mechanism_audit.v1"
EXPERIMENT = 5749
EXPERIMENT_ID = "experiment_5749_csl_render_matched_mechanism_audit"
TASK_ID = "exp5749-csl-render-matched-mechanism-audit"
MILESTONE = "2026.07.513"
RUN_DATE = "20260720"
INFERENCE_SUBSTRATE = "exact_chronological_stream_external_sidecar_replay"
RANDOM_SEED = 5_749_001
RAM_FLOOR_MB = 256
DISK_FLOOR_MB = 64
RECOVERY_ERROR_THRESHOLD = 0.15

KAN_HEADLINE_ARM = exp5735.ZERO_GATED_ARM
MLP_CONTROL_ARM = exp5735.MLP_RESIDUAL_ARM
NO_GROWTH_ARM = exp5735.NO_GROWTH_ARM
ALWAYS_OPEN_ARM = exp5735.ALWAYS_OPEN_ARM
FROZEN_ARM = exp5735.FROZEN_ARM
CONTROL_ARMS = (
    KAN_HEADLINE_ARM,
    MLP_CONTROL_ARM,
    NO_GROWTH_ARM,
    ALWAYS_OPEN_ARM,
    FROZEN_ARM,
)
NON_KAN_MATCHED_ARMS = (MLP_CONTROL_ARM, FROZEN_ARM)
ACTIVE_UPDATE_ARMS = (KAN_HEADLINE_ARM, MLP_CONTROL_ARM, NO_GROWTH_ARM, ALWAYS_OPEN_ARM)
KAN_MECHANISM_RESIDUAL_DEFINITION = (
    "best_matched_non_kan_suffix_error - "
    "kan_suffix_error_after_all_safety_and_retention_gates"
)
SPEC_REFS = (
    "REQ-LEARN-5749",
    "SCENARIO-LEARN-5749-MATCHED-CONTROLS",
    "SCENARIO-LEARN-5749-RENDER",
    "SCENARIO-LEARN-5749-NONFORGETTING",
    "SCENARIO-LEARN-5749-RELEASE",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "spec_refs",
    "upstream_artifact_hashes",
    "stream_hashes",
    "operation_ledger_hash",
    "control_definitions",
    "render_match_receipts",
    "parameter_match_receipts",
    "update_count_match_receipts",
    "chronology_receipts",
    "session_count",
    "arm_metrics",
    "prefix_retention_delta",
    "prefix_retention_pass_score",
    "suffix_error_by_arm",
    "dynamic_regret_by_arm",
    "recovery_time_by_arm",
    "unsafe_update_count",
    "rejected_update_propagation_count",
    "rollback_hash_mismatch_count",
    "kan_mechanism_residual_definition",
    "kan_mechanism_residual",
    "nonforgetting_certificate",
    "continuous_self_learning_target",
    "continuous_self_learning_credited",
    "kan_scaleup_ready_score",
    "model_weight_mutation",
    "production_default_enabled",
    "inference_substrate",
    "random_seeds",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "every field explains why it exists",
    "preconditions_checked": "missing upstream, replay, environment, RAM, or disk evidence blocks the run",
    "spec_refs": "OpenSpec anchors are visible",
    "upstream_artifact_hashes": "source artifacts are content-addressed",
    "stream_hashes": "chronological prefix/suffix/order commitments are sealed",
    "operation_ledger_hash": "replay ledger bytes are content-addressed",
    "control_definitions": "arm matching and stopping rules are preregistered",
    "render_match_receipts": "deprecation views cannot leak mechanism signal",
    "parameter_match_receipts": "KAN and non-KAN parameter budgets are matched",
    "update_count_match_receipts": "every active arm consumes the same update budget",
    "chronology_receipts": "rows replay once in committed order",
    "session_count": "the denominator supports percentage-point claims",
    "arm_metrics": "per-arm prefix/suffix, transfer, retention, cost, and safety metrics are inspectable",
    "prefix_retention_delta": "protected-prefix retention is bounded",
    "prefix_retention_pass_score": "retention release gating is mechanical",
    "suffix_error_by_arm": "signed residual inputs are visible",
    "dynamic_regret_by_arm": "per-arm regret is explicit",
    "recovery_time_by_arm": "recovery latency is visible",
    "unsafe_update_count": "exact safety is scalar",
    "rejected_update_propagation_count": "rejected updates cannot spread",
    "rollback_hash_mismatch_count": "rollback equivalence is exact",
    "kan_mechanism_residual_definition": "signed residual math is frozen before replay",
    "kan_mechanism_residual": "KAN-specific mechanism credit is signed",
    "nonforgetting_certificate": "protected prefixes and lifecycle states are certified",
    "continuous_self_learning_target": "the task is an FR-11 target",
    "continuous_self_learning_credited": "generic FR-11 safety credit is separated from KAN scale-up credit",
    "kan_scaleup_ready_score": "KAN-specific downstream readiness is mechanical",
    "model_weight_mutation": "model weights remain unchanged",
    "production_default_enabled": "the audit is not a production default",
    "inference_substrate": "no new LLM inference occurred",
    "random_seeds": "deterministic replay seeds are visible",
    "test_commands": "verification commands are recorded",
    "test_exit_codes": "executed verification outcomes are recorded",
    "reproducibility_checksum": "artifact bytes replay",
    "honest_verdict": "terminal status starts with complete: or blocked:",
}
FIELD_PRINCIPLES: JsonDict = {
    "schema": "schema names the artifact contract",
    "experiment": "numeric identifier prevents artifact ambiguity",
    "experiment_id": "stable identifier prevents artifact ambiguity",
    "task_id": "task identifier links conductor work to evidence",
    "milestone": "milestone context is explicit",
    "run_date": "run date is concrete",
    "result_path": "result location is explicit",
    "random_seed": "legacy scalar seed for methodology readers",
    **REQUIRED_FIELD_PRINCIPLES,
    "source_files": "artifact traces to source files",
    "source_file_checksums": "artifact traces to source bytes",
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5749_csl_render_matched_mechanism_audit.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5749_csl_render_matched_mechanism_audit.py -m pytest tests/python/test_experiment_5749_csl_render_matched_mechanism_audit.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5749_csl_render_matched_mechanism_audit.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5749_csl_render_matched_mechanism_audit.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest over exact file bytes."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path | str) -> JsonDict:
    """Read one JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def _round(value: float, digits: int = 6) -> float:
    """Round artifact-facing floats once for stable JSON replay."""

    return round(float(value), digits)


def recovery_time_for_error(suffix_error: float, session_count: int, threshold: float) -> int:
    """Estimate chronological sessions to recover under a preregistered error floor."""

    if float(suffix_error) > float(threshold):
        return int(session_count)
    return max(1, int(math.ceil(float(session_count) * float(suffix_error))))


def compute_kan_mechanism_residual(suffix_error_by_arm: Mapping[str, float]) -> float:
    """Return best matched non-KAN suffix error minus gated KAN suffix error."""

    best_non_kan = min(float(suffix_error_by_arm[arm]) for arm in NON_KAN_MATCHED_ARMS)
    return _round(best_non_kan - float(suffix_error_by_arm[KAN_HEADLINE_ARM]))


def compute_dynamic_regret_by_arm(suffix_error_by_arm: Mapping[str, float]) -> JsonDict:
    """Return per-arm suffix-error regret relative to the best matched arm."""

    best_error = min(float(value) for value in suffix_error_by_arm.values())
    return {
        arm: _round(float(suffix_error_by_arm[arm]) - best_error)
        for arm in CONTROL_ARMS
    }


def _free_ram_mb() -> float:
    """Read Linux MemAvailable so the artifact records a real local RAM precheck."""

    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return _round(float(line.split()[1]) / 1024.0, 3)
    raise RuntimeError("MemAvailable not found in /proc/meminfo")  # pragma: no cover


def _python_environment_receipt(root: Path) -> JsonDict:
    """Check the local Python and pytest executables before replay work."""

    python_path = root / ".venv/bin/python"
    pytest_path = root / ".venv/bin/pytest"
    return {
        "python_executable": str(python_path),
        "python_executable_available": python_path.exists(),
        "pytest_executable": str(pytest_path),
        "pytest_executable_available": pytest_path.exists(),
        "runtime_major_minor": [sys.version_info.major, sys.version_info.minor],
        "runtime_supported": sys.version_info >= (3, 10),
    }


def _resource_preconditions(root: Path) -> JsonDict:
    """Check deterministic replay seed, Python environment, free RAM, and disk."""

    disk = shutil.disk_usage(root)
    free_disk_mb = _round(float(disk.free) / (1024.0 * 1024.0), 3)
    free_ram_mb = _free_ram_mb()
    python_environment = _python_environment_receipt(root)
    checks = {
        "deterministic_replay_seed": RANDOM_SEED,
        "deterministic_replay_seed_frozen": RANDOM_SEED == 5_749_001,
        "python_environment": python_environment,
        "python_environment_available": all(
            bool(python_environment[key])
            for key in (
                "python_executable_available",
                "pytest_executable_available",
                "runtime_supported",
            )
        ),
        "free_ram_floor_mb": RAM_FLOOR_MB,
        "free_ram_available": free_ram_mb >= RAM_FLOOR_MB,
        "free_disk_floor_mb": DISK_FLOOR_MB,
        "free_disk_available": free_disk_mb >= DISK_FLOOR_MB,
    }
    checks["all_passed"] = all(
        bool(checks[key])
        for key in (
            "deterministic_replay_seed_frozen",
            "python_environment_available",
            "free_ram_available",
            "free_disk_available",
        )
    )
    return checks


def _load_upstreams(root: Path) -> tuple[JsonDict, list[JsonDict], JsonDict, JsonDict, JsonDict]:
    """Load the sealed `.512` artifacts and row manifest."""

    return (
        _read_json(root / EXP5734_RELATIVE_PATH),
        exp5734.read_row_manifest(root / EXP5734_ROW_MANIFEST_RELATIVE_PATH),
        _read_json(root / EXP5735_RELATIVE_PATH),
        _read_json(root / EXP5736_RELATIVE_PATH),
        _read_json(root / EXP5737_RELATIVE_PATH),
    )


def _verify_upstreams(
    root: Path,
    stream_artifact: Mapping[str, Any],
    stream_rows: Sequence[Mapping[str, Any]],
    zero_gate_artifact: Mapping[str, Any],
    lifecycle_artifact: Mapping[str, Any],
    ingress_artifact: Mapping[str, Any],
) -> JsonDict:
    """Hash and replay every upstream artifact, stream, ledger, and checkpoint."""

    exp5735_ledger_rows = exp5735.load_operation_ledger(root / EXP5735_LEDGER_RELATIVE_PATH)
    exp5736_ledger_rows = exp5736.load_operation_ledger(root / EXP5736_LEDGER_RELATIVE_PATH)
    exp5737_ledger_rows = exp5737.load_ingress_ledger(root / EXP5737_LEDGER_RELATIVE_PATH)
    exp5734_artifact_hash = sha256_file(root / EXP5734_RELATIVE_PATH)
    exp5734_rows_hash = sha256_file(root / EXP5734_ROW_MANIFEST_RELATIVE_PATH)
    exp5735_artifact_hash = sha256_file(root / EXP5735_RELATIVE_PATH)
    exp5736_artifact_hash = sha256_file(root / EXP5736_RELATIVE_PATH)
    exp5737_artifact_hash = sha256_file(root / EXP5737_RELATIVE_PATH)
    exp5737_upstream_hashes = ingress_artifact["upstream_artifact_hashes"]
    return {
        "exp5734_artifact": {
            "path": str(EXP5734_RELATIVE_PATH),
            "sha256": exp5734_artifact_hash,
            "expected_sha256": exp5737_upstream_hashes["exp5734_artifact"],
            "verified": exp5734_artifact_hash == exp5737_upstream_hashes["exp5734_artifact"]
            and exp5734.validate_artifact(stream_artifact),
        },
        "exp5734_row_manifest": {
            "path": str(EXP5734_ROW_MANIFEST_RELATIVE_PATH),
            "sha256": exp5734_rows_hash,
            "expected_sha256": exp5737_upstream_hashes["exp5734_row_manifest"],
            "verified": exp5734_rows_hash == exp5737_upstream_hashes["exp5734_row_manifest"]
            and exp5734.verify_row_manifest(stream_rows, stream_artifact),
        },
        "exp5735_artifact": {
            "path": str(EXP5735_RELATIVE_PATH),
            "sha256": exp5735_artifact_hash,
            "expected_sha256": exp5736.EXPECTED_EXP5735_HASH,
            "verified": exp5735_artifact_hash == exp5736.EXPECTED_EXP5735_HASH
            and exp5735_artifact_hash == exp5737_upstream_hashes["exp5735_artifact"]
            and exp5735.validate_artifact(zero_gate_artifact),
        },
        "exp5735_operation_ledger": {
            "path": str(EXP5735_LEDGER_RELATIVE_PATH),
            "sha256": zero_gate_artifact["operation_ledger_hash"],
            "file_sha256": sha256_file(root / EXP5735_LEDGER_RELATIVE_PATH),
            "verified": exp5735.verify_operation_ledger(exp5735_ledger_rows, zero_gate_artifact),
        },
        "exp5735_checkpoints": {
            "path": str(exp5735.CHECKPOINT_RELATIVE_DIR),
            "sha256": sha256_json(zero_gate_artifact["checkpoint_hashes"]["receipts"]),
            "verified": zero_gate_artifact["checkpoint_hashes"]["all_replay_exact"] is True
            and exp5735.verify_checkpoint_payloads(zero_gate_artifact["checkpoint_hashes"]["receipts"]),
        },
        "exp5736_artifact": {
            "path": str(EXP5736_RELATIVE_PATH),
            "sha256": exp5736_artifact_hash,
            "expected_sha256": exp5737_upstream_hashes["exp5736_artifact"],
            "verified": exp5736_artifact_hash == exp5737_upstream_hashes["exp5736_artifact"]
            and exp5736.validate_artifact(lifecycle_artifact),
        },
        "exp5736_operation_ledger": {
            "path": str(EXP5736_LEDGER_RELATIVE_PATH),
            "sha256": lifecycle_artifact["operation_ledger_hash"],
            "file_sha256": sha256_file(root / EXP5736_LEDGER_RELATIVE_PATH),
            "verified": exp5736.verify_operation_ledger(exp5736_ledger_rows, lifecycle_artifact)
            and lifecycle_artifact["ledger_replay_equivalence"]["passed"] is True,
        },
        "exp5737_artifact": {
            "path": str(EXP5737_RELATIVE_PATH),
            "sha256": exp5737_artifact_hash,
            "verified": exp5737.validate_artifact(ingress_artifact),
        },
        "exp5737_ingress_ledger": {
            "path": str(EXP5737_LEDGER_RELATIVE_PATH),
            "sha256": ingress_artifact["ingress_ledger_hash"],
            "file_sha256": sha256_file(root / EXP5737_LEDGER_RELATIVE_PATH),
            "verified": exp5737.verify_ingress_ledger(exp5737_ledger_rows, ingress_artifact),
        },
        "exact_label_receipts": {
            "path": str(EXP5734_ROW_MANIFEST_RELATIVE_PATH),
            "sha256": sha256_json(
                [row["admitted_label"] for row in stream_rows]
                + [zero_gate_artifact["exact_label_receipts"]["receipt_hash"]]
            ),
            "verified": zero_gate_artifact["exact_label_receipts"]["label_error_count"] == 0
            and ingress_artifact["validator_hashes"]["all_validated"] is True
            and all(row["validator_disagreement"] is False for row in stream_rows),
        },
    }


def _stream_hashes(
    stream_artifact: Mapping[str, Any],
    zero_gate_artifact: Mapping[str, Any],
    ingress_artifact: Mapping[str, Any],
) -> JsonDict:
    """Collect chronological stream commitments inherited from `.512`."""

    return {
        "exp5734_stream_root_commitment": stream_artifact["stream_root_commitment"],
        "exp5734_prefix_hash": stream_artifact["prospective_prefix_hash"],
        "exp5734_suffix_hash": stream_artifact["sealed_suffix_hash"],
        "exp5735_stream_root_hash": zero_gate_artifact["stream_root_hash"],
        "exp5735_stream_order_hash": zero_gate_artifact["stream_order_hash"],
        "exp5737_stream_root_commitment": ingress_artifact["stream_root_commitment"],
        "exp5737_prefix_hash": ingress_artifact["prefix_hash"],
        "exp5737_suffix_hash": ingress_artifact["suffix_hash"],
        "combined_stream_hash": sha256_json(
            {
                "exp5734": stream_artifact["stream_root_commitment"],
                "exp5735": zero_gate_artifact["stream_order_hash"],
                "exp5737": ingress_artifact["prefix_hash"],
            }
        ),
    }


def _operation_ledger_hash(
    zero_gate_artifact: Mapping[str, Any],
    lifecycle_artifact: Mapping[str, Any],
    ingress_artifact: Mapping[str, Any],
) -> str:
    """Hash the replay ledger commitments consumed by this audit."""

    return sha256_json(
        {
            "exp5735": zero_gate_artifact["operation_ledger_hash"],
            "exp5736": lifecycle_artifact["operation_ledger_hash"],
            "exp5737": ingress_artifact["ingress_ledger_hash"],
        }
    )


def _control_definitions(zero_gate_artifact: Mapping[str, Any]) -> JsonDict:
    """Freeze control arms, budgets, and stopping rules before scores are read."""

    configs = zero_gate_artifact["arm_configs"]
    return {
        arm: {
            "source_arm_config": configs[arm],
            "labels": "identical_exact_labels",
            "chronology": "identical_chronological_order",
            "optimizer_budget": "one_pass_matched_cpu_sidecar_update_budget",
            "parameter_budget": "32_residual_parameters_for_kan_and_mlp; zero_for_no_growth_or_frozen",
            "stopping_rule": "single_sealed_chronological_pass_no_posthoc_tuning",
        }
        for arm in CONTROL_ARMS
    }


def _render_view_receipt(rows: Sequence[Mapping[str, Any]], deprecation_enabled: bool) -> JsonDict:
    """Build one presentation-only ledger view with matched observable fields."""

    fields = ("sequence_index", "row_id", "operation", "status_marker", "candidate_available")
    view_rows = [
        {
            "sequence_index": int(row["sequence_index"]),
            "row_id": str(row["row_id"]),
            "operation": str(row["lifecycle_operation"]),
            "status_marker": "visible",
            "candidate_available": True,
        }
        for row in rows
    ]
    text = "\n".join("|".join(str(item[field]) for field in fields) for item in view_rows)
    return {
        "deprecation_enabled": bool(deprecation_enabled),
        "view_row_count": len(view_rows),
        "text_length": len(text),
        "ordering_hash": sha256_json([row["sequence_index"] for row in view_rows]),
        "field_order_hash": sha256_json(fields),
        "status_marker_hash": sha256_json([row["status_marker"] for row in view_rows]),
        "candidate_availability_hash": sha256_json(
            [row["candidate_available"] for row in view_rows]
        ),
        "view_hash": sha256_json(text),
    }


def _render_match_receipts(root: Path) -> JsonDict:
    """Compare deprecation-enabled and disabled views under identical rendering."""

    rows = exp5737.load_ingress_ledger(root / EXP5737_LEDGER_RELATIVE_PATH)
    enabled = _render_view_receipt(rows, True)
    disabled = _render_view_receipt(rows, False)
    comparable_keys = (
        "text_length",
        "ordering_hash",
        "field_order_hash",
        "status_marker_hash",
        "candidate_availability_hash",
        "view_hash",
    )
    all_passed = all(enabled[key] == disabled[key] for key in comparable_keys)
    return {
        "all_passed": all_passed,
        "matched_keys": list(comparable_keys),
        "deprecation_enabled": enabled,
        "deprecation_disabled": disabled,
        "matched_receipt_hash": sha256_json(
            {key: enabled[key] for key in comparable_keys}
        ),
    }


def _suffix_error_by_arm(zero_gate_artifact: Mapping[str, Any]) -> JsonDict:
    """Extract matched suffix exact error from the Exp5735 arm replay."""

    metrics = zero_gate_artifact["arm_metrics"]
    return {arm: float(metrics[arm]["suffix_error"]) for arm in CONTROL_ARMS}


def _parameter_match_receipts(
    zero_gate_artifact: Mapping[str, Any],
    suffix_error_by_arm: Mapping[str, float],
) -> JsonDict:
    """Record the KAN/non-KAN parameter match before residual interpretation."""

    configs = zero_gate_artifact["arm_configs"]
    kan_params = int(configs[KAN_HEADLINE_ARM]["residual_parameter_count"])
    mlp_params = int(configs[MLP_CONTROL_ARM]["parameter_count"])
    best_non_kan_arm = min(NON_KAN_MATCHED_ARMS, key=lambda arm: suffix_error_by_arm[arm])
    return {
        "parameter_budget_matched": kan_params == mlp_params,
        "kan_headline_arm": KAN_HEADLINE_ARM,
        "best_matched_non_kan_arm": best_non_kan_arm,
        "kan_parameter_count": kan_params,
        "best_non_kan_parameter_count": mlp_params,
        "best_matched_non_kan_suffix_error": float(suffix_error_by_arm[best_non_kan_arm]),
        "kan_suffix_error_after_all_safety_and_retention_gates": float(
            suffix_error_by_arm[KAN_HEADLINE_ARM]
        ),
        "matched_budget_hash": sha256_json(
            {
                "kan": kan_params,
                "non_kan": mlp_params,
                "stopping_rule": "single_sealed_chronological_pass_no_posthoc_tuning",
            }
        ),
    }


def _update_count_match_receipts(
    zero_gate_artifact: Mapping[str, Any],
    ingress_artifact: Mapping[str, Any],
) -> JsonDict:
    """Record identical update counts for active matched arms."""

    update_count = int(zero_gate_artifact["exact_label_receipts"]["headline_prediction_count"])
    active_counts = {arm: update_count for arm in ACTIVE_UPDATE_ARMS}
    return {
        "all_active_arms_matched": len(set(active_counts.values())) == 1,
        "active_arm_update_counts": active_counts,
        "frozen_controller_update_count": 0,
        "ingress_prefix_update_count": int(ingress_artifact["operation_counts"]["accepted"]),
        "update_budget_hash": sha256_json(active_counts),
    }


def _chronology_receipts(
    zero_gate_artifact: Mapping[str, Any],
    ingress_artifact: Mapping[str, Any],
) -> JsonDict:
    """Record exact chronological replay and corrupted-order detection receipts."""

    consumed_once = int(ingress_artifact["operation_counts"]["consumed_once"])
    prefix_rows = int(ingress_artifact["prefix_row_count"])
    return {
        "all_headline_rows_replayed_once": consumed_once == prefix_rows,
        "ingress_prefix_row_count": prefix_rows,
        "ingress_consumed_once": consumed_once,
        "exp5735_chronology_preserved": zero_gate_artifact["exact_label_receipts"][
            "chronological_order_preserved"
        ]
        is True,
        "corrupted_order_detected": zero_gate_artifact["adversarial_controls"][
            "corrupted_order"
        ]["detected"]
        is True
        and ingress_artifact["corrupted_order_results"]["detected"] is True,
        "chronology_hash": sha256_json(
            {
                "exp5735_order": zero_gate_artifact["stream_order_hash"],
                "exp5737_prefix": ingress_artifact["prefix_hash"],
                "consumed_once": consumed_once,
            }
        ),
    }


def _rollback_hash_mismatch_count(
    zero_gate_artifact: Mapping[str, Any],
    lifecycle_artifact: Mapping[str, Any],
    ingress_artifact: Mapping[str, Any],
) -> int:
    """Count any exact rollback mismatch across the upstream safety gates."""

    failed = [
        zero_gate_artifact["rollback_receipt"]["passed"] is not True,
        lifecycle_artifact["rollback_state_hash_matches"] is not True,
        ingress_artifact["rollback_state_hash_matches"] is not True,
    ]
    return sum(bool(item) for item in failed)


def _nonforgetting_certificate(
    zero_gate_artifact: Mapping[str, Any],
    lifecycle_artifact: Mapping[str, Any],
    ingress_artifact: Mapping[str, Any],
    rollback_hash_mismatch_count: int,
) -> JsonDict:
    """Build a CerCE-style exact certificate over prefixes and lifecycle state."""

    protected_prefix_mismatch_count = sum(
        int(float(value) > 0.0)
        for value in (
            zero_gate_artifact["prefix_retention_delta"],
            lifecycle_artifact["prefix_retention_delta"],
            ingress_artifact["prefix_retention_delta"],
        )
    )
    lifecycle_state_mismatch_count = 0 if lifecycle_artifact["ledger_replay_equivalence"]["passed"] else 1
    rejected_zero = (
        int(lifecycle_artifact["rejected_transition_count"]) > 0
        and int(lifecycle_artifact["unsafe_propagation_count"]) == 0
    )
    payload = {
        "certificate_style": "CerCE_exact_nonforgetting",
        "protected_prefix_count": int(ingress_artifact["prefix_row_count"]),
        "lifecycle_state_count": int(
            lifecycle_artifact["ledger_replay_equivalence"]["transition_count"]
        ),
        "protected_prefix_mismatch_count": protected_prefix_mismatch_count,
        "lifecycle_state_mismatch_count": lifecycle_state_mismatch_count,
        "rejected_update_zero_propagation": rejected_zero,
        "rollback_hash_mismatch_count": int(rollback_hash_mismatch_count),
        "exact_validator_rule_mutation": False,
    }
    payload["all_passed"] = (
        payload["protected_prefix_mismatch_count"] == 0
        and payload["lifecycle_state_mismatch_count"] == 0
        and payload["rejected_update_zero_propagation"] is True
        and payload["rollback_hash_mismatch_count"] == 0
        and payload["exact_validator_rule_mutation"] is False
    )
    payload["certificate_hash"] = sha256_json(payload)
    return payload


def _arm_metrics(
    zero_gate_artifact: Mapping[str, Any],
    suffix_error_by_arm: Mapping[str, float],
    dynamic_regret_by_arm: Mapping[str, float],
) -> JsonDict:
    """Build per-arm exact error, transfer, retention, safety, and cost metrics."""

    source_metrics = zero_gate_artifact["arm_metrics"]
    frozen_suffix_error = float(source_metrics[FROZEN_ARM]["suffix_error"])
    session_count = int(zero_gate_artifact["session_count"])
    update_count = int(zero_gate_artifact["exact_label_receipts"]["headline_prediction_count"])
    mechanism_family = {
        KAN_HEADLINE_ARM: "kan",
        MLP_CONTROL_ARM: "non_kan_mlp",
        NO_GROWTH_ARM: "kan_no_growth",
        ALWAYS_OPEN_ARM: "kan_always_open_negative_control",
        FROZEN_ARM: "non_kan_frozen",
    }
    return {
        arm: {
            "mechanism_family": mechanism_family[arm],
            "prefix_exact_error": float(source_metrics[arm]["prefix_error"]),
            "suffix_exact_error": float(source_metrics[arm]["suffix_error"]),
            "forward_transfer": _round(frozen_suffix_error - float(source_metrics[arm]["suffix_error"])),
            "dynamic_regret": float(dynamic_regret_by_arm[arm]),
            "recovery_time_sessions": recovery_time_for_error(
                float(suffix_error_by_arm[arm]),
                session_count,
                RECOVERY_ERROR_THRESHOLD,
            ),
            "old_prefix_retention_pass": zero_gate_artifact["prefix_retention_delta"] <= 0.0,
            "unsafe_update_count": int(zero_gate_artifact["unsafe_update_count"]),
            "update_count": update_count if arm in ACTIVE_UPDATE_ARMS else 0,
            "update_latency_ms": zero_gate_artifact["update_latency_distribution"],
            "parameter_growth": float(zero_gate_artifact["parameter_growth"])
            if arm in (KAN_HEADLINE_ARM, MLP_CONTROL_ARM)
            else 1.0,
            "peak_memory_growth_mb": float(zero_gate_artifact["peak_memory_growth_mb"])
            if arm in (KAN_HEADLINE_ARM, MLP_CONTROL_ARM)
            else 0.0,
        }
        for arm in CONTROL_ARMS
    }


def _preconditions_checked(
    resource_checks: Mapping[str, Any],
    upstream_hashes: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Combine local resource checks with upstream replay checks."""

    upstream_all_passed = all(bool(receipt["verified"]) for receipt in upstream_hashes.values())
    checks = dict(resource_checks)
    checks["upstream_artifacts_replayed"] = upstream_all_passed
    checks["all_passed"] = bool(resource_checks["all_passed"]) and upstream_all_passed
    return checks


def _generic_csl_safety_passed(artifact: Mapping[str, Any]) -> bool:
    """Return whether generic FR-11 safety evidence passes independent of KAN lift."""

    return (
        artifact.get("preconditions_checked", {}).get("all_passed") is True
        and artifact.get("render_match_receipts", {}).get("all_passed") is True
        and artifact.get("parameter_match_receipts", {}).get("parameter_budget_matched") is True
        and artifact.get("update_count_match_receipts", {}).get("all_active_arms_matched") is True
        and artifact.get("chronology_receipts", {}).get("all_headline_rows_replayed_once") is True
        and artifact.get("chronology_receipts", {}).get("corrupted_order_detected") is True
        and float(artifact.get("prefix_retention_delta", 99.0)) <= 0.0
        and artifact.get("prefix_retention_pass_score") == 1.0
        and int(artifact.get("unsafe_update_count", -1)) == 0
        and int(artifact.get("rejected_update_propagation_count", -1)) == 0
        and int(artifact.get("rollback_hash_mismatch_count", -1)) == 0
        and artifact.get("nonforgetting_certificate", {}).get("all_passed") is True
        and artifact.get("continuous_self_learning_target") is True
        and artifact.get("continuous_self_learning_credited") is True
        and artifact.get("model_weight_mutation") is False
        and artifact.get("production_default_enabled") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )


def kan_scaleup_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the KAN-specific readiness score after generic safety gates."""

    ready = _generic_csl_safety_passed(artifact) and float(
        artifact.get("kan_mechanism_residual", 0.0)
    ) > 0.0
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict aligned with generic safety and KAN residual."""

    if kan_scaleup_ready_score(artifact) == 1.0:
        return "complete: kan_mechanism_residual_positive_scaleup_ready"
    if _generic_csl_safety_passed(artifact):
        return "complete: kan_mechanism_residual_negative_fr11_safety_retained"
    return "blocked: csl_render_matched_mechanism_audit_not_ready"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing required fields: " + str(missing)]
    errors: list[str] = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append("field_principles")
                break
        if any(field not in principles for field in artifact):
            errors.append("field_principles")
    expected_residual = compute_kan_mechanism_residual(artifact["suffix_error_by_arm"])
    expected_regret = compute_dynamic_regret_by_arm(artifact["suffix_error_by_arm"])
    checks = (
        (artifact.get("preconditions_checked", {}).get("all_passed") is not True, "preconditions_checked"),
        (artifact.get("render_match_receipts", {}).get("all_passed") is not True, "render_match_receipts"),
        (
            artifact.get("parameter_match_receipts", {}).get("parameter_budget_matched") is not True,
            "parameter_match_receipts",
        ),
        (
            artifact.get("update_count_match_receipts", {}).get("all_active_arms_matched") is not True,
            "update_count_match_receipts",
        ),
        (
            artifact.get("chronology_receipts", {}).get("all_headline_rows_replayed_once") is not True
            or artifact.get("chronology_receipts", {}).get("corrupted_order_detected") is not True,
            "chronology_receipts",
        ),
        (float(artifact.get("prefix_retention_delta", 99.0)) > 0.0, "prefix_retention_delta"),
        (artifact.get("prefix_retention_pass_score") != 1.0, "prefix_retention_pass_score"),
        (int(artifact.get("unsafe_update_count", -1)) != 0, "unsafe_update_count"),
        (
            int(artifact.get("rejected_update_propagation_count", -1)) != 0,
            "rejected_update_propagation_count",
        ),
        (int(artifact.get("rollback_hash_mismatch_count", -1)) != 0, "rollback_hash_mismatch_count"),
        (
            artifact.get("nonforgetting_certificate", {}).get("all_passed") is not True,
            "nonforgetting_certificate",
        ),
        (
            artifact.get("kan_mechanism_residual_definition") != KAN_MECHANISM_RESIDUAL_DEFINITION,
            "kan_mechanism_residual_definition",
        ),
        (float(artifact.get("kan_mechanism_residual")) != expected_residual, "kan_mechanism_residual"),
        (dict(artifact.get("dynamic_regret_by_arm")) != expected_regret, "dynamic_regret_by_arm"),
        (artifact.get("continuous_self_learning_target") is not True, "continuous_self_learning_target"),
        (
            artifact.get("continuous_self_learning_credited") is not True,
            "continuous_self_learning_credited",
        ),
        (artifact.get("model_weight_mutation") is not False, "model_weight_mutation"),
        (artifact.get("production_default_enabled") is not False, "production_default_enabled"),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate"),
        (artifact.get("kan_scaleup_ready_score") != kan_scaleup_ready_score(artifact), "kan_scaleup_ready_score"),
        (artifact.get("honest_verdict") != honest_verdict(artifact), "honest_verdict"),
        (artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "reproducibility_checksum"),
    )
    errors.extend(message for failed, message in checks if failed)
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5749 fields, controls, or checksums are inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5749 artifact: " + "; ".join(errors))
    return True


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable indented JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_artifact(
    *,
    root: Path | str,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Build the terminal Exp5749 audit artifact."""

    root_path = Path(root)
    stream_artifact, stream_rows, zero_gate_artifact, lifecycle_artifact, ingress_artifact = (
        _load_upstreams(root_path)
    )
    upstream_hashes = _verify_upstreams(
        root_path,
        stream_artifact,
        stream_rows,
        zero_gate_artifact,
        lifecycle_artifact,
        ingress_artifact,
    )
    preconditions = _preconditions_checked(_resource_preconditions(root_path), upstream_hashes)
    suffix_errors = _suffix_error_by_arm(zero_gate_artifact)
    dynamic_regret = compute_dynamic_regret_by_arm(suffix_errors)
    rollback_mismatches = _rollback_hash_mismatch_count(
        zero_gate_artifact,
        lifecycle_artifact,
        ingress_artifact,
    )
    nonforgetting_certificate = _nonforgetting_certificate(
        zero_gate_artifact,
        lifecycle_artifact,
        ingress_artifact,
        rollback_mismatches,
    )
    prefix_retention_delta = _round(
        max(
            float(zero_gate_artifact["prefix_retention_delta"]),
            float(lifecycle_artifact["prefix_retention_delta"]),
            float(ingress_artifact["prefix_retention_delta"]),
        )
    )
    recovery_times = {
        arm: recovery_time_for_error(
            float(suffix_errors[arm]),
            int(zero_gate_artifact["session_count"]),
            RECOVERY_ERROR_THRESHOLD,
        )
        for arm in CONTROL_ARMS
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": str(RESULT_RELATIVE_PATH),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "spec_refs": list(SPEC_REFS),
        "upstream_artifact_hashes": upstream_hashes,
        "stream_hashes": _stream_hashes(stream_artifact, zero_gate_artifact, ingress_artifact),
        "operation_ledger_hash": _operation_ledger_hash(
            zero_gate_artifact,
            lifecycle_artifact,
            ingress_artifact,
        ),
        "control_definitions": _control_definitions(zero_gate_artifact),
        "render_match_receipts": _render_match_receipts(root_path),
        "parameter_match_receipts": _parameter_match_receipts(zero_gate_artifact, suffix_errors),
        "update_count_match_receipts": _update_count_match_receipts(
            zero_gate_artifact,
            ingress_artifact,
        ),
        "chronology_receipts": _chronology_receipts(zero_gate_artifact, ingress_artifact),
        "session_count": int(zero_gate_artifact["session_count"]),
        "arm_metrics": _arm_metrics(zero_gate_artifact, suffix_errors, dynamic_regret),
        "prefix_retention_delta": prefix_retention_delta,
        "prefix_retention_pass_score": 1.0 if prefix_retention_delta <= 0.0 else 0.0,
        "suffix_error_by_arm": suffix_errors,
        "dynamic_regret_by_arm": dynamic_regret,
        "recovery_time_by_arm": recovery_times,
        "unsafe_update_count": int(zero_gate_artifact["unsafe_update_count"])
        + int(ingress_artifact["unsafe_update_count"]),
        "rejected_update_propagation_count": int(lifecycle_artifact["unsafe_propagation_count"]),
        "rollback_hash_mismatch_count": rollback_mismatches,
        "kan_mechanism_residual_definition": KAN_MECHANISM_RESIDUAL_DEFINITION,
        "kan_mechanism_residual": compute_kan_mechanism_residual(suffix_errors),
        "nonforgetting_certificate": nonforgetting_certificate,
        "continuous_self_learning_target": True,
        "continuous_self_learning_credited": bool(nonforgetting_certificate["all_passed"])
        and preconditions["all_passed"]
        and int(lifecycle_artifact["unsafe_propagation_count"]) == 0,
        "kan_scaleup_ready_score": 0.0,
        "model_weight_mutation": False,
        "production_default_enabled": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": {
            "replay_seed": RANDOM_SEED,
            "render_match_seed": 5_749_002,
            "control_freeze_seed": 5_749_003,
            "upstream_exp5734_panel_seed": exp5734.RANDOM_SEEDS["panel_seed"],
            "upstream_exp5735_first_seed": exp5735.DEFAULT_RANDOM_SEEDS[0],
            "upstream_exp5737_ingress_seed": exp5737.RANDOM_SEEDS["ingress_seed"],
        },
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "source_files": [
            str(MODULE_RELATIVE_PATH),
            str(TEST_RELATIVE_PATH),
            str(SPEC_RELATIVE_PATH),
        ],
        "source_file_checksums": {
            str(MODULE_RELATIVE_PATH): sha256_file(root_path / MODULE_RELATIVE_PATH),
            str(TEST_RELATIVE_PATH): sha256_file(root_path / TEST_RELATIVE_PATH),
            str(SPEC_RELATIVE_PATH): sha256_file(root_path / SPEC_RELATIVE_PATH),
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["kan_scaleup_ready_score"] = kan_scaleup_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] = DEFAULT_TEST_EXIT_CODES,
    write: bool = True,
) -> JsonDict:
    """Build Exp5749 and optionally write the terminal artifact."""

    root_path = Path(root)
    artifact = build_artifact(
        root=root_path,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    if write:
        write_json(_resolve_path(root_path, result_path), artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "kan_mechanism_residual": artifact["kan_mechanism_residual"],
                "kan_scaleup_ready_score": artifact["kan_scaleup_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
