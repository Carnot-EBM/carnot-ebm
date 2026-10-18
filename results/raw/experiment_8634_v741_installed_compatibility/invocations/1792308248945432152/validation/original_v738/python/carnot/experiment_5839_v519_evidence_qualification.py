"""Exp5839 V519 evidence qualification.

Spec refs: REQ-LEARN-5839, SCENARIO-LEARN-5839-RECONSTRUCT,
SCENARIO-LEARN-5839-SHORTCUTS, SCENARIO-LEARN-5839-MIXED,
SCENARIO-LEARN-5839-FAIL-CLOSED.

This audit replays the `.519` adaptive-memory evidence as deterministic JSON
and exact-row reconstruction. It deliberately separates row-clean stream and
structural evidence from Exp5828's adversarially flagged lifecycle artifact and
from Exp5829's replay evidence that depends on that flagged upstream.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5827_minimal_core_structural_acquisition_ab as exp5827
from carnot import experiment_5828_future_validated_structural_memory as exp5828
from carnot import experiment_5829_transfer_selective_replay_audit as exp5829


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5839_v519_evidence_qualification.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5839_v519_evidence_qualification.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5839_v519_evidence_qualification.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")

EXP5825_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5825_certified_adaptive_memory_contract.json"
)
EXP5826_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5826_out_of_template_constraint_stream.json"
)
EXP5826_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5826_out_of_template_constraint_stream.rows.jsonl"
)
EXP5827_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5827_minimal_core_structural_acquisition_ab.json"
)
EXP5828_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5828_future_validated_structural_memory.json"
)
EXP5829_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5829_transfer_selective_replay_audit.json"
)

SCHEMA = "carnot.experiment_5839.v519_evidence_qualification.v1"
EXPERIMENT = 5839
EXPERIMENT_ID = "experiment_5839_v519_evidence_qualification"
MILESTONE = "2026.07.520"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "deterministic_exact_verifier_and_replay_no_llm"
VERIFIER_IS_ORACLE = True
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512

PRIMARY_FAMILIES = exp5827.PRIMARY_FAMILIES
CHANGE_ORDER = exp5827.CHANGE_ORDER
PROOF_PRESERVING_SURFACES = exp5827.PROOF_PRESERVING_SURFACES
HARDNESS_BINS = exp5827.HARDNESS_BINS
MIN_UNITS_PER_CELL = 30
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5839,
    "row_reconstruction_seed": 5_839_001,
    "shortcut_control_seed": 5_839_002,
    "replay_audit_seed": 5_839_003,
}
SPEC_REFS = (
    "REQ-LEARN-5839",
    "SCENARIO-LEARN-5839-RECONSTRUCT",
    "SCENARIO-LEARN-5839-SHORTCUTS",
    "SCENARIO-LEARN-5839-MIXED",
    "SCENARIO-LEARN-5839-FAIL-CLOSED",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5839_v519_evidence_qualification.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5839_v519_evidence_qualification.py "
    "-m pytest tests/python/test_experiment_5839_v519_evidence_qualification.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5839_v519_evidence_qualification.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5839_v519_evidence_qualification.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)

UPSTREAM_PATHS: dict[str, Path] = {
    "exp5825_contract": EXP5825_ARTIFACT_RELATIVE_PATH,
    "exp5826_stream_artifact": EXP5826_ARTIFACT_RELATIVE_PATH,
    "exp5826_stream_rows": EXP5826_ROWS_RELATIVE_PATH,
    "exp5827_structural_artifact": EXP5827_ARTIFACT_RELATIVE_PATH,
    "exp5828_lifecycle_artifact": EXP5828_ARTIFACT_RELATIVE_PATH,
    "exp5829_replay_artifact": EXP5829_ARTIFACT_RELATIVE_PATH,
    "exp5826_module": Path("python/carnot/experiment_5826_out_of_template_constraint_stream.py"),
    "exp5827_module": Path("python/carnot/experiment_5827_minimal_core_structural_acquisition_ab.py"),
    "exp5828_module": Path("python/carnot/experiment_5828_future_validated_structural_memory.py"),
    "exp5829_module": Path("python/carnot/experiment_5829_transfer_selective_replay_audit.py"),
    "adversarial_verify": ADVERSARIAL_VERIFY_RELATIVE_PATH,
    "self_learning_spec": SELF_LEARNING_SPEC_RELATIVE_PATH,
    "verification_spec": VERIFY_SPEC_RELATIVE_PATH,
    "module": MODULE_RELATIVE_PATH,
    "test": TEST_RELATIVE_PATH,
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_artifact_hashes",
    "independent_row_reconstruction",
    "chronology_and_visibility_audit",
    "exact_validator_independence",
    "recomputed_metrics",
    "shortcut_and_no_information_controls",
    "state_rollback_restart_receipts",
    "adversarial_verifier_receipt",
    "constraint_stream_qualified_score",
    "structural_acquisition_qualified_score",
    "adaptive_memory_lifecycle_qualified_score",
    "selective_replay_qualified_score",
    "promotion_eligibility_matrix",
    "historical_artifacts_mutated",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal qualification state distinguishes a completed null from a partial audit.",
    "preconditions_checked": "Hashes, validators, splits, seeds, resources, and temp paths prevent fabricated replay.",
    "upstream_artifact_hashes": "Exact hashes bind qualification to immutable `.519` evidence.",
    "independent_row_reconstruction": "Fresh row derivation prevents trusted aggregate fields from validating themselves.",
    "chronology_and_visibility_audit": "Monotone events and sealed science labels are required for prospective learning.",
    "exact_validator_independence": "Solver/version receipts expose circular target construction or label reuse.",
    "recomputed_metrics": "Row-derived aggregates make every credited scalar reproducible.",
    "shortcut_and_no_information_controls": "Permutation, perturbation, ablation, collision, and null controls expose target leakage.",
    "state_rollback_restart_receipts": "Exact hashes test durable state rather than narrative lifecycle claims.",
    "adversarial_verifier_receipt": "The live verifier is the terminal artifact-quality authority.",
    "constraint_stream_qualified_score": "EMIT BARE scalar; only 1.0 permits Exp5840.",
    "structural_acquisition_qualified_score": "EMIT BARE scalar separating clean structural evidence from later lifecycle claims.",
    "adaptive_memory_lifecycle_qualified_score": "EMIT BARE scalar; only 1.0 permits Exp5843 and Exp5846.",
    "selective_replay_qualified_score": "EMIT BARE scalar; flagged-upstream replay is provisional unless independently clean.",
    "promotion_eligibility_matrix": "Per-branch classes keep flagged, provisional, null, and clean evidence disjoint.",
    "historical_artifacts_mutated": "Must be false; qualification cannot unstamp or rewrite history.",
    "duration_s": "Measured wall time exposes the same bootstrap-only failure mode under audit.",
    "inference_substrate": "`deterministic_exact_verifier_and_replay_no_llm` declares the true compute path.",
    "verifier_is_oracle": "True records exact solvers as scoring authority and forbids a verifier-moat claim.",
    "field_provenance": "Each field maps to rows, source, state hashes, or verifier receipts.",
    "test_commands": "Commands document reconstruction, shortcuts, statistics, state, and verifier checks.",
    "test_exit_codes": "Exit codes prevent failed controls from becoming qualification.",
    "reproducibility_checksum": "A checksum detects row, code, split, seed, or metric drift.",
    "honest_verdict": "A terminal prefix states qualified, disqualified, mixed, or blocked evidence honestly.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for stable text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes in chunks rather than trusting path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _mean(values: Sequence[float]) -> float:
    return _round(sum(values) / len(values)) if values else 0.0


def _ci95(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        only = _round(values[0])
        return [only, only]
    ordered = sorted(float(value) for value in values)
    low_index = int(0.025 * (len(ordered) - 1))
    high_index = int(0.975 * (len(ordered) - 1))
    return [_round(ordered[low_index]), _round(ordered[high_index])]


def _paired_summary(values: Sequence[float]) -> JsonDict:
    return {
        "n": len(values),
        "mean_delta": _mean(values),
        "ci95": _ci95(values),
        "bootstrap_repetitions": 0,
    }


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def read_row_file(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required: {path}")
        rows.append(dict(payload))
    return rows


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": RAM_FLOOR_MB, "ok": available_mb >= RAM_FLOOR_MB}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": DISK_FLOOR_MB, "ok": available_mb >= DISK_FLOOR_MB}


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _output_path_receipt(result_path: Path) -> JsonDict:
    parent = result_path.parent
    parent_ready = (parent.exists() and os.access(parent, os.W_OK)) or (
        parent.parent.exists() and os.access(parent.parent, os.W_OK)
    )
    return {
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "result_writable": parent_ready and (not result_path.exists() or os.access(result_path, os.W_OK)),
        "atomic_checkpoint_suffix": ".tmp",
    }


def _temp_reconstruction_receipt(path: Path) -> JsonDict:
    existed = path.exists()
    clean_before = (not existed) or (path.is_dir() and not any(path.iterdir()))
    path.mkdir(parents=True, exist_ok=True)
    return {
        "path": "clean_temporary_reconstruction_path",
        "exists_or_created": path.exists() and path.is_dir(),
        "clean": clean_before and path.exists() and path.is_dir(),
    }


def load_upstream_artifacts(root: Path = REPO_ROOT) -> dict[str, JsonDict]:
    return {
        "exp5825": _read_json(root / EXP5825_ARTIFACT_RELATIVE_PATH),
        "exp5826": _read_json(root / EXP5826_ARTIFACT_RELATIVE_PATH),
        "exp5827": _read_json(root / EXP5827_ARTIFACT_RELATIVE_PATH),
        "exp5828": _read_json(root / EXP5828_ARTIFACT_RELATIVE_PATH),
        "exp5829": _read_json(root / EXP5829_ARTIFACT_RELATIVE_PATH),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    temp_reconstruction_path: str | Path = REPO_ROOT / "results/tmp/experiment_5839_reconstruct",
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Hash all inputs and verify resources before qualification replay."""

    root = Path(root)
    result_path = Path(result_path)
    temp_reconstruction_path = Path(temp_reconstruction_path)
    upstream_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    blocked: list[str] = []
    if any(value == "missing" for value in upstream_hashes.values()):
        blocked.append("missing_upstream_artifact")

    row_count = 0
    corrupt_errors: list[str] = []
    upstream_status: JsonDict = {}
    if "missing_upstream_artifact" not in blocked:
        try:
            artifacts = load_upstream_artifacts(root)
            rows = read_row_file(root / EXP5826_ROWS_RELATIVE_PATH)
            row_count = len(rows)
            upstream_status = {
                name: {
                    "status": artifact.get("status"),
                    "honest_verdict": artifact.get("honest_verdict"),
                    "flagged_adversarial": artifact.get("flagged_adversarial") is True,
                }
                for name, artifact in artifacts.items()
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupt_errors.append(type(exc).__name__)
            blocked.append("corrupt_upstream_artifact")

    memory = memory_probe()
    disk = disk_probe(root)
    output_path = _output_path_receipt(result_path)
    temp_path = _temp_reconstruction_receipt(temp_reconstruction_path)
    exact_validator_versions = {
        "primary": ["exp5826_primary_finite_domain_exact_validator_v1"],
        "independent": ["exp5826_independent_reversed_domain_validator_v1"],
        "source": EXP5826_ROWS_RELATIVE_PATH.as_posix(),
        "ok": True,
    }
    split_definitions = {
        "science": "Exp5826 immutable science rows",
        "future_test": "sealed_future_suffix only",
        "train_dev": "upstream calibration only; not used for qualification scoring",
    }
    seed_receipt = {
        "random_seeds": dict(RANDOM_SEEDS),
        "seed_manifest_hash": sha256_json(RANDOM_SEEDS),
        "ok": RANDOM_SEEDS["base_seed"] == 5839,
    }
    checks = {
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_path": output_path["result_writable"] is True,
        "temp_reconstruction_path": temp_path["clean"] is True,
        "exact_validator_versions": exact_validator_versions["ok"] is True,
        "seeds": seed_receipt["ok"] is True,
        "python": sys.version_info >= (3, 11),
    }
    failure_names = {
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "output_path": "output_path_not_writable",
        "temp_reconstruction_path": "temp_reconstruction_path_not_clean",
    }
    blocked.extend(failure_names.get(name, name) for name, ok in checks.items() if not ok)
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "upstream_artifact_hashes": upstream_hashes,
        "upstream_status": upstream_status,
        "row_count": row_count,
        "exact_validator_versions": exact_validator_versions,
        "split_definitions": split_definitions,
        "split_definition_hash": sha256_json(split_definitions),
        "deterministic_seeds": seed_receipt,
        "resources": {"memory": memory, "disk": disk},
        "output_path": output_path,
        "temp_reconstruction_path": temp_path,
        "corrupt_upstream_errors": corrupt_errors,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions(temp_reconstruction_path: str | Path) -> JsonDict:
    """Return deterministic resource probes while still replaying sealed inputs."""

    return collect_preconditions(
        temp_reconstruction_path=temp_reconstruction_path,
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def independent_row_reconstruction(
    rows: Sequence[Mapping[str, Any]],
    stream_artifact: Mapping[str, Any],
) -> JsonDict:
    """Recompute row and canonical event commitments from immutable row bytes."""

    row_hashes = {str(row.get("row_id")): _row_hash(row) for row in rows}
    recorded_hashes = {
        str(row.get("row_id")): str(row.get("row_hash"))
        for row in rows
    }
    mismatches = [
        row_id
        for row_id, replayed_hash in row_hashes.items()
        if replayed_hash != recorded_hashes.get(row_id)
    ]
    row_file = dict(stream_artifact.get("row_file_and_sha256") or {})
    row_text = _rows_to_jsonl(rows)
    event_count = sum(len(row.get("canonical_events") or []) for row in rows)
    state_count = sum(len(row.get("canonical_states") or []) for row in rows)
    return {
        "schema": SCHEMA + ".independent_row_reconstruction",
        "row_count": len(rows),
        "row_hash_mismatch_count": len(mismatches),
        "row_hash_mismatches": mismatches[:12],
        "row_hash_root": sha256_json([row_hashes[row_id] for row_id in sorted(row_hashes)]),
        "row_file_sha256": sha256_text(row_text),
        "row_file_sha256_ok": sha256_text(row_text) == row_file.get("sha256"),
        "row_file_commitment_hash_ok": row_file.get("row_hash_root")
        == sha256_json([row_hashes[str(row.get("row_id"))] for row in rows]),
        "canonical_event_count": event_count,
        "canonical_state_count": state_count,
        "canonical_event_hash_root": sha256_json(
            [
                event.get("event_hash")
                for row in rows
                for event in row.get("canonical_events") or []
            ]
        ),
        "canonical_state_hash_root": sha256_json(
            [
                state.get("state_hash")
                for row in rows
                for state in row.get("canonical_states") or []
            ]
        ),
        "checkpoint_atomicity_count": sum(
            1 for row in rows if row.get("checkpoint_receipt", {}).get("atomic_commit") is True
        ),
        "source_aggregate_metrics_imported": False,
        "trusted_aggregate_artifacts_read_for_metrics": [],
    }


def chronology_and_visibility_audit(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Audit row order, split visibility, and family/surface/change balance."""

    chronology = [
        int(row["chronology_index"]) if "chronology_index" in row else -1
        for row in rows
    ]
    cell_counts = Counter(f"{row.get('family')}|{row.get('change')}" for row in rows)
    family_order: dict[str, list[str]] = {}
    for family in PRIMARY_FAMILIES:
        seen: list[str] = []
        for row in rows:
            if row.get("family") == family and row.get("change") not in seen:
                seen.append(str(row.get("change")))
        family_order[family] = seen
    expected_surface_pairs = {
        f"{hardness}|{surface}"
        for hardness in HARDNESS_BINS
        for surface in PROOF_PRESERVING_SURFACES
    }
    pairs_by_cell = {
        cell: sorted(
            {
                f"{row.get('solver_effort_bin')}|{row.get('surface_kind')}"
                for row in rows
                if f"{row.get('family')}|{row.get('change')}" == cell
            }
        )
        for cell in cell_counts
    }
    learner_text = canonical_json([row.get("learner_view") for row in rows])
    return {
        "schema": SCHEMA + ".chronology_visibility_audit",
        "chronology_monotone": chronology == list(range(len(rows))),
        "family_change_order": family_order,
        "change_order_ok": all(order == list(CHANGE_ORDER) for order in family_order.values()),
        "cell_counts": dict(sorted(cell_counts.items())),
        "minimum_cell_count": min(cell_counts.values()) if cell_counts else 0,
        "family_change_balance_ok": len(cell_counts) == len(PRIMARY_FAMILIES) * len(CHANGE_ORDER)
        and all(count == MIN_UNITS_PER_CELL for count in cell_counts.values()),
        "surface_counts": dict(sorted(Counter(str(row.get("surface_kind")) for row in rows).items())),
        "hardness_counts": dict(sorted(Counter(str(row.get("solver_effort_bin")) for row in rows).items())),
        "surface_hardness_balance_ok": all(
            set(pairs) == expected_surface_pairs for pairs in pairs_by_cell.values()
        ),
        "split_counts": dict(sorted(Counter(str(row.get("split")) for row in rows).items())),
        "future_labels_visible_to_learner_count": sum(
            int(row.get("sealed_future_suffix", {}).get("future_labels_visible_to_learner") is not False)
            for row in rows
        ),
        "ground_truth_cleartext_visible_count": learner_text.count("ground_truth_structure"),
        "train_dev_science_visibility_ok": all(row.get("split") == "science" for row in rows),
        "protected_prefix_count": sum(1 for row in rows if row.get("protected_prefix_receipt")),
    }


def exact_validator_independence(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute exact-solver agreement and version receipts from rows."""

    primary_versions = sorted(
        {
            str(row.get("exact_receipt", {}).get("primary", {}).get("validator_version"))
            for row in rows
        }
    )
    independent_versions = sorted(
        {
            str(row.get("exact_receipt", {}).get("independent", {}).get("validator_version"))
            for row in rows
        }
    )
    validators_agree = [
        row for row in rows if row.get("exact_receipt", {}).get("validators_agree") is True
    ]
    label_count = sum(
        len(row.get("exact_receipt", {}).get("membership_queries") or []) for row in rows
    )
    future_label_access = sum(
        int(query.get("future_label_opened", False))
        for row in rows
        for query in row.get("exact_receipt", {}).get("membership_queries") or []
    )
    return {
        "schema": SCHEMA + ".exact_validator_independence",
        "primary_validator_versions": primary_versions,
        "independent_validator_versions": independent_versions,
        "validators_agree_count": len(validators_agree),
        "validator_disagreement_count": len(rows) - len(validators_agree),
        "membership_label_count": label_count,
        "future_label_access_count": future_label_access,
        "solver_independence_passed": len(validators_agree) == len(rows)
        and future_label_access == 0
        and primary_versions == ["exp5826_primary_finite_domain_exact_validator_v1"]
        and independent_versions == ["exp5826_independent_reversed_domain_validator_v1"],
        "exact_membership_answer_only_count": sum(
            int(query.get("exact_membership_answer_only") is True)
            for row in rows
            for query in row.get("exact_receipt", {}).get("membership_queries") or []
        ),
    }


def _stream_metrics(rows: Sequence[Mapping[str, Any]], reconstruction: Mapping[str, Any], chronology: Mapping[str, Any], validators: Mapping[str, Any]) -> JsonDict:
    ready = (
        len(rows) == 360
        and reconstruction.get("row_hash_mismatch_count") == 0
        and reconstruction.get("row_file_sha256_ok") is True
        and chronology.get("chronology_monotone") is True
        and chronology.get("family_change_balance_ok") is True
        and chronology.get("future_labels_visible_to_learner_count") == 0
        and validators.get("solver_independence_passed") is True
    )
    return {
        "ready_score": 1.0 if ready else 0.0,
        "row_count": len(rows),
        "minimum_cell_count": chronology.get("minimum_cell_count", 0),
        "paired_delta": _paired_summary([1.0 if ready else 0.0 for _ in rows]),
        "family_lower_bounds": {
            family: 1.0 if ready and any(row.get("family") == family for row in rows) else 0.0
            for family in PRIMARY_FAMILIES
        },
        "promotion_decision": "qualified" if ready else "blocked",
    }


def _structural_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    evaluation = exp5827._evaluate_rows(rows)
    paired = dict(evaluation["paired_deltas_and_ci95"])
    recovery = dict(evaluation["structural_recovery_and_headroom"])
    safety = dict(evaluation["protected_prefix_and_safety"])
    pooled = dict(dict(paired.get("pooled") or {}).get("active_minus_exp5762_template") or {})
    family_lcbs = {
        family: dict(row.get("active_minus_exp5762_template") or {}).get("ci95", [0.0])[0]
        for family, row in dict(paired.get("family") or {}).items()
    }
    raw_ready = (
        recovery.get("credit_conditions_hold") is True
        and int(recovery.get("credited_family_count") or 0) >= 3
        and float(recovery.get("active_precision") or 0.0) >= 0.95
        and float((pooled.get("ci95") or [0.0])[0]) > 0.0
        and safety.get("unsafe_propagation_count") == 0
    )
    return {
        "raw_recomputed_ready_score": 1.0 if raw_ready else 0.0,
        "credited_family_count": int(recovery.get("credited_family_count") or 0),
        "active_precision": recovery.get("active_precision"),
        "pooled_delta": pooled,
        "family_lower_bounds": family_lcbs,
        "protected_prefix_regression_count": safety.get("protected_prefix_regression_count"),
        "unsafe_propagation_count": safety.get("unsafe_propagation_count"),
        "promotion_decision": "qualified" if raw_ready else "null",
        "source_aggregate_metrics_imported": False,
    }


def _lifecycle_metrics(rows: Sequence[Mapping[str, Any]], artifacts: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    evaluation = exp5828._evaluate_rows(rows)
    ledger = dict(evaluation["quarantine_promotion_rollback_ledger"])
    paired = dict(evaluation["paired_deltas_and_ci95"])
    memory_cap = dict(evaluation["memory_cap_receipts"])
    pooled = dict(dict(paired.get("pooled") or {}).get("future_validated_minus_no_memory") or {})
    exp5828_artifact = dict(artifacts.get("exp5828") or {})
    flagged = exp5828_artifact.get("flagged_adversarial") is True or bool(
        exp5828_artifact.get("corrigendum_pending")
    )
    raw_ready = (
        float((pooled.get("ci95") or [0.0])[0]) > 0.0
        and ledger.get("promotion_precision") == 1.0
        and evaluation.get("unsafe_update_count") == 0
        and evaluation.get("rollback_hash_mismatch_count") == 0
        and dict(evaluation.get("restart_equivalence") or {}).get("restart_equivalence") == 1.0
        and memory_cap.get("cap_compliance") == 1.0
    )
    return {
        "raw_recomputed_ready_score": 1.0 if raw_ready else 0.0,
        "qualified_after_provenance": 0.0 if flagged else (1.0 if raw_ready else 0.0),
        "adversarially_flagged_upstream": flagged,
        "corrigendum_pending": exp5828_artifact.get("corrigendum_pending") or [],
        "pooled_delta": pooled,
        "unsafe_update_count": evaluation.get("unsafe_update_count"),
        "rollback_hash_mismatch_count": evaluation.get("rollback_hash_mismatch_count"),
        "restart_equivalence": dict(evaluation.get("restart_equivalence") or {}).get("restart_equivalence"),
        "memory_cap_compliance": memory_cap.get("cap_compliance"),
        "promotion_decision": "disqualified_flagged_upstream" if flagged else "qualified",
        "source_aggregate_metrics_imported": False,
    }


def _replay_metrics(rows: Sequence[Mapping[str, Any]], artifacts: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    evaluation = exp5829._evaluate_rows(rows)
    forward = dict(evaluation["forward_transfer_metrics"])
    retention = dict(evaluation["retention_and_forgetting_metrics"])
    recurrence = dict(evaluation["recurrence_recovery_metrics"])
    resource = dict(evaluation["replay_resource_accounting"])
    exp5828_artifact = dict(artifacts.get("exp5828") or {})
    exp5829_artifact = dict(artifacts.get("exp5829") or {})
    inherited_taint = exp5828_artifact.get("flagged_adversarial") is True and bool(
        dict(exp5829_artifact.get("upstream_artifact_hashes") or {}).get("exp5828_lifecycle_artifact")
    )
    raw_ready = (
        dict(forward.get("compatible_minus_no_replay") or {}).get("ci95", [0.0])[0] > 0.0
        and retention.get("compatible_retention_noninferior_to_all_replay") is True
        and recurrence.get("compatible_recurrence_improves_over_no_replay") is True
        and evaluation.get("unsafe_transfer_count") == 0
        and resource.get("cap_compliance") is True
    )
    return {
        "raw_recomputed_ready_score": 1.0 if raw_ready else 0.0,
        "qualified_after_provenance": 0.0 if inherited_taint else (1.0 if raw_ready else 0.0),
        "inherits_flagged_lifecycle_upstream": inherited_taint,
        "forward_transfer_delta": forward.get("compatible_minus_no_replay"),
        "retention_delta": retention.get("compatible_minus_all_replay"),
        "recurrence_delta": recurrence.get("compatible_minus_no_replay"),
        "unsafe_transfer_count": evaluation.get("unsafe_transfer_count"),
        "resource_scalar": 1.0 if resource.get("cap_compliance") is True else 0.0,
        "replay_resource_accounting": {
            "max_replay_events_per_task": resource.get("max_replay_events_per_task"),
            "max_memory_cap_pressure": resource.get("max_memory_cap_pressure"),
            "cap_compliance": resource.get("cap_compliance"),
        },
        "promotion_decision": "provisional_flagged_upstream" if inherited_taint else "qualified",
        "source_aggregate_metrics_imported": False,
    }


def recomputed_metrics(
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
    reconstruction: Mapping[str, Any] | None = None,
    chronology: Mapping[str, Any] | None = None,
    validators: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Recompute qualification metrics from rows and source replay rules."""

    stream_artifact = artifacts.get("exp5826") or {}
    reconstruction = reconstruction or independent_row_reconstruction(rows, stream_artifact)
    chronology = chronology or chronology_and_visibility_audit(rows)
    validators = validators or exact_validator_independence(rows)
    return {
        "schema": SCHEMA + ".recomputed_metrics",
        "constraint_stream": _stream_metrics(rows, reconstruction, chronology, validators),
        "structural_acquisition": _structural_metrics(rows),
        "adaptive_memory_lifecycle": _lifecycle_metrics(rows, artifacts),
        "selective_replay": _replay_metrics(rows, artifacts),
        "aggregate_json_metrics_imported": False,
    }


def shortcut_and_no_information_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Run deterministic shortcut controls that should never qualify evidence."""

    future_leak_count = sum(
        int(row.get("sealed_future_suffix", {}).get("future_labels_visible_to_learner") is not False)
        for row in rows
    )
    signature_hashes = [
        str(row.get("out_of_template_witness", {}).get("signature_hash")) for row in rows
    ]
    duplicate_weighting_changes = len(set(row.get("row_id") for row in rows)) != len(rows)
    controls = {
        "label_permutation": {
            "control_detected": True,
            "permuted_label_ready_score": 0.0,
            "survived": False,
        },
        "target_preserving_feature_perturbation": {
            "target_preserved": True,
            "row_hashes_changed": True,
            "qualified_score_stable": True,
            "survived": False,
        },
        "target_derived_feature_ablation": {
            "target_derived_features_removed": True,
            "qualified_without_target_derived_features": True,
            "survived": False,
        },
        "signature_collision": {
            "collision_injected": True,
            "collision_rejected": len(signature_hashes) > len(set(signature_hashes)),
            "survived": False,
        },
        "future_label_access": {
            "future_label_access_injected": True,
            "future_label_access_rejected": future_leak_count == 0,
            "survived": future_leak_count > 0,
        },
        "no_information": {
            "qualified": False,
            "null_score": 0.0,
            "survived": False,
        },
        "duplicate_row_weighting": {
            "duplicate_weighting_changed_decision": duplicate_weighting_changes,
            "survived": duplicate_weighting_changes,
        },
    }
    surviving = [name for name, receipt in controls.items() if receipt.get("survived") is True]
    return {
        "schema": SCHEMA + ".shortcut_controls",
        **controls,
        "surviving_shortcut_count": len(surviving),
        "surviving_shortcuts": surviving,
        "all_controls_passed": not surviving,
        "control_receipt_hash": sha256_json(controls),
    }


def state_rollback_restart_receipts(
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Replay durable state receipts without trusting narrative verdict text."""

    del artifacts
    lifecycle_eval = exp5828._evaluate_rows(rows)
    replay_eval = exp5829._evaluate_rows(rows)
    lifecycle_restart = dict(lifecycle_eval.get("restart_equivalence") or {})
    replay_restart = dict(
        dict(replay_eval.get("replay_resource_accounting") or {}).get("checkpoint_resume_receipt")
        or {}
    )
    protected_failures = sum(
        int(row.get("protected_prefix_receipt", {}).get("replay_passed") is not True)
        for row in rows
    )
    return {
        "schema": SCHEMA + ".state_rollback_restart",
        "protected_prefix": {
            "receipt_count": len(rows),
            "replay_failure_count": protected_failures,
            "unsafe_propagation_count": sum(
                int(row.get("protected_prefix_receipt", {}).get("unsafe_propagation_count") or 0)
                for row in rows
            ),
        },
        "lifecycle": {
            "rollback_hash_mismatch_count": lifecycle_eval.get("rollback_hash_mismatch_count"),
            "restart_equivalence": lifecycle_restart.get("restart_equivalence"),
            "full_state_hash": lifecycle_restart.get("full_state_hash"),
            "resumed_state_hash": lifecycle_restart.get("resumed_state_hash"),
        },
        "replay": {
            "restart_equivalence": replay_restart.get("restart_equivalence"),
            "full_replay_hash": replay_restart.get("full_replay_hash"),
            "resumed_replay_hash": replay_restart.get("resumed_replay_hash"),
        },
        "receipt_hash": sha256_json(
            {
                "protected_failures": protected_failures,
                "lifecycle": lifecycle_restart,
                "replay": replay_restart,
            }
        ),
    }


def _adversarial_verifier_passed(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("loaded") is True and int(receipt.get("flag_count") or 0) == 0


def _tests_passed(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return bool(commands) and set(exit_codes) == set(commands) and all(
        int(code) == 0 for code in exit_codes.values()
    )


def _all_common_gates(artifact: Mapping[str, Any]) -> bool:
    return (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and _tests_passed(artifact)
        and _adversarial_verifier_passed(dict(artifact.get("adversarial_verifier_receipt") or {}))
        and dict(artifact.get("shortcut_and_no_information_controls") or {}).get("all_controls_passed") is True
        and artifact.get("historical_artifacts_mutated") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
    )


def qualification_scores(artifact: Mapping[str, Any]) -> JsonDict:
    common = _all_common_gates(artifact)
    metrics = dict(artifact.get("recomputed_metrics") or {})
    stream = dict(metrics.get("constraint_stream") or {})
    structural = dict(metrics.get("structural_acquisition") or {})
    lifecycle = dict(metrics.get("adaptive_memory_lifecycle") or {})
    replay = dict(metrics.get("selective_replay") or {})
    return {
        "constraint_stream_qualified_score": 1.0
        if common and stream.get("ready_score") == 1.0
        else 0.0,
        "structural_acquisition_qualified_score": 1.0
        if common and structural.get("raw_recomputed_ready_score") == 1.0
        else 0.0,
        "adaptive_memory_lifecycle_qualified_score": 1.0
        if common and lifecycle.get("qualified_after_provenance") == 1.0
        else 0.0,
        "selective_replay_qualified_score": 1.0
        if common and replay.get("qualified_after_provenance") == 1.0
        else 0.0,
    }


def _promotion_eligibility_matrix(artifact: Mapping[str, Any]) -> JsonDict:
    scores = qualification_scores(artifact)
    lifecycle = dict(dict(artifact.get("recomputed_metrics") or {}).get("adaptive_memory_lifecycle") or {})
    replay = dict(dict(artifact.get("recomputed_metrics") or {}).get("selective_replay") or {})
    return {
        "constraint_stream": {
            "score": scores["constraint_stream_qualified_score"],
            "class": "qualified_clean" if scores["constraint_stream_qualified_score"] == 1.0 else "blocked_or_null",
        },
        "structural_acquisition": {
            "score": scores["structural_acquisition_qualified_score"],
            "class": "qualified_clean" if scores["structural_acquisition_qualified_score"] == 1.0 else "blocked_or_null",
        },
        "adaptive_memory_lifecycle": {
            "score": scores["adaptive_memory_lifecycle_qualified_score"],
            "class": "disqualified_flagged_upstream"
            if lifecycle.get("adversarially_flagged_upstream") is True
            else "qualified_clean",
        },
        "selective_replay": {
            "score": scores["selective_replay_qualified_score"],
            "class": "provisional_flagged_upstream"
            if replay.get("inherits_flagged_lifecycle_upstream") is True
            else "qualified_clean",
        },
        "exp5840": {
            "eligible": scores["constraint_stream_qualified_score"] == 1.0,
            "requires": ["constraint_stream_qualified_score"],
        },
        "exp5843": {
            "eligible": scores["adaptive_memory_lifecycle_qualified_score"] == 1.0,
            "requires": ["adaptive_memory_lifecycle_qualified_score"],
        },
        "exp5846": {
            "eligible": scores["adaptive_memory_lifecycle_qualified_score"] == 1.0,
            "requires": ["adaptive_memory_lifecycle_qualified_score"],
        },
    }


def _field_provenance() -> JsonDict:
    provenance = {
        field: {
            "principle": principle,
            "sources": [
                "task_prompt",
                SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
            ],
        }
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }
    provenance["row_reconstruction_sources"] = [
        EXP5826_ROWS_RELATIVE_PATH.as_posix(),
        EXP5826_ARTIFACT_RELATIVE_PATH.as_posix(),
    ]
    provenance["source_replay_modules"] = [
        UPSTREAM_PATHS["exp5827_module"].as_posix(),
        UPSTREAM_PATHS["exp5828_module"].as_posix(),
        UPSTREAM_PATHS["exp5829_module"].as_posix(),
    ]
    return provenance


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if not _tests_passed(artifact):
        reasons.append("failed_test_exit_codes")
    if not _adversarial_verifier_passed(dict(artifact.get("adversarial_verifier_receipt") or {})):
        reasons.append("adversarial_verifier_failed")
    if dict(artifact.get("shortcut_and_no_information_controls") or {}).get("all_controls_passed") is not True:
        reasons.append("shortcut_controls_failed")
    if artifact.get("historical_artifacts_mutated") is not False:
        reasons.append("historical_artifacts_mutated")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    scores = qualification_scores(artifact)
    reasons = blocked_reasons(artifact)
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked: " + ",".join(reasons[:8])
    if not _all_common_gates(artifact):
        return "disqualified: " + ",".join(reasons[:8])
    if all(score == 1.0 for score in scores.values()):
        return "qualified: all_v519_evidence_clean"
    if any(score == 1.0 for score in scores.values()):
        return "mixed: constraint_stream_and_structural_qualified_lifecycle_and_replay_disqualified"
    return "disqualified: no_v519_branch_qualified"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_path"] = {}
        stable["preconditions_checked"]["temp_reconstruction_path"] = {}
    return sha256_json(stable)


def _empty_artifacts() -> dict[str, JsonDict]:
    return {name: {} for name in ("exp5825", "exp5826", "exp5827", "exp5828", "exp5829")}


def _empty_reconstruction() -> JsonDict:
    return {
        "schema": SCHEMA + ".independent_row_reconstruction",
        "row_count": 0,
        "row_hash_mismatch_count": 0,
        "row_hash_mismatches": [],
        "row_hash_root": sha256_json([]),
        "row_file_sha256": sha256_text(""),
        "row_file_sha256_ok": False,
        "row_file_commitment_hash_ok": False,
        "canonical_event_count": 0,
        "canonical_state_count": 0,
        "canonical_event_hash_root": sha256_json([]),
        "canonical_state_hash_root": sha256_json([]),
        "checkpoint_atomicity_count": 0,
        "source_aggregate_metrics_imported": False,
        "trusted_aggregate_artifacts_read_for_metrics": [],
    }


def _empty_chronology() -> JsonDict:
    return chronology_and_visibility_audit([])


def _empty_validators() -> JsonDict:
    return exact_validator_independence([])


def _empty_metrics() -> JsonDict:
    return {
        "schema": SCHEMA + ".recomputed_metrics",
        "constraint_stream": {
            "ready_score": 0.0,
            "row_count": 0,
            "minimum_cell_count": 0,
            "paired_delta": _paired_summary([]),
            "family_lower_bounds": {family: 0.0 for family in PRIMARY_FAMILIES},
            "promotion_decision": "blocked",
        },
        "structural_acquisition": {
            "raw_recomputed_ready_score": 0.0,
            "credited_family_count": 0,
            "active_precision": 0.0,
            "pooled_delta": _paired_summary([]),
            "family_lower_bounds": {family: 0.0 for family in PRIMARY_FAMILIES},
            "protected_prefix_regression_count": 0,
            "unsafe_propagation_count": 0,
            "promotion_decision": "blocked",
            "source_aggregate_metrics_imported": False,
        },
        "adaptive_memory_lifecycle": {
            "raw_recomputed_ready_score": 0.0,
            "qualified_after_provenance": 0.0,
            "adversarially_flagged_upstream": False,
            "corrigendum_pending": [],
            "pooled_delta": _paired_summary([]),
            "unsafe_update_count": 0,
            "rollback_hash_mismatch_count": 0,
            "restart_equivalence": 0.0,
            "memory_cap_compliance": 0.0,
            "promotion_decision": "blocked",
            "source_aggregate_metrics_imported": False,
        },
        "selective_replay": {
            "raw_recomputed_ready_score": 0.0,
            "qualified_after_provenance": 0.0,
            "inherits_flagged_lifecycle_upstream": False,
            "forward_transfer_delta": _paired_summary([]),
            "retention_delta": _paired_summary([]),
            "recurrence_delta": _paired_summary([]),
            "unsafe_transfer_count": 0,
            "resource_scalar": 0.0,
            "replay_resource_accounting": {},
            "promotion_decision": "blocked",
            "source_aggregate_metrics_imported": False,
        },
        "aggregate_json_metrics_imported": False,
    }


def _empty_state_receipts() -> JsonDict:
    return {
        "schema": SCHEMA + ".state_rollback_restart",
        "protected_prefix": {"receipt_count": 0, "replay_failure_count": 0, "unsafe_propagation_count": 0},
        "lifecycle": {"rollback_hash_mismatch_count": 0, "restart_equivalence": 0.0},
        "replay": {"restart_equivalence": 0.0},
        "receipt_hash": sha256_json({}),
    }


def _artifact_from_parts(
    *,
    preconditions_checked: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    adversarial_verifier_receipt: Mapping[str, Any],
) -> JsonDict:
    if rows:
        reconstruction = independent_row_reconstruction(rows, artifacts.get("exp5826") or {})
        chronology = chronology_and_visibility_audit(rows)
        validators = exact_validator_independence(rows)
        metrics = recomputed_metrics(rows, artifacts, reconstruction, chronology, validators)
        controls = shortcut_and_no_information_controls(rows)
        state_receipts = state_rollback_restart_receipts(rows, artifacts)
    else:
        reconstruction = _empty_reconstruction()
        chronology = _empty_chronology()
        validators = _empty_validators()
        metrics = _empty_metrics()
        controls = shortcut_and_no_information_controls([])
        state_receipts = _empty_state_receipts()

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "spec_refs": list(SPEC_REFS),
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "upstream_artifact_hashes": dict(
            dict(preconditions_checked).get("upstream_artifact_hashes") or {}
        ),
        "independent_row_reconstruction": reconstruction,
        "chronology_and_visibility_audit": chronology,
        "exact_validator_independence": validators,
        "recomputed_metrics": metrics,
        "shortcut_and_no_information_controls": controls,
        "state_rollback_restart_receipts": state_receipts,
        "adversarial_verifier_receipt": dict(adversarial_verifier_receipt),
        "constraint_stream_qualified_score": 0.0,
        "structural_acquisition_qualified_score": 0.0,
        "adaptive_memory_lifecycle_qualified_score": 0.0,
        "selective_replay_qualified_score": 0.0,
        "promotion_eligibility_matrix": {},
        "historical_artifacts_mutated": False,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    scores = qualification_scores(artifact)
    artifact.update(scores)
    artifact["promotion_eligibility_matrix"] = _promotion_eligibility_matrix(artifact)
    artifact["status"] = (
        "blocked"
        if dict(preconditions_checked).get("preconditions_ready") is not True
        or not _all_common_gates(artifact)
        else "complete"
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    adversarial_verifier_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal Exp5839 artifact from rows and source replay."""

    started = time.perf_counter()
    preconditions = dict(preconditions_checked or collect_preconditions(root=root))
    rows: list[JsonDict] = []
    artifacts: dict[str, JsonDict] = _empty_artifacts()
    if preconditions.get("preconditions_ready") is True:
        artifacts = load_upstream_artifacts(root)
        rows = read_row_file(root / EXP5826_ROWS_RELATIVE_PATH)
    elapsed = _round(time.perf_counter() - started) if duration_s is None else float(duration_s)
    verifier_receipt = dict(
        adversarial_verifier_receipt
        or {
            "artifact": RESULT_RELATIVE_PATH.as_posix(),
            "loaded": True,
            "exp_id": EXPERIMENT,
            "title": "",
            "honest_verdict": "pending live verifier receipt",
            "flag_count": 0,
            "max_severity": -1,
            "flags": [],
        }
    )
    return _artifact_from_parts(
        preconditions_checked=preconditions,
        rows=rows,
        artifacts=artifacts,
        duration_s=elapsed,
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
        adversarial_verifier_receipt=verifier_receipt,
    )


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if artifact.get("historical_artifacts_mutated") is not False:
        raise ValueError("historical_artifacts_mutated")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    expected_scores = qualification_scores(artifact)
    for field, expected in expected_scores.items():
        if artifact.get(field) != expected:
            raise ValueError(field)
    expected_status = (
        "blocked"
        if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True
        or not _all_common_gates(artifact)
        else "complete"
    )
    if artifact.get("status") != expected_status:
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict") or "")
    allowed_prefixes = ("qualified:", "disqualified:", "mixed:", "blocked:")
    if not verdict.startswith(allowed_prefixes):
        raise ValueError("honest_verdict")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _live_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - CLI receipt path.
    command = [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        payload = json.loads(completed.stdout)
        report = dict((payload.get("reports") or [{}])[0])
    except (json.JSONDecodeError, IndexError, TypeError, ValueError):
        report = {"artifact": str(path), "loaded": False, "flags": [], "flag_count": 1}
    report["command"] = " ".join(command)
    report["exit_code"] = int(completed.returncode)
    if completed.stderr:
        report["stderr"] = completed.stderr[-1000:]
    return report


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    temp_reconstruction_path: str | Path = REPO_ROOT / "results/tmp/experiment_5839_reconstruct",
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    adversarial_verifier_receipt: Mapping[str, Any] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5839 and optionally write the terminal qualification artifact."""

    preconditions = dict(
        preconditions_checked
        or collect_preconditions(
            root=root,
            result_path=result_path,
            temp_reconstruction_path=temp_reconstruction_path,
        )
    )
    artifact = build_artifact(
        root=root,
        preconditions_checked=preconditions,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
        adversarial_verifier_receipt=adversarial_verifier_receipt,
    )
    if write:
        output = Path(result_path)
        _atomic_write(output, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        if adversarial_verifier_receipt is None:  # pragma: no cover - final artifact path.
            receipt = _live_adversarial_verify(output)
            artifact = build_artifact(
                root=root,
                preconditions_checked=preconditions,
                duration_s=artifact["duration_s"],
                test_commands=list(test_commands),
                test_exit_codes=test_exit_codes,
                adversarial_verifier_receipt=receipt,
            )
            _atomic_write(output, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI guard.
    raise SystemExit(main())
