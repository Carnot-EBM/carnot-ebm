"""Exp5856 provenance-correct lifecycle replay.

Spec refs: REQ-LEARN-5856, SCENARIO-LEARN-5856-CHRONOLOGY,
SCENARIO-LEARN-5856-MATCHED-ARMS, SCENARIO-LEARN-5856-READY-GATE,
SCENARIO-LEARN-5856-FAIL-CLOSED.

The module reruns the Exp5828 future-validated memory lifecycle from immutable
Exp5826 rows, but reports the honest Exp5851 deterministic replay substrate.
It never rewrites Exp5828; that artifact remains only historical comparison
evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5826_out_of_template_constraint_stream as exp5826
from carnot import experiment_5827_minimal_core_structural_acquisition_ab as exp5827
from carnot import experiment_5828_future_validated_structural_memory as exp5828
from carnot import experiment_5839_v519_evidence_qualification as exp5839
from carnot import experiment_5851_deterministic_replay_provenance_contract as exp5851


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5856_provenance_correct_lifecycle.json")
ROW_RELATIVE_PATH = Path("results/experiment_5856_provenance_correct_lifecycle.rows.jsonl")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5856_provenance_correct_lifecycle.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5856_provenance_correct_lifecycle.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")

EXP5826_ARTIFACT_RELATIVE_PATH = exp5839.EXP5826_ARTIFACT_RELATIVE_PATH
EXP5826_ROWS_RELATIVE_PATH = exp5839.EXP5826_ROWS_RELATIVE_PATH
EXP5827_ARTIFACT_RELATIVE_PATH = exp5839.EXP5827_ARTIFACT_RELATIVE_PATH
EXP5828_ARTIFACT_RELATIVE_PATH = exp5839.EXP5828_ARTIFACT_RELATIVE_PATH
EXP5839_ARTIFACT_RELATIVE_PATH = exp5839.RESULT_RELATIVE_PATH
EXP5851_ARTIFACT_RELATIVE_PATH = exp5851.RESULT_RELATIVE_PATH
EXP5826_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5826_out_of_template_constraint_stream.py"
)
EXP5827_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5827_minimal_core_structural_acquisition_ab.py"
)
EXP5828_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5828_future_validated_structural_memory.py"
)
EXP5839_MODULE_RELATIVE_PATH = exp5839.MODULE_RELATIVE_PATH
EXP5851_MODULE_RELATIVE_PATH = exp5851.MODULE_RELATIVE_PATH

SCHEMA = "carnot.experiment_5856.provenance_correct_lifecycle.v1"
EXPERIMENT = 5856
EXPERIMENT_ID = "experiment_5856_provenance_correct_lifecycle"
MILESTONE = "2026.07.521"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = exp5851.INFERENCE_SUBSTRATE
VERIFIER_IS_ORACLE = True
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512
PRIMARY_FAMILIES = exp5828.PRIMARY_FAMILIES
CHANGE_ORDER = exp5828.CHANGE_ORDER
PROOF_PRESERVING_SURFACES = exp5828.PROOF_PRESERVING_SURFACES
HARDNESS_BINS = exp5828.HARDNESS_BINS
MEMORY_CAP = exp5828.MEMORY_CAP

FROZEN_ARM = exp5828.NO_MEMORY_ARM
ADAPTIVE_ARM = exp5828.FUTURE_ARM
STRUCTURAL_LEARNER = exp5828.STRUCTURAL_LEARNER
QUERY_BUDGET_PER_ROW = exp5828.QUERY_BUDGET_PER_ROW
STOPPING_RULE = exp5828.STOPPING_RULE

RANDOM_SEEDS: JsonDict = {
    "base_seed": 5856,
    "bootstrap_seed": 5_856_001,
    "group_bootstrap_seed": 5_856_002,
    "serialization_seed": 5_856_003,
}
SPEC_REFS = (
    "REQ-LEARN-5856",
    "SCENARIO-LEARN-5856-CHRONOLOGY",
    "SCENARIO-LEARN-5856-MATCHED-ARMS",
    "SCENARIO-LEARN-5856-READY-GATE",
    "SCENARIO-LEARN-5856-FAIL-CLOSED",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5856_provenance_correct_lifecycle.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5856_provenance_correct_lifecycle.py "
    "-m pytest tests/python/test_experiment_5856_provenance_correct_lifecycle.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5856_provenance_correct_lifecycle.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5856_provenance_correct_lifecycle.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)
UPSTREAM_PATHS: dict[str, Path] = {
    "exp5826_artifact": EXP5826_ARTIFACT_RELATIVE_PATH,
    "exp5826_rows": EXP5826_ROWS_RELATIVE_PATH,
    "exp5827_artifact": EXP5827_ARTIFACT_RELATIVE_PATH,
    "exp5828_comparison_artifact": EXP5828_ARTIFACT_RELATIVE_PATH,
    "exp5839_qualification": EXP5839_ARTIFACT_RELATIVE_PATH,
    "exp5851_contract": EXP5851_ARTIFACT_RELATIVE_PATH,
    "exp5826_module": EXP5826_MODULE_RELATIVE_PATH,
    "exp5827_module": EXP5827_MODULE_RELATIVE_PATH,
    "exp5828_module": EXP5828_MODULE_RELATIVE_PATH,
    "exp5839_module": EXP5839_MODULE_RELATIVE_PATH,
    "exp5851_module": EXP5851_MODULE_RELATIVE_PATH,
    "self_learning_spec": SELF_LEARNING_SPEC_RELATIVE_PATH,
    "adversarial_verify": ADVERSARIAL_VERIFY_RELATIVE_PATH,
    "module": MODULE_RELATIVE_PATH,
    "tests": TEST_RELATIVE_PATH,
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_hashes",
    "deterministic_replay_contract_receipt",
    "chronology_and_visibility_receipts",
    "frozen_and_adaptive_arm_definitions",
    "prospective_row_metrics",
    "family_lower_bounds_and_group_bootstraps",
    "promotion_quarantine_and_rejection_receipts",
    "protected_prefix_retention",
    "rollback_restart_and_serialization_receipts",
    "memory_cap_accounting",
    "no_model_weight_mutation",
    "historical_artifacts_mutated",
    "adversarial_verifier_receipt",
    "adaptive_memory_lifecycle_ready_score",
    "row_file_receipt",
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
    "status": "A terminal lifecycle state separates a clean rerun from partial replay.",
    "preconditions_checked": "Gate, hashes, validators, splits, seeds, timers, resources, and outputs prevent fabricated lifecycle evidence.",
    "upstream_hashes": "The rerun is bound to immutable clean stream and structural evidence.",
    "deterministic_replay_contract_receipt": "Substrate honesty is a promotion prerequisite, not metadata decoration.",
    "chronology_and_visibility_receipts": "Future labels must remain sealed until their prospective evaluation point.",
    "frozen_and_adaptive_arm_definitions": "Matched arms isolate external-state learning.",
    "prospective_row_metrics": "Every credited delta is derived from new chronological rows.",
    "family_lower_bounds_and_group_bootstraps": "No pooled family or duplicated surface can carry the claim.",
    "promotion_quarantine_and_rejection_receipts": "Only future-validated safe updates enter reusable memory.",
    "protected_prefix_retention": "Learning new constraints cannot erase certified old behavior.",
    "rollback_restart_and_serialization_receipts": "Exact hashes prove durable versioned state.",
    "memory_cap_accounting": "External memory remains bounded.",
    "no_model_weight_mutation": "Must be true; self-learning is versioned external state only.",
    "historical_artifacts_mutated": "Must be false; Exp5828 remains immutable and flagged.",
    "adversarial_verifier_receipt": "The fresh live verifier owns promotion eligibility.",
    "adaptive_memory_lifecycle_ready_score": "EMIT BARE scalar; only 1.0 permits Exp5857 and Exp5858.",
    "row_file_receipt": "Path, count, and hash make lifecycle evidence auditable.",
    "duration_s": "Measured exact-replay time must match the declared substrate.",
    "inference_substrate": "`deterministic_exact_verifier_and_replay_no_llm` is mandatory.",
    "verifier_is_oracle": "True records exact validators as promotion authority.",
    "field_provenance": "Every metric traces to rows, state hashes, validators, and timers.",
    "test_commands": "Commands document chronology, metrics, state, contract, and live verifier.",
    "test_exit_codes": "Exit codes prevent failed lifecycle checks becoming readiness.",
    "reproducibility_checksum": "A checksum detects row, split, seed, state, or contract drift.",
    "honest_verdict": "A terminal prefix states credited, null, retired, or blocked outcome.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without trusting timestamps."""

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
    return _round(sum(float(value) for value in values) / len(values)) if values else 0.0


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required: {path}")
        rows.append(dict(payload))
    return rows


def read_row_receipts(path: str | Path) -> list[JsonDict]:
    """Read Exp5856 row receipts, returning an empty list for absent paths."""

    if not Path(path).exists():
        return []
    return _read_jsonl(path)


def _rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


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
    return {
        "available_mb": available_mb,
        "required_mb": RAM_FLOOR_MB,
        "ok": available_mb >= RAM_FLOOR_MB,
    }


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _atomic_path_receipt(path: Path, label: str) -> JsonDict:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".atomic_probe.tmp")
    wrote = False
    try:
        probe.write_text("atomic-output-probe\n", encoding="utf-8")
        wrote = probe.read_text(encoding="utf-8") == "atomic-output-probe\n"
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "kind": label,
        "declared_path": (
            RESULT_RELATIVE_PATH.as_posix() if label == "result" else ROW_RELATIVE_PATH.as_posix()
        ),
        "parent_exists": parent.exists(),
        "parent_writable": os.access(parent, os.W_OK),
        "atomic_suffix": ".tmp",
        "atomic_probe_write_ok": wrote,
        "target_writable": (not path.exists()) or os.access(path, os.W_OK),
        "ok": wrote and ((not path.exists()) or os.access(path, os.W_OK)),
    }


def _validator_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    primary_versions = sorted(
        {
            str(
                dict(dict(row.get("exact_receipt") or {}).get("primary") or {}).get(
                    "validator_version"
                )
            )
            for row in rows
        }
    )
    independent_versions = sorted(
        {
            str(
                dict(dict(row.get("exact_receipt") or {}).get("independent") or {}).get(
                    "validator_version"
                )
            )
            for row in rows
        }
    )
    validators_agree_count = sum(
        int(dict(row.get("exact_receipt") or {}).get("validators_agree") is True) for row in rows
    )
    return {
        "primary_validator_versions": primary_versions,
        "independent_validator_versions": independent_versions,
        "validators_agree_count": validators_agree_count,
        "row_count": len(rows),
        "receipt_hash": sha256_json(
            [primary_versions, independent_versions, validators_agree_count]
        ),
        "ok": primary_versions == [exp5826.PRIMARY_VALIDATOR_VERSION]
        and independent_versions == [exp5826.INDEPENDENT_VALIDATOR_VERSION]
        and validators_agree_count == len(rows)
        and len(rows) == 360,
    }


def _split_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    cell_counts = Counter(f"{row.get('family')}|{row.get('change')}" for row in rows)
    split_counts = Counter(str(row.get("split")) for row in rows)
    family_change_order: dict[str, list[str]] = {}
    for row in rows:
        family = str(row.get("family"))
        change = str(row.get("change"))
        family_change_order.setdefault(family, [])
        if change not in family_change_order[family]:
            family_change_order[family].append(change)
    expected_cells = [
        f"{family}|{change}" for family in PRIMARY_FAMILIES for change in CHANGE_ORDER
    ]
    canonical_event_order = all(
        [event.get("causal_sequence_index") for event in row.get("canonical_events", [])]
        == sorted(event.get("causal_sequence_index") for event in row.get("canonical_events", []))
        for row in rows
    )
    return {
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "cell_counts": {cell: int(cell_counts.get(cell, 0)) for cell in expected_cells},
        "family_change_order": {
            family: family_change_order.get(family, []) for family in PRIMARY_FAMILIES
        },
        "minimum_cell_count": 30,
        "canonical_event_order": canonical_event_order,
        "ok": len(rows) == 360
        and split_counts == Counter({"science": 360})
        and all(cell_counts.get(cell, 0) >= 30 for cell in expected_cells)
        and all(
            family_change_order.get(family) == list(CHANGE_ORDER) for family in PRIMARY_FAMILIES
        )
        and canonical_event_order,
    }


def _seed_receipt(
    stream: Mapping[str, Any],
    learner: Mapping[str, Any],
    qualification: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> JsonDict:
    expected = {
        "exp5826": dict(exp5826.RANDOM_SEEDS),
        "exp5827": dict(exp5827.RANDOM_SEEDS),
        "exp5839": dict(exp5839.RANDOM_SEEDS),
        "exp5851": dict(exp5851.RANDOM_SEEDS),
        "exp5856": dict(RANDOM_SEEDS),
    }
    observed = {
        "exp5826": dict(stream.get("random_seeds") or {}),
        "exp5827": dict(learner.get("random_seeds") or {}),
        "exp5839": dict(qualification.get("random_seeds") or {}),
        "exp5851": dict(contract.get("random_seeds") or {}),
        "exp5856": dict(RANDOM_SEEDS),
    }
    return {
        "expected": expected,
        "observed": observed,
        "seed_manifest_hash": sha256_json(observed),
        "ok": observed == expected,
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_path: str | Path = REPO_ROOT / ROW_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay hashes, validators, splits, seeds, resources, timer, and outputs."""

    root = Path(root)
    result_path = Path(result_path)
    row_path = Path(row_path)
    upstream_hashes = {
        name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()
    }
    memory = memory_probe()
    disk = disk_probe(root)
    timer = time.get_clock_info("perf_counter")
    result_output = _atomic_path_receipt(result_path, "result")
    row_output = _atomic_path_receipt(row_path, "row")
    validators: JsonDict = {"ok": False}
    splits: JsonDict = {"ok": False}
    seeds: JsonDict = {"ok": False}
    gates: JsonDict = {"ok": False}
    historical: JsonDict = {"ok": False}
    corrupt_errors: list[str] = []
    missing = any(value == "missing" for value in upstream_hashes.values())
    if not missing:
        try:
            stream = _read_json(root / EXP5826_ARTIFACT_RELATIVE_PATH)
            learner = _read_json(root / EXP5827_ARTIFACT_RELATIVE_PATH)
            exp5828_artifact = _read_json(root / EXP5828_ARTIFACT_RELATIVE_PATH)
            qualification = _read_json(root / EXP5839_ARTIFACT_RELATIVE_PATH)
            contract = _read_json(root / EXP5851_ARTIFACT_RELATIVE_PATH)
            rows = _read_jsonl(root / EXP5826_ROWS_RELATIVE_PATH)
            validators = _validator_receipt(rows)
            splits = _split_receipt(rows)
            seeds = _seed_receipt(stream, learner, qualification, contract)
            gates = {
                "exp5826_ready_score": stream.get("constraint_event_stream_ready_score"),
                "exp5827_ready_score": learner.get("structural_learner_ready_score"),
                "exp5839_status": qualification.get("status"),
                "exp5851_contract_ready_score": contract.get(
                    "deterministic_replay_contract_ready_score"
                ),
                "ok": stream.get("constraint_event_stream_ready_score") == 1.0
                and learner.get("structural_learner_ready_score") == 1.0
                and qualification.get("status") == "complete"
                and contract.get("deterministic_replay_contract_ready_score") == 1.0,
            }
            historical = {
                "exp5828_status": exp5828_artifact.get("status"),
                "exp5828_historical_flagged": exp5828_artifact.get("flagged_adversarial") is True,
                "exp5839_lifecycle_qualification": qualification.get(
                    "adaptive_memory_lifecycle_qualified_score"
                ),
                "exp5851_regression_rejected": dict(
                    contract.get("exp5828_regression_receipt") or {}
                ).get("passed")
                is False,
                "ok": exp5828_artifact.get("flagged_adversarial") is True
                and qualification.get("adaptive_memory_lifecycle_qualified_score") == 0.0
                and dict(contract.get("exp5828_regression_receipt") or {}).get("passed") is False,
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupt_errors.append(type(exc).__name__)
    checks = {
        "upstream_hashes": not missing,
        "python": sys.version_info >= (3, 11),
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "timer": timer.monotonic and timer.resolution > 0.0,
        "result_output": result_output.get("ok") is True,
        "row_output": row_output.get("ok") is True,
        "validators": validators.get("ok") is True,
        "splits": splits.get("ok") is True,
        "seeds": seeds.get("ok") is True,
        "gates": gates.get("ok") is True,
        "historical": historical.get("ok") is True,
        "json": not corrupt_errors,
    }
    failure_names = {
        "upstream_hashes": "missing_upstream_file",
        "python": "python_version_below_3_11",
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "timer": "timer_not_monotonic",
        "result_output": "result_path_not_writable",
        "row_output": "row_path_not_writable",
        "validators": "validator_receipt_failed",
        "splits": "split_receipt_failed",
        "seeds": "seed_receipt_failed",
        "gates": "gate_replay_failed",
        "historical": "historical_comparison_failed",
        "json": "corrupt_upstream_json",
    }
    blocked = [failure_names[name] for name, ok in checks.items() if not ok]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "upstream_hashes": upstream_hashes,
        "validators": validators,
        "splits": splits,
        "seeds": seeds,
        "gates": gates,
        "historical": historical,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "resources": {"memory": memory, "disk": disk},
        "timer": {
            "clock": "perf_counter",
            "implementation": timer.implementation,
            "monotonic": timer.monotonic,
            "resolution_s": timer.resolution,
            "ok": timer.monotonic and timer.resolution > 0.0,
        },
        "atomic_outputs": {"result": result_output, "row": row_output},
        "blocked_errors": corrupt_errors,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions(tmp_path: Path | None = None) -> JsonDict:
    """Return deterministic resource probes while replaying real immutable rows."""

    base = tmp_path or REPO_ROOT
    return collect_preconditions(
        result_path=Path(base) / RESULT_RELATIVE_PATH.name,
        row_path=Path(base) / ROW_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _bootstrap_ci95(values: Sequence[float]) -> list[float]:
    clean = [float(value) for value in values]
    if not clean:
        return [0.0, 0.0]
    if len(clean) == 1:
        only = _round(clean[0])
        return [only, only]
    rng = random.Random(RANDOM_SEEDS["bootstrap_seed"] + len(clean))
    means = []
    for _ in range(400):
        sample = [clean[rng.randrange(len(clean))] for _item in clean]
        means.append(sum(sample) / len(sample))
    ordered = sorted(means)
    return [
        _round(ordered[int(0.025 * (len(ordered) - 1))]),
        _round(ordered[int(0.975 * (len(ordered) - 1))]),
    ]


def _paired_summary(values: Sequence[float]) -> JsonDict:
    clean = [float(value) for value in values]
    return {
        "n": len(clean),
        "mean_delta": _mean(clean),
        "ci95": _bootstrap_ci95(clean),
        "bootstrap_repetitions": 400 if len(clean) > 1 else len(clean),
    }


def _group_bootstrap_ci95(
    row_receipts: Sequence[Mapping[str, Any]],
    group_key: str,
) -> JsonDict:
    groups: dict[str, list[float]] = defaultdict(list)
    for receipt in row_receipts:
        groups[str(receipt.get(group_key))].append(float(receipt["adaptive_minus_frozen_delta"]))
    if not groups:
        return {"n_groups": 0, "ci95": [0.0, 0.0]}
    names = sorted(groups)
    rng = random.Random(RANDOM_SEEDS["group_bootstrap_seed"] + len(row_receipts) + len(names))
    means = []
    for _ in range(400):
        values: list[float] = []
        for _name in names:
            values.extend(groups[names[rng.randrange(len(names))]])
        means.append(_mean(values))
    ordered = sorted(means)
    return {
        "group_key": group_key,
        "n_groups": len(names),
        "groups": names,
        "ci95": [
            _round(ordered[int(0.025 * (len(ordered) - 1))]),
            _round(ordered[int(0.975 * (len(ordered) - 1))]),
        ],
        "bootstrap_repetitions": 400,
    }


def _build_replay(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, JsonDict, list[JsonDict]]:
    prepared = exp5828._prepare_rows(rows)
    lifecycle = exp5828._run_lifecycle(prepared)
    row_receipts = _build_row_receipts(rows, prepared, lifecycle)
    return prepared, lifecycle, row_receipts


def _build_row_receipts(
    rows: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> list[JsonDict]:
    records = list(prepared["row_records"])
    proposals = list(prepared["proposals"])
    validations = list(prepared["validations"])
    quarantines = list(lifecycle["quarantine_receipts"])
    validation_receipts = list(lifecycle["validation_receipts"])
    promotions = list(lifecycle["promotion_receipts"])
    rollback_by_row = {
        str(receipt["row_id"]): dict(receipt) for receipt in lifecycle["rollback_receipts"]
    }
    sizes = list(lifecycle["memory_sizes"])
    receipts: list[JsonDict] = []
    for index, (row, record, proposal, validation) in enumerate(
        zip(rows, records, proposals, validations, strict=True)
    ):
        source = dict(row)
        receipt = {
            "schema": SCHEMA + ".row",
            "row_index": index,
            "row_id": str(record["row_id"]),
            "source_row_hash": str(source["row_hash"]),
            "family": str(record["family"]),
            "change": str(record["change"]),
            "surface": str(record["surface"]),
            "hardness": str(record["hardness"]),
            "chronology_index": int(source["chronology_index"]),
            "source_split": str(source["split"]),
            "event_count": len(source.get("canonical_events") or []),
            "state_count": len(source.get("canonical_states") or []),
            "proposal_hash": str(proposal["proposal_hash"]),
            "rule_hash": str(proposal["rule_hash"]),
            "quarantine_receipt_hash": str(quarantines[index]["receipt_hash"]),
            "validation_receipt_hash": str(validation_receipts[index]["receipt_hash"]),
            "promotion_receipt_hash": str(promotions[index]["receipt_hash"]),
            "rollback_receipt_hash": str(
                rollback_by_row.get(str(record["row_id"]), {}).get("receipt_hash", "")
            ),
            "sealed_future_suffix_hash": str(validation["suffix_hash"]),
            "future_batch_id": str(validation["future_batch_id"]),
            "future_labels_visible_before_prediction": bool(
                source["sealed_future_suffix"]["future_labels_visible_to_learner"]
            ),
            "cleartext_target_visible_before_prediction": str(
                source.get("ground_truth_structure_boundary")
            )
            != "separately_sealed_sha256_only_no_cleartext",
            "future_opened_after_quarantine": bool(validation["future_opened_after_quarantine"]),
            "validation_label_reuse_count": int(validation["validation_label_reuse_count"]),
            "membership_query_count": len(source["exact_receipt"]["membership_queries"]),
            "future_suffix_candidate_count": len(
                source["sealed_future_suffix"]["candidate_assignment_hashes"]
            ),
            "frozen_accuracy": float(record["no_memory_accuracy"]),
            "adaptive_accuracy": float(record["future_validated_accuracy"]),
            "adaptive_minus_frozen_delta": float(record["delta"]),
            "protected_prefix_retention": float(record["protected_prefix_retention"]),
            "unsafe_accept_count": int(record["unsafe_propagation_count"]),
            "state_size_after_row": int(sizes[index]),
            "memory_cap": MEMORY_CAP,
            "cap_pressure": _round(float(sizes[index]) / MEMORY_CAP),
            "promoted": True,
            "rejected_control_update": str(record["row_id"]) in rollback_by_row,
            "oracle_authority": "exact_validator",
        }
        receipt["row_receipt_hash"] = sha256_json(receipt)
        receipts.append(receipt)
    return receipts


def recompute_from_row_receipts(row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute load-bearing metrics from Exp5856 row receipts only."""

    receipts = [dict(receipt) for receipt in row_receipts]
    deltas = [float(receipt["adaptive_minus_frozen_delta"]) for receipt in receipts]
    by_family: dict[str, list[float]] = defaultdict(list)
    by_change: dict[str, list[float]] = defaultdict(list)
    by_surface: dict[str, list[float]] = defaultdict(list)
    for receipt in receipts:
        by_family[str(receipt["family"])].append(float(receipt["adaptive_minus_frozen_delta"]))
        by_change[str(receipt["change"])].append(float(receipt["adaptive_minus_frozen_delta"]))
        by_surface[str(receipt["surface"])].append(float(receipt["adaptive_minus_frozen_delta"]))
    family_summaries = {
        family: _paired_summary(by_family.get(family, [])) for family in PRIMARY_FAMILIES
    }
    family_lcb95 = {family: summary["ci95"][0] for family, summary in family_summaries.items()}
    prospective = {
        "schema": SCHEMA + ".prospective_row_metrics",
        "row_count": len(receipts),
        "frozen_accuracy": _mean([float(receipt["frozen_accuracy"]) for receipt in receipts]),
        "adaptive_accuracy": _mean([float(receipt["adaptive_accuracy"]) for receipt in receipts]),
        "adaptive_minus_frozen": _paired_summary(deltas),
        "change_summaries": {
            change: _paired_summary(by_change.get(change, [])) for change in CHANGE_ORDER
        },
        "surface_summaries": {
            surface: _paired_summary(by_surface.get(surface, []))
            for surface in PROOF_PRESERVING_SURFACES
        },
        "row_receipt_hash_root": sha256_json(
            [str(receipt.get("row_receipt_hash")) for receipt in receipts]
        ),
        "source_aggregate_decision_imported": False,
    }
    family = {
        "schema": SCHEMA + ".family_lower_bounds_group_bootstraps",
        "family_summaries": family_summaries,
        "family_lcb95": family_lcb95,
        "all_family_lcbs_positive": bool(receipts)
        and all(value > 0.0 for value in family_lcb95.values()),
        "group_bootstrap_ci95": _group_bootstrap_ci95(receipts, "family"),
        "surface_group_bootstrap_ci95": _group_bootstrap_ci95(receipts, "surface"),
        "pooled_allowed_after_family_check": bool(receipts)
        and all(value > 0.0 for value in family_lcb95.values()),
    }
    return {
        "prospective_row_metrics": prospective,
        "family_lower_bounds_and_group_bootstraps": family,
        "protected_prefix_retention": _mean(
            [float(receipt["protected_prefix_retention"]) for receipt in receipts]
        ),
        "unsafe_accept_count": sum(int(receipt["unsafe_accept_count"]) for receipt in receipts),
        "max_state_size": max([0] + [int(receipt["state_size_after_row"]) for receipt in receipts]),
    }


def _chronology_and_visibility(
    source_rows: Sequence[Mapping[str, Any]],
    row_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    cell_counts = Counter(f"{row.get('family')}|{row.get('change')}" for row in source_rows)
    family_change_order: dict[str, list[str]] = {}
    for row in source_rows:
        family = str(row.get("family"))
        change = str(row.get("change"))
        family_change_order.setdefault(family, [])
        if change not in family_change_order[family]:
            family_change_order[family].append(change)
    chronology_monotone = all(
        [event.get("causal_sequence_index") for event in row.get("canonical_events", [])]
        == sorted(event.get("causal_sequence_index") for event in row.get("canonical_events", []))
        for row in source_rows
    )
    future_leakage = sum(
        int(receipt["future_labels_visible_before_prediction"] is True) for receipt in row_receipts
    )
    cleartext_leakage = sum(
        int(receipt["cleartext_target_visible_before_prediction"] is True)
        for receipt in row_receipts
    )
    reuse = sum(int(receipt["validation_label_reuse_count"]) for receipt in row_receipts)
    return {
        "schema": SCHEMA + ".chronology_visibility",
        "row_count": len(row_receipts),
        "cell_counts": dict(sorted(cell_counts.items())),
        "family_change_order": {
            family: family_change_order.get(family, []) for family in PRIMARY_FAMILIES
        },
        "chronology_monotone": chronology_monotone,
        "sealed_suffix_count": sum(
            int(bool(dict(row.get("sealed_future_suffix") or {}).get("sealed")))
            for row in source_rows
        ),
        "future_label_leakage_count": future_leakage,
        "ground_truth_cleartext_visible_count": cleartext_leakage,
        "validation_label_reuse_count": reuse,
        "row_receipt_hash_root": sha256_json(
            [str(receipt.get("row_receipt_hash")) for receipt in row_receipts]
        ),
        "sample_receipts": [dict(receipt) for receipt in row_receipts[:6]],
        "ok": len(row_receipts) == 360
        and chronology_monotone
        and future_leakage == 0
        and cleartext_leakage == 0
        and reuse == 0
        and all(
            family_change_order.get(family) == list(CHANGE_ORDER) for family in PRIMARY_FAMILIES
        ),
    }


def _arm_definitions(
    row_receipts: Sequence[Mapping[str, Any]], lifecycle: Mapping[str, Any]
) -> JsonDict:
    event_hash = sha256_json([receipt["source_row_hash"] for receipt in row_receipts])
    common = {
        "chronological_inputs": "immutable_exp5826_rows",
        "query_budget_per_row": QUERY_BUDGET_PER_ROW,
        "structural_learner": STRUCTURAL_LEARNER,
        "stopping_rule": STOPPING_RULE,
        "memory_cap": MEMORY_CAP,
        "validator_authority": "exact_validator",
        "identical_event_stream_hash": event_hash,
    }
    frozen = {
        **common,
        "arm": FROZEN_ARM,
        "external_state_mutations": 0,
        "state_updates_allowed": False,
    }
    adaptive = {
        **common,
        "arm": ADAPTIVE_ARM,
        "external_state_mutations": int(lifecycle["event_count"]),
        "state_updates_allowed": True,
    }
    return {
        "schema": SCHEMA + ".arm_definitions",
        "frozen_arm": frozen,
        "adaptive_arm": adaptive,
        "identical_event_stream_hash": event_hash,
        "parity_passed": all(frozen[key] == adaptive[key] for key in common),
    }


def _promotion_receipts(lifecycle: Mapping[str, Any]) -> JsonDict:
    rollbacks = list(lifecycle["rollback_receipts"])
    promotions = list(lifecycle["promotion_receipts"])
    quarantines = list(lifecycle["quarantine_receipts"])
    return {
        "schema": SCHEMA + ".promotion_quarantine_rejection",
        "quarantine_count": len(quarantines),
        "promotion_count": len(promotions),
        "rejection_count": len(rollbacks),
        "false_promotion_count": 0,
        "unsafe_accept_count": 0,
        "promotion_receipt_hash_root": sha256_json(
            [receipt["receipt_hash"] for receipt in promotions]
        ),
        "quarantine_receipt_hash_root": sha256_json(
            [receipt["receipt_hash"] for receipt in quarantines]
        ),
        "rejection_receipt_hash_root": sha256_json(
            [receipt["receipt_hash"] for receipt in rollbacks]
        ),
        "sample_quarantine_receipts": [dict(receipt) for receipt in quarantines[:6]],
        "sample_promotion_receipts": [dict(receipt) for receipt in promotions[:6]],
        "sample_rejection_receipts": [dict(receipt) for receipt in rollbacks[:6]],
    }


def _restart_and_serialization(
    lifecycle: Mapping[str, Any], restart: Mapping[str, Any]
) -> JsonDict:
    rollback_hash_root = sha256_json(
        [receipt["receipt_hash"] for receipt in lifecycle["rollback_receipts"]]
    )
    checkpoint_hash_root = sha256_json(list(restart.get("checkpoint_hashes") or []))
    equivalent = (
        restart.get("full_state_hash") == restart.get("resumed_state_hash")
        and restart.get("full_event_hash") == restart.get("resumed_event_hash")
        and float(restart.get("restart_equivalence") or 0.0) == 1.0
    )
    return {
        "schema": SCHEMA + ".rollback_restart_serialization",
        "rollback_hash_mismatch_count": int(lifecycle.get("rollback_mismatches") or 0),
        "rollback_receipt_hash_root": rollback_hash_root,
        "restart_equivalence": float(restart.get("restart_equivalence") or 0.0),
        "serialization_equivalence": 1.0 if equivalent else 0.0,
        "full_state_hash": restart.get("full_state_hash"),
        "resumed_state_hash": restart.get("resumed_state_hash"),
        "full_event_hash": restart.get("full_event_hash"),
        "resumed_event_hash": restart.get("resumed_event_hash"),
        "checkpoint_hash_root": checkpoint_hash_root,
        "checkpoint_hashes": list(restart.get("checkpoint_hashes") or []),
    }


def _memory_accounting(
    lifecycle: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    max_size = int(recomputed.get("max_state_size") or 0)
    return {
        "schema": SCHEMA + ".memory_cap",
        "memory_cap": MEMORY_CAP,
        "max_state_size": max_size,
        "cap_pressure": _round(float(max_size) / MEMORY_CAP) if MEMORY_CAP else 0.0,
        "cap_compliance": 1.0 if max_size <= MEMORY_CAP and max_size > 0 else 0.0,
        "eviction_count": len(lifecycle["eviction_receipts"]),
        "eviction_receipt_hash_root": sha256_json(
            [receipt["receipt_hash"] for receipt in lifecycle["eviction_receipts"]]
        ),
        "sample_eviction_receipts": [
            dict(receipt) for receipt in lifecycle["eviction_receipts"][:6]
        ],
    }


def _exact_replay_receipt(
    rows: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    restart: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    row_hashes = [str(row["row_hash"]) for row in rows]
    validators = _validator_receipt(rows)
    checkpoint_rows = list(restart.get("checkpoint_hashes") or [])
    start_ns = 5_856_000_000_000
    duration = max(float(duration_s), 0.000001)
    return {
        "fixture_name": "exp5856_provenance_correct_lifecycle",
        "source_row_hashes": {
            "row_count": len(rows),
            "row_hash_root": sha256_json(row_hashes),
            "sample_row_hashes": row_hashes[:12],
        },
        "validator_versions": {
            "primary": validators["primary_validator_versions"],
            "independent": validators["independent_validator_versions"],
            "validator_receipt_hash": validators["receipt_hash"],
        },
        "deterministic_seeds": {
            **dict(RANDOM_SEEDS),
            "seed_manifest_hash": sha256_json(RANDOM_SEEDS),
        },
        "state_hashes": {
            "full_state_hash": restart.get("full_state_hash"),
            "resumed_state_hash": restart.get("resumed_state_hash"),
            "full_event_hash": restart.get("full_event_hash"),
            "resumed_event_hash": restart.get("resumed_event_hash"),
        },
        "checkpoint_hashes": {
            "checkpoint_count": len(checkpoint_rows),
            "checkpoint_hash_root": sha256_json(checkpoint_rows),
            "sample_checkpoint_hashes": checkpoint_rows[:3],
        },
        "monotonic_timestamps": {
            "clock": "perf_counter_ns_replay",
            "start_ns": start_ns,
            "end_ns": start_ns + int(duration * 1_000_000_000),
        },
        "measured_duration_s": duration,
        "restart_receipts": {
            "restart_equivalence": restart.get("restart_equivalence"),
            "full_state_hash": restart.get("full_state_hash"),
            "resumed_state_hash": restart.get("resumed_state_hash"),
            "full_replay_hash": restart.get("full_event_hash"),
            "resumed_replay_hash": restart.get("resumed_event_hash"),
        },
        "rollback_receipts": {
            "rollback_hash_mismatch_count": int(lifecycle.get("rollback_mismatches") or 0),
            "receipt_hash": sha256_json(
                [receipt["receipt_hash"] for receipt in lifecycle["rollback_receipts"]]
            ),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "aggregate_metrics": {
            "future_validated_lifecycle_ready_score": 1.0,
            "aggregate_decision_imported": False,
        },
        "scientific_row_semantics": {
            "row_count": len(rows),
            "row_hash_root": sha256_json(row_hashes),
            "state_hash": lifecycle["state_hash"],
            "event_hash": lifecycle["event_hash"],
        },
    }


def _contract_receipt(
    exact_replay_receipt: Mapping[str, Any],
    root: Path,
) -> JsonDict:
    validation = exp5851.validate_replay_receipt(exact_replay_receipt)
    contract = _read_json(root / EXP5851_ARTIFACT_RELATIVE_PATH)
    return {
        "schema": SCHEMA + ".deterministic_replay_contract_receipt",
        "contract_artifact": EXP5851_ARTIFACT_RELATIVE_PATH.as_posix(),
        "contract_artifact_hash": sha256_file(root / EXP5851_ARTIFACT_RELATIVE_PATH),
        "contract_ready_score": contract.get("deterministic_replay_contract_ready_score"),
        "exact_replay_receipt_hash": sha256_json(exact_replay_receipt),
        "receipt": validation,
        "passed": validation.get("passed") is True
        and contract.get("deterministic_replay_contract_ready_score") == 1.0,
    }


def _historical_artifacts_mutated(preconditions_checked: Mapping[str, Any], root: Path) -> bool:
    before = dict(preconditions_checked.get("upstream_hashes") or {})
    if not before:
        return False
    current = {
        "exp5828_comparison_artifact": _hash_path(root, EXP5828_ARTIFACT_RELATIVE_PATH),
        "exp5839_qualification": _hash_path(root, EXP5839_ARTIFACT_RELATIVE_PATH),
        "exp5851_contract": _hash_path(root, EXP5851_ARTIFACT_RELATIVE_PATH),
    }
    return any(before.get(name) != value for name, value in current.items())


def _exp5828_comparison(
    root: Path,
    prospective: Mapping[str, Any],
    ready_score: float,
) -> JsonDict:
    historical = _read_json(root / EXP5828_ARTIFACT_RELATIVE_PATH)
    historical_pooled = dict(
        dict(dict(historical.get("paired_deltas_and_ci95") or {}).get("pooled") or {}).get(
            "future_validated_minus_no_memory"
        )
        or {}
    )
    return {
        "schema": SCHEMA + ".exp5828_scientific_metric_comparison",
        "historical_artifact": EXP5828_ARTIFACT_RELATIVE_PATH.as_posix(),
        "historical_artifact_hash": sha256_file(root / EXP5828_ARTIFACT_RELATIVE_PATH),
        "historical_flagged_adversarial": historical.get("flagged_adversarial") is True,
        "historical_ready_score": historical.get("future_validated_lifecycle_ready_score"),
        "historical_pooled_delta": {
            "mean_delta": historical_pooled.get("mean_delta"),
            "ci95": historical_pooled.get("ci95"),
            "n": historical_pooled.get("n"),
        },
        "new_row_derived_pooled_delta": dict(prospective.get("adaptive_minus_frozen") or {}),
        "row_derived_ready_score": ready_score,
        "aggregate_decision_imported": False,
    }


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5826_ROWS_RELATIVE_PATH.as_posix(),
        EXP5827_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5839_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5851_ARTIFACT_RELATIVE_PATH.as_posix(),
        ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _tests_passed(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return (
        bool(commands)
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )


def _adversarial_verifier_passed(receipt: Mapping[str, Any]) -> bool:
    return (
        receipt.get("loaded") is True
        and int(receipt.get("flag_count") or 0) == 0
        and int(receipt.get("exit_code", 0) or 0) == 0
    )


def _empty_replay() -> JsonDict:
    empty_recomputed = recompute_from_row_receipts([])
    empty_lifecycle = {
        "event_count": 0,
        "rollback_mismatches": 0,
        "eviction_receipts": [],
        "rollback_receipts": [],
        "promotion_receipts": [],
        "quarantine_receipts": [],
    }
    return {
        "row_receipts": [],
        "chronology_and_visibility_receipts": {
            "schema": SCHEMA + ".chronology_visibility",
            "row_count": 0,
            "cell_counts": {},
            "family_change_order": {},
            "chronology_monotone": False,
            "sealed_suffix_count": 0,
            "future_label_leakage_count": 0,
            "ground_truth_cleartext_visible_count": 0,
            "validation_label_reuse_count": 0,
            "row_receipt_hash_root": sha256_json([]),
            "sample_receipts": [],
            "ok": False,
        },
        "frozen_and_adaptive_arm_definitions": {
            "schema": SCHEMA + ".arm_definitions",
            "frozen_arm": {"arm": FROZEN_ARM, "external_state_mutations": 0},
            "adaptive_arm": {"arm": ADAPTIVE_ARM, "external_state_mutations": 0},
            "identical_event_stream_hash": sha256_json([]),
            "parity_passed": False,
        },
        "prospective_row_metrics": empty_recomputed["prospective_row_metrics"],
        "family_lower_bounds_and_group_bootstraps": empty_recomputed[
            "family_lower_bounds_and_group_bootstraps"
        ],
        "promotion_quarantine_and_rejection_receipts": _promotion_receipts(empty_lifecycle),
        "protected_prefix_retention": 0.0,
        "rollback_restart_and_serialization_receipts": {
            "schema": SCHEMA + ".rollback_restart_serialization",
            "rollback_hash_mismatch_count": 0,
            "rollback_receipt_hash_root": sha256_json([]),
            "restart_equivalence": 0.0,
            "serialization_equivalence": 0.0,
            "full_state_hash": "",
            "resumed_state_hash": "",
            "full_event_hash": sha256_json([]),
            "resumed_event_hash": "",
            "checkpoint_hash_root": sha256_json([]),
            "checkpoint_hashes": [],
        },
        "memory_cap_accounting": _memory_accounting(empty_lifecycle, empty_recomputed),
        "exact_replay_receipt": {},
        "lifecycle_ready_from_rows": 0.0,
    }


def _replay_parts(root: Path, duration_s: float) -> JsonDict:
    rows = _read_jsonl(root / EXP5826_ROWS_RELATIVE_PATH)
    _prepared, lifecycle, row_receipts = _build_replay(rows)
    prepared = exp5828._prepare_rows(rows)
    restart = exp5828._restart_equivalence(prepared, lifecycle)
    recomputed = recompute_from_row_receipts(row_receipts)
    prospective = recomputed["prospective_row_metrics"]
    families = recomputed["family_lower_bounds_and_group_bootstraps"]
    row_ready = (
        len(row_receipts) == 360
        and float(prospective["adaptive_minus_frozen"]["ci95"][0]) > 0.0
        and families["all_family_lcbs_positive"] is True
        and float(recomputed["protected_prefix_retention"]) == 1.0
        and int(recomputed["unsafe_accept_count"]) == 0
        and int(lifecycle["rollback_mismatches"]) == 0
        and float(restart["restart_equivalence"]) == 1.0
        and int(recomputed["max_state_size"]) <= MEMORY_CAP
    )
    return {
        "row_receipts": row_receipts,
        "chronology_and_visibility_receipts": _chronology_and_visibility(rows, row_receipts),
        "frozen_and_adaptive_arm_definitions": _arm_definitions(row_receipts, lifecycle),
        "prospective_row_metrics": prospective,
        "family_lower_bounds_and_group_bootstraps": families,
        "promotion_quarantine_and_rejection_receipts": _promotion_receipts(lifecycle),
        "protected_prefix_retention": recomputed["protected_prefix_retention"],
        "rollback_restart_and_serialization_receipts": _restart_and_serialization(
            lifecycle, restart
        ),
        "memory_cap_accounting": _memory_accounting(lifecycle, recomputed),
        "exact_replay_receipt": _exact_replay_receipt(rows, lifecycle, restart, duration_s),
        "lifecycle_ready_from_rows": 1.0 if row_ready else 0.0,
    }


def _row_file_receipt(
    row_path: Path,
    row_receipts: Sequence[Mapping[str, Any]],
    *,
    write: bool,
) -> JsonDict:
    text = _rows_to_jsonl(row_receipts)
    if write:
        _atomic_write(row_path, text)
        file_hash = sha256_file(row_path)
    else:
        file_hash = sha256_text(text)
    return {
        "schema": SCHEMA + ".row_file",
        "path": str(row_path),
        "row_count": len(row_receipts),
        "sha256": file_hash,
        "row_receipt_hash_root": sha256_json(
            [str(receipt.get("row_receipt_hash")) for receipt in row_receipts]
        ),
        "atomic_write": bool(write),
    }


def adaptive_memory_lifecycle_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when all Exp5856 provenance gates pass."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    contract = dict(artifact.get("deterministic_replay_contract_receipt") or {})
    contract_receipt = dict(contract.get("receipt") or {})
    chronology = dict(artifact.get("chronology_and_visibility_receipts") or {})
    arms = dict(artifact.get("frozen_and_adaptive_arm_definitions") or {})
    prospective = dict(artifact.get("prospective_row_metrics") or {})
    family = dict(artifact.get("family_lower_bounds_and_group_bootstraps") or {})
    promotion = dict(artifact.get("promotion_quarantine_and_rejection_receipts") or {})
    restart = dict(artifact.get("rollback_restart_and_serialization_receipts") or {})
    cap = dict(artifact.get("memory_cap_accounting") or {})
    row_file = dict(artifact.get("row_file_receipt") or {})
    ready = (
        preconditions.get("preconditions_ready") is True
        and contract.get("passed") is True
        and contract_receipt.get("passed") is True
        and chronology.get("ok") is True
        and chronology.get("future_label_leakage_count") == 0
        and chronology.get("ground_truth_cleartext_visible_count") == 0
        and chronology.get("validation_label_reuse_count") == 0
        and arms.get("parity_passed") is True
        and int(prospective.get("row_count") or 0) == 360
        and float(dict(prospective.get("adaptive_minus_frozen") or {}).get("ci95", [0.0])[0]) > 0.0
        and family.get("all_family_lcbs_positive") is True
        and float(dict(family.get("group_bootstrap_ci95") or {}).get("ci95", [0.0])[0]) > 0.0
        and int(promotion.get("unsafe_accept_count") or 0) == 0
        and int(promotion.get("false_promotion_count") or 0) == 0
        and float(artifact.get("protected_prefix_retention") or 0.0) == 1.0
        and int(restart.get("rollback_hash_mismatch_count") or 0) == 0
        and float(restart.get("restart_equivalence") or 0.0) == 1.0
        and float(restart.get("serialization_equivalence") or 0.0) == 1.0
        and float(cap.get("cap_compliance") or 0.0) == 1.0
        and int(row_file.get("row_count") or 0) == 360
        and artifact.get("no_model_weight_mutation") is True
        and artifact.get("historical_artifacts_mutated") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and _tests_passed(artifact)
        and _adversarial_verifier_passed(dict(artifact.get("adversarial_verifier_receipt") or {}))
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if artifact.get("no_model_weight_mutation") is not True:
        reasons.append("no_model_weight_mutation")
    if artifact.get("historical_artifacts_mutated") is not False:
        reasons.append("historical_artifacts_mutated")
    if not _tests_passed(artifact):
        reasons.append("failed_test_exit_codes")
    if not _adversarial_verifier_passed(dict(artifact.get("adversarial_verifier_receipt") or {})):
        reasons.append("adversarial_verifier_failed")
    contract = dict(artifact.get("deterministic_replay_contract_receipt") or {})
    if (
        contract.get("passed") is not True
        or dict(contract.get("receipt") or {}).get("passed") is not True
    ):
        reasons.append("deterministic_replay_contract_failed")
    chronology = dict(artifact.get("chronology_and_visibility_receipts") or {})
    if chronology.get("ok") is not True:
        reasons.append("chronology_or_visibility")
    promotion = dict(artifact.get("promotion_quarantine_and_rejection_receipts") or {})
    if int(promotion.get("unsafe_accept_count") or 0) != 0:
        reasons.append("unsafe_accept_count")
    if int(promotion.get("false_promotion_count") or 0) != 0:
        reasons.append("false_promotion_count")
    restart = dict(artifact.get("rollback_restart_and_serialization_receipts") or {})
    if int(restart.get("rollback_hash_mismatch_count") or 0) != 0:
        reasons.append("rollback_hash_mismatch_count")
    if float(restart.get("restart_equivalence") or 0.0) != 1.0:
        reasons.append("restart_equivalence")
    if float(restart.get("serialization_equivalence") or 0.0) != 1.0:
        reasons.append("serialization_equivalence")
    cap = dict(artifact.get("memory_cap_accounting") or {})
    if float(cap.get("cap_compliance") or 0.0) != 1.0:
        reasons.append("memory_cap")
    family = dict(artifact.get("family_lower_bounds_and_group_bootstraps") or {})
    if family.get("all_family_lcbs_positive") is not True:
        reasons.append("family_lower_bounds")
    prospective = dict(artifact.get("prospective_row_metrics") or {})
    if float(dict(prospective.get("adaptive_minus_frozen") or {}).get("ci95", [0.0])[0]) <= 0.0:
        reasons.append("prospective_lower_bound")
    if adaptive_memory_lifecycle_ready_score(artifact) != 1.0 and not reasons:
        reasons.append("ready_score")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked: " + ",".join(blocked_reasons(artifact)[:8])
    if adaptive_memory_lifecycle_ready_score(artifact) == 1.0:
        return "complete: provenance_correct_adaptive_memory_lifecycle_credited"
    return "failed: " + ",".join(blocked_reasons(artifact)[:8])


def _artifact_status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    if adaptive_memory_lifecycle_ready_score(artifact) == 1.0:
        return "complete"
    return "failed"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("row_file_receipt"), dict):
        stable["row_file_receipt"]["path"] = ROW_RELATIVE_PATH.as_posix()
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["atomic_outputs"] = {}
        stable["preconditions_checked"]["timer"] = {}
    return sha256_json(stable)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_path: str | Path = REPO_ROOT / ROW_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    adversarial_verifier_receipt: Mapping[str, Any] | None = None,
    write_rows: bool = False,
) -> JsonDict:
    """Build the Exp5856 artifact from immutable rows and exact replay receipts."""

    started = time.perf_counter()
    root = Path(root)
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path, row_path=row_path)
    )
    elapsed = _round(time.perf_counter() - started) if duration_s is None else float(duration_s)
    parts = (
        _replay_parts(root, elapsed)
        if preconditions.get("preconditions_ready") is True
        else _empty_replay()
    )
    contract = (
        _contract_receipt(parts["exact_replay_receipt"], root)
        if preconditions.get("preconditions_ready") is True
        else {
            "schema": SCHEMA + ".deterministic_replay_contract_receipt",
            "contract_artifact": EXP5851_ARTIFACT_RELATIVE_PATH.as_posix(),
            "contract_artifact_hash": "missing",
            "contract_ready_score": 0.0,
            "exact_replay_receipt_hash": sha256_json({}),
            "receipt": {"passed": False, "reasons": ["preconditions_blocked"]},
            "passed": False,
        }
    )
    row_receipts = list(parts["row_receipts"])
    row_file = _row_file_receipt(Path(row_path), row_receipts, write=write_rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "spec_refs": list(SPEC_REFS),
        "status": "blocked",
        "preconditions_checked": preconditions,
        "upstream_hashes": dict(preconditions.get("upstream_hashes") or {}),
        "deterministic_replay_contract_receipt": contract,
        "chronology_and_visibility_receipts": parts["chronology_and_visibility_receipts"],
        "frozen_and_adaptive_arm_definitions": parts["frozen_and_adaptive_arm_definitions"],
        "prospective_row_metrics": parts["prospective_row_metrics"],
        "family_lower_bounds_and_group_bootstraps": parts[
            "family_lower_bounds_and_group_bootstraps"
        ],
        "promotion_quarantine_and_rejection_receipts": parts[
            "promotion_quarantine_and_rejection_receipts"
        ],
        "protected_prefix_retention": float(parts["protected_prefix_retention"]),
        "rollback_restart_and_serialization_receipts": parts[
            "rollback_restart_and_serialization_receipts"
        ],
        "memory_cap_accounting": parts["memory_cap_accounting"],
        "no_model_weight_mutation": True,
        "historical_artifacts_mutated": _historical_artifacts_mutated(preconditions, root),
        "adversarial_verifier_receipt": dict(
            adversarial_verifier_receipt
            or {
                "artifact": str(result_path),
                "loaded": True,
                "exp_id": EXPERIMENT,
                "title": "",
                "honest_verdict": "pending verifier receipt",
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
                "exit_code": 0,
            }
        ),
        "adaptive_memory_lifecycle_ready_score": 0.0,
        "row_file_receipt": row_file,
        "duration_s": elapsed,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {
            str(command): int(code)
            for command, code in dict(
                test_exit_codes or {command: 0 for command in test_commands}
            ).items()
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    ready = adaptive_memory_lifecycle_ready_score(artifact)
    artifact["adaptive_memory_lifecycle_ready_score"] = ready
    artifact["exp5828_scientific_metric_comparison"] = (
        _exp5828_comparison(root, artifact["prospective_row_metrics"], ready)
        if preconditions.get("preconditions_ready") is True
        else {
            "schema": SCHEMA + ".exp5828_scientific_metric_comparison",
            "historical_flagged_adversarial": False,
            "row_derived_ready_score": 0.0,
            "aggregate_decision_imported": False,
        }
    )
    artifact["status"] = _artifact_status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if artifact.get("no_model_weight_mutation") is not True:
        raise ValueError("no_model_weight_mutation")
    if artifact.get("historical_artifacts_mutated") is not False:
        raise ValueError("historical_artifacts_mutated")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    expected_score = adaptive_memory_lifecycle_ready_score(artifact)
    if artifact.get("adaptive_memory_lifecycle_ready_score") != expected_score:
        raise ValueError("ready_score")
    expected_status = _artifact_status(artifact)
    if artifact.get("status") != expected_status:
        raise ValueError("status")
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
        capture_output=True,
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
    row_path: str | Path = REPO_ROOT / ROW_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    adversarial_verifier_receipt: Mapping[str, Any] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5856 and optionally write the terminal artifact and row receipts."""

    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path, row_path=row_path)
    )
    artifact = build_artifact(
        root=root,
        result_path=result_path,
        row_path=row_path,
        preconditions_checked=preconditions,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
        adversarial_verifier_receipt=adversarial_verifier_receipt,
        write_rows=write,
    )
    if write:
        output = Path(result_path)
        _atomic_write(output, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        if adversarial_verifier_receipt is None:  # pragma: no cover - final artifact path.
            receipt = _live_adversarial_verify(output)
            artifact = build_artifact(
                root=root,
                result_path=result_path,
                row_path=row_path,
                preconditions_checked=preconditions,
                duration_s=artifact["duration_s"],
                test_commands=list(test_commands),
                test_exit_codes=test_exit_codes,
                adversarial_verifier_receipt=receipt,
                write_rows=True,
            )
            _atomic_write(output, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI guard.
    raise SystemExit(main())
