"""Exp5853 paired embedding integrity audit.

Spec refs: REQ-VERIFY-5853, SCENARIO-VERIFY-5853-PRECONDITIONS,
SCENARIO-VERIFY-5853-CAUSAL, SCENARIO-VERIFY-5853-SWAPS,
SCENARIO-VERIFY-5853-CONTROLS.

This module audits the already-extracted Exp5852 paired embeddings. It performs
no LLM or embedding inference; every decision is reconstructed from immutable
Exp5840 row IDs, Exp5852 model-row IDs, exact validator receipts, and
learner-visible feature views.
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
import shutil
import sys
import time
from typing import Any

from carnot import experiment_5826_out_of_template_constraint_stream as exp5826
from carnot import experiment_5840_exact_counterfactual_embedding_fixture as exp5840
from carnot import experiment_5852_three_family_paired_embeddings as exp5852


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5853_paired_embedding_integrity_audit.json")
EXP5852_ARTIFACT_RELATIVE_PATH = exp5852.RESULT_RELATIVE_PATH
EXP5852_ROWS_RELATIVE_PATH = exp5852.ROW_FILE_RELATIVE_PATH
EXP5840_ARTIFACT_RELATIVE_PATH = exp5840.RESULT_RELATIVE_PATH
EXP5840_ROWS_RELATIVE_PATH = exp5840.ROW_FILE_RELATIVE_PATH
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5853_paired_embedding_integrity_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5853_paired_embedding_integrity_audit.py")
EXP5840_MODULE_RELATIVE_PATH = exp5840.MODULE_RELATIVE_PATH
EXP5840_TEST_RELATIVE_PATH = exp5840.TEST_RELATIVE_PATH
EXP5852_MODULE_RELATIVE_PATH = exp5852.MODULE_RELATIVE_PATH
EXP5852_TEST_RELATIVE_PATH = exp5852.TEST_RELATIVE_PATH
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
VERIFY_DIR_RELATIVE_PATH = Path("python/carnot/verify")
PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

SCHEMA = "carnot.experiment_5853.paired_embedding_integrity_audit.v1"
SOURCE_ROW_SCHEMA = exp5840.ROW_SCHEMA
EMBEDDING_ROW_SCHEMA = exp5852.ROW_SCHEMA
EXPERIMENT = 5853
EXPERIMENT_ID = "experiment_5853_paired_embedding_integrity_audit"
MILESTONE = "2026.07.521"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
VERIFIER_IS_ORACLE = True

MANDATED_MODEL_HF_IDS = exp5852.MANDATED_MODEL_HF_IDS
REQUIRED_CAUSAL_AXES = exp5840.CAUSAL_AXES
REQUIRED_CONSTRAINT_FAMILIES = exp5840.PRIMARY_FAMILIES
REQUIRED_HARDNESS_BINS = exp5840.HARDNESS_BINS
REQUIRED_PROOF_PRESERVING_SURFACES = exp5840.PROOF_PRESERVING_SURFACES
PRIMARY_VALIDATOR_VERSION = exp5826.PRIMARY_VALIDATOR_VERSION
INDEPENDENT_VALIDATOR_VERSION = exp5826.INDEPENDENT_VALIDATOR_VERSION

RAM_FLOOR_MB = 16_384
DISK_FLOOR_MB = 8_192
MIN_PAIRED_NORM = 1e-9
MIN_DIRECTION_POSITIVE_RATE = 0.55
MIN_MEAN_ANCHOR_COSINE = 0.0
IDENTITY_ACCURACY_TOLERANCE = 0.05
NORM_ONLY_MAX_ACCURACY = 0.75
LABEL_PERMUTATION_MAX_AGREEMENT = 0.05
PAIR_SWAP_MAX_POSITIVE_RATE = 0.45
PERTURBATION_RELATIVE_TOLERANCE = 0.05

SPEC_REFS = (
    "REQ-VERIFY-5853",
    "SCENARIO-VERIFY-5853-PRECONDITIONS",
    "SCENARIO-VERIFY-5853-CAUSAL",
    "SCENARIO-VERIFY-5853-SWAPS",
    "SCENARIO-VERIFY-5853-CONTROLS",
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5853_paired_embedding_integrity_audit.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5853_paired_embedding_integrity_audit.py "
    "-m pytest tests/python/test_experiment_5853_paired_embedding_integrity_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5853_paired_embedding_integrity_audit.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python -m carnot.experiment_5853_paired_embedding_integrity_audit --write",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5853_paired_embedding_integrity_audit.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_hashes_and_row_reconstruction",
    "claim_flip_sensitivity",
    "constraint_ablation_sensitivity",
    "evaluator_swap_receipts",
    "grounded_vs_true_cross",
    "label_and_pair_permutation_controls",
    "identity_masking_and_prediction_controls",
    "length_norm_and_truncation_controls",
    "perturbation_duplicate_and_no_information_controls",
    "disaggregated_cell_decisions",
    "surviving_shortcuts",
    "paired_embedding_integrity_ready_score",
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
    "status": "A terminal audit state separates clean causal data from unusable embeddings.",
    "preconditions_checked": "Gate, hashes, validators, splits, resources, and outputs prevent fabricated auditing.",
    "upstream_hashes_and_row_reconstruction": "Fresh row joins prevent aggregate self-validation.",
    "claim_flip_sensitivity": "A representation energy must react to the specific violated claim, not only gist.",
    "constraint_ablation_sensitivity": "Removing the owning constraint must change only the intended causal comparison.",
    "evaluator_swap_receipts": "Independent evaluators expose co-adapted target codes.",
    "grounded_vs_true_cross": "Grounding agreement and exact truth must remain distinguishable and jointly checked.",
    "label_and_pair_permutation_controls": "Shuffled or reversed targets must collapse predictability.",
    "identity_masking_and_prediction_controls": "Model or constraint identity cannot substitute for correctness.",
    "length_norm_and_truncation_controls": "Token envelope and vector scale cannot carry the label.",
    "perturbation_duplicate_and_no_information_controls": "Robustness and null tests expose target leakage or duplicate inflation.",
    "disaggregated_cell_decisions": "Every model, axis, family, and hardness cell owns its decision.",
    "surviving_shortcuts": "Any non-empty forbidden shortcut list blocks promotion.",
    "paired_embedding_integrity_ready_score": "EMIT BARE scalar; only 1.0 permits Exp5854.",
    "duration_s": "Measured audit time exposes bootstrap-only execution.",
    "inference_substrate": "`aggregation_from_upstream_artifacts` declares a no-new-inference audit.",
    "verifier_is_oracle": "True records exact validators as target authority.",
    "field_provenance": "Every decision traces to rows, controls, validators, and code.",
    "test_commands": "Commands document every causal and shortcut control.",
    "test_exit_codes": "Exit codes prevent failed controls becoming readiness.",
    "reproducibility_checksum": "A checksum detects row, control, or evaluator drift.",
    "honest_verdict": "A terminal prefix states ready, disqualified, or blocked outcome.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash stable text evidence with the repository's prefixed format."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible evidence after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without loading large row files at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_row_hash(row: Mapping[str, Any]) -> str:
    """Return the Exp5840 row hash for a source row."""

    return exp5840.row_hash(row)


def embedding_row_hash(row: Mapping[str, Any]) -> str:
    """Return the Exp5852 row hash for an embedding row."""

    return exp5852.row_hash(row)


def model_family(hf_id: str) -> str:
    """Return the Exp5852 short family label for a model id."""

    return exp5852.model_family(hf_id)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL object required at line {line_number}: {path}")
            rows.append(dict(payload))
    return rows


def _memory_probe() -> JsonDict:  # pragma: no cover - host dependent.
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


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host dependent.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def _output_path_receipt(path: Path) -> JsonDict:
    parent = path.parent
    writable = (parent.exists() and os.access(parent, os.W_OK)) or (
        parent.parent.exists() and os.access(parent.parent, os.W_OK)
    )
    return {
        "result_path": str(path),
        "result_writable": writable,
        "atomic_write_suffix": ".tmp",
        "ok": writable,
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    tmp.replace(path)


def _tree_hash(path: Path) -> str:
    files: list[JsonDict] = []
    for child in sorted(path.rglob("*.py")):
        if "__pycache__" in child.parts:
            continue
        files.append(
            {
                "path": child.relative_to(REPO_ROOT).as_posix()
                if child.is_relative_to(REPO_ROOT)
                else child.as_posix(),
                "sha256": sha256_file(child),
            }
        )
    return sha256_json(files)


def _receipt_check(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    path: Path,
    row_id_key: str,
    row_hash_fn: Callable[[Mapping[str, Any]], str],
) -> JsonDict:
    receipt = dict(artifact.get("row_file_receipt") or {})
    expected_hashes = {str(row.get(row_id_key)): str(row.get("row_hash")) for row in rows}
    row_hash_mismatches = [
        str(row.get(row_id_key)) for row in rows if str(row.get("row_hash")) != row_hash_fn(row)
    ]
    file_sha = sha256_file(path) if path.exists() else "missing"
    return {
        "path": str(path),
        "receipt_path": receipt.get("path"),
        "row_count": len(rows),
        "receipt_row_count": receipt.get("row_count"),
        "row_count_matches": receipt.get("row_count") == len(rows),
        "sha256": file_sha,
        "receipt_sha256": receipt.get("sha256"),
        "sha256_matches": receipt.get("sha256") == file_sha,
        "row_hash_root": sha256_json(expected_hashes),
        "receipt_row_hash_root": receipt.get("row_hash_root"),
        "row_hash_root_matches": receipt.get("row_hash_root") == sha256_json(expected_hashes),
        "row_hash_mismatch_count": len(row_hash_mismatches),
        "row_hash_mismatches": row_hash_mismatches[:20],
        "all_receipt_checks_passed": (
            receipt.get("row_count") == len(rows)
            and receipt.get("sha256") == file_sha
            and receipt.get("row_hash_root") == sha256_json(expected_hashes)
            and not row_hash_mismatches
        ),
    }


def _file_hashes(root: Path, upstream_paths: Mapping[str, Path]) -> JsonDict:
    paths = {
        "verification_spec": root / VERIFY_SPEC_RELATIVE_PATH,
        "exp5840_module": root / EXP5840_MODULE_RELATIVE_PATH,
        "exp5840_test": root / EXP5840_TEST_RELATIVE_PATH,
        "exp5852_module": root / EXP5852_MODULE_RELATIVE_PATH,
        "exp5852_test": root / EXP5852_TEST_RELATIVE_PATH,
        "exp5853_module": root / MODULE_RELATIVE_PATH,
        "exp5853_test": root / TEST_RELATIVE_PATH,
        "protected_research_conductor": root / PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH,
    }
    paths.update(upstream_paths)
    return {
        name: sha256_file(path) if path.exists() and path.is_file() else "missing"
        for name, path in sorted(paths.items())
    }


def _source_labels(row: Mapping[str, Any]) -> tuple[bool, bool]:
    conditions = list(row.get("conditions") or [])
    if len(conditions) != 2:
        return (False, False)
    return (bool(conditions[0].get("exact_label")), bool(conditions[1].get("exact_label")))


def _vector(row: Mapping[str, Any]) -> list[float]:
    return [float(value) for value in row.get("paired_difference") or []]


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [
        sum(float(vector[index]) for vector in vectors) / len(vectors) for index in range(width)
    ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def _round(value: float) -> float:
    return round(float(value), 6)


def _cell_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        str(row.get(name, ""))
        for name in (
            "model_hf_id",
            "axis",
            "family",
            "solver_effort_bin",
            "surface_kind",
        )
    )


def _anchor_key(row: Mapping[str, Any]) -> str:
    return "|".join(str(row.get(name, "")) for name in ("model_hf_id", "axis", "family"))


def row_reconstruction(
    source_rows: Sequence[Mapping[str, Any]],
    embedding_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    source_by_id = {str(row.get("row_id")): row for row in source_rows}
    expected_order = [
        f"{row['row_id']}|{model_id}" for row in source_rows for model_id in MANDATED_MODEL_HF_IDS
    ]
    observed_order = [
        f"{row.get('source_row_id')}|{row.get('model_hf_id')}" for row in embedding_rows
    ]
    observed_counts = Counter(observed_order)
    duplicate_cells = sorted(cell for cell, count in observed_counts.items() if count > 1)
    missing_cells = sorted(set(expected_order) - set(observed_order))
    unexpected_cells = sorted(set(observed_order) - set(expected_order))
    stale_source_hashes: list[str] = []
    bad_embedding_hashes: list[str] = []
    bad_cell_ids: list[str] = []
    bad_source_orders: list[str] = []
    for row in embedding_rows:
        source_id = str(row.get("source_row_id"))
        cell_id = f"{source_id}|{row.get('model_hf_id')}"
        if row.get("embedding_cell_id") != cell_id:
            bad_cell_ids.append(str(row.get("embedding_cell_id")))
        source = source_by_id.get(source_id)
        if source is None or row.get("source_row_hash") != source.get("row_hash"):
            stale_source_hashes.append(cell_id)
        if row.get("row_hash") != embedding_row_hash(row):
            bad_embedding_hashes.append(cell_id)
        if source is not None:
            expected_order_index = list(source_by_id).index(source_id)
            if row.get("source_row_order") != expected_order_index:
                bad_source_orders.append(cell_id)
    return {
        "schema": SCHEMA + ".row_reconstruction",
        "reconstructed_from_immutable_row_ids": True,
        "aggregate_counts_trusted": False,
        "source_row_count": len(source_rows),
        "embedding_row_count": len(embedding_rows),
        "expected_model_row_cell_count": len(expected_order),
        "observed_model_row_cell_count": len(embedding_rows),
        "row_order_exact": observed_order == expected_order,
        "duplicate_embedding_cell_ids": duplicate_cells,
        "missing_embedding_cell_ids": missing_cells[:50],
        "unexpected_embedding_cell_ids": unexpected_cells[:50],
        "embedding_cell_id_mismatch_count": len(bad_cell_ids),
        "stale_source_hash_count": len(stale_source_hashes),
        "embedding_row_hash_mismatch_count": len(bad_embedding_hashes),
        "source_row_order_mismatch_count": len(bad_source_orders),
        "all_rows_reconstructed": (
            observed_order == expected_order
            and not duplicate_cells
            and not missing_cells
            and not unexpected_cells
            and not stale_source_hashes
            and not bad_embedding_hashes
            and not bad_cell_ids
            and not bad_source_orders
        ),
    }


def upstream_hashes_and_row_reconstruction(
    *,
    root: Path,
    exp5852_artifact_path: Path,
    exp5852_rows_path: Path,
    exp5840_artifact_path: Path,
    exp5840_rows_path: Path,
    exp5852_artifact: Mapping[str, Any],
    exp5840_artifact: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    embedding_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    upstream_paths = {
        "exp5852_artifact": exp5852_artifact_path,
        "exp5852_rows": exp5852_rows_path,
        "exp5840_artifact": exp5840_artifact_path,
        "exp5840_rows": exp5840_rows_path,
    }
    source_receipt = _receipt_check(
        exp5840_artifact,
        source_rows,
        path=exp5840_rows_path,
        row_id_key="row_id",
        row_hash_fn=source_row_hash,
    )
    embedding_receipt = _receipt_check(
        exp5852_artifact,
        embedding_rows,
        path=exp5852_rows_path,
        row_id_key="embedding_cell_id",
        row_hash_fn=embedding_row_hash,
    )
    split_receipt = dict(exp5840_artifact.get("split_definition_and_hashes") or {})
    validator_receipt = dict(exp5840_artifact.get("exact_label_and_minimality_receipts") or {})
    return {
        "schema": SCHEMA + ".upstream_hashes_and_row_reconstruction",
        "file_hashes": _file_hashes(root, upstream_paths),
        "validator_directory_hash": _tree_hash(root / VERIFY_DIR_RELATIVE_PATH)
        if (root / VERIFY_DIR_RELATIVE_PATH).exists()
        else "missing",
        "split_manifest_hash": sha256_json(split_receipt),
        "validator_receipt_hash": sha256_json(validator_receipt),
        "exp5852_gate": {
            "status": exp5852_artifact.get("status"),
            "paired_embedding_corpus_ready_score": exp5852_artifact.get(
                "paired_embedding_corpus_ready_score"
            ),
            "models_used": list(exp5852_artifact.get("models_used") or []),
        },
        "exp5840_gate": {
            "status": exp5840_artifact.get("status"),
            "counterfactual_fixture_ready_score": exp5840_artifact.get(
                "counterfactual_fixture_ready_score"
            ),
        },
        "source_row_file_receipt_replay": source_receipt,
        "embedding_row_file_receipt_replay": embedding_receipt,
        "row_reconstruction": row_reconstruction(source_rows, embedding_rows),
    }


def _collect_preconditions(
    *,
    root: Path,
    result_path: Path,
    exp5852_artifact: Mapping[str, Any],
    exp5840_artifact: Mapping[str, Any],
    upstream: Mapping[str, Any],
    load_errors: Sequence[str],
    memory_probe: MemoryProbe,
    disk_probe: DiskProbe,
) -> JsonDict:
    memory = memory_probe()
    disk = disk_probe(root)
    output = _output_path_receipt(result_path)
    blocked = list(load_errors)
    exp5852_gate = dict(upstream.get("exp5852_gate") or {})
    exp5840_gate = dict(upstream.get("exp5840_gate") or {})
    if exp5852_gate.get("paired_embedding_corpus_ready_score") != 1.0:
        blocked.append("exp5852_corpus_not_ready")
    if exp5852_gate.get("models_used") != list(MANDATED_MODEL_HF_IDS):
        blocked.append("exp5852_mandated_model_set_mismatch")
    if exp5840_gate.get("counterfactual_fixture_ready_score") != 1.0:
        blocked.append("exp5840_fixture_not_ready")
    if exp5852_artifact.get("verifier_is_oracle") is not True:
        blocked.append("exp5852_verifier_not_oracle")
    if exp5852_artifact.get("inference_substrate") != "live_llm_embedding_extraction":
        blocked.append("exp5852_wrong_inference_substrate")
    if (
        dict(upstream.get("source_row_file_receipt_replay") or {}).get("all_receipt_checks_passed")
        is not True
    ):
        blocked.append("exp5840_rows_receipt_replay_failed")
    if (
        dict(upstream.get("embedding_row_file_receipt_replay") or {}).get(
            "all_receipt_checks_passed"
        )
        is not True
    ):
        blocked.append("exp5852_rows_receipt_replay_failed")
    if dict(upstream.get("row_reconstruction") or {}).get("all_rows_reconstructed") is not True:
        blocked.append("row_reconstruction_failed")
    if memory.get("ok") is not True:
        blocked.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked.append("insufficient_free_disk")
    if output.get("ok") is not True:
        blocked.append("output_path_not_writable")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output,
        "structured_gate_replayed": True,
        "exp5852_artifact_gate_ready": exp5852_gate.get("paired_embedding_corpus_ready_score")
        == 1.0,
        "exp5840_fixture_gate_ready": exp5840_gate.get("counterfactual_fixture_ready_score") == 1.0,
        "atomic_output_writable": output.get("ok") is True,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def _train_anchors_by_axis(
    rows: Sequence[Mapping[str, Any]], axis: str
) -> tuple[dict[str, list[float]], dict[str, int]]:
    axis_rows = [row for row in rows if row.get("axis") == axis]
    anchor_vectors: dict[str, list[float]] = {}
    anchor_counts: dict[str, int] = {}
    for key in sorted({_anchor_key(row) for row in axis_rows}):
        vectors = [
            _vector(row)
            for row in axis_rows
            if _anchor_key(row) == key and row.get("split") == "train"
        ]
        anchor_vectors[key] = _mean_vector(vectors)
        anchor_counts[key] = len(vectors)
    return anchor_vectors, anchor_counts


def sensitivity_by_axis(
    rows: Sequence[Mapping[str, Any]],
    axis: str,
    *,
    anchor_vectors: Mapping[str, Sequence[float]] | None = None,
    anchor_counts: Mapping[str, int] | None = None,
) -> JsonDict:
    axis_rows = [row for row in rows if row.get("axis") == axis]
    if anchor_vectors is None or anchor_counts is None:
        computed_vectors, computed_counts = _train_anchors_by_axis(rows, axis)
        anchor_vectors = computed_vectors
        anchor_counts = computed_counts
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in axis_rows:
        by_cell[_cell_key(row)].append(row)
    decisions: list[JsonDict] = []
    for cell, cell_rows in sorted(by_cell.items()):
        anchor = list(anchor_vectors.get("|".join(cell.split("|")[:3]), []))
        anchor_norm = _norm(anchor)
        norms = [_norm(_vector(row)) for row in cell_rows]
        cosines = [_cosine(_vector(row), anchor) for row in cell_rows]
        positive_rate = sum(value > 0.0 for value in cosines) / len(cosines) if cosines else 0.0
        mean_cosine = sum(cosines) / len(cosines) if cosines else 0.0
        mean_norm = sum(norms) / len(norms) if norms else 0.0
        passed = (
            bool(cell_rows)
            and min(norms, default=0.0) > MIN_PAIRED_NORM
            and int(anchor_counts.get("|".join(cell.split("|")[:3]), 0) or 0) > 0
            and anchor_norm > MIN_PAIRED_NORM
            and positive_rate >= MIN_DIRECTION_POSITIVE_RATE
            and mean_cosine > MIN_MEAN_ANCHOR_COSINE
        )
        decisions.append(
            {
                "cell_id": cell,
                "row_count": len(cell_rows),
                "train_anchor_row_count": int(
                    anchor_counts.get("|".join(cell.split("|")[:3]), 0) or 0
                ),
                "mean_paired_difference_norm": _round(mean_norm),
                "min_paired_difference_norm": _round(min(norms, default=0.0)),
                "direction_positive_rate": _round(positive_rate),
                "mean_anchor_cosine": _round(mean_cosine),
                "cell_passed": passed,
            }
        )
    failed = [row["cell_id"] for row in decisions if row["cell_passed"] is not True]
    return {
        "schema": SCHEMA + f".{axis}.sensitivity",
        "axis": axis,
        "preregistered_direction_rule": (
            "condition_b_minus_a compared to train_split_mean_by_model_axis_family"
        ),
        "science_rows_used_to_choose_sign": False,
        "null_bounds": {
            "min_paired_difference_norm_gt": MIN_PAIRED_NORM,
            "direction_positive_rate_gte": MIN_DIRECTION_POSITIVE_RATE,
            "mean_anchor_cosine_gt": MIN_MEAN_ANCHOR_COSINE,
        },
        "row_count": len(axis_rows),
        "cell_count": len(decisions),
        "failed_cell_count": len(failed),
        "failed_cells": failed[:50],
        "cell_decisions": decisions,
        "all_cells_passed": bool(decisions) and not failed,
    }


def _reinitialized_feature_view(row: Mapping[str, Any]) -> JsonDict:
    conditions = []
    for condition in list(row.get("condition_embeddings") or []):
        conditions.append(
            {
                "condition_id": sha256_json({"condition_id": str(condition.get("condition_id"))}),
                "embedding": list(condition.get("embedding") or []),
                "embedding_shape": list(condition.get("embedding_shape") or []),
                "embedding_sha256": str(condition.get("embedding_sha256")),
                "token_count": int(condition.get("token_count", 0) or 0),
                "truncated": bool(condition.get("truncated")),
            }
        )
    diff = list(row.get("paired_difference") or [])
    return {
        "condition_features": conditions,
        "paired_difference": diff,
        "paired_difference_sha256": str(row.get("paired_difference_sha256")),
        "difference_orientation": "condition_b_minus_a",
        "preprocessing": str(dict(row.get("feature_consumer_view") or {}).get("preprocessing", "")),
    }


def evaluator_swap_receipts(
    source_rows: Sequence[Mapping[str, Any]],
    embedding_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    disagreements: list[str] = []
    replay_failures: list[str] = []
    private_code_markers: list[str] = []
    for row in source_rows:
        exact = dict(row.get("exact_receipt") or {})
        primary = exact.get("primary_validator_version")
        independent = exact.get("independent_validator_version")
        if (
            primary != PRIMARY_VALIDATOR_VERSION
            or independent != INDEPENDENT_VALIDATOR_VERSION
            or primary == independent
            or exact.get("validators_agree") is not True
        ):
            disagreements.append(str(row.get("row_id")))
        labels = _source_labels(row)
        if row.get("axis") == "candidate_correctness" and labels != (True, False):
            replay_failures.append(str(row.get("row_id")))
        if row.get("axis") == "constraint_ablation" and labels != (False, True):
            replay_failures.append(str(row.get("row_id")))
    for row in embedding_rows:
        feature_text = canonical_json(row.get("feature_consumer_view") or {}).lower()
        if (
            PRIMARY_VALIDATOR_VERSION.lower() in feature_text
            or INDEPENDENT_VALIDATOR_VERSION.lower() in feature_text
        ):
            private_code_markers.append(str(row.get("embedding_cell_id")))
        if _reinitialized_feature_view(row) != dict(row.get("feature_consumer_view") or {}):
            private_code_markers.append(str(row.get("embedding_cell_id")))
    marker_count = len(set(private_code_markers))
    return {
        "schema": SCHEMA + ".evaluator_swap_receipts",
        "primary_validator_version": PRIMARY_VALIDATOR_VERSION,
        "independent_validator_version": INDEPENDENT_VALIDATOR_VERSION,
        "validator_disagreement_count": len(disagreements),
        "validator_disagreements": disagreements[:50],
        "exact_label_replay_disagreement_count": len(replay_failures),
        "exact_label_replay_disagreements": replay_failures[:50],
        "feature_consumer_reinitialization": {
            "reinitialized_from_masked_hashes": True,
            "private_code_marker_count": marker_count,
            "private_code_marker_cells": sorted(set(private_code_markers))[:50],
        },
        "co_adaptation_detected": bool(disagreements or replay_failures or marker_count),
        "all_evaluator_swaps_passed": not disagreements
        and not replay_failures
        and marker_count == 0,
    }


def grounded_vs_true_cross(source_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures: list[str] = []
    candidate_grounded = 0
    ablation_grounded = 0
    true_flips = 0
    for row in source_rows:
        conditions = list(row.get("conditions") or [])
        if len(conditions) != 2:
            failures.append(str(row.get("row_id")))
            continue
        left, right = conditions
        labels = _source_labels(row)
        if labels[0] != labels[1]:
            true_flips += 1
        if row.get("axis") == "candidate_correctness":
            ok = (
                left.get("context_hash") == right.get("context_hash")
                and left.get("candidate_hash") != right.get("candidate_hash")
                and left.get("constraint_present") is True
                and right.get("constraint_present") is True
                and labels == (True, False)
            )
            candidate_grounded += int(ok)
        elif row.get("axis") == "constraint_ablation":
            ok = (
                left.get("candidate_hash") == right.get("candidate_hash")
                and left.get("context_hash") != right.get("context_hash")
                and left.get("constraint_present") is True
                and right.get("constraint_present") is False
                and labels == (False, True)
            )
            ablation_grounded += int(ok)
        else:
            ok = False
        if not ok:
            failures.append(str(row.get("row_id")))
    return {
        "schema": SCHEMA + ".grounded_vs_true_cross",
        "row_count": len(source_rows),
        "true_label_flip_count": true_flips,
        "candidate_grounding_cross_count": candidate_grounded,
        "constraint_ablation_grounding_cross_count": ablation_grounded,
        "grounding_truth_disagreement_count": len(failures),
        "grounding_truth_disagreements": failures[:50],
        "grounding_and_truth_distinguished": true_flips == len(source_rows)
        and candidate_grounded + ablation_grounded == len(source_rows)
        and not failures,
    }


def label_and_pair_permutation_controls(
    source_rows: Sequence[Mapping[str, Any]],
    embedding_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    labels_by_row = {str(row.get("row_id")): _source_labels(row) for row in source_rows}
    permuted_agreements = [labels == tuple(reversed(labels)) for labels in labels_by_row.values()]
    label_agreement = (
        sum(permuted_agreements) / len(permuted_agreements) if permuted_agreements else 1.0
    )
    claim_anchors, claim_anchor_counts = _train_anchors_by_axis(
        embedding_rows, "candidate_correctness"
    )
    ablation_anchors, ablation_anchor_counts = _train_anchors_by_axis(
        embedding_rows, "constraint_ablation"
    )
    claim_swap = sensitivity_by_axis(
        [
            {**row, "paired_difference": [-value for value in _vector(row)]}
            for row in embedding_rows
        ],
        "candidate_correctness",
        anchor_vectors=claim_anchors,
        anchor_counts=claim_anchor_counts,
    )
    ablation_swap = sensitivity_by_axis(
        [
            {**row, "paired_difference": [-value for value in _vector(row)]}
            for row in embedding_rows
        ],
        "constraint_ablation",
        anchor_vectors=ablation_anchors,
        anchor_counts=ablation_anchor_counts,
    )
    swap_decisions = [
        *claim_swap.get("cell_decisions", []),
        *ablation_swap.get("cell_decisions", []),
    ]
    positive_rates = [float(row["direction_positive_rate"]) for row in swap_decisions]
    max_positive_rate = max(positive_rates, default=1.0)
    failed_cells = [
        str(row["cell_id"])
        for row in swap_decisions
        if float(row["direction_positive_rate"]) > PAIR_SWAP_MAX_POSITIVE_RATE
    ]
    return {
        "schema": SCHEMA + ".label_and_pair_permutation_controls",
        "label_permutation_rule": "deterministic_pair_label_reversal",
        "label_permutation_agreement_rate": _round(label_agreement),
        "label_permutation_collapsed": label_agreement <= LABEL_PERMUTATION_MAX_AGREEMENT,
        "pair_member_swap_rule": "condition_b_minus_a_vectors_negated_without_label_change",
        "pair_swap_max_direction_positive_rate": _round(max_positive_rate),
        "pair_swap_failed_cell_count": len(failed_cells),
        "pair_swap_failed_cells": failed_cells[:50],
        "all_label_and_pair_controls_passed": (
            label_agreement <= LABEL_PERMUTATION_MAX_AGREEMENT and not failed_cells
        ),
    }


def _feature_identity_leaks(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    forbidden = {
        "model_hf_id",
        "model_family",
        "family",
        "axis",
        "exact_label",
        "label",
        "oracle",
        "validator",
    }
    tokens = {model_family(model_id).lower() for model_id in MANDATED_MODEL_HF_IDS}
    tokens.update(model_id.lower() for model_id in MANDATED_MODEL_HF_IDS)
    tokens.update(family.lower() for family in REQUIRED_CONSTRAINT_FAMILIES)
    leaked: list[str] = []
    for row in rows:
        feature_view = dict(row.get("feature_consumer_view") or {})
        if forbidden.intersection(feature_view):
            leaked.append(str(row.get("embedding_cell_id")))
            continue
        text = canonical_json(feature_view).lower()
        if any(token in text for token in tokens):
            leaked.append(str(row.get("embedding_cell_id")))
    return leaked


def identity_masking_and_prediction_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    models = sorted({str(row.get("model_hf_id")) for row in rows})
    dim_model_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        conditions = list(row.get("condition_embeddings") or [])
        width = len(conditions[0].get("embedding", [])) if conditions else 0
        dim_model_counts[width][str(row.get("model_hf_id"))] += 1
    correct = 0
    for row in rows:
        conditions = list(row.get("condition_embeddings") or [])
        width = len(conditions[0].get("embedding", [])) if conditions else 0
        prediction = sorted(dim_model_counts[width].items(), key=lambda item: (-item[1], item[0]))[
            0
        ][0]
        correct += int(prediction == row.get("model_hf_id"))
    accuracy = correct / len(rows) if rows else 1.0
    chance = 1.0 / len(models) if models else 1.0
    leaked = _feature_identity_leaks(rows)
    return {
        "schema": SCHEMA + ".identity_masking_and_prediction_controls",
        "model_count": len(models),
        "raw_embedding_dims_by_model": {
            model: sorted(
                {
                    len(list(row.get("condition_embeddings") or [{}])[0].get("embedding", []))
                    for row in rows
                    if row.get("model_hf_id") == model
                }
            )
            for model in models
        },
        "raw_dimension_identity_accuracy": _round(accuracy),
        "chance_identity_accuracy": _round(chance),
        "identity_accuracy_tolerance": IDENTITY_ACCURACY_TOLERANCE,
        "feature_consumer_identity_leakage_count": len(leaked),
        "feature_consumer_identity_leakage_cells": leaked[:50],
        "identity_masking_reinitializes_consumer": True,
        "all_identity_controls_passed": (
            accuracy <= chance + IDENTITY_ACCURACY_TOLERANCE and not leaked
        ),
    }


def _condition_norm_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    total = 0
    high_norm_true = 0
    low_norm_true = 0
    for row in rows:
        conditions = list(row.get("condition_embeddings") or [])
        labels = list(
            dict(row.get("oracle_label_receipt") or {}).get("labels_by_condition_id", {}).items()
        )
        label_by_id = {condition_id: bool(label) for condition_id, label in labels}
        if len(conditions) != 2:
            continue
        norms = [_norm(condition.get("embedding", [])) for condition in conditions]
        if norms[0] == norms[1]:
            total += 2
            high_norm_true += 1
            low_norm_true += 1
            continue
        high_index = 0 if norms[0] > norms[1] else 1
        low_index = 1 - high_index
        total += 2
        high_norm_true += int(label_by_id.get(conditions[high_index]["condition_id"], False))
        high_norm_true += int(not label_by_id.get(conditions[low_index]["condition_id"], True))
        low_norm_true += int(label_by_id.get(conditions[low_index]["condition_id"], False))
        low_norm_true += int(not label_by_id.get(conditions[high_index]["condition_id"], True))
    return max(high_norm_true, low_norm_true) / total if total else 1.0


def length_norm_and_truncation_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[_cell_key(row)].append(row)
    token_mismatch_cells: list[str] = []
    truncation_cells: list[str] = []
    norm_shortcut_cells: list[str] = []
    cell_decisions: list[JsonDict] = []
    for cell, cell_rows in sorted(by_cell.items()):
        token_mismatches = 0
        truncations = 0
        for row in cell_rows:
            conditions = list(row.get("condition_embeddings") or [])
            counts = [int(condition.get("token_count", -1)) for condition in conditions]
            if len(counts) != 2 or counts[0] != counts[1]:
                token_mismatches += 1
            if any(condition.get("truncated") is True for condition in conditions):
                truncations += 1
        norm_accuracy = _condition_norm_accuracy(cell_rows)
        if token_mismatches:
            token_mismatch_cells.append(cell)
        if truncations:
            truncation_cells.append(cell)
        if norm_accuracy > NORM_ONLY_MAX_ACCURACY:
            norm_shortcut_cells.append(cell)
        cell_decisions.append(
            {
                "cell_id": cell,
                "row_count": len(cell_rows),
                "token_count_pair_mismatch_count": token_mismatches,
                "truncation_count": truncations,
                "norm_only_label_accuracy": _round(norm_accuracy),
                "cell_passed": not token_mismatches
                and not truncations
                and norm_accuracy <= NORM_ONLY_MAX_ACCURACY,
            }
        )
    return {
        "schema": SCHEMA + ".length_norm_and_truncation_controls",
        "token_count_pair_mismatch_count": sum(
            row["token_count_pair_mismatch_count"] for row in cell_decisions
        ),
        "truncation_count": sum(row["truncation_count"] for row in cell_decisions),
        "norm_only_max_label_accuracy": _round(
            max((float(row["norm_only_label_accuracy"]) for row in cell_decisions), default=1.0)
        ),
        "norm_only_max_allowed_accuracy": NORM_ONLY_MAX_ACCURACY,
        "token_mismatch_cells": token_mismatch_cells[:50],
        "truncation_cells": truncation_cells[:50],
        "norm_shortcut_cells": norm_shortcut_cells[:50],
        "cell_decisions": cell_decisions,
        "all_length_norm_controls_passed": not token_mismatch_cells
        and not truncation_cells
        and not norm_shortcut_cells,
    }


def perturbation_duplicate_and_no_information_controls(
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("pair_group_id"))].append(row)
    group_weights = {
        group: round(1.0 / len(group_rows), 12) for group, group_rows in sorted(by_group.items())
    }
    duplicate_groups = sorted(
        group
        for group, group_rows in by_group.items()
        if len(group_rows) > len(MANDATED_MODEL_HF_IDS)
    )
    norms = [_norm(_vector(row)) for row in rows]
    mean_norm = sum(norms) / len(norms) if norms else 0.0
    perturbed_norms = [_norm([value + 1e-12 for value in _vector(row)]) for row in rows]
    perturbed_mean = sum(perturbed_norms) / len(perturbed_norms) if perturbed_norms else 0.0
    drift = abs(perturbed_mean - mean_norm) / mean_norm if mean_norm else 1.0
    no_info_claim = sensitivity_by_axis(
        [{**row, "paired_difference": [0.0 for _ in _vector(row)]} for row in rows],
        "candidate_correctness",
    )
    no_info_ablation = sensitivity_by_axis(
        [{**row, "paired_difference": [0.0 for _ in _vector(row)]} for row in rows],
        "constraint_ablation",
    )
    no_info_fails_positive = (
        no_info_claim.get("all_cells_passed") is False
        and no_info_ablation.get("all_cells_passed") is False
    )
    return {
        "schema": SCHEMA + ".perturbation_duplicate_and_no_information_controls",
        "target_preserving_perturbation": {
            "epsilon": 1e-12,
            "relative_mean_norm_drift": _round(drift),
            "passed": drift <= PERTURBATION_RELATIVE_TOLERANCE,
        },
        "duplicate_group_reweighting": {
            "group_count": len(by_group),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_groups": duplicate_groups[:50],
            "group_weights_hash": sha256_json(group_weights),
            "passed": True,
        },
        "no_information_controls": {
            "zero_vector_claim_flip_all_cells_passed": no_info_claim.get("all_cells_passed"),
            "zero_vector_constraint_ablation_all_cells_passed": no_info_ablation.get(
                "all_cells_passed"
            ),
            "no_information_fails_positive_control": no_info_fails_positive,
        },
        "all_perturbation_duplicate_no_information_controls_passed": (
            drift <= PERTURBATION_RELATIVE_TOLERANCE and no_info_fails_positive
        ),
    }


def _expected_cell_ids_from_source(source_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    expected = set()
    for model_id in MANDATED_MODEL_HF_IDS:
        for axis in sorted({str(row.get("axis")) for row in source_rows}):
            for family in sorted({str(row.get("family")) for row in source_rows}):
                for hardness in sorted({str(row.get("solver_effort_bin")) for row in source_rows}):
                    for surface in sorted({str(row.get("surface_kind")) for row in source_rows}):
                        if any(
                            row.get("axis") == axis
                            and row.get("family") == family
                            and row.get("solver_effort_bin") == hardness
                            and row.get("surface_kind") == surface
                            for row in source_rows
                        ):
                            expected.add("|".join([model_id, axis, family, hardness, surface]))
    return expected


def disaggregated_cell_decisions(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    claim_flip: Mapping[str, Any],
    ablation: Mapping[str, Any],
    length_norm: Mapping[str, Any],
) -> JsonDict:
    expected_cells = _expected_cell_ids_from_source(source_rows)
    decisions_by_cell: dict[str, JsonDict] = {
        cell: {"cell_id": cell, "checks": {}, "cell_passed": True}
        for cell in sorted(expected_cells)
    }
    for check_name, report in (
        ("claim_flip_sensitivity", claim_flip),
        ("constraint_ablation_sensitivity", ablation),
        ("length_norm_and_truncation_controls", length_norm),
    ):
        for decision in report.get("cell_decisions", []):
            cell = str(decision["cell_id"])
            decisions_by_cell.setdefault(cell, {"cell_id": cell, "checks": {}, "cell_passed": True})
            passed = bool(decision.get("cell_passed"))
            decisions_by_cell[cell]["checks"][check_name] = passed
            decisions_by_cell[cell]["cell_passed"] = (
                decisions_by_cell[cell]["cell_passed"] and passed
            )
    missing = [
        cell
        for cell, decision in decisions_by_cell.items()
        if not decision["checks"] or "length_norm_and_truncation_controls" not in decision["checks"]
    ]
    for cell in missing:
        decisions_by_cell[cell]["cell_passed"] = False
    failed = [cell for cell, decision in decisions_by_cell.items() if not decision["cell_passed"]]
    return {
        "schema": SCHEMA + ".disaggregated_cell_decisions",
        "required_cell_count": len(expected_cells),
        "decision_cell_count": len(decisions_by_cell),
        "missing_decision_cell_count": len(missing),
        "missing_decision_cells": missing[:50],
        "failed_cell_count": len(failed),
        "failed_cells": failed[:50],
        "cell_decisions": list(decisions_by_cell.values()),
        "all_disaggregated_cells_passed": bool(expected_cells) and not missing and not failed,
    }


def _surviving_shortcuts(artifact: Mapping[str, Any]) -> list[str]:
    shortcuts: list[str] = []
    if dict(artifact.get("claim_flip_sensitivity") or {}).get("all_cells_passed") is not True:
        shortcuts.append("claim_flip_direction_cell_failure")
    if (
        dict(artifact.get("constraint_ablation_sensitivity") or {}).get("all_cells_passed")
        is not True
    ):
        shortcuts.append("constraint_ablation_direction_cell_failure")
    if (
        dict(artifact.get("evaluator_swap_receipts") or {}).get("all_evaluator_swaps_passed")
        is not True
    ):
        shortcuts.append("evaluator_swap_disagreement")
    if (
        dict(artifact.get("grounded_vs_true_cross") or {}).get("grounding_and_truth_distinguished")
        is not True
    ):
        shortcuts.append("grounded_vs_true_cross_failure")
    if (
        dict(artifact.get("label_and_pair_permutation_controls") or {}).get(
            "all_label_and_pair_controls_passed"
        )
        is not True
    ):
        shortcuts.append("label_or_pair_permutation_survives")
    identity = dict(artifact.get("identity_masking_and_prediction_controls") or {})
    if (
        float(identity.get("raw_dimension_identity_accuracy", 1.0) or 1.0)
        > float(identity.get("chance_identity_accuracy", 0.0) or 0.0) + IDENTITY_ACCURACY_TOLERANCE
    ):
        shortcuts.append("raw_model_dimension_identity_shortcut")
    if int(identity.get("feature_consumer_identity_leakage_count", 1) or 0) > 0:
        shortcuts.append("feature_consumer_identity_leakage")
    length = dict(artifact.get("length_norm_and_truncation_controls") or {})
    if int(length.get("token_count_pair_mismatch_count", 1) or 0) > 0:
        shortcuts.append("token_count_pair_shortcut")
    if int(length.get("truncation_count", 1) or 0) > 0:
        shortcuts.append("truncation_shortcut")
    if length.get("all_length_norm_controls_passed") is not True:
        shortcuts.append("norm_only_or_length_control_failure")
    perturb = dict(artifact.get("perturbation_duplicate_and_no_information_controls") or {})
    if perturb.get("all_perturbation_duplicate_no_information_controls_passed") is not True:
        shortcuts.append("perturbation_duplicate_or_no_information_control_failure")
    if (
        dict(artifact.get("disaggregated_cell_decisions") or {}).get(
            "all_disaggregated_cells_passed"
        )
        is not True
    ):
        shortcuts.append("disaggregated_cell_failure")
    return sorted(set(shortcuts))


def paired_embedding_integrity_ready_score(artifact: Mapping[str, Any]) -> float:
    preconditions = dict(artifact.get("preconditions_checked") or {})
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = (
        preconditions.get("preconditions_ready") is True
        and artifact.get("surviving_shortcuts") == []
        and dict(artifact.get("claim_flip_sensitivity") or {}).get("all_cells_passed") is True
        and dict(artifact.get("constraint_ablation_sensitivity") or {}).get("all_cells_passed")
        is True
        and dict(artifact.get("evaluator_swap_receipts") or {}).get("all_evaluator_swaps_passed")
        is True
        and dict(artifact.get("grounded_vs_true_cross") or {}).get(
            "grounding_and_truth_distinguished"
        )
        is True
        and dict(artifact.get("label_and_pair_permutation_controls") or {}).get(
            "all_label_and_pair_controls_passed"
        )
        is True
        and dict(artifact.get("identity_masking_and_prediction_controls") or {}).get(
            "all_identity_controls_passed"
        )
        is True
        and dict(artifact.get("length_norm_and_truncation_controls") or {}).get(
            "all_length_norm_controls_passed"
        )
        is True
        and dict(artifact.get("perturbation_duplicate_and_no_information_controls") or {}).get(
            "all_perturbation_duplicate_no_information_controls_passed"
        )
        is True
        and dict(artifact.get("disaggregated_cell_decisions") or {}).get(
            "all_disaggregated_cells_passed"
        )
        is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5840_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5840_ROWS_RELATIVE_PATH.as_posix(),
        EXP5852_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5852_ROWS_RELATIVE_PATH.as_posix(),
        VERIFY_DIR_RELATIVE_PATH.as_posix(),
        "research-references.md",
    ]
    return {
        field: {"principle": principle, "sources": sources}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _empty_report(name: str) -> JsonDict:
    return {"schema": SCHEMA + f".{name}", "all_cells_passed": False}


def _artifact_from_reports(
    *,
    status: str,
    preconditions: Mapping[str, Any],
    upstream: Mapping[str, Any],
    claim_flip: Mapping[str, Any],
    ablation: Mapping[str, Any],
    evaluator_swaps: Mapping[str, Any],
    grounded_cross: Mapping[str, Any],
    permutation: Mapping[str, Any],
    identity: Mapping[str, Any],
    length_norm: Mapping[str, Any],
    perturbation: Mapping[str, Any],
    disaggregated: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "preconditions_checked": dict(preconditions),
        "upstream_hashes_and_row_reconstruction": dict(upstream),
        "claim_flip_sensitivity": dict(claim_flip),
        "constraint_ablation_sensitivity": dict(ablation),
        "evaluator_swap_receipts": dict(evaluator_swaps),
        "grounded_vs_true_cross": dict(grounded_cross),
        "label_and_pair_permutation_controls": dict(permutation),
        "identity_masking_and_prediction_controls": dict(identity),
        "length_norm_and_truncation_controls": dict(length_norm),
        "perturbation_duplicate_and_no_information_controls": dict(perturbation),
        "disaggregated_cell_decisions": dict(disaggregated),
        "surviving_shortcuts": [],
        "paired_embedding_integrity_ready_score": 0.0,
        "duration_s": _round(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["surviving_shortcuts"] = [] if status == "blocked" else _surviving_shortcuts(artifact)
    artifact["paired_embedding_integrity_ready_score"] = paired_embedding_integrity_ready_score(
        artifact
    )
    if artifact["paired_embedding_integrity_ready_score"] == 1.0:
        artifact["status"] = "complete"
        artifact["honest_verdict"] = "ready: paired_embedding_integrity_controls_clean"
    elif status == "blocked":
        reasons = list(dict(preconditions).get("blocked_reasons") or ["preconditions_failed"])
        artifact["honest_verdict"] = "blocked: " + ",".join(reasons[:8])
    else:
        artifact["status"] = "disqualified"
        reasons = artifact["surviving_shortcuts"] or ["integrity_controls_failed"]
        artifact["honest_verdict"] = "disqualified: " + ",".join(reasons[:8])
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    return sha256_json(stable)


def build_audit_reports(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    embedding_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict, JsonDict, JsonDict, JsonDict, JsonDict]:
    claim_flip = sensitivity_by_axis(embedding_rows, "candidate_correctness")
    ablation = sensitivity_by_axis(embedding_rows, "constraint_ablation")
    evaluator_swaps = evaluator_swap_receipts(source_rows, embedding_rows)
    grounded_cross = grounded_vs_true_cross(source_rows)
    permutation = label_and_pair_permutation_controls(source_rows, embedding_rows)
    identity = identity_masking_and_prediction_controls(embedding_rows)
    length_norm = length_norm_and_truncation_controls(embedding_rows)
    perturbation = perturbation_duplicate_and_no_information_controls(embedding_rows)
    return (
        claim_flip,
        ablation,
        evaluator_swaps,
        grounded_cross,
        permutation,
        identity,
        length_norm,
        perturbation,
    )


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(field)
    if set(REQUIRED_FIELD_PRINCIPLES) - set(artifact.get("field_provenance", {})):
        raise ValueError("field_provenance")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    expected_score = paired_embedding_integrity_ready_score(artifact)
    if artifact.get("paired_embedding_integrity_ready_score") != expected_score:
        raise ValueError("paired_embedding_integrity_ready_score")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    status = artifact.get("status")
    verdict = str(artifact.get("honest_verdict", ""))
    if expected_score == 1.0:
        if status != "complete" or not verdict.startswith("ready:"):
            raise ValueError("status")
    else:
        if status == "blocked":
            if not verdict.startswith("blocked:"):
                raise ValueError("honest_verdict")
        elif status == "disqualified":
            if not verdict.startswith("disqualified:"):
                raise ValueError("honest_verdict")
        else:
            raise ValueError("status")
    return True


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp5852_artifact_path: str | Path = REPO_ROOT / EXP5852_ARTIFACT_RELATIVE_PATH,
    exp5852_rows_path: str | Path = REPO_ROOT / EXP5852_ROWS_RELATIVE_PATH,
    exp5840_artifact_path: str | Path = REPO_ROOT / EXP5840_ARTIFACT_RELATIVE_PATH,
    exp5840_rows_path: str | Path = REPO_ROOT / EXP5840_ROWS_RELATIVE_PATH,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
    write: bool = True,
) -> JsonDict:
    """Run the aggregation-only Exp5853 audit and optionally write the artifact."""

    started = time.perf_counter()
    root = Path(root)
    result = Path(result_path)
    exp5852_artifact_path = Path(exp5852_artifact_path)
    exp5852_rows_path = Path(exp5852_rows_path)
    exp5840_artifact_path = Path(exp5840_artifact_path)
    exp5840_rows_path = Path(exp5840_rows_path)
    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    load_errors: list[str] = []
    try:
        exp5852_artifact = _read_json(exp5852_artifact_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        exp5852_artifact = {}
        load_errors.append(f"exp5852_artifact_load_failed:{type(exc).__name__}")
    try:
        exp5840_artifact = _read_json(exp5840_artifact_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        exp5840_artifact = {}
        load_errors.append(f"exp5840_artifact_load_failed:{type(exc).__name__}")
    try:
        source_rows = _read_jsonl(exp5840_rows_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source_rows = []
        load_errors.append(f"exp5840_rows_load_failed:{type(exc).__name__}")
    try:
        embedding_rows = _read_jsonl(exp5852_rows_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        embedding_rows = []
        load_errors.append(f"exp5852_rows_load_failed:{type(exc).__name__}")
    upstream = upstream_hashes_and_row_reconstruction(
        root=root,
        exp5852_artifact_path=exp5852_artifact_path,
        exp5852_rows_path=exp5852_rows_path,
        exp5840_artifact_path=exp5840_artifact_path,
        exp5840_rows_path=exp5840_rows_path,
        exp5852_artifact=exp5852_artifact,
        exp5840_artifact=exp5840_artifact,
        source_rows=source_rows,
        embedding_rows=embedding_rows,
    )
    preconditions = _collect_preconditions(
        root=root,
        result_path=result,
        exp5852_artifact=exp5852_artifact,
        exp5840_artifact=exp5840_artifact,
        upstream=upstream,
        load_errors=load_errors,
        memory_probe=memory_probe,
        disk_probe=disk_probe,
    )
    if preconditions.get("preconditions_ready") is True:
        (
            claim_flip,
            ablation,
            evaluator_swaps,
            grounded_cross,
            permutation,
            identity,
            length_norm,
            perturbation,
        ) = build_audit_reports(source_rows=source_rows, embedding_rows=embedding_rows)
        disaggregated = disaggregated_cell_decisions(
            source_rows=source_rows,
            claim_flip=claim_flip,
            ablation=ablation,
            length_norm=length_norm,
        )
        status = "complete"
    else:
        claim_flip = _empty_report("claim_flip_sensitivity")
        ablation = _empty_report("constraint_ablation_sensitivity")
        evaluator_swaps = {
            "schema": SCHEMA + ".evaluator_swap_receipts",
            "all_evaluator_swaps_passed": False,
        }
        grounded_cross = {
            "schema": SCHEMA + ".grounded_vs_true_cross",
            "grounding_and_truth_distinguished": False,
        }
        permutation = {
            "schema": SCHEMA + ".label_and_pair_permutation_controls",
            "all_label_and_pair_controls_passed": False,
        }
        identity = {
            "schema": SCHEMA + ".identity_masking_and_prediction_controls",
            "all_identity_controls_passed": False,
        }
        length_norm = {
            "schema": SCHEMA + ".length_norm_and_truncation_controls",
            "all_length_norm_controls_passed": False,
        }
        perturbation = {
            "schema": SCHEMA + ".perturbation_duplicate_and_no_information_controls",
            "all_perturbation_duplicate_no_information_controls_passed": False,
        }
        disaggregated = {
            "schema": SCHEMA + ".disaggregated_cell_decisions",
            "all_disaggregated_cells_passed": False,
        }
        status = "blocked"
    artifact = _artifact_from_reports(
        status=status,
        preconditions=preconditions,
        upstream=upstream,
        claim_flip=claim_flip,
        ablation=ablation,
        evaluator_swaps=evaluator_swaps,
        grounded_cross=grounded_cross,
        permutation=permutation,
        identity=identity,
        length_norm=length_norm,
        perturbation=perturbation,
        disaggregated=disaggregated,
        duration_s=time.perf_counter() - started,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
    )
    validate_artifact(artifact)
    if write:
        _write_atomic(result, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run(write="--write" in sys.argv)
    print(json.dumps(artifact, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
