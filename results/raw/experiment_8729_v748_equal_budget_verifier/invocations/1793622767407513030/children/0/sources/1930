"""Exp5638 FR-11 gate schema corrigendum.

Spec refs: REQ-LEARN-5638,
SCENARIO-LEARN-5638-ORIGINAL-SHAPE,
SCENARIO-LEARN-5638-FAIL-CLOSED,
SCENARIO-LEARN-5638-DETERMINISTIC.

This module does not rerun Exp5628. It reads the immutable Exp5628 artifact as
bytes, verifies the exact source hash, and emits a separate downstream receipt
that exposes the existing `unsafe_false_accept_count.total` value as the scalar
integer expected by conductor gates. The raw structured evidence is kept intact
so the schema correction cannot silently discard per-arm safety evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5638_fr11_gate_schema_corrigendum.json")
SOURCE_ARTIFACT_RELATIVE_PATH = Path("results/experiment_5628_conformal_active_spline_kan_csl.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5638_fr11_gate_schema_corrigendum.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5638_fr11_gate_schema_corrigendum.py")

EXPECTED_SOURCE_ARTIFACT_SHA256 = (
    "sha256:241d7dfb5db9c5984c4d6353d3e6dc7ef64dece658bf6f4babeb247403307bea"
)
NORMALIZATION_JSON_PATH = "unsafe_false_accept_count.total"
INFERENCE_SUBSTRATE = "deterministic_artifact_schema_normalization"
SCHEMA = "carnot.experiment_5638.fr11_gate_schema_corrigendum.v1"
EXPERIMENT = 5638
EXPERIMENT_ID = "experiment_5638_fr11_gate_schema_corrigendum"
TASK_ID = "exp5638-fr11-gate-schema-corrigendum"
MILESTONE = "2026.07.509"
RUN_DATE = "20260714"
TERMINAL_PREFIXES = ("complete:", "blocked:")
FORBIDDEN_VERDICT_PHRASE = "independent FR-11 validation"

SPEC_REFS = (
    "REQ-LEARN-5638",
    "SCENARIO-LEARN-5638-ORIGINAL-SHAPE",
    "SCENARIO-LEARN-5638-FAIL-CLOSED",
    "SCENARIO-LEARN-5638-DETERMINISTIC",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "source_artifact_path",
    "source_artifact_sha256",
    "source_honest_verdict",
    "raw_unsafe_false_accept_count",
    "normalization_json_path",
    "unsafe_false_accept_count_total",
    "by_arm_sum",
    "by_arm_reconciliation_pass",
    "source_continuous_self_learning_ready",
    "scientific_recompute_performed",
    "source_artifact_modified",
    "gate_contract_ready_score",
    "inference_substrate",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "receipt fields explain why they exist",
    "source_artifact_path": "provenance is explicit",
    "source_artifact_sha256": "immutability is enforced",
    "source_honest_verdict": "upstream status is preserved",
    "raw_unsafe_false_accept_count": "structured evidence is not discarded",
    "normalization_json_path": "derivation is unambiguous",
    "unsafe_false_accept_count_total": "the conductor-safe value is a scalar integer",
    "by_arm_sum": "reconciliation is explicit",
    "by_arm_reconciliation_pass": "shape consistency is tested",
    "source_continuous_self_learning_ready": "no readiness value is invented",
    "scientific_recompute_performed": "this is not a rerun",
    "source_artifact_modified": "Exp5628 remains immutable",
    "gate_contract_ready_score": "downstream gating is mechanical",
    "inference_substrate": "no model inference occurred",
    "reproducibility_checksum": "receipt is stable",
    "honest_verdict": (
        "terminal status starts with complete: or blocked: without claiming independent FR-11"
    ),
    "source_hash_exact": "pre-registered source hash match is explicit",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest tests/python/test_experiment_5638_fr11_gate_schema_corrigendum.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5638_fr11_gate_schema_corrigendum.py -m pytest tests/python/test_experiment_5638_fr11_gate_schema_corrigendum.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5638_fr11_gate_schema_corrigendum.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5638_fr11_gate_schema_corrigendum.json",
)


def _resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths without hiding absolute test fixtures."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(root) / candidate


def _display_path(root: Path | str, path: Path | str) -> str:
    """Return a stable repository-relative path when the source lives in the repo."""

    root_path = Path(root).resolve()
    target = Path(path).resolve()
    try:
        return target.relative_to(root_path).as_posix()
    except ValueError:
        return target.as_posix()


def _reject_duplicate_object_pairs(pairs: Sequence[tuple[str, Any]]) -> JsonDict:
    """Parse JSON objects while treating duplicate keys as ambiguous evidence."""

    parsed: JsonDict = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON key: {key}")
        parsed[key] = value
    return parsed


def load_json_object_from_bytes(raw_bytes: bytes) -> JsonDict:
    """Decode one JSON object and reject duplicate keys before normalization."""

    try:
        parsed = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON source artifact: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("source artifact must be a JSON object")
    return parsed


def _strict_non_negative_int(value: Any, path: str) -> int:
    """Accept only real JSON integers, excluding Python's bool-as-int subclass."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be a non-boolean integer")
    if value < 0:
        raise ValueError(f"{path} must be non-negative")
    return value


def normalize_unsafe_false_accept_count(source_artifact: Mapping[str, Any]) -> JsonDict:
    """Normalize only `unsafe_false_accept_count.total` and reconcile by-arm evidence."""

    raw_count = source_artifact.get("unsafe_false_accept_count")
    if not isinstance(raw_count, Mapping):
        raise ValueError("unsafe_false_accept_count must be an object")
    if "total" not in raw_count:
        raise ValueError("unsafe_false_accept_count.total is missing")
    if "by_arm" not in raw_count:
        raise ValueError("unsafe_false_accept_count.by_arm is missing")

    total = _strict_non_negative_int(raw_count["total"], "unsafe_false_accept_count.total")
    by_arm = raw_count["by_arm"]
    if not isinstance(by_arm, Mapping) or not by_arm:
        raise ValueError("unsafe_false_accept_count.by_arm must be a non-empty object")

    by_arm_sum = 0
    for arm, arm_value in by_arm.items():
        by_arm_sum += _strict_non_negative_int(
            arm_value,
            f"unsafe_false_accept_count.by_arm.{arm}",
        )
    if by_arm_sum != total:
        raise ValueError(
            f"by_arm_sum {by_arm_sum} does not equal unsafe_false_accept_count.total {total}"
        )
    return {
        "raw_unsafe_false_accept_count": dict(raw_count),
        "unsafe_false_accept_count_total": total,
        "by_arm_sum": by_arm_sum,
        "by_arm_reconciliation_pass": True,
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    source_path: Path | str = SOURCE_ARTIFACT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
) -> JsonDict:
    """Build the hash-bound scalar gate contract without modifying Exp5628."""

    root_path = Path(root)
    resolved_source_path = _resolve_path(root_path, source_path)
    source_bytes = resolved_source_path.read_bytes()
    source_sha256 = sha256_bytes(source_bytes)
    source_artifact = load_json_object_from_bytes(source_bytes)
    normalization = normalize_unsafe_false_accept_count(source_artifact)
    source_artifact_modified = resolved_source_path.read_bytes() != source_bytes
    source_hash_exact = source_sha256 == EXPECTED_SOURCE_ARTIFACT_SHA256
    gate_contract_ready = (
        source_hash_exact
        and normalization["unsafe_false_accept_count_total"] == 0
        and normalization["by_arm_reconciliation_pass"] is True
        and source_artifact_modified is False
    )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifact_path": _display_path(root_path, resolved_source_path),
        "source_artifact_sha256": source_sha256,
        "expected_source_artifact_sha256": EXPECTED_SOURCE_ARTIFACT_SHA256,
        "source_hash_exact": source_hash_exact,
        "source_honest_verdict": source_artifact.get("honest_verdict"),
        "normalization_json_path": NORMALIZATION_JSON_PATH,
        "source_continuous_self_learning_ready": source_artifact.get(
            "continuous_self_learning_ready"
        ),
        "scientific_recompute_performed": False,
        "source_artifact_modified": source_artifact_modified,
        "gate_contract_ready_score": 1.0 if gate_contract_ready else 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
    }
    artifact.update(normalization)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    source_path: Path | str = SOURCE_ARTIFACT_RELATIVE_PATH,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the deterministic Exp5638 corrigendum."""

    root_path = Path(root)
    artifact = build_artifact(
        root=root_path,
        source_path=source_path,
        tests_added_or_reused=tests_added_or_reused,
    )
    if write:
        write_json(_resolve_path(root_path, result_path), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the corrigendum cannot safely feed a scalar conductor gate."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5638 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while recomputing only schema-derived checks."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if not artifact.get("tests_added_or_reused"):
        errors.append("tests_added_or_reused")

    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        missing_principles = [
            field for field in REQUIRED_ARTIFACT_FIELDS if not principles.get(field)
        ]
        if missing_principles:
            errors.append(f"field_principles missing: {missing_principles}")

    normalized: JsonDict | None = None
    try:
        normalized = normalize_unsafe_false_accept_count(
            {"unsafe_false_accept_count": artifact.get("raw_unsafe_false_accept_count")}
        )
    except ValueError as exc:
        errors.append(f"raw_unsafe_false_accept_count: {exc}")

    if normalized is not None:
        if (
            artifact.get("unsafe_false_accept_count_total")
            != normalized["unsafe_false_accept_count_total"]
        ):
            errors.append("unsafe_false_accept_count_total")
        if artifact.get("by_arm_sum") != normalized["by_arm_sum"]:
            errors.append("by_arm_sum")
        if artifact.get("by_arm_reconciliation_pass") is not True:
            errors.append("by_arm_reconciliation_pass")

    source_hash_exact = artifact.get("source_artifact_sha256") == artifact.get(
        "expected_source_artifact_sha256"
    )
    if artifact.get("source_hash_exact") is not source_hash_exact:
        errors.append("source_artifact_sha256/source_hash_exact")
    if artifact.get("scientific_recompute_performed") is not False:
        errors.append("scientific_recompute_performed")
    if artifact.get("source_artifact_modified") is not False:
        errors.append("source_artifact_modified")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")

    expected_score = expected_gate_contract_ready_score(artifact, source_hash_exact)
    if artifact.get("gate_contract_ready_score") != expected_score:
        errors.append("gate_contract_ready_score")

    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES) or FORBIDDEN_VERDICT_PHRASE in verdict:
        errors.append("honest_verdict")
    if artifact.get("gate_contract_ready_score") == 1.0 and not verdict.startswith("complete:"):
        errors.append("honest_verdict")
    if artifact.get("gate_contract_ready_score") == 0.0 and not verdict.startswith("blocked:"):
        errors.append("honest_verdict")

    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def expected_gate_contract_ready_score(
    artifact: Mapping[str, Any],
    source_hash_exact: bool,
) -> float:
    """Compute the mechanical downstream score from already-normalized fields."""

    if (
        source_hash_exact
        and artifact.get("unsafe_false_accept_count_total") == 0
        and artifact.get("by_arm_reconciliation_pass") is True
        and artifact.get("scientific_recompute_performed") is False
        and artifact.get("source_artifact_modified") is False
    ):
        return 1.0
    return 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict without claiming independent FR-11 validation."""

    source_hash_exact = bool(artifact.get("source_hash_exact"))
    if expected_gate_contract_ready_score(artifact, source_hash_exact) == 1.0:
        return "complete: hash_bound_scalar_gate_contract_ready"
    return "blocked: hash_bound_scalar_gate_contract_not_ready"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the receipt with the checksum field removed."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(payload)


def source_file_checksums(root: Path | str) -> JsonDict:
    """Record the files that define this schema-normalization receipt."""

    root_path = Path(root)
    return {
        "module": sha256_file(root_path / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root_path / SPEC_RELATIVE_PATH),
        "test": sha256_file(root_path / TEST_RELATIVE_PATH),
    }


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable JSON so repeated corrigendum runs are byte-identical."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a prefixed SHA-256 digest for stable JSON payloads."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return sha256_bytes(blob)


def sha256_bytes(payload: bytes) -> str:
    """Return a prefixed SHA-256 digest over exact bytes."""

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest over a file's exact bytes."""

    return sha256_bytes(Path(path).read_bytes())


def main() -> int:  # pragma: no cover
    """CLI entrypoint for writing the repository Exp5638 receipt."""

    artifact = run()
    print(_resolve_path(REPO_ROOT, RESULT_RELATIVE_PATH).as_posix())
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
