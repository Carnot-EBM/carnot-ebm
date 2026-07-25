"""Exp5907 ConstraintIR replay contract repair.

Spec refs: REQ-BENCH-5907, SCENARIO-BENCH-5907-CANONICAL,
SCENARIO-BENCH-5907-FRESH-PROCESS, SCENARIO-BENCH-5907-TAMPER,
SCENARIO-BENCH-5907-LEGACY.

This module repairs only the deterministic artifact replay contract between
the Exp5896 ConstraintIR fixture producer and the Exp5897 consumer gate. It
does not run a model, rewrite historical artifacts, or make a model-science
claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import constraint_ir_replay_contract
from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5897_sota_constraint_ir_repair_ab as exp5897


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5907_constraint_ir_replay_contract.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5907_constraint_ir_replay_contract.py")
HELPER_RELATIVE_PATH = Path("python/carnot/constraint_ir_replay_contract.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5907_constraint_ir_replay_contract.py")
BENCH_SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
RUN_DATE = "20260725"
EXPERIMENT_ID = "experiment_5907_constraint_ir_replay_contract"
INFERENCE_SUBSTRATE = "deterministic_artifact_replay_no_llm"
VERIFIER_IS_ORACLE = True

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_artifact_hashes",
    "mismatch_reproduction_and_root_cause",
    "canonical_projection_schema_and_version",
    "excluded_and_bound_fields",
    "shared_helper_receipt",
    "fresh_twin_producer_consumer_replay",
    "fresh_process_replay_receipt",
    "tamper_detection_matrix",
    "legacy_exp5896_adjudication",
    "historical_artifacts_unchanged",
    "protected_files_unchanged",
    "constraint_ir_replay_contract_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "canonical_projection_schema_and_version": (
        "One explicit public projection owns every producer-consumer checksum."
    ),
    "historical_artifacts_unchanged": "A repaired contract cannot rewrite prior evidence.",
    "constraint_ir_replay_contract_ready_score": (
        "Emit bare 1.0 only for shared-helper parity, fresh-process exact replay, "
        "row binding, and complete tamper rejection."
    ),
    "inference_substrate": "Use deterministic_artifact_replay_no_llm.",
    "verifier_is_oracle": "True only for checksum/schema/tamper adjudication.",
    "honest_verdict": "Use complete_ready:, retired:, or blocked:.",
}

PRE_EDIT_INPUT_HASHES = {
    "AGENTS.md": "sha256:6afad92d58a6c7860487402caf9b70c6c9a0a8734724efc8112fd200e1c8a93a",
    "CLAUDE.md": "sha256:20cbc11c703e7fb7209a0e1745b64f8626aca9cb8905503f12650d04fa1c5db9",
    "CODEX.md": "sha256:1db7ebc1009714f872db83447828a8957c3abb72061832cc1bc307735cd96bbe",
    "research-program.md": "sha256:53ed14d8257e92794b841f3959dc3337cafe56adce714fe3447fd62977e4d9f9",
    str(
        exp5896.RESULT_RELATIVE_PATH
    ): "sha256:7e5d6cb2169e006c4b26f2fbd980e5ad38df52acafee6267bc54fa0d316fffea",
    str(
        exp5896.ROW_FILE_RELATIVE_PATH
    ): "sha256:e26db6998de83696d296974acc6cb4731f0a2956846083ad6e00da8cb282a55b",
    str(
        exp5897.RESULT_RELATIVE_PATH
    ): "sha256:3481f27c0ba603ead764c52c4d617ac80c001852ab3bb6559d6abbac0b6dc91d",
    str(
        exp5896.MODULE_RELATIVE_PATH
    ): "sha256:6b69b6f65b10228ecc5934a4681f118b080e438e1bf3486ea93dbe741dec6c62",
    str(
        exp5897.MODULE_RELATIVE_PATH
    ): "sha256:ff4a9c335cfef7847a48492dbdc3eeb33a55791c4b16a90590d84a86fd2b5d20",
    str(
        exp5896.TEST_RELATIVE_PATH
    ): "sha256:61aeb6065275fe13aa817e4b655ea2ddea11ac30b57663a921de4168f3b45e2b",
    str(
        exp5897.TEST_RELATIVE_PATH
    ): "sha256:0b22193ee3d18d6ccf8f47379673695b09090614aed314166b964058f20c7d60",
    "scripts/adversarial_verify.py": "sha256:a002b15ab311865bcd009254bd27b5bbf0e0e182eff06c410db11aa4fc8388de",
    str(
        BENCH_SPEC_RELATIVE_PATH
    ): "sha256:43ff0d276570a42db2bffeb7d0797ce3299cc3aa9841092c850dba3122751a76",
    "openspec/capabilities/verification/spec.md": "sha256:88d758aae7663c9af71fafdc22dfa70bb1d15cee59b426bb78b564f955cb561b",
}
HISTORICAL_ARTIFACT_HASHES = {
    str(exp5896.RESULT_RELATIVE_PATH): PRE_EDIT_INPUT_HASHES[str(exp5896.RESULT_RELATIVE_PATH)],
    str(exp5896.ROW_FILE_RELATIVE_PATH): PRE_EDIT_INPUT_HASHES[str(exp5896.ROW_FILE_RELATIVE_PATH)],
    str(exp5897.RESULT_RELATIVE_PATH): PRE_EDIT_INPUT_HASHES[str(exp5897.RESULT_RELATIVE_PATH)],
}
PROTECTED_FILE_HASHES = {
    "scripts/research_conductor.py": "sha256:353e0a26ec6c7f9cb144cb172dbd0d1b3409196c5e07f18f8e01c9a276694771",
    "ops/changelog.md": "sha256:f322d42482f84d49a3d853cda09fa4e13084b61d5e17a27f6ff4dcda6d27d477",
    "ops/status.md": "sha256:eda040b1c97c3108e440468451d60fec994d29291f770d657688dd9df1e104a1",
    "_bmad/traceability.md": "sha256:e9727749b5389bc9f9687f5ae322134c902e4147922e8caeb4f0935f130c6db2",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5907_constraint_ir_replay_contract.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/constraint_ir_replay_contract.py,python/carnot/experiment_5907_constraint_ir_replay_contract.py "
    "-m pytest tests/python/test_experiment_5907_constraint_ir_replay_contract.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/constraint_ir_replay_contract.py,python/carnot/experiment_5907_constraint_ir_replay_contract.py --fail-under=100",
    ".venv/bin/pytest tests/python/test_experiment_5896_typed_constraint_ir_fixture.py "
    "tests/python/test_experiment_5897_sota_constraint_ir_repair_ab.py "
    "tests/python/test_experiment_5907_constraint_ir_replay_contract.py -q --no-cov -n 0",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5907_constraint_ir_replay_contract",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5907_constraint_ir_replay_contract.json",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def reproduce_historical_mismatch(*, root: Path = REPO_ROOT) -> JsonDict:
    """Reproduce the pre-repair Exp5896 checksum mismatch without old code."""

    result_path = root / exp5896.RESULT_RELATIVE_PATH
    row_path = root / exp5896.ROW_FILE_RELATIVE_PATH
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines() if line]
    rebuilt = exp5896.build_artifact(rows, root=root, duration_s=0.0, test_exit_codes={})

    artifact_projection = _old_exp5896_projection_text(artifact)
    rebuilt_projection = _old_exp5896_projection_text(rebuilt)
    artifact_checksum = sha256_text(artifact_projection)
    rebuilt_checksum = sha256_text(rebuilt_projection)
    differing_paths = [
        path
        for path, _left, _right in _diff_paths(
            json.loads(artifact_projection), json.loads(rebuilt_projection)
        )
    ]
    old_replay_error = (
        "ConstraintIRReplayError: artifact reproducibility checksum mismatch"
        if artifact_checksum != rebuilt_checksum
        else ""
    )
    return {
        "old_replay_error": old_replay_error,
        "stored_checksum": artifact["reproducibility_checksum"],
        "artifact_projection_checksum": artifact_checksum,
        "rebuilt_projection_checksum": rebuilt_checksum,
        "artifact_projection_byte_length": len(artifact_projection.encode("utf-8")),
        "rebuilt_projection_byte_length": len(rebuilt_projection.encode("utf-8")),
        "artifact_projection_utf8": artifact_projection,
        "rebuilt_projection_utf8": rebuilt_projection,
        "root_cause": {
            "differing_paths": differing_paths,
            "summary": "old projection included protected ops/status hashes that changed after Exp5896",
        },
    }


def adjudicate_legacy_exp5896(*, root: Path = REPO_ROOT) -> JsonDict:
    """Replay checked-in Exp5896 as immutable legacy evidence under the new projection."""

    mismatch = reproduce_historical_mismatch(root=root)
    replay = exp5896.replay_artifact(root=root)
    projection = dict(replay["canonical_projection"])
    return {
        "historical_checksum_mismatch_preserved": bool(mismatch["old_replay_error"]),
        "new_contract_replay_ready": bool(replay["ok"]),
        "legacy_mode_without_projection_field": bool(
            projection.get("legacy_mode_without_projection_field")
        ),
        "stored_checksum_matched_new_projection": bool(projection.get("stored_checksum_matched")),
        "stored_historical_checksum": projection.get("stored_checksum"),
        "new_contract_checksum": projection.get("checksum"),
        "row_count": replay["row_count"],
        "row_file_sha256": projection.get("bound_fields", {}).get("row_file_sha256"),
    }


def run_fresh_twin_producer_consumer_replay() -> JsonDict:
    """Generate a deterministic twin fixture and replay it through both entrypoints."""

    with tempfile.TemporaryDirectory(prefix="exp5907-twin-") as tmp:
        root = Path(tmp)
        artifact = exp5896.write_fixture(root=root, duration_s=0.0)
        producer = exp5896.replay_artifact(root=root)
        consumer = exp5897._upstream_gate_receipt(root)
        producer_checksum = producer["canonical_projection"]["checksum"]
        consumer_checksum = (
            consumer["canonical_projection"]["checksum"] if consumer["replay_ok"] else None
        )
        return {
            "producer_replay_ok": bool(producer["ok"]),
            "consumer_replay_ok": bool(consumer["replay_ok"]),
            "producer_checksum": producer_checksum,
            "consumer_checksum": consumer_checksum,
            "artifact_checksum": artifact["reproducibility_checksum"],
            "shared_helper_parity": producer_checksum
            == consumer_checksum
            == artifact["reproducibility_checksum"],
            "row_file_sha256": producer["canonical_projection"]["bound_fields"]["row_file_sha256"],
        }


def run_fresh_process_replay() -> JsonDict:
    """Replay a fresh twin fixture in a separate Python interpreter."""

    with tempfile.TemporaryDirectory(prefix="exp5907-fresh-process-") as tmp:
        root = Path(tmp)
        artifact = exp5896.write_fixture(root=root, duration_s=0.0)
        code = """
import json
import sys
from pathlib import Path
from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
receipt = exp5896.replay_artifact(root=Path(sys.argv[1]))
print(json.dumps(receipt, sort_keys=True))
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python")
        proc = subprocess.run(
            [sys.executable, "-c", code, str(root)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        parsed = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {}
        checksum = parsed.get("canonical_projection", {}).get("checksum")
        return {
            "ok": proc.returncode == 0 and bool(parsed.get("ok")),
            "returncode": proc.returncode,
            "stdout_sha256": sha256_text(proc.stdout),
            "stderr_sha256": sha256_text(proc.stderr),
            "checksum": checksum,
            "matches_producer_checksum": checksum == artifact["reproducibility_checksum"],
            "row_file_sha256": parsed.get("canonical_projection", {})
            .get("bound_fields", {})
            .get("row_file_sha256"),
            "error": proc.stderr if proc.returncode else None,
        }


def run_tamper_detection_matrix() -> JsonDict:
    """Tamper every bound replay component and require fail-closed replay."""

    cases = [
        ("row_file_bytes", _tamper_row_file_bytes),
        ("row_file_sha256_receipt", _tamper_row_file_receipt),
        ("constraint_ir_schema_version", _tamper_schema_version),
        ("projection_schema_version", _tamper_projection_version),
        ("reproducibility_checksum", _tamper_reproducibility_checksum),
    ]
    results = []
    for component, tamper in cases:
        with tempfile.TemporaryDirectory(prefix=f"exp5907-tamper-{component}-") as tmp:
            root = Path(tmp)
            exp5896.write_fixture(root=root, duration_s=0.0)
            tamper(root)
            try:
                exp5896.replay_artifact(root=root)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "component": component,
                        "rejected": True,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                results.append(
                    {
                        "component": component,
                        "rejected": False,
                        "error_type": None,
                        "error": None,
                    }
                )
    return {
        "cases": results,
        "all_rejected": all(case["rejected"] for case in results),
        "row_binding_checked": _case_rejected(results, "row_file_bytes")
        and _case_rejected(results, "row_file_sha256_receipt"),
        "schema_binding_checked": _case_rejected(results, "constraint_ir_schema_version"),
        "projection_version_checked": _case_rejected(results, "projection_schema_version"),
        "checksum_checked": _case_rejected(results, "reproducibility_checksum"),
    }


def write_contract_artifact(
    *,
    output_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build and write the Exp5907 terminal artifact."""

    started = time.monotonic()
    target = output_path or REPO_ROOT / RESULT_RELATIVE_PATH
    artifact = _build_artifact(
        output_path=target,
        duration_s=duration_s if duration_s is not None else 0.0,
        test_exit_codes=test_exit_codes,
    )
    if duration_s is None:
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be deterministic_artifact_replay_no_llm")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true for checksum/schema/tamper adjudication")
    if artifact["canonical_projection_schema_and_version"]["projection_schema_version"] != (
        constraint_ir_replay_contract.PROJECTION_SCHEMA_VERSION
    ):
        raise ValueError("canonical projection version mismatch")
    score = float(artifact["constraint_ir_replay_contract_ready_score"])
    if score not in {0.0, 1.0}:
        raise ValueError("constraint_ir_replay_contract_ready_score must be bare 0.0 or 1.0")
    if score == 1.0 and not str(artifact["honest_verdict"]).startswith("complete_ready:"):
        raise ValueError("ready score requires complete_ready verdict")


def _build_artifact(
    *,
    output_path: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> JsonDict:
    mismatch = reproduce_historical_mismatch(root=REPO_ROOT)
    fresh = run_fresh_twin_producer_consumer_replay()
    fresh_process = run_fresh_process_replay()
    tamper = run_tamper_detection_matrix()
    legacy = adjudicate_legacy_exp5896(root=REPO_ROOT)
    historical = _unchanged_receipt(HISTORICAL_ARTIFACT_HASHES)
    protected = _unchanged_receipt(PROTECTED_FILE_HASHES)
    ready = (
        fresh["shared_helper_parity"]
        and fresh_process["ok"]
        and fresh_process["matches_producer_checksum"]
        and tamper["all_rejected"]
        and tamper["row_binding_checked"]
        and legacy["new_contract_replay_ready"]
        and historical["unchanged"]
        and protected["unchanged"]
    )
    score = 1.0 if ready else 0.0
    artifact: JsonDict = {
        "schema": "carnot.experiment_5907.constraint_ir_replay_contract.v1",
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": FIELD_PRINCIPLES,
        "status": "complete_ready" if ready else "blocked",
        "preconditions_checked": _preconditions(output_path),
        "immutable_artifact_hashes": _current_hashes(HISTORICAL_ARTIFACT_HASHES),
        "mismatch_reproduction_and_root_cause": mismatch,
        "canonical_projection_schema_and_version": _canonical_projection_schema_receipt(),
        "excluded_and_bound_fields": _excluded_and_bound_fields_receipt(),
        "shared_helper_receipt": {
            "helper_path": str(HELPER_RELATIVE_PATH),
            "projection_schema_version": constraint_ir_replay_contract.PROJECTION_SCHEMA_VERSION,
            "producer_entrypoint": str(exp5896.MODULE_RELATIVE_PATH) + ":_artifact_checksum",
            "consumer_entrypoint": str(exp5897.MODULE_RELATIVE_PATH) + ":_upstream_gate_receipt",
            "producer_consumer_parity": fresh["shared_helper_parity"],
        },
        "fresh_twin_producer_consumer_replay": fresh,
        "fresh_process_replay_receipt": fresh_process,
        "tamper_detection_matrix": tamper,
        "legacy_exp5896_adjudication": legacy,
        "historical_artifacts_unchanged": historical,
        "protected_files_unchanged": protected,
        "constraint_ir_replay_contract_ready_score": score,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_ready: ConstraintIR replay contract is shared, row-bound, and tamper-rejecting"
            if ready
            else "blocked: ConstraintIR replay contract did not clear all replay gates"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["test_exit_codes"] = {}
    stable["reproducibility_checksum"] = ""
    return sha256_text(canonical_json(stable))


def _old_exp5896_projection_text(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["test_exit_codes"] = {}
    stable["reproducibility_checksum"] = ""
    stable["row_file_receipt"]["sha256"] = None
    stable["preconditions_checked"]["disk"]["available_mb"] = 0
    stable["preconditions_checked"]["ram"]["available_mb"] = 0
    return canonical_json(stable)


def _diff_paths(left: Any, right: Any, path: str = "$") -> list[tuple[str, Any, Any]]:
    if type(left) is not type(right):
        return [(path, left, right)]
    if isinstance(left, dict):
        out: list[tuple[str, Any, Any]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                out.append((f"{path}.{key}", "<missing>", right[key]))
            elif key not in right:
                out.append((f"{path}.{key}", left[key], "<missing>"))
            else:
                out.extend(_diff_paths(left[key], right[key], f"{path}.{key}"))
        return out
    if isinstance(left, list):
        out = []
        if len(left) != len(right):
            out.append((f"{path}.length", len(left), len(right)))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            out.extend(_diff_paths(left_item, right_item, f"{path}[{index}]"))
        return out
    return [] if left == right else [(path, left, right)]


def _preconditions(output_path: Path) -> JsonDict:
    return {
        "run_date": RUN_DATE,
        "pre_edit_input_hashes": PRE_EDIT_INPUT_HASHES,
        "current_input_hashes": _current_hashes(PRE_EDIT_INPUT_HASHES),
        "output_path": str(output_path),
        "output_path_existed_before_write": output_path.exists(),
        "disk": _disk_probe(REPO_ROOT),
        "ram": _memory_probe(),
        "protected_file_hashes_before_edit": PROTECTED_FILE_HASHES,
        "no_model_load_required": True,
    }


def _canonical_projection_schema_receipt() -> JsonDict:
    return {
        "projection_schema_version": constraint_ir_replay_contract.PROJECTION_SCHEMA_VERSION,
        "normalization_version": constraint_ir_replay_contract.NORMALIZATION_VERSION,
        "checksum_field": "reproducibility_checksum",
        "principle": FIELD_PRINCIPLES["canonical_projection_schema_and_version"],
    }


def _excluded_and_bound_fields_receipt() -> JsonDict:
    return {
        "excluded_top_level_fields": list(constraint_ir_replay_contract.EXCLUDED_TOP_LEVEL_FIELDS),
        "excluded_nested_paths": [
            ".".join(path) for path in constraint_ir_replay_contract.EXCLUDED_NESTED_PATHS
        ],
        "bound_field_names": list(constraint_ir_replay_contract.BOUND_FIELD_NAMES),
        "bound_row_file_sha256": sha256_file(REPO_ROOT / exp5896.ROW_FILE_RELATIVE_PATH),
        "bound_schema_versions": {
            "artifact_schema": exp5896.ARTIFACT_SCHEMA_VERSION,
            "constraint_ir_schema_version": exp5896.CONSTRAINT_IR_SCHEMA_VERSION,
            "row_schema_version": exp5896.ROW_SCHEMA_VERSION,
        },
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "satisfied_by": "generated_by_exp5907_constraint_ir_replay_contract",
            "principle": FIELD_PRINCIPLES.get(field, "Replay-contract audit field."),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _current_hashes(expected: Mapping[str, str]) -> JsonDict:
    rows = []
    for relative, expected_sha in expected.items():
        path = REPO_ROOT / relative
        current = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "path": relative,
                "exists": path.exists(),
                "expected_sha256": expected_sha,
                "sha256": current,
                "matches_expected": current == expected_sha,
            }
        )
    return {
        "files": rows,
        "all_present": all(row["exists"] for row in rows),
        "all_match_expected": all(row["matches_expected"] for row in rows),
    }


def _unchanged_receipt(expected: Mapping[str, str]) -> JsonDict:
    current = _current_hashes(expected)
    return {
        "unchanged": bool(current["all_present"] and current["all_match_expected"]),
        "files": current["files"],
    }


def _disk_probe(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _memory_probe() -> JsonDict:
    required_mb = 512
    meminfo = Path("/proc/meminfo")
    available_mb = 0
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover - non-Linux fallback.
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _case_rejected(cases: Sequence[Mapping[str, Any]], component: str) -> bool:
    return any(
        case.get("component") == component and case.get("rejected") is True for case in cases
    )


def _artifact_path(root: Path) -> Path:
    return root / exp5896.RESULT_RELATIVE_PATH


def _row_path(root: Path) -> Path:
    return root / exp5896.ROW_FILE_RELATIVE_PATH


def _read_artifact(root: Path) -> JsonDict:
    return json.loads(_artifact_path(root).read_text(encoding="utf-8"))


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    _artifact_path(root).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _tamper_row_file_bytes(root: Path) -> None:
    _row_path(root).write_text(
        _row_path(root).read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
    )


def _tamper_row_file_receipt(root: Path) -> None:
    artifact = _read_artifact(root)
    artifact["row_file_receipt"]["sha256"] = "sha256:" + "0" * 64
    _write_artifact(root, artifact)


def _tamper_schema_version(root: Path) -> None:
    artifact = _read_artifact(root)
    artifact["constraint_ir_schema_and_version"]["schema_version"] = "carnot.constraint_ir.v0"
    _write_artifact(root, artifact)


def _tamper_projection_version(root: Path) -> None:
    artifact = _read_artifact(root)
    artifact["canonical_projection_schema_and_version"]["projection_schema_version"] = (
        "carnot.constraint_ir.replay_contract_projection.v0"
    )
    _write_artifact(root, artifact)


def _tamper_reproducibility_checksum(root: Path) -> None:
    artifact = _read_artifact(root)
    artifact["reproducibility_checksum"] = "sha256:" + "1" * 64
    _write_artifact(root, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    write_contract_artifact(output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
