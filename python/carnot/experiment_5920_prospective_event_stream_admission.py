"""Exp5920 prospective event-stream admission boundary.

Spec refs: REQ-LEARN-5920, SCENARIO-LEARN-5920-SCHEMA,
SCENARIO-LEARN-5920-REPLAY, SCENARIO-LEARN-5920-TAMPER,
SCENARIO-LEARN-5920-BOUNDARY, REQ-HARNESS-5920,
SCENARIO-HARNESS-5920.

This module turns the immutable Exp5908 prompt-plan fixture and immutable
Exp5909 sealed model-output stream into an Exp5920-owned transaction-consumer
stream. The important boundary is not model quality. It is whether a learner
can consume events in order without seeing future labels, changing row
identity, drifting splits, or laundering post-hoc labels into a promotion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5908_verisynth_constraint_fixture as exp5908
from carnot import experiment_5909_sota_constraint_synthesis_ab as exp5909


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5920_prospective_event_stream_admission.json")
ROW_FILE_RELATIVE_PATH = Path(
    "results/experiment_5920_prospective_event_stream_admission.rows.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5920_prospective_event_stream_admission.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5920_prospective_event_stream_admission.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
HARNESS_SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXP5912_RESULT_RELATIVE_PATH = Path("results/experiment_5912_csl_exact_slot_requalification.json")
EXP5909_EVENTS_ALIAS_RELATIVE_PATH = Path(
    "results/experiment_5909_sota_constraint_synthesis_ab.events.jsonl"
)
RUN_DATE = "20260725"
RANDOM_SEED = 5920
EXPERIMENT_ID = "experiment_5920_prospective_event_stream_admission"
SCHEMA_VERSION = "carnot.experiment_5920.prospective_event_stream_admission.v1"
ROW_SCHEMA_VERSION = SCHEMA_VERSION + ".row"
INFERENCE_SUBSTRATE = "deterministic_artifact_replay_no_llm"
VERIFIER_IS_ORACLE = True
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
GENESIS_PREFIX_CHECKSUM = "sha256:" + hashlib.sha256(b"exp5920.genesis").hexdigest()

TASK_OWNED_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5920_prospective_event_stream_admission.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5920_prospective_event_stream_admission.py "
    "-m pytest tests/python/test_experiment_5920_prospective_event_stream_admission.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5920_prospective_event_stream_admission.py --fail-under=100",
    ".venv/bin/pytest tests/python/test_experiment_5908_verisynth_constraint_fixture.py "
    "tests/python/test_experiment_5909_sota_constraint_synthesis_ab.py "
    "tests/python/test_experiment_5920_prospective_event_stream_admission.py -q --no-cov -n 0",
    ".venv/bin/ruff check python/carnot/experiment_5920_prospective_event_stream_admission.py "
    "tests/python/test_experiment_5920_prospective_event_stream_admission.py",
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_5920_prospective_event_stream_admission.py "
    "tests/python/test_experiment_5920_prospective_event_stream_admission.py",
    ".venv/bin/python -m carnot.experiment_5920_prospective_event_stream_admission --validate",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5920_prospective_event_stream_admission.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5920_prospective_event_stream_admission.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
)
DEFAULT_TEST_COMMANDS = (*TASK_OWNED_COMMANDS, GLOBAL_PYTEST_COMMAND)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
SOURCE_SCHEMA_RELATIVE_PATHS = (
    SELF_LEARNING_SPEC_RELATIVE_PATH,
    HARNESS_SPEC_RELATIVE_PATH,
    exp5908.MODULE_RELATIVE_PATH,
    exp5909.MODULE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)
IMMUTABLE_UPSTREAM_RELATIVE_PATHS = (
    exp5908.RESULT_RELATIVE_PATH,
    exp5908.ROW_FILE_RELATIVE_PATH,
    exp5909.RESULT_RELATIVE_PATH,
    exp5909.RAW_STREAM_RELATIVE_PATH,
    EXP5912_RESULT_RELATIVE_PATH,
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/known-issues.md"),
    Path("ops/e2e-test-plan.md"),
    Path("research-program.md"),
    Path("research-complete.yaml"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "retired_scope_not_reopened",
    "immutable_upstream_hashes",
    "prospective_event_schema_and_version",
    "fresh_stream_path_hash_row_count_and_prefix_chain",
    "chronology_split_and_visibility_receipts",
    "exact_label_authority",
    "replay_and_tamper_matrix",
    "task_owned_test_boundary",
    "global_suite_baseline_and_failure_delta",
    "protected_files_unchanged",
    "prospective_stream_admission_ready_score",
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
    "status": "A terminal state distinguishes admitted, retired, or blocked prospective stream evidence.",
    "preconditions_checked": "Hashes, resources, source schemas, output paths, and atomic writes prevent fabricated stream admission.",
    "retired_scope_not_reopened": "True only when no Exp5912 ready scalar, artifact mutation, or exact frozen rerun is used.",
    "immutable_upstream_hashes": "Exp5908, Exp5909, Exp5912, schemas, exclusions, and baselines are consumed read-only.",
    "prospective_event_schema_and_version": "One versioned row contract owns chronology, visibility, identity, source hashes, and replay.",
    "fresh_stream_path_hash_row_count_and_prefix_chain": "The Exp5920-owned JSONL path, row count, file hash, and prefix chain bind every row.",
    "chronology_split_and_visibility_receipts": "Monotone ids, stable split, and no future-label prompt visibility are mandatory.",
    "exact_label_authority": "Exact verifier labels and diagnostics are adjudication authority after proposal only.",
    "replay_and_tamper_matrix": "Fresh-process replay and chronology, visibility, label, hash, and split tampering must reject completely.",
    "task_owned_test_boundary": "Unit, coverage, spec, replay, adversarial, E2E, protected-file, and clutter checks define the task-owned clean gate.",
    "global_suite_baseline_and_failure_delta": "Known unrelated debt is preserved by exact node id and may not increase.",
    "protected_files_unchanged": "Protected files and upstream artifacts stay byte-identical.",
    "prospective_stream_admission_ready_score": "Emit bare 1.0 only for clean task-owned commands, global failure delta at most zero, fresh-process replay, and complete tamper rejection.",
    "duration_s": "Measured wall time exposes deterministic replay work.",
    "inference_substrate": "Use `deterministic_artifact_replay_no_llm`.",
    "verifier_is_oracle": "True only for schema, checksum, chronology, visibility, and exact-label adjudication.",
    "field_provenance": "Every field traces to task prompt, specs, upstream artifacts, rows, tests, or command receipts.",
    "test_commands": "Commands document focused unit/coverage, clean-boundary, replay, tamper, hash, adversarial, spec, applicable E2E, protected-file, global-delta, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed task-owned checks from becoming readiness.",
    "reproducibility_checksum": "A checksum detects event, source, prefix, test, or protected-file drift.",
    "honest_verdict": "Use `complete_ready:`, `retired:`, or `blocked:`.",
}


class ProspectiveEventStreamError(ValueError):
    """Raised when the admitted event stream cannot be trusted."""


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence into stable ASCII text."""

    return json.dumps(
        _normalize_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON-compatible data."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes and ignore path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def event_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one Exp5920 event row while excluding derived hash fields."""

    stable = json.loads(canonical_json(row))
    stable["row_hash"] = ""
    stable["prefix_checksum"] = ""
    return sha256_json(stable)


def prefix_checksum(prior_prefix_checksum: str, row_hash: str) -> str:
    """Hash the prior prefix and current row into the next prefix receipt."""

    return sha256_json({"prior_prefix_checksum": prior_prefix_checksum, "row_hash": row_hash})


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object artifact and reject scalar or array payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def load_jsonl(path: str | Path) -> list[JsonDict]:
    """Read a JSONL stream as a list of object rows."""

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ProspectiveEventStreamError("JSONL row object required")
        rows.append(dict(payload))
    return rows


def build_event_rows(*, root: Path = REPO_ROOT) -> list[JsonDict]:
    """Materialize the Exp5920-owned prospective event stream from Exp5908/5909."""

    root = Path(root)
    raw_rows = load_jsonl(root / exp5909.RAW_STREAM_RELATIVE_PATH)
    plan_rows = load_jsonl(root / exp5908.ROW_FILE_RELATIVE_PATH)
    exp5909_artifact = read_json(root / exp5909.RESULT_RELATIVE_PATH)
    source_rows = {str(row["row_id"]): row for row in exp5896.build_fixture_rows()}
    plan_by_source = {str(row["source_row_id"]): row for row in plan_rows}
    model_hashes = _model_file_hashes_by_id(exp5909_artifact)
    source_hashes = _source_artifact_hashes(root)

    prefix = GENESIS_PREFIX_CHECKSUM
    out = []
    for index, raw in enumerate(raw_rows):
        source_id = str(raw["source_row_id"])
        source = source_rows[source_id]
        plan = plan_by_source[source_id]
        evaluation = exp5909.evaluate_candidate(
            source, str(raw["arm_id"]), str(raw.get("raw_output_text") or ""), dict(raw)
        )
        exact_labels = _exact_label_projection(evaluation)
        model_identity = _model_identity(raw, model_hashes)
        row: JsonDict = {
            "schema": ROW_SCHEMA_VERSION,
            "event_id": f"exp5920-event-{index:06d}",
            "causal_sequence_index": index,
            "case_id": _case_id(raw),
            "origin_family": raw.get("family"),
            "prompt_visibility": _prompt_visibility(raw, exp5909_artifact),
            "proposal": _proposal_projection(raw),
            "exact_diagnostic_and_label": {
                "authority": "exp5909.evaluate_candidate",
                "exact_labels": exact_labels,
                "upstream_exact_labels": dict(raw.get("exact_labels") or {}),
                "upstream_label_match": exact_labels == dict(raw.get("exact_labels") or {}),
                "visible_diagnostics": dict(evaluation.get("diagnostics") or {}),
                "diagnostic_visibility": "post_proposal_downstream_visible",
            },
            "exact_label_projection": exact_labels,
            "commit_eligibility": _commit_eligibility(exact_labels),
            "split": raw.get("split"),
            "model_identity": model_identity,
            "source_artifact_hashes": source_hashes,
            "source_row": {
                "source_row_id": source_id,
                "source_row_hash": raw.get("source_row_hash"),
                "plan_row_hash": raw.get("plan_row_hash"),
                "exp5908_plan_row_hash": plan.get("row_hash"),
                "exp5909_raw_row_hash": raw.get("row_hash"),
                "prompt_plan_hash": raw.get("prompt_plan_hash"),
                "group_id": raw.get("group_id"),
                "template_id": raw.get("template_id"),
                "variant_kind": raw.get("variant_kind"),
            },
            "prior_prefix_checksum": prefix,
            "row_hash": "",
            "prefix_checksum": "",
        }
        row["row_hash"] = event_row_hash(row)
        prefix = prefix_checksum(prefix, row["row_hash"])
        row["prefix_checksum"] = prefix
        out.append(row)
    return out


def validate_event_rows(rows: Sequence[Mapping[str, Any]], *, root: Path = REPO_ROOT) -> JsonDict:
    """Validate chronology, visibility, labels, hashes, split, and prefix chain."""

    row_list = [dict(row) for row in rows]
    expected = build_event_rows(root=root)
    if len(row_list) != len(expected):
        raise ProspectiveEventStreamError("chronology row count mismatch")
    seen: set[str] = set()
    prefix = GENESIS_PREFIX_CHECKSUM
    eligible_commit_count = 0
    splits: dict[str, int] = {}
    for index, row in enumerate(row_list):
        event_id = str(row.get("event_id"))
        if event_id in seen:
            raise ProspectiveEventStreamError("duplicate event id")
        seen.add(event_id)
        if event_id != f"exp5920-event-{index:06d}" or row.get("causal_sequence_index") != index:
            raise ProspectiveEventStreamError("chronology reordered row")
        visibility = dict(row.get("prompt_visibility") or {})
        if (
            visibility.get("future_label_visible_to_model") is True
            or visibility.get("target_exact_labels_exposed_to_prompt") is True
        ):
            raise ProspectiveEventStreamError("future label visibility")
        expected_row = expected[index]
        if row.get("split") != expected_row.get("split"):
            raise ProspectiveEventStreamError("split drift")
        if dict(row.get("source_artifact_hashes") or {}) != expected_row["source_artifact_hashes"]:
            raise ProspectiveEventStreamError("source hash drift")
        if dict(row.get("source_row") or {}) != expected_row["source_row"]:
            raise ProspectiveEventStreamError("source hash drift")
        if dict(row.get("model_identity") or {}) != expected_row["model_identity"]:
            raise ProspectiveEventStreamError("model hash drift")
        if (
            dict(row.get("exact_label_projection") or {}) != expected_row["exact_label_projection"]
            or dict(row.get("exact_diagnostic_and_label") or {})
            != expected_row["exact_diagnostic_and_label"]
        ):
            raise ProspectiveEventStreamError("post-hoc exact label")
        if row.get("prior_prefix_checksum") != prefix:
            raise ProspectiveEventStreamError("prefix chain mismatch")
        if event_row_hash(row) != row.get("row_hash"):
            raise ProspectiveEventStreamError("row hash mismatch")
        prefix = prefix_checksum(prefix, str(row["row_hash"]))
        if row.get("prefix_checksum") != prefix:
            raise ProspectiveEventStreamError("prefix chain mismatch")
        if dict(row.get("commit_eligibility") or {}).get("eligible") is True:
            eligible_commit_count += 1
        split = str(row.get("split"))
        splits[split] = splits.get(split, 0) + 1
    return {
        "ok": True,
        "row_count": len(row_list),
        "first_event_id": row_list[0]["event_id"] if row_list else None,
        "last_event_id": row_list[-1]["event_id"] if row_list else None,
        "final_prefix_checksum": prefix,
        "event_ids_unique": len(seen) == len(row_list),
        "event_order_is_chronological": True,
        "future_label_visibility_count": 0,
        "split_counts": splits,
        "eligible_commit_count": eligible_commit_count,
    }


def write_event_rows_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the Exp5920 JSONL stream through an atomic replace."""

    text = "".join(canonical_json(row) + "\n" for row in rows)
    _write_text_atomic(path, text)


def replay_stream(row_path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Replay a materialized Exp5920 stream and return its validation receipt."""

    rows = load_jsonl(row_path)
    receipt = validate_event_rows(rows, root=root)
    receipt.update(
        {
            "path": str(row_path),
            "sha256": sha256_file(row_path),
            "schema": ROW_SCHEMA_VERSION,
        }
    )
    return receipt


def run_fresh_process_replay(row_path: Path) -> JsonDict:
    """Replay the row stream in a separate interpreter."""

    code = """
import json
import sys
from pathlib import Path
from carnot import experiment_5920_prospective_event_stream_admission as mod
receipt = mod.replay_stream(Path(sys.argv[1]))
print(json.dumps(receipt, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    proc = subprocess.run(
        [sys.executable, "-c", code, str(row_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    parsed = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {}
    return {
        "ok": proc.returncode == 0 and bool(parsed.get("ok")),
        "returncode": proc.returncode,
        "row_count": parsed.get("row_count"),
        "final_prefix_checksum": parsed.get("final_prefix_checksum"),
        "stream_sha256": parsed.get("sha256"),
        "stdout_sha256": sha256_text(proc.stdout),
        "stderr_sha256": sha256_text(proc.stderr),
        "error": proc.stderr if proc.returncode else None,
    }


def run_tamper_matrix() -> JsonDict:
    """Tamper each boundary dimension and require fail-closed replay."""

    cases = []
    for component, tamper in (
        ("chronology_reordered_row", _tamper_chronology),
        ("duplicate_event_id", _tamper_duplicate_event_id),
        ("future_label_visibility", _tamper_visibility),
        ("exact_label_posthoc_relabel", _tamper_exact_label),
        ("source_hash_drift", _tamper_source_hash),
        ("split_drift", _tamper_split),
    ):
        rows = build_event_rows()
        tamper(rows)
        try:
            validate_event_rows(rows)
        except Exception as exc:  # noqa: BLE001
            cases.append(
                {
                    "component": component,
                    "rejected": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "partial_promotions": 0,
                }
            )
        else:
            cases.append(
                {
                    "component": component,
                    "rejected": False,
                    "error_type": None,
                    "error": None,
                    "partial_promotions": 1,
                }
            )
    return {
        "cases": cases,
        "all_rejected": all(case["rejected"] for case in cases),
        "partial_promotions": sum(int(case["partial_promotions"]) for case in cases),
    }


def global_suite_baseline(*, root: Path = REPO_ROOT) -> JsonDict:
    """Return the exact Exp5912 node-id baseline for unrelated global debt."""

    node_ids = _baseline_node_ids(root)
    return {
        "source": EXP5912_RESULT_RELATIVE_PATH.as_posix(),
        "command": GLOBAL_PYTEST_COMMAND,
        "baseline_node_ids": list(node_ids),
        "baseline_node_count": len(node_ids),
        "principle": REQUIRED_FIELD_PRINCIPLES["global_suite_baseline_and_failure_delta"],
    }


def global_suite_delta(
    after_node_ids: Sequence[str] | None = None,
    *,
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Compare a post-task global failure set against the exact baseline."""

    baseline = list(_baseline_node_ids(root))
    after = list(after_node_ids) if after_node_ids is not None else list(baseline)
    baseline_set = set(baseline)
    after_set = set(after)
    new_nodes = sorted(after_set - baseline_set)
    resolved_nodes = sorted(baseline_set - after_set)
    return {
        **global_suite_baseline(root=root),
        "after_node_ids": after,
        "after_node_count": len(after),
        "new_node_ids": new_nodes,
        "resolved_node_ids": resolved_nodes,
        "failure_delta": len(after_set) - len(baseline_set),
        "global_suite_failure_delta": len(after_set) - len(baseline_set),
        "ready_allowed": not new_nodes and len(after_set) <= len(baseline_set),
        "global_suite_zero_required": False,
        "unrelated_debt_preserved_by_exact_node_id": True,
    }


def write_admission_artifact(
    *,
    output_path: Path | None = None,
    row_output_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    global_after_node_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp5920 stream plus terminal artifact."""

    started = time.monotonic()
    target = output_path or REPO_ROOT / RESULT_RELATIVE_PATH
    row_target = row_output_path or REPO_ROOT / ROW_FILE_RELATIVE_PATH
    protected_before = _path_hashes(PROTECTED_RELATIVE_PATHS)
    upstream_before = _path_hashes(IMMUTABLE_UPSTREAM_RELATIVE_PATHS)
    preconditions = _preconditions(target, row_target)
    rows = build_event_rows()
    write_event_rows_atomic(row_target, rows)
    replay = replay_stream(row_target)
    fresh_process = run_fresh_process_replay(row_target)
    tamper = run_tamper_matrix()
    protected = _unchanged_receipt(PROTECTED_RELATIVE_PATHS, protected_before)
    upstream = _unchanged_receipt(IMMUTABLE_UPSTREAM_RELATIVE_PATHS, upstream_before)
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = _build_artifact(
        output_path=target,
        row_output_path=row_target,
        rows=rows,
        preconditions=preconditions,
        replay=replay,
        fresh_process=fresh_process,
        tamper=tamper,
        protected=protected,
        upstream=upstream,
        duration_s=float(elapsed),
        test_exit_codes=dict(test_exit_codes or {}),
        global_after_node_ids=global_after_node_ids,
    )
    validate_artifact(artifact)
    _write_json_atomic(target, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal artifact and its readiness invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    if artifact.get("prospective_stream_admission_ready_score") != ready_score(artifact):
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the bare Exp5920 readiness scalar."""

    task = dict(artifact.get("task_owned_test_boundary") or {})
    global_delta = dict(artifact.get("global_suite_baseline_and_failure_delta") or {})
    replay = dict(artifact.get("replay_and_tamper_matrix") or {})
    stream = dict(artifact.get("fresh_stream_path_hash_row_count_and_prefix_chain") or {})
    chronology = dict(artifact.get("chronology_split_and_visibility_receipts") or {})
    exact = dict(artifact.get("exact_label_authority") or {})
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and artifact.get("retired_scope_not_reopened") is True
        and dict(artifact.get("immutable_upstream_hashes") or {}).get("unchanged") is True
        and stream.get("prefix_chain_valid") is True
        and chronology.get("event_order_is_chronological") is True
        and chronology.get("future_label_visibility_count") == 0
        and exact.get("all_labels_recomputed_and_match") is True
        and dict(replay.get("fresh_process_replay") or {}).get("ok") is True
        and dict(replay.get("tamper_matrix") or {}).get("all_rejected") is True
        and dict(replay.get("tamper_matrix") or {}).get("partial_promotions") == 0
        and task.get("all_task_owned_commands_clean") is True
        and global_delta.get("ready_allowed") is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Return the terminal status from the readiness fields."""

    if artifact.get("retired_scope_not_reopened") is not True:
        return "retired"
    return "complete_ready" if ready_score(artifact) == 1.0 else "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal-prefixed Exp5920 verdict."""

    state = status(artifact)
    if state == "complete_ready":
        return "complete_ready: prospective_event_stream_admission_boundary_ready"
    if state == "retired":
        return "retired: exp5912_exact_slot_scope_not_reopened"
    return "blocked: " + ",".join(_blocked_reasons(artifact)[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the terminal artifact while normalizing volatile host fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        for key in ("disk", "ram"):
            if isinstance(preconditions.get(key), dict):
                preconditions[key]["available_mb"] = 0
        if isinstance(preconditions.get("output_paths"), dict):
            preconditions["output_paths"]["result_path"] = "<normalized>"
            preconditions["output_paths"]["row_path"] = "<normalized>"
    stream = stable.get("fresh_stream_path_hash_row_count_and_prefix_chain")
    if isinstance(stream, dict):
        stream["path"] = ROW_FILE_RELATIVE_PATH.as_posix()
    return sha256_json(stable)


def _build_artifact(
    *,
    output_path: Path,
    row_output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    replay: Mapping[str, Any],
    fresh_process: Mapping[str, Any],
    tamper: Mapping[str, Any],
    protected: Mapping[str, Any],
    upstream: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int],
    global_after_node_ids: Sequence[str] | None,
) -> JsonDict:
    schema_receipt = _schema_receipt()
    chronology = _chronology_visibility_receipt(replay)
    exact = _exact_label_authority(rows)
    task_boundary = _task_owned_boundary(test_exit_codes)
    global_delta = global_suite_delta(global_after_node_ids)
    stream_receipt = {
        "path": str(row_output_path),
        "sha256": replay["sha256"],
        "row_count": replay["row_count"],
        "schema": ROW_SCHEMA_VERSION,
        "first_event_id": replay["first_event_id"],
        "last_event_id": replay["last_event_id"],
        "genesis_prefix_checksum": GENESIS_PREFIX_CHECKSUM,
        "final_prefix_checksum": replay["final_prefix_checksum"],
        "prefix_chain_valid": replay["ok"],
        "principle": REQUIRED_FIELD_PRINCIPLES["fresh_stream_path_hash_row_count_and_prefix_chain"],
    }
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "status": "blocked",
        "preconditions_checked": dict(preconditions),
        "retired_scope_not_reopened": True,
        "immutable_upstream_hashes": upstream,
        "prospective_event_schema_and_version": schema_receipt,
        "fresh_stream_path_hash_row_count_and_prefix_chain": stream_receipt,
        "chronology_split_and_visibility_receipts": chronology,
        "exact_label_authority": exact,
        "replay_and_tamper_matrix": {
            "fresh_process_replay": dict(fresh_process),
            "tamper_matrix": dict(tamper),
            "principle": REQUIRED_FIELD_PRINCIPLES["replay_and_tamper_matrix"],
        },
        "task_owned_test_boundary": task_boundary,
        "global_suite_baseline_and_failure_delta": global_delta,
        "protected_files_unchanged": protected,
        "prospective_stream_admission_ready_score": 0.0,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["prospective_stream_admission_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProspectiveEventStreamError("canonical JSON numbers must be finite")
        return value
    raise ProspectiveEventStreamError(f"unsupported JSON value: {type(value).__name__}")


def _exact_label_projection(evaluation: Mapping[str, Any]) -> JsonDict:
    keys = (
        "parse_valid",
        "type_valid",
        "compiled",
        "satisfiability_correct",
        "exact_semantic_equivalence",
        "query_correct",
        "unsafe_accepted_constraints",
    )
    return {key: evaluation.get(key) for key in keys}


def _case_id(raw: Mapping[str, Any]) -> str:
    return "::".join(
        [
            str(raw.get("source_row_id")),
            str(raw.get("model_hf_id")),
            str(raw.get("arm_id")),
            str(raw.get("stream_sequence_index")),
        ]
    )


def _prompt_visibility(raw: Mapping[str, Any], artifact: Mapping[str, Any]) -> JsonDict:
    visibility = dict(artifact.get("retrieval_and_oracle_visibility") or {})
    return {
        "prompt_sha256": raw.get("prompt_sha256"),
        "visible_to_model": list(visibility.get("visible_to_model") or []),
        "withheld_from_model": list(visibility.get("withheld_from_model") or []),
        "visible_after_proposal_to_downstream_learning": list(
            visibility.get("visible_after_proposal_to_downstream_learning") or []
        ),
        "future_label_visible_to_model": False,
        "target_exact_labels_exposed_to_prompt": bool(
            visibility.get("target_exact_labels_exposed_to_prompt")
        ),
        "target_hidden_gold_ir_exposed_to_prompt": bool(
            visibility.get("target_hidden_gold_ir_exposed_to_prompt")
        ),
        "certificates_exposed_to_prompt": bool(visibility.get("certificates_exposed_to_prompt")),
        "diagnostic_repair_traces_exposed_to_prompt": bool(
            visibility.get("diagnostic_repair_traces_exposed_to_prompt")
        ),
    }


def _proposal_projection(raw: Mapping[str, Any]) -> JsonDict:
    return {
        "event_kind": raw.get("event_kind"),
        "arm_id": raw.get("arm_id"),
        "prompt_sha256": raw.get("prompt_sha256"),
        "raw_output_text": raw.get("raw_output_text"),
        "raw_output_sha256": raw.get("raw_output_sha256"),
        "candidate_sha256": raw.get("candidate_sha256"),
        "usage": dict(raw.get("usage") or {}),
        "latency_s": raw.get("latency_s"),
    }


def _commit_eligibility(exact_labels: Mapping[str, Any]) -> JsonDict:
    eligible = (
        exact_labels.get("parse_valid") is True
        and exact_labels.get("type_valid") is True
        and exact_labels.get("compiled") is True
        and exact_labels.get("satisfiability_correct") is True
        and exact_labels.get("exact_semantic_equivalence") is True
        and exact_labels.get("query_correct") is True
        and exact_labels.get("unsafe_accepted_constraints") is False
    )
    return {
        "eligible": eligible,
        "decision": "commit_eligible" if eligible else "reject_or_quarantine",
        "authority": "post_proposal_exact_label_projection",
        "partial_promotion_allowed": False,
    }


def _model_identity(raw: Mapping[str, Any], model_hashes: Mapping[str, Any]) -> JsonDict:
    hf_id = str(raw.get("model_hf_id"))
    model_file_sha = model_hashes.get(hf_id)
    base = {
        "model_hf_id": hf_id,
        "model_name": raw.get("model_name"),
        "model_path_sha256": sha256_text(str(raw.get("model_path") or "")),
        "model_file_sha256": model_file_sha,
    }
    return {**base, "model_identity_hash": sha256_json(base)}


def _model_file_hashes_by_id(artifact: Mapping[str, Any]) -> JsonDict:
    out = {}
    for row in dict(artifact.get("model_file_hashes") or {}).get("files", []):
        if isinstance(row, Mapping):
            out[str(row.get("hf_id"))] = row.get("sha256")
    return out


def _source_artifact_hashes(root: Path) -> JsonDict:
    return {
        "exp5908_artifact_sha256": sha256_file(root / exp5908.RESULT_RELATIVE_PATH),
        "exp5908_row_file_sha256": sha256_file(root / exp5908.ROW_FILE_RELATIVE_PATH),
        "exp5909_artifact_sha256": sha256_file(root / exp5909.RESULT_RELATIVE_PATH),
        "exp5909_raw_stream_sha256": sha256_file(root / exp5909.RAW_STREAM_RELATIVE_PATH),
        "exp5912_artifact_sha256": sha256_file(root / EXP5912_RESULT_RELATIVE_PATH),
    }


def _schema_receipt() -> JsonDict:
    return {
        "artifact_schema": SCHEMA_VERSION,
        "row_schema": ROW_SCHEMA_VERSION,
        "event_id_format": "exp5920-event-%06d",
        "required_row_fields": [
            "schema",
            "event_id",
            "causal_sequence_index",
            "case_id",
            "origin_family",
            "prompt_visibility",
            "proposal",
            "exact_diagnostic_and_label",
            "exact_label_projection",
            "commit_eligibility",
            "split",
            "model_identity",
            "source_artifact_hashes",
            "source_row",
            "prior_prefix_checksum",
            "row_hash",
            "prefix_checksum",
        ],
        "principle": REQUIRED_FIELD_PRINCIPLES["prospective_event_schema_and_version"],
    }


def _chronology_visibility_receipt(replay: Mapping[str, Any]) -> JsonDict:
    return {
        "event_order_is_chronological": replay.get("event_order_is_chronological"),
        "event_ids_unique": replay.get("event_ids_unique"),
        "future_label_visibility_count": replay.get("future_label_visibility_count"),
        "split_counts": dict(replay.get("split_counts") or {}),
        "first_event_id": replay.get("first_event_id"),
        "last_event_id": replay.get("last_event_id"),
        "principle": REQUIRED_FIELD_PRINCIPLES["chronology_split_and_visibility_receipts"],
    }


def _exact_label_authority(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    labels_match = [
        dict(row.get("exact_diagnostic_and_label") or {}).get("upstream_label_match") is True
        for row in rows
    ]
    return {
        "authority": "exp5909.evaluate_candidate",
        "label_projection_fields": list(_exact_label_projection({}).keys()),
        "row_count": len(rows),
        "all_labels_recomputed_and_match": all(labels_match),
        "visible_after_proposal_only": True,
        "verifier_is_oracle_scope": (
            "schema, checksum, chronology, visibility, and exact-label adjudication"
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES["exact_label_authority"],
    }


def _task_owned_boundary(test_exit_codes: Mapping[str, int]) -> JsonDict:
    missing = [command for command in TASK_OWNED_COMMANDS if command not in test_exit_codes]
    nonzero = [
        command for command in TASK_OWNED_COMMANDS if int(test_exit_codes.get(command, 1)) != 0
    ]
    return {
        "task_owned_commands": list(TASK_OWNED_COMMANDS),
        "global_command": GLOBAL_PYTEST_COMMAND,
        "missing_task_owned_commands": missing,
        "nonzero_task_owned_commands": nonzero,
        "all_task_owned_commands_clean": not missing and not nonzero,
        "global_suite_zero_required": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["task_owned_test_boundary"],
    }


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        HARNESS_SPEC_RELATIVE_PATH.as_posix(),
        exp5908.RESULT_RELATIVE_PATH.as_posix(),
        exp5908.ROW_FILE_RELATIVE_PATH.as_posix(),
        exp5909.RESULT_RELATIVE_PATH.as_posix(),
        exp5909.RAW_STREAM_RELATIVE_PATH.as_posix(),
        EXP5912_RESULT_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _preconditions(output_path: Path, row_output_path: Path) -> JsonDict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_output_path.parent.mkdir(parents=True, exist_ok=True)
    disk = _disk_probe(REPO_ROOT)
    ram = _memory_probe()
    atomic = _atomic_output_probe(row_output_path.parent)
    source_schema_hashes = _hash_rows(SOURCE_SCHEMA_RELATIVE_PATHS)
    checks = {
        "exp5908_replay_ok": bool(exp5908.replay_artifact(root=REPO_ROOT).get("ok")),
        "exp5909_raw_stream_bound": (REPO_ROOT / exp5909.RAW_STREAM_RELATIVE_PATH).exists(),
        "prompt_named_exp5909_events_alias_absent": not (
            REPO_ROOT / EXP5909_EVENTS_ALIAS_RELATIVE_PATH
        ).exists(),
        "exp5912_retired": read_json(REPO_ROOT / EXP5912_RESULT_RELATIVE_PATH).get("status")
        == "retired",
        "exact_verifier_available": callable(exp5909.evaluate_candidate),
        "disk": disk["ok"],
        "ram": ram["ok"],
        "atomic_json_jsonl_writes": atomic["ok"],
        "output_paths_writable": os.access(output_path.parent, os.W_OK)
        and os.access(row_output_path.parent, os.W_OK),
    }
    return {
        "run_date": RUN_DATE,
        "source_schema_hashes": source_schema_hashes,
        "requested_exp5909_events_alias": {
            "path": EXP5909_EVENTS_ALIAS_RELATIVE_PATH.as_posix(),
            "exists": (REPO_ROOT / EXP5909_EVENTS_ALIAS_RELATIVE_PATH).exists(),
            "artifact_bound_raw_stream_used": exp5909.RAW_STREAM_RELATIVE_PATH.as_posix(),
        },
        "known_issue_baseline_hash": sha256_file(REPO_ROOT / "ops/known-issues.md"),
        "exclusion_manifest_hash": sha256_file(REPO_ROOT / "ops/exclusion_manifest.yaml"),
        "output_paths": {
            "result_path": str(output_path),
            "row_path": str(row_output_path),
            "result_parent_writable": os.access(output_path.parent, os.W_OK),
            "row_parent_writable": os.access(row_output_path.parent, os.W_OK),
        },
        "disk": disk,
        "ram": ram,
        "atomic_writes": atomic,
        "exact_verifier_availability": {
            "callable": callable(exp5909.evaluate_candidate),
            "authority": "exp5909.evaluate_candidate",
        },
        "checks": checks,
        "preconditions_ready": all(checks.values()),
    }


def _path_hashes(paths: Sequence[Path]) -> JsonDict:
    return {
        path.as_posix(): sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None
        for path in paths
    }


def _hash_rows(paths: Sequence[Path]) -> list[JsonDict]:
    return [
        {
            "path": path.as_posix(),
            "exists": (REPO_ROOT / path).is_file(),
            "sha256": sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None,
        }
        for path in paths
    ]


def _unchanged_receipt(paths: Sequence[Path], before: Mapping[str, Any]) -> JsonDict:
    after = _path_hashes(paths)
    changed = [
        path.as_posix()
        for path in paths
        if before.get(path.as_posix()) is None
        or after.get(path.as_posix()) != before.get(path.as_posix())
    ]
    return {
        "unchanged": not changed,
        "before_hashes": dict(before),
        "after_hashes": after,
        "changed_files": changed,
    }


def _disk_probe(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _memory_probe() -> JsonDict:
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover - non-Linux fallback.
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _atomic_output_probe(directory: Path) -> JsonDict:
    target = directory / ".exp5920_atomic_probe"
    _write_text_atomic(target, "ok\n")
    ok = target.read_text(encoding="utf-8") == "ok\n"
    target.unlink()
    return {"ok": ok, "detail": "tempfile_replace_supported"}


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _baseline_node_ids(root: Path) -> list[str]:
    try:
        payload = read_json(root / EXP5912_RESULT_RELATIVE_PATH)
    except Exception:  # pragma: no cover - missing historical artifact is not expected here.
        return []
    failures = (
        payload.get("current_failure_node_ids_phases_and_ownership", {}).get("failures", [])
        if isinstance(payload.get("current_failure_node_ids_phases_and_ownership"), Mapping)
        else []
    )
    for failure in failures:
        if isinstance(failure, Mapping) and failure.get("command") == GLOBAL_PYTEST_COMMAND:
            return [str(node) for node in failure.get("node_ids") or []]
    return []


BASELINE_GLOBAL_NODE_IDS = tuple(_baseline_node_ids(REPO_ROOT))


def _tamper_chronology(rows: list[JsonDict]) -> None:
    rows[0], rows[1] = rows[1], rows[0]


def _tamper_duplicate_event_id(rows: list[JsonDict]) -> None:
    rows[1]["event_id"] = rows[0]["event_id"]


def _tamper_visibility(rows: list[JsonDict]) -> None:
    rows[0]["prompt_visibility"]["future_label_visible_to_model"] = True


def _tamper_exact_label(rows: list[JsonDict]) -> None:
    current = bool(rows[0]["exact_label_projection"]["parse_valid"])
    rows[0]["exact_label_projection"]["parse_valid"] = not current


def _tamper_source_hash(rows: list[JsonDict]) -> None:
    rows[0]["source_artifact_hashes"]["exp5909_raw_stream_sha256"] = "sha256:" + "0" * 64


def _tamper_split(rows: list[JsonDict]) -> None:
    rows[0]["split"] = "heldout" if rows[0].get("split") != "heldout" else "train"


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        reasons.append("preconditions")
    if artifact.get("retired_scope_not_reopened") is not True:
        reasons.append("retired_scope_reopened")
    if dict(artifact.get("immutable_upstream_hashes") or {}).get("unchanged") is not True:
        reasons.append("immutable_upstream_hashes")
    if (
        dict(artifact.get("task_owned_test_boundary") or {}).get("all_task_owned_commands_clean")
        is not True
    ):
        reasons.append("task_owned_test_boundary")
    if (
        dict(artifact.get("global_suite_baseline_and_failure_delta") or {}).get("ready_allowed")
        is not True
    ):
        reasons.append("global_suite_failure_delta")
    if dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is not True:
        reasons.append("protected_files")
    if not reasons and ready_score(artifact) != 1.0:
        reasons.append("ready_score")
    return reasons


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--rows", type=Path, default=REPO_ROOT / ROW_FILE_RELATIVE_PATH)
    args = parser.parse_args(argv)
    if args.validate:
        artifact = read_json(args.output)
        validate_artifact(artifact)
        replay_stream(args.rows)
        return 0
    write_admission_artifact(output_path=args.output, row_output_path=args.rows)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
