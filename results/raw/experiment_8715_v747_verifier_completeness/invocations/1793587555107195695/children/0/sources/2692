"""Build the REQ-REPORT-7111 forward ARC provenance canary artifact.

The run uses synthetic dictionaries at the real evaluator serialization and
dashboard read boundaries. It never opens game source, steps an environment,
loads a model, edits historical rows, or changes the solve registry.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping

from carnot.agentic.arc_eval_provenance import (
    NO_LLM_INFERENCE_SUBSTRATE,
    NOT_APPLICABLE,
    ArcEvalProvenanceInput,
    build_arc_eval_provenance,
    build_arc_level_claim_receipts,
    serialize_arc_evaluation_payload,
    validate_arc_evaluation_row,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = Path("results/experiment_7111_v624_arc_provenance_canary.json")
MODULE_PATH = Path("python/carnot/experiment_7111_v624_arc_provenance_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7111_v624_arc_provenance_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7111_v624_arc_provenance_canary.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EVALUATOR_PATH = Path("scripts/arc_leaderboard_eval.py")
DASHBOARD_PATH = Path("scripts/outer_loop_dashboard.py")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
BENCH_PATH = Path("ops/arc_bench_latest.json")
KNOWN_ISSUES_PATH = Path("ops/known-issues.md")
CONSUMER_LINT_PATH = Path("scripts/eval_run_consumer_field_lint.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
PROVENANCE_PATH = Path("python/carnot/agentic/arc_eval_provenance.py")

# REQ-ARC-WMTE-6642: the synthetic file handed to the real dashboard has the
# same required row surface as a future evaluator run. Declaring it here keeps
# the canary itself inside the producer/consumer field join.
EVAL_RUN_FIELDS_READ = (
    "per_game",
    "policy",
    "game",
    "levels",
    "solve_provenance",
    "arc_eval_provenance",
    "started_at",
    "finished_at",
    "actions",
    "frame_sequence",
    "attempt_receipt",
    "runtime_re_receipt",
    "arc_provenance_valid",
    "arc_headline_eligible",
    "arc_headline_ineligibility",
)

INFERENCE_SUBSTRATE = "deterministic ARC serialization and consumer canaries"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
RANDOM_SEED = 711120260907
POSITIVE_VERDICT = "complete: positive V624 ARC forward provenance canaries passed"
DISQUALIFIED_VERDICT = "complete: disqualified V624 ARC forward provenance canary failed"
BLOCKED_VERDICT = "complete: blocked V624 ARC forward provenance precondition unavailable"

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "preconditions_checked",
        "run_date",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "source_artifact_hashes",
        "rows",
        "canary_provenance_rows",
        "evaluator_writer_rows",
        "dashboard_consumer_rows",
        "missing_provenance_rejection_rows",
        "headline_eligibility_rows",
        "historical_rows_backfilled",
        "solve_provenance",
        "offline_reproduced",
        "arc_registry_hash_before",
        "arc_registry_hash_after",
        "arc_registry_delta",
        "arc_forward_provenance_ready_score",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
)

FIELD_PRINCIPLES = {
    "field_principles": "Every artifact field states the scientific reason it is retained.",
    "preconditions_checked": "Unavailable writer, consumer, registry, or output paths must block before a canary runs.",
    "run_date": "The execution date places this forward-only rule after the historical audit boundary.",
    "inference_substrate": "The substrate names deterministic serialization and consumption rather than a solve attempt.",
    "inference_substrate_class": "The class proves that no model load is being credited as ARC inference.",
    "execution_venue": "The venue identifies where the deterministic host checks executed.",
    "duration_s": "Measured wall time bounds the canary execution receipt.",
    "source_artifact_hashes": "Source hashes bind results to the exact writer, consumer, registry, tests, and contract.",
    "rows": "Small gate rows let an independent reader recompute readiness.",
    "canary_provenance_rows": "One row per synthetic case preserves the provenance class and decision.",
    "evaluator_writer_rows": "Writer rows prove accepted evidence was validated and annotated before bytes were emitted.",
    "dashboard_consumer_rows": "Consumer rows prove displayed live credit is separated from measured uncredited evidence.",
    "missing_provenance_rejection_rows": "Negative rows prove absent and invalid provenance cannot cross the writer boundary.",
    "headline_eligibility_rows": "Eligibility rows expose why each positive level count is or is not creditable.",
    "historical_rows_backfilled": "False preserves historical absence as an audit fact rather than manufactured provenance.",
    "solve_provenance": "Development-proxy labels this replay itself and prevents it from becoming a solve claim.",
    "offline_reproduced": "False confirms that this canary did not reproduce or claim a game level.",
    "arc_registry_hash_before": "The starting registry digest fixes the no-recredit baseline.",
    "arc_registry_hash_after": "The ending registry digest detects any forbidden registry mutation.",
    "arc_registry_delta": "A zero change indicator is required because the canary creates no solve evidence.",
    "arc_forward_provenance_ready_score": "Readiness requires every writer and consumer canary plus registry stability.",
    "random_seed": "A fixed canary identity makes the synthetic case set reproducible.",
    "reproducibility_checksum": "A canonical digest detects any changed result or gate decision.",
    "gate_check_summary": "The first exact gate outcome distinguishes completion, disqualification, and a blocked no-run.",
    "verifier_is_oracle": "False prevents structural hygiene from being mistaken for game correctness.",
    "verdict_class": "A closed verdict class separates readiness from blocked or failed execution.",
    "honest_verdict": "A terminal prefix gives automation one unambiguous result.",
}

SOURCE_PATHS = (
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
    EVALUATOR_PATH,
    DASHBOARD_PATH,
    PROVENANCE_PATH,
    REGISTRY_PATH,
    BENCH_PATH,
    KNOWN_ISSUES_PATH,
    CONSUMER_LINT_PATH,
    ADVERSARIAL_PATH,
)


def _sha256(path: Path) -> str:
    """Return a labeled digest of the exact file bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _checksum(payload: Mapping[str, Any]) -> str:
    """Hash every terminal artifact field except the digest itself."""

    projection = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def _preconditions(repo_root: Path, output_path: Path) -> list[dict[str, Any]]:
    """Check exact readable inputs and writable contract, test, and artifact parents."""

    rows: list[dict[str, Any]] = []
    for name, relative in (
        ("arc_evaluator_readable", EVALUATOR_PATH),
        ("dashboard_readable", DASHBOARD_PATH),
        ("solve_registry_readable", REGISTRY_PATH),
    ):
        path = repo_root / relative
        passed = path.is_file() and os.access(path, os.R_OK)
        rows.append(
            {
                "check": name,
                "expected_value": "readable file",
                "observed_value": "readable file" if passed else str(path),
                "passed": passed,
            }
        )
    for name, path in (
        ("test_path_writable", repo_root / TEST_PATH),
        ("spec_path_writable", repo_root / SPEC_PATH),
        ("artifact_path_writable", output_path),
    ):
        parent = path.parent
        passed = parent.is_dir() and os.access(parent, os.W_OK)
        rows.append(
            {
                "check": name,
                "expected_value": "writable parent directory",
                "observed_value": "writable parent directory" if passed else str(parent),
                "passed": passed,
            }
        )
    return rows


def _no_llm_record(solve_provenance: str) -> dict[str, Any]:
    """Build strict synthetic nested provenance without touching a model."""

    na = NOT_APPLICABLE
    return build_arc_eval_provenance(
        ArcEvalProvenanceInput(
            inference_substrate=NO_LLM_INFERENCE_SUBSTRATE,
            gpu_uuid=na,
            gpu_model=na,
            cuda_device=na,
            model_repository=na,
            model_filename=na,
            model_hash=na,
            n_ctx=na,
            server_binary=na,
            server_binary_hash=na,
            server_command_hash=na,
            endpoint=na,
            port=na,
            lease_id=na,
            lease_hash=na,
            lease_issued_at=na,
            lease_expires_at=na,
            lease_checked_at=na,
            request_count=0,
            completion_count=0,
            error_count=0,
            policy_hash="sha256:" + "7" * 64,
            factory_hash="sha256:" + "8" * 64,
            git_commit="9" * 40,
            solve_provenance=solve_provenance,
        )
    )


def _canary_row(solve_provenance: str, game: str, levels: int) -> dict[str, Any]:
    """Create one deterministic level-bearing row with same-row receipts."""

    frames = [
        {"action_index": 1, "levels_completed": 0},
        {"action_index": 2, "levels_completed": 1},
        {"action_index": 3, "levels_completed": 2},
        {"action_index": 4, "levels_completed": levels},
    ]
    row: dict[str, Any] = {
        "game": game,
        "solve_provenance": solve_provenance,
        "started_at": "2026-09-07T00:00:00+00:00",
        "finished_at": "2026-09-07T00:00:04+00:00",
        "actions": 4,
        "levels": levels,
        "frame_sequence": frames,
        "arc_eval_provenance": _no_llm_record(solve_provenance),
    }
    row.update(
        build_arc_level_claim_receipts(
            game=game,
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            actions=4,
            level_up_actions=list(range(1, levels + 1)),
            frame_sequence=frames,
            induction_attempts=[],
            level_induction_events=[],
        )
    )
    return row


def _gate(name: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Create one exact recomputable equality gate."""

    return {
        "check": name,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
    }


def _base_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hashes: dict[str, str | None],
    preconditions: list[dict[str, Any]],
    registry_before: str | None,
) -> dict[str, Any]:
    """Create every required field before blocked or completed evidence is added."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_hashes,
        "rows": [],
        "canary_provenance_rows": [],
        "evaluator_writer_rows": [],
        "dashboard_consumer_rows": [],
        "missing_provenance_rejection_rows": [],
        "headline_eligibility_rows": [],
        "historical_rows_backfilled": False,
        "solve_provenance": "development_proxy",
        "offline_reproduced": False,
        "arc_registry_hash_before": registry_before,
        "arc_registry_hash_after": registry_before,
        "arc_registry_delta": 0,
        "arc_forward_provenance_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }


def build_artifact(
    repo_root: Path = REPO_ROOT,
    *,
    execution_date: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run synthetic rows through the real writer and dashboard boundaries."""

    started = time.monotonic()
    repo_root = Path(repo_root)
    destination = output_path or (repo_root / DEFAULT_OUTPUT_PATH)
    preconditions = _preconditions(repo_root, destination)
    registry = repo_root / REGISTRY_PATH
    registry_before = _sha256(registry) if registry.is_file() else None
    source_hashes = {
        str(relative): (_sha256(repo_root / relative) if (repo_root / relative).is_file() else None)
        for relative in SOURCE_PATHS
    }
    artifact = _base_artifact(
        run_date=execution_date,
        duration_s=0.0,
        source_hashes=source_hashes,
        preconditions=preconditions,
        registry_before=registry_before,
    )
    failed_precondition = next((row for row in preconditions if not row["passed"]), None)
    if failed_precondition is not None:
        artifact["inference_substrate_class"] = "blocked_no_run"
        artifact["rows"] = [failed_precondition]
        artifact["gate_check_summary"] = {
            "status": "blocked",
            "failed_check": failed_precondition["check"],
            "expected_value": failed_precondition["expected_value"],
            "observed_value": failed_precondition["observed_value"],
        }
        artifact["duration_s"] = round(max(time.monotonic() - started, 0.001), 6)
        artifact["reproducibility_checksum"] = _checksum(artifact)
        return artifact

    try:
        scripts_path = str(repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        dashboard = importlib.import_module("outer_loop_dashboard")

        allowed = [
            _canary_row("live_agent_self_discovery", "canary-live", 2),
            _canary_row("development_proxy", "canary-proxy", 3),
            _canary_row("outer_loop_re", "canary-outer", 4),
        ]
        live_missing_receipt = _canary_row("live_agent_self_discovery", "canary-live-missing-re", 5)
        del live_missing_receipt["runtime_re_receipt"]
        serialized = serialize_arc_evaluation_payload(
            {"policy": "e3", "per_game": [*allowed, live_missing_receipt]}
        )
        emitted_rows = json.loads(serialized)["per_game"]
        writer_rows = [
            {
                "game": row["game"],
                "solve_provenance": row["solve_provenance"],
                "accepted": True,
                "arc_provenance_valid": row["arc_provenance_valid"],
                "arc_headline_eligible": row["arc_headline_eligible"],
                "arc_headline_ineligibility": row["arc_headline_ineligibility"],
            }
            for row in emitted_rows
        ]

        rejection_rows = []
        for case, mutate in (
            ("missing_top_level", lambda row: row.pop("solve_provenance")),
            ("invalid_top_level", lambda row: row.__setitem__("solve_provenance", "guessed")),
            ("missing_nested_record", lambda row: row.pop("arc_eval_provenance")),
        ):
            defective = copy.deepcopy(allowed[0])
            mutate(defective)
            error = None
            try:
                serialize_arc_evaluation_payload({"policy": "e3", "per_game": [defective]})
            except ValueError as exc:
                error = str(exc)
            rejection_rows.append({"case": case, "rejected": error is not None, "error": error})

        with tempfile.TemporaryDirectory(prefix="carnot-exp7111-") as temp_name:
            temp_repo = Path(temp_name)
            run_dir = temp_repo / "results" / "arc_leaderboard_eval_runs"
            run_dir.mkdir(parents=True)
            (run_dir / "serialized.json").write_text(serialized, encoding="utf-8")
            legacy = run_dir / "legacy-missing-provenance.json"
            legacy.write_text(
                json.dumps({"policy": "e3", "per_game": [{"game": "legacy-missing", "levels": 6}]}),
                encoding="utf-8",
            )
            legacy_before = _sha256(legacy)
            original_repo = dashboard.REPO
            try:
                dashboard.REPO = temp_repo
                dashboard_result = dashboard.generalization_levels()
            finally:
                dashboard.REPO = original_repo
            legacy_after = _sha256(legacy)

        expected_level_counts = {
            "development_proxy": 3,
            "live_agent_self_discovery": 7,
            "outer_loop_re": 4,
            "missing_or_invalid": 6,
        }
        dashboard_passed = (
            dashboard_result.get("headline_levels") == 2
            and dashboard_result.get("headline_games") == 1
            and dashboard_result.get("provenance_level_counts") == expected_level_counts
            and dashboard_result.get("provenance_rejected_games") == 4
        )
        dashboard_rows = [
            {
                "consumer": "outer_loop_dashboard.generalization_levels",
                "expected_headline_levels": 2,
                "observed_headline_levels": dashboard_result.get("headline_levels"),
                "expected_provenance_level_counts": expected_level_counts,
                "observed_provenance_level_counts": dashboard_result.get("provenance_level_counts"),
                "passed": dashboard_passed,
            },
            {
                "consumer": "legacy_read_byte_preservation",
                "hash_before": legacy_before,
                "hash_after": legacy_after,
                "passed": legacy_before == legacy_after,
            },
        ]
        canary_rows = [
            {
                "game": row["game"],
                "solve_provenance": row.get("solve_provenance"),
                "levels": row["levels"],
                "provenance_valid": validate_arc_evaluation_row(row).valid,
                "headline_eligible": validate_arc_evaluation_row(row).headline_eligible,
            }
            for row in emitted_rows
        ]
        headline_rows = [
            {
                "game": row["game"],
                "solve_provenance": row["solve_provenance"],
                "levels": row["levels"],
                "headline_eligible": validate_arc_evaluation_row(row).headline_eligible,
                "ineligibility": list(validate_arc_evaluation_row(row).headline_ineligibility),
            }
            for row in emitted_rows
        ]
        registry_after = _sha256(registry)
        artifact["arc_registry_hash_after"] = registry_after
        artifact["arc_registry_delta"] = int(registry_after != registry_before)
        artifact["canary_provenance_rows"] = canary_rows
        artifact["evaluator_writer_rows"] = writer_rows
        artifact["dashboard_consumer_rows"] = dashboard_rows
        artifact["missing_provenance_rejection_rows"] = rejection_rows
        artifact["headline_eligibility_rows"] = headline_rows
        artifact["rows"] = [
            _gate(
                "writer_accepts_all_allowed_provenance_values",
                ["development_proxy", "live_agent_self_discovery", "outer_loop_re"],
                sorted({row["solve_provenance"] for row in writer_rows[:3]}),
            ),
            _gate(
                "writer_rejects_missing_or_invalid_provenance",
                True,
                all(row["rejected"] for row in rejection_rows),
            ),
            _gate(
                "only_receipted_live_row_is_headline_eligible",
                [True, False, False, False],
                [row["headline_eligible"] for row in headline_rows],
            ),
            _gate("dashboard_preserves_provenance_groups", True, dashboard_passed),
            _gate("historical_row_bytes_unchanged", legacy_before, legacy_after),
            _gate("arc_registry_hash_unchanged", registry_before, registry_after),
        ]
    except Exception as exc:  # noqa: BLE001 - a canary fault must become a terminal artifact
        artifact["rows"] = [
            {
                "check": "deterministic_writer_consumer_canaries",
                "expected_value": "completed without exception",
                "observed_value": f"{type(exc).__name__}: {exc}",
                "passed": False,
            }
        ]

    first_failure = next((row for row in artifact["rows"] if not row["passed"]), None)
    ready = first_failure is None and artifact["arc_registry_delta"] == 0
    artifact["arc_forward_provenance_ready_score"] = int(ready)
    if ready:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = POSITIVE_VERDICT
        artifact["gate_check_summary"] = {
            "status": "passed",
            "failed_check": None,
            "expected_value": None,
            "observed_value": None,
        }
    else:
        failure = first_failure or {
            "check": "arc_registry_delta",
            "expected_value": 0,
            "observed_value": artifact["arc_registry_delta"],
        }
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["gate_check_summary"] = {
            "status": "disqualified",
            "failed_check": failure["check"],
            "expected_value": failure["expected_value"],
            "observed_value": failure["observed_value"],
        }
    artifact["duration_s"] = round(max(time.monotonic() - started, 0.001), 6)
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Independently recompute field, readiness, verdict, and checksum invariants."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("artifact fields do not exactly match REQ-REPORT-7111")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover every artifact field exactly")
    elif any(not isinstance(value, str) or not value.strip() for value in principles.values()):
        errors.append("every field principle must be non-empty")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue must be host")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("the deterministic canary must be development_proxy")
    if artifact.get("offline_reproduced") is not False:
        errors.append("the deterministic canary must not claim offline reproduction")
    if artifact.get("historical_rows_backfilled") is not False:
        errors.append("historical rows must not be backfilled")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("the verifier is not an ARC oracle")
    rows = artifact.get("rows")
    all_gates_pass = (
        isinstance(rows, list)
        and bool(rows)
        and all(isinstance(row, dict) and row.get("passed") is True for row in rows)
    )
    registry_unchanged = (
        isinstance(artifact.get("arc_registry_hash_before"), str)
        and artifact.get("arc_registry_hash_before") == artifact.get("arc_registry_hash_after")
        and artifact.get("arc_registry_delta") == 0
    )
    expected_ready = int(all_gates_pass and registry_unchanged)
    if artifact.get("arc_forward_provenance_ready_score") != expected_ready:
        errors.append("arc_forward_provenance_ready_score does not recompute from gates")
    verdict_class = artifact.get("verdict_class")
    if verdict_class == "positive":
        if expected_ready != 1 or artifact.get("inference_substrate_class") != "no_model_load":
            errors.append("positive verdict requires ready score one and no_model_load")
        if not str(artifact.get("honest_verdict", "")).startswith("complete:"):
            errors.append("positive honest_verdict lacks a terminal complete prefix")
    elif verdict_class == "blocked":
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked verdict requires blocked_no_run")
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, dict) or any(
            summary.get(key) is None for key in ("failed_check", "expected_value", "observed_value")
        ):
            errors.append("blocked verdict requires an exact failed precondition summary")
    elif verdict_class != "disqualified":
        errors.append("canary verdict_class must be positive, blocked, or disqualified")
    observed_checksum = artifact.get("reproducibility_checksum")
    if observed_checksum != _checksum(artifact):
        errors.append("reproducibility_checksum does not match canonical artifact bytes")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the canary, atomically write its terminal artifact, and validate it."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    artifact = build_artifact(REPO_ROOT, execution_date=args.date, output_path=output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"artifact": str(output), "validation_errors": errors}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "artifact": str(output),
                "arc_forward_provenance_ready_score": artifact[
                    "arc_forward_provenance_ready_score"
                ],
                "verdict_class": artifact["verdict_class"],
            },
            indent=2,
        )
    )
    return 0 if artifact["verdict_class"] == "positive" else 1


if __name__ == "__main__":
    raise SystemExit(main())
