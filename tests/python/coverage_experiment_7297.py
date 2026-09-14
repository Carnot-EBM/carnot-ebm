"""Execute Exp7297 paths for scoped coverage measurement."""

from copy import deepcopy
from contextlib import suppress
from pathlib import Path
import tempfile

from carnot import experiment_7297_v641_mixture_audit as audit


def main() -> None:
    """Exercise measured, blocked, validation and helper paths."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp7297-coverage-") as directory:
        root = Path(directory)
        paths = audit.ExperimentPaths.under(root)
        artifact = audit.build_and_seal(
            audit.REPO_ROOT,
            paths,
            stream_ids=("evaluation-01", "evaluation-13"),
            bootstrap_draws=20,
            progress=True,
        )
        direct = audit._cold_replay_impl(
            audit.REPO_ROOT,
            ("evaluation-01", "evaluation-13"),
        )
        assert direct["cold_replay_parity"]["mismatch_rows"] == []
        capture = audit._load_object(audit.REPO_ROOT / audit.UPSTREAM_CAPTURE)
        contract = audit._load_object(audit.REPO_ROOT / audit.UPSTREAM_CONTRACT)
        views = audit._load_views(audit.REPO_ROOT, contract)
        step_path = audit._resolve(
            audit.REPO_ROOT,
            capture["raw_evidence_receipts"]["step_rows"]["path"],
        )
        feedback_path = audit._resolve(
            audit.REPO_ROOT,
            capture["raw_evidence_receipts"]["feedback_update_rows"]["path"],
        )
        producer_steps = [
            row
            for row in audit.fixture._read_jsonl(step_path)
            if row["stream_id"] == "evaluation-01"
        ]
        producer_feedback = [
            row
            for row in audit.fixture._read_jsonl(feedback_path)
            if row["stream_id"] == "evaluation-01"
        ]
        step_map = {(row["arm"], row["chronology_index"]): row for row in producer_steps}
        feedback_map = {row["release_index"]: row for row in producer_feedback}
        step_map[("fixed_share_mixture", 128)] = deepcopy(step_map[("fixed_share_mixture", 128)])
        step_map[("fixed_share_mixture", 128)]["prediction"] = "wrong"
        step_map[("fixed_share_mixture", 128)]["state_hash_before_prediction"] = "wrong"
        first_release = min(feedback_map)
        feedback_map[first_release] = deepcopy(feedback_map[first_release])
        feedback_map[first_release]["arm_updates"]["fixed_share_mixture"]["state_hash_after"] = (
            "wrong"
        )
        mismatched = audit._replay_stream(
            views,
            "evaluation-01",
            step_map,
            feedback_map,
        )
        assert mismatched["prediction_mismatches"] == 1
        assert mismatched["state_mismatches"] == 1
        assert mismatched["transition_mismatches"] == 1
        audit.ExperimentPaths.defaults()
        audit.validate_artifact(
            artifact,
            expected_stream_ids=("evaluation-01", "evaluation-13"),
            check_files=True,
        )
        receipt = {
            "command": "coverage",
            "exit_code": 0,
            "classification": "passed",
            "duration_s": 0.1,
            "log_sha256": "sha256:" + "0" * 64,
        }
        sealed = audit.attach_validation_receipts(artifact, [receipt])
        audit.write_artifact(
            paths.artifact,
            sealed,
            expected_stream_ids=("evaluation-01", "evaluation-13"),
        )
        checks, hashes, _, _ = audit.collect_preconditions(
            audit.REPO_ROOT,
            paths,
            capture_path=root / "missing.json",
        )
        blocked = audit.build_blocked_artifact(
            checks,
            hashes,
            ("evaluation-01",),
            started_at="2026-09-14T00:00:00+00:00",
            duration_s=0.01,
        )
        audit.validate_artifact(blocked)
        changed = dict(artifact)
        changed["schema"] = "wrong"
        audit.validate_artifact(changed)
        with suppress(ValueError):
            audit.attach_validation_receipts({}, [{"command": "invalid"}])
        with suppress(ValueError):
            audit.write_artifact(paths.artifact, changed)
        original_capture = audit.UPSTREAM_CAPTURE
        audit.UPSTREAM_CAPTURE = root / "missing-for-build.json"
        audit.build_and_seal(audit.REPO_ROOT, paths, progress=False)
        audit.UPSTREAM_CAPTURE = original_capture
        audit.main(
            [
                "--date",
                audit.RUN_DATE,
                "--audit-worker",
                "--output-root",
                str(root / "worker"),
                "--stream-ids",
                "evaluation-01",
            ]
        )
        audit.main(
            [
                "--date",
                audit.RUN_DATE,
                "--validate",
                str(paths.artifact),
            ]
        )
        original_build = audit.build_and_seal
        original_write = audit.write_artifact
        original_commands = audit._validation_commands
        original_receipt = audit.fixture._command_receipt
        audit.build_and_seal = lambda *args, **kwargs: blocked
        audit.write_artifact = lambda *args, **kwargs: {"sha256": "sha256:" + "1" * 64}
        audit.main(["--date", audit.RUN_DATE, "--output-root", str(root / "blocked-main")])
        audit.build_and_seal = lambda *args, **kwargs: artifact
        audit._validation_commands = lambda candidate: [["coverage-command"]]
        audit.fixture._command_receipt = lambda command: (receipt.copy(), "coverage log")
        audit.main(["--date", audit.RUN_DATE, "--output-root", str(root / "complete-main")])
        failed_receipt = {**receipt, "exit_code": 1, "classification": "failed"}
        audit.fixture._command_receipt = lambda command: (failed_receipt.copy(), "failed log")
        with suppress(RuntimeError):
            audit.main(["--date", audit.RUN_DATE, "--output-root", str(root / "failed-main")])
        audit.build_and_seal = original_build
        audit.write_artifact = original_write
        audit._validation_commands = original_commands
        audit.fixture._command_receipt = original_receipt
        audit.derive_terminal_scores(False, True, True)
        audit.derive_terminal_scores(True, True, False)
        audit._validation_commands(paths.terminal_candidate)
        audit._parse_args(["--date", audit.RUN_DATE])


if __name__ == "__main__":
    main()
