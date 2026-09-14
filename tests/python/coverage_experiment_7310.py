"""Exercise defensive and CLI glue for Exp7310 scoped coverage.

The behavior assertions live in test_experiment_7310_v642_factor_prototype.py.
This script covers rejection branches and entrypoint wiring without adding a
second source of scientific expectations.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from typing import Iterator

from carnot import experiment_7310_v642_factor_prototype as exp


@contextmanager
def rejected(message: str = "") -> Iterator[None]:
    """Require one defensive branch to reject instead of changing state."""

    try:
        yield
    except (exp.FactorRevisionRejected, RuntimeError, ValueError) as error:
        assert not message or message in str(error)
    else:
        raise AssertionError(f"expected rejection: {message}")


def masks() -> dict[str, int]:
    """Return a complete finite mask set for defensive probes."""

    return dict.fromkeys(exp.FAMILIES, exp.FULL_MASK)


def event(event_id: str = "e", *, source: int = 0) -> dict[str, object]:
    """Return one valid public event."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 10,
        "chronology_index": source,
    }


def release(event_id: str = "e", *, source: int = 0, due: int = 4) -> dict[str, object]:
    """Return one matching released witness."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 10,
        "observed_label": "accept",
        "source_index": source,
        "release_index": due,
    }


with tempfile.TemporaryDirectory(prefix="carnot-exp7310-coverage-", dir="/tmp") as name:
    root = Path(name)
    exp.ExperimentPaths.defaults()
    exp._progress(0, "coverage", "entrypoint glue")
    assert exp._sha256_path(root / "missing") is None
    assert exp._load_object(root / "missing") == {}
    malformed = root / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert exp._load_object(malformed) == {}

    immutable = root / "immutable.json"
    first = exp._write_immutable(immutable, {"value": 1})
    second = exp._write_immutable(immutable, {"value": 1})
    assert first["reused"] is False and second["reused"] is True
    with rejected("immutable_evidence_mismatch"):
        exp._write_immutable(immutable, {"value": 2})
    with rejected("unknown_family"):
        exp.evaluator_exact_label("unknown", 0, 0)

    with rejected("invalid_mask_families"):
        exp.FactorLocalController.from_masks({})
    invalid_masks = masks()
    invalid_masks["lower_bound"] = 0
    with rejected("invalid_survivor_mask"):
        exp.FactorLocalController.from_masks(invalid_masks)
    with rejected("invalid_controller_mode"):
        exp.FactorLocalController.from_masks(masks(), mode="unknown")
    with rejected("factor_memory_cap"):
        exp.FactorLocalController.from_masks(masks(), memory_cap_bytes=1)

    base = exp.FactorLocalController.from_masks(masks())
    assert base.state_dict()["schema"] == exp.STATE_SCHEMA
    assert base.memory_usage()["within_cap"] is True
    with rejected("invalid_family"):
        base.family_state("unknown")
    with rejected("invalid_public_event"):
        base.predict({"family_id": "lower_bound", "numeric_value": True})
    with rejected("private_authority_in_prediction"):
        base.predict({**event("private"), "exact_label": "accept"})
    with rejected("invalid_public_event"):
        base.seal_prediction({"family_id": "lower_bound", "numeric_value": 1}, release_index=4)
    with rejected("invalid_release_schedule"):
        base.seal_prediction(event("past", source=4), release_index=3)
    base.seal_prediction(event("duplicate"), release_index=4)
    with rejected("duplicate_pending_release"):
        base.seal_prediction(event("duplicate"), release_index=4)
    with rejected("invalid_release"):
        base.apply_release({}, current_index=4)
    with rejected("release_without_sealed_prediction"):
        base.apply_release(release("missing"), current_index=4)
    mismatch = release("duplicate")
    mismatch["numeric_value"] = 11
    with rejected("release_receipt_mismatch"):
        base.apply_release(mismatch, current_index=4)
    with rejected("release_not_due"):
        base.apply_release(release("duplicate"), current_index=3)

    duplicate_state = base.state_dict()
    duplicate_state["used_release_ids"] = [duplicate_state["pending_releases"][0]["evidence_id"]]
    duplicate = exp.FactorLocalController.from_state(duplicate_state)
    with rejected("duplicate_release"):
        duplicate.apply_release(release("duplicate"), current_index=4)

    bad_states = []
    state = base.state_dict()
    state["schema"] = "bad"
    bad_states.append(state)
    state = base.state_dict()
    state["families"].pop("upper_bound")
    bad_states.append(state)
    state = base.state_dict()
    state["families"]["lower_bound"]["survivor_mask"] = 0
    bad_states.append(state)
    state = base.state_dict()
    state["used_release_ids"] = ["x"] * (exp.DEDUPLICATION_LIMIT + 1)
    bad_states.append(state)
    state = base.state_dict()
    state.pop("families")
    bad_states.append(state)
    for state in bad_states:
        with rejected("invalid_factor_state"):
            exp.FactorLocalController.from_state(state)
    state = base.state_dict()
    state["memory_cap_bytes"] = 1
    with rejected("factor_memory_cap"):
        exp.FactorLocalController.from_state(state)
    with rejected("invalid_factor_state"):
        exp.FactorLocalController.load(root / "absent.json")

    impossible_masks = masks()
    impossible_masks["lower_bound"] = 1 << 5
    impossible = exp.FactorLocalController.from_masks(impossible_masks)
    impossible_event = {**event("impossible"), "numeric_value": 32}
    impossible.seal_prediction(impossible_event, release_index=4)
    impossible_release = {**release("impossible"), "numeric_value": 32, "observed_label": "reject"}
    with rejected("no_consistent_suffix"):
        impossible.apply_release(impossible_release, current_index=4)

    independence_guard = exp.FactorLocalController.from_masks(impossible_masks)
    independence_guard.seal_prediction(event("guard"), release_index=4)
    original_family_bytes = independence_guard.family_bytes
    calls = 0

    def changed_family_bytes(family: str) -> bytes:
        global calls
        calls += 1
        data = original_family_bytes(family)
        return data if calls <= len(exp.FAMILIES) - 1 else data + b"x"

    independence_guard.family_bytes = changed_family_bytes  # type: ignore[method-assign]
    with rejected("unaffected_factor_changed"):
        independence_guard.apply_release(release("guard"), current_index=4)

    cap_probe = exp.FactorLocalController.from_masks(masks())
    cap_probe.seal_prediction(event("update-cap"), release_index=4)
    cap_state = cap_probe.state_dict()
    cap_state["memory_cap_bytes"] = len(cap_probe.state_bytes()) + 4
    cap_probe = exp.FactorLocalController.from_state(cap_state)
    with rejected("factor_memory_cap"):
        cap_probe.apply_release(release("update-cap"), current_index=4)

    with rejected("stale_rollback"):
        base.rollback({"receipt_id": "none"})
    rollback_controller = exp.FactorLocalController.from_masks(impossible_masks)
    rollback_controller.seal_prediction(event("rollback"), release_index=4)
    receipt = rollback_controller.apply_release(
        {**release("rollback"), "observed_label": "reject"}, current_index=4
    )
    corrupt = rollback_controller.state_dict()
    corrupt["rollback"]["parent_bytes_zlib_b64"] = "bad"
    corrupt_controller = exp.FactorLocalController.from_state(corrupt)
    with rejected("invalid_rollback"):
        corrupt_controller.rollback(receipt)
    corrupt = rollback_controller.state_dict()
    corrupt["rollback"]["parent_hash"] = "sha256:" + "0" * 64
    corrupt_controller = exp.FactorLocalController.from_state(corrupt)
    with rejected("invalid_rollback"):
        corrupt_controller.rollback(receipt)

    disabled = exp.FactorPipelineHook(root / "disabled")
    assert disabled.predict(event()) == "abstain"
    with rejected("factor_hook_disabled"):
        disabled.pre_label(event(), release_index=4)
    with rejected("factor_hook_disabled"):
        disabled.release(release(), current_index=4)

    with rejected("invalid_stream_kind"):
        exp.build_stream_views("bad")
    views = exp.build_stream_views("development")
    broken = deepcopy(views)
    broken.public.pop()
    assert "public_count" in exp.stream_conformance_errors(broken, "development")
    assert "stream_kind" in exp.stream_conformance_errors(broken, "bad")
    with rejected("incomplete_warmup"):
        exp._warmup_masks([], "none")
    inconsistent = [
        dict(row)
        for row in views.releases
        if row["stream_id"] == "development-01" and row["role"] == "warmup"
    ]
    lower = [row for row in inconsistent if row["family_id"] == "lower_bound"]
    lower[0]["numeric_value"] = 10
    lower[0]["observed_label"] = "accept"
    lower[1]["numeric_value"] = 10
    lower[1]["observed_label"] = "reject"
    with rejected("inconsistent_warmup"):
        exp._warmup_masks(inconsistent, "development-01")

    assert exp._task_identity("{") == {}
    assert exp._task_identity("milestone: wrong\ntasks: []\n") == {}
    assert exp._task_identity(f"milestone: {exp.MILESTONE}\ntasks: []\n") == {}
    summary = exp.gate_summary([exp._gate("x", "u", "f", 1, 0, False, "purpose")])
    assert summary["first_failure"]["check"] == "x"

    blocked_paths = exp.ExperimentPaths.under(root / "blocked")
    blocked = exp.build_and_seal(exp.REPO_ROOT, blocked_paths, progress=True)
    assert blocked["status"] == "blocked"
    with rejected("blocked_artifact_without_failure"):
        exp.build_blocked_artifact([], {})

    complete_root = root / "complete"
    complete_paths = exp.ExperimentPaths.under(complete_root)
    complete_paths.historical_artifact.parent.mkdir(parents=True, exist_ok=True)
    complete_paths.historical_artifact.write_bytes(
        (exp.REPO_ROOT / exp.HISTORICAL_ARTIFACT).read_bytes()
    )
    artifact = exp.build_and_seal(exp.REPO_ROOT, complete_paths, progress=True)
    assert artifact["status"] == "complete"
    assert exp._receipt_error({}) is True
    with rejected("validation_receipt_schema"):
        exp.attach_validation_receipts(artifact, [{}])
    invalid = deepcopy(artifact)
    invalid["schema"] = "bad"
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    with rejected("artifact_validation_failed"):
        exp.write_artifact(root / "invalid.json", invalid)

    for key, value in (
        ("MODEL_SPECS", ["bad"]),
        ("inference_substrate", "bad"),
        ("verifier_is_oracle", False),
        ("verdict_class", "positive"),
        ("field_principles", {}),
        ("validation_receipts", [{}]),
        ("status", "partial"),
        ("rows", []),
        ("stream_manifest", {}),
        ("factor_state_schema", {}),
        ("hardware_path", {"future_acceleration_target_verified": True}),
    ):
        changed = deepcopy(artifact)
        changed[key] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert exp.validate_artifact(changed)
    blocked_changed = deepcopy(blocked)
    blocked_changed["rows"] = [{}]
    blocked_changed["reproducibility_checksum"] = exp.reproducibility_checksum(blocked_changed)
    assert "blocked_contract" in exp.validate_artifact(blocked_changed)

    receipt_row, log = exp._command_receipt(
        [str(exp.REPO_ROOT / ".venv/bin/python"), "-c", "print('ok')"],
        scope="coverage",
    )
    assert receipt_row["exit_code"] == 0 and log == "ok\n"
    assert exp._ControllerMemoryView(immutable).state_hash().startswith("sha256:")
    assert exp._validation_commands(complete_paths.terminal_candidate)
    assert exp._parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE

    exp.write_artifact(complete_paths.terminal_candidate, artifact)
    assert (
        exp.main(["--date", exp.RUN_DATE, "--validate", str(complete_paths.terminal_candidate)])
        == 0
    )

    main_root = root / "main"
    main_root.mkdir()
    (main_root / exp.HISTORICAL_ARTIFACT.name).write_bytes(
        (exp.REPO_ROOT / exp.HISTORICAL_ARTIFACT).read_bytes()
    )
    original_commands = exp._validation_commands
    exp._validation_commands = lambda _candidate: [
        (
            [str(exp.REPO_ROOT / ".venv/bin/python"), "-c", "print('main-check')"],
            "coverage main validation",
        )
    ]
    try:
        assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(main_root)]) == 0
        assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(root / "main-blocked")]) == 0
    finally:
        exp._validation_commands = original_commands

    failed_root = root / "main-failed"
    failed_root.mkdir()
    (failed_root / exp.HISTORICAL_ARTIFACT.name).write_bytes(
        (exp.REPO_ROOT / exp.HISTORICAL_ARTIFACT).read_bytes()
    )
    exp._validation_commands = lambda _candidate: [
        (
            [str(exp.REPO_ROOT / ".venv/bin/python"), "-c", "raise SystemExit(2)"],
            "coverage failed validation",
        )
    ]
    try:
        with rejected("focused_validation_failed"):
            exp.main(["--date", exp.RUN_DATE, "--output-root", str(failed_root)])
    finally:
        exp._validation_commands = original_commands

    validation_failure_root = root / "build-validation-failure"
    failure_paths = exp.ExperimentPaths.under(validation_failure_root)
    failure_paths.historical_artifact.parent.mkdir(parents=True, exist_ok=True)
    failure_paths.historical_artifact.write_bytes(
        (exp.REPO_ROOT / exp.HISTORICAL_ARTIFACT).read_bytes()
    )
    original_validate = exp.validate_artifact
    exp.validate_artifact = lambda *_args, **_kwargs: ["injected"]
    try:
        with rejected("artifact_validation_failed"):
            exp.build_and_seal(exp.REPO_ROOT, failure_paths)
    finally:
        exp.validate_artifact = original_validate

assert True
