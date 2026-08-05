"""REQ-REPORT-6143 tests for experiment artifact isolation.

Spec refs:
  REQ-REPORT-6143
  SCENARIO-REPORT-6143-PRODUCTION-DEFAULT
  SCENARIO-REPORT-6143-PYTEST-TEMP-ROOT
  SCENARIO-REPORT-6143-INVALID-OVERRIDE
  SCENARIO-REPORT-6143-DIRECT-WRITER-COMPATIBILITY
  SCENARIO-REPORT-6143-ATTEMPTED-TRACKED-WRITE-DETECTION
  SCENARIO-REPORT-6143-QUARANTINE-PRESERVATION
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from carnot import experiment_artifacts as artifact_mod
from carnot.experiment_artifacts import (
    ARTIFACT_ROOT_ENV,
    ArtifactPathError,
    atomic_write_json,
    atomic_write_text,
    artifact_output_root,
    resolve_experiment_artifact_path,
    validate_artifact_output_root,
)
from carnot.paths import repo_root
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.testing import tracked_results_guard


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _json_fields(path: Path, fields: tuple[str, ...]) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {field: payload.get(field) for field in fields if field in payload}


def _valid_override_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifact-root"
    root.mkdir()
    return root


def test_production_default_keeps_historical_results_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-6143-PRODUCTION-DEFAULT: legacy paths resolve to production results."""
    monkeypatch.delenv(ARTIFACT_ROOT_ENV, raising=False)
    production_root = repo_root() / "results"

    assert artifact_output_root() == production_root.resolve()
    assert (
        resolve_experiment_artifact_path("results/experiment_6143_probe.json")
        == production_root / "experiment_6143_probe.json"
    )


def test_valid_test_override_redirects_legacy_results_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-PYTEST-TEMP-ROOT: test override owns result writes."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))

    resolved = resolve_experiment_artifact_path("results/experiment_6143_redirected.json")

    assert resolved == override / "experiment_6143_redirected.json"
    assert str(resolved).startswith(str(override))


def test_invalid_override_targets_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-INVALID-OVERRIDE: broad and repository roots are rejected."""
    broad_tmp = Path(tempfile.gettempdir()).resolve()
    invalid_roots = ["", str(repo_root()), str(repo_root() / "results"), str(broad_tmp)]

    for invalid in invalid_roots:
        monkeypatch.setenv(ARTIFACT_ROOT_ENV, invalid)
        with pytest.raises(ArtifactPathError):
            artifact_output_root()

    valid = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(valid))
    assert artifact_output_root() == valid.resolve()


def test_missing_and_non_temp_override_targets_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6143-INVALID-OVERRIDE: missing and non-temp dirs are rejected."""
    missing = tmp_path / "missing"

    with pytest.raises(ArtifactPathError):
        validate_artifact_output_root(missing)
    with pytest.raises(ArtifactPathError):
        validate_artifact_output_root(Path.home())


def test_unresolvable_override_path_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-6143-INVALID-OVERRIDE: path resolution errors are reported."""

    class _BadPath:
        def __init__(self, raw: object) -> None:
            self.raw = raw

        def expanduser(self) -> "_BadPath":
            return self

        def resolve(self) -> Path:
            raise OSError(f"cannot resolve {self.raw}")

    monkeypatch.setattr(artifact_mod, "Path", _BadPath)
    with pytest.raises(ArtifactPathError):
        validate_artifact_output_root("bad")


def test_empty_and_root_only_artifact_paths_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-INVALID-OVERRIDE: empty artifact names cannot resolve."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))

    with pytest.raises(ArtifactPathError):
        resolve_experiment_artifact_path("")
    with pytest.raises(ArtifactPathError):
        resolve_experiment_artifact_path("results")


def test_traversal_and_symlink_escape_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-6143-INVALID-OVERRIDE: path escapes never resolve."""
    override = _valid_override_root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (override / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))

    with pytest.raises(ArtifactPathError):
        resolve_experiment_artifact_path("../outside.json")
    with pytest.raises(ArtifactPathError):
        resolve_experiment_artifact_path("results/../outside.json")
    with pytest.raises(ArtifactPathError):
        resolve_experiment_artifact_path("escape/leak.json")


def test_absolute_paths_map_only_when_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-DIRECT-WRITER-COMPATIBILITY: absolute path handling is narrow."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    production_path = repo_root() / "results" / "experiment_6143_absolute.json"
    already_redirected = override / "already.json"

    assert (
        resolve_experiment_artifact_path(production_path)
        == override / "experiment_6143_absolute.json"
    )
    assert resolve_experiment_artifact_path(already_redirected) == already_redirected
    with pytest.raises(ArtifactPathError):
        resolve_experiment_artifact_path(tmp_path / "elsewhere.json")

    monkeypatch.delenv(ARTIFACT_ROOT_ENV, raising=False)
    external = tmp_path / "external.json"
    assert resolve_experiment_artifact_path(
        external, allow_external_absolute=True
    ) == external.resolve(strict=False)


def test_atomic_json_and_text_replacement_use_redirected_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-DIRECT-WRITER-COMPATIBILITY: atomic replacement is redirected."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))

    json_path = atomic_write_json("results/experiment_6143_atomic.json", {"value": 1})
    json_path_2 = atomic_write_json("results/experiment_6143_atomic.json", {"value": 2})
    text_path = atomic_write_text("results/experiment_6143_atomic.txt", "ok\n")

    assert json_path == json_path_2 == override / "experiment_6143_atomic.json"
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 2}
    assert text_path.read_text(encoding="utf-8") == "ok\n"
    assert list(override.glob("*.tmp")) == []
    assert not (repo_root() / "results" / "experiment_6143_atomic.json").exists()


def test_atomic_write_cleans_temp_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-DIRECT-WRITER-COMPATIBILITY: failed atomic writes clean temp files."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))

    def _boom(src: Path | str, dst: Path | str) -> None:
        raise RuntimeError(f"replace failed for {src} -> {dst}")

    monkeypatch.setattr(artifact_mod.os, "replace", _boom)
    with pytest.raises(RuntimeError):
        atomic_write_text("results/experiment_6143_atomic_fail.txt", "nope\n")

    assert list(override.glob("*.tmp")) == []


def test_deliverable_guard_checks_redirected_relative_results_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-DIRECT-WRITER-COMPATIBILITY: guard follows resolver paths."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))

    atomic_write_json("results/experiment_6143_guard.json", {"status": "complete"})

    DeliverableGuard("results/experiment_6143_guard.json").assert_written()
    assert (override / "experiment_6143_guard.json").exists()


def test_experiment_template_uses_redirected_output_and_checkpoint_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-PYTEST-TEMP-ROOT: template paths use the test root."""
    from scripts.experiment_template import ExperimentTemplate

    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))

    template = ExperimentTemplate(
        6143, "Artifact isolation", "results/experiment_6143_template.json"
    )
    template.checkpoint_save({"ok": True}, step=1)

    assert template._output_path == override / "experiment_6143_template.json"
    assert (override / "checkpoints" / "experiment_6143" / "checkpoint.json").exists()


def test_tracked_results_guard_observes_real_forbidden_attempt() -> None:
    """SCENARIO-REPORT-6143-ATTEMPTED-TRACKED-WRITE-DETECTION: legacy writes are caught."""
    tracked_results_guard.install()
    tracked_results_guard.clear_violations()
    target = repo_root() / "results" / "experiment_1938_nrgpt_loss_probe.json"
    before = _sha256(target)

    try:
        with pytest.raises(tracked_results_guard.TrackedResultWriteError):
            target.write_text("forbidden\n", encoding="utf-8")
        violations = tracked_results_guard.recorded_violations()
        assert violations
        assert violations[-1]["path"] == "results/experiment_1938_nrgpt_loss_probe.json"
        assert _sha256(target) == before
    finally:
        tracked_results_guard.clear_violations()


def test_tracked_results_guard_helper_and_event_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-6143-ATTEMPTED-TRACKED-WRITE-DETECTION: guard branches are live."""
    target = repo_root() / "results" / "experiment_1938_nrgpt_loss_probe.json"
    tracked_results_guard.clear_violations()

    assert str(target.resolve()) in tracked_results_guard._tracked_result_paths()
    monkeypatch.setattr(
        tracked_results_guard.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("git missing")),
    )
    assert tracked_results_guard._tracked_result_paths() == frozenset()
    monkeypatch.undo()
    assert tracked_results_guard._is_write_intent("w", None) is True
    assert tracked_results_guard._is_write_intent("r", None) is False
    assert tracked_results_guard._is_write_intent(None, artifact_mod.os.O_CREAT) is True
    assert tracked_results_guard._is_write_intent(None, None) is True
    assert tracked_results_guard._looks_like_results_path(str(target)) is True
    assert tracked_results_guard._looks_like_results_path("results/example.json") is True
    assert tracked_results_guard._looks_like_results_path("elsewhere/example.json") is False
    assert tracked_results_guard._violation_for(123) is None
    assert tracked_results_guard._violation_for("") is None
    assert tracked_results_guard._violation_for("results/not_tracked_6143.json") is None
    assert tracked_results_guard._violation_for(str(target).encode()) == str(target.resolve())

    tracked_results_guard._audit_hook("open", (str(target), "r", 0))
    tracked_results_guard._audit_hook("open", ("not-results.txt", "w", 0))
    tracked_results_guard._audit_hook("os.replace", ("tmp",))
    tracked_results_guard._audit_hook("os.replace", ("tmp", "not-results.txt"))
    tracked_results_guard._audit_hook("os.remove", tuple())
    tracked_results_guard._audit_hook("os.remove", ("not-results.txt",))

    with pytest.raises(tracked_results_guard.TrackedResultWriteError):
        tracked_results_guard._audit_hook("open", (str(target), "w", 0))
    tracked_results_guard.clear_violations()
    with pytest.raises(tracked_results_guard.TrackedResultWriteError):
        tracked_results_guard._audit_hook("os.replace", ("tmp", str(target)))
    tracked_results_guard.clear_violations()
    with pytest.raises(tracked_results_guard.TrackedResultWriteError):
        tracked_results_guard._audit_hook("os.remove", (str(target),))
    tracked_results_guard.clear_violations()

    class _BadTrackedPath:
        def __init__(self, raw: object) -> None:
            self.raw = raw

        def resolve(self, *, strict: bool = False) -> Path:
            raise OSError(f"cannot resolve {self.raw}")

    monkeypatch.setattr(tracked_results_guard, "Path", _BadTrackedPath)
    assert (
        tracked_results_guard._violation_for("results/experiment_1938_nrgpt_loss_probe.json")
        is None
    )
    monkeypatch.undo()

    calls: list[object] = []
    monkeypatch.setattr(tracked_results_guard, "_installed", False)
    monkeypatch.setattr(tracked_results_guard.sys, "addaudithook", lambda hook: calls.append(hook))
    assert tracked_results_guard.install() is True
    assert calls
    assert tracked_results_guard.is_installed() is True
    assert tracked_results_guard.install() is False
    tracked_results_guard.clear_violations()


def test_redirected_writer_does_not_record_tracked_result_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-ATTEMPTED-TRACKED-WRITE-DETECTION: redirected writes are allowed."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    tracked_results_guard.install()
    tracked_results_guard.clear_violations()

    try:
        atomic_write_json("results/experiment_6143_allowed.json", {"status": "redirected"})
        assert tracked_results_guard.recorded_violations() == []
    finally:
        tracked_results_guard.clear_violations()


def test_quarantine_fields_and_hash_survive_redirected_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6143-QUARANTINE-PRESERVATION: sentinel artifact is immutable."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    sentinel = repo_root() / "results" / "experiment_1938_nrgpt_loss_probe.json"
    fields = (
        "flagged_adversarial",
        "corrigendum_pending",
        "corrigendum_note",
        "flagged_adversarial_restoration_note",
        "flagged_adversarial_restored_fields",
    )
    before_hash = _sha256(sentinel)
    before_fields = _json_fields(sentinel, fields)

    atomic_write_json(
        "results/experiment_1938_nrgpt_loss_probe.json",
        {"status": "redirected copy", "flagged_adversarial": False},
    )

    assert _sha256(sentinel) == before_hash
    assert _json_fields(sentinel, fields) == before_fields
    assert (override / "experiment_1938_nrgpt_loss_probe.json").exists()
