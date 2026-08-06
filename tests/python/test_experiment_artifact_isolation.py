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

import ast
import builtins
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from carnot import experiment_artifacts as artifact_mod
from carnot import experiment_6157_repo_wide_artifact_isolation_closure as exp6157
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
from carnot.pipeline.atomic_writer import AtomicResultWriter
from carnot.pipeline.deliverable_guard import DeliverableGuard, DocOnlyClassifier
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


def test_deliverable_guard_allows_existing_production_artifact_for_read_only_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-EARLY-OVERRIDE-COLLECTION: read-only imports can collect."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    sentinel = repo_root() / "results" / "experiment_1938_nrgpt_loss_probe.json"
    before = _sha256(sentinel)

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/python/test_import.py::case (call)")
    with pytest.raises(FileNotFoundError):
        DeliverableGuard("results/experiment_1938_nrgpt_loss_probe.json").assert_written()

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    DeliverableGuard("results/experiment_1938_nrgpt_loss_probe.json").assert_written()

    assert _sha256(sentinel) == before
    assert not (override / "experiment_1938_nrgpt_loss_probe.json").exists()


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


def test_legacy_relative_results_open_and_pathlib_redirect_to_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-LEGACY-WRITER-COMPATIBILITY: literal writers redirect."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    monkeypatch.chdir(repo_root())
    tracked_results_guard.install()
    tracked_results_guard.install_legacy_results_write_compat()
    tracked_results_guard.clear_violations()
    tracked_results_guard.clear_legacy_compat_redirects()
    open_target = repo_root() / "results" / "experiment_1938_nrgpt_loss_probe.json"
    pathlib_target = repo_root() / "results" / "experiment_2085_pem_sudoku_eval.json"
    before = {_target: _sha256(_target) for _target in (open_target, pathlib_target)}

    try:
        with open(  # noqa: PTH123 - this is the legacy-writer compatibility control.
            "results/experiment_1938_nrgpt_loss_probe.json", "w", encoding="utf-8"
        ) as fh:
            fh.write("redirected by Exp6157\n")
        Path("results/experiment_2085_pem_sudoku_eval.json").write_text(
            "redirected by pathlib\n", encoding="utf-8"
        )

        assert (override / "experiment_1938_nrgpt_loss_probe.json").read_text(
            encoding="utf-8"
        ) == "redirected by Exp6157\n"
        assert (override / "experiment_2085_pem_sudoku_eval.json").read_text(
            encoding="utf-8"
        ) == "redirected by pathlib\n"
        assert {_target: _sha256(_target) for _target in before} == before
        assert tracked_results_guard.recorded_violations() == []
        redirects = tracked_results_guard.recorded_legacy_compat_redirects()
        assert {row["requested"] for row in redirects} >= {
            "results/experiment_1938_nrgpt_loss_probe.json",
            "results/experiment_2085_pem_sudoku_eval.json",
        }
    finally:
        tracked_results_guard.clear_violations()
        tracked_results_guard.clear_legacy_compat_redirects()


def test_legacy_relative_atomic_replace_redirects_source_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-LEGACY-WRITER-COMPATIBILITY: legacy os.replace is safe."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    monkeypatch.chdir(repo_root())
    tracked_results_guard.install()
    tracked_results_guard.install_legacy_results_write_compat()
    tracked_results_guard.clear_violations()
    target = repo_root() / "results" / "experiment_1938_nrgpt_loss_probe.json"
    before = _sha256(target)

    try:
        Path("results/experiment_1938_nrgpt_loss_probe.json.tmp").write_text(
            "replacement\n", encoding="utf-8"
        )
        os.replace(
            "results/experiment_1938_nrgpt_loss_probe.json.tmp",
            "results/experiment_1938_nrgpt_loss_probe.json",
        )

        assert (override / "experiment_1938_nrgpt_loss_probe.json").read_text(
            encoding="utf-8"
        ) == "replacement\n"
        assert not (override / "experiment_1938_nrgpt_loss_probe.json.tmp").exists()
        assert _sha256(target) == before
        assert tracked_results_guard.recorded_violations() == []
    finally:
        tracked_results_guard.clear_violations()
        tracked_results_guard.clear_legacy_compat_redirects()


def test_legacy_compat_rejects_traversal_bare_results_and_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-LEGACY-WRITER-COMPATIBILITY: bad targets fail closed."""
    override = _valid_override_root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (override / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    monkeypatch.chdir(repo_root())
    tracked_results_guard.install_legacy_results_write_compat()

    with pytest.raises(ArtifactPathError):
        open("results/../experiment_6157_escape.json", "w", encoding="utf-8")  # noqa: PTH123
    with pytest.raises(ArtifactPathError):
        open("results", "w", encoding="utf-8")  # noqa: PTH123
    with pytest.raises(ArtifactPathError):
        open("results/escape/leak.json", "w", encoding="utf-8")  # noqa: PTH123
    assert not (repo_root() / "experiment_6157_escape.json").exists()
    assert not (outside / "leak.json").exists()


def test_absolute_tracked_write_negative_control_still_hits_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-ATTEMPTED-WRITE-CONTROL: real tracked writes are caught."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    tracked_results_guard.install()
    tracked_results_guard.install_legacy_results_write_compat()
    tracked_results_guard.clear_violations()
    target = repo_root() / "results" / "experiment_1938_nrgpt_loss_probe.json"
    before = _sha256(target)

    try:
        with pytest.raises(tracked_results_guard.TrackedResultWriteError):
            target.write_text("absolute write must be caught\n", encoding="utf-8")
        assert tracked_results_guard.recorded_violations()[-1]["path"] == (
            "results/experiment_1938_nrgpt_loss_probe.json"
        )
        assert _sha256(target) == before
    finally:
        tracked_results_guard.clear_violations()


def test_atomic_result_writer_resolves_legacy_results_path_through_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-QUARANTINE-AND-ATOMIC-PRESERVATION: shared writer redirects."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    tracked_results_guard.install()
    tracked_results_guard.clear_violations()
    target = repo_root() / "results" / "experiment_1938_nrgpt_loss_probe.json"
    before = _sha256(target)

    writer = AtomicResultWriter("results/experiment_1938_nrgpt_loss_probe.json")
    writer.write({"status": "redirected"})

    assert writer._final == override / "experiment_1938_nrgpt_loss_probe.json"
    assert writer.verify_exists() is True
    assert json.loads(writer._final.read_text(encoding="utf-8")) == {"status": "redirected"}
    assert _sha256(target) == before
    assert tracked_results_guard.recorded_violations() == []


def test_exp6157_census_manifests_and_failure_classification(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6157-CENSUS-MANIFESTS: census rows become reviewed ledger entries."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "legacy_open.py").write_text(
        "import json\nwith open('results/a.json', 'w') as fh:\n    json.dump({'a': 1}, fh)\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "atomic_result.py").write_text(
        "from carnot.pipeline.atomic_writer import AtomicResultWriter\n"
        "AtomicResultWriter('results/b.json').write({'b': 1})\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "canonical.py").write_text(
        "from carnot.experiment_artifacts import atomic_write_json\n"
        "atomic_write_json('results/c.json', {'c': 1})\n",
        encoding="utf-8",
    )

    census = exp6157.collect_writer_census(tmp_path, roots=("scripts",))
    assert census["total_rows"] == 3
    assert census["grouping"]["mechanism_counts"]["legacy_open_or_json_dump"] == 1
    assert census["grouping"]["mechanism_counts"]["atomic_result_writer"] == 1
    assert census["grouping"]["mechanism_counts"]["canonical_artifact_writer"] == 1

    exception_manifest = exp6157.build_exception_manifest(
        census, reviewed_at="2026-08-06", expiry="2026-09-06"
    )
    migration_ledger = exp6157.build_migration_ledger(census)
    assert exception_manifest["reviewed"] is True
    assert exception_manifest["entry_count"] == 2
    assert all(
        row["owner"] and row["reason"] and row["expiry"] for row in exception_manifest["entries"]
    )
    assert migration_ledger["covered_row_count"] == 3
    assert migration_ledger["entries"][0]["migration_key"].count(":") == 1

    classified = exp6157.classify_test_failures(
        [
            {
                "name": "negative-control",
                "command": "pytest negative",
                "exit_code": 1,
                "stderr": "Test attempted to write tracked result evidence",
            },
            {
                "name": "known-import",
                "command": "pytest unrelated",
                "exit_code": 2,
                "stderr": "ModuleNotFoundError: No module named 'missing_fixture'",
            },
            {
                "name": "new-assertion",
                "command": "pytest new",
                "exit_code": 1,
                "stderr": "AssertionError: changed behavior",
            },
        ],
        known_unrelated_patterns=("ModuleNotFoundError",),
    )
    assert classified["counts"] == {
        "artifact_isolation": 1,
        "unrelated_preexisting": 1,
        "new_regression": 1,
        "unclassified": 0,
    }

    unclassified = exp6157.classify_test_failures([{"name": "", "exit_code": 1}])
    assert unclassified["counts"]["unclassified"] == 1


def test_exp6157_low_level_census_snapshot_and_fallback_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-CENSUS-MANIFESTS: fallback accounting paths are deterministic."""
    assert exp6157.path_sha256(tmp_path / "missing.json") is None
    assert isinstance(exp6157._git(repo_root(), ["status", "--short"]), str)
    assert isinstance(exp6157._git_status_short(repo_root()), list)

    monkeypatch.setattr(exp6157, "_git", lambda *_a, **_k: "scripts/a.py\nREADME.md\n")
    assert exp6157._source_files(tmp_path, ("scripts",)) == [Path("scripts/a.py")]

    def _raise_git(*_args: object, **_kwargs: object) -> str:
        raise OSError("git unavailable")

    monkeypatch.setattr(exp6157, "_git", _raise_git)
    assert exp6157._git_status_short(tmp_path) == []
    assert exp6157._tracked_results(tmp_path) == []

    (tmp_path / "results").mkdir()
    good = tmp_path / "results" / "good.json"
    bad = tmp_path / "results" / "bad.json"
    good.write_text('{"flagged_adversarial": true, "corrigendum_note": "kept"}\n')
    bad.write_text("{", encoding="utf-8")

    assert exp6157._tracked_results(tmp_path) == [
        Path("results/bad.json"),
        Path("results/good.json"),
    ]
    assert exp6157._aggregate_digest(
        tmp_path, [Path("results/good.json"), Path("results/missing.json")]
    ).startswith("sha256:")
    assert exp6157._snapshot_quarantine_fields(tmp_path, [Path("results/good.json")]) == {
        "results/good.json": {"flagged_adversarial": True, "corrigendum_note": "kept"}
    }
    assert "_unreadable" in exp6157._snapshot_quarantine_fields(
        tmp_path, [Path("results/bad.json")]
    )["results/bad.json"]

    assert exp6157._call_name(ast.Constant(1)) == ""
    assert exp6157._interesting_call_name("pathlib.Path.write_text") == "write_text"
    assert exp6157._mechanism_for_calls([{"call": "os.replace"}]) == "legacy_atomic_replace"
    assert exp6157._mechanism_for_calls([{"call": "unknown"}]) == "other_writer"
    assert exp6157._risk_for_mechanism("other_writer") == "legacy_literal_write_requires_compat"

    source = tmp_path / "source.py"
    source.write_text("x = 'results/no_writer.json'\n", encoding="utf-8")
    assert exp6157._writer_row(tmp_path, Path("source.py")) is None
    source.write_text("def nope(:\n", encoding="utf-8")
    assert exp6157._writer_row(tmp_path, Path("source.py")) is None

    assert exp6157.build_migration_ledger({"rows": [object()]})["covered_row_count"] == 0

    real_import = builtins.__import__

    def _raise_operator_guard(name: str, *args: object, **kwargs: object) -> object:
        if name == "carnot.testing.operator_curated_doc_guard":
            raise ImportError("operator guard unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_operator_guard)
    assert exp6157._protected_file_paths() == (
        Path("README.md"),
        Path("LICENSE"),
        Path("NOTICE"),
    )
    snapshot = exp6157.snapshot_repository(tmp_path)
    assert snapshot["tracked_results_count"] == 2
    assert snapshot["protected_matrix"]["README.md"] is None


def test_exp6157_sidecars_run_and_validation_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-FAILURE-CLASSIFICATION: artifact generation is replayable."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "legacy.py").write_text(
        "from pathlib import Path\nPath('results/legacy.json').write_text('x')\n",
        encoding="utf-8",
    )
    pre_path = tmp_path / "preconditions.json"
    pre_path.write_text(
        json.dumps(
            {
                "tracked_results_digest": "sha256:before",
                "quarantine_fields": {},
                "prior_6143_direct_writer_census_count": 6198,
            }
        ),
        encoding="utf-8",
    )

    loaded = exp6157._load_pre_snapshot(tmp_path, pre_path)
    assert "protected_matrix" in loaded

    invalid_pre = tmp_path / "invalid-preconditions.json"
    invalid_pre.write_text("{", encoding="utf-8")
    assert "tracked_results_digest" in exp6157._load_pre_snapshot(tmp_path, invalid_pre)

    sidecar = exp6157._write_sidecar(tmp_path, Path("results/sidecar.json"), {"ok": True})
    assert sidecar["path"] == "results/sidecar.json"
    assert sidecar["sha256"].startswith("sha256:")

    artifact = exp6157.run(
        tmp_path,
        command_receipts=[
            {
                "name": "focused",
                "command": "pytest focused",
                "exit_code": 0,
                "stderr": "",
            },
            {
                "name": "determination-preservation",
                "command": "python scripts/determination_preservation_lint.py",
                "exit_code": 0,
                "stderr": "",
            },
        ],
        pre_snapshot_path=pre_path,
        duration_s=1.25,
    )

    assert artifact["duration_s"] == 1.25
    assert artifact["determination_preservation_lint_receipt"]["name"] == (
        "determination-preservation"
    )
    assert (tmp_path / exp6157.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / exp6157.EXCEPTION_MANIFEST_RELATIVE_PATH).exists()
    assert (tmp_path / exp6157.MIGRATION_LEDGER_RELATIVE_PATH).exists()

    bad = dict(artifact)
    bad.pop("status")
    bad["field_provenance"] = []
    bad["inference_substrate"] = "wrong"
    bad["honest_verdict"] = "maybe"
    bad["reproducibility_checksum"] = "wrong"
    errors = exp6157.validate_artifact(bad)
    assert "missing:status" in errors
    assert "field_provenance:not_mapping" in errors
    assert "inference_substrate" in errors
    assert "honest_verdict_prefix" in errors
    assert "reproducibility_checksum" in errors

    bad_provenance = dict(artifact)
    bad_provenance["field_provenance"] = dict(artifact["field_provenance"])
    bad_provenance["field_provenance"]["status"] = {"principle": "wrong"}
    bad_provenance["reproducibility_checksum"] = exp6157.payload_checksum(bad_provenance)
    assert "field_provenance:status" in exp6157.validate_artifact(bad_provenance)

    monkeypatch.setattr(exp6157, "validate_artifact", lambda _payload: ["forced"])
    with pytest.raises(ValueError):
        exp6157.run(tmp_path, pre_snapshot_path=pre_path, duration_s=1.0)


def test_exp6157_closure_artifact_schema_and_ready_score() -> None:
    """SCENARIO-REPORT-6157-FAILURE-CLASSIFICATION: ready score follows isolation evidence."""
    census = {
        "total_rows": 1,
        "rows": [
            {
                "path": "scripts/legacy.py",
                "source_sha256": "a" * 64,
                "mechanism": "legacy_open_or_json_dump",
                "risk": "legacy_literal_write_requires_compat",
            }
        ],
        "grouping": {
            "mechanism_counts": {"legacy_open_or_json_dump": 1},
            "risk_counts": {"legacy_literal_write_requires_compat": 1},
        },
        "checksum": "sha256:census",
    }
    exception_manifest = {
        "reviewed": True,
        "entry_count": 1,
        "entries": [
            {
                "source_path": "scripts/legacy.py",
                "source_sha256": "a" * 64,
                "owner": "artifact-isolation",
                "reason": "covered by legacy compatibility",
                "expiry": "2026-09-06",
            }
        ],
        "sha256": "sha256:exceptions",
        "path": "results/experiment_6157_writer_exception_manifest.json",
    }
    migration_ledger = {
        "covered_row_count": 1,
        "entries": [{"migration_key": "scripts/legacy.py:" + "a" * 64}],
        "sha256": "sha256:migration",
        "path": "results/experiment_6157_resumable_migration_ledger.json",
    }

    artifact = exp6157.build_closure_artifact(
        pre_snapshot={"tracked_results_digest": "same", "quarantine_fields": {}},
        post_snapshot={"tracked_results_digest": "same", "quarantine_fields": {}},
        prior_failure_receipt={"residual_call_site_rows": 6198},
        writer_census_before={"total_rows": 6198, "grouping": {}},
        writer_census_after=census,
        exception_manifest=exception_manifest,
        migration_ledger=migration_ledger,
        command_receipts=[
            {"name": "focused", "command": "pytest focused", "exit_code": 0, "stderr": ""}
        ],
        duration_s=1.0,
    )

    assert exp6157.validate_artifact(artifact) == []
    assert artifact["artifact_isolation_closure_ready_score"] == 1
    assert artifact["isolation_violation_count"] == 0
    assert artifact["inference_substrate"] == "deterministic_repository_test_isolation"


def test_legacy_results_helper_and_guard_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-LEGACY-WRITER-COMPATIBILITY: helper branches stay narrow."""
    override = _valid_override_root(tmp_path)
    env = {ARTIFACT_ROOT_ENV: str(override)}

    assert artifact_mod.is_legacy_results_path(b"results/bytes.json") is True
    assert artifact_mod.is_legacy_results_path("") is False
    assert artifact_mod.is_legacy_results_path(object()) is False
    assert artifact_mod.resolve_legacy_results_write_path(
        b"results/bytes.json", env=env
    ) == override / "bytes.json"

    production_target = repo_root() / "results" / "experiment_6157_absolute_helper.json"
    assert artifact_mod.resolve_legacy_results_write_path(
        production_target, env=env
    ) == override / "experiment_6157_absolute_helper.json"

    external_target = tmp_path / "external" / "result.json"
    assert artifact_mod.resolve_legacy_results_write_path(
        external_target, ensure_parent=True
    ) == external_target
    assert external_target.parent.exists()

    monkeypatch.chdir(tmp_path)
    assert artifact_mod.resolve_legacy_results_write_path(
        "nested/result.json", ensure_parent=True
    ) == Path("nested/result.json")
    assert (tmp_path / "nested").exists()

    assert tracked_results_guard.is_legacy_compat_installed() is True
    assert tracked_results_guard._path_text(b"results/bytes.json") == "results/bytes.json"
    assert tracked_results_guard._path_text(object()).startswith("<object object")
    assert tracked_results_guard._redirect_legacy_write_path(
        "results/no-env.json", "open", ensure_parent=True
    ) != "results/no-env.json"


def test_atomic_result_writer_failure_preserves_tmp_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-QUARANTINE-AND-ATOMIC-PRESERVATION: rename failures are loud."""
    writer = AtomicResultWriter(str(tmp_path / "result.json"))

    def _raise_rename(_src: object, _dst: object) -> None:
        raise RuntimeError("rename failed")

    monkeypatch.setattr(os, "rename", _raise_rename)
    with pytest.raises(RuntimeError):
        writer.write({"status": "partial"})
    assert writer._tmp.read_text(encoding="utf-8") == '{\n  "status": "partial"\n}'

    original_write_text = Path.write_text

    def _raise_for_historical_tmp(
        self: Path, data: str, *args: object, **kwargs: object
    ) -> int:
        if self == writer._tmp:
            raise OSError("diagnostic tmp unavailable")
        return original_write_text(self, data, *args, **kwargs)

    writer._tmp.unlink()
    monkeypatch.setattr(Path, "write_text", _raise_for_historical_tmp)
    with pytest.raises(RuntimeError):
        writer.write({"status": "partial"})
    assert not writer._tmp.exists()


def test_deliverable_guard_and_doc_only_classifier_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6157-EARLY-OVERRIDE-COLLECTION: read guards remain explicit."""
    override = _valid_override_root(tmp_path)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(override))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    external = tmp_path / "external" / "deliverable.json"

    assert DeliverableGuard._resolve_guard_path(str(external)) == external
    with pytest.raises(ArtifactPathError):
        DeliverableGuard._resolve_guard_path("results/../bad.json")
    assert DeliverableGuard._resolve_production_fallback("results/../bad.json", external) is None
    production_path = repo_root() / "results" / "experiment_1938_nrgpt_loss_probe.json"
    assert (
        DeliverableGuard._resolve_production_fallback(str(production_path), production_path)
        is None
    )

    partial = tmp_path / "partial.json"
    partial.write_text("{}", encoding="utf-8")
    guard = DeliverableGuard(str(external))
    guard.assert_written_or_partial(str(partial))
    with pytest.raises(FileNotFoundError):
        guard.assert_written_or_partial(str(tmp_path / "missing-partial.json"))

    classifier = DocOnlyClassifier()
    assert classifier.is_doc_only_diff([]) is False
    assert classifier.is_doc_only_diff(["ops/status.md", "docs/api-reference.md"]) is True
    assert classifier.is_doc_only_diff(["python/carnot/module.py"]) is False
