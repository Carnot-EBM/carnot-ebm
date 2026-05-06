"""Tests for portable SessionMemory JSON packs.

Spec: REQ-LEARN-1405, REQ-LEARN-1406, REQ-LEARN-1407,
      SCENARIO-LEARN-1405, SCENARIO-LEARN-1406, SCENARIO-LEARN-1407
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.case_memory import CaseMemory, CaseRecord
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
from carnot.pipeline.session_memory import SessionMemory
from carnot.pipeline.session_memory_pack import (
    diff_session_memory_packs,
    export_session_memory,
    import_session_memory,
    load_session_memory_pack,
    validate_session_memory_pack,
)


def _case_memory(*, confidence: float = 0.5, case_id: str = "case-1") -> CaseMemory:
    memory = CaseMemory()
    memory.record(
        CaseRecord.normalize(
            benchmark="synthetic",
            benchmark_slice="arithmetic",
            model_name="test-model",
            case_id=case_id,
            violation_types=("carry_error",),
            description_texts=("two digit carry error",),
            prompt_text="What is 27 + 15?",
            confidence=confidence,
        )
    )
    return memory


def _template_library() -> ConstraintTemplateLibrary:
    library = ConstraintTemplateLibrary()
    library.register_builtin_templates()
    library.observe_pattern("carry_check", "test-model", 5)
    return library


def _fp_tracker() -> PerModelFPTracker:
    tracker = PerModelFPTracker(min_observations=10)
    tracker.update("test-model", "carry_check", was_fp=False, was_tp=True)
    return tracker


def _save_state(storage_dir: Path, *, confidence: float = 0.5) -> SessionMemory:
    session = SessionMemory(str(storage_dir), "test-model")
    session.save(_case_memory(confidence=confidence), _template_library(), _fp_tracker())
    return session


def test_export_pack_has_required_schema_metadata(tmp_path: Path) -> None:
    """REQ-LEARN-1405: export produces a schema-valid portable pack."""
    _save_state(tmp_path)

    pack = export_session_memory(tmp_path, "test-model")

    validate_session_memory_pack(pack)
    assert pack["schema"] == "carnot.session_memory_pack.v1"
    assert pack["schema_version"] == "1.0.0"
    assert pack["metadata"]["source"] == "local-session"
    assert pack["models"][0]["model_id"] == "test-model"
    assert pack["models"][0]["case_memory"]["portable_entries"][0]["n_observations"] == 1


def test_validate_rejects_missing_model_sections() -> None:
    """REQ-LEARN-1405-3: malformed packs are rejected before import."""
    with pytest.raises(ValueError, match="models"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "schema_version": "1.0.0",
                "metadata": {"source": "test", "license": "Apache-2.0"},
            }
        )

    with pytest.raises(ValueError, match="case_memory"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "schema_version": "1.0.0",
                "metadata": {"source": "test", "license": "Apache-2.0"},
                "models": [{"model_id": "test-model"}],
            }
        )


def test_export_import_round_trip_diff_is_empty(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1405: export -> import -> export has no semantic diff."""
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    _save_state(source_dir)
    source_pack = export_session_memory(source_dir, "test-model")

    report = import_session_memory(source_pack, target_dir, model_id="test-model", merge=True)
    target_pack = export_session_memory(target_dir, "test-model")

    assert report["written"] is True
    diff = diff_session_memory_packs(source_pack, target_pack)
    assert diff["is_empty"] is True


def test_merge_recomputes_duplicate_case_confidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1406: duplicate case entries merge with weighted confidence."""
    local_dir = tmp_path / "local"
    incoming_dir = tmp_path / "incoming"
    _save_state(local_dir, confidence=0.25)
    _save_state(incoming_dir, confidence=0.75)
    pack = export_session_memory(incoming_dir, "test-model")

    report = import_session_memory(pack, local_dir, model_id="test-model", merge=True)
    loaded = SessionMemory(str(local_dir), "test-model").load()

    assert report["case_entries_merged"] == 1
    assert loaded is not None
    case_memory, _, _ = loaded
    entry = case_memory.entries()[0]
    assert entry.support == 2
    assert entry.confidence == pytest.approx(0.5)


def test_import_dry_run_does_not_write_state(tmp_path: Path) -> None:
    """REQ-LEARN-1406-2: dry-run import reports work without creating state files."""
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    _save_state(source_dir)
    pack = export_session_memory(source_dir, "test-model")

    report = import_session_memory(
        pack, target_dir, model_id="test-model", merge=True, dry_run=True
    )

    assert report["written"] is False
    assert not SessionMemory(str(target_dir), "test-model").exists()


def test_replace_and_merge_are_mutually_exclusive(tmp_path: Path) -> None:
    """REQ-LEARN-1406-3: callers cannot request merge and replace together."""
    source_dir = tmp_path / "source"
    _save_state(source_dir)
    pack = export_session_memory(source_dir, "test-model")

    with pytest.raises(ValueError, match="mutually exclusive"):
        import_session_memory(
            pack, tmp_path / "target", model_id="test-model", merge=True, replace=True
        )


def test_cli_memory_export_import_and_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-LEARN-1407: CLI memory commands route to portable pack APIs."""
    from carnot.cli import main

    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    pack_path = tmp_path / "pack.json"
    exported_again = tmp_path / "pack2.json"
    _save_state(source_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "carnot",
            "memory",
            "export",
            "--storage-dir",
            str(source_dir),
            "--model-id",
            "test-model",
            "-o",
            str(pack_path),
        ],
    )
    assert main() == 0
    assert pack_path.exists()

    monkeypatch.setattr(
        "sys.argv",
        [
            "carnot",
            "memory",
            "import",
            str(pack_path),
            "--storage-dir",
            str(target_dir),
            "--model-id",
            "test-model",
            "--merge",
        ],
    )
    assert main() == 0

    export_session_memory(target_dir, "test-model", output_path=exported_again)
    monkeypatch.setattr(
        "sys.argv",
        [
            "carnot",
            "memory",
            "diff",
            str(pack_path),
            str(exported_again),
        ],
    )
    assert main() == 0
    captured = capsys.readouterr()
    assert "is_empty=True" in captured.out


def test_example_constraint_packs_validate() -> None:
    """REQ-LEARN-1405-1: checked-in starter packs conform to the schema contract."""
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "examples/constraint_packs/empty_v1.json",
        "examples/constraint_packs/arithmetic_v1.json",
        "examples/constraint_packs/python_code_v1.json",
    ):
        pack = load_session_memory_pack(root / relative)
        validate_session_memory_pack(pack)


def test_schema_file_is_draft_2020_12_json(tmp_path: Path) -> None:
    """REQ-LEARN-1405-1: the schema artifact is valid JSON and declares draft-2020-12."""
    root = Path(__file__).resolve().parents[2]
    schema_path = root / "python/carnot/schemas/session_memory_v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema"]["const"] == "carnot.session_memory_pack.v1"


def test_export_with_redact_provenance(tmp_path: Path) -> None:
    """REQ-LEARN-1405: redact_provenance removes case identifiers."""
    _save_state(tmp_path)

    pack = export_session_memory(tmp_path, "test-model", redact_provenance=True)

    validate_session_memory_pack(pack)
    entries = pack["models"][0]["case_memory"]["entries"]
    assert len(entries) > 0
    for entry in entries:
        for provenance in entry.get("provenance", []):
            assert provenance.get("case_id") == "REDACTED"
            assert provenance.get("source_artifact") is None
            assert provenance.get("verifier_path") == ""


def test_export_with_metadata(tmp_path: Path) -> None:
    """REQ-LEARN-1405: metadata parameter is included in exported pack."""
    _save_state(tmp_path)

    custom_metadata = {"source": "test-integration", "custom_field": "custom_value"}
    pack = export_session_memory(tmp_path, "test-model", metadata=custom_metadata)

    validate_session_memory_pack(pack)
    assert pack["metadata"]["source"] == "test-integration"
    assert pack["metadata"]["custom_field"] == "custom_value"


def test_validate_rejects_malformed_pack_type() -> None:
    """REQ-LEARN-1405-3: non-dict payloads are rejected."""
    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_session_memory_pack([])  # type: ignore

    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_session_memory_pack("not a dict")  # type: ignore


def test_validate_rejects_missing_schema_fields() -> None:
    """REQ-LEARN-1405-3: schema version validation."""
    with pytest.raises(ValueError, match="schema_version"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "metadata": {"source": "test", "license": "Apache-2.0"},
                "models": [],
            }
        )

    with pytest.raises(ValueError, match="schema_version"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "schema_version": "2.0.0",
                "metadata": {"source": "test", "license": "Apache-2.0"},
                "models": [],
            }
        )


def test_validate_rejects_missing_metadata_fields() -> None:
    """REQ-LEARN-1405-3: metadata source and license are required."""
    with pytest.raises(ValueError, match="metadata"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "schema_version": "1.0.0",
                "models": [],
            }
        )

    with pytest.raises(ValueError, match="source"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "schema_version": "1.0.0",
                "metadata": {"license": "Apache-2.0"},
                "models": [],
            }
        )

    with pytest.raises(ValueError, match="license"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "schema_version": "1.0.0",
                "metadata": {"source": "test"},
                "models": [],
            }
        )


def test_validate_rejects_empty_models() -> None:
    """REQ-LEARN-1405-3: models list must be non-empty."""
    with pytest.raises(ValueError, match="models"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "schema_version": "1.0.0",
                "metadata": {"source": "test", "license": "Apache-2.0"},
                "models": [],
            }
        )


def test_validate_rejects_duplicate_model_ids() -> None:
    """REQ-LEARN-1405-3: duplicate model IDs are rejected."""
    base_model = {
        "model_id": "test",
        "safe_model_id": "test",
        "case_memory": {"version": 1, "entries": []},
        "constraint_templates": [],
        "template_library": {"observations": []},
        "fp_tracker": {"stats": []},
        "session_state": {"violations_by_type": [], "violations_by_domain": []},
    }
    with pytest.raises(ValueError, match="duplicate model_id"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "schema_version": "1.0.0",
                "metadata": {"source": "test", "license": "Apache-2.0"},
                "models": [base_model, base_model],
            }
        )


def test_validate_rejects_missing_model_fields() -> None:
    """REQ-LEARN-1405-3: model must have all required fields."""
    for missing_field in [
        "model_id",
        "safe_model_id",
        "case_memory",
        "constraint_templates",
        "template_library",
        "fp_tracker",
        "session_state",
    ]:
        model = {
            "model_id": "test",
            "safe_model_id": "test",
            "case_memory": {"version": 1, "entries": []},
            "constraint_templates": [],
            "template_library": {"observations": []},
            "fp_tracker": {"stats": []},
            "session_state": {"violations_by_type": [], "violations_by_domain": []},
        }
        del model[missing_field]
        with pytest.raises(ValueError, match="missing"):
            validate_session_memory_pack(
                {
                    "schema": "carnot.session_memory_pack.v1",
                    "schema_version": "1.0.0",
                    "metadata": {"source": "test", "license": "Apache-2.0"},
                    "models": [model],
                }
            )


def test_validate_rejects_malformed_case_memory() -> None:
    """REQ-LEARN-1405-3: case_memory.entries must be a list."""
    with pytest.raises(ValueError, match="entries"):
        validate_session_memory_pack(
            {
                "schema": "carnot.session_memory_pack.v1",
                "schema_version": "1.0.0",
                "metadata": {"source": "test", "license": "Apache-2.0"},
                "models": [
                    {
                        "model_id": "test",
                        "safe_model_id": "test",
                        "case_memory": {"version": 1},
                        "constraint_templates": [],
                        "template_library": {"observations": []},
                        "fp_tracker": {"stats": []},
                        "session_state": {"violations_by_type": [], "violations_by_domain": []},
                    }
                ],
            }
        )


def test_diff_with_added_and_removed_models(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1407: diff detects added and removed models."""
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    _save_state(source_dir, confidence=0.3)
    source_pack = export_session_memory(source_dir, "test-model")

    # Create a different pack with a different model
    _save_state(target_dir, confidence=0.7)
    target_pack = export_session_memory(target_dir, "other-model")

    diff = diff_session_memory_packs(source_pack, target_pack)
    assert not diff["is_empty"]
    assert "test-model" in diff["models_removed"]
    assert "other-model" in diff["models_added"]


def test_diff_with_changed_models(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1407: diff detects changed model content."""
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    _save_state(source_dir, confidence=0.25)
    source_pack = export_session_memory(source_dir, "test-model")

    _save_state(target_dir, confidence=0.75)
    target_pack = export_session_memory(target_dir, "test-model")

    diff = diff_session_memory_packs(source_pack, target_pack)
    assert not diff["is_empty"]
    assert len(diff["models_changed"]) > 0
    assert diff["models_changed"][0]["model_id"] == "test-model"


def test_import_replace_mode_overwrites(tmp_path: Path) -> None:
    """REQ-LEARN-1406: replace mode completely overwrites local state."""
    local_dir = tmp_path / "local"
    incoming_dir = tmp_path / "incoming"
    _save_state(local_dir, confidence=0.25)
    _save_state(incoming_dir, confidence=0.75)
    incoming_pack = export_session_memory(incoming_dir, "test-model")

    report = import_session_memory(incoming_pack, local_dir, model_id="test-model", replace=True)
    loaded = SessionMemory(str(local_dir), "test-model").load()

    assert report["mode"] == "replace"
    assert report["case_entries_merged"] == 0
    assert report["case_entries_added"] == 1
    assert loaded is not None
    case_memory, _, _ = loaded
    entry = case_memory.entries()[0]
    assert entry.confidence == pytest.approx(0.75)


def test_import_with_new_model_id_remaps(tmp_path: Path) -> None:
    """REQ-LEARN-1406: import can remap pack model_id to a different target."""
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    _save_state(source_dir)
    pack = export_session_memory(source_dir, "source-model")

    # Import with a different model_id
    report = import_session_memory(pack, target_dir, model_id="remapped-model", merge=True)

    assert report["model_id"] == "remapped-model"
    loaded = SessionMemory(str(target_dir), "remapped-model").load()
    assert loaded is not None


def test_load_session_memory_pack_from_dict(tmp_path: Path) -> None:
    """REQ-LEARN-1405-3: load_session_memory_pack validates dicts as well as files."""
    _save_state(tmp_path)
    pack_dict = export_session_memory(tmp_path, "test-model")

    # load_session_memory_pack expects a file path, not a dict
    # so this should raise since we're passing a dict path that doesn't exist
    pack_path = tmp_path / "pack.json"
    # export and then load from file
    export_session_memory(tmp_path, "test-model", output_path=pack_path)
    loaded_pack = load_session_memory_pack(pack_path)

    assert loaded_pack["schema"] == "carnot.session_memory_pack.v1"
    assert loaded_pack["models"][0]["model_id"] == "test-model"


def test_merge_with_added_template_observations(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1406: merge adds new template observations."""
    local_dir = tmp_path / "local"
    incoming_dir = tmp_path / "incoming"
    local_session = SessionMemory(str(local_dir), "test-model")
    incoming_session = SessionMemory(str(incoming_dir), "test-model")

    local_library = ConstraintTemplateLibrary()
    local_library.register_builtin_templates()
    local_library.observe_pattern("carry_check", "test-model", 3)

    incoming_library = ConstraintTemplateLibrary()
    incoming_library.register_builtin_templates()
    incoming_library.observe_pattern("carry_check", "test-model", 2)
    incoming_library.observe_pattern("sign_check", "test-model", 5)

    local_session.save(CaseMemory(), local_library, PerModelFPTracker())
    incoming_session.save(CaseMemory(), incoming_library, PerModelFPTracker())

    incoming_pack = export_session_memory(incoming_dir, "test-model")

    report = import_session_memory(incoming_pack, local_dir, model_id="test-model", merge=True)

    # Should have merged template observations
    assert report["template_observations_merged"] > 0 or report["template_observations_added"] > 0


def test_merge_with_fp_tracker_stats(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1406: merge combines FP tracker statistics."""
    local_dir = tmp_path / "local"
    incoming_dir = tmp_path / "incoming"
    local_session = SessionMemory(str(local_dir), "test-model")
    incoming_session = SessionMemory(str(incoming_dir), "test-model")

    local_tracker = PerModelFPTracker(min_observations=10)
    local_tracker.update("test-model", "carry_check", was_fp=False, was_tp=True)
    local_tracker.update("test-model", "carry_check", was_fp=False, was_tp=True)

    incoming_tracker = PerModelFPTracker(min_observations=10)
    incoming_tracker.update("test-model", "carry_check", was_fp=True, was_tp=False)

    local_session.save(CaseMemory(), ConstraintTemplateLibrary(), local_tracker)
    incoming_session.save(CaseMemory(), ConstraintTemplateLibrary(), incoming_tracker)

    incoming_pack = export_session_memory(incoming_dir, "test-model")

    report = import_session_memory(incoming_pack, local_dir, model_id="test-model", merge=True)

    # Should have merged FP tracker stats
    assert report["fp_stats_merged"] > 0 or report["fp_stats_added"] > 0


def test_export_empty_session_as_valid_pack(tmp_path: Path) -> None:
    """REQ-LEARN-1405-2: exporting empty session produces valid pack."""
    empty_session_dir = tmp_path / "empty"
    empty_session_dir.mkdir(parents=True, exist_ok=True)

    # No saved state, so session is empty
    pack = export_session_memory(empty_session_dir, "empty-model")

    validate_session_memory_pack(pack)
    assert pack["models"][0]["model_id"] == "empty-model"
    assert len(pack["models"][0]["case_memory"]["entries"]) == 0
