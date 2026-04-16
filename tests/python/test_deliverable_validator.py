"""Tests for DeliverableContentValidator, CloudGPUInstructions, and related helpers.

Spec: REQ-INFRA-019, REQ-INFRA-020,
      SCENARIO-INFRA-022, SCENARIO-INFRA-023, SCENARIO-INFRA-024
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from carnot.pipeline.deliverable_validator import (
    CloudGPUInstructions,
    DeliverableContentValidator,
    build_cloud_gpu_instructions,
    generate_cloud_gpu_script,
)


# ---------------------------------------------------------------------------
# DeliverableContentValidator.is_valid_python
# ---------------------------------------------------------------------------


class TestIsValidPython:
    """SCENARIO-INFRA-022/023: is_valid_python returns True only for valid Python."""

    def test_valid_python_returns_true(self, tmp_path: Path) -> None:
        # REQ-INFRA-019: valid Python module content must be accepted
        f = tmp_path / "mod.py"
        f.write_text("def hello():\n    return 42\n")
        assert DeliverableContentValidator.is_valid_python(str(f)) is True

    def test_json_content_returns_false(self, tmp_path: Path) -> None:
        # SCENARIO-INFRA-022: JSON content must be rejected
        f = tmp_path / "mod.py"
        f.write_text('{"experiment": 375, "status": "partial"}\n')
        assert DeliverableContentValidator.is_valid_python(str(f)) is False

    def test_empty_file_returns_false(self, tmp_path: Path) -> None:
        # Empty file has no Python content — treat as corrupt
        f = tmp_path / "empty.py"
        f.write_text("")
        assert DeliverableContentValidator.is_valid_python(str(f)) is False

    def test_missing_file_returns_false(self, tmp_path: Path) -> None:
        # Missing file cannot be valid
        missing = str(tmp_path / "nonexistent.py")
        assert DeliverableContentValidator.is_valid_python(missing) is False

    def test_syntax_error_returns_false(self, tmp_path: Path) -> None:
        # SyntaxError in file content must be rejected
        f = tmp_path / "broken.py"
        f.write_text("def foo(:\n    pass\n")
        assert DeliverableContentValidator.is_valid_python(str(f)) is False

    def test_multiline_valid_python_returns_true(self, tmp_path: Path) -> None:
        # Multi-line valid module must be accepted
        f = tmp_path / "big.py"
        f.write_text(textwrap.dedent("""\
            from dataclasses import dataclass

            @dataclass
            class Foo:
                x: int
                y: float

            def compute(a: int, b: int) -> int:
                return a + b
        """))
        assert DeliverableContentValidator.is_valid_python(str(f)) is True

    def test_unicode_decode_error_returns_false(self, tmp_path: Path) -> None:
        # Binary / non-UTF-8 file must be rejected
        f = tmp_path / "binary.py"
        f.write_bytes(b"\xff\xfe\x00invalid binary content\x00")
        assert DeliverableContentValidator.is_valid_python(str(f)) is False

    def test_only_comments_returns_true(self, tmp_path: Path) -> None:
        # A file with only comments is syntactically valid Python
        f = tmp_path / "comments.py"
        f.write_text("# This is a comment\n# Another comment\n")
        assert DeliverableContentValidator.is_valid_python(str(f)) is True

    def test_json_array_returns_false(self, tmp_path: Path) -> None:
        # JSON array is not valid Python
        f = tmp_path / "array.py"
        f.write_text('[{"key": "value"}]\n')
        assert DeliverableContentValidator.is_valid_python(str(f)) is False


# ---------------------------------------------------------------------------
# DeliverableContentValidator.validate_and_clear
# ---------------------------------------------------------------------------


class TestValidateAndClear:
    """SCENARIO-INFRA-022/023: validate_and_clear deletes corrupt files, keeps valid ones."""

    def test_valid_python_returns_true_no_deletion(self, tmp_path: Path) -> None:
        # SCENARIO-INFRA-023: valid file must not be deleted
        f = tmp_path / "good.py"
        f.write_text("x = 1\n")
        result = DeliverableContentValidator.validate_and_clear(str(f))
        assert result is True
        assert f.exists(), "valid file must NOT be deleted"

    def test_json_content_returns_false_and_deletes(self, tmp_path: Path) -> None:
        # SCENARIO-INFRA-022: corrupt JSON file must be deleted and False returned
        f = tmp_path / "corrupt.py"
        f.write_text('{"experiment": 375}\n')
        result = DeliverableContentValidator.validate_and_clear(str(f))
        assert result is False
        assert not f.exists(), "corrupt file must be deleted"

    def test_empty_file_deleted(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_text("")
        result = DeliverableContentValidator.validate_and_clear(str(f))
        assert result is False
        assert not f.exists()

    def test_missing_file_returns_false_no_error(self, tmp_path: Path) -> None:
        # Missing file: is_valid_python=False, os.remove would fail — must not raise
        missing = str(tmp_path / "ghost.py")
        result = DeliverableContentValidator.validate_and_clear(missing)
        assert result is False

    def test_warning_logged_on_corrupt_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Corrupt file must trigger a warning log containing the path
        import logging

        f = tmp_path / "json_imposter.py"
        f.write_text('{"status": "bad"}\n')
        with caplog.at_level(logging.WARNING, logger="carnot.pipeline.deliverable_validator"):
            DeliverableContentValidator.validate_and_clear(str(f))
        assert str(f) in caplog.text or "json_imposter" in caplog.text


# ---------------------------------------------------------------------------
# DeliverableContentValidator.audit_known_corrupt_files
# ---------------------------------------------------------------------------


class TestAuditKnownCorruptFiles:
    """REQ-INFRA-019: audit_known_corrupt_files returns correct status per file."""

    # The five known corrupt files from RETRO-023
    KNOWN_FILES = [
        "python/carnot/models/cikan_energy.py",
        "python/carnot/pipeline/jitrl_memory.py",
        "python/carnot/models/safety_kan.py",
        "python/carnot/pipeline/semantic_energy_scorer.py",
        "python/carnot/pipeline/crane_extractor.py",
    ]

    def test_returns_dict_with_all_known_files(self, tmp_path: Path) -> None:
        # All five keys must be present in the returned dict
        result = DeliverableContentValidator.audit_known_corrupt_files(str(tmp_path))
        assert set(result.keys()) == set(self.KNOWN_FILES)

    def test_missing_file_status(self, tmp_path: Path) -> None:
        # Files absent from tmp_path must be reported as 'missing'
        result = DeliverableContentValidator.audit_known_corrupt_files(str(tmp_path))
        for path, status in result.items():
            assert status == "missing", f"{path} should be 'missing' in empty tmp_path"

    def test_valid_python_status(self, tmp_path: Path) -> None:
        # Create one file as valid Python — should report 'valid_python'
        target_rel = "python/carnot/models/cikan_energy.py"
        target_abs = tmp_path / target_rel
        target_abs.parent.mkdir(parents=True, exist_ok=True)
        target_abs.write_text("def energy(): return 0.0\n")
        result = DeliverableContentValidator.audit_known_corrupt_files(str(tmp_path))
        assert result[target_rel] == "valid_python"

    def test_corrupt_json_status(self, tmp_path: Path) -> None:
        # Create one file as JSON — should report 'corrupt_json'
        target_rel = "python/carnot/pipeline/jitrl_memory.py"
        target_abs = tmp_path / target_rel
        target_abs.parent.mkdir(parents=True, exist_ok=True)
        target_abs.write_text('{"experiment": 999, "status": "partial"}\n')
        result = DeliverableContentValidator.audit_known_corrupt_files(str(tmp_path))
        assert result[target_rel] == "corrupt_json"

    def test_mixed_statuses(self, tmp_path: Path) -> None:
        # Set up: one valid, one corrupt, rest missing
        valid_rel = "python/carnot/models/safety_kan.py"
        corrupt_rel = "python/carnot/pipeline/semantic_energy_scorer.py"

        (tmp_path / valid_rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / valid_rel).write_text("class SafetyKAN: pass\n")

        (tmp_path / corrupt_rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / corrupt_rel).write_text('{"data": []}\n')

        result = DeliverableContentValidator.audit_known_corrupt_files(str(tmp_path))
        assert result[valid_rel] == "valid_python"
        assert result[corrupt_rel] == "corrupt_json"
        # Remaining three should be missing
        for rel in self.KNOWN_FILES:
            if rel not in (valid_rel, corrupt_rel):
                assert result[rel] == "missing"

    def test_all_valid_python_status(self, tmp_path: Path) -> None:
        # All five files are valid Python
        for rel in self.KNOWN_FILES:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# valid\npass\n")
        result = DeliverableContentValidator.audit_known_corrupt_files(str(tmp_path))
        for status in result.values():
            assert status == "valid_python"


# ---------------------------------------------------------------------------
# CloudGPUInstructions dataclass
# ---------------------------------------------------------------------------


class TestCloudGPUInstructions:
    """REQ-INFRA-020: CloudGPUInstructions dataclass fields."""

    def test_fields_present(self) -> None:
        inst = CloudGPUInstructions(
            lambda_command="lambdalabs instance create --instance-type gpu_1x_a100",
            vastai_command="vastai create instance <id>",
            runpod_command="runpodctl create pod --gpuType NVIDIA_A100_80GB",
            estimated_cost_per_hour_usd=1.10,
        )
        assert inst.lambda_command.startswith("lambdalabs")
        assert inst.vastai_command.startswith("vastai")
        assert inst.runpod_command.startswith("runpodctl")
        assert inst.estimated_cost_per_hour_usd == pytest.approx(1.10)

    def test_is_dataclass(self) -> None:
        from dataclasses import fields as dc_fields

        field_names = {f.name for f in dc_fields(CloudGPUInstructions)}
        assert field_names == {
            "lambda_command",
            "vastai_command",
            "runpod_command",
            "estimated_cost_per_hour_usd",
        }


# ---------------------------------------------------------------------------
# build_cloud_gpu_instructions
# ---------------------------------------------------------------------------


class TestBuildCloudGPUInstructions:
    """REQ-INFRA-020: build_cloud_gpu_instructions returns correct commands."""

    def test_returns_cloud_gpu_instructions_instance(self) -> None:
        result = build_cloud_gpu_instructions()
        assert isinstance(result, CloudGPUInstructions)

    def test_lambda_command(self) -> None:
        result = build_cloud_gpu_instructions()
        assert "lambdalabs" in result.lambda_command
        assert "gpu_1x_a100" in result.lambda_command
        assert "us-west-2" in result.lambda_command

    def test_vastai_command(self) -> None:
        result = build_cloud_gpu_instructions()
        assert "vastai" in result.vastai_command
        assert "pytorch" in result.vastai_command

    def test_runpod_command(self) -> None:
        result = build_cloud_gpu_instructions()
        assert "runpodctl" in result.runpod_command
        assert "A100" in result.runpod_command

    def test_estimated_cost(self) -> None:
        result = build_cloud_gpu_instructions()
        assert result.estimated_cost_per_hour_usd == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# generate_cloud_gpu_script
# ---------------------------------------------------------------------------


class TestGenerateCloudGPUScript:
    """REQ-INFRA-020 / SCENARIO-INFRA-024: generate_cloud_gpu_script writes shell script."""

    def test_creates_file(self, tmp_path: Path) -> None:
        output = str(tmp_path / "setup_cloud_gpu.sh")
        inst = build_cloud_gpu_instructions()
        generate_cloud_gpu_script(inst, output)
        assert os.path.exists(output)

    def test_file_contains_lambda_command(self, tmp_path: Path) -> None:
        output = str(tmp_path / "setup_cloud_gpu.sh")
        inst = build_cloud_gpu_instructions()
        generate_cloud_gpu_script(inst, output)
        content = Path(output).read_text()
        assert inst.lambda_command in content

    def test_file_contains_vastai_command(self, tmp_path: Path) -> None:
        output = str(tmp_path / "setup_cloud_gpu.sh")
        inst = build_cloud_gpu_instructions()
        generate_cloud_gpu_script(inst, output)
        content = Path(output).read_text()
        assert inst.vastai_command in content

    def test_file_contains_runpod_command(self, tmp_path: Path) -> None:
        output = str(tmp_path / "setup_cloud_gpu.sh")
        inst = build_cloud_gpu_instructions()
        generate_cloud_gpu_script(inst, output)
        content = Path(output).read_text()
        assert inst.runpod_command in content

    def test_file_contains_all_three_providers(self, tmp_path: Path) -> None:
        # All three cloud provider sections must appear
        output = str(tmp_path / "setup_cloud_gpu.sh")
        inst = build_cloud_gpu_instructions()
        generate_cloud_gpu_script(inst, output)
        content = Path(output).read_text()
        assert "Lambda" in content or "lambda" in content.lower()
        assert "vast" in content.lower()
        assert "runpod" in content.lower() or "RunPod" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output = str(tmp_path / "deep" / "nested" / "setup.sh")
        inst = build_cloud_gpu_instructions()
        generate_cloud_gpu_script(inst, output)
        assert os.path.exists(output)

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        output = str(tmp_path / "setup.sh")
        Path(output).write_text("old content\n")
        inst = build_cloud_gpu_instructions()
        generate_cloud_gpu_script(inst, output)
        content = Path(output).read_text()
        assert "old content" not in content
        assert inst.lambda_command in content
