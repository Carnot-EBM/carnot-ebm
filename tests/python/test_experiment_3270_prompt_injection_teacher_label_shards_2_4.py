"""Tests for Exp 3270 prompt-injection teacher-label shards 2-4.

Spec refs: REQ-REPORT-3270, SCENARIO-REPORT-3270.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import prompt_injection_teacher_label_shards_2_4_3270 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"
SELECTED_PYTHON = "/repo/.venv/bin/python"

REQUIRED_FIELDS = {
    "teacher_label_shards_2_4_ready",
    "cumulative_label_count",
    "new_label_count",
    "shard_counts",
    "label_distribution",
    "model_specs",
    "models_used",
    "preconditions_checked",
    "output_paths",
    "checksums",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_model(cache_root: Path, hf_id: str, *, content: bytes = b"gguf fixture") -> Path:
    owner, name = hf_id.split("/", 1)
    stem = name.removesuffix("-GGUF")
    path = (
        cache_root
        / f"models--{owner}--{name}"
        / "snapshots"
        / "rev1"
        / f"{stem}-Q4_K_M.gguf"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _command(command: list[str], *, returncode: int = 0, stdout: str = "", stderr: str = "") -> dict[str, Any]:
    return {
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_summary": stdout,
        "stderr_summary": stderr,
    }


def _cuda_probe_stdout(*, cuda_available: bool = True, llama_cpp_cuda: bool = True) -> str:
    return (
        json.dumps(
            {
                "python": SELECTED_PYTHON,
                "torch_import_ok": True,
                "cuda_available": cuda_available,
                "cuda_device_count": 2 if cuda_available else 0,
                "cuda_device_name": "NVIDIA GeForce RTX 3090" if cuda_available else "",
                "llama_cpp_import_ok": True,
                "llama_cpp_supports_gpu_offload": llama_cpp_cuda,
                "llama_cpp_system_info": "CUDA : ARCHS = 860" if llama_cpp_cuda else "CPU",
            },
            sort_keys=True,
        )
        + "\n"
    )


def _runner(
    *,
    nvidia_ok: bool = True,
    cuda_available: bool = True,
    llama_cpp_cuda: bool = True,
) -> tuple[mod.CommandRunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append({"command": command, "kwargs": kwargs})
        joined = "\n".join(command)
        if command[:1] == ["nvidia-smi"]:
            if not nvidia_ok:
                return _command(command, returncode=1, stderr="nvidia-smi failed\n")
            return _command(
                command,
                stdout=(
                    "0, NVIDIA GeForce RTX 3090, 24576, 4, 0, 595.71.05\n"
                    "1, NVIDIA GeForce RTX 3090, 24576, 4, 0, 595.71.05\n"
                ),
            )
        if "exp3268_cuda_probe" in joined:
            return _command(
                command,
                returncode=0 if cuda_available else 1,
                stdout=_cuda_probe_stdout(
                    cuda_available=cuda_available,
                    llama_cpp_cuda=llama_cpp_cuda,
                ),
                stderr="" if cuda_available else "cuda unavailable\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    return run, calls


def _write_upstreams(root: Path, *, manifest_ready: bool = True, seed_size: int = 2000) -> None:
    _write_json(
        root,
        mod.EXP3268_REL_PATH,
        {
            "artifact": "experiment_3268_sota_receipt_methodology_supplement_v1",
            "clean_sota_receipt_eligible": True,
            "models_used": [
                {
                    "model_id": GEMMA26,
                    "attempted_live_receipt": True,
                    "clean_row": True,
                }
            ],
            "honest_verdict": "complete: clean_sota_receipt_eligible=true",
        },
    )
    _write_json(
        root,
        mod.EXP3264_REL_PATH,
        {
            "artifact": "experiment_3264_prompt_injection_teacher_label_shard_v3",
            "teacher_label_shard_ready": seed_size > 0,
            "teacher_label_shard_v3_ready": seed_size > 0,
            "shard_size": seed_size,
            "label_counts": {"benign": 1459, "injection": 541} if seed_size else {},
        },
    )
    _write_json(
        root,
        mod.EXP3269_REL_PATH,
        {
            "artifact": "experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1",
            "full_corpus_manifest_ready": manifest_ready,
            "shard_plan": [
                {
                    "shard_id": f"v4-shard-{index:03d}",
                    "target_examples": 4,
                    "category_focus": mod.CATEGORY_FOCUS_BY_SHARD[index],
                    "teacher_label_deliverable": mod.OUTPUT_REL_PATH.as_posix(),
                }
                for index in mod.TARGET_SHARD_NUMBERS
            ],
        },
    )


def _panel_labeler(rows: list[dict[str, Any]], model_specs: dict[str, Any]) -> list[dict[str, Any]]:
    assert model_specs["selected_mandated_model_id"] == GEMMA26
    return [
        {
            "example_id": row["example_id"],
            "teacher_label": row["source_label"],
            "raw_output": row["source_label"],
            "parse_status": "parsed",
            "latency_s": 0.01,
            "tokens_generated": 2,
            "prompt_tokens": 32,
        }
        for row in rows
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_req_report_3270_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3270: OpenSpec declares Exp 3270 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3270" in spec
    assert "SCENARIO-REPORT-3270" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "teacher_label_shards_2_4_ready" in spec
    assert "new_label_count=6000" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3270_gated_skip_when_manifest_not_ready(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3270: closed upstream gates produce complete skip artifacts."""

    _write_upstreams(tmp_path, manifest_ready=False)
    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, GEMMA26)
    runner, calls = _runner()

    def fail_labeler(_rows: list[dict[str, Any]], _model_specs: dict[str, Any]) -> list[dict[str, Any]]:
        raise AssertionError("SOTA labeler must not run when Exp 3269 is closed")

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        sota_labeler=fail_labeler,
        monotonic=iter([1.0, 1.25]).__next__,
        shard_target_size=4,
        panel_rows_per_category=1,
    )
    saved = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["teacher_label_shards_2_4_ready"] is False
    assert artifact["blocked_reason"] == "gated_exp3269_full_corpus_manifest_not_ready"
    assert artifact["new_label_count"] == 0
    assert artifact["cumulative_label_count"] == 2000
    assert artifact["shard_counts"] == {}
    assert artifact["checksums"]["shard_files"] == {}
    assert artifact["output_paths"] == [mod.OUTPUT_REL_PATH.as_posix()]
    assert artifact["honest_verdict"].startswith("complete:")
    assert "teacher_label_shards_2_4_ready=false" in artifact["honest_verdict"]
    assert any(call["command"][:1] == ["nvidia-smi"] for call in calls)


def test_scenario_report_3270_writes_ready_shards_with_counts_and_checksums(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3270: shards 2-4 add auditable labels and integrity checks."""

    _write_upstreams(tmp_path)
    cache_root = tmp_path / "hf-cache"
    model_path = _write_model(cache_root, GEMMA26)
    runner, _calls = _runner()

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        env={"EXTRA_ENV_FOR_TEST": "1"},
        command_runner=runner,
        sota_labeler=_panel_labeler,
        monotonic=iter([10.0, 12.5]).__next__,
        shard_target_size=4,
        panel_rows_per_category=1,
    )
    second = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        sota_labeler=_panel_labeler,
        monotonic=iter([20.0, 21.0]).__next__,
        shard_target_size=4,
        panel_rows_per_category=1,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3270"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.303"
    assert artifact["teacher_label_shards_2_4_ready"] is True
    assert artifact["blocked_reason"] == ""
    assert artifact["new_label_count"] == 12
    assert artifact["cumulative_label_count"] == 2012
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    assert "new_label_count=12" in artifact["honest_verdict"]

    assert artifact["shard_counts"] == {
        "v4-shard-002": {
            "total": 4,
            "benign": 2,
            "injection": 2,
            "aligned_instruction": 2,
            "misaligned_instruction": 2,
            "non_instruction": 0,
            "category_counts": {
                "aligned_instruction_benign": 2,
                "misaligned_instruction_attack": 2,
            },
        },
        "v4-shard-003": {
            "total": 4,
            "benign": 2,
            "injection": 2,
            "aligned_instruction": 0,
            "misaligned_instruction": 2,
            "non_instruction": 2,
            "category_counts": {
                "encoding_attack": 2,
                "non_instruction_benign": 2,
            },
        },
        "v4-shard-004": {
            "total": 4,
            "benign": 0,
            "injection": 4,
            "aligned_instruction": 0,
            "misaligned_instruction": 4,
            "non_instruction": 0,
            "category_counts": {
                "dataflip_kad_adaptive_attack": 2,
                "tool_rag_indirect_injection_attack": 2,
            },
        },
    }
    assert artifact["label_distribution"]["benign"] == 4
    assert artifact["label_distribution"]["injection"] == 8
    assert artifact["label_distribution"]["aligned_instruction"] == 2
    assert artifact["label_distribution"]["misaligned_instruction"] == 8
    assert artifact["label_distribution"]["non_instruction"] == 2

    assert artifact["model_specs"]["selected_mandated_model_id"] == GEMMA26
    assert artifact["model_specs"]["selected_mandated_model_path"] == str(model_path)
    assert artifact["model_specs"]["mandated_models"][GEMMA26]["cached"] is True
    assert artifact["models_used"][0]["model_id"] == GEMMA26
    assert artifact["models_used"][0]["examples_labeled"] == 6
    assert artifact["models_used"][1]["model_id"] == mod.MANIFEST_LABELER_ID
    assert artifact["models_used"][1]["examples_labeled"] == 6
    assert artifact["headline_label_evidence"]["parsed_count"] == 6
    assert artifact["headline_label_evidence"]["agreement_with_source_label"] == pytest.approx(1.0)

    shard_paths = [
        tmp_path / mod.shard_output_rel_path(shard_number)
        for shard_number in mod.TARGET_SHARD_NUMBERS
    ]
    assert artifact["output_paths"] == [
        mod.OUTPUT_REL_PATH.as_posix(),
        *[path.relative_to(tmp_path).as_posix() for path in shard_paths],
    ]
    for path in shard_paths:
        rows = _read_jsonl(path)
        assert len(rows) == 4
        rel = path.relative_to(tmp_path).as_posix()
        assert artifact["checksums"]["shard_files"][rel] == _sha256(path)
        assert rows[0]["example_id"].startswith(rows[0]["shard_id"])
        assert rows[0]["teacher_label"] in mod.ALLOWED_LABELS
        assert rows[0]["parse_status"] == "parsed"
        assert rows[0]["text_sha256"] == mod.sha256_text(rows[0]["text"])
        assert "model_id" in rows[0]["provenance"]


def test_req_report_3270_sota_evidence_failures_do_not_write_ready_shards(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3270: missing parsed SOTA evidence fails closed."""

    _write_upstreams(tmp_path)
    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, GEMMA26)
    runner, _calls = _runner()

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        sota_labeler=lambda _rows, _specs: [],
        monotonic=iter([3.0, 3.5]).__next__,
        shard_target_size=4,
        panel_rows_per_category=1,
    )

    assert artifact["teacher_label_shards_2_4_ready"] is False
    assert artifact["blocked_reason"] == "sota_label_evidence_incomplete_or_unparseable"
    assert artifact["new_label_count"] == 0
    assert artifact["checksums"]["shard_files"] == {}
    assert not (tmp_path / mod.shard_output_rel_path(2)).exists()
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3270_helpers_parse_validate_and_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3270: helper behavior is deterministic and validated."""

    assert mod.parse_teacher_label(" benign\n") == ("benign", "parsed")
    assert mod.parse_teacher_label("FINAL_LABEL: injection") == ("injection", "parsed")
    assert mod.parse_teacher_label("not sure") == ("abstain", "parse_failed")
    assert mod.safe_int("42") == 42
    assert mod.safe_int("nope") == 0
    assert mod.duration(3.0, 2.0) == 0.0
    assert mod.terminal_prefix_ok("success: done")
    assert not mod.terminal_prefix_ok("blocked")

    rows = mod.generate_shard_rows(shard_number=2, shard_target_size=4, random_seed=3270)
    assert [row["category_id"] for row in rows].count("aligned_instruction_benign") == 2
    assert [row["category_id"] for row in rows].count("misaligned_instruction_attack") == 2
    assert mod.compute_shard_counts(rows)["total"] == 4
    assert mod.compute_label_distribution(rows)["benign"] == 2
    bad_panel = mod.normalize_panel_evidence(
        panel_rows=[rows[0]],
        label_outputs=[{"teacher_label": "other", "raw_output": "other"}],
        model_specs={
            "selected_mandated_model_id": GEMMA26,
            "selected_mandated_model_path": "/model.gguf",
        },
    )
    assert bad_panel["rows"][0]["teacher_label"] == "abstain"
    assert bad_panel["rows"][0]["parse_status"] == "parse_failed"

    assert mod.completed_seed_count({"teacher_label_shard_ready": False}) == 0
    assert (
        mod.completed_seed_count(
            {
                "teacher_label_shard_ready": True,
                "shard_size": 0,
                "label_counts": {"benign": "2", "injection": 3},
            }
        )
        == 5
    )
    assert (
        mod.completed_seed_count(
            {"teacher_label_shard_ready": True, "shard_size": 0, "label_counts": []}
        )
        == 0
    )

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}

    missing = {
        "experiment_id": mod.EXPERIMENT_ID,
        "honest_verdict": "complete: ok",
    }
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_verdict = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    bad_verdict.update(
        {
            "experiment_id": mod.EXPERIMENT_ID,
            "honest_verdict": "blocked",
        }
    )
    with pytest.raises(ValueError, match="terminal success prefix"):
        mod.validate_artifact(bad_verdict)

    bad_experiment = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    bad_experiment.update(
        {
            "experiment_id": "exp0000",
            "honest_verdict": "complete: wrong experiment",
        }
    )
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(bad_experiment)
