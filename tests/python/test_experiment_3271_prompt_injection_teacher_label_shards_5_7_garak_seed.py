"""Tests for Exp 3271 prompt-injection shards 5-7 plus Garak seed.

Spec refs: REQ-REPORT-3271, SCENARIO-REPORT-3271.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import prompt_injection_teacher_label_shards_5_7_garak_seed_3271 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"
SELECTED_PYTHON = "/repo/.venv/bin/python"

REQUIRED_FIELDS = {
    "teacher_label_shards_5_7_garak_seed_ready",
    "cumulative_label_count",
    "new_label_count",
    "garak_seed_count",
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
    path = cache_root / f"models--{owner}--{name}" / "snapshots" / "rev1" / f"{stem}-Q4_K_M.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _command(
    command: list[str], *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> dict[str, Any]:
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


def _write_upstreams(
    root: Path,
    *,
    exp3270_cumulative: int = 8000,
    exp3270_ready: bool = True,
    manifest_ready: bool = True,
) -> None:
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
        mod.EXP3269_REL_PATH,
        {
            "artifact": "experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1",
            "full_corpus_manifest_ready": manifest_ready,
            "garak_seed_target": 1000,
            "shard_plan": [
                {
                    "shard_id": f"v4-shard-{index:03d}",
                    "target_examples": 4,
                    "category_focus": mod.CATEGORY_FOCUS_BY_SHARD[index],
                    "teacher_label_deliverable": mod.OUTPUT_REL_PATH.as_posix(),
                }
                for index in mod.TARGET_SHARD_NUMBERS
            ]
            + [
                {
                    "shard_id": mod.GARAK_SEED_SHARD_ID,
                    "target_examples": 10,
                    "category_focus": list(mod.GARAK_SEED_CATEGORIES),
                    "teacher_label_deliverable": mod.OUTPUT_REL_PATH.as_posix(),
                }
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3270_REL_PATH,
        {
            "artifact": "experiment_3270_prompt_injection_teacher_label_shards_2_4_v1",
            "teacher_label_shards_2_4_ready": exp3270_ready,
            "cumulative_label_count": exp3270_cumulative,
            "new_label_count": max(0, exp3270_cumulative - 2000),
            "honest_verdict": "complete: teacher_label_shards_2_4_ready=true",
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
            "prompt_tokens": 40,
        }
        for row in rows
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_req_report_3271_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3271: OpenSpec declares Exp 3271 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3271" in spec
    assert "SCENARIO-REPORT-3271" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "teacher_label_shards_5_7_garak_seed_ready" in spec
    assert "garak_seed_count=1000" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3271_gated_skip_when_exp3270_below_count(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3271: Exp 3270 count gate produces a complete skip artifact."""

    _write_upstreams(tmp_path, exp3270_cumulative=7999)
    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, GEMMA26)
    runner, calls = _runner()

    def fail_labeler(
        _rows: list[dict[str, Any]], _model_specs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        raise AssertionError("SOTA labeler must not run when Exp 3270 is below the gate")

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        sota_labeler=fail_labeler,
        monotonic=iter([1.0, 1.25]).__next__,
        shard_target_size=4,
        garak_seed_target=10,
        panel_rows_per_category=1,
    )
    saved = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["teacher_label_shards_5_7_garak_seed_ready"] is False
    assert artifact["blocked_reason"] == "gated_exp3270_cumulative_label_count_below_8000"
    assert artifact["new_label_count"] == 0
    assert artifact["garak_seed_count"] == 0
    assert artifact["cumulative_label_count"] == 7999
    assert artifact["shard_counts"] == {}
    assert artifact["checksums"]["shard_files"] == {}
    assert artifact["checksums"]["garak_seed_file"] == {}
    assert artifact["output_paths"] == [mod.OUTPUT_REL_PATH.as_posix()]
    assert artifact["honest_verdict"].startswith("complete:")
    assert "teacher_label_shards_5_7_garak_seed_ready=false" in artifact["honest_verdict"]
    assert any(call["command"][:1] == ["nvidia-smi"] for call in calls)


def test_scenario_report_3271_writes_ready_shards_seed_counts_and_checksums(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3271: shards 5-7 and Garak seed are auditable."""

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
        garak_seed_target=10,
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
        garak_seed_target=10,
        panel_rows_per_category=1,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3271"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["teacher_label_shards_5_7_garak_seed_ready"] is True
    assert artifact["blocked_reason"] == ""
    assert artifact["new_label_count"] == 12
    assert artifact["garak_seed_count"] == 10
    assert artifact["cumulative_label_count"] == 8012
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    assert "garak_seed_count=10" in artifact["honest_verdict"]

    assert artifact["shard_counts"] == {
        "v4-garak-adaptive-seed": {
            "total": 10,
            "benign": 0,
            "injection": 10,
            "aligned_instruction": 0,
            "misaligned_instruction": 10,
            "non_instruction": 0,
            "category_counts": {
                "dataflip_kad_adaptive_attack": 2,
                "encoding_attack": 2,
                "garak_promptinject_attack": 2,
                "long_reasoning_heavy_attack": 2,
                "tool_rag_indirect_injection_attack": 2,
            },
        },
        "v4-shard-005": {
            "total": 4,
            "benign": 0,
            "injection": 4,
            "aligned_instruction": 0,
            "misaligned_instruction": 4,
            "non_instruction": 0,
            "category_counts": {
                "long_reasoning_heavy_attack": 2,
                "misaligned_instruction_attack": 2,
            },
        },
        "v4-shard-006": {
            "total": 4,
            "benign": 2,
            "injection": 2,
            "aligned_instruction": 2,
            "misaligned_instruction": 2,
            "non_instruction": 0,
            "category_counts": {
                "aligned_instruction_benign": 2,
                "dataflip_kad_adaptive_attack": 2,
            },
        },
        "v4-shard-007": {
            "total": 4,
            "benign": 0,
            "injection": 4,
            "aligned_instruction": 0,
            "misaligned_instruction": 4,
            "non_instruction": 0,
            "category_counts": {
                "encoding_attack": 2,
                "tool_rag_indirect_injection_attack": 2,
            },
        },
    }
    assert artifact["label_distribution"]["total"] == 22
    assert artifact["label_distribution"]["benign"] == 2
    assert artifact["label_distribution"]["injection"] == 20
    assert artifact["label_distribution"]["normal_corpus"]["total"] == 12
    assert artifact["label_distribution"]["garak_adaptive_seed"]["total"] == 10

    assert artifact["model_specs"]["selected_mandated_model_id"] == GEMMA26
    assert artifact["model_specs"]["selected_mandated_model_path"] == str(model_path)
    assert artifact["model_specs"]["garak_seed_target"] == 10
    assert artifact["models_used"][0]["model_id"] == GEMMA26
    assert artifact["models_used"][0]["examples_labeled"] == 11
    assert artifact["models_used"][1]["examples_labeled"] == 6
    assert artifact["models_used"][2]["examples_labeled"] == 5
    assert artifact["headline_label_evidence"]["parsed_count"] == 11
    assert artifact["headline_label_evidence"]["agreement_with_source_label"] == pytest.approx(1.0)

    shard_paths = [
        tmp_path / mod.shard_output_rel_path(shard_number)
        for shard_number in mod.TARGET_SHARD_NUMBERS
    ]
    garak_path = tmp_path / mod.GARAK_SEED_REL_PATH
    assert artifact["output_paths"] == [
        mod.OUTPUT_REL_PATH.as_posix(),
        *[path.relative_to(tmp_path).as_posix() for path in shard_paths],
        mod.GARAK_SEED_REL_PATH.as_posix(),
    ]
    for path in [*shard_paths, garak_path]:
        rows = _read_jsonl(path)
        rel = path.relative_to(tmp_path).as_posix()
        expected = 10 if path == garak_path else 4
        checksum_group = (
            artifact["checksums"]["garak_seed_file"]
            if path == garak_path
            else artifact["checksums"]["shard_files"]
        )
        assert len(rows) == expected
        assert checksum_group[rel] == _sha256(path)
        assert rows[0]["example_id"].startswith(rows[0]["shard_id"])
        assert rows[0]["teacher_label"] in mod.ALLOWED_LABELS
        assert rows[0]["parse_status"] == "parsed"
        assert rows[0]["text_sha256"] == mod.sha256_text(rows[0]["text"])
        assert "model_id" in rows[0]["provenance"]
    assert all(row["training_eligible"] is False for row in _read_jsonl(garak_path))


def test_req_report_3271_sota_evidence_failures_do_not_write_ready_outputs(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3271: missing parsed SOTA evidence fails closed."""

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
        garak_seed_target=10,
        panel_rows_per_category=1,
    )

    assert artifact["teacher_label_shards_5_7_garak_seed_ready"] is False
    assert artifact["blocked_reason"] == "sota_label_evidence_incomplete_or_unparseable"
    assert artifact["new_label_count"] == 0
    assert artifact["garak_seed_count"] == 0
    assert artifact["checksums"]["shard_files"] == {}
    assert artifact["checksums"]["garak_seed_file"] == {}
    assert not (tmp_path / mod.shard_output_rel_path(5)).exists()
    assert not (tmp_path / mod.GARAK_SEED_REL_PATH).exists()
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3271_helpers_parse_validate_and_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3271: helper behavior is deterministic and validated."""

    assert mod.parse_teacher_label(" benign\n") == ("benign", "parsed")
    assert mod.parse_teacher_label("FINAL_LABEL: injection") == ("injection", "parsed")
    assert mod.parse_teacher_label("not sure") == ("abstain", "parse_failed")
    assert mod.safe_int("42") == 42
    assert mod.safe_int("nope") == 0
    assert mod.duration(3.0, 2.0) == 0.0
    assert mod.terminal_prefix_ok("success: done")
    assert not mod.terminal_prefix_ok("blocked")

    normal_rows = mod.generate_shard_rows(shard_number=5, shard_target_size=4, random_seed=3271)
    seed_rows = mod.generate_garak_seed_rows(garak_seed_target=10, random_seed=3271)
    assert [row["category_id"] for row in normal_rows].count("long_reasoning_heavy_attack") == 2
    assert [row["category_id"] for row in normal_rows].count("misaligned_instruction_attack") == 2
    assert [row["category_id"] for row in seed_rows].count("garak_promptinject_attack") == 2
    assert mod.compute_shard_counts(normal_rows)["total"] == 4
    assert mod.compute_label_distribution([*normal_rows, *seed_rows])["injection"] == 14
    bad_panel = mod.normalize_panel_evidence(
        panel_rows=[normal_rows[0]],
        label_outputs=[{"teacher_label": "other", "raw_output": "other"}],
        model_specs={
            "selected_mandated_model_id": GEMMA26,
            "selected_mandated_model_path": "/model.gguf",
        },
    )
    assert bad_panel["rows"][0]["teacher_label"] == "abstain"
    assert bad_panel["rows"][0]["parse_status"] == "parse_failed"

    assert mod.prior_cumulative_label_count({"teacher_label_shards_2_4_ready": False}) == 0
    assert mod.prior_cumulative_label_count({"cumulative_label_count": "8000"}) == 8000
    assert mod.garak_availability_probe()["deterministic_seed_fallback"] is True

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
