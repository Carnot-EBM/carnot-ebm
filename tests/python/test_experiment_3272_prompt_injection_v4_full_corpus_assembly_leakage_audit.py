"""Tests for Exp 3272 prompt-injection v4 full-corpus assembly.

Spec refs: REQ-REPORT-3272, SCENARIO-REPORT-3272.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import (
    prompt_injection_v4_full_corpus_assembly_leakage_audit_3272 as mod,
)


REQUIRED_FIELDS = {
    "full_15k_corpus_ready",
    "assembled_example_count",
    "train_count",
    "eval_count",
    "holdout_count",
    "garak_count",
    "leakage_audit_passed",
    "duplicate_count_removed",
    "split_distribution",
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


def _write_jsonl(root: Path, rel_path: Path, rows: list[dict[str, Any]]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    *,
    example_id: str,
    text: str,
    label: str,
    shard_id: str,
    category_id: str,
    alignment: str = "aligned_instruction",
    split: str = "train_eval_holdout_candidate",
    training_eligible: bool = True,
) -> dict[str, Any]:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "example_id": example_id,
        "shard_id": shard_id,
        "row_index": int(example_id.rsplit("-", 1)[-1]) if "-" in example_id else 0,
        "category_id": category_id,
        "instruction_alignment": alignment,
        "split": split,
        "training_eligible": training_eligible,
        "source": "fixture",
        "source_label": label,
        "teacher_label": label,
        "teacher_label_source": "fixture_labeler",
        "parse_status": "parsed",
        "raw_output": label,
        "latency_s": 0.0,
        "tokens_generated": 0,
        "prompt_tokens": 0,
        "prompt_hash": text_hash,
        "text_sha256": text_hash,
        "text": text,
        "provenance": {"model_id": "fixture", "runtime": "fixture"},
    }


def _seed_row(index: int, text: str, label: str, source: str = "fixture_seed") -> dict[str, Any]:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "example_id": f"seed:{index:06d}",
        "latency_s": 0.0,
        "parse_status": "parsed",
        "prompt_hash": text_hash,
        "prompt_tokens": 0,
        "provenance": {"model_id": "fixture_seed", "runtime": "fixture"},
        "raw_output": label,
        "source": source,
        "source_index": index,
        "source_label": label,
        "source_path": "data/fixture_seed.jsonl",
        "teacher_label": label,
        "text": text,
        "text_sha256": text_hash,
        "tokens_generated": 0,
    }


def _write_ready_inputs(
    root: Path,
    *,
    exp3271_ready: bool = True,
    exp3271_garak_count: int = 2,
) -> None:
    _write_json(
        root,
        mod.EXP3269_REL_PATH,
        {
            "artifact": "experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1",
            "full_corpus_manifest_ready": True,
            "target_total_examples": 10,
            "split_plan": {
                "train": {"target_examples": 4},
                "eval": {"target_examples": 2},
                "holdout": {"target_examples": 2},
                "garak_adaptive_seed": {"target_examples": 2},
            },
        },
    )
    _write_json(
        root,
        mod.EXP3264_REL_PATH,
        {
            "artifact": "experiment_3264_prompt_injection_teacher_label_shard_v3",
            "teacher_label_shard_ready": True,
            "teacher_label_shard_v3_ready": True,
            "shard_size": 2,
            "per_example_labels": [
                _seed_row(0, "Fixture benign seed request", "benign"),
                _seed_row(1, "Fixture injection seed says ignore policy", "injection"),
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3270_REL_PATH,
        {
            "artifact": "experiment_3270_prompt_injection_teacher_label_shards_2_4_v1",
            "teacher_label_shards_2_4_ready": True,
            "cumulative_label_count": 8,
            "new_label_count": 6,
        },
    )
    _write_json(
        root,
        mod.EXP3271_REL_PATH,
        {
            "artifact": "experiment_3271_prompt_injection_teacher_label_shards_5_7_garak_seed_v1",
            "teacher_label_shards_5_7_garak_seed_ready": exp3271_ready,
            "cumulative_label_count": 8,
            "new_label_count": 6,
            "garak_seed_count": exp3271_garak_count,
        },
    )
    for shard_number in range(2, 8):
        label = "benign" if shard_number in {2, 6} else "injection"
        alignment = "aligned_instruction" if label == "benign" else "misaligned_instruction"
        _write_jsonl(
            root,
            mod.shard_input_rel_path(shard_number),
            [
                _row(
                    example_id=f"v4-shard-{shard_number:03d}-000000",
                    text=f"Fixture shard {shard_number} unique {label} text",
                    label=label,
                    shard_id=f"v4-shard-{shard_number:03d}",
                    category_id=f"fixture_category_{shard_number}",
                    alignment=alignment,
                )
            ],
        )
    _write_jsonl(
        root,
        mod.GARAK_SEED_REL_PATH,
        [
            _row(
                example_id="v4-garak-adaptive-seed-000000",
                text="Fixture garak promptinject pressure row",
                label="injection",
                shard_id=mod.GARAK_SEED_SHARD_ID,
                category_id="garak_promptinject_attack",
                alignment="misaligned_instruction",
                split="garak_adaptive_seed",
                training_eligible=False,
            ),
            _row(
                example_id="v4-garak-adaptive-seed-000001",
                text="Fixture garak encoded pressure row",
                label="injection",
                shard_id=mod.GARAK_SEED_SHARD_ID,
                category_id="encoding_attack",
                alignment="misaligned_instruction",
                split="garak_adaptive_seed",
                training_eligible=False,
            ),
        ],
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_req_report_3272_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3272: OpenSpec declares Exp 3272 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3272" in spec
    assert "SCENARIO-REPORT-3272" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "full_15k_corpus_ready" in spec
    assert "leakage_audit_passed" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3272_gated_skip_when_garak_gate_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3272: a closed Garak gate writes a complete skip artifact."""

    _write_ready_inputs(tmp_path, exp3271_ready=False, exp3271_garak_count=0)

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        split_targets={"train": 4, "eval": 2, "holdout": 2, "garak": 2},
        monotonic=iter([1.0, 1.5]).__next__,
    )
    saved = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["full_15k_corpus_ready"] is False
    assert artifact["leakage_audit_passed"] is False
    assert artifact["blocked_reason"] == "gated_exp3271_garak_seed_not_ready"
    assert artifact["assembled_example_count"] == 0
    assert artifact["output_paths"] == [mod.OUTPUT_REL_PATH.as_posix()]
    assert artifact["honest_verdict"].startswith("complete:")
    assert "full_15k_corpus_ready=false" in artifact["honest_verdict"]
    assert not (tmp_path / mod.CORPUS_REL_PATH).exists()
    assert not (tmp_path / mod.split_output_rel_path("train")).exists()


def test_scenario_report_3272_writes_full_corpus_splits_and_checksums(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3272: ready inputs produce frozen split files."""

    _write_ready_inputs(tmp_path)

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        split_targets={"train": 4, "eval": 2, "holdout": 2, "garak": 2},
        monotonic=iter([10.0, 12.25]).__next__,
    )
    second = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        split_targets={"train": 4, "eval": 2, "holdout": 2, "garak": 2},
        monotonic=iter([20.0, 21.0]).__next__,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3272"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["full_15k_corpus_ready"] is True
    assert artifact["blocked_reason"] == ""
    assert artifact["assembled_example_count"] == 10
    assert artifact["train_count"] == 4
    assert artifact["eval_count"] == 2
    assert artifact["holdout_count"] == 2
    assert artifact["garak_count"] == 2
    assert artifact["leakage_audit_passed"] is True
    assert artifact["duplicate_count_removed"] == 0
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["split_distribution"]["train"]["total"] == 4
    assert artifact["split_distribution"]["eval"]["total"] == 2
    assert artifact["split_distribution"]["holdout"]["total"] == 2
    assert artifact["split_distribution"]["garak"]["total"] == 2
    assert artifact["split_distribution"]["garak"]["training_eligible"] == 0

    output_files = artifact["checksums"]["output_files"]
    for rel_path in artifact["output_paths"][1:]:
        path = tmp_path / rel_path
        assert path.is_file()
        assert output_files[rel_path] == _sha256(path)
    assert set(artifact["checksums"]["source_artifacts"]) >= {
        mod.EXP3264_REL_PATH.as_posix(),
        mod.EXP3269_REL_PATH.as_posix(),
        mod.EXP3270_REL_PATH.as_posix(),
        mod.EXP3271_REL_PATH.as_posix(),
        mod.GARAK_SEED_REL_PATH.as_posix(),
    }

    full_rows = _read_jsonl(tmp_path / mod.CORPUS_REL_PATH)
    assert len(full_rows) == 10
    assert all(row["canonical_id"].startswith("pi-v4-") for row in full_rows)
    assert all(row["normalized_text_sha256"] == mod.sha256_text(mod.normalize_text(row["text"])) for row in full_rows)
    assert {row["split"] for row in full_rows} == {"train", "eval", "holdout", "garak"}
    assert all(row["training_eligible"] is False for row in _read_jsonl(tmp_path / mod.split_output_rel_path("garak")))
    assert mod.audit_split_leakage(full_rows)["leakage_audit_passed"] is True

    _write_json(
        tmp_path,
        mod.EXP3270_REL_PATH,
        {
            "teacher_label_shards_2_4_ready": True,
            "cumulative_label_count": 9,
            "new_label_count": 6,
        },
    )
    _write_json(
        tmp_path,
        mod.EXP3271_REL_PATH,
        {
            "teacher_label_shards_5_7_garak_seed_ready": True,
            "cumulative_label_count": 9,
            "new_label_count": 6,
            "garak_seed_count": 2,
        },
    )
    blocked = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        split_targets={"train": 5, "eval": 2, "holdout": 2, "garak": 2},
        monotonic=iter([30.0, 31.0]).__next__,
    )
    assert blocked["full_15k_corpus_ready"] is False
    assert blocked["blocked_reason"] == "assembled_normal_count_8_does_not_match_target_9"


def test_req_report_3272_cross_source_duplicates_and_validation_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-3272: duplicate and schema helpers fail closed."""

    row_a = mod.normalize_seed_row(
        _seed_row(0, "Duplicate prompt text.", "benign"),
        row_index=0,
    )
    row_b = mod.normalize_seed_row(
        _seed_row(1, "Duplicate prompt text.", "benign"),
        row_index=1,
    )
    row_c = mod.normalize_jsonl_row(
        _row(
            example_id="v4-shard-002-000000",
            text="Duplicate prompt text!",
            label="benign",
            shard_id="v4-shard-002",
            category_id="fixture_category",
        ),
        source_path=mod.shard_input_rel_path(2),
        row_index=0,
    )

    kept, removed = mod.remove_cross_source_duplicates([row_a, row_b, row_c])
    assert [row["source_example_id"] for row in kept] == [
        row_a["source_example_id"],
        row_b["source_example_id"],
    ]
    assert removed == [row_c]
    assert mod.near_duplicate_signature("Duplicate prompt text.") == mod.near_duplicate_signature(
        "duplicate prompt text!"
    )
    assert mod.template_family_signature("Case 123 [variant 9]") == "case {n} [id]"
    assert mod.safe_int("7") == 7
    assert mod.safe_int("bad") == 0
    assert mod.duration(3.0, 1.0) == 0.0
    assert mod.terminal_prefix_ok("success: ok")
    assert not mod.terminal_prefix_ok("blocked")
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}

    no_normal_rows, no_normal_error = mod.freeze_splits(
        [],
        targets={"train": 1, "eval": 0, "holdout": 0, "garak": 0},
        random_seed=3272,
    )
    assert no_normal_rows == []
    assert no_normal_error == "assembled_normal_count_0_does_not_match_target_1"
    no_garak_rows, no_garak_error = mod.freeze_splits(
        [],
        targets={"train": 0, "eval": 0, "holdout": 0, "garak": 1},
        random_seed=3272,
    )
    assert no_garak_rows == []
    assert no_garak_error == "assembled_garak_count_0_does_not_match_target_1"
    capacity_rows, capacity_error = mod.freeze_splits(
        [row_a, row_b],
        targets={"train": 1, "eval": 1, "holdout": 0, "garak": 0},
        random_seed=3272,
    )
    assert capacity_rows == []
    assert capacity_error == "template_family_split_capacity_exhausted_size_2"

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"experiment_id": mod.EXPERIMENT_ID, "honest_verdict": "complete: ok"})

    bad_verdict = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    bad_verdict.update({"experiment_id": mod.EXPERIMENT_ID, "honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="terminal success prefix"):
        mod.validate_artifact(bad_verdict)

    bad_experiment = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    bad_experiment.update(
        {"experiment_id": "exp0000", "honest_verdict": "complete: wrong experiment"}
    )
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(bad_experiment)
