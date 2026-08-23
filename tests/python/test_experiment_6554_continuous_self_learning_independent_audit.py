"""Tests for Exp6554 independent prospective CSL audit.

Spec refs: REQ-CL-6554, SCENARIO-CL-6554-MISSING-INPUT,
SCENARIO-CL-6554-RECEIPTS, SCENARIO-CL-6554-REPLAY,
SCENARIO-CL-6554-ROWS, SCENARIO-CL-6554-ATTACKS,
SCENARIO-CL-6554-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6553_prospective_sota_continuous_self_learning as upstream_mod
from carnot import experiment_6554_continuous_self_learning_independent_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6554", "exit_code": 0}]


class FakeInferenceBackend:
    """Small llama.cpp-shaped backend for SCENARIO-CL-6554-RECEIPTS."""

    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load_model(self, spec: dict[str, Any]) -> dict[str, Any]:
        self.loaded.append(spec["hf_id"])
        return {
            "hf_id": spec["hf_id"],
            "model_path": spec["model_path"],
            "loader": "llama_cpp.Llama",
            "load_ok": True,
            "smoke_ok": True,
            "embedded_tokenizer_ok": True,
            "process_id": 655300 + len(self.loaded),
            "load_s": 0.2,
            "smoke_s": 0.01,
            "error": "",
        }

    def infer(
        self,
        *,
        spec: dict[str, Any],
        query: dict[str, Any],
        arm_id: str,
        seed: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        del seed, timeout_s
        token_bonus = {"frozen": 8, "hysteretic": 2, "same_query_mutation": 1}.get(arm_id, 4)
        prompt_tokens = 24 + int(query["query_index"])
        output_tokens = token_bonus
        return {
            "terminal_status": "terminal",
            "exit_status": "ok",
            "timeout": False,
            "censored": False,
            "request_text": f"{spec['hf_id']} {query['query_id']} {arm_id}",
            "response_text": f"FINAL: exact-satisfying {arm_id} {query['query_id']}",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "model_wall_time_s": round(0.18 + output_tokens * 0.01, 6),
            "first_token_time_s": 0.03,
            "gpu_samples": [
                {
                    "gpu": spec["gpu"],
                    "memory_used_mb": 12000 + output_tokens,
                    "utilization_pct": 82,
                }
            ],
        }

    def close(self) -> None:
        return None


def _runtime_state() -> dict[str, Any]:
    return {
        "platform": "test",
        "python": "3.12",
        "gpu": {
            "available": True,
            "driver_version": "610.43.03",
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "vram_total_mb": 24576,
                    "vram_free_mb": 24576,
                    "driver_version": "610.43.03",
                },
                {
                    "index": 1,
                    "name": "NVIDIA GeForce RTX 3090",
                    "vram_total_mb": 24576,
                    "vram_free_mb": 24576,
                    "driver_version": "610.43.03",
                },
            ],
        },
        "llama_cpp": {
            "available": True,
            "version": "0.3.33",
            "cuda_backend_available": True,
            "gpu_offload_supported": True,
            "system_info": "CUDA",
            "error": "",
        },
        "llama_cpp_binary": {
            "path": "/tmp/llama-cli",
            "exists": True,
            "executable": True,
            "version": "version: 9606",
        },
        "z3": {"available": True, "version": "4.16.0"},
        "disk": {"checkpoint_dir_writable": True, "disk_free_bytes": 2_000_000_000},
    }


@pytest.fixture()
def clean_upstream_path(tmp_path: Path) -> Path:
    """SCENARIO-CL-6554-RECEIPTS: build stored evidence with raw files."""

    model_specs = []
    for index, hf_id in enumerate(upstream_mod.MANDATED_HF_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"fake gguf {hf_id}".encode())
        model_specs.append(
            {
                "name": upstream_mod.MODEL_NAMES_BY_HF_ID[hf_id],
                "hf_id": hf_id,
                "role": upstream_mod.MODEL_ROLES_BY_HF_ID[hf_id],
                "gpu": index % 2,
                "quantization": "Q4_K_M",
                "model_path": str(path),
            }
        )

    upstream = upstream_mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "experiment_6553.json",
        work_root=tmp_path / "work",
        write=False,
        duration_s=120.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=model_specs,
        runtime_state_override=_runtime_state(),
        tokenizer_probe=lambda _path: (True, "embedded tokenizer ok"),
        inference_backend=FakeInferenceBackend(),
    )
    model_hashes = {row["hf_id"]: row["gguf_sha256"] for row in upstream["MODEL_SPECS"]}
    model_paths = {row["hf_id"]: row["model_path"] for row in upstream["MODEL_SPECS"]}
    for row in upstream["per_unit_rows"]:
        request_text = f"{row['model_hf_id']} {row['query_id']} {row['arm_id']}"
        response_text = f"FINAL: exact-satisfying {row['arm_id']} {row['query_id']}"
        row.update(
            {
                "raw_request_text": request_text,
                "raw_response_text": response_text,
                "process_id": 700000 + int(row["query_index"]),
                "command": f"llama-cli --model {model_paths[row['model_hf_id']]}",
                "model_file_sha256": model_hashes[row["model_hf_id"]],
                "raw_receipt_path": str(tmp_path / "raw_receipts.jsonl"),
            }
        )
        row["row_hash"] = upstream_mod.row_hash(row)

    raw_path = tmp_path / "raw_receipts.jsonl"
    raw_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "row_hash": row["row_hash"],
                    "request_hash": row["request_hash"],
                    "response_hash": row["response_hash"],
                    "model_hf_id": row["model_hf_id"],
                    "query_id": row["query_id"],
                    "arm_id": row["arm_id"],
                },
                sort_keys=True,
            )
            for row in upstream["per_unit_rows"]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    journal_path = tmp_path / "journal.jsonl"
    checkpoint_path.write_text(json.dumps({"head": "clean"}), encoding="utf-8")
    journal_path.write_text(json.dumps({"event": "clean"}) + "\n", encoding="utf-8")
    upstream["raw_model_receipts"] = [{"path": str(raw_path), "sha256": mod.sha256_file(raw_path)}]
    upstream["checkpoint_receipts"] = [
        {"path": str(checkpoint_path), "sha256": mod.sha256_file(checkpoint_path)}
    ]
    upstream["journal_receipts"] = [
        {"path": str(journal_path), "sha256": mod.sha256_file(journal_path)}
    ]
    upstream["reproducibility_checksum"] = upstream_mod.reproducibility_checksum(upstream)
    upstream_path = tmp_path / "clean-upstream.json"
    upstream_path.write_text(json.dumps(upstream, indent=2, sort_keys=True) + "\n")
    return upstream_path


def test_req_cl_6554_spec_declares_independent_audit_contract() -> None:
    """REQ-CL-6554: OpenSpec owns the Exp6554 audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CL-6554") :]
    normalized = " ".join(section.split())
    for marker in (
        "SCENARIO-CL-6554-MISSING-INPUT",
        "SCENARIO-CL-6554-RECEIPTS",
        "SCENARIO-CL-6554-REPLAY",
        "SCENARIO-CL-6554-ROWS",
        "SCENARIO-CL-6554-ATTACKS",
        "SCENARIO-CL-6554-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "continuous_self_learning_audited_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_cl_6554_clean_audit_replays_rows(
    clean_upstream_path: Path, tmp_path: Path
) -> None:
    """SCENARIO-CL-6554-REPLAY/ROWS: clean evidence derives readiness."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        input_path=clean_upstream_path,
        result_path=tmp_path / "audit.json",
        write=True,
        duration_s=3.5,
        tests_run=TESTS_RUN,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_continuous_self_learning_independent_audit"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verdict_class"] == "null"
    assert artifact["continuous_self_learning_audited_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["input_existence_and_hash_receipts"]["all_required_inputs_present"] is True
    assert len(artifact["independent_live_receipt_audit_rows"]) == 756
    assert all(row["receipt_authentic"] for row in artifact["independent_live_receipt_audit_rows"])
    assert all(row["exact_replay_passed"] for row in artifact["independent_exact_replay_rows"])
    assert all(
        row["transition_replay_passed"] for row in artifact["independent_transition_replay_rows"]
    )
    assert artifact["dose_and_coobservation_audit"]["matched_update_dose"] is True
    assert artifact["unsafe_write_and_use_audit"]["safe_arm_unsafe_write_count"] == 0
    assert artifact["restart_rollback_and_persistence_audit"]["restart_rollback_passed"] is True
    assert artifact["attack_matrix"]["all_attacks_fail_closed"] is True
    assert artifact["aggregate_row_recomputation"] == mod.aggregate_row_recomputation(artifact)
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert json.loads((tmp_path / "audit.json").read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_cl_6554_missing_input_blocks_current_exp6553(tmp_path: Path) -> None:
    """SCENARIO-CL-6554-MISSING-INPUT: checked-in blocked Exp6553 stays blocked."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        hash_model_files=False,
    )

    assert artifact["status"] == "blocked_continuous_self_learning_audit_missing_inputs"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["continuous_self_learning_audited_ready_score"] == 0.0
    assert artifact["input_existence_and_hash_receipts"]["exp6553"]["exists"] is True
    assert artifact["independent_live_receipt_audit_rows"] == []
    assert artifact["per_unit_rows"] == []
    assert artifact["missing_input_disposition"]["terminal_disposition"] == "blocked"
    assert {
        "exp6553_per_unit_rows",
        "raw_model_receipts",
        "checkpoint_receipts",
        "journal_receipts",
    } <= set(artifact["missing_input_disposition"]["missing_inputs"])
    assert "gpu_vram_contract" in artifact["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_cl_6554_attack_validation_and_cli(
    clean_upstream_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-CL-6554-ATTACKS/ATOMIC: tampering and CLI paths are explicit."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        input_path=clean_upstream_path,
        result_path=tmp_path / "audit.json",
        write=False,
        duration_s=3.5,
        tests_run=TESTS_RUN,
    )

    assert mod._read_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json(bad_json) == {}
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod.sha256_file(None) == "missing"
    assert mod._arms({}) == mod.DEFAULT_ARMS  # noqa: SLF001
    assert mod._safe_arms({}) == tuple(  # noqa: SLF001
        arm for arm in mod.DEFAULT_ARMS if arm != mod.DIAGNOSTIC_ARM
    )
    assert mod.independent_current_effect_rows({"per_unit_rows": []}) == []
    assert mod.independent_retention_and_support_rows({"per_unit_rows": []}) == []
    assert (
        mod._status_and_verdict(  # noqa: SLF001
            {"verdict_class_from_rows": "disqualified"},
            {},
        )[2]
        == "disqualified"
    )
    assert mod._status_and_verdict({"verdict_class_from_rows": "partial"}, {})[2] == "partial"  # noqa: SLF001

    bad_row = deepcopy(artifact)
    bad_row["per_unit_rows"][0]["row_hash"] = "sha256:bad"
    bad_row["reproducibility_checksum"] = mod.reproducibility_checksum(bad_row)
    assert "per_unit_rows row_hash mismatch" in mod.validate_artifact(bad_row)

    bad_score = deepcopy(artifact)
    bad_score["continuous_self_learning_audited_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    assert "ready score mismatch" in mod.validate_artifact(bad_score)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "wrong"
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad_substrate)

    bad_aggregate = deepcopy(artifact)
    bad_aggregate["aggregate_row_recomputation"]["row_count"] = -1
    bad_aggregate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_aggregate)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad_aggregate)

    assert "required field set mismatch" in mod.validate_artifact({})
    validation_cases = [
        ("status", "running", "status lacks terminal prefix"),
        ("honest_verdict", "ready", "honest_verdict lacks terminal prefix"),
        ("verdict_class", "positive", "verdict_class must be closed"),
        ("verifier_is_oracle", True, "verifier_is_oracle must be false"),
        ("field_provenance", {}, "field_provenance must cover required fields"),
    ]
    for key, value, expected in validation_cases:
        changed = deepcopy(artifact)
        changed[key] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert expected in mod.validate_artifact(changed)

    wrong_clean_class = deepcopy(artifact)
    wrong_clean_class["verdict_class"] = "partial"
    wrong_clean_class["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_clean_class)
    assert "clean audit must use verdict_class null" in mod.validate_artifact(wrong_clean_class)

    protected_changed = deepcopy(artifact)
    protected_changed["protected_files_unchanged"]["all_protected_files_unchanged"] = False
    protected_changed["aggregate_row_recomputation"] = mod.aggregate_row_recomputation(
        protected_changed
    )
    protected_changed["continuous_self_learning_audited_ready_score"] = 0.0
    protected_changed["status"] = "partial_continuous_self_learning_independent_audit"
    protected_changed["honest_verdict"] = "partial: protected file check failed"
    protected_changed["verdict_class"] = "partial"
    protected_changed["reproducibility_checksum"] = mod.reproducibility_checksum(protected_changed)
    assert "protected files changed" in mod.validate_artifact(protected_changed)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    duplicate = deepcopy(artifact)
    duplicate["per_unit_rows"].append(deepcopy(duplicate["per_unit_rows"][0]))
    duplicate["aggregate_row_recomputation"] = mod.aggregate_row_recomputation(duplicate)
    duplicate["continuous_self_learning_audited_ready_score"] = 0.0
    duplicate["verdict_class"] = "disqualified"
    duplicate["status"] = "disqualified_continuous_self_learning_audit"
    duplicate["honest_verdict"] = "disqualified: duplicate row attack passed"
    duplicate["reproducibility_checksum"] = mod.reproducibility_checksum(duplicate)
    assert "duplicate audit rows detected" in mod.validate_artifact(duplicate)

    missing_repo_artifact = mod.build_artifact(
        repo_root=tmp_path,
        input_path=tmp_path / "absent.json",
        result_path=tmp_path / "missing-repo.json",
        write=False,
        duration_s=0.1,
        tests_run=TESTS_RUN,
        hash_model_files=False,
    )
    assert (
        "exp6553_artifact" in missing_repo_artifact["missing_input_disposition"]["missing_inputs"]
    )
    assert (
        "exp6552_artifact" in missing_repo_artifact["missing_input_disposition"]["missing_inputs"]
    )

    blocked_nonzero = deepcopy(missing_repo_artifact)
    blocked_nonzero["continuous_self_learning_audited_ready_score"] = 1.0
    blocked_nonzero["reproducibility_checksum"] = mod.reproducibility_checksum(blocked_nonzero)
    assert "blocked verdict requires zero ready score" in mod.validate_artifact(blocked_nonzero)

    assert mod.main(["--validate", "--result-path", str(tmp_path / "missing-cli.json")]) == 1
    assert "artifact not found" in capsys.readouterr().out
    bad_cli = tmp_path / "bad-cli.json"
    bad_cli.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_cli)]) == 1
    assert "required field set mismatch" in capsys.readouterr().out

    result_path = tmp_path / "cli.json"
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--input-path",
                str(clean_upstream_path),
                "--result-path",
                str(result_path),
            ]
        )
        == 0
    )
    assert "wrote" in capsys.readouterr().out
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    assert "validated" in capsys.readouterr().out

    real_validate = mod.validate_artifact
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["forced"])
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--input-path",
                str(clean_upstream_path),
                "--result-path",
                str(tmp_path / "forced.json"),
            ]
        )
        == 1
    )
    assert "forced" in capsys.readouterr().out
    monkeypatch.setattr(mod, "validate_artifact", real_validate)

    original_build = mod.build_artifact
    try:
        monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: artifact)
        monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["post-build"])
        assert (
            mod.main(
                [
                    "--date",
                    "20260823",
                    "--input-path",
                    str(clean_upstream_path),
                    "--result-path",
                    str(tmp_path / "post-build.json"),
                ]
            )
            == 1
        )
        assert "post-build" in capsys.readouterr().out
    finally:
        monkeypatch.setattr(mod, "build_artifact", original_build)
        monkeypatch.setattr(mod, "validate_artifact", real_validate)
