"""Tests for Exp1550 SATQuest SOTA re-evaluation.

Spec: REQ-BENCH-1550, SCENARIO-BENCH-1550.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import satquest_cnf_verifier_benchmark as exp1536
from carnot.eval import satquest_sota_reeval_zero_false_accepts as exp1550


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _ready_repair_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    artifact_path = tmp_path / "experiment_1549.json"
    manifest_path = tmp_path / "repair.jsonl"
    _write_json(
        artifact_path,
        {
            "status": "complete",
            "satquest_oracle_repair_ready": True,
            "satquest_zero_false_accepts": True,
            "solver_oracle_false_accepts_after": 0,
        },
    )
    _write_jsonl(
        manifest_path,
        [
            {
                "case_id": "ok-sat",
                "repaired_false_accept": False,
                "oracle_evidence": {"label": "SAT", "assignment_witness_checked": True},
            },
            {
                "case_id": "ok-unsat",
                "repaired_false_accept": False,
                "oracle_evidence": {"label": "UNSAT", "unsat_certificate_checked": True},
            },
        ],
    )
    return artifact_path, manifest_path


def _fake_collect(spec: dict[str, Any], prompt_cases: list[exp1536.PromptCase]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(prompt_cases):
        if index % 3 == 0:
            output_text = exp1536.gold_answer_for_prompt_case(case)
        elif case.oracle_label == "SAT":
            gold = json.loads(exp1536.gold_answer_for_prompt_case(case))
            output_text = json.dumps(
                {
                    "answer": "UNSAT",
                    "assignment": None,
                    "verifier": {"accept": True},
                    "candidate_answers": [gold],
                    "repair_hint_answer": gold,
                }
            )
        else:
            full_assignment = {f"x{i}": True for i in range(1, case.instance.n_vars + 1)}
            output_text = json.dumps(
                {
                    "answer": "SAT",
                    "assignment": full_assignment,
                    "verifier": {"accept": True},
                    "candidate_answers": [{"answer": "UNSAT"}],
                    "repair_hint_answer": {"answer": "UNSAT"},
                }
            )
        rows.append(
            {
                "case_id": case.case_id,
                "instance_id": case.instance.instance_id,
                "format_name": case.format_name,
                "model_hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "generation_source": "live_sota_llamacpp",
                "output_text": output_text,
                "elapsed_seconds": 0.01,
                "blocker": None,
            }
        )
    return {
        "summary": {
            "hf_id": spec["hf_id"],
            "model_name": spec.get("name"),
            "model_used": True,
            "blocker": None,
            "cases_returned": len(rows),
        },
        "rows": rows,
    }


def test_req_bench_1550_loads_only_zero_false_accept_repair_gate(tmp_path: Path) -> None:
    """REQ-BENCH-1550: Exp1550 refuses SATQuest runs without repaired oracle proof."""

    artifact_path, manifest_path = _ready_repair_artifacts(tmp_path)
    gate = exp1550.load_repaired_satquest_gate(artifact_path, manifest_path)

    assert gate.ready is True
    assert gate.satquest_zero_false_accepts is True
    assert gate.rows_checked == 2
    assert gate.repaired_false_accepts == 0

    _write_json(
        artifact_path,
        {
            "status": "blocked",
            "satquest_oracle_repair_ready": False,
            "satquest_zero_false_accepts": False,
            "solver_oracle_false_accepts_after": 1,
        },
    )
    with pytest.raises(exp1550.RepairedGateError, match="satquest_zero_false_accepts"):
        exp1550.load_repaired_satquest_gate(artifact_path, manifest_path)

    _write_json(
        artifact_path,
        {
            "status": "blocked",
            "satquest_oracle_repair_ready": False,
            "satquest_zero_false_accepts": True,
            "solver_oracle_false_accepts_after": 0,
        },
    )
    with pytest.raises(exp1550.RepairedGateError, match="satquest_oracle_repair_ready"):
        exp1550.load_repaired_satquest_gate(artifact_path, manifest_path)

    _write_json(
        artifact_path,
        {
            "status": "blocked",
            "satquest_oracle_repair_ready": True,
            "satquest_zero_false_accepts": True,
            "solver_oracle_false_accepts_after": 1,
        },
    )
    with pytest.raises(exp1550.RepairedGateError, match="solver_oracle_false_accepts_after"):
        exp1550.load_repaired_satquest_gate(artifact_path, manifest_path)

    _write_json(
        artifact_path,
        {
            "status": "complete",
            "satquest_oracle_repair_ready": True,
            "satquest_zero_false_accepts": True,
            "solver_oracle_false_accepts_after": 0,
        },
    )
    _write_jsonl(manifest_path, [{"case_id": "bad", "repaired_false_accept": True}])
    with pytest.raises(exp1550.RepairedGateError, match="repaired manifest"):
        exp1550.load_repaired_satquest_gate(artifact_path, manifest_path)


def test_req_bench_1550_builds_thirty_cross_format_cases_with_evidence() -> None:
    """REQ-BENCH-1550: bounded machine, symbolic, and narrative CNFs have proofs."""

    cases = exp1550.build_reeval_prompt_cases(min_cases=30)

    assert len(cases) >= 30
    assert {case.format_name for case in cases} == {"machine", "symbolic", "narrative"}
    assert {case.oracle_label for case in cases} == {"SAT", "UNSAT"}
    for case in cases:
        evidence = exp1550.oracle_evidence_for_case(case)
        assert evidence.label == case.oracle_label
        assert evidence.assignment_witness_checked or evidence.unsat_certificate_checked

    with pytest.raises(ValueError, match="at least 31"):
        exp1550.build_reeval_prompt_cases(min_cases=31)


def test_req_bench_1550_resolver_and_auc_edge_paths_are_deterministic() -> None:
    """REQ-BENCH-1550: resolver failures and degenerate AUC remain explicit."""

    def cached_pair_raises(**_kwargs: Any) -> list[dict[str, Any]] | None:
        raise RuntimeError("cache boom")

    specs, cached_details, blockers = exp1550.resolve_mandated_model_specs(
        cached_pair_fn=cached_pair_raises,
        resolver_fn=lambda hf_id: "/tmp/gemma31.gguf" if hf_id == exp1550.MODEL_SPECS[1] else None,
    )

    assert cached_details == []
    assert any(spec.get("model_path") == "/tmp/gemma31.gguf" for spec in specs)
    assert blockers == ["cached_sota_pair_error:RuntimeError: cache boom"]

    def resolver_raises(_hf_id: str) -> str | None:
        raise RuntimeError("resolver boom")

    _specs, _cached_details, failing_blockers = exp1550.resolve_mandated_model_specs(
        cached_pair_fn=lambda gpu_indices=(0, 1): None,
        resolver_fn=resolver_raises,
    )

    assert "cached_sota_pair_not_available" in failing_blockers
    assert any(blocker.startswith("model_resolver_error:") for blocker in failing_blockers)
    assert exp1550.aggregate_repaired_rows([])["energy_ranking_auc"] is None
    assert exp1550._binary_auc([True, True], [1.0, 0.5]) is None
    assert exp1550._binary_auc([True, False], [0.5, 0.5]) == pytest.approx(0.5)


def test_scenario_bench_1550_runner_writes_solver_grounded_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-1550: live rows are accepted only by repaired solver evidence."""

    repair_artifact, repair_manifest = _ready_repair_artifacts(tmp_path)
    output_path = tmp_path / "experiment_1550.json"
    manifest_path = tmp_path / "reeval.jsonl"
    cached_spec = {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": exp1550.MODEL_SPECS[0],
        "gpu": 0,
        "model_path": "/tmp/qwen.gguf",
    }

    artifact = exp1550.run_reeval(
        output_path=output_path,
        manifest_path=manifest_path,
        repaired_artifact_path=repair_artifact,
        repaired_manifest_path=repair_manifest,
        collect_model_outputs_fn=_fake_collect,
        cached_pair_fn=lambda gpu_indices=(0, 1): [cached_spec],
        resolver_fn=lambda _hf_id: None,
        gpu_probe_fn=lambda: {"gpu_count": 1},
        focused_tests_passed=True,
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact == persisted
    assert set(exp1550.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["satquest_sota_reeval_ready"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["models_attempted"][0]["hf_id"] == exp1550.MODEL_SPECS[0]
    assert artifact["cases_attempted"] == len(rows) >= 30
    assert artifact["formats_attempted"] == ["machine", "narrative", "symbolic"]
    assert artifact["solver_oracle_false_accepts"] == 0
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert 0.0 < artifact["answer_accuracy"] < 1.0
    assert artifact["witness_validity_rate"] == pytest.approx(1.0)
    assert artifact["energy_ranking_auc"] is not None
    assert "json_object_schema_prompt" in artifact["automata_or_format_constraints_used"]
    assert artifact["model_availability_blockers"] == []
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert all(row["repaired_false_accept"] is False for row in rows)
    assert all(row["oracle_evidence_valid"] is True for row in rows)


def test_req_bench_1550_blocks_without_available_mandated_model(tmp_path: Path) -> None:
    """REQ-BENCH-1550: legacy small GGUFs are not used as headline substitutes."""

    repair_artifact, repair_manifest = _ready_repair_artifacts(tmp_path)

    def fail_collect(_spec: dict[str, Any], _cases: list[exp1536.PromptCase]) -> dict[str, Any]:
        raise AssertionError("collector should not run without a mandated model path")

    artifact = exp1550.run_reeval(
        output_path=tmp_path / "blocked.json",
        manifest_path=tmp_path / "blocked.jsonl",
        repaired_artifact_path=repair_artifact,
        repaired_manifest_path=repair_manifest,
        collect_model_outputs_fn=fail_collect,
        cached_pair_fn=lambda gpu_indices=(0, 1): None,
        resolver_fn=lambda _hf_id: None,
        gpu_probe_fn=lambda: {"gpu_count": 0},
    )

    assert artifact["status"] == "blocked"
    assert artifact["satquest_sota_reeval_ready"] is False
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["cases_attempted"] == 0
    assert "cached_sota_pair_not_available" in artifact["model_availability_blockers"]
    assert artifact["honest_verdict"].startswith("complete_blocked:")


def test_req_bench_1550_blocks_before_model_probe_when_repair_gate_is_not_ready(
    tmp_path: Path,
) -> None:
    """REQ-BENCH-1550: model quality is not measured until Exp1549 is clean."""

    repair_artifact = tmp_path / "experiment_1549.json"
    repair_manifest = tmp_path / "repair.jsonl"
    _write_json(
        repair_artifact,
        {
            "status": "blocked",
            "satquest_oracle_repair_ready": False,
            "satquest_zero_false_accepts": False,
            "solver_oracle_false_accepts_after": 1,
        },
    )
    _write_jsonl(repair_manifest, [])

    artifact = exp1550.run_reeval(
        output_path=tmp_path / "gate_blocked.json",
        manifest_path=tmp_path / "gate_blocked.jsonl",
        repaired_artifact_path=repair_artifact,
        repaired_manifest_path=repair_manifest,
        collect_model_outputs_fn=lambda _spec, _cases: (_ for _ in ()).throw(
            AssertionError("collector should not run")
        ),
        cached_pair_fn=lambda gpu_indices=(0, 1): [
            {"name": "Qwen", "hf_id": exp1550.MODEL_SPECS[0], "gpu": 0, "model_path": "/tmp/qwen.gguf"}
        ],
        resolver_fn=lambda _hf_id: "/tmp/qwen.gguf",
        gpu_probe_fn=lambda: {"gpu_count": 1},
    )

    assert artifact["status"] == "blocked"
    assert artifact["solver_oracle_false_accepts"] == 0
    assert artifact["cases_attempted"] == 0
    assert "repaired_satquest_gate_not_ready" in artifact["model_availability_blockers"][0]
