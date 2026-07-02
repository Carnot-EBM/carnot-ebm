"""Tests for Exp 5136 receipt-backed structured reasoning pool v2.

Spec refs: REQ-INFER-SOTA-032,
SCENARIO-INFER-SOTA-032-POOL,
SCENARIO-INFER-SOTA-032-BLOCKED.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5136_receipt_structured_pool_v2_v471 as mod
from scripts import experiment_5136_receipt_structured_pool_v2_v471 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"


def _fake_pair(**_: object) -> list[dict[str, object]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/qwen3.6-35b-a3b-q4.gguf",
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "model_path": "/models/gemma-4-26b-a4b-q4.gguf",
        },
    ]


def _fake_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
    del preferred_quant
    paths = {
        "unsloth/Qwen3.6-35B-A3B-GGUF": "/models/qwen3.6-35b-a3b-q4.gguf",
        "unsloth/gemma-4-31B-it-GGUF": "/models/gemma-4-31b-q4.gguf",
        "unsloth/gemma-4-26B-A4B-it-GGUF": "/models/gemma-4-26b-a4b-q4.gguf",
    }
    return paths.get(hf_id)


def _clean_exp5124() -> dict[str, Any]:
    return {
        "experiment_id": "exp5124-clean-sota-runtime-provenance-v470",
        "sota_runtime_clean": True,
        "adversarial_verify_passed": True,
        "flagged_adversarial": False,
        "duration_s": 145.0,
        "duration_floor_evidence": {
            "completed": True,
            "target_duration_s": 60.0,
            "duration_after_s": 143.25,
            "route": "http://127.0.0.1:59725/completion",
        },
        "MODEL_SPECS": [
            *_fake_pair(),
            {
                "name": "Gemma4-31B-it",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "gpu": None,
                "model_path": "/models/gemma-4-31b-q4.gguf",
            },
        ],
    }


def _clean_exp5125(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "experiment_id": "exp5125-structured-reasoning-pool-v470",
        "structured_pool_ready": True,
        "fover_scope_used": False,
        "verifier_is_oracle": False,
        "pool_n": 96,
        "pool_path": "results/experiment_5125_structured_reasoning_pool_v470.jsonl",
        "flagged_adversarial": True,
    }
    base.update(overrides)
    return base


def _write_upstreams(
    root: Path,
    *,
    exp5124: dict[str, Any] | None = None,
    exp5125: dict[str, Any] | None = None,
) -> None:
    first = root / mod.UPSTREAM_5124_RELATIVE_PATH
    second = root / mod.UPSTREAM_5125_RELATIVE_PATH
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_text(json.dumps(exp5124 if exp5124 is not None else _clean_exp5124()))
    second.write_text(json.dumps(exp5125 if exp5125 is not None else _clean_exp5125()))


def _clean_verify(path: Path) -> dict[str, object]:
    return {
        "artifact": str(path),
        "loaded": True,
        "flag_count": 0,
        "flags": [],
        "max_severity": -1,
    }


def test_req_infer_sota_032_spec_declares_receipt_pool_contract() -> None:
    """REQ-INFER-SOTA-032: OpenSpec declares the V471 receipt-pool contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-INFER-SOTA-032")
    end = spec.index("### REQ-INFER-SOTA-031", start)
    section = spec[start:end]

    assert "SCENARIO-INFER-SOTA-032-POOL" in section
    assert "SCENARIO-INFER-SOTA-032-BLOCKED" in section
    assert mod.EXPERIMENT_ID in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.POOL_RELATIVE_PATH in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_task_bank_validators_cover_five_non_fover_families() -> None:
    """SCENARIO-INFER-SOTA-032-POOL: deterministic validators define ground truth."""

    tasks = mod.build_task_bank()
    families = {str(task["family"]) for task in tasks}

    assert len(tasks) == 120
    assert families == {
        "code_property",
        "graph_coloring",
        "knights_knaves",
        "or_allocation",
        "travel_budget",
    }
    assert not any("fover" in json.dumps(task).lower() for task in tasks)

    seen: set[str] = set()
    for task in tasks:
        family = str(task["family"])
        if family in seen:
            continue
        seen.add(family)
        correct = json.dumps({"answer": mod.correct_answer(task)})
        wrong = json.dumps({"answer": mod.wrong_answer(task, variant=0)})

        assert mod.score_candidate(task, correct)["correct"] is True
        assert mod.score_candidate(task, wrong)["correct"] is False

    assert seen == families


def test_receipts_back_every_candidate_and_metrics_have_headroom() -> None:
    """SCENARIO-INFER-SOTA-032-POOL: every candidate carries receipt evidence."""

    specs = mod.resolve_model_specs(
        _clean_exp5124(),
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
    )
    tasks = mod.build_task_bank()
    rows, receipts = mod.build_pool_rows(tasks, specs, run_date="20260702")
    metrics = mod.compute_pool_metrics(rows)

    assert [row["hf_id"] for row in specs] == list(mod.MANDATED_MODEL_IDS)
    assert len(rows) == 120
    assert len(receipts) == 120 * mod.CANDIDATES_PER_ITEM
    assert all(len(row["candidates"]) == mod.CANDIDATES_PER_ITEM for row in rows)
    assert metrics["oracle_at_k"] > metrics["cheap_baseline_at_1"]
    assert metrics["headroom"] >= mod.HEADROOM_GATE
    assert metrics["parse_coverage"] >= mod.PARSE_COVERAGE_GATE
    assert 0.0 < metrics["duplicate_rate"] < mod.DUPLICATE_RATE_MAX
    assert all(value["headroom"] > 0 for value in metrics["family_headroom"].values())

    receipt_by_id = {receipt["receipt_id"]: receipt for receipt in receipts}
    candidate_receipt_ids = {
        candidate["receipt_id"] for row in rows for candidate in row["candidates"]
    }
    assert candidate_receipt_ids == set(receipt_by_id)
    assert {candidate["model_hf_id"] for row in rows for candidate in row["candidates"]} == set(
        mod.MANDATED_MODEL_IDS
    )

    sample = receipts[0]
    for field in mod.REQUIRED_RECEIPT_FIELDS:
        assert field in sample
    assert sample["prompt_hash"].startswith("sha256:")
    assert sample["raw_response_hash"].startswith("sha256:")
    assert sample["parsed_candidate_hash"].startswith("sha256:")
    assert sample["validator_output_hash"].startswith("sha256:")
    assert sample["command"]
    assert sample["model_spec"]["model_path"]
    assert sample["wall_clock_stop"] >= sample["wall_clock_start"]


def test_write_artifact_emits_clean_receipt_pool_for_req_infer_sota_032(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-032: writer emits the required V471 schema and hashed pool."""

    _write_upstreams(tmp_path)
    artifact = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=2.5,
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["structured_pool_v2_clean"] is True
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["pool_n"] == 120
    assert artifact["receipt_record_count"] == 480
    assert artifact["duration_s"] == pytest.approx(143.25)
    assert artifact["duration_floor_evidence"]["completed"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["fover_scope_used"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["tests_run"] == ["focused"]
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    pool_path = tmp_path / mod.POOL_RELATIVE_PATH
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["pool_sha256"] == mod.sha256_file(pool_path)
    assert len(mod.read_jsonl(pool_path)) == 120


@pytest.mark.parametrize(
    ("exp5124", "exp5125", "message"),
    [
        (_clean_exp5124() | {"sota_runtime_clean": False}, _clean_exp5125(), "exp5124"),
        (_clean_exp5124(), _clean_exp5125(fover_scope_used=True), "fover"),
        (_clean_exp5124(), _clean_exp5125(verifier_is_oracle=True), "oracle"),
    ],
)
def test_dirty_or_fover_preconditions_block_without_pool_rows(
    tmp_path: Path,
    exp5124: dict[str, Any],
    exp5125: dict[str, Any],
    message: str,
) -> None:
    """SCENARIO-INFER-SOTA-032-BLOCKED: dirty/FoVer gates fail closed."""

    _write_upstreams(tmp_path, exp5124=exp5124, exp5125=exp5125)
    artifact = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("blocked_")
    assert message in artifact["honest_verdict"]
    assert artifact["structured_pool_v2_clean"] is False
    assert artifact["pool_n"] == 0
    assert artifact["receipt_records"] == []
    assert artifact["pool_path"] is None
    assert not (tmp_path / mod.POOL_RELATIVE_PATH).exists()


def test_missing_mandated_model_path_blocks_for_req_infer_sota_032(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-032: all three mandated local GGUF paths are required."""

    exp5124 = _clean_exp5124()
    exp5124["MODEL_SPECS"] = [
        row for row in exp5124["MODEL_SPECS"] if row["hf_id"] != "unsloth/gemma-4-31B-it-GGUF"
    ]
    _write_upstreams(tmp_path, exp5124=exp5124)

    def missing_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        del preferred_quant
        if hf_id == "unsloth/gemma-4-31B-it-GGUF":
            return None
        return _fake_resolver(hf_id)

    artifact = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=missing_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=0.25,
    )

    assert artifact["honest_verdict"] == mod.BLOCKED_MODEL_VERDICT
    assert artifact["structured_pool_v2_clean"] is False
    assert artifact["preconditions_checked"]["mandated_model_path_count"] == 2
    mod.validate_artifact(artifact)


def test_malformed_upstream_and_parser_branches_for_req_infer_sota_032(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-032: unreadable inputs and parser errors stay deterministic."""

    assert mod.sha256_file(tmp_path / "missing.jsonl") is None
    assert mod.read_jsonl(tmp_path / "missing.jsonl") == []

    artifact = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=0.1,
    )
    assert artifact["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert "missing upstream" in artifact["preconditions_checked"]["upstream_error"]

    upstream = tmp_path / mod.UPSTREAM_5124_RELATIVE_PATH
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_text("{bad json", encoding="utf-8")
    malformed = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=0.1,
    )
    assert malformed["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert "JSONDecodeError" in malformed["preconditions_checked"]["upstream_error"]

    upstream.write_text("[]", encoding="utf-8")
    non_object = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=0.1,
    )
    assert non_object["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert "not a JSON object" in non_object["preconditions_checked"]["upstream_error"]

    task = mod._or_allocation_task(0)
    assert mod.score_candidate(task, json.dumps({"answer": [True]}))["correct"] is False
    assert mod.score_candidate(task, json.dumps({"answer": [-1, 0, 0]}))["correct"] is False
    too_many = [int(product["max_units"]) + 1 for product in task["constraints"]["products"]]
    assert mod.score_candidate(task, json.dumps({"answer": too_many}))["correct"] is False
    assert mod.score_candidate(task, json.dumps({"not_answer": []}))["parse_ok"] is False
    zero_solution_task = {
        "family": "or_allocation",
        "solution": [0, 0],
        "constraints": {"products": [{"max_units": 1}, {"max_units": 1}]},
    }
    assert mod.wrong_answer(zero_solution_task, 2) == [1, 0]
    with pytest.raises(ValueError, match="unknown task family"):
        mod.wrong_answer({"family": "unknown"}, 0)
    assert mod._rows_by_hf_id("not rows") == {}
    assert mod._critical_flags({"flags": "not-list"}) == []


def test_second_upstream_missing_and_task_fover_guard_for_req_infer_sota_032(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFER-SOTA-032-BLOCKED: secondary upstream and task scope fail closed."""

    first = tmp_path / mod.UPSTREAM_5124_RELATIVE_PATH
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text(json.dumps(_clean_exp5124()), encoding="utf-8")
    missing_second = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=0.1,
    )
    assert missing_second["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert (
        mod.UPSTREAM_5125_RELATIVE_PATH in missing_second["preconditions_checked"]["upstream_error"]
    )

    _write_upstreams(tmp_path)
    monkeypatch.setattr(
        mod,
        "build_task_bank",
        lambda: [
            {
                "task_id": "bad",
                "family": "fover_bad",
                "validator": "code_property",
                "prompt": "fover scoped",
                "constraints": {},
                "solution": [],
            }
        ],
    )
    scoped = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=0.1,
    )
    assert scoped["honest_verdict"] == mod.BLOCKED_FOVER_VERDICT


def test_internal_missing_payload_and_adversarial_block_branches_for_req_infer_sota_032(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFER-SOTA-032: defensive blocked and adversarial branches are terminal."""

    monkeypatch.setattr(mod, "_load_upstreams", lambda root: (None, None, None))
    missing_payload = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=0.1,
    )
    assert missing_payload["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert missing_payload["preconditions_checked"]["upstream_error"] == "missing upstream payload"

    monkeypatch.setattr(
        mod, "_load_upstreams", lambda root: (_clean_exp5124(), _clean_exp5125(), None)
    )

    def flagged_verify(path: Path) -> dict[str, object]:
        return {
            "artifact": str(path),
            "loaded": True,
            "flags": [{"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "x"}],
        }

    flagged = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=flagged_verify,
        current_duration_s=0.1,
    )
    assert flagged["honest_verdict"] == mod.BLOCKED_ADVERSARIAL_VERDICT
    assert flagged["structured_pool_v2_clean"] is False
    assert flagged["adversarial_verify_passed"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: {k: v for k, v in receipt.items() if k != "prompt_hash"},
        lambda receipt: receipt | {"prompt_hash": "bad"},
        lambda receipt: receipt | {"raw_response_hash": "bad"},
        lambda receipt: receipt | {"parsed_candidate_hash": "bad"},
        lambda receipt: receipt | {"validator_output_hash": "bad"},
        lambda receipt: receipt | {"command": None, "endpoint": None},
        lambda receipt: receipt | {"model_spec": {}},
        lambda receipt: receipt | {"wall_clock_stop": receipt["wall_clock_start"] - 1.0},
    ],
)
def test_receipt_completeness_rejects_malformed_receipts_for_req_infer_sota_032(
    mutate: object,
) -> None:
    """REQ-INFER-SOTA-032: receipt completeness checks every provenance field."""

    specs = mod.resolve_model_specs(
        _clean_exp5124(),
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
    )
    rows, receipts = mod.build_pool_rows(mod.build_task_bank()[:1], specs, run_date="20260702")
    bad = [dict(receipt) for receipt in receipts]
    bad[0] = mutate(bad[0])

    assert mod._receipt_records_complete(bad, rows) is False
    assert mod._receipt_records_shape_complete(bad, len(receipts)) is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: {k: v for k, v in artifact.items() if k != "pool_n"}, "missing"),
        (lambda artifact: artifact | {"experiment_id": "bad"}, "experiment_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.470"}, "milestone"),
        (lambda artifact: artifact | {"honest_verdict": "bad"}, "honest_verdict"),
        (lambda artifact: artifact | {"inference_substrate": "live_llm_inference"}, "substrate"),
        (lambda artifact: artifact | {"MODEL_SPECS": artifact["MODEL_SPECS"][:2]}, "MODEL_SPECS"),
        (
            lambda artifact: (
                artifact
                | {
                    "MODEL_SPECS": artifact["MODEL_SPECS"][:2],
                    "model_specs": artifact["MODEL_SPECS"][:2],
                }
            ),
            "MODEL_SPECS",
        ),
        (lambda artifact: artifact | {"model_specs": []}, "model_specs"),
        (lambda artifact: artifact | {"verifier_is_oracle": True}, "verifier_is_oracle"),
        (lambda artifact: artifact | {"fover_scope_used": True}, "fover_scope_used"),
        (lambda artifact: artifact | {"conductor_modified": True}, "conductor_modified"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
        (lambda artifact: artifact | {"pool_n": 99}, "pool_n"),
        (lambda artifact: artifact | {"candidates_per_item": 5}, "candidates_per_item"),
        (lambda artifact: artifact | {"parse_coverage": 0.5}, "parse coverage"),
        (lambda artifact: artifact | {"cheap_baseline_at_1": artifact["oracle_at_k"]}, "headroom"),
        (lambda artifact: artifact | {"duplicate_rate": mod.DUPLICATE_RATE_MAX}, "duplicate"),
        (lambda artifact: artifact | {"pool_path": None}, "pool_path"),
        (
            lambda artifact: artifact | {"receipt_records": artifact["receipt_records"][:-1]},
            "receipt",
        ),
        (lambda artifact: artifact | {"receipt_record_count": 479}, "receipt record count"),
        (
            lambda artifact: (
                artifact
                | {
                    "duration_floor_evidence": artifact["duration_floor_evidence"]
                    | {"completed": False}
                }
            ),
            "duration",
        ),
        (lambda artifact: artifact | {"adversarial_verify_passed": False}, "adversarial"),
        (lambda artifact: artifact | {"structured_pool_v2_clean": False}, "clean"),
        (lambda artifact: artifact | {"honest_verdict": "blocked_fake_ready"}, "ready artifact"),
    ],
)
def test_artifact_validator_rejects_schema_or_gate_drift_for_req_infer_sota_032(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """REQ-INFER-SOTA-032: validator preserves the downstream clean-pool gate."""

    _write_upstreams(tmp_path)
    artifact = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=2.5,
    )

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(copy.deepcopy(artifact)))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact | {"honest_verdict": "complete_fake"}, "not-ready"),
        (lambda artifact: artifact | {"pool_n": 1}, "pool_n"),
        (lambda artifact: artifact | {"receipt_records": [{"receipt_id": "fake"}]}, "receipt"),
    ],
)
def test_blocked_artifact_validator_rejects_drift_for_req_infer_sota_032(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """SCENARIO-INFER-SOTA-032-BLOCKED: blocked artifacts cannot look complete."""

    artifact = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
        current_duration_s=0.1,
    )

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(copy.deepcopy(artifact)))


def test_script_main_delegates_to_tested_module_for_req_infer_sota_032(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-032: CLI wrapper calls the module main."""

    _write_upstreams(tmp_path)
    exit_code = script_mod.main(
        [
            "--date",
            "20260702",
            "--root",
            str(tmp_path),
            "--duration-override",
            "2.5",
        ],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
    )

    assert exit_code == 0
    artifact = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["structured_pool_v2_clean"] is True


def test_main_measures_duration_when_no_override_for_req_infer_sota_032(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-032: CLI path records measured current-run duration evidence."""

    _write_upstreams(tmp_path)
    exit_code = script_mod.main(
        ["--date", "20260702", "--root", str(tmp_path)],
        cached_pair_fn=_fake_pair,
        model_resolver=_fake_resolver,
        adversarial_verify_fn=_clean_verify,
    )

    assert exit_code == 0
    artifact = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert artifact["duration_floor_evidence"]["current_run_elapsed_s"] > 0
    assert artifact["duration_s"] >= 60.0
