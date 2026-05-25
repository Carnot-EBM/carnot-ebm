"""Tests for Exp 3085 I-CALM task-abstention SOTA panel.

Spec refs: REQ-VERIFY-3085,
           SCENARIO-VERIFY-3085,
           SCENARIO-VERIFY-3085-BLOCKED.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import icalm_task_abstention_sota_panel_v2 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / exp.SCRIPT_FILENAME


class FakeClock:
    def __init__(self) -> None:
        self.value = 50.0

    def __call__(self) -> float:
        self.value += 1.25
        return self.value


class FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == exp.DEFAULT_SEED
        assert kwargs["logprobs"] == exp.DEFAULT_LOGPROBS
        case_hash = _field(prompt, "Case Hash")
        policy = _field(prompt, "Policy")
        if policy == "baseline":
            text = {
                "case0001": "SAT | confidence=0.94",
                "case0002": "UNSAT | confidence=0.93",
                "case0003": "VALID | confidence=0.95",
                "case0004": "INVALID | confidence=0.91",
                "case0005": "VALID | confidence=0.88",
                "case0006": "REPAIRABLE | confidence=0.44",
            }[case_hash]
        else:
            text = {
                "case0001": "ACCEPT | answer=SAT | confidence=0.94",
                "case0002": "REJECT | answer=UNSAT | confidence=0.93",
                "case0003": "ACCEPT | answer=VALID | confidence=0.95",
                "case0004": "REJECT | answer=INVALID | confidence=0.91",
                "case0005": "ABSTAIN | answer=UNKNOWN | confidence=0.20",
                "case0006": "ABSTAIN | answer=REPAIRABLE | confidence=0.44",
            }[case_hash]
        return _completion(text)

    def close(self) -> None:
        self.closed = True


class NoConfidenceLlama(FakeLlama):
    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        result = super().__call__(prompt, **kwargs)
        result["choices"][0].pop("logprobs")
        return result


class RaisingLlama:
    def __init__(self, **_kwargs: Any) -> None:
        raise RuntimeError("load failed")


def _completion(text: str) -> dict[str, Any]:
    token = text.split()[0]
    other = "ABSTAIN" if token != "ABSTAIN" else "ACCEPT"
    return {
        "choices": [
            {
                "text": text,
                "logprobs": {
                    "tokens": [token],
                    "token_logprobs": [math.log(0.97)],
                    "top_logprobs": [{token: math.log(0.97), other: math.log(0.03)}],
                },
            }
        ]
    }


def _field(prompt: str, name: str) -> str:
    prefix = f"{name}: "
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _model_path(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"small fake gguf for unit tests")
    return path


def _resolve_one_model(path: Path) -> exp.ResolveGgufFn:
    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        if hf_id == "unsloth/gemma-4-26B-A4B-it-GGUF":
            return str(path)
        return None

    return resolve


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_sources(root: Path) -> None:
    _write_json(
        root,
        exp.EXP3070_REL_PATH,
        {
            "abstention_precision": 0.5,
            "rejection_recall": 0.25,
            "abstention_coverage": 0.5,
            "first_token_panel_ready": True,
        },
    )


def _row(
    *,
    fixture_id: str,
    family: str,
    task_axis: str,
    perturbation_family: str,
    prompt_hash: str,
    prompt_payload: dict[str, Any],
    exact_label: dict[str, Any],
    label_source: str,
) -> dict[str, Any]:
    return {
        "schema": "carnot.resyn_exact_fixture.v1",
        "fixture_id": fixture_id,
        "family": family,
        "task_axis": task_axis,
        "perturbation_family": perturbation_family,
        "leakage_safe_prompt_payload": prompt_payload,
        "prompt_payload_sha256": prompt_hash,
        "exact_label": exact_label,
        "label_source": label_source,
        "authority_payload": {"hidden": "exact authority must never enter prompts"},
    }


def _fixture_rows() -> list[dict[str, Any]]:
    return [
        _row(
            fixture_id="smt-sat",
            family="smt_constraints",
            task_axis="solving",
            perturbation_family="smt_sat_solving",
            prompt_hash="case0001",
            prompt_payload={
                "task": "Classify integer constraints as SAT or UNSAT.",
                "variables": ["x", "y"],
                "constraints": ["x >= 0", "y >= 0", "x + y == 2"],
                "response_schema": {"verdict": "SAT_OR_UNSAT"},
            },
            exact_label={"kind": "smt_satisfiability", "is_satisfiable": True},
            label_source="z3_solver",
        ),
        _row(
            fixture_id="smt-unsat",
            family="smt_constraints",
            task_axis="abstaining",
            perturbation_family="smt_unsat_abstention",
            prompt_hash="case0002",
            prompt_payload={
                "task": "Classify integer constraints as SAT or UNSAT.",
                "variables": ["x"],
                "constraints": ["x >= 3", "x <= 1"],
                "response_schema": {"verdict": "SAT_OR_UNSAT"},
            },
            exact_label={"kind": "smt_satisfiability", "is_satisfiable": False},
            label_source="z3_solver",
        ),
        _row(
            fixture_id="arith-valid",
            family="arithmetic_code_assertions",
            task_axis="verifying",
            perturbation_family="arithmetic_true_verification",
            prompt_hash="case0003",
            prompt_payload={
                "task": "Classify the candidate arithmetic assertion.",
                "expression": "(2 + 3) * 4",
                "candidate_assertion": "assert ((2 + 3) * 4) == 20",
                "response_schema": {"verdict": "VALID_OR_INVALID"},
            },
            exact_label={"kind": "arithmetic_assertion", "assertion_passes": True},
            label_source="python_ast_runtime_execution",
        ),
        _row(
            fixture_id="arith-invalid",
            family="arithmetic_code_assertions",
            task_axis="verifying",
            perturbation_family="arithmetic_false_verification",
            prompt_hash="case0004",
            prompt_payload={
                "task": "Classify the candidate arithmetic assertion.",
                "expression": "(2 + 3) * 4",
                "candidate_assertion": "assert ((2 + 3) * 4) == 21",
                "response_schema": {"verdict": "VALID_OR_INVALID"},
            },
            exact_label={"kind": "arithmetic_assertion", "assertion_passes": False},
            label_source="python_ast_runtime_execution",
        ),
        _row(
            fixture_id="repair-json",
            family="repairable_invalid_candidates",
            task_axis="repairing",
            perturbation_family="json_syntax_repair",
            prompt_hash="case0005",
            prompt_payload={
                "task": "Assess whether this invalid candidate needs repair.",
                "candidate": '{"mode": "bounded" "limit": 4}',
                "required_fields": ["mode", "limit"],
            },
            exact_label={"kind": "repairability", "candidate_valid": False, "repairable": True},
            label_source="json_parser",
        ),
        _row(
            fixture_id="repair-py",
            family="repairable_invalid_candidates",
            task_axis="repairing",
            perturbation_family="python_assertion_repair",
            prompt_hash="case0006",
            prompt_payload={
                "task": "Assess whether this failed assertion needs repair.",
                "expression": "(4 * 2)",
                "candidate_assertion": "assert (4 * 2) == 9",
            },
            exact_label={"kind": "repairability", "candidate_valid": False, "repairable": True},
            label_source="python_ast_runtime_execution",
        ),
    ]


def _write_manifest(root: Path, rows: list[dict[str, Any]] | None = None) -> Path:
    manifest = root / exp.FIXTURE_MANIFEST_REL_PATH
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = rows or _fixture_rows()
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / exp.OUTPUT_REL_PATH,
        rows_path=tmp_path / exp.PANEL_ROWS_REL_PATH,
        fixture_manifest_path=tmp_path / exp.FIXTURE_MANIFEST_REL_PATH,
        sample_per_family=2,
        tests_run=("pytest focused",),
    )


def _successful_artifact(tmp_path: Path, llama_factory: Any = FakeLlama) -> dict[str, Any]:
    _write_sources(tmp_path)
    _write_manifest(tmp_path)
    return exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        llama_factory=llama_factory,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": [{"index": 0, "memory_free_mib": 24000}]},
        python_environment_func=lambda: {"executable": "python-test"},
    )


def test_req_verify_3085_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3085: OpenSpec declares the panel and terminal schema."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3085" in spec
    assert "SCENARIO-VERIFY-3085" in spec
    assert "SCENARIO-VERIFY-3085-BLOCKED" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "abstention_panel_v2_ready" in spec
    assert "blocked_sota_or_fixture_precondition_failed" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3085_balanced_sampling_avoids_label_leakage(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3085: selected prompts use leakage-safe payloads only."""
    rows = _fixture_rows()
    manifest = _write_manifest(tmp_path, rows)
    loaded = exp.load_fixture_manifest(manifest)
    sample = exp.sample_balanced_fixtures(loaded, per_family=2, seed=exp.DEFAULT_SEED)

    assert len(sample) == 6
    assert {row["family"] for row in sample} == {
        "arithmetic_code_assertions",
        "repairable_invalid_candidates",
        "smt_constraints",
    }
    assert [row["fixture_id"] for row in sample] == sorted(row["fixture_id"] for row in rows)

    for row in sample:
        prompt = exp.build_prompt(row, policy="task_abstention")
        assert row["fixture_id"] not in prompt
        assert json.dumps(row["exact_label"], sort_keys=True) not in prompt
        assert json.dumps(row["authority_payload"], sort_keys=True) not in prompt
        assert row["prompt_payload_sha256"] in prompt
        assert exp.expected_answer_and_action(row)["expected_answer"] in {
            "SAT",
            "UNSAT",
            "VALID",
            "INVALID",
            "REPAIRABLE",
            "UNREPAIRABLE",
        }


def test_scenario_verify_3085_live_panel_reports_required_metrics(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3085: fake live rows produce non-vacuous panel metrics."""
    artifact = _successful_artifact(tmp_path)
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    rows = exp.load_jsonl(tmp_path / artifact["panel_rows_path"])

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["abstention_panel_v2_ready"] is True
    assert artifact["first_token_panel_ready"] is True
    assert artifact["exact_ground_truth_count"] == 6
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["model_specs"][0]["model_path"].endswith(".gguf")
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["baseline_row_count"] == 6
    assert artifact["task_abstention_row_count"] == 6
    assert artifact["accepted_count"] == 2
    assert artifact["rejected_count"] == 2
    assert artifact["abstained_count"] == 2
    assert artifact["abstention_precision"] == pytest.approx(1.0)
    assert artifact["abstention_precision_reaches_0_7"] is True
    assert artifact["rejection_recall"] == pytest.approx(0.5)
    assert artifact["abstention_coverage"] == pytest.approx(2 / 6)
    assert artifact["overacceptance_rate"] == pytest.approx(0.0)
    assert artifact["baseline_overacceptance_rate"] > artifact["overacceptance_rate"]
    assert artifact["exp3070_comparison"]["abstention_precision_delta"] == pytest.approx(0.5)
    assert artifact["preconditions_checked"]["selected_model_load"]["ok"] is True
    assert artifact["preconditions_checked"]["fixture_manifest"]["ok"] is True
    assert artifact["inference_substrate"]["repo_commit"] == "test-commit"
    assert artifact["inference_substrate"]["confidence_support"]["first_token_available"] is True
    assert artifact["prompt_hash_count"] == len(artifact["prompt_hashes"]) == 12
    assert artifact["panel_rows_sha256"] == exp.sha256_file(tmp_path / artifact["panel_rows_path"])
    assert artifact["tests_or_checks_run"] == ["pytest focused"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert {row["policy"] for row in rows} == {"baseline", "task_abstention"}
    assert any(row["decision"] == "abstain" for row in rows)

    exp.validate_artifact(artifact)


def test_scenario_verify_3085_confidence_unavailable_keeps_abstention_ready(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3085: first-token readiness is separate from abstention readiness."""
    artifact = _successful_artifact(tmp_path, llama_factory=NoConfidenceLlama)

    assert artifact["abstention_panel_v2_ready"] is True
    assert artifact["first_token_panel_ready"] is False
    assert artifact["first_token_confidence_coverage"] == 0.0
    assert artifact["inference_substrate"]["confidence_support"]["first_token_available"] is False
    exp.validate_artifact(artifact)


def test_scenario_verify_3085_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3085-BLOCKED: missing manifest or model fails closed."""
    _write_sources(tmp_path)
    missing_manifest = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        python_environment_func=lambda: {"executable": "python-test"},
    )

    assert missing_manifest["abstention_panel_v2_ready"] is False
    assert missing_manifest["first_token_panel_ready"] is False
    assert missing_manifest["exact_ground_truth_count"] == 0
    assert missing_manifest["models_used"] == []
    assert missing_manifest["model_specs"] == []
    assert missing_manifest["prompt_hashes"] == []
    assert missing_manifest["preconditions_checked"]["fixture_manifest"]["ok"] is False
    assert missing_manifest["honest_verdict"].startswith(
        "blocked_sota_or_fixture_precondition_failed"
    )
    exp.validate_artifact(missing_manifest)

    _write_manifest(tmp_path)
    no_model = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=lambda _hf_id, _preferred_quant="Q4_K_M": None,
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        python_environment_func=lambda: {"executable": "python-test"},
    )
    assert no_model["preconditions_checked"]["gguf_cache"]["ok"] is False
    assert no_model["runtime_blocker"] == "no_mandated_gguf_resolved"
    exp.validate_artifact(no_model)


def test_req_verify_3085_load_failure_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3085: model load failures and malformed artifacts fail closed."""
    _write_sources(tmp_path)
    _write_manifest(tmp_path)
    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        llama_factory=RaisingLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        python_environment_func=lambda: {"executable": "python-test"},
    )

    assert artifact["runtime_blocker"].startswith("model_load_failed:")
    assert artifact["preconditions_checked"]["selected_model_load"]["ok"] is False
    assert artifact["honest_verdict"].startswith("blocked_sota_or_fixture_precondition_failed")

    good = _successful_artifact(tmp_path)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="legacy"):
        exp.validate_artifact(good | {"legacy_smoke_only_used": True})
    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(good | {"model_specs": []})
    with pytest.raises(ValueError, match="exact_ground_truth_count"):
        exp.validate_artifact(good | {"exact_ground_truth_count": 0})
    with pytest.raises(ValueError, match="prompt_hashes"):
        exp.validate_artifact(good | {"prompt_hashes": []})
    with pytest.raises(ValueError, match="baseline_row_count"):
        exp.validate_artifact(good | {"baseline_row_count": 0})
    with pytest.raises(ValueError, match="task_abstention_row_count"):
        exp.validate_artifact(good | {"task_abstention_row_count": 0})
    with pytest.raises(ValueError, match="first-token confidence coverage"):
        exp.validate_artifact(good | {"first_token_confidence_coverage": 0.0})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(good | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked_sota_or_fixture_precondition_failed"):
        exp.validate_artifact(good | {"abstention_panel_v2_ready": False})


def test_req_verify_3085_parsing_metrics_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3085: parser and metric helpers are deterministic and defensive."""
    rows = _fixture_rows()
    config = exp.ExperimentConfig(
        repo_root=tmp_path,
        decode_config={"max_tokens": 3},
        load_config={"n_batch": 8},
    )
    assert config.effective_decode_config()["max_tokens"] == 3
    assert config.effective_load_config(1)["n_batch"] == 8
    assert config.effective_load_config(1)["main_gpu"] == 1

    assert exp._parse_policy_response("ACCEPT | answer=SAT | confidence=0.75") == {
        "raw_action": "accept",
        "answer": "SAT",
        "verbal_confidence": 0.75,
    }
    assert exp._parse_policy_response("reject answer invalid high confidence") == {
        "raw_action": "reject",
        "answer": "INVALID",
        "verbal_confidence": 0.9,
    }
    assert exp._parse_policy_response("not sure") == {
        "raw_action": None,
        "answer": None,
        "verbal_confidence": None,
    }
    assert exp._verbal_confidence("medium confidence") == pytest.approx(0.5)
    assert exp._verbal_confidence("low confidence") == pytest.approx(0.2)
    assert exp._verbal_confidence("confidence=2.0") == pytest.approx(1.0)
    assert exp._verbal_confidence("confidence=-1.0") == pytest.approx(0.0)
    assert exp._derive_decision(None, "SAT", 0.9, 0.9, 0.7) == "accept"
    assert exp._derive_decision(None, "INVALID", 0.9, 0.9, 0.7) == "reject"
    assert exp._derive_decision("accept", "SAT", 0.2, 0.9, 0.7) == "abstain"
    assert exp._derive_decision("accept", "SAT", None, None, 0.7) == "accept"
    assert exp._answer_matches("sat.", "SAT") is True
    assert exp._answer_matches(None, "SAT") is False
    assert exp._decision_matches_expected("reject", "reject") is True
    assert exp._decision_matches_expected("accept", "reject") is False
    assert exp._baseline_decision(None) == "abstain"
    assert exp._sha256_text("x") == exp._sha256_text("x")
    assert exp._relative_path(tmp_path, Path("/outside/rows.jsonl")) == "/outside/rows.jsonl"
    assert exp._model_family("unsloth/Qwen3.6-35B-A3B-GGUF") == "qwen"
    assert exp._model_family("vendor/Other-GGUF") == "vendor"
    assert exp._float("bad") == 0.0
    assert exp._selected_model({}) is None
    assert exp._empty_metrics()["exact_ground_truth_count"] == 0

    manifest = _write_manifest(tmp_path, rows)
    loaded = exp.load_fixture_manifest(manifest)
    assert exp.sample_balanced_fixtures(loaded, per_family=99, seed=1) == sorted(
        loaded, key=lambda row: row["fixture_id"]
    )
    assert exp._fixture_manifest_status(manifest)["row_count"] == 6
    missing = tmp_path / "missing.jsonl"
    assert exp._fixture_manifest_status(missing)["ok"] is False
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{bad-json\n", encoding="utf-8")
    assert exp._fixture_manifest_status(bad)["ok"] is False
    assert exp._exp3070_comparison(tmp_path, 0.8)["abstention_precision_delta"] == pytest.approx(
        0.3
    )
    _write_json(tmp_path, exp.EXP3070_REL_PATH, {"abstention_precision": "bad"})
    assert exp._exp3070_comparison(tmp_path, 0.8)["exp3070_abstention_precision"] == 0.0
    (tmp_path / exp.EXP3070_REL_PATH).write_text("{bad-json", encoding="utf-8")
    assert exp._exp3070_comparison(tmp_path, 0.8)["exp3070_abstention_precision"] == 0.0

    candidate_valid = dict(rows[-1])
    candidate_valid["exact_label"] = {"kind": "repairability", "candidate_valid": True}
    assert exp.expected_answer_and_action(candidate_valid) == {
        "expected_answer": "VALID",
        "expected_action": "accept",
    }
    unrepairable = dict(rows[-1])
    unrepairable["exact_label"] = {
        "kind": "repairability",
        "candidate_valid": False,
        "repairable": False,
    }
    assert exp.expected_answer_and_action(unrepairable) == {
        "expected_answer": "UNREPAIRABLE",
        "expected_action": "reject",
    }
    unknown_family = dict(rows[-1]) | {"family": "unknown_family"}
    with pytest.raises(ValueError, match="unknown fixture family"):
        exp.expected_answer_and_action(unknown_family)
    assert exp._allowed_answers(unknown_family) == ("UNKNOWN",)
    assert (
        exp._first_precondition_failure(
            {
                "fixture_manifest": {"ok": True},
                "cuda_gpu": {"ok": False},
                "gguf_cache": {"ok": True},
            }
        )
        == "cuda_gpu_unavailable"
    )

    payload = {"z": 1, "a": 2}
    assert (
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()
        == exp._prompt_payload_hash(payload)
    )

    token_only = exp._confidence_from_output(
        {"choices": [{"text": "SAT", "logprobs": {"tokens": ["SAT"], "token_logprobs": [math.log(0.7)]}}]}
    )
    assert token_only["confidence_signal"] == "first_token_logprob_proxy"
    assert token_only["confidence_score"] == pytest.approx(0.7)
    assert exp._first_choice({"choices": []}) == {}
    skipped_space = exp._confidence_from_output(
        {
            "choices": [
                {
                    "text": "SAT",
                    "logprobs": {
                        "tokens": [" ", "SAT"],
                        "token_logprobs": [math.log(0.6), math.log(0.8)],
                    },
                }
            ]
        }
    )
    assert skipped_space["first_token"] == "SAT"
    assert exp._confidence_from_output({"choices": [{"text": "SAT", "logprobs": {}}]})[
        "confidence_available"
    ] is False
    assert exp._first_content_index([], []) == 0
    assert exp._topk_entropy_confidence({"bad": "not-a-number"})["confidence_available"] is False
    assert exp._topk_entropy_confidence({})["confidence_available"] is False
    assert exp._float_list(None) == []
    assert exp._float_list([1, "bad", 2.5]) == [1.0, 2.5]

    large = tmp_path / "large.gguf"
    large.write_bytes(b"a" * 2048)
    evidence = exp._file_evidence(str(large), full_limit_bytes=1)
    assert evidence["method"] == "bounded_head_tail_sha256"
    assert evidence["full_sha256_feasible"] is False
