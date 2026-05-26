"""Tests for Exp 3099 local SOTA confidence abstention panel v3.

Spec refs: REQ-VERIFY-3099,
           SCENARIO-VERIFY-3099,
           SCENARIO-VERIFY-3099-BLOCKED.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import local_sota_confidence_abstention_panel_v3 as exp
from carnot.eval import maxsat_abstention_routing_policy_v1 as policy_mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
REQUIRED_ARTIFACT_FIELDS = {
    "abstention_panel_v3_ready",
    "model_specs",
    "exact_ground_truth_count",
    "abstention_precision",
    "rejection_recall",
    "abstention_coverage",
    "false_accept_rate",
    "false_reject_rate",
    "solve_accuracy",
    "verification_accuracy",
    "maxsat_policy_used",
    "thermodynamic_decode_telemetry",
    "prompt_hashes",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


class FakeClock:
    def __init__(self) -> None:
        self.value = 20.0

    def __call__(self) -> float:
        self.value += 0.5
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
        text = {
            "hash-accept-sat": "ACCEPT | answer=SAT | confidence=0.96",
            "hash-reject-unsat": "ACCEPT | answer=SAT | confidence=0.95",
            "hash-reject-invalid": "REJECT | answer=INVALID | confidence=0.91",
            "hash-accept-valid-wrong": "ACCEPT | answer=INVALID | confidence=0.93",
        }[case_hash]
        return _completion(text)

    def close(self) -> None:
        self.closed = True


class RaisingLlama:
    def __init__(self, **_kwargs: Any) -> None:
        raise RuntimeError("load failed")


class NoVerbalConfidenceLlama(FakeLlama):
    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        result = super().__call__(prompt, **kwargs)
        text = result["choices"][0]["text"]
        result["choices"][0]["text"] = text.split("| confidence", 1)[0].strip()
        return result


class FailingGenerationLlama(FakeLlama):
    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("generation boom")


def _field(prompt: str, name: str) -> str:
    prefix = f"{name}: "
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _completion(text: str) -> dict[str, Any]:
    token = text.split()[0]
    return {
        "choices": [
            {
                "text": text,
                "logprobs": {
                    "tokens": [token],
                    "token_logprobs": [math.log(0.8)],
                    "top_logprobs": [{token: math.log(0.8), "ABSTAIN": math.log(0.2)}],
                },
            }
        ]
    }


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _model_path(root: Path) -> Path:
    path = root / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake gguf for exp 3099 tests")
    return path


def _resolve_one_model(path: Path) -> exp.ResolveGgufFn:
    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        if hf_id == "unsloth/gemma-4-26B-A4B-it-GGUF":
            return str(path)
        return None

    return resolve


def _manifest_row(
    *,
    source_fixture_id: str,
    prompt_hash: str,
    task_family: str,
    perturbation_type: str,
    expected_answer: str,
    expected_action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": source_fixture_id,
        "source_prompt_payload_sha256": prompt_hash,
        "task_family": task_family,
        "task_axis": "solving",
        "perturbation_type": perturbation_type,
        "expected_answer": expected_answer,
        "solver_label": expected_answer.lower(),
        "label_source": "unit_exact_authority",
        "exact_label_kind": "unit",
        "leakage_safe_prompt_payload": payload,
        "verifier_target": {
            "expected_action": expected_action,
            "expected_reject": expected_action == "reject",
        },
        "repair_target": {"applicable": False, "reason": "not_a_repair_fixture"},
        "evaluation_tasks": ["abstention_sota_panel_v3"],
        "stratum_key": f"{task_family}|{perturbation_type}|{expected_answer}",
    }


def _manifest_rows() -> list[dict[str, Any]]:
    return [
        _manifest_row(
            source_fixture_id="case-accept-sat",
            prompt_hash="hash-accept-sat",
            task_family="smt_constraints",
            perturbation_type="smt_sat_solving",
            expected_answer="SAT",
            expected_action="accept",
            payload={
                "task": "Classify the integer constraints as SAT or UNSAT.",
                "constraints": ["x >= 0", "x <= 3"],
                "response_schema": {"verdict": "SAT_OR_UNSAT"},
            },
        ),
        _manifest_row(
            source_fixture_id="case-reject-unsat",
            prompt_hash="hash-reject-unsat",
            task_family="smt_constraints",
            perturbation_type="smt_unsat_abstention",
            expected_answer="UNSAT",
            expected_action="reject",
            payload={
                "task": "Classify the integer constraints as SAT or UNSAT.",
                "constraints": ["x >= 3", "x <= 1"],
                "response_schema": {"verdict": "SAT_OR_UNSAT"},
            },
        ),
        _manifest_row(
            source_fixture_id="case-reject-invalid",
            prompt_hash="hash-reject-invalid",
            task_family="arithmetic_code_assertions",
            perturbation_type="arithmetic_false_verification",
            expected_answer="INVALID",
            expected_action="reject",
            payload={
                "task": "Classify the candidate arithmetic assertion.",
                "candidate_assertion": "assert 2 + 2 == 5",
                "response_schema": {"verdict": "VALID_OR_INVALID"},
            },
        ),
        _manifest_row(
            source_fixture_id="case-accept-valid-wrong",
            prompt_hash="hash-accept-valid-wrong",
            task_family="arithmetic_code_assertions",
            perturbation_type="arithmetic_true_verification",
            expected_answer="VALID",
            expected_action="accept",
            payload={
                "task": "Classify the candidate arithmetic assertion.",
                "candidate_assertion": "assert 2 + 2 == 4",
                "response_schema": {"verdict": "VALID_OR_INVALID"},
            },
        ),
    ]


def _write_sources(root: Path, rows: list[dict[str, Any]] | None = None) -> None:
    rows = rows or _manifest_rows()
    (root / "CODEX.md").write_text("Spec First\nWrite Tests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No tiny panels\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "arXiv:2604.07867 diagnostic telemetry only\n", encoding="utf-8"
    )
    _write_jsonl(root, exp.STRATIFIED_MANIFEST_REL_PATH, rows)
    _write_json(
        root,
        exp.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "minimum_live_eval_count": 4,
            "usable_fixture_count": len(rows),
            "stratified_eval_manifest_path": exp.STRATIFIED_MANIFEST_REL_PATH.as_posix(),
            "honest_verdict": "complete: eval_protocol_ready=true",
        },
    )
    policy = policy_mod.build_policy_document()
    _write_json(root, exp.POLICY_REL_PATH, policy)
    _write_json(
        root,
        exp.EXP3098_REL_PATH,
        {
            "artifact": "experiment_3098_maxsat_abstention_routing_policy_v1",
            "maxsat_policy_ready": True,
            "routing_policy_path": exp.POLICY_REL_PATH.as_posix(),
            "honest_verdict": "complete: maxsat_policy_ready=true",
        },
    )


def _config(root: Path) -> exp.PanelConfig:
    return exp.PanelConfig(
        repo_root=root,
        output_path=root / exp.OUTPUT_REL_PATH,
        rows_path=root / exp.PANEL_ROWS_REL_PATH,
        minimum_live_eval_count=4,
        started_s=10.0,
        clock=FakeClock(),
        tests_run=("pytest exp3099 focused",),
    )


def test_req_verify_3099_spec_anchor_exists() -> None:
    """REQ-VERIFY-3099: OpenSpec declares the v3 panel contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3099" in spec
    assert "SCENARIO-VERIFY-3099" in spec
    assert "SCENARIO-VERIFY-3099-BLOCKED" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "maxsat_policy_used" in spec
    assert "thermodynamic_decode_telemetry" in spec


def test_scenario_verify_3099_routes_minimum_exact_panel(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3099: fake live rows are routed through the MaxSAT policy."""
    _write_sources(tmp_path)

    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        cached_pair_func=lambda **_: None,
        llama_factory=FakeLlama,
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": [{"index": 0}]},
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    rows = exp.load_jsonl(tmp_path / artifact["panel_rows_path"])

    assert saved == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["abstention_panel_v3_ready"] is True
    assert artifact["maxsat_policy_used"] is True
    assert artifact["exact_ground_truth_count"] == 4
    assert artifact["available_exact_fixture_count"] == 4
    assert artifact["minimum_live_eval_count"] == 4
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["cached_sota_pair"]["called"] is True
    assert artifact["cached_sota_pair"]["ready"] is False
    assert len(artifact["model_specs"]) == 3
    assert [row["cache_status"] for row in artifact["model_specs"]].count("cached") == 1
    assert artifact["route_decision_counts"] == {"accept": 1, "reject": 3, "abstain": 0}
    assert artifact["solve_accuracy"] == pytest.approx(0.5)
    assert artifact["verification_accuracy"] == pytest.approx(0.75)
    assert artifact["rejection_recall"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["false_reject_rate"] == pytest.approx(0.0)
    assert artifact["abstention_precision"] == pytest.approx(0.0)
    assert artifact["abstention_coverage"] == pytest.approx(0.0)
    assert artifact["prompt_hash_count"] == len(artifact["prompt_hashes"]) == 4
    assert artifact["panel_rows_sha256"] == exp.sha256_file(tmp_path / artifact["panel_rows_path"])
    assert artifact["thermodynamic_decode_telemetry"]["available"] is True
    assert artifact["thermodynamic_decode_telemetry"]["diagnostic_only"] is True
    assert artifact["blocked_outcomes"] == []
    assert artifact["skipped_outcomes"] == []
    assert artifact["cache_missing_outcomes"] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    assert artifact["inference_substrate"]["repo_commit"] == "test-commit"
    assert artifact["honest_verdict"].startswith("complete:")

    assert len(rows) == 4
    assert all(row["maxsat_policy_used"] is True for row in rows)
    assert {row["route_decision"] for row in rows} == {"accept", "reject"}
    assert rows[0]["prompt_hash"] == artifact["prompt_hashes"][0]
    exp.validate_artifact(artifact)


def test_scenario_verify_3099_blocked_paths_are_terminal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3099-BLOCKED: missing model or policy fails closed."""
    _write_sources(tmp_path)
    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=lambda _hf_id, _preferred_quant="Q4_K_M": None,
        cached_pair_func=lambda **_: None,
        llama_factory=FakeLlama,
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )

    assert artifact["abstention_panel_v3_ready"] is False
    assert artifact["maxsat_policy_used"] is True
    assert artifact["exact_ground_truth_count"] == 0
    assert artifact["prompt_hashes"] == []
    assert artifact["models_used"] == []
    assert artifact["runtime_blocker"] == "no_mandated_gguf_resolved"
    assert set(artifact["cache_missing_outcomes"]) == set(exp.MANDATED_MODEL_IDS)
    assert artifact["honest_verdict"].startswith("blocked_sota_or_panel_precondition_failed")
    exp.validate_artifact(artifact)

    (tmp_path / exp.POLICY_REL_PATH).unlink()
    missing_policy = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        cached_pair_func=lambda **_: None,
        llama_factory=FakeLlama,
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )
    assert missing_policy["abstention_panel_v3_ready"] is False
    assert missing_policy["maxsat_policy_used"] is False
    assert missing_policy["runtime_blocker"] == "maxsat_policy_unavailable"
    exp.validate_artifact(missing_policy)

    _write_sources(tmp_path)
    load_failed = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        cached_pair_func=lambda **_: None,
        llama_factory=RaisingLlama,
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )
    assert load_failed["runtime_blocker"].startswith("model_load_failed:RuntimeError")
    exp.validate_artifact(load_failed)

    generation_failed = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        cached_pair_func=lambda **_: None,
        llama_factory=FailingGenerationLlama,
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )
    assert generation_failed["runtime_blocker"].startswith("generation_failed:ValueError")
    exp.validate_artifact(generation_failed)


def test_req_verify_3099_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3099: helper functions are deterministic and defensive."""
    rows = _manifest_rows()
    selected = exp.select_eval_rows(rows, minimum_count=3)
    assert [row["source_fixture_id"] for row in selected] == [
        "case-reject-invalid",
        "case-accept-valid-wrong",
        "case-accept-sat",
    ]

    prompt = exp.build_prompt(rows[0])
    assert "SAT_OR_UNSAT" in prompt
    assert "expected_answer" not in prompt
    assert rows[0]["source_fixture_id"] not in prompt
    assert rows[0]["source_prompt_payload_sha256"] in prompt

    assert exp.parse_response("abstain answer unknown low confidence") == {
        "raw_action": "abstain",
        "answer": "UNKNOWN",
        "verbal_confidence": 0.2,
    }
    assert exp.parse_response("accept answer valid high confidence")["verbal_confidence"] == 0.9
    assert exp.parse_response("accept answer valid medium confidence")["verbal_confidence"] == 0.5
    assert exp.parse_response("reject | answer=invalid | confidence=2.0") == {
        "raw_action": "reject",
        "answer": "INVALID",
        "verbal_confidence": 1.0,
    }
    assert exp.parse_response("nothing parseable") == {
        "raw_action": None,
        "answer": None,
        "verbal_confidence": None,
    }
    assert exp.answer_matches("sat.", "SAT") is True
    assert exp.answer_matches(None, "SAT") is False
    assert exp.safe_load_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp.safe_load_json(bad) == {}
    assert exp.load_jsonl(tmp_path / "missing.jsonl") == []

    output_without_logprobs = {"choices": [{"text": "ACCEPT | answer=SAT"}]}
    assert exp.confidence_from_output(output_without_logprobs)["confidence_available"] is False
    output_without_token_logprobs = {
        "choices": [{"text": "ACCEPT", "logprobs": {"top_logprobs": [{"A": math.log(1.0)}]}}]
    }
    no_token_confidence = exp.confidence_from_output(output_without_token_logprobs)
    assert no_token_confidence["confidence_available"] is False
    assert no_token_confidence["first_token_entropy"] == pytest.approx(0.0)
    assert exp.topk_entropy(None) is None
    assert exp.topk_entropy({"A": -999999.0}) is None
    assert exp.first_choice({"choices": []}) == {}
    assert exp.float_or_none("bad") is None
    assert exp.clamp01(-1.0) == 0.0
    assert exp.clamp01(2.0) == 1.0
    assert exp.relative_path(tmp_path, Path("/outside/file.json")) == "/outside/file.json"

    _write_sources(tmp_path)
    good = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        cached_pair_func=lambda **_: None,
        llama_factory=FakeLlama,
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="success prefix"):
        exp.validate_artifact(good | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(good | {"model_specs": []})
    with pytest.raises(ValueError, match="maxsat_policy_used"):
        exp.validate_artifact(good | {"maxsat_policy_used": False})
    with pytest.raises(ValueError, match="prompt_hashes"):
        exp.validate_artifact(good | {"prompt_hashes": []})
    with pytest.raises(ValueError, match="minimum_live_eval_count"):
        exp.validate_artifact(good | {"exact_ground_truth_count": 0})
    with pytest.raises(ValueError, match="blocked_sota_or_panel_precondition_failed"):
        exp.validate_artifact(good | {"abstention_panel_v3_ready": False})

    blocked = exp.blocked_metrics()
    assert blocked["solve_accuracy"] == 0.0
    assert exp.metrics_from_rows([]) == blocked
    telemetry = exp.thermodynamic_decode_telemetry([])
    assert telemetry["available"] is False
    no_model_result = exp.run_live_rows(
        [],
        None,
        policy_mod.build_policy_document(),
        _config(tmp_path),
        llama_factory=FakeLlama,
    )
    assert no_model_result == (
        [],
        "no_mandated_gguf_resolved",
    )
    selected_model = {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "model_path": str(_model_path(tmp_path)),
        "gpu": 0,
    }
    no_verbal_rows, no_verbal_blocker = exp.run_live_rows(
        [rows[0]],
        selected_model,
        policy_mod.build_policy_document(),
        _config(tmp_path),
        llama_factory=NoVerbalConfidenceLlama,
    )
    assert no_verbal_blocker is None
    assert no_verbal_rows[0]["confidence"] == pytest.approx(0.8)
    assert exp.exercise_cached_sota_pair(
        lambda **_: (_ for _ in ()).throw(RuntimeError("pair boom"))
    )["error"] == "RuntimeError:pair boom"
    assert (
        exp.first_precondition_failure(
            exp3097={},
            exp3098={"maxsat_policy_ready": True},
            policy=policy_mod.build_policy_document(),
            manifest_rows=rows,
            selected_rows=rows,
            minimum_count=4,
            cuda_status={"cuda_available": True},
            selected_model=selected_model,
        )
        == "exact_eval_protocol_unavailable"
    )
    assert (
        exp.first_precondition_failure(
            exp3097={"eval_protocol_ready": True},
            exp3098={"maxsat_policy_ready": True},
            policy=policy_mod.build_policy_document(),
            manifest_rows=rows[:2],
            selected_rows=rows[:2],
            minimum_count=4,
            cuda_status={"cuda_available": True},
            selected_model=selected_model,
        )
        == "minimum_live_eval_count_unavailable"
    )
    assert (
        exp.first_precondition_failure(
            exp3097={"eval_protocol_ready": True},
            exp3098={"maxsat_policy_ready": True},
            policy=policy_mod.build_policy_document(),
            manifest_rows=rows,
            selected_rows=rows,
            minimum_count=4,
            cuda_status={"cuda_available": False},
            selected_model=selected_model,
        )
        == "cuda_unavailable"
    )
    assert exp.negative_outcomes(
        [
            {"expected_action": "reject", "exact_answer_match": False, "route_decision": "accept"},
            {"expected_action": "accept", "exact_answer_match": True, "route_decision": "reject"},
        ]
    ) == [
        "false_accepts_observed",
        "false_rejects_observed",
        "solver_answer_errors_observed",
        "no_correct_abstentions_observed",
    ]
