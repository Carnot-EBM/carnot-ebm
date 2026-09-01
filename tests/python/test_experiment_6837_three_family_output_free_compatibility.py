"""REQ-CONSTRAINT-6837 output-free compatibility scoring tests."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6837_three_family_output_free_compatibility as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / exp.SPEC_RELATIVE_PATH


class FakeForcedSequenceScorer:
    """Small scorer that returns candidate-only token logprobs."""

    start_order: list[str] = []
    close_order: list[str] = []

    def __init__(self, model_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.model_spec = dict(model_spec)
        self.config = dict(config)
        self.closed = False
        self.score_calls: list[str] = []

    def start(self) -> dict[str, Any]:
        FakeForcedSequenceScorer.start_order.append(str(self.model_spec["hf_id"]))
        return {
            "command": [
                ".venv/bin/python",
                "-m",
                "carnot.experiment_6837_three_family_output_free_compatibility",
                "--score-worker",
                "--model-path",
                self.model_spec["model_path"],
                "--port",
                str(9100 + len(FakeForcedSequenceScorer.start_order)),
            ],
            "pid": 7000 + len(FakeForcedSequenceScorer.start_order),
            "process_start": f"fixture-start-{len(FakeForcedSequenceScorer.start_order)}",
            "port": 9100 + len(FakeForcedSequenceScorer.start_order),
            "gpu_uuid": f"GPU-fixture-{self.model_spec['family']}",
            "visible_devices": [self.model_spec["gpu"]],
            "model_hash": self.model_spec["model_sha256"],
            "tokenizer_hash": self.model_spec["tokenizer_receipt"]["receipt_hash"],
            "owned_by_task": True,
            "token_logprob_support": True,
            "first_score": None,
            "final_score": None,
        }

    def score(
        self, prompt_text: str, candidate_text: str, row_identity: Mapping[str, Any]
    ) -> dict[str, Any]:
        del prompt_text
        self.score_calls.append(str(row_identity["row_identity"]))
        token_ids = [len(piece) + index for index, piece in enumerate(candidate_text.split("|"))]
        if "unequal" in candidate_text:
            token_ids.append(999)
        base = -0.1 if row_identity["label"] == "compatible" else -0.4
        if self.config.get("reverse_margin"):
            base = -0.8 if row_identity["label"] == "compatible" else -0.2
        token_logprobs = [round(base - index * 0.01, 6) for index in range(len(token_ids))]
        score = round(sum(token_logprobs), 6)
        return {
            "prompt_token_ids": [101, 102, 103],
            "candidate_token_ids": token_ids,
            "token_logprobs": token_logprobs,
            "conditional_log_likelihood": score,
            "raw_receipt": {
                "row_identity": row_identity["row_identity"],
                "label": row_identity["label"],
                "prompt_token_count": 3,
                "candidate_token_count": len(token_ids),
                "token_logprob_count": len(token_logprobs),
            },
        }

    def close(self) -> dict[str, Any]:
        FakeForcedSequenceScorer.close_order.append(str(self.model_spec["hf_id"]))
        self.closed = True
        return {
            "action": "terminated",
            "bounded": True,
            "leak_free": True,
            "unrelated_process_kill_count_delta": 0,
        }


class CountingScorer(FakeForcedSequenceScorer):
    """Scorer variant used to prove checkpoint rows are not regenerated."""

    calls_by_model: dict[str, list[str]] = {}

    def score(
        self, prompt_text: str, candidate_text: str, row_identity: Mapping[str, Any]
    ) -> dict[str, Any]:
        identities = CountingScorer.calls_by_model.setdefault(str(self.model_spec["hf_id"]), [])
        identity = str(row_identity["row_identity"])
        if identity not in identities:
            identities.append(identity)
        return super().score(prompt_text, candidate_text, row_identity)


class MisalignedScorer(FakeForcedSequenceScorer):
    """Scorer variant that violates token/logprob alignment."""

    def score(
        self, prompt_text: str, candidate_text: str, row_identity: Mapping[str, Any]
    ) -> dict[str, Any]:
        del prompt_text, candidate_text, row_identity
        return {
            "prompt_token_ids": [1],
            "candidate_token_ids": [2, 3],
            "token_logprobs": [-0.1],
            "conditional_log_likelihood": -0.1,
            "raw_receipt": {},
        }


def _model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, hf_id in enumerate(exp.MANDATED_MODEL_HF_IDS):
        model_path = tmp_path / f"model-{index}.gguf"
        model_path.write_bytes(f"GGUF fixture {hf_id}".encode())
        specs.append(
            {
                "name": hf_id.rsplit("/", 1)[-1],
                "hf_id": hf_id,
                "family": exp.model_family(hf_id),
                "role": "fixture",
                "gpu": index,
                "model_path": str(model_path),
                "cache_path": str(model_path),
                "local_model_present": True,
                "model_sha256": exp.sha256_file(model_path),
                "tokenizer_receipt": {
                    "source": "fixture",
                    "loadable": True,
                    "detail": "fixture tokenizer metadata",
                    "receipt_hash": exp.sha256_text(f"tokenizer:{hf_id}"),
                },
                "headline_eligible": True,
                "quantization": "Q4_K_M",
            }
        )
    return specs


def _ready_preconditions() -> dict[str, Any]:
    return {
        "preconditions_ready": True,
        "blocked_reasons": [],
        "checks": [
            exp.gate_check("cached_sota_pair_called", True, True),
            exp.gate_check(
                "all_three_exact_gguf_files",
                list(exp.MANDATED_MODEL_HF_IDS),
                list(exp.MANDATED_MODEL_HF_IDS),
            ),
            exp.gate_check("model_hashes_present", True, True),
            exp.gate_check("native_tokenizer_metadata", True, True),
            exp.gate_check("token_log_probability_support", True, True),
            exp.gate_check("cuda_available", True, True),
            exp.gate_check("exclusive_gpu_leases", True, True),
            exp.gate_check("free_ports", True, True),
            exp.gate_check("sufficient_disk", True, True),
            exp.gate_check("exp6836_typed_obligation_program_ready_score", 1, 1),
            exp.gate_check("exp6836_obligation_pair_fixture_ready_score", 1, 1),
            exp.gate_check("live_canary_per_model", True, True),
        ],
        "accelerator_samples": [
            {"gpu_uuid": "GPU-fixture-0", "free_vram_mb": 24000, "owned_by_task": True}
        ],
        "free_ports": [9101, 9102, 9103],
    }


def _fixture_rows(limit: int = 2) -> list[dict[str, Any]]:
    payload = exp.read_json(REPO_ROOT / exp.EXP6836_RELATIVE_PATH)
    return [dict(row) for row in payload["rows"][:limit]]


def test_req_constraint_6837_spec_declares_contract_and_fields() -> None:
    """REQ-CONSTRAINT-6837 is anchored before implementation behavior."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("## REQ-CONSTRAINT-6837:", 1)[1]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CONSTRAINT-6837-PRECONDITIONS",
        "SCENARIO-CONSTRAINT-6837-FORCED-SCORING",
        "SCENARIO-CONSTRAINT-6837-TOKEN-ALIGNMENT",
        "SCENARIO-CONSTRAINT-6837-CHECKPOINT-RESTART",
        "SCENARIO-CONSTRAINT-6837-PROCESS-OWNERSHIP",
        "SCENARIO-CONSTRAINT-6837-ARTIFACT",
        exp.RESULT_RELATIVE_PATH.as_posix(),
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for hf_id in exp.MANDATED_MODEL_HF_IDS:
        assert hf_id in section
    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6837_preconditions_call_cached_pair_and_resolve_three(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CONSTRAINT-6837-PRECONDITIONS requires exact model files."""

    calls: list[str] = []
    for name in ("qwen.gguf", "gemma31.gguf", "gemma26.gguf"):
        (tmp_path / name).write_bytes(name.encode("utf-8"))

    def fake_cached_sota_pair(*, gpu_indices: tuple[int, int] = (0, 1)) -> list[dict[str, Any]]:
        calls.append(f"cached:{gpu_indices}")
        return [
            {
                "name": "qwen",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": str(tmp_path / "qwen.gguf"),
            },
            {
                "name": "gemma26",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 1,
                "model_path": str(tmp_path / "gemma26.gguf"),
            },
        ]

    def fake_resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str:
        calls.append(f"resolve:{hf_id}:{preferred_quant}")
        return str(tmp_path / "gemma31.gguf")

    monkeypatch.setattr(exp, "cached_sota_pair", fake_cached_sota_pair)
    monkeypatch.setattr(exp, "resolve_cached_gguf", fake_resolve)
    monkeypatch.setattr(exp, "gguf_tokenizer_loadable", lambda path: (True, f"ok:{path}"))

    specs = exp.resolve_model_specs()

    assert calls[0] == "cached:(0, 1)"
    assert calls[1] == "resolve:unsloth/gemma-4-31B-it-GGUF:Q4_K_M"
    assert [row["hf_id"] for row in specs] == list(exp.MANDATED_MODEL_HF_IDS)
    assert all(row["local_model_present"] is True for row in specs)
    assert all(row["model_sha256"].startswith("sha256:") for row in specs)
    assert all(row["tokenizer_receipt"]["loadable"] is True for row in specs)


def test_scenario_6837_forced_scoring_masks_prompt_and_records_raw_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6837-FORCED-SCORING stores token ids and logprobs."""

    scorer = FakeForcedSequenceScorer(_model_specs(tmp_path)[0], {})
    row = exp.score_fixture_row(
        fixture_row=_fixture_rows(1)[0],
        model_spec=_model_specs(tmp_path)[0],
        scorer=scorer,
    )

    assert row["compatible_label_position"] == 0
    assert row["candidate_length"] == len(row["compatible"]["candidate_token_ids"])
    assert row["compatible"]["prompt_token_ids"] == [101, 102, 103]
    assert len(row["compatible"]["token_logprobs"]) == row["candidate_length"]
    assert len(row["violation"]["token_logprobs"]) == row["candidate_length"]
    assert (
        row["compatible"]["conditional_log_likelihood"]
        > row["violation"]["conditional_log_likelihood"]
    )
    assert row["log_likelihood_margin"] > 0
    assert row["compatible"]["raw_receipt"]["prompt_token_count"] == 3
    assert "prompt_token_logprobs" not in row["compatible"]


def test_scenario_6837_token_alignment_rejects_unequal_candidate_lengths(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6837-TOKEN-ALIGNMENT forbids unequal-token margins."""

    fixture_row = _fixture_rows(1)[0]
    fixture_row = json.loads(json.dumps(fixture_row))
    fixture_row["candidates"][1]["expected_tokenization_inputs"]["candidate_text"] += "|unequal"

    with pytest.raises(exp.SequenceScoringError, match="unequal_candidate_token_length"):
        exp.score_fixture_row(
            fixture_row=fixture_row,
            model_spec=_model_specs(tmp_path)[0],
            scorer=FakeForcedSequenceScorer(_model_specs(tmp_path)[0], {}),
        )


def test_scenario_6837_checkpoint_restart_skips_complete_rows(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6837-CHECKPOINT-RESTART never regenerates complete rows."""

    model_spec = _model_specs(tmp_path)[0]
    rows = _fixture_rows(2)
    first = exp.score_fixture_row(
        fixture_row=rows[0],
        model_spec=model_spec,
        scorer=FakeForcedSequenceScorer(model_spec, {}),
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    exp.write_checkpoint(
        checkpoint_path,
        exp.build_checkpoint_manifest([first], expected_row_count=2),
    )
    CountingScorer.calls_by_model = {}

    phase = exp.run_model_phase(
        model_spec=model_spec,
        fixture_rows=rows,
        scorer_factory=CountingScorer,
        checkpoint_path=checkpoint_path,
    )

    assert phase["row_count"] == 2
    assert phase["rows"][0] == first
    assert CountingScorer.calls_by_model[model_spec["hf_id"]] == [
        f"{model_spec['family']}::{rows[1]['row_id']}"
    ]
    assert phase["checkpoint_manifest"]["resumed_row_count"] == 1
    assert phase["checkpoint_manifest"]["missing_row_count"] == 1


def test_scenario_6837_process_ownership_model_isolation_and_teardown(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6837-PROCESS-OWNERSHIP isolates model phases."""

    FakeForcedSequenceScorer.start_order = []
    FakeForcedSequenceScorer.close_order = []
    artifact = exp.run(
        root=REPO_ROOT,
        result_path=tmp_path / "artifact.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_ready_preconditions(),
        scorer_factory=FakeForcedSequenceScorer,
        fixture_row_limit=1,
        write=True,
    )

    assert [row["hf_id"] for row in artifact["process_receipts"]] == list(exp.MANDATED_MODEL_HF_IDS)
    assert len({row["pid"] for row in artifact["process_receipts"]}) == 3
    assert all(row["owned_by_task"] is True for row in artifact["process_receipts"])
    assert all(row["teardown"]["leak_free"] is True for row in artifact["process_receipts"])
    assert FakeForcedSequenceScorer.start_order == list(exp.MANDATED_MODEL_HF_IDS)
    assert FakeForcedSequenceScorer.close_order == list(exp.MANDATED_MODEL_HF_IDS)
    assert (tmp_path / "artifact.json").is_file()


def test_scenario_6837_readiness_depends_on_rows_not_margin_direction(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6837-ARTIFACT separates readiness from effect size."""

    artifact = exp.run(
        root=REPO_ROOT,
        result_path=tmp_path / "artifact.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_ready_preconditions(),
        scorer_factory=FakeForcedSequenceScorer,
        scorer_config={"reverse_margin": True},
        fixture_row_limit=1,
        write=False,
    )

    assert artifact["obligation_compatibility_stream_ready_score"] == 1
    assert artifact["rows"][0]["log_likelihood_margin"] < 0
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["method_parity_limits"]["effect_estimate_is_truth_proof"] is False


def test_scenario_6837_blocked_output_names_exact_failed_gate(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6837-PRECONDITIONS emits blocked artifact rows only."""

    blocked = dict(_ready_preconditions())
    blocked["preconditions_ready"] = False
    blocked["blocked_reasons"] = ["cuda_available"]
    blocked["checks"] = [exp.gate_check("cuda_available", True, False)]

    artifact = exp.run(
        root=REPO_ROOT,
        result_path=tmp_path / "blocked.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=_model_specs(tmp_path),
        preconditions_checked=blocked,
        scorer_factory=FakeForcedSequenceScorer,
        fixture_row_limit=1,
        write=True,
    )

    assert artifact["rows"] == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_output_free_compatibility"
    assert artifact["gate_check_summary"]["failed_check"] == "cuda_available"
    assert artifact["gate_check_summary"]["observed"] is False
    assert artifact["obligation_compatibility_stream_ready_score"] == 0
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_6837_guard_failures_are_explicit(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6837 exposes malformed rows and checkpoints as errors."""

    assert exp.sha256_bytes(b"x").startswith("sha256:")

    invalid_json = tmp_path / "array.json"
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(exp.SequenceScoringError, match="json_object_required"):
        exp.read_json(invalid_json)

    with pytest.raises(exp.SequenceScoringError, match="unknown_model_hf_id"):
        exp.model_family("not-a-model")

    missing_specs = exp.normalize_model_specs([])
    assert all(row["local_model_present"] is False for row in missing_specs)
    assert all(
        row["tokenizer_receipt"]["receipt_hash"].startswith("sha256:") for row in missing_specs
    )

    empty_root = tmp_path / "empty-root"
    exp.write_json_atomic(empty_root / exp.EXP6836_RELATIVE_PATH, {"rows": []})
    with pytest.raises(exp.SequenceScoringError, match="exp6836_rows_missing"):
        exp.load_fixture_rows(empty_root)

    fixture_row = json.loads(json.dumps(_fixture_rows(1)[0]))
    fixture_row["candidates"][1]["exact_check"]["satisfaction_predicate"] = True
    with pytest.raises(exp.SequenceScoringError, match="one_compatible_one_violation_required"):
        exp.score_fixture_row(
            fixture_row=fixture_row,
            model_spec=_model_specs(tmp_path)[0],
            scorer=FakeForcedSequenceScorer(_model_specs(tmp_path)[0], {}),
        )

    with pytest.raises(exp.SequenceScoringError, match="token_logprob_alignment_failed"):
        exp.score_fixture_row(
            fixture_row=_fixture_rows(1)[0],
            model_spec=_model_specs(tmp_path)[0],
            scorer=MisalignedScorer(_model_specs(tmp_path)[0], {}),
        )

    model_spec = _model_specs(tmp_path)[0]
    scored = exp.score_fixture_row(
        fixture_row=_fixture_rows(1)[0],
        model_spec=model_spec,
        scorer=FakeForcedSequenceScorer(model_spec, {}),
    )
    scored["row_hash"] = "sha256:bad"
    bad_checkpoint = tmp_path / "bad-checkpoint.json"
    exp.write_checkpoint(
        bad_checkpoint,
        exp.build_checkpoint_manifest([scored], expected_row_count=1),
    )
    with pytest.raises(exp.SequenceScoringError, match="checkpoint_row_hash_mismatch"):
        exp.run_model_phase(
            model_spec=model_spec,
            fixture_rows=_fixture_rows(1),
            scorer_factory=FakeForcedSequenceScorer,
            checkpoint_path=bad_checkpoint,
        )


def test_req_constraint_6837_main_cli_dispatches_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CONSTRAINT-6837 exposes the requested script entrypoint."""

    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        return {"honest_verdict": "complete_blocked_output_free_compatibility"}

    monkeypatch.setattr(exp, "run", fake_run)

    result_path = tmp_path / "artifact.json"
    assert exp.main(["--date", "20260901", "--result-path", str(result_path)]) == 0
    assert calls[0]["result_path"] == result_path
    assert "complete_blocked_output_free_compatibility" in capsys.readouterr().out

    with pytest.raises(exp.SequenceScoringError, match="run_date_mismatch"):
        exp.main(["--date", "20260902"])
