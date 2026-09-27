"""REQ-REPORT-7782 keeps historical imports and inputs stable after rollover."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

from carnot import experiment_7131_v626_model_facing_csl as csl
from carnot import experiment_7495_v656_window_calibration as window
from carnot import experiment_7496_v656_causal_update_fixture as causal
from carnot import experiment_7782_v677_historical_compatibility as receipt
from carnot.experiment_7753_v675_contract_methods import resolve_authority
from carnot.inference.sota_models import LEGACY_COMPARATOR_GGUF_MODELS, current_model


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/python/fixtures"


@pytest.mark.parametrize(
    "module",
    [
        "experiment_5512_structured_output_positive_control",
        "experiment_5759_sota_exact_proposal_utility_panel",
        "experiment_5786_sota_constraint_stream",
        "experiment_5799_sota_answer_channel_canary",
        "experiment_6607_gemma4_26b_direct_headroom",
        "experiment_7131_v626_model_facing_csl",
        "experiment_7495_v656_window_calibration",
        "experiment_7496_v656_causal_update_fixture",
    ],
)
def test_historical_imports_keep_named_owners(module: str) -> None:
    """SCENARIO-REPORT-7782-COLLECTION: old modules must import by name."""
    imported = importlib.import_module(f"carnot.{module}")
    assert imported.__name__.endswith(module)


def test_current_registry_remains_single_qwen38() -> None:
    """SCENARIO-REPORT-7782-HISTORICAL: old metadata cannot change mandate."""
    assert current_model()["hf_id"] == "unsloth/Qwen3.8-27B-GGUF"
    assert [row["hf_id"] for row in LEGACY_COMPARATOR_GGUF_MODELS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]


@pytest.mark.parametrize(
    ("filename", "digest"),
    [
        (
            "roadmap_2026_09_675.yaml",
            "0fc9604cda91cc5198a95aad47ea0f48f700d5f5cd18cb5ce9b3fa3bc7f87171",
        ),
        (
            "roadmap_design_2026_09_675.md",
            "bef49b376955937ca54df4651063dd4e117b192bce18ee3f4a8a1f2103175bef",
        ),
    ],
)
def test_v675_fixture_bytes_are_immutable(filename: str, digest: str) -> None:
    """SCENARIO-REPORT-7782-HISTORICAL: fixture bytes have Git origin."""
    assert hashlib.sha256((FIXTURES / filename).read_bytes()).hexdigest() == digest


def test_later_roadmap_cannot_replace_v675_contract(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7782-HISTORICAL: current resolver stays strict."""
    original = (FIXTURES / "roadmap_2026_09_675.yaml").read_bytes()
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.09.676\ntasks: []\n")
    with pytest.raises(ValueError, match="matching V675 authority missing"):
        resolve_authority(tmp_path)
    assert (FIXTURES / "roadmap_2026_09_675.yaml").read_bytes() == original


def test_csl_cold_reader_rejects_invalid_lifecycle(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7782-COLLECTION: restored memory fails closed."""
    source = {"prompt_hash": csl.sha256_text("prompt"), "family": "sat_logic"}
    record = csl.signed_record(
        source_row=source,
        outcome={"exact_correct": True},
        decision_sequence=1,
        outcome_sequence=2,
        close_sequence=3,
        admitted_sequence=4,
        admitted_for_event_index=1,
    )
    store = csl.FixedSchemaMemory("fixed")
    assert store.commit(record, current_event_index=1)["committed"]
    assert store.retrieve("sat_logic", current_event_index=2) == [record]
    bad_time = {**record, "outcome_sequence": 0}
    bad_time["signature"] = csl.sha256_text(
        csl._canonical({key: value for key, value in bad_time.items() if key != "signature"})
    )
    with pytest.raises(csl.MemoryProtocolError, match="event_time_invalid"):
        csl.FixedSchemaMemory("other").commit(bad_time, current_event_index=1)
    with pytest.raises(csl.MemoryProtocolError, match="recovery_not_prepared"):
        store.recover({}, record, current_event_index=2)

    upstream = tmp_path / "upstream.json"
    upstream.write_text('{"sota_constraint_bank_ready_score":1,"rows":[]}')
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"fixture")
    specs = [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "model_path": str(model)}]
    common = {
        "repo_root": tmp_path,
        "upstream_path": upstream,
        "artifact_path": tmp_path / "result.json",
        "raw_dir": tmp_path / "raw",
        "run_date": "20260908",
        "model_specs": specs,
        "model_call": lambda *_args, **_kwargs: {},
    }
    with pytest.raises(RuntimeError, match="model_preflight_failed"):
        csl.run_experiment(**common, preflight_func=lambda **_kwargs: {"passed": False})
    with pytest.raises(ValueError, match="source_stream_invalid"):
        csl.run_experiment(**common, preflight_func=lambda **_kwargs: {"passed": True})
    blocked = csl._base_artifact("20260908", upstream, specs)
    csl._finish(blocked)
    blocked["model_facing_csl_complete_score"] = 1
    assert "blocked_gate_mismatch" in csl.validate_artifact(blocked)
    blocked["field_principles"] = {}
    assert "field_principles_incomplete" in csl.validate_artifact(blocked)
    assert "checksum_mismatch" in csl.validate_artifact(blocked)


def test_window_reader_rejects_missing_rows_and_changed_score(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7782-RECEIPT: raw evidence and scores stay bound."""
    assert (
        window.select_cost_policy([0, 1], [0.1, 0.9], false_accept_cost=5, escalation_cost=0.1)[
            "cost"
        ]
        == 0
    )
    artifact = window.build_artifact_for_test(tmp_path)
    artifact["raw_rows_sha256"] = "sha256:wrong"
    assert "raw_rows_hash_mismatch" in window.validate_artifact(
        artifact, root=tmp_path, require_validation=False
    )
    artifact = window.build_artifact_for_test(tmp_path)
    artifact["raw_rows_path"] = str(tmp_path / "missing.json")
    assert "raw_rows_invalid" in window.validate_artifact(
        artifact, root=tmp_path, require_validation=False
    )
    artifact = window.build_artifact_for_test(tmp_path)
    artifact["probability_benefit_score"] = 1
    assert "score_mismatch:probability_benefit_score" in window.validate_artifact(
        artifact, root=tmp_path, require_validation=False
    )


def test_causal_invalid_state_and_private_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7782-RECEIPT: CLI writes only private fixture bytes."""
    with pytest.raises(ValueError, match="training_fixture_invalid"):
        causal.BrierResidualHead.from_training([], seed=1, learning_rate=0.03, residual_bound=1)
    head = causal.BrierResidualHead.from_training(
        causal.training_fixture(), seed=1, learning_rate=0.03, residual_bound=1
    )
    head.seal_prediction(
        event_id="same",
        source_version="s",
        prediction_time=2,
        reveal_time=2,
        features=causal.training_fixture()[0],
        frozen_probability=0.5,
    )
    with pytest.raises(ValueError, match="event_identity_duplicate"):
        head.seal_prediction(
            event_id="same",
            source_version="s",
            prediction_time=2,
            reveal_time=2,
            features=causal.training_fixture()[0],
            frozen_probability=0.5,
        )
    assert head.apply_feedback("same", label=1, visible_at=1)["status"] == "not_revealed"
    head.seal_prediction(
        event_id="early",
        source_version="s",
        prediction_time=3,
        reveal_time=1,
        features=causal.training_fixture()[0],
        frozen_probability=0.5,
    )
    assert head.apply_feedback("early", label=1, visible_at=1)["status"] == "reordered"
    spec = tmp_path / "openspec/capabilities/kan/spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-KAN-7496")
    monkeypatch.setattr(causal, "REPO_ROOT", tmp_path)
    result = causal.run_experiment(tmp_path, causal.RUN_DATE, output_path=tmp_path / "direct.json")
    assert causal.validate_artifact(result) == []
    assert causal.main(["--date", causal.RUN_DATE]) == 0
    assert (tmp_path / "results/experiment_7496_v656_causal_update_fixture.json").is_file()


def test_receipt_reduces_every_frozen_node_and_rejects_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7782-RECEIPT: all 23 owners bind one checksum."""
    manifest = receipt.load_manifest(ROOT)
    log = tmp_path / "pass.log"
    log.write_text("190 passed\n")
    receipts = [
        receipt.command_receipt("affected", ["pytest", "affected"], 0, log, 1.0),
        receipt.command_receipt("collection", ["pytest", "--collect-only"], 0, log, 2.0),
    ]
    artifact = receipt.build_candidate(
        ROOT,
        manifest,
        receipts,
        [{"phase": "validation", "duration_s": 3.0}],
        flagged_adversarial=False,
    )
    assert len(artifact["compatibility_rows"]) == 23
    assert artifact["collection_ready_score"] == 1
    assert artifact["historical_fixture_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert receipt.cold_validate(artifact, ROOT) == []
    artifact["compatibility_rows"][0]["disposition"] = "unstarted"
    assert "compatibility_rows_mismatch" in receipt.cold_validate(artifact, ROOT)


def test_failed_validation_disqualifies_every_readiness_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7782-RECEIPT: a failed child cannot open a gate."""
    log = tmp_path / "failed.log"
    log.write_text("failure\n")
    failed = receipt.command_receipt("affected", ["pytest", "affected"], 1, log, 1.0)
    artifact = receipt.build_candidate(
        ROOT, receipt.load_manifest(ROOT), [failed], [], flagged_adversarial=False
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified")
    assert artifact["acceptance_gate_results"]["readiness"] == 0
    assert artifact["collection_ready_score"] == 0
    assert artifact["gate_check_summary"][0]["observed"] == 1
