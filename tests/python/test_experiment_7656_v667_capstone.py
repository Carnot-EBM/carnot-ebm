"""V667 capstone contract checks: REQ-REPORT-7656."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7656_v667_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


def test_ordered_custody_and_external_block() -> None:
    """SCENARIO-REPORT-7656-ORDER: count actual producers and pre-gates."""

    artifact = capstone.build_artifact(ROOT)
    dispositions = artifact["milestone_dispositions"]
    assert len(dispositions) == 14
    assert [row["order"] for row in dispositions] == list(range(1, 15))
    assert dispositions[4]["custody_kind"] == "conductor_pre_gate"
    assert dispositions[4]["actual_path"].endswith("experiment_7647_witness_energy.json")
    assert dispositions[-1]["custody_kind"] == "current_self"
    assert dispositions[-1]["actual_path"] is None
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert all(
        set(check) >= {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        for check in artifact["gate_check_summary"]["failed_checks"]
    )


def test_measured_null_and_branch_boundaries() -> None:
    """SCENARIO-REPORT-7656-REDUCTION: raw groups, controls, and ARC stay separate."""

    artifact = capstone.build_artifact(ROOT)
    summary = artifact["evidence_summary"]
    assert summary["source_corpus"]["independent_groups"] == 240
    assert summary["source_corpus"]["checked_predicates"] == 0
    assert summary["source_corpus"]["coverage_denominator"] > 0
    assert summary["fixture"]["verifier_is_oracle"] is True
    assert summary["source_audit"]["benefit_eligible"] is False
    assert summary["arc_wrapper"]["verdict_class"] == "disqualified"
    assert summary["arc_live"]["verdict_class"] == "disqualified"
    assert artifact["sample_size_budget"]["source_groups"]["observed"] == 240
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["capstone_complete_score"] == 1
    assert {row["gate"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "probability_benefit",
        "utility",
        "retention",
        "freshness",
    }


def test_cold_replay_rejects_four_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7656-COLD-REPLAY: reload sources and reject corrupted claims."""

    artifact = capstone.build_artifact(ROOT)
    assert capstone.independent_reduce(artifact, ROOT) == []
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert capstone.read_candidate(path, ROOT) == []
    for mutation in ("deleted", "reordered", "wrong_field", "self_input"):
        changed = capstone.mutate_candidate(deepcopy(artifact), mutation)
        path.write_text(json.dumps(changed), encoding="utf-8")
        assert capstone.read_candidate(path, ROOT), mutation
    with pytest.raises(ValueError, match="unknown mutation"):
        capstone.mutate_candidate(deepcopy(artifact), "unknown")


def test_publication_and_decisions_are_scoped() -> None:
    """REQ-REPORT-7656: a failed mechanism needs a changed premise."""

    artifact = capstone.build_artifact(ROOT)
    assert artifact["unmet_gates"] == [
        name for name, gate in artifact["publication_gates"]["gates"].items() if not gate["pass"]
    ]
    assert {"G1", "G2", "G3", "G4"} <= set(artifact["publication_gates"]["gates"])
    assert all(row["changed_premise_required"] for row in artifact["next_decisions"])
    assert any(
        row["scope"] == "source_witness_parser" and row["decision"] == "retire"
        for row in artifact["next_decisions"]
    )
    assert artifact["capstone_note_path"] == "docs/research-notes/v667-capstone.md"


def test_authority_and_raw_reduction_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7656: corrupt authority or raw counts cannot pass."""

    authority = capstone.load_authority(ROOT)
    with pytest.raises(ValueError, match="roster order"):
        capstone.collect_dispositions(ROOT, authority["tasks"][:-1])
    monkeypatch.setattr(
        capstone.contract,
        "compare_authorities",
        lambda *_: {"passed": False, "errors": ["private"]},
    )
    with pytest.raises(ValueError, match="authority mismatch"):
        capstone.load_authority(ROOT)
    monkeypatch.undo()

    original = capstone.prior.load_json

    def corrupt_corpus(path: Path) -> dict:
        value = original(path)
        if path.name.startswith("experiment_7646_"):
            value["rows"] = [
                row
                for row in value["rows"]
                if row.get("source_group_id")
                != "sha256:e14d2875c04e4d824abed9333e316e607c5a798ef44afdd49e47a345a54e4218"
            ]
        return value

    monkeypatch.setattr(capstone.prior, "load_json", corrupt_corpus)
    with pytest.raises(ValueError, match="raw sidecars"):
        capstone.reduce_evidence(ROOT)


def test_cold_reader_diagnostics_and_mutation_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7656-COLD-REPLAY: malformed claims name their failure."""

    artifact = capstone.build_artifact(ROOT)
    assert capstone.independent_reduce(None, ROOT) == ["artifact_object_required"]
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "wrong"
    assert "checksum_mismatch" in capstone.independent_reduce(changed, ROOT)
    changed = deepcopy(artifact)
    changed["milestone_dispositions"][0] = {}
    assert "milestone_disposition_order" in capstone.independent_reduce(changed, ROOT)
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["uninvoked"]
    assert "MODEL_SPECS_mismatch" in capstone.independent_reduce(changed, ROOT)
    changed = deepcopy(artifact)
    changed["field_principles"].pop("field_principles")
    assert "field_principles_incomplete" in capstone.independent_reduce(changed, ROOT)
    assert capstone.task_specific_e2e(artifact, ROOT) == []

    def fail_builder(_root: Path) -> dict:
        raise ValueError("corrupt private source")

    monkeypatch.setattr(capstone, "build_artifact", fail_builder)
    assert "cold_reduction_failed:ValueError" in capstone.independent_reduce(artifact, ROOT)
    monkeypatch.undo()
    monkeypatch.setattr(capstone, "independent_reduce", lambda *_: [])
    assert len(capstone.task_specific_e2e(artifact, ROOT)) == 4


def test_missing_named_input_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7656: planned output is not a required input."""

    authority = capstone.load_authority(ROOT)
    dispositions = capstone.collect_dispositions(ROOT, authority["tasks"])
    monkeypatch.setattr(capstone, "INPUTS", (*capstone.INPUTS, "missing-private-input"))
    hashes = capstone._sources(ROOT, dispositions, authority)
    assert "missing-private-input" in hashes["missing_inputs"]
    assert capstone.OUTPUT not in hashes["missing_inputs"]


@pytest.mark.parametrize(
    ("producer", "message"),
    [
        (7650, "raw reduction disagrees"),
        (7652, "ARC wrapper producer rows disagree"),
        (7653, "ARC live producer rows disagree"),
    ],
)
def test_raw_branch_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch, producer: int, message: str
) -> None:
    """SCENARIO-REPORT-7656-REDUCTION: three raw branches require byte-consistent rows."""

    original = capstone.prior.load_json

    def corrupt(path: Path) -> dict:
        value = original(path)
        if path.name.startswith(f"experiment_{producer}_v667_"):
            if producer == 7650:
                value["independent_findings"]["exp7646"]["checked_predicates"] = 1
            else:
                value["rows"] = value["rows"][1:]
        return value

    monkeypatch.setattr(capstone.prior, "load_json", corrupt)
    with pytest.raises(ValueError, match=message):
        capstone.reduce_evidence(ROOT)
