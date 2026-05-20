import json

from scripts.experiment_2581_retro_v247 import build_retro, write_retro


def _load_capstone():
    with open("results/experiment_2580_capstone_v247.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_build_retro_v247_carries_required_capstone_fields():
    """REQ-REPORT-009: the .247 retro reports source capstone fields honestly."""
    capstone = _load_capstone()

    data = build_retro(capstone)

    assert data["honest_verdict"].startswith("complete:")
    assert data["schema"] == "carnot.operational_retro.v69"
    assert data["n_experiments_completed"] == capstone["n_experiments_completed"] == 0
    assert data["best_247_auroc"] == capstone["best_247_auroc"]
    assert data["safety_classifier_viable"] is capstone["safety_classifier_viable"] is False
    assert data["tier0s_real_improvement"] is capstone["tier0s_real_improvement"] is False
    assert data["tier0u_real_improvement"] is capstone["tier0u_real_improvement"] is False
    assert data["gatemate_status"] == capstone["gatemate_status"]
    assert data["kv260_status"] == capstone["kv260_status"]
    assert data["operator_recommendation"] == "hardware_terminal_pending"


def test_build_retro_v247_identifies_three_successes_and_three_gaps():
    """REQ-REPORT-009: planner-facing successes and gaps are fixed-size summaries."""
    data = build_retro(_load_capstone())

    assert len(data["top_3_successes"]) == 3
    assert [item["rank"] for item in data["top_3_successes"]] == [1, 2, 3]
    assert len(data["top_3_gaps_for_248"]) == 3
    assert [item["rank"] for item in data["top_3_gaps_for_248"]] == [1, 2, 3]
    assert data["field_principles"]["honest_verdict"].startswith("Terminal-prefix")
    assert data["acceptance_gates"][0]["passed"] is True


def test_write_retro_v247_writes_requested_deliverable(tmp_path):
    """REQ-REPORT-009: the writer persists the terminal retro artifact."""
    out_path = tmp_path / "experiment_2581_retro_v247.json"

    data = write_retro(output_path=out_path)

    with out_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data
    assert loaded["schema"] == "carnot.operational_retro.v69"
