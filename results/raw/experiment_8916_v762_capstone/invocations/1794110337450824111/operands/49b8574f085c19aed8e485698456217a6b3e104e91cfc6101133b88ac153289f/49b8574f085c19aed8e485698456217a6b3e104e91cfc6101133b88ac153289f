"""REQ-REPORT-8802 / REQ-VERIFY-8802: causal state and numerical readiness."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot.memory import causal_memory_8802 as m


def test_role_normalization_and_five_checks() -> None:
    """SCENARIO-VERIFY-8802-ENERGY-CONTROLS: additions change a later predicate."""
    first = m.world("first", "insufficient")
    renamed = m.world("renamed", "insufficient")
    assert m.normalize(first) == m.normalize(renamed)
    before = m.observe(first, [])
    after = m.observe(renamed, [m.normalize(first)])
    assert before["primitive_checks"] == after["primitive_checks"] == 5
    assert before["counts"][-1] > after["counts"][-1]
    assert m.observe(first, ["useless"]) == before
    for key in ["label", "world_id", "solver_answer", "future_labels", "condition_id"]:
        with pytest.raises(ValueError):
            m.observe(dict(first, **{key: "secret"}), [])
    with pytest.raises(ValueError):
        m.observe(dict(first, query=first["query"][:4]), [])


def test_gradients_and_thresholds() -> None:
    """REQ-VERIFY-8802: genuine NCE and Brier gradients train the fixed Gibbs model."""
    params = m.initial_parameters(11)
    counts = m.observe(m.world("fit", "support"), [m.normalize(m.world("fit", "support"))])[
        "counts"
    ]
    changed, receipt = m.update(params, counts, "support")
    assert receipt["gradient_norm"] > 0 and receipt["parameter_delta"] > 0
    assert np.isfinite(receipt["loss"])
    assert len(m.probabilities(changed, counts)) == 3
    assert sum(m.probabilities(changed, counts)) == pytest.approx(1)
    assert len(m.features(counts, 0)) == 9
    assert m.action([0.9, 0.05, 0.05], 0.6) == "accept"
    assert m.action([0.05, 0.9, 0.05], 0.6) == "reject"
    assert m.action([0.3, 0.3, 0.4], 0.6) == "escalate"
    assert m.tune(changed, [m.world("tune", "support")]) in [0.5, 0.6, 0.7, 0.8]


def test_frozen_pending_release_duplicate_and_restart(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8802-TRANSACTION: every state component survives restart."""
    store = m.Store(tmp_path / "state.json", m.initial_parameters(11))
    query = m.world("query", "insufficient")
    sealed = store.predict("e0", query, 0.6)
    assert sealed["primitive_checks"] == 5
    with pytest.raises(RuntimeError, match="active"):
        store.predict("other", query, 0.6)
    with pytest.raises(RuntimeError, match="active"):
        store.release("e0", query, 8, "combined")
    assert store.read() == store.read()
    store.close()
    before = deepcopy(store.state)
    reloaded = m.Store(store.path, m.initial_parameters(23))
    assert reloaded.state == before and reloaded.state["pending"]
    with pytest.raises(ValueError, match="early"):
        reloaded.release("e0", query, 7, "combined")
    with pytest.raises(ValueError, match="evidence"):
        reloaded.release("e0", m.world("changed", "support"), 8, "combined")
    receipt = reloaded.release("e0", query, 8, "combined")
    assert receipt["template_added"] and receipt["coefficient_committed"]
    assert reloaded.state["acks"] == ["e0"] and not reloaded.state["pending"]
    with pytest.raises(ValueError, match="duplicate"):
        reloaded.release("e0", query, 9, "combined")
    assert m.Store(store.path, {}).state == reloaded.state


def test_controls_and_complete_counts(tmp_path: Path) -> None:
    """REQ-REPORT-8802: all scheduled prototype arms retain equal budgets."""
    result = m.prototype(tmp_path, seeds=[11])
    assert len(result["rows"]) == 5 * 12
    assert all(r["primitive_checks"] == 5 for r in result["rows"])
    assert all(r["passed"] for r in result["leakage_controls"])
    assert {"duplicate_feedback", "label_leakage_rejected", "active_write_rejected"} <= {
        r["control"] for r in result["leakage_controls"]
    }
    assert all(r["passed"] for r in result["restart_rows"])
    assert all(r["within_query_unchanged"] for r in result["rows"])
    assert len(result["gradient_update_rows"]) == 5 * 4
    assert {r["arm"] for r in result["rows"]} == set(m.ARMS)
    assert result["template_addition_rows"]
    for row in result["gradient_update_rows"]:
        assert row["feedback_label"] in row["released_prefix"]
    manifest = m.freeze_manifest()
    assert len(manifest["streams"]) == 30
    assert all(len(s["worlds"]) == 32 and len(s["retention"]) == 12 for s in manifest["streams"])
    assert manifest == m.freeze_manifest()
    assert json.loads((tmp_path / "combined-11.json").read_text())["pending"]


def test_signed_evidence_and_duplicate_prediction(tmp_path: Path) -> None:
    """REQ-VERIFY-8802: explicit refutation and conflict remain distinct from support."""
    payload = m.world("negative", "refute")
    assert m.observe(payload, [m.normalize(payload)])["decision"] == "refute"
    payload["facts"].append(payload["facts"][0][:3] + [True])
    assert m.observe(payload, [m.normalize(payload)])["decision"] == "insufficient"
    store = m.Store(tmp_path / "state.json", m.initial_parameters(11))
    store.predict("e0", m.world("original", "support"), 0.6)
    store.close()
    with pytest.raises(ValueError, match="duplicate"):
        store.predict("e0", m.world("original", "support"), 0.6)
    store.state["pending"] = {f"e{i}": {} for i in range(32)}
    with pytest.raises(ValueError, match="capacity"):
        store.predict("e32", m.world("overflow", "support"), 0.6)
