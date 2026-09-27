"""Evidence-view fixtures for REQ-VERIFY-7756 and REQ-REPORT-7756."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.verify import evidence_views as views
from carnot.verify import source_set_energy as energy
from carnot.verify import training_runtime as runtime


def test_bytes_offsets_and_intervention() -> None:
    """SCENARIO-VERIFY-7756-WITNESS: both views retain every input byte."""
    source = "Élan!  東京? Third value is 12. final fragment".encode()
    answer = "東京? Answer continues".encode()
    pair = views.prepare_views(source, answer)
    a, b = pair["a"], pair["b"]
    for view in (a, b):
        assert view["source_bytes"] == source
        assert view["answer_bytes"] == answer
        assert b"".join(view["source_sentences"]) == source
        assert b"".join(view["answer_units"]) == answer
        for field, units, original in (
            ("source_offsets", view["source_sentences"], source),
            ("answer_offsets", view["answer_units"], answer),
        ):
            assert [original[start:end] for start, end in view[field]] == units
        assert view["source_sha256"] == views.digest(source)
        assert view["answer_sha256"] == views.digest(answer)
    assert len(a["windows"]) == 6
    assert len(b["windows"]) == 7
    assert a["windows"][-1] == b"".join(a["source_sentences"][1:4])
    assert b["windows"][-1] == b"".join(b["source_sentences"][2:4])
    assert a["pair_features"] != b["pair_features"]
    assert views.prepare_views(source, answer) == pair


def test_duplicate_null_budget_and_invalid() -> None:
    """SCENARIO-VERIFY-7756-WITNESS: equal group mass and shared abstention."""
    pair = views.prepare_views(b"A. A.", b"A.")
    assert pair["a"]["group_ids"] == [0, 0]
    assert views.location_prior(pair["a"]) == pytest.approx([0.25, 0.25, 0.5])
    empty = views.prepare_views(b"", b"A.")
    assert empty["a"]["windows"] == []
    assert views.location_prior(empty["a"]) == [1.0]
    assert energy.distribution(empty["a"], energy.zero_parameters())["null_mass"] == [1.0]
    over = views.prepare_views(b"A. " * 65, b"A.")
    assert len(over["a"]["windows"]) <= 128
    assert len(over["b"]["windows"]) > 128
    assert over["a"]["abstention"] == over["b"]["abstention"] == "source_windows_over_budget"
    assert views.prepare_views(b"A.", b"B. " * 17)["a"]["abstention"] == "answer_units_over_budget"
    assert views.prepare_views(b"A.", b"")["b"]["abstention"] == "empty_answer"
    assert views.prepare_views(b"\xff", b"A.")["a"]["abstention"] == "invalid_utf8"


def test_pooled_sentence_shapes_padding_and_decision() -> None:
    """SCENARIO-VERIFY-7756-DECISION: matched heads see each sentence and one policy."""
    pair = views.prepare_views(b"Alpha 12. Beta 30. Gamma 50.", b"Alpha 12. Beta 31.")
    rows = [
        {"view_a": pair["a"], "view_b": pair["b"], "known": [1, -1], "label": 1},
        {
            "view_a": views.prepare_views(b"A.", b"A.")["a"],
            "view_b": views.prepare_views(b"A.", b"A.")["b"],
            "known": [1],
            "label": 0,
        },
    ]
    batch = runtime.prepare_batch(rows)
    assert batch["a"]["x"].shape[:2] == (2, 2)
    assert np.asarray(batch["a"]["unit_mask"])[1, 1] == 0
    assert np.asarray(batch["a"]["place_mask"])[1, -1] == 0
    for arm in ("energy_local", "logistic_local", "mlp_local"):
        params = runtime.init_params(arm, 67501)
        support = np.asarray(runtime.sentence_support(params, batch["a"], arm))
        assert support.shape == (2, 2)
        assert support[1, 1] == 1.0
        assert float(runtime.predict(params, batch["a"], arm)[0]) == pytest.approx(
            1 - float(np.prod(support[0]))
        )
    assert len(views.pooled_sentence_features(pair["a"])) == 2
    assert len(views.pooled_sentence_features(pair["a"])[0]) == 132
    assert set(views.ARMS) == {
        "response_set",
        "local_set",
        "augmented_set",
        "constrained_set",
        "augmented_mlp",
        "constrained_mlp",
        "local_logistic",
        "source_erased_constrained_set",
        "complete_static_constrained_set",
    }
    assert views.decision("augmented_set", 0.1, 0.9, 2.0)["raw_risk"] == 0.5
    assert views.decision("local_set", 0.1, 0.9, 2.0)["raw_risk"] == 0.1
    forced = views.decision("constrained_set", None, None, 2.0)
    assert forced["action"] == "escalate" and forced["probability_unsupported"] == 0.5
    assert forced["brier"] == forced["realized_cost"] == 0.25
    with pytest.raises(ValueError, match="arm"):
        views.decision("unregistered", 0.1, 0.9, 1.0)


def test_real_child_basetemp_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7756-TERMINAL: child setup and serialized view replay."""
    parent = tmp_path / "nested"
    parent.mkdir()
    target = parent / "child"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        f"--basetemp={target}",
        "--version",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0 and target.parent.is_dir()
    pair = views.prepare_views(b"A. B. C.", b"A.")
    path = tmp_path / "views.json"
    path.write_text(json.dumps(views.serialize_pair(pair)))
    reopened = views.deserialize_pair(json.loads(path.read_text()))
    assert reopened == pair
    assert math.isclose(
        sum(
            sum(x) for x in energy.distribution(reopened["b"], energy.zero_parameters())["joint"][0]
        ),
        1.0,
    )


def test_feature_average_and_input_guards() -> None:
    """SCENARIO-VERIFY-7756-DECISION: logistic averages features before its head."""
    pair = views.prepare_views(b"A 12. B 30. C 50.", b"A 12. B 31.")
    params = runtime.init_params("logistic_local", 67501)
    params["w"] = params["w"].at[128, 1].set(5.0)
    probability = views.logistic_averaged_feature_risk(params, pair)
    assert 0 < probability < 1
    assert views.decision("local_logistic", probability, None, 1.0, 1)["realized_cost"] is not None
    assert views.decision("local_set", 0.2, None, 1.0)["brier"] is None
    with pytest.raises(TypeError, match="bytes"):
        views.prepare_views("A.", b"A.")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="abstained"):
        views.pooled_sentence_features(views.prepare_views(b"A.", b"")["a"])
