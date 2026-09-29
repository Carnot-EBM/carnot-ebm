"""Private custody and numerical checks for REQ-VERIFY-7894-V685."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from carnot.verify import natural_training, training_runtime
from carnot.verify import energy_fit_7894 as fit
from scripts.experiments import experiment_7894_v685_energy_fit as cli


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    public = tmp_path / "public.jsonl"
    evaluator = tmp_path / "evaluator.jsonl"
    family = "private-family"
    source = "Lumen has 12 apples. Orion is present."
    answer = "Lumen has 13 apples."
    public.write_text(
        json.dumps(
            {
                "family_id": family,
                "source_bytes": source.encode().hex(),
                "answer_bytes": answer.encode().hex(),
            }
        )
        + "\n"
    )
    evaluator.write_text(
        json.dumps(
            {
                "family_id": family,
                "role": "fit",
                "human_label": 1,
                "observed": 1,
                "annotation_byte_offsets": [[0, len(answer)]],
                "response_id": "private",
                "label_scope": "response",
            }
        )
        + "\n"
    )
    artifact = {
        "experiment_id": 7892,
        "source_boundary_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "circular_positive",
        "public_shards": [
            {
                "path": str(public),
                "sha256": "sha256:" + hashlib.sha256(public.read_bytes()).hexdigest(),
            }
        ],
        "evaluator_shards": [
            {
                "path": str(evaluator),
                "sha256": "sha256:" + hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            }
        ],
    }
    upstream = tmp_path / "upstream.json"
    upstream.write_text(json.dumps(artifact))
    return upstream, artifact


def test_private_custody_and_mask(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7894-CUSTODY: raw labels never enter source features."""
    upstream, artifact = _fixture(tmp_path)
    records, sources = fit.load_records(upstream)
    assert len(records) == 1 and records[0]["known"] == [0]
    assert records[0]["source"] == b"Lumen has 12 apples. Orion is present."
    assert len(sources) == 3
    public = Path(artifact["public_shards"][0]["path"])
    public.write_text(public.read_text() + "\n")
    with pytest.raises(fit.InputBlocked, match="sha256"):
        fit.load_records(upstream)


def test_gate_missing_field_is_terminal_external_block(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7894-CUSTODY: name the failed upstream operand."""
    upstream, artifact = _fixture(tmp_path)
    del artifact["source_boundary_ready_score"]
    upstream.write_text(json.dumps(artifact))
    with pytest.raises(fit.InputBlocked) as error:
        fit.load_records(upstream)
    operand = error.value.operands[0]
    assert operand["upstream_id"] == "exp7892-source-boundary"
    assert operand["artifact_field"] == "source_boundary_ready_score"
    assert operand["observed"] is None


def test_unknown_targets_have_zero_gradient() -> None:
    """SCENARIO-VERIFY-7894-FIT: absent spans cannot become negative targets."""
    rows = [
        {
            "id": "fit-a",
            "group": "fit-a",
            "role": "fit",
            "source": b"Lumen has 12.",
            "answer": b"Lumen has 13.",
            "label": 1,
            "known": [-1],
        }
    ]
    batch, excluded = natural_training.prepare(rows, "local_set")
    assert excluded == []
    params = training_runtime.init_params("energy_local", 67801)
    assert float(training_runtime._local_loss(params, batch, "energy_local", "a")) == 0
    gradient = jax.grad(training_runtime._local_loss)(params, batch, "energy_local", "a")
    assert all(np.all(np.asarray(value) == 0) for value in gradient.values())


def test_small_current_fit_interface() -> None:
    """SCENARIO-VERIFY-7894-FIT: the callable current library seals a head."""
    train = [
        {
            "id": "fit",
            "group": "fit",
            "role": "fit",
            "source": b"Lumen has 12.",
            "answer": b"Lumen has 13.",
            "label": 1,
            "known": [0],
        }
    ]
    tune = [{**train[0], "id": "tune", "group": "tune", "role": "tune"}]
    head = natural_training.fit(train, tune, "constrained_set", 67801, 0.01, 1)
    assert head["parameter_count"] <= 4096
    assert head["temperature"] in training_runtime.temperature_grid()
    assert len(head["curve"]) == 1
    assert 0 <= natural_training.predict(head, tune)[0]["probability_unsupported"] <= 1


def test_batched_scoring_matches_current_predict() -> None:
    """SCENARIO-VERIFY-7894-FIT: one batched risk preserves the library decision."""
    train = [
        {
            "id": "fit",
            "group": "fit",
            "role": "fit",
            "source": b"Lumen has 12.",
            "answer": b"Lumen has 13.",
            "label": 1,
            "known": [0],
        }
    ]
    tune = [{**train[0], "id": "tune", "group": "tune", "role": "tune"}]
    head = natural_training.fit(train, tune, "constrained_set", 67801, 0.01, 1)
    expected = natural_training.predict(head, tune)[0]
    assert cli.predict_batch(head, tune)[0] == expected


def test_scoring_reuses_public_features_across_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7894-FIT: one role/view preparation serves both seeds."""
    records = [
        {
            "id": f"private-{role}",
            "group": f"private-{role}",
            "role": role,
            "source": b"Lumen has 12 apples.",
            "answer": b"Lumen has 13 apples.",
            "label": 1,
            "known": [0],
            "source_cluster_id": f"private-{role}",
        }
        for role in ("fit", "tune")
    ]
    head = {
        "params": training_runtime.init_params("energy_local", 67801),
        "arm": "energy_local",
        "view_arm": "local_set",
        "paired": False,
        "temperature": 1.0,
    }
    monkeypatch.setattr(training_runtime, "load", lambda _: head)
    real_prepare = natural_training.prepare
    calls: list[str] = []

    def counted_prepare(rows: list[dict[str, object]], arm: str):
        calls.append(arm)
        return real_prepare(rows, arm)

    monkeypatch.setattr(natural_training, "prepare", counted_prepare)
    manifest = [
        {"arm": "local_set", "seed": seed, "path": "private", "sha256": "private"}
        for seed in (67801, 67802)
    ]
    path, _ = cli.score(records, manifest, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 4
    assert calls == ["local_set", "local_set"]


def test_frozen_validation_keeps_repository_health_separate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7894-TERMINAL: broad health is not a required child."""
    manifest = json.loads(cli.freeze(tmp_path).read_text())
    names = {item["name"] for item in manifest["commands"]}
    assert "affected_unit" in names
    assert "full_pytest" not in names
    assert "tests/python" not in [arg for item in manifest["commands"] for arg in item["argv"]]
    spec = next(item for item in manifest["commands"] if item["name"] == "spec_coverage")
    assert str(cli.ROOT / "tests/python/test_energy_fit_7894.py") in spec["argv"]


def test_source_permutation_stays_local_for_odd_count() -> None:
    """SCENARIO-VERIFY-7894-FIT: the longest source never wraps to shortest."""
    records = [{"id": f"private-{length}", "source": b"x" * length} for length in (1, 2, 3, 4, 5)]
    ordered, donors = cli.length_matched_donors(records)
    assert all(row["id"] != donor["id"] for row, donor in zip(ordered, donors, strict=True))
    assert (
        max(
            abs(len(row["source"]) - len(donor["source"]))
            for row, donor in zip(ordered, donors, strict=True)
        )
        <= 2
    )


def test_corrected_run_retains_prior_required_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7894-TERMINAL: corrected scope preserves its old failure."""
    upstream = tmp_path / "upstream.json"
    upstream.write_text(json.dumps({"historical_required_failures": []}))
    output = tmp_path / "previous.json"
    output.write_text(
        json.dumps(
            {
                "historical_required_failures": [],
                "validation_receipts": [
                    {
                        "name": "spec_coverage",
                        "actual_exit": 1,
                        "passed": False,
                        "log_path": "private.log",
                        "log_sha256": "sha256:private",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(cli, "UPSTREAM", upstream)
    monkeypatch.setattr(cli, "OUTPUT", output)
    history = cli.base(upstream, [], [])["historical_required_failures"]
    assert history[0]["name"] == "spec_coverage"
    assert history[0]["log_sha256"] == "sha256:private"
