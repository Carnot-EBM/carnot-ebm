"""Tests for Exp 5209's GAP-1 set-search holdout hardening.

Spec refs: REQ-VERIFY-5209, SCENARIO-VERIFY-5209.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import arc_gap1_autopyverifier_pilot as pilot
from carnot.verify import arc_gap1_set_search_holdout_hardening as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _gold_grid() -> list[list[int]]:
    return [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]


def _same_directional_distractor() -> list[list[int]]:
    return [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]


def _fixture_pool(task_id: str) -> pilot.TaskPool:
    gold = _gold_grid()
    same_directional = _same_directional_distractor()
    return pilot.TaskPool(
        task_id=task_id,
        train_pairs=({"input": gold, "output": gold},),
        test_input=gold,
        candidates=(
            pilot.CandidateGrid("a_same_directional", "same_directional", same_directional, False),
            pilot.CandidateGrid("b_same_directional", "same_directional", same_directional, False),
            pilot.CandidateGrid("c_transposed_gold", "transposed_gold", pilot.transpose_grid(gold), False),
            pilot.CandidateGrid("z_gold", "gold", gold, True),
        ),
    )


def _fixture_pools(n: int = 10) -> list[pilot.TaskPool]:
    return [_fixture_pool(f"fixture-{idx:02d}:0") for idx in range(n)]


def _write_exp5205_artifacts(root: Path) -> None:
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (root / pilot.SOURCE_ARTIFACT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "experiment": "arc_grid_verifier_invariants_v2_combined",
                "no_test_gold_leak": True,
                "no_llm_used": True,
                "no_induction": True,
            }
        ),
        encoding="utf-8",
    )
    (root / pilot.RESULT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "experiment": pilot.EXPERIMENT,
                "best_subset_found": {"value": list(mod.EXP5205_BEST_SUBSET)},
                "candidate_discriminators_authored": {
                    "value": pilot.candidate_discriminator_metadata()
                },
                "inference_substrate": {"value": pilot.INFERENCE_SUBSTRATE},
                "source_context": {
                    "source_artifact": pilot.SOURCE_ARTIFACT_RELATIVE_PATH,
                    "square_transpose_subset_n": 239,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_verifier_gaps(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / pilot.VERIFIER_GAPS_RELATIVE_PATH).write_text(
        "### GAP-1: transpose / orientation discrimination\n"
        "- status: open -- prior baseline refuted\n"
        "<!-- experiment_5205_autopyverifier_gap1_pilot_v476 start -->\n"
        "- exp5205 old line\n"
        "<!-- experiment_5205_autopyverifier_gap1_pilot_v476 end -->\n"
        "\n### GAP-2: next\n",
        encoding="utf-8",
    )


def test_req_verify_5209_spec_declares_holdout_hardening_contract() -> None:
    """REQ-VERIFY-5209: OpenSpec declares the exp5210 gate fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5209") :]

    for marker in (
        "REQ-VERIFY-5209",
        "SCENARIO-VERIFY-5209",
        mod.RESULT_RELATIVE_PATH,
        "gap1_hardened_positive",
        "heldout_pass_at_2_mean",
        "baseline_always_on_pass_at_2_mean",
        "single_refuted_directional_pass_at_2_mean",
        "paired_delta_ci95",
        "leakage_audit_passed",
        "best_subset_stable",
    ):
        assert marker in section


def test_scenario_verify_5209_repeated_grouped_splits_write_gate_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5209: repeated grouped splits produce held-out gate fields."""

    _write_exp5205_artifacts(tmp_path)
    _write_verifier_gaps(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        pools=_fixture_pools(),
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        n_grouped_splits=20,
        duration_s=0.25,
        update_gap_doc=True,
    )

    assert artifact["gap1_hardened_positive"]["value"] is True
    assert artifact["heldout_pass_at_2_mean"]["value"] == pytest.approx(1.0)
    assert artifact["baseline_always_on_pass_at_2_mean"]["value"] == pytest.approx(0.0)
    assert artifact["single_refuted_directional_pass_at_2_mean"]["value"] == pytest.approx(0.0)
    assert artifact["delta_over_always_on"]["value"] == pytest.approx(1.0)
    assert artifact["delta_over_single_refuted"]["value"] == pytest.approx(1.0)
    assert artifact["paired_delta_ci95"]["value"] == "[1.000000, 1.000000]"
    assert artifact["n_grouped_splits"]["value"] == 20
    assert artifact["leakage_audit_passed"]["value"] is True
    assert artifact["best_subset_stable"]["value"] is True
    assert artifact["ops_verifier_gaps_updated"]["value"] is True
    assert artifact["inference_substrate"]["value"] == pilot.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "set_search_remains_positive_after_hardening" in artifact["honest_verdict"]["value"]
    assert artifact["candidate_discriminator_matrix"]["row_count"] == 40
    assert artifact["candidate_discriminator_matrix"]["columns"][:3] == [
        "object_count",
        "palette_histogram_shape",
        "__always_on__",
    ]
    assert artifact["candidate_discriminator_matrix"]["exp5205_best_subset"] == list(
        mod.EXP5205_BEST_SUBSET
    )
    assert len(artifact["split_details"]) == 20
    assert all(
        set(row["train_groups"]).isdisjoint(row["heldout_groups"])
        for row in artifact["split_details"]
    )
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    mod.update_verifier_gap_doc(tmp_path, artifact)
    gap_doc = (tmp_path / pilot.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "experiment_5209" in gap_doc
    assert gap_doc.count("experiment_5209 GAP-1 set-search holdout hardening") == 1
    assert "Do not promote" in gap_doc


def test_req_verify_5209_leakage_and_matrix_guards_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5209: leakage and artifact reconstruction guards reject unsafe inputs."""

    _write_exp5205_artifacts(tmp_path)
    pools = _fixture_pools()

    with pytest.raises(ValueError, match="at least two task-id groups"):
        mod.grouped_split([_fixture_pool("only-one-group:0")], seed=1)

    bad_split = mod.GroupedSplit(
        split_index=0,
        seed=1,
        train_groups=("dup",),
        heldout_groups=("dup",),
        train_pools=(pools[0],),
        heldout_pools=(pools[1],),
    )
    assert any(
        "duplicate task ids across train/eval" in error
        for error in mod.leakage_audit_errors(
            [bad_split],
            source_artifact={"no_test_gold_leak": True},
        )
    )
    assert any(
        "source artifact does not assert no_test_gold_leak=true" in error
        for error in mod.leakage_audit_errors(
            [bad_split],
            source_artifact={"no_test_gold_leak": False},
        )
    )

    exp5205_path = tmp_path / pilot.RESULT_RELATIVE_PATH
    exp5205 = json.loads(exp5205_path.read_text(encoding="utf-8"))
    exp5205["candidate_discriminators_authored"]["value"] = [
        {"name": "not_from_exp5205", "description": "bad", "transpose_sensitive": True}
    ]
    exp5205_path.write_text(json.dumps(exp5205), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate discriminator mismatch"):
        mod.reconstruct_exp5205_candidate_matrix(pools, root=tmp_path)

    assert mod.payload_checksum({"reproducibility_checksum": ""}).startswith("sha256:")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {key: value for key, value in artifact.items() if key != "gap1_hardened_positive"},
            "missing required fields",
        ),
        (
            lambda artifact: artifact
            | {
                "n_grouped_splits": {
                    "value": 19,
                    "principle": mod.FIELD_PRINCIPLES["n_grouped_splits"],
                }
            },
            "n_grouped_splits",
        ),
        (
            lambda artifact: artifact
            | {
                "leakage_audit_passed": {
                    "value": False,
                    "principle": mod.FIELD_PRINCIPLES["leakage_audit_passed"],
                }
            },
            "leakage_audit_passed",
        ),
        (
            lambda artifact: artifact
            | {
                "paired_delta_ci95": {
                    "value": "bad",
                    "principle": mod.FIELD_PRINCIPLES["paired_delta_ci95"],
                }
            },
            "paired_delta_ci95",
        ),
        (
            lambda artifact: artifact
            | {
                "inference_substrate": {
                    "value": "live_llm",
                    "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                }
            },
            "inference_substrate",
        ),
        (
            lambda artifact: artifact
            | {
                "random_seed": {
                    "value": 520900,
                    "principle": "wrong principle",
                }
            },
            "principle mismatch",
        ),
        (
            lambda artifact: artifact
            | {
                "honest_verdict": {
                    "value": "done",
                    "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                }
            },
            "honest_verdict",
        ),
        (
            lambda artifact: artifact
            | {
                "reproducibility_checksum": {
                    "value": "bad",
                    "principle": mod.FIELD_PRINCIPLES["reproducibility_checksum"],
                }
            },
            "reproducibility_checksum must be sha256",
        ),
        (
            lambda artifact: artifact
            | {
                "reproducibility_checksum": {
                    "value": "sha256:bad",
                    "principle": mod.FIELD_PRINCIPLES["reproducibility_checksum"],
                }
            },
            "reproducibility_checksum mismatch",
        ),
    ],
)
def test_req_verify_5209_schema_rejects_bad_artifacts(tmp_path: Path, mutate, message: str) -> None:
    """REQ-VERIFY-5209: malformed exp5210 gate artifacts fail closed."""

    _write_exp5205_artifacts(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        pools=_fixture_pools(),
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        n_grouped_splits=20,
        update_gap_doc=False,
    )

    errors = mod.artifact_schema_errors(mutate(artifact))

    assert any(message in error for error in errors)


def test_req_verify_5209_build_artifact_raises_on_schema_error(tmp_path: Path, monkeypatch) -> None:
    """REQ-VERIFY-5209: build-time schema validation blocks malformed artifacts."""

    _write_exp5205_artifacts(tmp_path)
    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced schema failure"])

    with pytest.raises(ValueError, match="forced schema failure"):
        mod.build_artifact(_fixture_pools(), root=tmp_path, n_grouped_splits=20)
