"""Tests for Exp 5205's AutoPyVerifier-inspired GAP-1 set-search pilot.

Spec refs: REQ-VERIFY-5205, SCENARIO-VERIFY-5205.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from carnot.verify import arc_gap1_autopyverifier_pilot as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _gold_grid() -> list[list[int]]:
    return [
        [1, 1, 0],
        [2, 0, 0],
        [2, 0, 0],
    ]


def _fixture_pool(task_id: str) -> mod.TaskPool:
    gold = _gold_grid()
    return mod.TaskPool(
        task_id=task_id,
        train_pairs=({"input": gold, "output": gold},),
        test_input=gold,
        candidates=(
            mod.CandidateGrid("a_transposed_gold", "transposed_gold", mod.transpose_grid(gold), False),
            mod.CandidateGrid("b_rotated_gold", "rotated_gold", mod.rotate_180_grid(gold), False),
            mod.CandidateGrid("z_gold", "gold", gold, True),
        ),
    )


def _fixture_pools(n: int = 6) -> list[mod.TaskPool]:
    return [_fixture_pool(f"fixture-{idx:02d}") for idx in range(n)]


def test_req_verify_5205_spec_declares_gap1_set_search_contract() -> None:
    """REQ-VERIFY-5205: OpenSpec declares required Exp 5205 fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5205") :]

    for marker in (
        "REQ-VERIFY-5205",
        "SCENARIO-VERIFY-5205",
        mod.RESULT_RELATIVE_PATH,
        "candidate_discriminators_authored",
        "best_subset_found",
        "pass_at_2_baseline_always_on_only",
        "pass_at_2_best_subset",
        "transpose_misvotes_captured",
        "verifier_is_oracle",
        "inference_substrate",
    ):
        assert marker in section


def test_req_verify_5205_candidate_library_is_transpose_sensitive() -> None:
    """REQ-VERIFY-5205: authored candidates include the refuted member but are a set."""

    metadata = mod.candidate_discriminator_metadata()
    names = {row["name"] for row in metadata}
    train_pairs = _fixture_pool("library").train_pairs
    gold = _gold_grid()
    transposed = mod.transpose_grid(gold)

    assert 5 <= len(metadata) <= 10
    assert "directional_adjacency_refuted_20260609" in names
    assert all(row["transpose_sensitive"] is True for row in metadata)
    assert mod.always_on_score(gold, train_pairs) == pytest.approx(
        mod.always_on_score(transposed, train_pairs)
    )
    gold_scores = mod.score_candidate_discriminators(gold, train_pairs)
    transposed_scores = mod.score_candidate_discriminators(transposed, train_pairs)
    assert any(gold_scores[name] < transposed_scores[name] for name in names)


def test_scenario_verify_5205_search_improves_fixture_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5205: subset search records a complete artifact."""

    (tmp_path / "ops").mkdir()
    (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text(
        "### GAP-1: transpose / orientation discrimination\n- status: open\n",
        encoding="utf-8",
    )
    artifact = mod.run(
        root=tmp_path,
        pools=_fixture_pools(),
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        random_seed=11,
        duration_s=1.25,
        update_gap_doc=True,
    )

    assert artifact["pass_at_2_baseline_always_on_only"]["value"] == pytest.approx(0.0)
    assert artifact["pass_at_2_best_subset"]["value"] == pytest.approx(1.0)
    assert artifact["transpose_misvotes_captured"]["value"] == "6 out of 6"
    assert artifact["best_subset_found"]["value"]
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["inference_substrate"]["value"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["reproducibility_checksum"]["value"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert "experiment_5205" in (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")


def test_req_verify_5205_reconstructs_square_transpose_subset_from_arc_files(tmp_path: Path) -> None:
    """REQ-VERIFY-5205: loader keeps only dimension-preserving transpose rows."""

    source = tmp_path / mod.SOURCE_ARTIFACT_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"experiment": "arc_grid_verifier_invariants_v2_combined"}), encoding="utf-8")
    arc_root = tmp_path / "arc"
    arc_root.mkdir()
    square = _gold_grid()
    nonsquare = [[1, 0, 0], [2, 0, 0]]
    challenges = {
        "square": {"train": [{"input": square, "output": square}], "test": [{"input": square}]},
        "wide": {"train": [{"input": nonsquare, "output": nonsquare}], "test": [{"input": nonsquare}]},
        "missing_solution": {
            "train": [{"input": square, "output": square}],
            "test": [{"input": square}],
        },
        "extra_test": {
            "train": [{"input": square, "output": square}],
            "test": [{"input": square}, {"input": square}],
        },
    }
    solutions = {"square": [square], "wide": [nonsquare], "extra_test": [nonsquare]}
    (arc_root / "arc-agi_training_challenges.json").write_text(
        json.dumps(challenges),
        encoding="utf-8",
    )
    (arc_root / "arc-agi_training_solutions.json").write_text(json.dumps(solutions), encoding="utf-8")

    pools = mod.load_square_transpose_subset(root=tmp_path, arc_root=arc_root, seed=0)

    assert [pool.task_id for pool in pools] == ["square:0"]
    assert any(candidate.kind == "gold" and candidate.correct for candidate in pools[0].candidates)
    assert any(candidate.kind == "transposed_gold" for candidate in pools[0].candidates)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {key: value for key, value in artifact.items() if key != "best_subset_found"},
            "missing required fields",
        ),
        (
            lambda artifact: artifact
            | {"verifier_is_oracle": {"value": True, "principle": mod.FIELD_PRINCIPLES["verifier_is_oracle"]}},
            "verifier_is_oracle",
        ),
        (
            lambda artifact: artifact | {"random_seed": 11},
            "principle-wrapped",
        ),
        (
            lambda artifact: artifact
            | {"honest_verdict": {"value": "done", "principle": mod.FIELD_PRINCIPLES["honest_verdict"]}},
            "honest_verdict",
        ),
        (
            lambda artifact: artifact
            | {
                "inference_substrate": {
                    "value": "live_llm_inference",
                    "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                }
            },
            "inference_substrate",
        ),
        (
            lambda artifact: artifact
            | {
                "random_seed": {
                    "value": 11,
                    "principle": "wrong principle",
                }
            },
            "principle mismatch",
        ),
        (
            lambda artifact: artifact
            | {
                "candidate_discriminators_authored": {
                    "value": "bad",
                    "principle": mod.FIELD_PRINCIPLES["candidate_discriminators_authored"],
                }
            },
            "candidate_discriminators_authored must list",
        ),
        (
            lambda artifact: artifact
            | {
                "candidate_discriminators_authored": {
                    "value": [
                        {"name": f"candidate_{idx}", "description": "x", "transpose_sensitive": True}
                        for idx in range(5)
                    ],
                    "principle": mod.FIELD_PRINCIPLES["candidate_discriminators_authored"],
                }
            },
            "refuted directional",
        ),
        (
            lambda artifact: artifact
            | {
                "pass_at_2_best_subset": {
                    "value": 1.5,
                    "principle": mod.FIELD_PRINCIPLES["pass_at_2_best_subset"],
                }
            },
            "pass_at_2_best_subset",
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
    ],
)
def test_req_verify_5205_schema_rejects_bad_artifacts(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-VERIFY-5205: malformed terminal artifacts fail closed."""

    artifact = mod.run(
        root=tmp_path,
        pools=_fixture_pools(),
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        random_seed=11,
        update_gap_doc=False,
    )

    errors = mod.artifact_schema_errors(mutate(artifact))

    assert any(message in error for error in errors)


def test_req_verify_5205_defensive_paths_are_covered(tmp_path: Path, monkeypatch) -> None:
    """REQ-VERIFY-5205: helper edge paths stay deterministic and covered."""

    discriminators = mod.default_discriminators()
    by_name = {row.name: row for row in discriminators}
    pools = _fixture_pools(1)
    artifact = mod.build_artifact(pools, root=tmp_path, random_seed=11)
    table = mod._score_table(pools, by_name)  # noqa: SLF001 - white-box coverage for deterministic helpers.

    assert mod._color_swap([[1]], random.Random(0)) is None  # noqa: SLF001
    assert mod._combined_score(_gold_grid(), pools[0].train_pairs, ("row_column_run_profile",), by_name) >= 0
    assert mod._pass_at_2([], (), by_name) == 0.0  # noqa: SLF001
    assert mod._split_pools(pools, seed=11) == (pools, pools)  # noqa: SLF001
    assert mod._transpose_capture(  # noqa: SLF001
        [
            mod.TaskPool(
                task_id="no-transpose",
                train_pairs=pools[0].train_pairs,
                test_input=_gold_grid(),
                candidates=(mod.CandidateGrid("z_gold", "gold", _gold_grid(), True),),
            )
        ],
        (),
        by_name,
    ) == (0, 0)
    assert mod._transpose_capture(pools, ("row_column_run_profile",), by_name)[1] == 1  # noqa: SLF001
    assert mod._value({"plain": 3}, "plain") == 3  # noqa: SLF001
    assert mod.payload_checksum({"reproducibility_checksum": ""}).startswith("sha256:")
    assert any(
        row["effect"] == "neutral"
        for row in mod._candidate_effects(  # noqa: SLF001
            pools,
            mod._pass_at_2(pools, ("row_column_run_profile",), by_name, table),  # noqa: SLF001
            by_name,
            table,
        )
    )
    assert all(
        row["effect"] == "hurt"
        for row in mod._candidate_effects(pools, 2.0, by_name, table)  # noqa: SLF001
    )

    with pytest.raises(FileNotFoundError):
        mod.load_square_transpose_subset(root=tmp_path, arc_root=tmp_path)

    mod.update_verifier_gap_doc(tmp_path / "missing-root", artifact)
    (tmp_path / "ops").mkdir(exist_ok=True)
    gap_path = tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH
    gap_path.write_text(
        "<!-- experiment_5205_autopyverifier_gap1_pilot_v476 start -->\n"
        "old\n"
        "<!-- experiment_5205_autopyverifier_gap1_pilot_v476 end -->\n",
        encoding="utf-8",
    )
    mod.update_verifier_gap_doc(tmp_path, artifact)
    assert "old" not in gap_path.read_text(encoding="utf-8")

    root_with_gap2 = tmp_path / "gap2-root"
    (root_with_gap2 / "ops").mkdir(parents=True)
    (root_with_gap2 / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text("intro\n### GAP-2: next\n", encoding="utf-8")
    mod.update_verifier_gap_doc(root_with_gap2, artifact)
    assert "experiment_5205" in (root_with_gap2 / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced schema failure"])
    with pytest.raises(ValueError, match="forced schema failure"):
        mod.build_artifact(pools, root=tmp_path, random_seed=11)
