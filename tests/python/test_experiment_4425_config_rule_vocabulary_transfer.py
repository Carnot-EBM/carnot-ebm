"""Tests for Exp 4425 Qwen config-rule vocabulary transfer.

Spec refs: REQ-LEARN-4425, SCENARIO-LEARN-4425,
SCENARIO-LEARN-4425-NULL.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from carnot import experiment_4425_config_rule_vocabulary_transfer as mod


def _seed_rule_sources(root: Path) -> None:
    (root / "ops").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        "schema_version: 1\n"
        "games:\n"
        "  - game: ka59\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 1\n"
        "    win_condition: 'editable count equals reference count; match reference count'\n"
        "  - game: tr87\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 1\n"
        "    win_condition: 'glyph rewrite sequence follows LHS RHS map rules'\n"
        "  - game: sc25\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 1\n"
        "    win_condition: 'cast grid 3x3 cross shape pattern match'\n"
        "  - game: s5i5\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 1\n"
        "    win_condition: 'progress fill extends controlled markers until target marker coverage'\n"
        "  - game: blocked\n"
        "    reproducibility: unsolved\n",
        encoding="utf-8",
    )
    (root / "results" / "experiment_4414_config_rule_induction_solve.json").write_text(
        json.dumps(
            {
                "config_win_rules_grounded": [
                    {
                        "game": "ka59",
                        "tier": 2,
                        "predicate": "editable_count_4_equals_reference_count_4_32",
                        "false_positive_rate": 0.0,
                    },
                    {
                        "game": "ignored",
                        "tier": 0,
                        "predicate": "ungrounded",
                        "false_positive_rate": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "results" / "experiment_4421_config_rule_solve_unseen.json").write_text(
        json.dumps(
            {
                "target_game": "s5i5",
                "offline_reproduced": True,
                "grounded_win_condition": {
                    "predicate": (
                        "all target marker coordinates are occupied by controlled marker "
                        "coordinates"
                    )
                },
            }
        ),
        encoding="utf-8",
    )


def _repeat_payload(per_game_runs: dict[str, list[bool]]) -> dict[str, object]:
    return {
        "experiment": "arc3_layerb_repeat_bench",
        "model_key": "qwen3.5-9b-mtp",
        "mtp": True,
        "repeat": 4,
        "n_predict": 2560,
        "per_game": {
            game: {
                "runs": [
                    {"seed": seed, "grounded": grounded, "tokens": 100 + seed}
                    for seed, grounded in enumerate(runs)
                ]
            }
            for game, runs in per_game_runs.items()
        },
    }


def test_scenario_learn_4425_prompt_vocabulary_and_positive_transfer(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4425: seeded grounding lift with CI excluding zero is true."""

    _seed_rule_sources(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        cold_repeat_bench=_repeat_payload(
            {"bp35": [False, False, False, False], "dc22": [False, False, False, False]}
        ),
        seeded_repeat_bench=_repeat_payload(
            {"bp35": [True, True, True, True], "dc22": [True, True, True, True]}
        ),
        started_at=1.0,
        ended_at=2.25,
    )

    names = [primitive["name"] for primitive in artifact["config_rule_vocabulary"]]
    sources = mod.extract_grounded_rule_sources(tmp_path)

    assert any(source["source"] == mod.EXP4414_RELATIVE_PATH for source in sources)
    assert "editable_count==reference_count" in names
    assert "match-reference" in names
    assert "progress-fill" in names
    assert "glyph-rewrite" in names
    assert "marker-coverage" in names
    assert artifact["vocabulary_seeded_prompt"].index("RELATIONAL WIN-RULE VOCABULARY") < (
        artifact["vocabulary_seeded_prompt"].index("You are inducing")
    )
    assert "Generator: Qwen3.5-9B-MTP, iGPU, /no_think, MTP, four seeds" in (
        artifact["vocabulary_seeded_prompt"]
    )
    assert artifact["config_rule_vocabulary_transfers"] is True
    assert artifact["overall_grounding_rate_lift"] == 1.0
    assert artifact["overall_lift_ci95"] == [1.0, 1.0]
    assert {row["held_out_game"] for row in artifact["transfer_learning_curve"]} == {
        "bp35",
        "dc22",
    }
    assert all(row["seed_count"] == 4 for row in artifact["transfer_learning_curve"])
    assert artifact["verifier_is_oracle"] is False
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["model_specs"]["generator"] == "Qwen3.5-9B-MTP"
    assert artifact["model_specs"]["no_think"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_learn_4425_null_seeded_arm_logs_gap(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4425-NULL: missing seeded arm is logged, not gated."""

    _seed_rule_sources(tmp_path)
    cold_path = tmp_path / mod.COLD_REPEAT_BENCH_RELATIVE_PATH
    cold_path.parent.mkdir(parents=True, exist_ok=True)
    cold_path.write_text(
        json.dumps(
            _repeat_payload(
                {"ka59": [True, True, False, True], "tn36": [False, True, True, False]}
            )
        ),
        encoding="utf-8",
    )

    output_path = mod.run(tmp_path, now=lambda: 10.0)
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == (
        "complete: null_config_rule_vocabulary_transfer_seeded_arm_missing"
    )
    assert artifact["config_rule_vocabulary_transfers"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert "missing_vocabulary_seeded_repeat_bench" in artifact["logged_gaps"]
    assert artifact["transfer_learning_curve"][0]["seeded_grounding_rate"] is None
    assert artifact["transfer_learning_curve"][0]["lift"] is None
    assert artifact["model_specs"]["repeat_bench"] == "arc3_layerb_repeat_bench.py"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_learn_4425_schema_validation_and_runner(tmp_path: Path) -> None:
    """REQ-LEARN-4425: schema validation rejects non-bare and non-terminal fields."""

    artifact = {
        "honest_verdict": "null",
        "config_rule_vocabulary_transfers": "false",
        "config_rule_vocabulary": [],
        "vocabulary_seeded_prompt": 1,
        "transfer_learning_curve": {},
        "verifier_is_oracle": True,
        "random_seed": "4425",
        "logged_gaps": "gap",
        "model_specs": [],
        "reproducibility_checksum": "bad",
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "config_rule_vocabulary_transfers must be bare bool" in errors
    assert "vocabulary_seeded_prompt must be str" in errors
    assert "transfer_learning_curve must be list" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "random_seed must be bare int" in errors
    assert "logged_gaps must be list" in errors
    assert "model_specs must be dict" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors

    _seed_rule_sources(tmp_path)
    cold_path = tmp_path / mod.COLD_REPEAT_BENCH_RELATIVE_PATH
    cold_path.parent.mkdir(parents=True, exist_ok=True)
    cold_path.write_text(json.dumps(_repeat_payload({"ka59": [True, False, True, False]})))

    runner_path = Path(__file__).resolve().parents[2] / "results" / (
        "experiment_4425_config_rule_vocabulary_transfer.py"
    )
    spec = importlib.util.spec_from_file_location("exp4425_runner", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp4425_runner"] = module
    spec.loader.exec_module(module)

    assert module.main(tmp_path) == 0
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_learn_4425_defensive_helpers(tmp_path: Path) -> None:
    """REQ-LEARN-4425: defensive helpers keep null artifacts reproducible."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[1, 2, 3]", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    assert mod._load_registry(tmp_path) == {"games": []}
    assert mod._bootstrap_ci95([], seed=mod.RANDOM_SEED) == [0.0, 0.0]
    assert mod._bootstrap_ci95([0.25, 0.25], seed=mod.RANDOM_SEED) == [0.25, 0.25]
    varied = mod._bootstrap_ci95([0.0, 1.0], seed=mod.RANDOM_SEED, resamples=20)
    assert len(varied) == 2

    source = {"game": "custom", "rule_text": "mirror symmetry command object"}
    names = [primitive["name"] for primitive in mod.build_rule_vocabulary([source])]
    assert names == ["symmetry", "program-command-map"]
    assert mod._honest_verdict(False, True, 0.0) == (
        "complete: null_config_rule_vocabulary_transfer_lift_ci_includes_zero"
    )

    no_pair_artifact = mod.build_artifact(
        root=tmp_path,
        cold_repeat_bench={"per_game": {}},
        seeded_repeat_bench=_repeat_payload({"newgame": [True, True, True, True]}),
        started_at=0.0,
        ended_at=0.1,
    )
    assert no_pair_artifact["honest_verdict"] == (
        "complete: null_config_rule_vocabulary_transfer_no_paired_heldout"
    )
    assert "missing_cold_start_observations:newgame" in no_pair_artifact["logged_gaps"]
    assert "missing_config_rule_vocabulary" in no_pair_artifact["logged_gaps"]

    try:
        mod.write_artifact(tmp_path, {})
    except ValueError as exc:
        assert "missing honest_verdict" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("write_artifact should reject invalid artifacts")
