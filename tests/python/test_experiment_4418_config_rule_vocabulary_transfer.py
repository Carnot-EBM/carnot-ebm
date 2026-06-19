"""Tests for Exp 4418 config-rule vocabulary transfer.

Spec refs: REQ-LEARN-4418, SCENARIO-LEARN-4418,
SCENARIO-LEARN-4418-BLOCKED.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from carnot import experiment_4418_config_rule_vocabulary_transfer as mod


def _seed_repo(root: Path) -> None:
    (root / "ops").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        "schema_version: 1\n"
        "games:\n"
        "  - game: ka59\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 1\n"
        "    win_condition: 'editable count equals reference count'\n"
        "  - game: tr87\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 6\n"
        "    win_condition: 'editable glyph sequence equals greedy rewrite of target through LHS->RHS rules'\n"
        "  - game: tn36\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 7\n"
        "    win_condition: 'program command object must match target x y scale rotation and property attributes'\n"
        "  - game: sc25\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 1\n"
        "    win_condition: 'cast-grid 3x3 cross shape alignment fires the active spell'\n"
        "  - not-a-mapping\n"
        "  - game: bp35\n"
        "    reproducibility: unsolved\n"
        "    levels_reproduced: 0\n",
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
                        "fires_on_win": True,
                        "false_positive_rate": 0.0,
                        "literal_hardcode": False,
                    },
                    "not-a-rule",
                    {
                        "game": "dc22",
                        "tier": 0,
                        "predicate": "ungrounded",
                        "false_positive_rate": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _available_model(_root: Path) -> mod.ModelProbe:
    return mod.ModelProbe(
        available=True,
        status="ok",
        model="unsloth/gemma-4-12B-it-GGUF Q4",
        port=8920,
        endpoint="http://127.0.0.1:8920/v1/models",
    )


def _unavailable_model(_root: Path) -> mod.ModelProbe:
    return mod.ModelProbe(
        available=False,
        status="blocked_local_model_unavailable",
        model=None,
        port=8920,
        endpoint="http://127.0.0.1:8920/v1/models",
    )


def test_scenario_learn_4418_blocks_when_local_model_unavailable(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4418-BLOCKED: no iGPU inducer means no transfer claim."""

    _seed_repo(tmp_path)

    artifact_path = mod.run(
        tmp_path,
        model_probe=_unavailable_model,
        held_out_games=("ka59", "tr87", "bp35"),
        now=lambda: 10.0,
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "blocked_local_model_unavailable"
    assert artifact["config_rule_vocabulary_transfers"] is False
    assert artifact["transfer_learning_curve"] == []
    assert artifact["config_rule_vocabulary"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["preconditions_checked"]["grounded_rules"]["count"] == 4
    assert artifact["preconditions_checked"]["local_model_server"]["status"] == (
        "blocked_local_model_unavailable"
    )
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["model_specs"]["inducer_status"] == "blocked_local_model_unavailable"


def test_req_learn_4418_extracts_vocabulary_and_transfer_curve(tmp_path: Path) -> None:
    """REQ-LEARN-4418: seeded grounding beats cold-start only with CI95 excl-0."""

    _seed_repo(tmp_path)

    def arm_runner(
        game: str,
        arm: str,
        vocabulary: tuple[str, ...],
    ) -> mod.ArmResult:
        assert game in {"ka59", "tr87", "bp35"}
        assert "count-equality" in vocabulary
        grounded = arm == "vocabulary_seeded"
        return mod.ArmResult(
            grounded=grounded,
            tier=2 if grounded else 0,
            false_positive_rate=0.0,
            status="grounded" if grounded else "not_grounded",
        )

    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions=mod.check_preconditions(tmp_path, model_probe=_available_model),
        started_at=1.0,
        ended_at=2.5,
        held_out_games=("ka59", "tr87", "bp35"),
        arm_runner=arm_runner,
    )

    assert artifact["honest_verdict"] == "success_config_rule_vocabulary_transfers"
    assert artifact["config_rule_vocabulary_transfers"] is True
    assert artifact["config_rule_vocabulary"] == [
        "count-equality",
        "editable-reference-relation",
        "position-region-match",
        "attribute-match",
        "glyph-map",
        "sequence-rewrite",
        "shape-pattern-match",
        "program-command-map",
    ]
    assert artifact["overall_delta"] == 1.0
    assert artifact["overall_delta_ci95"] == [1.0, 1.0]
    assert len(artifact["transfer_learning_curve"]) == 3
    assert all(row["delta_ci95"] == [1.0, 1.0] for row in artifact["transfer_learning_curve"])
    assert artifact["verifier_is_oracle"] is False
    assert len(artifact["reproducibility_checksum"]) == 64

    errors = mod.artifact_schema_errors(artifact)
    assert errors == []


def test_req_learn_4418_schema_validation_and_runner(tmp_path: Path) -> None:
    """REQ-LEARN-4418: schema validation and results runner are wired."""

    artifact = {
        "honest_verdict": "success_config_rule_vocabulary_transfers",
        "config_rule_vocabulary_transfers": "true",
        "transfer_learning_curve": [],
        "config_rule_vocabulary": [],
        "verifier_is_oracle": True,
        "preconditions_checked": {},
        "random_seed": "4418",
        "reproducibility_checksum": "bad",
        "model_specs": {},
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "config_rule_vocabulary_transfers must be bare bool" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors

    _seed_repo(tmp_path)
    runner_path = Path(__file__).resolve().parents[2] / "results" / (
        "experiment_4418_config_rule_vocabulary_transfer.py"
    )
    spec = importlib.util.spec_from_file_location("exp4418_runner", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp4418_runner"] = module
    spec.loader.exec_module(module)

    assert module.main(tmp_path) == 0
    assert (tmp_path / "results" / "experiment_4418_config_rule_vocabulary_transfer.json").exists()


def test_req_learn_4418_defensive_paths_and_helpers(tmp_path: Path, monkeypatch) -> None:
    """REQ-LEARN-4418: helper branches keep blocked/null artifacts honest."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    assert mod._load_registry(tmp_path) == {"games": []}
    assert mod._model_names({"data": "not-a-list"}) == []
    assert mod._model_probe_from_payload(
        {"data": [{"id": "unsloth/gemma-4-12B-it-GGUF"}]}
    ).available
    monkeypatch.setattr(
        mod,
        "_fetch_model_payload",
        lambda: {"data": [{"id": "unsloth/gemma-4-12B-it-GGUF"}]},
    )
    assert mod.default_model_probe(tmp_path).status == "ok"

    assert mod._classify_primitives("custom", "mirror symmetry") == ["symmetry"]
    empty_rules: dict[str, dict] = {}
    mod._merge_rule(empty_rules, game="custom", source="x", rule_text="")
    assert empty_rules == {}
    assert mod._bootstrap_ci95([], seed=mod.RANDOM_SEED) == [0.0, 0.0]
    varied = mod._bootstrap_ci95([0.0, 1.0], seed=mod.RANDOM_SEED, resamples=20)
    assert len(varied) == 2
    assert mod._unwired_arm_runner("ka59", "cold_start", ()).status == (
        "blocked_live_inducer_runner_not_invoked"
    )

    insufficient = mod.build_artifact(
        root=tmp_path,
        preconditions={
            "grounded_rules": {"count": 0, "sources": []},
            "local_model_server": {"available": True, "status": "ok"},
            "trm_training_stood_down": True,
        },
        started_at=0.0,
        ended_at=0.0,
    )
    assert insufficient["honest_verdict"] == "blocked_insufficient_grounded_rules"

    _seed_repo(tmp_path)
    trm_preconditions = mod.check_preconditions(tmp_path, model_probe=_available_model)
    trm_preconditions["trm_training_stood_down"] = False
    trm_blocked = mod.build_artifact(
        root=tmp_path,
        preconditions=trm_preconditions,
        started_at=0.0,
        ended_at=0.0,
    )
    assert trm_blocked["honest_verdict"] == "blocked_trm_training_not_stood_down"

    null_path = mod.run(
        tmp_path,
        model_probe=_available_model,
        held_out_games=("ka59",),
        now=lambda: 1.0,
    )
    null_artifact = json.loads(null_path.read_text(encoding="utf-8"))
    assert null_artifact["honest_verdict"] == "complete_clean_null_config_rule_vocabulary_heterogeneous"

    bad_errors = mod.artifact_schema_errors({})
    assert "missing honest_verdict" in bad_errors
    assert "transfer_learning_curve must be list" in bad_errors
    assert "config_rule_vocabulary must be list[str]" in bad_errors
    assert "preconditions_checked must be dict" in bad_errors
    assert "model_specs must be dict" in bad_errors
    try:
        mod.write_artifact(tmp_path, {})
    except ValueError as exc:
        assert "missing honest_verdict" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("write_artifact should reject invalid artifacts")
