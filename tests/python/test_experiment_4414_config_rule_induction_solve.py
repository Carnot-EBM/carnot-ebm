"""Tests for Exp 4414 config-rule induction/solve reporting.

Spec refs: REQ-REPORT-4414, SCENARIO-REPORT-4414.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from carnot import experiment_4414_config_rule_induction_solve as mod


def _seed_repo(root: Path) -> None:
    for game in ("ka59", "bp35", "dc22"):
        env = root / "environment_files" / game / "envhash"
        env.mkdir(parents=True)
        (env / f"{game}.py").write_text("# offline env\n", encoding="utf-8")
        (env / "metadata.json").write_text("{}", encoding="utf-8")
    (root / "results" / "arc_config_layerb").mkdir(parents=True)
    (root / "results" / "arc_config_layerb" / "ka59_scaffolded_is_win.py").write_text(
        "import numpy as np\n\n"
        "def is_win(grid):\n"
        "    e = grid[63:64, 26:64]\n"
        "    return bool(np.sum(e == 4) == 32)\n",
        encoding="utf-8",
    )
    (root / "results" / "arc3_config_layerb_scaffolded_ka59.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete_scaffolded_tier2_GROUNDED_relational_rule",
                "verification": {
                    "fires_on_win": True,
                    "false_positive_rate": 0.0,
                    "n_nonwin": 6,
                },
                "rule_grounded": True,
                "literal_hardcode": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "ops").mkdir()
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        "schema_version: 1\n"
        "reproducible_total_levels: 34\n"
        "games:\n"
        "  - game: ka59\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 1\n"
        "  - game: bp35\n"
        "    reproducibility: unsolved\n"
        "    levels_reproduced: 0\n"
        "  - game: dc22\n"
        "    reproducibility: unsolved\n"
        "    levels_reproduced: 0\n",
        encoding="utf-8",
    )


def test_scenario_report_4414_model_unavailable_blocks_without_fabrication(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4414: no local Gemma blocks fresh induction, not ka59 reuse."""

    _seed_repo(tmp_path)

    artifact_path = mod.run(
        tmp_path,
        targets=("ka59", "bp35", "dc22"),
        model_probe=lambda _root: mod.ModelProbe(
            cached=False,
            server_started=False,
            status="blocked_local_model_unavailable",
            model_path=None,
            port=8920,
        ),
        now=lambda: 12.5,
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "complete_config_rule_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["local_model_server"]["status"] == (
        "blocked_local_model_unavailable"
    )
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64

    scorecards = {row["game"]: row for row in artifact["per_target_scorecard"]}
    assert scorecards["ka59"]["grounding_tier"] == 2
    assert scorecards["ka59"]["win_rule_predicate"] == "editable_count_4_equals_reference_count_4_32"
    assert scorecards["ka59"]["fires_on_win"] is True
    assert scorecards["ka59"]["false_positive_rate"] == 0.0
    assert scorecards["ka59"]["offline_reproduced"] is False
    assert scorecards["ka59"]["search_blocker"] == "no_registered_next_level_config_adapter"
    assert scorecards["bp35"]["honest_verdict"] == "blocked_local_model_unavailable"
    assert scorecards["dc22"]["honest_verdict"] == "blocked_local_model_unavailable"

    assert artifact["config_win_rules_grounded"] == [
        {
            "game": "ka59",
            "tier": 2,
            "predicate": "editable_count_4_equals_reference_count_4_32",
            "fires_on_win": True,
            "false_positive_rate": 0.0,
            "literal_hardcode": False,
        }
    ]
    assert "results/arc_config_layerb/ka59_scaffolded_is_win.py" in artifact["world_model_paths"]
    assert artifact["model_specs"]["proposer_status"] == "blocked_local_model_unavailable"


def test_req_report_4414_schema_validation_catches_bad_artifact() -> None:
    """REQ-REPORT-4414: required fields and bare monotonic counters are enforced."""

    artifact = {
        "honest_verdict": "complete_config_rule_partial",
        "per_target_scorecard": [],
        "reproducible_total_levels": "34",
        "new_levels_reproduced": 0,
        "config_win_rules_grounded": [],
        "world_model_paths": [],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4414,
        "reproducibility_checksum": "x",
        "model_specs": {},
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "reproducible_total_levels must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors


def test_req_report_4414_results_runner_imports_module(tmp_path: Path) -> None:
    """REQ-REPORT-4414: the operator-requested results runner delegates to the module."""

    runner_path = Path(__file__).resolve().parents[2] / "results" / (
        "experiment_4414_config_rule_induction_solve.py"
    )
    spec = importlib.util.spec_from_file_location("exp4414_runner", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp4414_runner"] = module
    spec.loader.exec_module(module)

    assert module.main(tmp_path) == 0
    assert (tmp_path / "results" / "experiment_4414_config_rule_induction_solve.json").exists()
