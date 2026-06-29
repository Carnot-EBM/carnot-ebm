"""Tests for Exp 4963 .457 bank and pivot-turnkey audit.

Spec refs: REQ-CAPSTONE-4963, SCENARIO-CAPSTONE-4963,
SCENARIO-CAPSTONE-4963-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_4963_bank_and_pivot_audit as exp4963


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
LOADER_RELATIVE_PATH = "data/experiment_4922_travelplanner_structured_slice.jsonl"


def _registry_text(*, tr87: int = 6, s5i5: int = 2) -> str:
    return f"""schema_version: 1
updated: '2026-06-29'
reproducible_total_levels: 69
games:
- game: tr87
  reproducibility: reproduced
  levels_reproduced: {tr87}
  solver: GameAdapter _tr87 plus scripts/arc_loop_solve.py.
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: {s5i5}
  solver: GameAdapter _s5i5 plus scripts/arc_loop_solve.py.
"""


def _bank_artifact(
    *,
    game: str,
    exp_id: int,
    reached: int = 2,
    prior: int = 2,
    claimed: bool = False,
    provenance: str = "live_agent_self_discovery",
    live_path: bool = True,
    outer_loop: bool = False,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "experiment": f"experiment_{exp_id}_levelup_attempt",
        "honest_verdict": f"complete_{game}_no_new_level_residual_no_grounded_delta",
        "target_game": game,
        "solve_provenance": provenance,
        "offline_reproduced": False,
        "reproduced_levels": prior,
        "new_levels_banked": 0,
        "live_path_reachable": live_path,
        "standing_loop_result_path": f"results/arc_loop_solve_{game}.json",
        "standing_loop_ran": False,
        "reproduction_gate": {},
        "registry_update": {
            "target_game": game,
            "prior_game_levels": prior,
            "new_game_levels": reached,
            "banked_levels": 0,
            "reason": "no_grounded_delta",
        },
        "offline_ground_truth_bfs": outer_loop,
    }
    if claimed:
        artifact.update(
            {
                "honest_verdict": f"success_{game}_levelup_banked",
                "offline_reproduced": True,
                "reproduced_levels": reached,
                "new_levels_banked": max(1, reached - prior),
                "standing_loop_ran": True,
                "reproduction_gate": {
                    "game": game,
                    "claimed_level": reached,
                    "reached_level": reached,
                    "reproduced": True,
                    "mode": "offline_reproduction_gate_no_quota",
                },
                "registry_update": {
                    "target_game": game,
                    "prior_game_levels": prior,
                    "new_game_levels": reached,
                    "banked_levels": max(1, reached - prior),
                    "reason": "banked_offline_reproduced_level",
                },
            }
        )
    return artifact


def _loop_artifact(game: str, *, reached: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "mode": "standing_arc_loop_offline_no_quota",
        "reached_level": reached,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached if reproduced else 0,
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached,
            "reached_level": reached,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
    }


def _dry_run_rows(n: int = 3) -> list[dict[str, object]]:
    return [
        {
            "problem_id": f"tp4963-{idx}",
            "self_consistency": {
                "selected_candidate_id": f"sc-{idx}",
                "correct_by_cached_oracle": idx == 1,
            },
            "decomposed_energy_verifier": {
                "selected_candidate_id": f"energy-{idx}",
                "verifier_is_oracle": False,
                "analytical_penalty_source": {"verifier_id": "fover_production_ensemble"},
            },
            "oracle": {
                "selected_candidate_id": f"energy-{idx}",
                "oracle_used_for_correctness_only": True,
            },
        }
        for idx in range(n)
    ]


def _citations() -> dict[str, dict[str, object]]:
    return {
        arxiv_id: {
            "http_status": 200,
            "title": f"Paper {arxiv_id}",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        }
        for arxiv_id in exp4963.REQUIRED_ARXIV_IDS
    }


def _pivot_artifact(**overrides: object) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": "success_distributional_energy_verifier_pivot_turnkey_ready",
        "verifier_is_oracle": False,
        "moat_proven_claimed": False,
        "arxiv_ids_cited": sorted(exp4963.REQUIRED_ARXIV_IDS),
        "already_ingested_reconfirmed": ["2605.18871", "2504.16828", "2502.01989"],
        "citations": _citations(),
        "turnkey_spec": {
            "entrypoint_command": exp4963.PIVOT_ENTRYPOINT_COMMAND,
            "real_loader": LOADER_RELATIVE_PATH,
            "decomposed_energy_verifier_column": {
                "model_identity_features_allowed": False,
                "oracle_labels_allowed_in_verifier": False,
            },
            "oracle_column": "cached labels used only to score correctness",
        },
        "dry_run_three_columns": {
            "columns": ["self_consistency", "decomposed_energy_verifier", "oracle"],
            "n_rows": 3,
            "rows": _dry_run_rows(),
            "full_benchmark_run": False,
        },
        "pivot_turnkey": True,
        "three_column_dry_run_ok": True,
        "post_sprint_first_experiment_pointer": {
            "entrypoint_command": exp4963.PIVOT_ENTRYPOINT_COMMAND,
            "real_benchmark_executed_by_exp4962": False,
        },
        "validation_gate": {
            "beats_self_consistency_ci95_excludes_zero_required": True,
            "oracle_distinct_required": True,
            "no_model_identity_shortcut_required": True,
            "verifier_is_oracle_required_value": False,
            "claimed_met": False,
        },
        "preconditions_checked": {
            "domain_slice_present": True,
            "domain_slice_valid": True,
            "domain_slice_rows": 3,
            "self_consistency_saturated": False,
            "exp4922_harness_present": True,
            "fover_registry_present": True,
            "fover_active_ensemble_present": True,
            "blocked_resource": None,
            "real_benchmark_executed": False,
            "model_load": False,
            "training_launched": False,
            "scripts_research_conductor_modified": False,
        },
        "source_artifacts": {
            "domain_slice": LOADER_RELATIVE_PATH,
            "exp4922_harness": "python/carnot/experiment_4922_distributional_energy_verifier_scaffold.py",
            "fover_registry": "ops/verifier_registry.yaml",
        },
        "field_principles": {
            "verifier_is_oracle": {
                "principle": (
                    "false -- the DESIGN TARGET is oracle-distinct, not the "
                    "executable oracle that defines correctness."
                )
            }
        },
    }
    artifact.update(overrides)
    return artifact


def _write_inputs(root: Path, *, include_d: bool = True, registry_text: str | None = None) -> None:
    (root / "results").mkdir()
    (root / "ops").mkdir()
    (root / "scripts").mkdir()
    (root / "data").mkdir()
    (root / "openspec" / "capabilities" / "capstone").mkdir(parents=True)
    (root / "scripts" / "adversarial_verify.py").write_text("", encoding="utf-8")
    (root / "scripts" / "summarize_artifact.py").write_text("", encoding="utf-8")
    (root / "scripts" / "arc_orphan_solver_lint.py").write_text("", encoding="utf-8")
    (root / "openspec" / "capabilities" / "capstone" / "spec.md").write_text(
        "REQ-CAPSTONE-4963\nSCENARIO-CAPSTONE-4963\n",
        encoding="utf-8",
    )
    (root / exp4963.REGISTRY_RELATIVE_PATH).write_text(
        registry_text or _registry_text(),
        encoding="utf-8",
    )
    (root / LOADER_RELATIVE_PATH).write_text("{}", encoding="utf-8")
    for relative, payload in (
        (exp4963.A1_RELATIVE_PATH, _bank_artifact(game="tr87", exp_id=4958, reached=6, prior=6)),
        (exp4963.A2_RELATIVE_PATH, _bank_artifact(game="s5i5", exp_id=4959, reached=2, prior=2)),
        ("results/arc_loop_solve_tr87.json", _loop_artifact("tr87", reached=6)),
        ("results/arc_loop_solve_s5i5.json", _loop_artifact("s5i5", reached=2)),
    ):
        (root / relative).write_text(json.dumps(payload), encoding="utf-8")
    if include_d:
        (root / exp4963.PIVOT_RELATIVE_PATH).write_text(
            json.dumps(_pivot_artifact()),
            encoding="utf-8",
        )


def test_req_capstone_4963_spec_declares_audit_contract() -> None:
    """REQ-CAPSTONE-4963: OpenSpec names the .457 audit checks and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4963.SPEC_REFS:
        assert ref in spec
    for check in exp4963.CHECK_KEYS:
        assert check in spec
    for field, principle in exp4963.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec
    assert exp4963.RESULT_RELATIVE_PATH in spec
    assert "dry_run_three_columns" in spec
    assert "2508.16665" in spec


def test_scenario_capstone_4963_no_bank_claims_are_vacuously_trusted(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4963: honest A1/A2 no-bank dead ends are not failures."""

    _write_inputs(tmp_path)

    artifact = exp4963.run(
        root=tmp_path,
        write=True,
        lint_runner=lambda _root: {"passed": True, "command": "fixture lint"},
        now=lambda: 10.0,
    )
    written = json.loads((tmp_path / exp4963.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert artifact["honest_verdict"] == "complete_v457_banks_and_pivot_audited_trusted"
    assert artifact["checks"] == {key: True for key in exp4963.CHECK_KEYS}
    assert artifact["banks_trustworthy"] is True
    assert artifact["pivot_readiness_trustworthy"] is True
    assert artifact["audit_failure_reasons"] == []
    assert artifact["bank_evidence"]["A1"]["bank_claimed"] is False
    assert artifact["bank_evidence"]["A2"]["bank_claimed"] is False
    assert artifact["pivot_readiness_evidence"]["turnkey_wiring_evidence"]["loader_present"] is True
    assert artifact["duration_s"] == 1.0
    assert exp4963.artifact_schema_errors(artifact) == []


def test_req_capstone_4963_claimed_bank_checks_catch_adversarial_patterns() -> None:
    """REQ-CAPSTONE-4963: claimed banks must be genuine, new, self-discovered, and live-path."""

    registry = yaml.safe_load(_registry_text())
    clean = exp4963.audit_bank(
        label="A1",
        artifact=_bank_artifact(game="tr87", exp_id=4958, reached=7, prior=6, claimed=True),
        loop_artifact=_loop_artifact("tr87", reached=7),
        registry=registry,
        lint_result={"passed": True},
    )

    assert clean["bank_claimed"] is True
    assert clean["checks"] == {key: True for key in exp4963.BANK_CHECK_KEYS}
    assert clean["failure_reasons"] == []

    bad = exp4963.audit_bank(
        label="A2",
        artifact=_bank_artifact(
            game="s5i5",
            exp_id=4959,
            reached=2,
            prior=2,
            claimed=True,
            provenance="outer_loop_re",
            live_path=False,
            outer_loop=True,
        ),
        loop_artifact=_loop_artifact("s5i5", reached=1, reproduced=False),
        registry=registry,
        lint_result={"passed": False, "returncode": 1},
    )

    assert bad["checks"] == {
        "reproduction_genuine": False,
        "not_duplicate": False,
        "self_discovery_provenance": False,
        "live_path_reachable": False,
    }
    assert bad["failure_reasons"] == [
        "A2_reproduction_genuine_failed_loop_or_gate_mismatch_s5i5",
        "A2_not_duplicate_failed_duplicate_depth_s5i5_L2",
        "A2_self_discovery_provenance_failed_outer_loop_re",
        "A2_live_path_reachable_failed",
    ]


def test_scenario_capstone_4963_pivot_checks_oracle_distinct_and_turnkey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-4963: D readiness is gated on oracle-distinct turnkey wiring."""

    _write_inputs(tmp_path)
    clean = exp4963.audit_pivot_readiness(_pivot_artifact(), root=tmp_path)

    assert clean["checks"] == {"oracle_distinct_design": True, "honest_readiness": True}
    assert clean["failure_reasons"] == []
    assert clean["turnkey_wiring_evidence"]["dry_run_rows_have_required_columns"] is True

    bad = exp4963.audit_pivot_readiness(
        _pivot_artifact(
            verifier_is_oracle=True,
            moat_proven_claimed=True,
            arxiv_ids_cited=["2605.18871"],
            citations={"2605.18871": {"http_status": 404, "title": "", "url": "bad"}},
            turnkey_spec={
                "entrypoint_command": "wrong",
                "real_loader": "missing.jsonl",
                "decomposed_energy_verifier_column": {
                    "model_identity_features_allowed": True,
                    "oracle_labels_allowed_in_verifier": True,
                },
                "matm_similarity_retrieval": {"proposed": True},
            },
            dry_run_three_columns={
                "columns": ["self_consistency"],
                "n_rows": 0,
                "rows": [],
                "full_benchmark_run": True,
            },
            pivot_turnkey=False,
            three_column_dry_run_ok=False,
            post_sprint_first_experiment_pointer={"entrypoint_command": "wrong"},
            validation_gate={
                "beats_self_consistency_ci95_excludes_zero_required": False,
                "oracle_distinct_required": False,
                "no_model_identity_shortcut_required": False,
                "verifier_is_oracle_required_value": True,
                "claimed_met": True,
            },
            field_principles={"verifier_is_oracle": {"principle": "oracle"}},
        ),
        root=tmp_path,
    )

    assert bad["checks"] == {"oracle_distinct_design": False, "honest_readiness": False}
    assert bad["failure_reasons"] == [
        "D_oracle_distinct_design_failed_verifier_is_oracle_not_false",
        "D_oracle_distinct_design_failed_design_target_not_declared",
        "D_oracle_distinct_design_failed_model_identity_shortcut_allowed",
        "D_oracle_distinct_design_failed_oracle_labels_allowed",
        "D_honest_readiness_failed_arxiv_ids_not_exact",
        "D_honest_readiness_failed_citation_metadata_not_real",
        "D_honest_readiness_failed_validation_gate_not_precise",
        "D_honest_readiness_failed_pivot_turnkey_wiring_not_genuine",
        "D_honest_readiness_failed_moat_proven_claimed",
        "D_honest_readiness_failed_matm_reproposed",
    ]

    monkeypatch.setattr(
        exp4963,
        "_critical_circular_moat_flags",
        lambda _artifact: [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}],
    )
    circular = exp4963.audit_pivot_readiness(_pivot_artifact(), root=tmp_path)
    assert circular["checks"]["oracle_distinct_design"] is False
    assert "D_oracle_distinct_design_failed_circular_moat_overclaim" in circular[
        "failure_reasons"
    ]


def test_scenario_capstone_4963_missing_d_blocks_but_audits_present_banks(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4963-BLOCKED-PRECONDITION: missing D is recorded."""

    _write_inputs(tmp_path, include_d=False)

    artifact = exp4963.run(
        root=tmp_path,
        write=False,
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 20.0,
    )

    assert artifact["honest_verdict"] == (
        "blocked_experiment_4962_distributional_energy_verifier_turnkey_missing"
    )
    assert artifact["preconditions_checked"]["d_artifact_present"] is False
    assert artifact["checks"] == {
        "reproduction_genuine": True,
        "not_duplicate": True,
        "self_discovery_provenance": True,
        "live_path_reachable": True,
        "oracle_distinct_design": False,
        "honest_readiness": False,
    }
    assert artifact["banks_trustworthy"] is True
    assert artifact["pivot_readiness_trustworthy"] is False
    assert "D_missing_experiment_4962_distributional_energy_verifier_turnkey" in artifact[
        "audit_failure_reasons"
    ]
    assert artifact["pivot_readiness_evidence"]["present"] is False
    assert exp4963.artifact_schema_errors(artifact) == []


def test_req_capstone_4963_schema_and_blocked_registry_paths(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4963: malformed registries and malformed artifacts fail closed."""

    _write_inputs(tmp_path)
    (tmp_path / exp4963.REGISTRY_RELATIVE_PATH).write_text("not: [valid", encoding="utf-8")

    blocked = exp4963.run(
        root=tmp_path,
        write=False,
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 30.0,
    )
    assert blocked["honest_verdict"] == "blocked_arc_solve_registry_unloadable"
    assert blocked["preconditions_checked"]["registry_loadable"] is False
    assert blocked["banks_trustworthy"] is False
    assert blocked["pivot_readiness_trustworthy"] is True

    valid = exp4963.run(
        root=tmp_path,
        write=False,
        registry_loader=lambda _root: yaml.safe_load(_registry_text()),
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 40.0,
    )
    bad = copy.deepcopy(valid)
    bad["banks_trustworthy"] = "yes"
    bad["pivot_readiness_trustworthy"] = "yes"
    bad["checks"] = {"reproduction_genuine": True}
    bad["audit_failure_reasons"] = "none"
    bad["inference_substrate"] = "bad"
    bad["field_principles"] = {}
    bad["preconditions_checked"] = []
    bad["reproducibility_checksum"] = "bad"
    bad.pop("honest_verdict")

    errors = exp4963.artifact_schema_errors(bad)

    assert "missing required field honest_verdict" in errors
    assert "banks_trustworthy must be bare bool" in errors
    assert "pivot_readiness_trustworthy must be bare bool" in errors
    assert "checks must contain the six required bare booleans" in errors
    assert "audit_failure_reasons must be a list" in errors
    assert "inference_substrate mismatch" in errors
    assert "field_principles mismatch" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    mismatch = copy.deepcopy(valid)
    mismatch["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    assert "reproducibility_checksum mismatch" in exp4963.artifact_schema_errors(mismatch)

    with_errors = copy.deepcopy(valid)
    with_errors["schema_errors"] = ["stale"]
    assert "schema_errors must be empty" in exp4963.artifact_schema_errors(with_errors)

    out = exp4963.write_artifact(valid, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == valid
    with pytest.raises(ValueError, match="banks_trustworthy must be bare bool"):
        exp4963.write_artifact({**valid, "banks_trustworthy": "bad"}, root=tmp_path)


def test_req_capstone_4963_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4963: helper paths keep malformed evidence untrusted."""

    _write_inputs(tmp_path)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        exp4963._read_json(bad_json)

    (tmp_path / exp4963.REGISTRY_RELATIVE_PATH).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="registry did not contain a mapping"):
        exp4963._load_registry(tmp_path)

    assert exp4963._mapping([]) == {}
    assert exp4963._int_value("bad", default=7) == 7
    assert exp4963._load_json_if_present(tmp_path, "missing.json") is None
    assert exp4963._citation_metadata_real(_pivot_artifact(citations={})) is False
    assert exp4963._citation_metadata_real(
        _pivot_artifact(
            citations={
                arxiv_id: {"http_status": 200, "title": "x", "url": "bad"}
                for arxiv_id in exp4963.REQUIRED_ARXIV_IDS
            }
        )
    ) is False
    assert exp4963._citation_metadata_real(
        _pivot_artifact(
            citations={
                arxiv_id: {
                    "http_status": 200,
                    "title": "",
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                }
                for arxiv_id in exp4963.REQUIRED_ARXIV_IDS
            }
        )
    ) is False
    assert exp4963._validation_gate_precise(_pivot_artifact(validation_gate={})) is False
    assert exp4963._contains_matm_reproposal("retired MATM null") is False
    assert exp4963._contains_matm_reproposal("MATM similarity-keyed retrieval proposed") is True
    assert exp4963._artifact_path_exists(tmp_path, "") is False
    assert exp4963._artifact_path_exists(tmp_path, str(tmp_path / LOADER_RELATIVE_PATH)) is True
    assert exp4963._turnkey_rows_have_required_columns(
        {
            "columns": ["self_consistency", "decomposed_energy_verifier", "oracle"],
            "n_rows": 3,
            "rows": _dry_run_rows(),
            "full_benchmark_run": True,
        }
    ) is False
    assert exp4963._turnkey_rows_have_required_columns(
        {
            "columns": ["self_consistency", "decomposed_energy_verifier", "oracle"],
            "n_rows": 3,
            "rows": [object(), *_dry_run_rows()],
            "full_benchmark_run": False,
        }
    ) is False
    missing_column_rows = _dry_run_rows()
    missing_column_rows[0].pop("oracle")
    assert exp4963._turnkey_rows_have_required_columns(
        {
            "columns": ["self_consistency", "decomposed_energy_verifier", "oracle"],
            "n_rows": 3,
            "rows": missing_column_rows,
            "full_benchmark_run": False,
        }
    ) is False
    verifier_oracle_rows = _dry_run_rows()
    verifier_oracle_rows[0]["decomposed_energy_verifier"]["verifier_is_oracle"] = True
    assert exp4963._turnkey_rows_have_required_columns(
        {
            "columns": ["self_consistency", "decomposed_energy_verifier", "oracle"],
            "n_rows": 3,
            "rows": verifier_oracle_rows,
            "full_benchmark_run": False,
        }
    ) is False
    bad_oracle_rows = _dry_run_rows()
    bad_oracle_rows[0]["oracle"]["oracle_used_for_correctness_only"] = False
    assert exp4963._turnkey_rows_have_required_columns(
        {
            "columns": ["self_consistency", "decomposed_energy_verifier", "oracle"],
            "n_rows": 3,
            "rows": bad_oracle_rows,
            "full_benchmark_run": False,
        }
    ) is False
    assert exp4963._turnkey_rows_have_required_columns(
        {
            "columns": ["self_consistency", "decomposed_energy_verifier", "oracle"],
            "n_rows": 1,
            "rows": [
                {
                    "self_consistency": {},
                    "decomposed_energy_verifier": {"verifier_is_oracle": False},
                    "oracle": {"oracle_used_for_correctness_only": True},
                }
            ],
            "full_benchmark_run": False,
        }
    ) is False

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    checked, registry_none = exp4963._preconditions(empty_root, None)
    assert registry_none is None
    assert checked["registry_loadable"] is False
    assert "arc_solve_registry" in checked["absent_inputs"]

    checked_bad_loader, registry_bad_loader = exp4963._preconditions(
        tmp_path,
        lambda _root: [],
    )
    assert registry_bad_loader is None
    assert checked_bad_loader["registry_loadable"] is False

    all_ok = {
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "d_artifact_present": True,
        "registry_present": True,
        "registry_loadable": True,
        "adversarial_verify_present": True,
        "summarize_artifact_present": True,
        "arc_orphan_solver_lint_present": True,
        "spec_has_req_4963": True,
    }
    assert exp4963._blocked_verdict(all_ok) is None
    for key, expected in (
        ("a1_artifact_present", "blocked_experiment_4958_levelup_attempt_missing"),
        ("a2_artifact_present", "blocked_experiment_4959_levelup_attempt_missing"),
        ("d_artifact_present", "blocked_experiment_4962_distributional_energy_verifier_turnkey_missing"),
        ("adversarial_verify_present", "blocked_scripts_adversarial_verify_missing"),
        ("summarize_artifact_present", "blocked_scripts_summarize_artifact_missing"),
        ("arc_orphan_solver_lint_present", "blocked_scripts_arc_orphan_solver_lint_missing"),
        ("spec_has_req_4963", "blocked_capstone_spec_req_4963_missing"),
    ):
        assert exp4963._blocked_verdict({**all_ok, key: False}) == expected

    missing_bank = exp4963._missing_bank_evidence("A1", "A1_missing")
    assert missing_bank["failure_reasons"] == ["A1_missing"]
    missing_pivot = exp4963._missing_pivot_evidence("D_missing")
    assert missing_pivot["checks"] == {"oracle_distinct_design": False, "honest_readiness": False}

    no_loop = exp4963.audit_bank(
        label="A1",
        artifact=_bank_artifact(game="tr87", exp_id=4958, reached=7, prior=6, claimed=True),
        loop_artifact=None,
        registry=yaml.safe_load(_registry_text()),
        lint_result={"passed": True},
    )
    assert "A1_missing_loop_artifact" in no_loop["failure_reasons"]
    assert exp4963._aggregate_checks(
        {"A1": no_loop},
        {"checks": {"oracle_distinct_design": True, "honest_readiness": True}},
    )["reproduction_genuine"] is False
    assert exp4963._load_loop_for_bank(tmp_path, {}) is None

    missing_root = tmp_path / "missing_inputs"
    missing_root.mkdir()
    _write_inputs(missing_root)
    (missing_root / exp4963.A1_RELATIVE_PATH).unlink()
    missing = exp4963.run(
        root=missing_root,
        write=False,
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 50.0,
    )
    assert missing["bank_evidence"]["A1"]["failure_reasons"] == [
        "A1_missing_experiment_4958_levelup_attempt"
    ]

    bad_schema = exp4963._with_checksum_and_schema({"experiment": "wrong"})
    assert bad_schema["schema_errors"]
    assert bad_schema["reproducibility_checksum"].startswith("sha256:")
