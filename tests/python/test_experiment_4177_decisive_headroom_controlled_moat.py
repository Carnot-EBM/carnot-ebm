"""Tests for Exp 4177 decisive headroom-controlled verifier moat test.

Spec refs: REQ-VERIFY-4177, SCENARIO-VERIFY-4177.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import decisive_headroom_controlled_moat_4177 as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_repo(tmp_path: Path, *, headroom: float = 0.5, domain: str = "code") -> Path:
    results = tmp_path / "results"
    source = results / "experiment_1999_code_verification_humaneval.json"
    _write_json(
        source,
        {
            "honest_verdict": "complete: fixture",
            "results": [
                {
                    "task_id": "HumanEval/0",
                    "baseline_passed": False,
                    "repair_passed": True,
                    "extracted_constraints": 2,
                },
                {
                    "task_id": "HumanEval/1",
                    "baseline_passed": False,
                    "repair_passed": True,
                    "extracted_constraints": 2,
                },
                {
                    "task_id": "HumanEval/2",
                    "baseline_passed": True,
                    "repair_passed": True,
                    "extracted_constraints": 0,
                },
                {
                    "task_id": "HumanEval/3",
                    "baseline_passed": False,
                    "repair_passed": False,
                    "extracted_constraints": 3,
                },
                {
                    "task_id": "HumanEval/4",
                    "baseline_passed": True,
                    "repair_passed": True,
                    "extracted_constraints": 0,
                },
                {
                    "task_id": "HumanEval/5",
                    "baseline_passed": False,
                    "repair_passed": True,
                    "extracted_constraints": 1,
                },
                {
                    "task_id": "HumanEval/6",
                    "baseline_passed": False,
                    "repair_passed": True,
                    "extracted_constraints": 2,
                },
                {
                    "task_id": "HumanEval/7",
                    "baseline_passed": True,
                    "repair_passed": True,
                    "extracted_constraints": 0,
                },
                {
                    "task_id": "HumanEval/8",
                    "baseline_passed": False,
                    "repair_passed": True,
                    "extracted_constraints": 2,
                },
                {
                    "task_id": "HumanEval/9",
                    "baseline_passed": True,
                    "repair_passed": True,
                    "extracted_constraints": 0,
                },
            ],
        },
    )
    _write_json(
        results / "experiment_4175_headroom_gate_executable_census.json",
        {
            "honest_verdict": "complete: headroom fixture",
            "headroom_present_domain": domain if headroom >= mod.HEADROOM_THRESHOLD else "",
            "max_selectable_headroom": headroom,
            "per_domain_headroom": {
                "code": {
                    "artifact_flags": {
                        "candidate_pool_detected": True,
                        "census_incomplete": False,
                        "source": str(source),
                    },
                    "oracle_at_k": 9 / 10,
                    "sc_vote_pass1": 4 / 10,
                    "selectable_headroom": headroom,
                }
            },
        },
    )
    _write_json(
        results / "experiment_4176_vstar_selector_model.json",
        {
            "accepted_rejected_n": {"accepted": 7, "rejected": 5, "total": 12},
            "coefficients": [2.0, 0.0, 0.0, 0.0],
            "feature_names": list(mod.FEATURE_NAMES),
            "intercept": 0.0,
            "model_type": "logistic_regression",
            "random_seed": 4176,
            "reproducibility_checksum": "selector-fixture",
            "spec_refs": ["REQ-VERIFY-4176", "SCENARIO-VERIFY-4176"],
        },
    )
    _write_json(
        results / "experiment_4176_vstar_learned_selector.json",
        {
            "honest_verdict": "complete: selector fixture",
            "domain": "code",
            "selector_path": str(results / "experiment_4176_vstar_selector_model.json"),
            "reproducibility_checksum": "selector-fixture",
        },
    )
    return tmp_path


def test_req_4177_spec_declares_decisive_moat_contract() -> None:
    """REQ-VERIFY-4177: OpenSpec declares the runner, fields, and principles."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4177",
        "SCENARIO-VERIFY-4177",
        "python/carnot/reporting/decisive_headroom_controlled_moat_4177.py",
        "results/experiment_4177_decisive_headroom_controlled_moat_test.py",
        "results/experiment_4177_decisive_headroom_controlled_moat_test.json",
        "verifier_value_added",
        "moat_delta_vs_vote",
        "moat_vs_matched_control",
        "accuracy_cost_pareto",
        "positive_control_confirmed",
        "arXiv:2511.02886",
        "arXiv:2504.01005",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4177_headroom_present_measures_three_arms(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4177: verifier, SC-vote, and matched control are compared."""

    root = _make_repo(tmp_path)

    artifact = mod.run(
        root,
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=400,
        oracle_checker=lambda: (True, "subprocess import ok"),
        adversarial_runner=lambda _path: {"flagged": [], "returncode": 0},
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["positive_control_confirmed"] is True
    assert artifact["verifier_value_added"] is True
    assert artifact["moat_delta_vs_vote"]["arm_a_pass1"] == pytest.approx(9 / 10)
    assert artifact["moat_delta_vs_vote"]["arm_b_sc_vote_pass1"] == pytest.approx(4 / 10)
    assert artifact["moat_delta_vs_vote"]["delta"] == pytest.approx(0.5)
    assert artifact["moat_delta_vs_vote"]["ci95"][0] > 0.0
    assert artifact["moat_vs_matched_control"]["arm_c_no_verifier_pass1"] == pytest.approx(4 / 10)
    assert artifact["moat_vs_matched_control"]["delta"] == pytest.approx(0.5)
    assert artifact["accuracy_cost_pareto"]["same_candidate_budget"] is True
    assert artifact["accuracy_cost_pareto"]["efficiency_parity"] is False
    assert artifact["accuracy_cost_pareto"]["value_added_basis"] == "accuracy_lift_ci95_excludes_zero"
    assert artifact["positive_control"]["oracle_at_k"] == pytest.approx(9 / 10)
    assert artifact["positive_control"]["selection_flips_vs_vote"] >= 1
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == ["REQ-VERIFY-4177", "SCENARIO-VERIFY-4177"]
    assert artifact["inference_substrate"] == "deterministic_verifier_plus_replay"
    assert artifact["adversarial_verify"]["returncode"] == 0

    written = json.loads(
        (root / "results" / "experiment_4177_decisive_headroom_controlled_moat_test.json").read_text(
            encoding="utf-8"
        )
    )
    assert written == artifact


def test_scenario_4177_defers_when_headroom_gate_is_absent(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4177: no headroom writes a complete gated deferral."""

    artifact = mod.run(
        _make_repo(tmp_path, headroom=0.05),
        oracle_checker=lambda: pytest.fail("oracle checker should not run after no-headroom gate"),
        adversarial_runner=lambda _path: {"flagged": [], "returncode": 0},
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete_moat_test_deferred_no_headroom_present"
    assert artifact["positive_control_confirmed"] is False
    assert artifact["verifier_value_added"] is False
    assert artifact["moat_delta_vs_vote"]["status"] == "deferred_no_headroom_present"
    assert artifact["accuracy_cost_pareto"]["status"] == "deferred_no_headroom_present"
    assert artifact["acceptance_gate"] is True


def test_req_4177_blocked_and_schema_paths_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4177: malformed inputs and invalid artifacts fail loudly."""

    assert mod._finite_float(True, 7.0) == 7.0
    assert mod._finite_float("bad", 3.0) == 3.0
    assert mod._bootstrap_ci([], random_seed=mod.RANDOM_SEED, resamples=10) == [0.0, 0.0]
    oracle_ok, oracle_detail = mod._default_oracle_checker()
    assert oracle_ok is True
    assert isinstance(oracle_detail, str)

    blocked = mod.run(
        _make_repo(tmp_path, domain="math"),
        oracle_checker=lambda: (True, "ok"),
        adversarial_runner=lambda _path: {"flagged": [], "returncode": 0},
    )
    assert blocked["honest_verdict"] == "blocked_unsupported_headroom_domain_math"
    assert blocked["acceptance_gate"] is False

    missing = mod.run(
        tmp_path / "missing",
        oracle_checker=lambda: (True, "ok"),
        adversarial_runner=lambda _path: {"flagged": [], "returncode": 0},
    )
    assert missing["honest_verdict"] == "blocked_missing_headroom_gate"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match="blocked_malformed_json_artifact"):
        mod._read_json_object(bad_json)
    with pytest.raises(mod.BlockedRun, match="blocked_missing_headroom_gate"):
        mod._load_corpus(tmp_path / "no-headroom")
    with pytest.raises(mod.BlockedRun, match="blocked_missing_vstar_selector"):
        mod._load_selector(tmp_path / "no-selector")

    selector_root = tmp_path / "bad-selector"
    _write_json(
        selector_root / "results" / "experiment_4176_vstar_selector_model.json",
        {"feature_names": ["wrong"]},
    )
    with pytest.raises(mod.BlockedRun, match="blocked_selector_feature_mismatch"):
        mod._load_selector(selector_root)

    oracle_blocked = mod.run(
        _make_repo(tmp_path / "oracle-blocked"),
        oracle_checker=lambda: (False, "subprocess import failed"),
        adversarial_runner=lambda _path: {"flagged": [], "returncode": 0},
    )
    assert oracle_blocked["honest_verdict"] == "blocked_executable_oracle_unavailable"

    adv_root = tmp_path / "adv"
    scripts = adv_root / "scripts"
    scripts.mkdir(parents=True)
    adv_script = scripts / "adversarial_verify.py"
    adv_script.write_text("print('not json')\n", encoding="utf-8")
    adv_report = mod._run_adversarial_verify(adv_root, adv_root / "artifact.json")
    assert adv_report["stdout"] == "not json\n"
    assert adv_report["returncode"] == 0

    json_adv_root = _make_repo(tmp_path / "json-adv")
    (json_adv_root / "scripts").mkdir(parents=True)
    (json_adv_root / "scripts" / "adversarial_verify.py").write_text(
        "import json\nprint(json.dumps({'flagged_count': 0, 'reports': []}))\n",
        encoding="utf-8",
    )
    json_adv = mod.run(json_adv_root, oracle_checker=lambda: (True, "ok"))
    assert json_adv["adversarial_verify"]["flagged_count"] == 0

    valid = mod._empty_artifact("blocked_fixture", "fixture", mod.RANDOM_SEED, 0.1)
    invalid_cases = [
        ({k: v for k, v in valid.items() if k != "honest_verdict"}, "missing required"),
        ({**valid, "honest_verdict": ""}, "terminal-prefixed"),
        ({**valid, "verifier_value_added": 1}, "bare bool"),
        ({**valid, "positive_control_confirmed": 0}, "bare bool"),
        ({**valid, "random_seed": True}, "bare int"),
        ({**valid, "reproducibility_checksum": ""}, "checksum"),
        ({**valid, "field_principles": {}}, "field_principles"),
        ({**valid, "spec_refs": []}, "spec_refs"),
        ({**valid, "inference_substrate": "live"}, "inference_substrate"),
        ({**valid, "moat_delta_vs_vote": []}, "moat_delta_vs_vote"),
        ({**valid, "moat_vs_matched_control": []}, "moat_vs_matched_control"),
        ({**valid, "accuracy_cost_pareto": []}, "accuracy_cost_pareto"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
