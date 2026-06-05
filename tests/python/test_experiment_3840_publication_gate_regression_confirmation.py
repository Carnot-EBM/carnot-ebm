"""Tests for Exp 3840 publication-gate regression confirmation.

Spec refs: REQ-PUBLISH-3840, SCENARIO-PUBLISH-3840.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from carnot.reporting import publication_gate_regression_3840 as exp3840


SPEC_PATH = Path("openspec/capabilities/publication/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate_data(*, unmet: tuple[str, ...] = ()) -> dict[str, object]:
    gates: dict[str, dict[str, object]] = {
        "G1": {
            "pass": "G1" not in unmet,
            "detail": "headline measured",
            "source": "experiment_2850_fover_dual_condition_integrity_v4.json",
        },
        "G2": {"pass": "G2" not in unmet, "detail": "external reproducer"},
        "G3": {"pass": "G3" not in unmet, "detail": "narrowing clean", "hits": []},
        "G4": {
            "pass": "G4" not in unmet,
            "detail": "seed/checksum present",
            "source": "experiment_2850_fover_dual_condition_integrity_v4.json",
        },
    }
    return {
        "paper_ready": not unmet,
        "gates": gates,
        "unmet_gates": list(unmet),
        "note": "Stable 4-condition gate",
    }


def _seed_repo(root: Path) -> dict[Path, str]:
    gate_script = root / "scripts" / "publication_gate.py"
    gate_script.parent.mkdir(parents=True, exist_ok=True)
    gate_payload = json.dumps(_gate_data(), sort_keys=True)
    gate_script.write_text(
        "import json\n"
        f"GATE = json.loads({gate_payload!r})\n"
        "if __name__ == '__main__':\n"
        "    print(json.dumps(GATE))\n",
        encoding="utf-8",
    )

    summarizer = root / "scripts" / "summarize_artifact.py"
    summarizer.write_text(
        "import sys\n"
        "exp = sys.argv[1]\n"
        "print('ARTIFACT experiment_' + exp)\n"
        "if exp == '3836':\n"
        "    print('flagged_adversarial (stamped): None   |   LIVE re-check: CRITICAL')\n"
        "    sys.exit(2)\n"
        "print('flagged_adversarial (stamped): None   |   LIVE re-check: clean')\n",
        encoding="utf-8",
    )

    _write_json(
        root / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json",
        {
            "condition_a_production_auroc_mean": 0.9131335999999999,
            "n_seeds": 5,
            "random_seed": 42,
            "random_seeds_used": [42, 137, 271, 314, 1729],
            "reproducibility_checksum": "headline-checksum",
        },
    )
    for exp_id, suffix in (
        (3835, "formal_core_5seed_ci"),
        (3836, "formal_core_certified_abstention_operating_point"),
        (3837, "fover_error_category_learned_contribution"),
        (3838, "tier4_adaptive_structure"),
    ):
        _write_json(
            root / "results" / f"experiment_{exp_id}_{suffix}.json",
            {
                "experiment": exp_id,
                "honest_verdict": f"complete: exp{exp_id}",
                "flagged_adversarial": None,
                "random_seed": exp_id,
                "reproducibility_checksum": f"checksum-{exp_id}",
            },
        )

    files = {
        root / "ops" / "north-star.md": "fixed gate definitions\n",
        root / "scripts" / "research_conductor.py": "conductor unchanged\n",
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return files


def _gate_run(data: dict[str, object], *, ok: bool = True) -> dict[str, object]:
    return {
        "ok": ok,
        "blocked_resource": "" if ok else "publication_gate_json",
        "command": [sys.executable, "scripts/publication_gate.py", "--json"],
        "returncode": 0 if ok else 3,
        "stdout": json.dumps(data, sort_keys=True) if ok else "",
        "stderr": "" if ok else "failed",
        "data": data if ok else {},
    }


def _spot_checks(*, gate_feeder: bool = False) -> dict[str, object]:
    return {
        "ok": True,
        "blocked_resource": "",
        "checked_experiments": [3835, 3836, 3837, 3838],
        "artifacts": {
            "3835": [
                {
                    "path": "results/experiment_3835_formal_core_5seed_ci.json",
                    "stamped_flagged_adversarial": gate_feeder,
                    "feeds_publication_gate": gate_feeder,
                }
            ],
            "3836": [],
            "3837": [],
            "3838": [],
        },
        "summaries": {},
        "flagged_adversarial_true_gate_feeders": (
            [{"experiment": 3835, "path": "results/experiment_3835_formal_core_5seed_ci.json"}]
            if gate_feeder
            else []
        ),
        "live_critical_artifacts": ["3836"],
    }


def test_req_publish_3840_spec_anchor_exists() -> None:
    """REQ-PUBLISH-3840: OpenSpec declares the v353 gate-regression contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PUBLISH-3840" in spec
    assert "SCENARIO-PUBLISH-3840" in spec
    assert exp3840.OUTPUT_REL_PATH.as_posix() in spec
    assert exp3840.TERMINAL_VERDICT in spec


def test_scenario_publish_3840_records_gate_and_spot_check_evidence(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3840: G1-G4 pass and .353 additions do not feed gates."""

    untouched = _seed_repo(tmp_path)

    out_path = exp3840.run(
        tmp_path,
        started_s=10.0,
        now_s=12.5,
        executable=sys.executable,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3840.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3840.TERMINAL_VERDICT
    assert set(exp3840.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3840.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["g1"] is True
    assert artifact["g2"] is True
    assert artifact["g3"] is True
    assert artifact["g4"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["frozen_fover_auroc"] == 0.9131
    assert artifact["frozen_fover_auroc_unchanged"] is True
    assert artifact["flagged_adversarial_true_gate_feeders"] == []
    assert artifact["spot_check_summaries"]["3836"]["returncode"] == 2
    assert artifact["spot_check_summaries"]["3836"]["live_recheck"] == "critical"
    assert artifact["live_critical_spot_check_experiments"] == ["3836"]
    assert artifact["random_seed"] == 3840
    assert artifact["duration_s"] == 2.5
    assert artifact["reproducibility_checksum"] == exp3840.payload_checksum(artifact)
    assert (
        artifact["field_provenance"]["g1"]["principle"]
        == "each gate boolean \u2014 the four conditions for paper_ready"
    )
    assert (
        artifact["field_provenance"]["paper_ready"]["principle"]
        == "G1^G2^G3^G4 \u2014 the standing convergence invariant, MUST be true"
    )

    for path, original in untouched.items():
        assert path.read_text(encoding="utf-8") == original


def test_req_publish_3840_regression_and_blocked_verdicts_are_explicit(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-3840: regressed and blocked outcomes use closed-set prefixes."""

    _seed_repo(tmp_path)
    regressed = exp3840.build_artifact(
        tmp_path,
        publication_gate_run=_gate_run(_gate_data(unmet=("G3",))),
        spot_check_result=_spot_checks(),
        started_s=1.0,
        now_s=1.5,
        executable=sys.executable,
    )
    exp3840.validate_artifact(regressed)
    assert regressed["honest_verdict"] == "complete: publication_gate_REGRESSION_DETECTED_unmet_G3"
    assert regressed["g3"] is False
    assert regressed["paper_ready"] is False

    blocked = exp3840.build_artifact(
        tmp_path,
        publication_gate_run=_gate_run(_gate_data(), ok=False),
        spot_check_result=_spot_checks(),
        started_s=2.0,
        now_s=2.5,
        executable=sys.executable,
    )
    exp3840.validate_artifact(blocked)
    assert blocked["honest_verdict"] == "blocked_publication_gate_json"
    assert blocked["publication_gate_json"] == {}

    blocked_without_injected_spot_check = exp3840.build_artifact(
        tmp_path,
        publication_gate_run=_gate_run(_gate_data(), ok=False),
        started_s=2.5,
        now_s=2.75,
        executable=sys.executable,
    )
    exp3840.validate_artifact(blocked_without_injected_spot_check)
    assert blocked_without_injected_spot_check["honest_verdict"] == "blocked_publication_gate_json"

    spot_check_blocked = exp3840.build_artifact(
        tmp_path,
        publication_gate_run=_gate_run(_gate_data()),
        spot_check_result={
            "ok": False,
            "blocked_resource": "summarize_artifact",
            "artifacts": {},
            "summaries": {},
            "flagged_adversarial_true_gate_feeders": [],
            "live_critical_artifacts": [],
        },
        started_s=2.75,
        now_s=3.0,
        executable=sys.executable,
    )
    exp3840.validate_artifact(spot_check_blocked)
    assert spot_check_blocked["honest_verdict"] == "blocked_summarize_artifact"

    flagged_gate_feeder = exp3840.build_artifact(
        tmp_path,
        publication_gate_run=_gate_run(_gate_data()),
        spot_check_result=_spot_checks(gate_feeder=True),
        started_s=3.0,
        now_s=3.5,
        executable=sys.executable,
    )
    exp3840.validate_artifact(flagged_gate_feeder)
    assert (
        flagged_gate_feeder["honest_verdict"]
        == "complete: publication_gate_REGRESSION_DETECTED_unmet_flagged_adversarial_gate_feeder"
    )
    assert exp3840._gate_bool({}, "G1") is False
    assert exp3840._unmet_gates({"gates": {"G1": {"pass": False}}}) == [
        "G1",
        "G2",
        "G3",
        "G4",
    ]
    assert exp3840._gate_source_names({}) == set()
    assert (
        exp3840._regression_verdict([], paper_ready=False)
        == "complete: publication_gate_REGRESSION_DETECTED_unmet_paper_ready"
    )

    _write_json(
        tmp_path / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json",
        {"condition_a_production_auroc_mean": 0.9},
    )
    fover_moved = exp3840.build_artifact(
        tmp_path,
        publication_gate_run=_gate_run(_gate_data()),
        spot_check_result=_spot_checks(),
        started_s=4.0,
        now_s=4.5,
        executable=sys.executable,
    )
    exp3840.validate_artifact(fover_moved)
    assert (
        fover_moved["honest_verdict"]
        == "complete: publication_gate_REGRESSION_DETECTED_unmet_frozen_fover_auroc"
    )


def test_req_publish_3840_command_preconditions_and_missing_resources(tmp_path: Path) -> None:
    """REQ-PUBLISH-3840: missing commands or artifacts produce blocked resources."""

    _seed_repo(tmp_path)
    gate_script = tmp_path / "scripts" / "publication_gate.py"

    missing_python = exp3840.run_publication_gate_json(tmp_path, executable=tmp_path / "missing-python")
    assert missing_python["ok"] is False
    assert missing_python["blocked_resource"] == "publication_gate_python"

    no_gate_root = tmp_path / "no-gate"
    no_gate_root.mkdir()
    missing_gate = exp3840.run_publication_gate_json(no_gate_root, executable=sys.executable)
    assert missing_gate["ok"] is False
    assert missing_gate["blocked_resource"] == "publication_gate_script"

    gate_script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    failed_gate = exp3840.run_publication_gate_json(tmp_path, executable=sys.executable)
    assert failed_gate["ok"] is False
    assert failed_gate["blocked_resource"] == "publication_gate_json"
    assert failed_gate["returncode"] == 3

    gate_script.write_text("print('not json')\n", encoding="utf-8")
    invalid_gate = exp3840.run_publication_gate_json(tmp_path, executable=sys.executable)
    assert invalid_gate["ok"] is False
    assert invalid_gate["data"] == {}

    gate_script.write_text("print('[]')\n", encoding="utf-8")
    non_dict_gate = exp3840.run_publication_gate_json(tmp_path, executable=sys.executable)
    assert non_dict_gate["ok"] is False
    assert non_dict_gate["data"] == {}

    flagged_path = tmp_path / "results" / "experiment_3835_formal_core_5seed_ci.json"
    flagged_payload = json.loads(flagged_path.read_text(encoding="utf-8"))
    flagged_payload["flagged_adversarial"] = True
    _write_json(flagged_path, flagged_payload)
    feeder_gate_data = _gate_data()
    gates = feeder_gate_data["gates"]
    assert isinstance(gates, dict)
    gates["G1"]["source"] = flagged_path.name
    feeder_spot_check = exp3840.spot_check_353_artifacts(
        tmp_path,
        executable=sys.executable,
        publication_gate_data=feeder_gate_data,
    )
    assert feeder_spot_check["ok"] is True
    assert feeder_spot_check["flagged_adversarial_true_gate_feeders"] == [
        {"experiment": 3835, "path": str(flagged_path)}
    ]

    missing_summarizer_python = exp3840.spot_check_353_artifacts(
        tmp_path,
        executable=tmp_path / "missing-python",
        publication_gate_data=_gate_data(),
    )
    assert missing_summarizer_python["ok"] is False
    assert missing_summarizer_python["blocked_resource"] == "summarizer_python"

    (tmp_path / "scripts" / "summarize_artifact.py").unlink()
    spot_checks = exp3840.spot_check_353_artifacts(
        tmp_path,
        executable=sys.executable,
        publication_gate_data=_gate_data(),
    )
    assert spot_checks["ok"] is False
    assert spot_checks["blocked_resource"] == "summarize_artifact"

    (tmp_path / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json").unlink()
    headline_missing = exp3840.build_artifact(
        tmp_path,
        publication_gate_run=_gate_run(_gate_data()),
        spot_check_result=_spot_checks(),
        started_s=4.0,
        now_s=4.5,
        executable=sys.executable,
    )
    exp3840.validate_artifact(headline_missing)
    assert headline_missing["honest_verdict"] == "blocked_frozen_fover_headline_source"
    assert headline_missing["frozen_fover_auroc"] is None
