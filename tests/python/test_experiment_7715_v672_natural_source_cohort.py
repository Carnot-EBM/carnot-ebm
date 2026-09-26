"""Natural RAGTruth cohort contracts: REQ-REPORT-7715, REQ-VERIFY-7715."""

from __future__ import annotations

import json
from pathlib import Path
import time

from copy import deepcopy

import pytest

from carnot.reporting import natural_source_cohort as cohort
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot import experiment_7715_v672_natural_source_cohort as runner


def source(number: int, text: str | None = None) -> dict:
    return {
        "source_id": f"source-{number}",
        "task_type": "Summary",
        "source_info": text or f"Original source family {number} has unique evidence.",
    }


def response(number: int, split: str = "train", source_number: int | None = None) -> dict:
    return {
        "id": f"response-{number}",
        "source_id": f"source-{source_number if source_number is not None else number}",
        "split": split,
        "quality": "good",
        "response": f"Unique answer {number}.",
        "labels": [],
        "model": "historical-generator-only",
    }


def fixture_rows() -> tuple[list[dict], list[dict]]:
    sources = [source(index) for index in range(4)]
    replies = [response(0), response(1), response(2), response(3, "test")]
    return sources, replies


def test_scenario_report_7715_custody_groups_sources_before_labels() -> None:
    # SCENARIO-REPORT-7715-CUSTODY: model IDs and labels cannot choose a response.
    sources, replies = fixture_rows()
    replies.append(response(9, source_number=1))
    families, excluded = cohort.build_public_families(sources, replies, "fixture")
    assert excluded == []
    assert len(families) == 4
    first = [row for row in families if "source-1" in row["instance_ids"]][0]
    assert first["member_count"] == 2
    replies[1]["labels"] = [{"start": 0, "end": 1, "text": "U", "label_type": "x"}]
    replies[1]["model"] = "changed-historical-generator"
    assert cohort.build_public_families(sources, replies, "fixture")[0] == families


def test_scenario_report_7715_custody_cross_split_and_exposure() -> None:
    # SCENARIO-REPORT-7715-CUSTODY: copied sources cannot cross roles.
    sources, replies = fixture_rows()
    sources.append(source(4, sources[0]["source_info"].swapcase()))
    replies.append(response(4, "test"))
    families, collisions = cohort.build_public_families(sources, replies, "fixture")
    assert len(collisions) == 1
    assert collisions[0]["reason"] == "cross_official_split_family"
    exposed = {"source-1"}
    planned = cohort.subtract_and_assign(families, exposed, {"fit": 2, "evaluation": 1}, "fixture")
    assert planned["excluded_prior_exposure_count"] == 1
    assert planned["shortages"]["train"] == 1
    assert planned["shortages"]["test"] == 0
    assert planned["selected"] == []
    assert planned["candidate_counts"]["train"] == 1


def test_scenario_verify_7715_isolation_seals_before_labels(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-7715-ISOLATION: public bytes precede annotation reads.
    sources, replies = fixture_rows()
    families, _ = cohort.build_public_families(sources, replies, "fixture")
    counts = {"fit": 2, "evaluation": 1}
    planned = cohort.subtract_and_assign(families, {"source-2"}, counts, "fixture")
    assert not any(planned["shortages"].values())
    seen: list[str] = []

    def label_reader(row: dict) -> int:
        assert (tmp_path / "public_manifest.json").is_file()
        seen.append(row["view"]["response_id"])
        return 1

    manifest = cohort.seal(
        tmp_path,
        planned["selected"],
        counts,
        "fixture",
        label_reader,
        {"files": [], "uncertainties": []},
        {"path": "windows.json", "sha256": "sha256:fixture"},
    )
    assert len(seen) == 3
    assert cohort.cold_reduce(tmp_path / "manifest.json", counts)["families"] == 3
    for role in counts:
        path = tmp_path / manifest["roles"][role]["public_path"]
        rows = cohort.predictor_inputs(path, role)
        assert all("label" not in row and "model" not in row for row in rows)
        assert all(row["source_sha256"] and row["response_sha256"] for row in rows)
        label_path = tmp_path / manifest["roles"][role]["evaluator_path"]
        assert label_path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(ValueError, match="public_role_path"):
            cohort.predictor_inputs(label_path, role)
        row = json.loads(path.read_text().splitlines()[0])
        row["label"] = 1
        path.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="public_hash_mismatch"):
            cohort.cold_reduce(tmp_path / "manifest.json", counts)
        break


def test_scenario_verify_7715_isolation_rejects_private_mutations(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-7715-ISOLATION: byte, role, path and duplicate mutations fail.
    sources, replies = fixture_rows()
    families, _ = cohort.build_public_families(sources, replies, "fixture")
    counts = {"fit": 2, "evaluation": 1}
    selected = cohort.subtract_and_assign(families, {"source-2"}, counts, "fixture")["selected"]
    with pytest.raises(ValueError, match="duplicate_family"):
        cohort.seal(tmp_path, [*selected, selected[0]], counts, "fixture", lambda _: 0, {}, {})
    manifest = cohort.seal(tmp_path, selected, counts, "fixture", lambda _: 0, {}, {})
    public = tmp_path / manifest["roles"]["fit"]["public_path"]
    rows = cohort.predictor_inputs(public, "fit")
    altered = dict(rows[0], complete_source="changed")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cohort.validate_predictor(altered, "fit")
    with pytest.raises(ValueError, match="role_mismatch"):
        cohort.validate_predictor(rows[0], "evaluation")
    with pytest.raises(ValueError, match="public_role_path"):
        cohort.predictor_inputs(tmp_path / "../fit_public.jsonl", "fit")


def test_scenario_report_7715_terminal_exact_shortage(tmp_path: Path) -> None:
    # SCENARIO-REPORT-7715-TERMINAL: V651 exposed families block 400 fresh roles.
    candidate = runner.build_candidate(runner.ROOT, tmp_path, "20260926")
    assert candidate["honest_verdict"].startswith("complete_blocked_")
    assert candidate["verdict_class"] == "blocked"
    assert candidate["natural_cohort_ready_score"] == 0
    assert candidate["fresh_source_score"] == 0
    assert candidate["MODEL_SPECS"] == []
    assert candidate["inference_substrate_class"] == "no_model_load"
    assert candidate["gate_check_summary"]
    for gate in candidate["gate_check_summary"]:
        assert {"check", "upstream", "path", "field", "operator", "expected", "observed"} <= set(
            gate
        )
    assert candidate["sample_size_budget"]["eligible_families"] < 400
    assert runner.cold_reduce(candidate, tmp_path)["verdict_class"] == "blocked"


def test_scenario_verify_7715_isolation_invalid_public_inputs() -> None:
    # REQ-VERIFY-7715: malformed public identities fail before labels.
    sources, replies = fixture_rows()
    with pytest.raises(ValueError, match="source_identity_invalid"):
        cohort.build_public_families([sources[0], sources[0]], replies, "fixture")
    with pytest.raises(ValueError, match="response_identity_invalid"):
        cohort.build_public_families(sources, [replies[0], replies[0]], "fixture")
    wrong_source = dict(replies[0], source_id="absent")
    with pytest.raises(ValueError, match="response_source_or_split_invalid"):
        cohort.build_public_families(sources, [wrong_source], "fixture")
    wrong_split = dict(replies[0], split="unknown")
    with pytest.raises(ValueError, match="response_source_or_split_invalid"):
        cohort.build_public_families(sources, [wrong_split], "fixture")
    empty_response = dict(replies[0], response="")
    with pytest.raises(ValueError, match="response_text_invalid"):
        cohort.build_public_families(sources, [empty_response], "fixture")
    poor = dict(replies[0], quality="truncated")
    assert cohort.build_public_families(sources, [poor], "fixture")[1] == [
        {"response_id": "response-0", "reason": "quality_excluded"}
    ]
    families, _ = cohort.build_public_families(sources, replies, "fixture")
    with pytest.raises(ValueError, match="duplicate_family"):
        cohort.subtract_and_assign([families[0], families[0]], set(), {}, "fixture")


def sealed_fixture(path: Path) -> tuple[dict, dict]:
    sources, replies = fixture_rows()
    families, _ = cohort.build_public_families(sources, replies, "fixture")
    counts = {"fit": 2, "evaluation": 1}
    selected = cohort.subtract_and_assign(families, {"source-2"}, counts, "fixture")["selected"]
    manifest = cohort.seal(path, selected, counts, "fixture", lambda _: 1, {}, {})
    return manifest, counts


def _update_public_hash(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["public_manifest_sha256"] = sha256_file(path / "public_manifest.json")
    atomic_json(path / "manifest.json", manifest)


def _rewrite_info(path: Path, manifest: dict, field: str, value: object) -> None:
    manifest["roles"]["fit"][field] = value
    public = json.loads((path / "public_manifest.json").read_text())
    public["roles"]["fit"][field] = value
    atomic_json(path / "public_manifest.json", public)
    atomic_json(path / "manifest.json", manifest)
    _update_public_hash(path)


def _rewrite_role(path: Path, manifest: dict, role: str, kind: str, rows: list[dict]) -> None:
    info = manifest["roles"][role]
    target = path / info[f"{kind}_path"]
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    info[f"{kind}_sha256"] = sha256_file(target)
    if kind == "public":
        public = json.loads((path / "public_manifest.json").read_text())
        public["roles"][role]["public_sha256"] = info["public_sha256"]
        atomic_json(path / "public_manifest.json", public)
    atomic_json(path / "manifest.json", manifest)
    if kind == "public":
        _update_public_hash(path)


def test_scenario_verify_7715_isolation_rejects_schema_and_label_errors(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-7715-ISOLATION: all public and evaluator identities bind.
    manifest, counts = sealed_fixture(tmp_path / "good")
    fit_path = tmp_path / "good" / manifest["roles"]["fit"]["public_path"]
    row = cohort.predictor_inputs(fit_path, "fit")[0]
    with pytest.raises(ValueError, match="public_field_mismatch"):
        cohort.validate_predictor(dict(row, label=1), "fit")
    with pytest.raises(ValueError, match="split_mismatch"):
        cohort.validate_predictor(dict(row, official_split="test"), "fit")
    with pytest.raises(ValueError, match="response_hash_mismatch"):
        cohort.validate_predictor(dict(row, complete_response="changed"), "fit")
    families, _ = cohort.build_public_families(*fixture_rows(), "fixture")
    selected = cohort.subtract_and_assign(families, {"source-2"}, counts, "fixture")["selected"]
    with pytest.raises(ValueError, match="role_count_mismatch"):
        cohort.seal(tmp_path / "few", selected[:-1], counts, "fixture", lambda _: 0, {}, {})
    wrong_role = [dict(item) for item in selected]
    wrong_role[0]["role"] = "evaluation"
    with pytest.raises(ValueError, match="role_count_mismatch"):
        cohort.seal(tmp_path / "wrong", wrong_role, counts, "fixture", lambda _: 0, {}, {})
    with pytest.raises(ValueError, match="human_label_invalid"):
        cohort.seal(tmp_path / "bad_label", selected, counts, "fixture", lambda _: 2, {}, {})

    cases = (
        (
            "public_manifest_hash_mismatch",
            lambda p, m: (p / "public_manifest.json").write_text("{}"),
        ),
        (
            "public_manifest_mismatch",
            lambda p, m: atomic_json(p / "public_manifest.json", {"roles": {}, "counts": counts}),
        ),
        (
            "evaluator_hash_mismatch",
            lambda p, m: (p / m["roles"]["fit"]["evaluator_path"]).write_text("{}"),
        ),
        ("role_count_mismatch", lambda p, m: _rewrite_role(p, m, "fit", "public", [])),
        (
            "family_roster_mismatch",
            lambda p, m: _rewrite_info(p, m, "families", ["wrong", "wrong"]),
        ),
        (
            "source_roster_mismatch",
            lambda p, m: _rewrite_info(p, m, "source_hashes", ["wrong", "wrong"]),
        ),
        (
            "response_roster_mismatch",
            lambda p, m: _rewrite_info(p, m, "response_hashes", ["wrong", "wrong"]),
        ),
        (
            "label_join_mismatch",
            lambda p, m: _rewrite_role(p, m, "fit", "evaluator", [{"label": 1}] * 2),
        ),
    )
    for index, (error, mutation) in enumerate(cases):
        path = tmp_path / f"mutation-{index}"
        sealed, _ = sealed_fixture(path)
        mutation(path, sealed)
        if error == "public_manifest_mismatch":
            _update_public_hash(path)
        with pytest.raises(ValueError, match=error):
            cohort.cold_reduce(path / "manifest.json", counts)


def test_scenario_verify_7715_isolation_duplicate_family_cold(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-7715-ISOLATION: hashes cannot license a duplicate family.
    manifest, counts = sealed_fixture(tmp_path)
    fit = cohort.predictor_inputs(tmp_path / manifest["roles"]["fit"]["public_path"], "fit")
    evaluation = cohort.predictor_inputs(
        tmp_path / manifest["roles"]["evaluation"]["public_path"], "evaluation"
    )
    evaluation[0]["family_id"] = fit[0]["family_id"]
    _rewrite_role(tmp_path, manifest, "evaluation", "public", evaluation)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    _rewrite_info(tmp_path, manifest, "families", manifest["roles"]["fit"]["families"])
    # The roster edit above targets fit; set evaluation's roster directly.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["roles"]["evaluation"]["families"] = [fit[0]["family_id"]]
    public = json.loads((tmp_path / "public_manifest.json").read_text())
    public["roles"]["evaluation"]["families"] = [fit[0]["family_id"]]
    atomic_json(tmp_path / "public_manifest.json", public)
    atomic_json(tmp_path / "manifest.json", manifest)
    _update_public_hash(tmp_path)
    with pytest.raises(ValueError, match="duplicate_family"):
        cohort.cold_reduce(tmp_path / "manifest.json", counts)


def test_scenario_report_7715_terminal_cold_mutations(tmp_path: Path) -> None:
    # SCENARIO-REPORT-7715-TERMINAL: exact saved inventory controls the class.
    candidate = runner.build_candidate(runner.ROOT, tmp_path, "20260926")
    mutations = (
        ("family_roster_mismatch", lambda x: x["rows"].pop()),
        ("eligible_count_mismatch", lambda x: x["sample_size_budget"].update(eligible_families=1)),
        ("gate_summary_mismatch", lambda x: x.update(gate_check_summary=[])),
        (
            "shortage_promoted",
            lambda x: x.update(verdict_class="null", natural_cohort_ready_score=1),
        ),
    )
    for error, mutation in mutations:
        changed = deepcopy(candidate)
        mutation(changed)
        with pytest.raises(ValueError, match=error):
            runner.cold_reduce(changed, tmp_path)
    path = tmp_path / "candidate_manifest.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="role_manifest_hash_mismatch"):
        runner.cold_reduce(candidate, tmp_path)


def test_scenario_report_7715_custody_scans_mutated_exposure(tmp_path: Path) -> None:
    # SCENARIO-REPORT-7715-CUSTODY: missing shards and later IDs stay visible.
    directory = tmp_path / "results/raw/experiment_7423_v651_annotated_protocol"
    directory.mkdir(parents=True)
    atomic_json(
        directory / "corpus_manifest.json",
        {
            "schema": "carnot.exp7423.corpus_manifest.v1",
            "shards": [
                {"kind": "evaluator", "path": "missing.jsonl", "sha256": "sha256:absent", "rows": 1}
            ],
        },
    )
    ids, checks, _ = runner._v651_exposure(tmp_path, time.monotonic())
    assert ids == set()
    assert any(not row["passed"] for row in checks)
    later = tmp_path / "results/raw/experiment_7500_v657_test"
    later.mkdir(parents=True)
    (later / "source_manifest.json").write_text('{"source_id":"123"}')
    found, receipts = runner._later_exposure_inventory(tmp_path, time.monotonic())
    assert found == {"123"}
    assert receipts[0]["sha256"].startswith("sha256:")
    with pytest.raises(ValueError, match="repo_or_date_mismatch"):
        runner.build_candidate(runner.ROOT, tmp_path / "unused", "wrong")


def test_scenario_report_7715_terminal_orchestrates_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SCENARIO-REPORT-7715-TERMINAL: subprocess exits govern the saved result.
    candidate = runner.build_candidate(runner.ROOT, tmp_path, "20260926")
    monkeypatch.setattr(runner, "RAW", tmp_path)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "final.json")
    monkeypatch.setattr(runner, "build_candidate", lambda _root, _raw, _date: deepcopy(candidate))
    scoped = {
        "validation_receipts": [
            {"name": "focused_pytest", "exit_code": 0, "log_sha256": "sha256:test"}
        ],
        "required_checks_passed": True,
        "repository_health": {"global_debt": "kept separate"},
    }
    monkeypatch.setattr(runner.validation, "run_scoped_validation", lambda *a, **k: scoped)
    terminal = [{"passed": True, "exit_code": 0, "log_sha256": "sha256:test"} for _ in range(3)]
    monkeypatch.setattr(runner.validation, "run_commands", lambda *a, **k: terminal)
    result = runner.run_experiment(runner.ROOT, "20260926")
    assert result["verdict_class"] == "blocked"
    assert json.loads((tmp_path / "final.json").read_text())["verdict_class"] == "blocked"
    assert runner.main(["--date", "20260926"]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "terminal_candidate.json")]) == 0
    scoped["required_checks_passed"] = False
    result = runner.run_experiment(runner.ROOT, "20260926")
    assert result["honest_verdict"] == "complete_disqualified_required_validation"
    scoped["required_checks_passed"] = True
    terminal[1] = {"passed": False, "exit_code": 1, "log_sha256": "sha256:failed"}
    result = runner.run_experiment(runner.ROOT, "20260926")
    assert result["honest_verdict"] == "complete_disqualified_terminal_reader"
    assert result["flagged_adversarial"] is True
