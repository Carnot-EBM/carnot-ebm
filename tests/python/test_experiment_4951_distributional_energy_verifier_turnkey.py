"""Tests for Exp 4951 distributional-energy verifier turnkey readiness.

Spec refs: REQ-KONA-4951, SCENARIO-KONA-4951-TURNKEY-DRY-RUN,
SCENARIO-KONA-4951-BLOCKED, SCENARIO-KONA-4951-VALIDATION-GATE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4951_distributional_energy_verifier_turnkey as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase3-kona" / "spec.md"


def _candidate(
    candidate_id: str,
    answer: str,
    *,
    sample_count: int,
    quality: float,
    penalty: float,
    stddev: float,
    correct: bool,
    model_id: str = "cached-generator-a",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "answer": answer,
        "sample_count": sample_count,
        "learned_quality_mean": quality,
        "learned_quality_stddev": stddev,
        "deterministic_constraint_penalty": penalty,
        "uncertainty": stddev,
        "label_correct": correct,
        "model_id": model_id,
    }


def _row(problem_id: str = "tp4951-test-1") -> dict[str, object]:
    return {
        "problem_id": problem_id,
        "domain": "travelplanner_style_structured_reasoning",
        "prompt": "Build a small constrained itinerary.",
        "oracle_distinct": True,
        "cheap_executable_oracle_available": False,
        "candidates": [
            _candidate(
                "sc-majority",
                "invalid but common plan",
                sample_count=3,
                quality=0.55,
                penalty=0.42,
                stddev=0.12,
                correct=False,
            ),
            _candidate(
                "energy-best",
                "constraint-respecting plan",
                sample_count=2,
                quality=0.88,
                penalty=0.03,
                stddev=0.05,
                correct=True,
                model_id="cached-generator-b",
            ),
        ],
    }


def _saturated_row() -> dict[str, object]:
    return {
        "problem_id": "tp4951-saturated",
        "domain": "travelplanner_style_structured_reasoning",
        "prompt": "Easy row where self-consistency is already correct.",
        "oracle_distinct": True,
        "cheap_executable_oracle_available": False,
        "candidates": [
            _candidate(
                "majority-correct",
                "correct plan",
                sample_count=5,
                quality=0.91,
                penalty=0.01,
                stddev=0.03,
                correct=True,
            ),
            _candidate(
                "minority-wrong",
                "wrong plan",
                sample_count=1,
                quality=0.40,
                penalty=0.33,
                stddev=0.15,
                correct=False,
            ),
        ],
    }


def _write_slice(path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    rows = rows if rows is not None else [_row("tp4951-test-1"), _row("tp4951-test-2")]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=REPO,
        domain_slice_path=_write_slice(tmp_path / "slice.jsonl"),
        net_available=True,
    )


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-KONA-4951")
    end = spec.index("## Latent Symbol Bridge Falsification", start)
    return spec[start:end]


def _stage_repo_with(path: Path, *relative_paths: str, registry_has_active: bool = True) -> Path:
    for relative_path in relative_paths:
        target = path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative_path == mod.FOVER_REGISTRY_RELATIVE_PATH:
            text = (
                "verifiers:\n- verifier_id: fover_production_ensemble\n"
                if registry_has_active
                else "verifiers: []\n"
            )
        else:
            text = "{}\n" if relative_path.endswith(".json") else "# staged\n"
        target.write_text(text, encoding="utf-8")
    return path


def test_req_kona_4951_spec_declares_turnkey_contract() -> None:
    """REQ-KONA-4951: OpenSpec anchors the turnkey paths, fields, and guardrails."""

    section = _spec_section()

    for marker in (
        "REQ-KONA-4951",
        "SCENARIO-KONA-4951-TURNKEY-DRY-RUN",
        "SCENARIO-KONA-4951-BLOCKED",
        "SCENARIO-KONA-4951-VALIDATION-GATE",
        mod.RESULT_RELATIVE_PATH,
        mod.NOTE_RELATIVE_PATH,
        mod.MODULE_RELATIVE_PATH,
        mod.ENTRYPOINT_COMMAND,
        "research-studying.md",
        "scripts/research_conductor.py",
        "ops/changelog.md",
        "ops/status.md",
        "_bmad/traceability.md",
        "self_consistency",
        "decomposed_energy_verifier",
        "oracle",
        "pivot_turnkey",
        "post_sprint_first_experiment_pointer",
        "verifier_is_oracle=false",
        "moat_proven_claimed=false",
        mod.HONEST_VERDICT,
    ):
        assert marker in section
    for arxiv_id in mod.ARXIV_IDS:
        assert arxiv_id in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_req_kona_4951_citations_mapping_and_next_milestone_are_complete() -> None:
    """REQ-KONA-4951: SOTA ingestion maps every required arXiv paper."""

    assert set(mod.CITATIONS) == set(mod.ARXIV_IDS)
    assert set(mod.SOTA_TO_CARNOT_MAPPING) == set(mod.ARXIV_IDS)
    assert "decomposed_energy_lora_ensemble" in mod.FLAGGED_FOR_NEXT_MILESTONE[0]
    for arxiv_id, citation in mod.CITATIONS.items():
        assert citation["url"] == f"https://arxiv.org/abs/{arxiv_id}"
        assert citation["http_status"] == 200
        assert citation["title"]
    for arxiv_id, mapping in mod.SOTA_TO_CARNOT_MAPPING.items():
        assert set(mapping) == mod.REQUIRED_MAPPING_FIELDS
        assert mapping["source_id"] == arxiv_id
        assert mapping["strongest_method"]
        assert "FoVer" in mapping["implementation_cost_over_current_stack"]
        assert mapping["pitfalls"]
        assert mapping["roadmap_input"]


def test_req_kona_4951_loader_reads_real_slice_and_enforces_non_saturation(tmp_path: Path) -> None:
    """REQ-KONA-4951: real JSONL loader returns a small SC-not-saturated slice."""

    loaded = mod.load_turnkey_domain_slice(_write_slice(tmp_path / "slice.jsonl"), limit=1)

    assert len(loaded) == 1
    assert loaded[0]["problem_id"] == "tp4951-test-1"
    assert mod.slice_self_consistency_saturated(loaded) is False
    with pytest.raises(ValueError, match="self-consistency saturated"):
        mod.load_turnkey_domain_slice(_write_slice(tmp_path / "saturated.jsonl", [_saturated_row()]))


def test_req_kona_4951_energy_uses_mean_and_fover_penalty_not_identity_or_labels() -> None:
    """REQ-KONA-4951: verifier score ignores model identity and oracle labels."""

    candidate = _candidate(
        "c",
        "plan",
        sample_count=1,
        quality=0.80,
        penalty=0.20,
        stddev=0.10,
        correct=False,
        model_id="generator-a",
    )
    identity_and_label_mutated = dict(candidate) | {
        "model_id": "generator-b",
        "label_correct": True,
    }

    assert mod.learned_quality_ensemble(candidate) == {"mean": 0.8, "stddev": 0.1}
    assert mod.fover_analytical_penalty(candidate) == pytest.approx(0.20)
    assert mod.decomposed_energy(candidate) == pytest.approx(-0.60)
    assert mod.decomposed_energy(identity_and_label_mutated) == pytest.approx(-0.60)
    assert mod.candidate_abstention(candidate) is False
    assert mod.candidate_abstention(dict(candidate) | {"learned_quality_stddev": 0.50}) is True


def test_scenario_kona_4951_dry_run_emits_self_consistency_energy_and_oracle() -> None:
    """SCENARIO-KONA-4951-TURNKEY-DRY-RUN: three columns wire end-to-end."""

    dry_run = mod.run_three_column_dry_run([_row()], limit=1)

    assert dry_run["columns"] == list(mod.THREE_DRY_RUN_COLUMNS)
    assert dry_run["n_rows"] == 1
    assert dry_run["full_benchmark_run"] is False
    scored = dry_run["rows"][0]
    assert scored["self_consistency"]["selected_candidate_id"] == "sc-majority"
    assert scored["decomposed_energy_verifier"]["selected_candidate_id"] == "energy-best"
    assert scored["decomposed_energy_verifier"]["verifier_is_oracle"] is False
    assert scored["decomposed_energy_verifier"]["analytical_penalty_source"]["verifier_id"] == (
        mod.FOVER_ACTIVE_ENSEMBLE_ID
    )
    assert scored["decomposed_energy_verifier"]["abstention_recommended"] is False
    assert scored["oracle"]["selected_candidate_id"] == "energy-best"
    assert scored["oracle"]["oracle_used_for_correctness_only"] is True


def test_req_kona_4951_self_consistency_uses_sample_counts() -> None:
    """REQ-KONA-4951: self-consistency column follows cached sample counts."""

    row = _row()
    candidates = list(row["candidates"])
    candidates[0] = dict(candidates[0]) | {"sample_count": 1}
    candidates[1] = dict(candidates[1]) | {"sample_count": 5}

    selected = mod.select_self_consistency(candidates)

    assert selected["candidate_id"] == "energy-best"
    assert mod.self_consistency_accuracy([]) == 0.0


def test_scenario_kona_4951_success_artifact_has_required_guardrails(tmp_path: Path) -> None:
    """SCENARIO-KONA-4951-VALIDATION-GATE: success artifact is readiness only."""

    artifact = _artifact(tmp_path)

    mod.validate_artifact(artifact)
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.FIELD_PRINCIPLES)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["arxiv_ids_cited"] == list(mod.ARXIV_IDS)
    assert artifact["pivot_executable_on_7_1"] is True
    assert artifact["pivot_turnkey"] is True
    assert artifact["three_column_dry_run_ok"] is True
    assert artifact["sc_not_saturated_domain"] == mod.SC_NOT_SATURATED_DOMAIN
    assert artifact["post_sprint_first_experiment_pointer"]["entrypoint_command"] == (
        mod.ENTRYPOINT_COMMAND
    )
    assert artifact["validation_gate"]["claimed_met"] is False
    assert artifact["validation_gate"]["beats_self_consistency_ci95_excludes_zero_required"] is True
    assert artifact["validation_gate"]["oracle_distinct_required"] is True
    assert artifact["validation_gate"]["no_model_identity_shortcut_required"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["moat_proven_claimed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["net_available"] is True
    assert artifact["preconditions_checked"]["blocked_resource"] is None
    assert artifact["preconditions_checked"]["self_consistency_saturated"] is False
    assert artifact["no_real_benchmark_run"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenario_kona_4951_blocked_preconditions_do_not_fabricate(tmp_path: Path) -> None:
    """SCENARIO-KONA-4951-BLOCKED: missing resources block honestly."""

    domain_slice = _write_slice(tmp_path / "slice.jsonl")
    missing_spec = mod.build_artifact(
        repo_root=tmp_path / "empty",
        domain_slice_path=domain_slice,
        net_available=False,
    )
    mod.validate_artifact(missing_spec)
    assert missing_spec["honest_verdict"] == "blocked_executable_spec_artifact_missing"
    assert missing_spec["pivot_executable_on_7_1"] is False
    assert missing_spec["pivot_turnkey"] is False
    assert missing_spec["preconditions_checked"]["net_available"] is False

    missing_module_repo = _stage_repo_with(
        tmp_path / "missing_module_repo",
        mod.EXP4940_RESULT_RELATIVE_PATH,
    )
    missing_module = mod.build_artifact(
        repo_root=missing_module_repo,
        domain_slice_path=domain_slice,
        net_available=True,
    )
    mod.validate_artifact(missing_module)
    assert missing_module["honest_verdict"] == "blocked_executable_spec_module_missing"

    missing_harness_repo = _stage_repo_with(
        tmp_path / "missing_harness_repo",
        mod.EXP4940_RESULT_RELATIVE_PATH,
        mod.EXP4940_MODULE_RELATIVE_PATH,
    )
    missing_harness = mod.build_artifact(
        repo_root=missing_harness_repo,
        domain_slice_path=domain_slice,
        net_available=True,
    )
    mod.validate_artifact(missing_harness)
    assert missing_harness["honest_verdict"] == "blocked_exp4922_harness_missing"

    missing_registry_repo = _stage_repo_with(
        tmp_path / "missing_registry_repo",
        mod.EXP4940_RESULT_RELATIVE_PATH,
        mod.EXP4940_MODULE_RELATIVE_PATH,
        mod.EXP4922_HARNESS_RELATIVE_PATH,
    )
    missing_registry = mod.build_artifact(
        repo_root=missing_registry_repo,
        domain_slice_path=domain_slice,
        net_available=True,
    )
    mod.validate_artifact(missing_registry)
    assert missing_registry["honest_verdict"] == "blocked_fover_registry_missing"

    missing_active_repo = _stage_repo_with(
        tmp_path / "missing_active_repo",
        mod.EXP4940_RESULT_RELATIVE_PATH,
        mod.EXP4940_MODULE_RELATIVE_PATH,
        mod.EXP4922_HARNESS_RELATIVE_PATH,
        mod.FOVER_REGISTRY_RELATIVE_PATH,
        registry_has_active=False,
    )
    missing_active = mod.build_artifact(
        repo_root=missing_active_repo,
        domain_slice_path=domain_slice,
        net_available=True,
    )
    mod.validate_artifact(missing_active)
    assert missing_active["honest_verdict"] == "blocked_fover_active_ensemble_missing"

    missing_slice = mod.build_artifact(
        repo_root=REPO,
        domain_slice_path=tmp_path / "missing.jsonl",
        net_available=True,
    )
    mod.validate_artifact(missing_slice)
    assert missing_slice["honest_verdict"] == "blocked_domain_slice_missing"

    saturated = mod.build_artifact(
        repo_root=REPO,
        domain_slice_path=_write_slice(tmp_path / "saturated.jsonl", [_saturated_row()]),
        net_available=True,
    )
    mod.validate_artifact(saturated)
    assert saturated["honest_verdict"] == "blocked_self_consistency_saturated"

    invalid_slice = tmp_path / "invalid.jsonl"
    invalid_slice.write_text("{not json}\n", encoding="utf-8")
    invalid = mod.check_preconditions(
        repo_root=REPO,
        domain_slice_path=invalid_slice,
        net_available=True,
    )
    assert invalid["blocked_resource"] == "domain_slice_invalid"
    assert invalid["domain_error"]


def test_req_kona_4951_defensive_helpers_fail_closed() -> None:
    """REQ-KONA-4951: missing candidates, oracle labels, and mapping drift fail closed."""

    with pytest.raises(ValueError, match="candidate list is empty"):
        mod.select_self_consistency([])
    with pytest.raises(ValueError, match="cached correct label"):
        mod.select_oracle(
            [
                _candidate(
                    "wrong",
                    "wrong",
                    sample_count=1,
                    quality=0.2,
                    penalty=0.3,
                    stddev=0.1,
                    correct=False,
                )
            ]
        )

    bad_fields = copy.deepcopy(mod.SOTA_TO_CARNOT_MAPPING)
    bad_fields["2605.18871"] = {
        key: value for key, value in bad_fields["2605.18871"].items() if key != "pitfalls"
    }
    with pytest.raises(ValueError, match="fields invalid"):
        mod.validate_mapping(bad_fields)

    bad_source = copy.deepcopy(mod.SOTA_TO_CARNOT_MAPPING)
    bad_source["2605.18871"]["source_id"] = "9999.99999"
    with pytest.raises(ValueError, match="source_id mismatch"):
        mod.validate_mapping(bad_source)

    bad_text = copy.deepcopy(mod.SOTA_TO_CARNOT_MAPPING)
    bad_text["2605.18871"]["strongest_method"] = ""
    with pytest.raises(ValueError, match="strongest_method"):
        mod.validate_mapping(bad_text)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"honest_verdict": "draft"}, "terminal prefix"),
        ({"honest_verdict": "success_wrong"}, "honest_verdict"),
        ({"arxiv_ids_cited": ["2605.18871"]}, "arxiv_ids_cited"),
        ({"sota_to_carnot_mapping": {}}, "sota_to_carnot_mapping"),
        ({"pivot_executable_on_7_1": False}, "pivot_executable"),
        ({"pivot_turnkey": False}, "pivot_turnkey"),
        ({"three_column_dry_run_ok": False}, "three_column_dry_run_ok"),
        ({"sc_not_saturated_domain": "ARC"}, "sc_not_saturated_domain"),
        ({"post_sprint_first_experiment_pointer": {"entrypoint_command": "wrong"}}, "entrypoint"),
        ({"validation_gate": {"claimed_met": True}}, "validation_gate"),
        ({"verifier_is_oracle": True}, "verifier_is_oracle"),
        ({"moat_proven_claimed": True}, "moat_proven_claimed"),
        ({"inference_substrate": "live_benchmark"}, "inference_substrate"),
        ({"random_seed": 4951}, "random_seed"),
        ({"dry_run_three_columns": {"columns": ["self_consistency"]}}, "dry_run_three_columns"),
        (
            {"dry_run_three_columns": mod.run_three_column_dry_run([_row()]) | {"full_benchmark_run": True}},
            "full benchmark",
        ),
        ({"citations": {}}, "citations"),
        (
            {
                "citations": mod.CITATIONS
                | {
                    "2605.18871": {
                        key: value
                        for key, value in mod.CITATIONS["2605.18871"].items()
                        if key != "title"
                    }
                }
            },
            "fields invalid",
        ),
        (
            {
                "citations": mod.CITATIONS
                | {
                    "2605.18871": mod.CITATIONS["2605.18871"]
                    | {"url": "https://arxiv.org/abs/9999.99999"}
                }
            },
            "URL invalid",
        ),
        (
            {
                "citations": mod.CITATIONS
                | {"2605.18871": mod.CITATIONS["2605.18871"] | {"http_status": 404}}
            },
            "http_status",
        ),
        ({"field_principles": {}}, "field_principles"),
        ({"no_real_benchmark_run": False}, "no_real_benchmark_run"),
    ],
)
def test_validate_artifact_rejects_guardrail_violations(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    """SCENARIO-KONA-4951-VALIDATION-GATE: invalid guardrails fail closed."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad.update(updates)
    if "reproducibility_checksum" in bad:
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad)


def test_validate_artifact_rejects_missing_extra_and_bad_checksum(tmp_path: Path) -> None:
    """REQ-KONA-4951: artifact schema and checksum drift are rejected."""

    artifact = _artifact(tmp_path)
    missing = copy.deepcopy(artifact)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="fields mismatch"):
        mod.validate_artifact(missing)

    extra = copy.deepcopy(artifact)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        mod.validate_artifact(extra)

    bad_checksum = copy.deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:wrong"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_network_checker_reports_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KONA-4951: network precondition is recorded but not required."""

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def ok(_url: str, timeout: float) -> Response:
        assert timeout == mod.NETWORK_TIMEOUT_S
        return Response()

    def fail(_url: str, timeout: float) -> Response:
        raise OSError("network down")

    monkeypatch.setattr(mod.request, "urlopen", ok)
    assert mod.check_network_available() is True
    monkeypatch.setattr(mod.request, "urlopen", fail)
    assert mod.check_network_available() is False


def test_main_writes_result_note_and_research_studying_section(tmp_path: Path) -> None:
    """REQ-KONA-4951: main writes the artifact, note, and INGESTED study marker."""

    result_path = tmp_path / "artifact.json"
    note_path = tmp_path / "note.md"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("before\n", encoding="utf-8")

    artifact = mod.main(
        repo_root=REPO,
        domain_slice_path=_write_slice(tmp_path / "slice.jsonl"),
        result_path=result_path,
        note_path=note_path,
        studying_path=studying_path,
        net_available=True,
    )

    written = json.loads(result_path.read_text(encoding="utf-8"))
    note = note_path.read_text(encoding="utf-8")
    studying = studying_path.read_text(encoding="utf-8")
    assert written == artifact
    assert "Exp 4951" in note
    assert mod.ENTRYPOINT_COMMAND in note
    assert "INGESTED" in studying
    assert "flagged_for_next_milestone" in studying
    assert studying.count(mod.STUDYING_SECTION_START) == 1

    mod.write_research_studying_section(studying_path, artifact)
    updated = studying_path.read_text(encoding="utf-8")
    assert updated.count(mod.STUDYING_SECTION_START) == 1


def test_research_studying_writer_can_build_default_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-KONA-4951: studying writer has a deterministic default artifact path."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        domain_slice_path=_write_slice(tmp_path / "slice.jsonl"),
        net_available=True,
    )
    studying_path = tmp_path / "research-studying.md"

    monkeypatch.setattr(mod, "build_artifact", lambda: artifact)
    mod.write_research_studying_section(studying_path)

    studying = studying_path.read_text(encoding="utf-8")
    assert "Exp 4951" in studying
    assert "INGESTED" in studying
