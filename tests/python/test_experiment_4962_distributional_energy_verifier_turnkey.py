"""Tests for Exp 4962 distributional-energy verifier turnkey backlog extension.

Spec refs: REQ-KONA-4962, SCENARIO-KONA-4962-TURNKEY-BACKLOG,
SCENARIO-KONA-4962-BLOCKED, SCENARIO-KONA-4962-VALIDATION-GATE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4962_distributional_energy_verifier_turnkey as mod


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


def _row(problem_id: str = "tp4962-test-1") -> dict[str, object]:
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
        "problem_id": "tp4962-saturated",
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
    rows = rows if rows is not None else [_row("tp4962-test-1"), _row("tp4962-test-2")]
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
    start = spec.index("### REQ-KONA-4962")
    end = spec.index("## Latent Symbol Bridge Falsification", start)
    return spec[start:end]


def _stage_repo_with(
    path: Path,
    *relative_paths: str,
    registry_has_active: bool = True,
    spec_has_req: bool = True,
) -> Path:
    for relative_path in relative_paths:
        target = path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative_path == mod.FOVER_REGISTRY_RELATIVE_PATH:
            text = (
                "verifiers:\n- verifier_id: fover_production_ensemble\n"
                if registry_has_active
                else "verifiers: []\n"
            )
        elif relative_path == mod.KONA_SPEC_RELATIVE_PATH:
            text = "### REQ-KONA-4962\n" if spec_has_req else "### REQ-KONA-4951\n"
        else:
            text = "{}\n" if relative_path.endswith(".json") else "# staged\n"
        target.write_text(text, encoding="utf-8")
    return path


def test_req_kona_4962_spec_declares_turnkey_backlog_contract() -> None:
    """REQ-KONA-4962: OpenSpec anchors paths, fields, papers, and guardrails."""

    section = _spec_section()

    for marker in (
        "REQ-KONA-4962",
        "SCENARIO-KONA-4962-TURNKEY-BACKLOG",
        "SCENARIO-KONA-4962-BLOCKED",
        "SCENARIO-KONA-4962-VALIDATION-GATE",
        mod.RESULT_RELATIVE_PATH,
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


def test_req_kona_4962_citations_mapping_and_taxonomy_are_complete() -> None:
    """REQ-KONA-4962: new and re-confirmed arXiv papers map onto Carnot."""

    assert mod.ARXIV_IDS == mod.NEW_ARXIV_IDS + mod.RECONFIRMED_ARXIV_IDS
    assert set(mod.CITATIONS) == set(mod.ARXIV_IDS)
    assert set(mod.SOTA_TO_CARNOT_MAPPING) == set(mod.ARXIV_IDS)
    assert set(mod.ALREADY_INGESTED_RECONFIRMED) == set(mod.RECONFIRMED_ARXIV_IDS)
    assert "decomposed_energy_verifier" in mod.TAXONOMY_POSITION["carnot_design_cell"]
    assert "fast_slow_process_router" in mod.NEXT_MILESTONE_ROADMAP_INPUTS
    for arxiv_id, citation in mod.CITATIONS.items():
        assert citation["url"] == f"https://arxiv.org/abs/{arxiv_id}"
        assert citation["http_status"] == 200
        assert citation["title"]
        assert citation["ingestion_status"] in {"new", "reconfirmed"}
    for arxiv_id, mapping in mod.SOTA_TO_CARNOT_MAPPING.items():
        assert set(mapping) == mod.REQUIRED_MAPPING_FIELDS
        assert mapping["source_id"] == arxiv_id
        assert mapping["strongest_method"]
        assert mapping["implementation_cost_over_current_stack"]
        assert mapping["pitfalls"]
        assert mapping["roadmap_input"]


def test_scenario_kona_4962_dry_run_reconfirms_three_columns() -> None:
    """SCENARIO-KONA-4962-TURNKEY-BACKLOG: three columns wire end-to-end."""

    rows = [_row()]
    dry_run = mod.run_three_column_dry_run(rows, limit=1)

    assert dry_run["columns"] == list(mod.THREE_DRY_RUN_COLUMNS)
    assert dry_run["n_rows"] == 1
    assert dry_run["full_benchmark_run"] is False
    assert dry_run["reconfirmed_from_exp4951"] is True
    scored = dry_run["rows"][0]
    assert scored["self_consistency"]["selected_candidate_id"] == "sc-majority"
    assert scored["decomposed_energy_verifier"]["selected_candidate_id"] == "energy-best"
    assert scored["decomposed_energy_verifier"]["verifier_is_oracle"] is False
    assert scored["decomposed_energy_verifier"]["abstention_recommended"] is False
    assert scored["oracle"]["selected_candidate_id"] == "energy-best"
    assert scored["oracle"]["oracle_used_for_correctness_only"] is True


def test_req_kona_4962_loader_reads_real_slice_and_enforces_non_saturation(tmp_path: Path) -> None:
    """REQ-KONA-4962: real JSONL loader returns a small SC-not-saturated slice."""

    loaded = mod.load_turnkey_domain_slice(_write_slice(tmp_path / "slice.jsonl"), limit=1)

    assert len(loaded) == 1
    assert loaded[0]["problem_id"] == "tp4962-test-1"
    assert mod.slice_self_consistency_saturated(loaded) is False
    with pytest.raises(ValueError, match="self-consistency saturated"):
        mod.load_turnkey_domain_slice(_write_slice(tmp_path / "saturated.jsonl", [_saturated_row()]))


def test_scenario_kona_4962_success_artifact_has_required_guardrails(tmp_path: Path) -> None:
    """SCENARIO-KONA-4962-VALIDATION-GATE: success artifact is readiness only."""

    artifact = _artifact(tmp_path)

    mod.validate_artifact(artifact)
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.FIELD_PRINCIPLES)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["arxiv_ids_cited"] == list(mod.ARXIV_IDS)
    assert artifact["already_ingested_reconfirmed"] == list(mod.RECONFIRMED_ARXIV_IDS)
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
    assert artifact["preconditions_checked"]["exp4951_turnkey_artifact_present"] is True
    assert artifact["preconditions_checked"]["self_consistency_saturated"] is False
    assert artifact["no_real_benchmark_run"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenario_kona_4962_blocked_preconditions_do_not_fabricate(tmp_path: Path) -> None:
    """SCENARIO-KONA-4962-BLOCKED: missing resources block honestly."""

    domain_slice = _write_slice(tmp_path / "slice.jsonl")
    missing_spec = mod.build_artifact(
        repo_root=tmp_path / "empty",
        domain_slice_path=domain_slice,
        net_available=False,
    )
    mod.validate_artifact(missing_spec)
    assert missing_spec["honest_verdict"] == "blocked_kona_spec_missing"
    assert missing_spec["pivot_executable_on_7_1"] is False
    assert missing_spec["pivot_turnkey"] is False
    assert missing_spec["preconditions_checked"]["net_available"] is False

    missing_req_repo = _stage_repo_with(
        tmp_path / "missing_req_repo",
        mod.KONA_SPEC_RELATIVE_PATH,
        spec_has_req=False,
    )
    missing_req = mod.build_artifact(
        repo_root=missing_req_repo,
        domain_slice_path=domain_slice,
        net_available=True,
    )
    mod.validate_artifact(missing_req)
    assert missing_req["honest_verdict"] == "blocked_kona_spec_req_missing"

    staged = _stage_repo_with(
        tmp_path / "staged",
        mod.KONA_SPEC_RELATIVE_PATH,
        mod.EXP4951_RESULT_RELATIVE_PATH,
        mod.EXP4951_MODULE_RELATIVE_PATH,
        mod.EXP4922_HARNESS_RELATIVE_PATH,
        mod.FOVER_REGISTRY_RELATIVE_PATH,
    )
    missing_slice = mod.build_artifact(
        repo_root=staged,
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


@pytest.mark.parametrize(
    ("missing", "resource"),
    [
        ("exp4951_artifact", "exp4951_turnkey_artifact_missing"),
        ("exp4951_module", "exp4951_turnkey_module_missing"),
        ("exp4922_harness", "exp4922_harness_missing"),
        ("fover_registry", "fover_registry_missing"),
        ("fover_active", "fover_active_ensemble_missing"),
    ],
)
def test_blocked_resource_helper_names_each_missing_precondition(
    missing: str, resource: str
) -> None:
    """REQ-KONA-4962: precondition naming is exact for conductor audits."""

    kwargs = {
        "kona_spec_present": True,
        "kona_spec_has_req": True,
        "exp4951_artifact_present": True,
        "exp4951_module_present": True,
        "exp4922_harness_present": True,
        "fover_registry_present": True,
        "fover_active_ensemble_present": True,
        "domain_slice_present": True,
        "domain_slice_valid": True,
        "self_consistency_saturated": False,
    }
    if missing == "exp4951_artifact":
        kwargs["exp4951_artifact_present"] = False
    elif missing == "exp4951_module":
        kwargs["exp4951_module_present"] = False
    elif missing == "exp4922_harness":
        kwargs["exp4922_harness_present"] = False
    elif missing == "fover_registry":
        kwargs["fover_registry_present"] = False
    else:
        kwargs["fover_active_ensemble_present"] = False

    assert mod.blocked_resource_from_preconditions(**kwargs) == resource


def test_blocked_resource_helper_accepts_complete_preconditions() -> None:
    """REQ-KONA-4962: complete staged resources leave the run unblocked."""

    assert (
        mod.blocked_resource_from_preconditions(
            kona_spec_present=True,
            kona_spec_has_req=True,
            exp4951_artifact_present=True,
            exp4951_module_present=True,
            exp4922_harness_present=True,
            fover_registry_present=True,
            fover_active_ensemble_present=True,
            domain_slice_present=True,
            domain_slice_valid=True,
            self_consistency_saturated=False,
        )
        is None
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"honest_verdict": "draft"}, "terminal prefix"),
        ({"honest_verdict": "success_wrong"}, "honest_verdict"),
        ({"arxiv_ids_cited": ["2508.16665"]}, "arxiv_ids_cited"),
        ({"sota_to_carnot_mapping": {}}, "sota_to_carnot_mapping"),
        ({"already_ingested_reconfirmed": []}, "already_ingested_reconfirmed"),
        ({"taxonomy_position": {"carnot_design_cell": "wrong"}}, "taxonomy_position"),
        ({"next_milestone_roadmap_inputs": []}, "next_milestone"),
        ({"pivot_executable_on_7_1": False}, "pivot_executable"),
        ({"pivot_turnkey": False}, "pivot_turnkey"),
        ({"three_column_dry_run_ok": False}, "three_column_dry_run_ok"),
        ({"sc_not_saturated_domain": "ARC"}, "sc_not_saturated_domain"),
        ({"post_sprint_first_experiment_pointer": {"entrypoint_command": "wrong"}}, "entrypoint"),
        ({"validation_gate": {"claimed_met": True}}, "validation_gate"),
        ({"verifier_is_oracle": True}, "verifier_is_oracle"),
        ({"moat_proven_claimed": True}, "moat_proven_claimed"),
        ({"inference_substrate": "live_benchmark"}, "inference_substrate"),
        ({"random_seed": 4962}, "random_seed"),
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
                    "2508.16665": {
                        key: value
                        for key, value in mod.CITATIONS["2508.16665"].items()
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
                    "2508.16665": mod.CITATIONS["2508.16665"]
                    | {"url": "https://arxiv.org/abs/9999.99999"}
                }
            },
            "URL invalid",
        ),
        (
            {
                "citations": mod.CITATIONS
                | {"2508.16665": mod.CITATIONS["2508.16665"] | {"http_status": 404}}
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
    """SCENARIO-KONA-4962-VALIDATION-GATE: invalid guardrails fail closed."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad.update(updates)
    if "reproducibility_checksum" in bad:
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad)


def test_validate_artifact_rejects_mapping_schema_and_checksum(tmp_path: Path) -> None:
    """REQ-KONA-4962: artifact and mapping schemas reject drift."""

    artifact = _artifact(tmp_path)
    missing = copy.deepcopy(artifact)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="fields mismatch"):
        mod.validate_artifact(missing)

    extra = copy.deepcopy(artifact)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        mod.validate_artifact(extra)

    bad_fields = copy.deepcopy(mod.SOTA_TO_CARNOT_MAPPING)
    bad_fields["2508.16665"] = {
        key: value for key, value in bad_fields["2508.16665"].items() if key != "pitfalls"
    }
    with pytest.raises(ValueError, match="fields invalid"):
        mod.validate_mapping(bad_fields)

    bad_source = copy.deepcopy(mod.SOTA_TO_CARNOT_MAPPING)
    bad_source["2508.16665"]["source_id"] = "9999.99999"
    with pytest.raises(ValueError, match="source_id mismatch"):
        mod.validate_mapping(bad_source)

    bad_text = copy.deepcopy(mod.SOTA_TO_CARNOT_MAPPING)
    bad_text["2508.16665"]["strongest_method"] = ""
    with pytest.raises(ValueError, match="strongest_method"):
        mod.validate_mapping(bad_text)

    bad_checksum = copy.deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:wrong"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_network_checker_reports_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KONA-4962: network precondition is recorded but not required."""

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
    """REQ-KONA-4962: main writes the artifact, note, and INGESTED marker."""

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
    assert "Exp 4962" in note
    assert "2508.16665" in note
    assert mod.ENTRYPOINT_COMMAND in note
    assert "INGESTED" in studying
    assert "fast_slow_process_router" in studying
    assert studying.count(mod.STUDYING_SECTION_START) == 1

    mod.write_research_studying_section(studying_path, artifact)
    updated = studying_path.read_text(encoding="utf-8")
    assert updated.count(mod.STUDYING_SECTION_START) == 1


def test_research_studying_writer_can_build_default_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-KONA-4962: studying writer has a deterministic default artifact path."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        domain_slice_path=_write_slice(tmp_path / "slice.jsonl"),
        net_available=True,
    )
    studying_path = tmp_path / "research-studying.md"

    monkeypatch.setattr(mod, "build_artifact", lambda: artifact)
    mod.write_research_studying_section(studying_path)

    studying = studying_path.read_text(encoding="utf-8")
    assert "Exp 4962" in studying
    assert "INGESTED" in studying
