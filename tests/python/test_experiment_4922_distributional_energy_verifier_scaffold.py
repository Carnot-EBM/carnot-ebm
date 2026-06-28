"""Tests for Exp 4922 distributional-energy verifier pivot scaffold.

Spec refs: REQ-KONA-4922, SCENARIO-KONA-4922-DRY-RUN,
SCENARIO-KONA-4922-BLOCKED, SCENARIO-KONA-4922-NO-WIN-CLAIM.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_4922_distributional_energy_verifier_scaffold as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase3-kona" / "spec.md"


def _candidate(
    candidate_id: str,
    answer: str,
    *,
    sample_count: int,
    quality: float,
    penalty: float,
    uncertainty: float,
    judge: float,
    correct: bool,
    model_id: str = "cached-generator-a",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "answer": answer,
        "sample_count": sample_count,
        "learned_quality_mean": quality,
        "deterministic_constraint_penalty": penalty,
        "uncertainty": uncertainty,
        "llm_judge_score": judge,
        "label_correct": correct,
        "model_id": model_id,
    }


def _row(problem_id: str = "tp-test-1") -> dict[str, object]:
    return {
        "problem_id": problem_id,
        "domain": "travelplanner_style_structured_reasoning",
        "oracle_distinct": True,
        "cheap_executable_oracle_available": False,
        "candidates": [
            _candidate(
                "sc-majority",
                "invalid hotel-heavy plan",
                sample_count=3,
                quality=0.55,
                penalty=0.42,
                uncertainty=0.12,
                judge=0.90,
                correct=False,
            ),
            _candidate(
                "energy-best",
                "constraint-respecting plan",
                sample_count=2,
                quality=0.88,
                penalty=0.03,
                uncertainty=0.05,
                judge=0.70,
                correct=True,
            ),
        ],
    }


def _write_slice(path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    rows = rows if rows is not None else [_row("tp-test-1"), _row("tp-test-2")]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n\n",
        encoding="utf-8",
    )
    return path


def test_req_kona_4922_spec_declares_scaffold_contract() -> None:
    """REQ-KONA-4922: OpenSpec anchors paths, fields, and guardrails."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    start = section.index("### REQ-KONA-4922")
    end = section.index("## Latent Symbol Bridge Falsification", start)
    req = section[start:end]

    for marker in (
        "REQ-KONA-4922",
        "SCENARIO-KONA-4922-DRY-RUN",
        "SCENARIO-KONA-4922-BLOCKED",
        "SCENARIO-KONA-4922-NO-WIN-CLAIM",
        mod.RESULT_RELATIVE_PATH,
        mod.DEFAULT_DOMAIN_SLICE_RELATIVE_PATH,
        mod.HARNESS_SKELETON_PATH,
        mod.ARXIV_ID,
        "distributional_energy_verifier",
        "self_consistency",
        "llm_judge",
        "verifier_is_oracle=false",
        "self_consistency_saturated=false",
        "no_verifier_win_claimed=true",
        "scripts/research_conductor.py",
    ):
        assert marker in req
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in req
        assert principle["principle"] in req


def test_req_kona_4922_distributional_energy_formula_ignores_model_identity() -> None:
    """REQ-KONA-4922: energy uses quality, constraints, and uncertainty only."""

    candidate = _candidate(
        "c",
        "plan",
        sample_count=1,
        quality=0.80,
        penalty=0.20,
        uncertainty=0.10,
        judge=0.50,
        correct=True,
        model_id="generator-a",
    )
    identity_mutated = dict(candidate) | {"model_id": "generator-b"}

    assert mod.distributional_energy(candidate) == pytest.approx(-0.50)
    assert mod.distributional_energy(identity_mutated) == pytest.approx(-0.50)


def test_scenario_kona_4922_dry_run_emits_required_three_columns() -> None:
    """SCENARIO-KONA-4922-DRY-RUN: each row has the three comparison columns."""

    dry_run = mod.run_dry_run([_row("tp-test-1")], limit=1)

    assert dry_run["columns"] == list(mod.THREE_COMPARISON_COLUMNS)
    assert dry_run["n_rows"] == 1
    scored = dry_run["rows"][0]
    assert set(mod.THREE_COMPARISON_COLUMNS) <= set(scored)
    assert scored["distributional_energy_verifier"]["selected_candidate_id"] == "energy-best"
    assert scored["self_consistency"]["selected_candidate_id"] == "sc-majority"
    assert scored["llm_judge"]["selected_candidate_id"] == "sc-majority"
    assert "NOT a headline" in dry_run["dry_run_note"]


def test_scenario_kona_4922_success_artifact_has_guardrail_fields(tmp_path: Path) -> None:
    """SCENARIO-KONA-4922-NO-WIN-CLAIM: success artifact is scaffold-only."""

    domain_slice = _write_slice(tmp_path / "slice.jsonl")
    artifact = mod.build_artifact(repo_root=REPO, domain_slice_path=domain_slice)

    mod.validate_artifact(artifact)
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["pivot_executable_on_6_30"] is True
    assert artifact["harness_skeleton_path"] == mod.HARNESS_SKELETON_PATH
    assert artifact["arxiv_id_cited"] == mod.ARXIV_ID
    assert artifact["verifier_is_oracle"] is False
    assert artifact["self_consistency_saturated"] is False
    assert artifact["no_verifier_win_claimed"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["comparison_stubbed"] is True
    assert artifact["duration_s"] == mod.DURATION_S
    assert artifact["preconditions_checked"]["fover_harness_present"] is True
    assert artifact["preconditions_checked"]["domain_slice_present"] is True
    assert artifact["preconditions_checked"]["domain_slice_non_saturated"] is True
    assert artifact["validation_gate"]["ci95_excludes_zero_required"] is True
    assert artifact["validation_gate"]["adversarial_verify_no_model_identity_shortcut_required"] is True
    assert artifact["validation_gate"]["oracle_distinct_required"] is True


def test_scenario_kona_4922_blocked_missing_domain_slice(tmp_path: Path) -> None:
    """SCENARIO-KONA-4922-BLOCKED: missing slice blocks honestly."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        domain_slice_path=tmp_path / "missing.jsonl",
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_domain_slice_missing"
    assert artifact["pivot_executable_on_6_30"] is False
    assert artifact["preconditions_checked"]["domain_slice_present"] is False
    assert artifact["dry_run_three_columns"]["columns"] == list(mod.THREE_COMPARISON_COLUMNS)
    assert artifact["dry_run_three_columns"]["rows"] == []


def test_scenario_kona_4922_blocked_missing_fover_harness(tmp_path: Path) -> None:
    """SCENARIO-KONA-4922-BLOCKED: missing FoVer harness blocks honestly."""

    domain_slice = _write_slice(tmp_path / "slice.jsonl")
    artifact = mod.build_artifact(repo_root=tmp_path, domain_slice_path=domain_slice)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_fover_harness_missing"
    assert artifact["pivot_executable_on_6_30"] is False
    assert artifact["preconditions_checked"]["fover_harness_present"] is False


def test_req_kona_4922_schema_defensive_paths_are_explicit(tmp_path: Path) -> None:
    """REQ-KONA-4922: malformed cached rows fail closed with named errors."""

    with pytest.raises(ValueError, match="candidate list is empty"):
        mod.select_distributional_energy([])
    with pytest.raises(ValueError, match="no rows"):
        mod.validate_rows([])
    with pytest.raises(ValueError, match="no candidates"):
        mod.validate_rows([{"problem_id": "bad", "cheap_executable_oracle_available": False}])
    with pytest.raises(ValueError, match="oracle-distinct"):
        mod.validate_rows(
            [
                {
                    "problem_id": "bad",
                    "cheap_executable_oracle_available": True,
                    "candidates": [_candidate("c", "a", sample_count=1, quality=0, penalty=0, uncertainty=0, judge=0, correct=True)],
                }
            ]
        )
    with pytest.raises(ValueError, match="lacks cached labels"):
        mod.validate_rows(
            [
                {
                    "problem_id": "bad",
                    "cheap_executable_oracle_available": False,
                    "candidates": [{"candidate_id": "c", "answer": "a"}],
                }
            ]
        )
    assert mod.self_consistency_accuracy([]) == 0.0

    invalid_slice = tmp_path / "invalid.jsonl"
    invalid_slice.write_text("{not json}\n", encoding="utf-8")
    invalid_preconditions = mod.check_preconditions(repo_root=REPO, domain_slice_path=invalid_slice)
    assert invalid_preconditions["blocked_resource"] == "domain_slice_invalid"
    assert invalid_preconditions["domain_error"]


def test_scenario_kona_4922_blocks_saturated_slice(tmp_path: Path) -> None:
    """SCENARIO-KONA-4922-BLOCKED: near-ceiling self-consistency is not a moat domain."""

    saturated = _row("tp-saturated")
    saturated["candidates"] = [
        _candidate(
            "majority-correct",
            "correct plan",
            sample_count=5,
            quality=0.9,
            penalty=0.01,
            uncertainty=0.02,
            judge=0.9,
            correct=True,
        ),
        _candidate(
            "minority-wrong",
            "wrong plan",
            sample_count=1,
            quality=0.2,
            penalty=0.4,
            uncertainty=0.2,
            judge=0.2,
            correct=False,
        ),
    ]
    domain_slice = _write_slice(tmp_path / "saturated.jsonl", [saturated])
    artifact = mod.build_artifact(repo_root=REPO, domain_slice_path=domain_slice)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_self_consistency_saturated"
    assert artifact["preconditions_checked"]["blocked_resource"] == "self_consistency_saturated"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"honest_verdict": "not_terminal"}, "honest_verdict"),
        ({"arxiv_id_cited": "9999.99999"}, "arxiv_id_cited"),
        ({"verifier_is_oracle": True}, "verifier_is_oracle"),
        ({"self_consistency_saturated": True}, "self_consistency_saturated"),
        ({"no_verifier_win_claimed": False}, "no_verifier_win_claimed"),
        ({"inference_substrate": "live_llm_inference"}, "inference_substrate"),
        ({"dry_run_three_columns": {"columns": ["distributional_energy_verifier"], "rows": []}}, "columns"),
        ({"validation_gate": "bad"}, "CI95"),
        ({"validation_gate": {"ci95_excludes_zero_required": True}}, "adversarial_verify"),
        (
            {"validation_gate": dict(mod.VALIDATION_GATE) | {"oracle_distinct_required": False}},
            "oracle-distinct",
        ),
        ({"field_principles": {}}, "field_principles"),
    ],
)
def test_validate_artifact_rejects_guardrail_violations(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    """SCENARIO-KONA-4922-NO-WIN-CLAIM: invalid guardrails fail closed."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        domain_slice_path=_write_slice(tmp_path / "slice.jsonl"),
    )
    bad = copy.deepcopy(artifact)
    bad.update(updates)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad)


def test_validate_artifact_rejects_missing_required_fields(tmp_path: Path) -> None:
    """REQ-KONA-4922: artifact schema mismatch is a validator error."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        domain_slice_path=_write_slice(tmp_path / "slice.jsonl"),
    )
    artifact.pop("duration_s")

    with pytest.raises(ValueError, match="fields mismatch"):
        mod.validate_artifact(artifact)


def test_main_writes_stable_result_json(tmp_path: Path) -> None:
    """REQ-KONA-4922: main writes the result artifact path supplied by caller."""

    domain_slice = _write_slice(tmp_path / "slice.jsonl")
    result_path = tmp_path / "artifact.json"

    artifact = mod.main(
        repo_root=REPO,
        domain_slice_path=domain_slice,
        result_path=result_path,
    )

    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["honest_verdict"] == mod.HONEST_VERDICT
