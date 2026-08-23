"""Tests for Exp6555 proof-preserving constraint saturation fixture.

Spec refs: REQ-BENCH-6555, SCENARIO-BENCH-6555-GATE,
SCENARIO-BENCH-6555-SOTA, SCENARIO-BENCH-6555-VARIANTS,
SCENARIO-BENCH-6555-PROOFS, SCENARIO-BENCH-6555-SPLITS,
SCENARIO-BENCH-6555-ATTACKS, SCENARIO-BENCH-6555-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from urllib.error import URLError

from carnot import experiment_6555_proof_preserving_constraint_saturation_fixture as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6555", "exit_code": 0}]


def _availability_rows() -> dict[str, dict[str, Any]]:
    return {
        row["arxiv_id"]: {
            "availability_checked": True,
            "direct_arxiv_available": True,
            "http_status": 200,
            "query_timestamp_utc": "2026-08-23T12:00:00Z",
            "url": row["arxiv_url"],
        }
        for row in mod.SOTA_SOURCE_CATALOG
    }


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "experiment_6555.json",
        fixture_path=tmp_path / "v567_constraint_saturation.jsonl",
        note_path=tmp_path / "v567-constraint-saturation-sota-mapping.md",
        transaction_work_dir=tmp_path / "tx",
        write=write,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
        arxiv_availability=_availability_rows(),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_bench_6555_spec_declares_fixture_contract() -> None:
    """REQ-BENCH-6555: OpenSpec owns the Exp6555 fixture contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6555") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6555-GATE",
        "SCENARIO-BENCH-6555-SOTA",
        "SCENARIO-BENCH-6555-VARIANTS",
        "SCENARIO-BENCH-6555-PROOFS",
        "SCENARIO-BENCH-6555-SPLITS",
        "SCENARIO-BENCH-6555-ATTACKS",
        "SCENARIO-BENCH-6555-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.FIXTURE_RELATIVE_PATH.as_posix(),
        mod.NOTE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenarios_bench_6555_complete_fixture_closes(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6555-GATE/SOTA/VARIANTS/PROOFS/SPLITS/ATOMIC: ready closes."""

    artifact = _artifact(tmp_path)
    fixture_rows = _read_jsonl(tmp_path / "v567_constraint_saturation.jsonl")
    written = json.loads((tmp_path / "experiment_6555.json").read_text(encoding="utf-8"))
    note = (tmp_path / "v567-constraint-saturation-sota-mapping.md").read_text(encoding="utf-8")

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_proof_preserving_constraint_saturation_fixture"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] is None
    assert artifact["constraint_saturation_fixture_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    assert len(artifact["literature_source_rows"]) == 5
    assert all(row["direct_arxiv_available"] for row in artifact["literature_source_rows"])
    assert all(
        row["query_timestamp_utc"] == "2026-08-23T12:00:00Z"
        for row in artifact["literature_source_rows"]
    )
    assert "https://arxiv.org/abs/2608.12426" in note
    assert "Bottom-line fixture contract" in note

    assert len(fixture_rows) == 144
    assert artifact["fixture_path_and_hash"]["row_count"] == 144
    assert artifact["fixture_path_and_hash"]["sha256"] == mod.sha256_file(
        tmp_path / "v567_constraint_saturation.jsonl"
    )
    assert artifact["fixture_path_and_hash"]["roundtrip_checker_passed"] is True
    assert artifact["per_unit_rows"] == fixture_rows

    contract = artifact["frozen_variant_and_split_contract"]
    assert contract["lineage_count"] == 36
    assert contract["domains"] == ["logic_grid", "scheduling", "seating"]
    assert contract["constraint_counts"] == list(range(1, 13))
    assert contract["surfaces"] == ["brief", "table"]
    assert contract["variant_modes"] == ["equivalent", "hardened"]
    assert contract["domain_lineage_counts"] == {
        "logic_grid": 12,
        "scheduling": 12,
        "seating": 12,
    }
    assert contract["split_lineage_counts"] == {
        "development": 12,
        "held": 12,
        "train": 12,
    }

    proofs = artifact["equivalence_and_hardening_proof_rows"]
    assert len(proofs) == len(fixture_rows)
    assert {row["variant_mode"] for row in proofs} == {"equivalent", "hardened"}
    assert all(row["proof_status"] == "passed" for row in proofs)
    assert all(row["terminal_status"] == "terminal" for row in proofs)
    assert all(
        row["solution_set_preserved"] for row in proofs if row["variant_mode"] == "equivalent"
    )
    assert all(
        row["declared_hardening_only"] for row in proofs if row["variant_mode"] == "hardened"
    )
    assert all(
        row["hardening_strictly_narrows"] for row in proofs if row["variant_mode"] == "hardened"
    )

    assert mod.roundtrip_fixture_checkers(fixture_rows)["passed"] is True
    assert artifact["exact_clause_checker_contract"]["roundtrip_checker_passed"] is True
    assert all(row["passed"] for row in artifact["attack_matrix"])


def test_scenario_bench_6555_validation_edges_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6555-ATTACKS: invalid proof, split, and checksum fail closed."""

    artifact = _artifact(tmp_path)

    validations: list[tuple[str, dict[str, Any], str]] = []

    missing = deepcopy(artifact)
    del missing["status"]
    validations.append(("missing", missing, "missing required fields"))

    bad_prefix = deepcopy(artifact)
    bad_prefix["honest_verdict"] = "ready"
    validations.append(("bad_prefix", bad_prefix, "honest_verdict lacks terminal prefix"))

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "llm"
    validations.append(("bad_substrate", bad_substrate, "inference_substrate mismatch"))

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    validations.append(("bad_oracle", bad_oracle, "verifier_is_oracle must be true"))

    failed_attack = deepcopy(artifact)
    failed_attack["attack_matrix"][0]["passed"] = False
    validations.append(
        ("failed_attack", failed_attack, "ready score cannot open with failed attacks")
    )

    bad_proof = deepcopy(artifact)
    bad_proof["equivalence_and_hardening_proof_rows"][0]["proof_status"] = "failed"
    validations.append(("bad_proof", bad_proof, "ready score cannot open with failed proofs"))

    split_leak = deepcopy(artifact)
    split_leak["frozen_variant_and_split_contract"]["lineage_isolation_passed"] = False
    validations.append(("split_leak", split_leak, "lineage isolation failed"))

    bad_fixture_hash = deepcopy(artifact)
    bad_fixture_hash["fixture_path_and_hash"]["sha256"] = "sha256:" + "0" * 64
    validations.append(("bad_fixture_hash", bad_fixture_hash, "fixture hash mismatch"))

    for _name, payload, expected in validations:
        if expected != "fixture hash mismatch":
            payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
        assert any(expected in error for error in mod.validate_artifact(payload)), expected

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(checksum)


def test_scenario_bench_6555_blocked_gate_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6555-GATE: a failed upstream gate cannot generate readiness."""

    gate_payload = deepcopy(mod.load_json(REPO / mod.UPSTREAM_GATE_RELATIVE_PATH))
    gate_payload["v566_external_transfer_eligible_score"] = 0.0
    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        fixture_path=tmp_path / "blocked.jsonl",
        note_path=tmp_path / "blocked.md",
        transaction_work_dir=tmp_path / "tx-blocked",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
        arxiv_availability=_availability_rows(),
        upstream_gate_payload=gate_payload,
    )

    assert blocked["status"] == "blocked_proof_preserving_constraint_saturation_fixture"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["constraint_saturation_fixture_ready_score"] == 0.0
    assert blocked["per_unit_rows"] == []
    assert blocked["fixture_path_and_hash"]["row_count"] == 0
    assert "upstream_gate_ready" in blocked["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_bench_6555_helper_and_partial_edges(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """REQ-BENCH-6555: helper edges stay deterministic and closed."""

    class _Response:
        status = 200

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    calls: list[str] = []

    def _fake_urlopen(request: Any, timeout: float) -> _Response:
        del timeout
        calls.append(request.full_url)
        if len(calls) == 1:
            return _Response()
        raise URLError("offline")

    monkeypatch.setattr(mod, "urlopen", _fake_urlopen)
    availability = mod.check_arxiv_availability("2026-08-23T12:00:00Z", timeout_s=0.01)
    assert availability["2608.12426"]["direct_arxiv_available"] is True
    assert availability["2602.13217"]["direct_arxiv_available"] is False

    assert mod.sha256_file(tmp_path / "missing.txt") == "missing"
    assert mod.load_json(tmp_path / "missing.json") == {}
    assert mod._load_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._utc_now().endswith("Z")
    assert mod._source_root_from_intake(REPO, {}) == mod.DEFAULT_SOURCE_CACHE_ROOT
    assert mod._problem_context({"domain": "unknown"}) == {}
    assert (
        list(
            mod._candidate_hardening_constraints(
                {"domain": "seating", "entities": ["A"]},
                {"gold_solution": []},
            )
            or []
        )
        == []
    )
    (tmp_path / "manual-probe" / "corrupt_resume_probe").mkdir(parents=True)
    assert mod._corrupt_resume_probe(tmp_path / "manual-probe")["corrupt_resume_rejected"]

    partial_availability = {
        row["arxiv_id"]: {
            "availability_checked": True,
            "direct_arxiv_available": False,
            "http_status": 503,
            "query_timestamp_utc": "2026-08-23T12:00:00Z",
            "url": row["arxiv_url"],
        }
        for row in mod.SOTA_SOURCE_CATALOG
    }
    partial = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "partial.json",
        fixture_path=tmp_path / "partial.jsonl",
        note_path=tmp_path / "partial.md",
        transaction_work_dir=tmp_path / "tx-partial",
        write=False,
        duration_s=None,
        tests_run=TESTS_RUN,
        arxiv_availability=partial_availability,
    )
    assert partial["status"] == "partial_proof_preserving_constraint_saturation_fixture"
    assert partial["verdict_class"] == "partial"
    assert partial["honest_verdict"].startswith("partial_")
    assert partial["constraint_saturation_fixture_ready_score"] == 0.0
    assert mod.validate_artifact(partial) == []

    complete = _artifact(tmp_path)
    _artifact(tmp_path)

    more_validations: list[tuple[dict[str, Any], str]] = []
    bad_status = deepcopy(complete)
    bad_status["status"] = "ready"
    more_validations.append((bad_status, "status lacks terminal prefix"))

    bad_class = deepcopy(complete)
    bad_class["verdict_class"] = "positive"
    more_validations.append((bad_class, "verdict_class is outside closed class"))

    bad_provenance = deepcopy(complete)
    bad_provenance["field_provenance"] = {}
    more_validations.append((bad_provenance, "field_provenance must cover required fields"))

    bad_principles = deepcopy(complete)
    bad_principles["field_principles"] = {}
    more_validations.append((bad_principles, "field_principles must cover required fields"))

    bad_aggregate = deepcopy(complete)
    bad_aggregate["aggregate_row_recomputation"]["recomputed_ready_score"] = 0.0
    more_validations.append(
        (bad_aggregate, "ready score must derive from aggregate row recomputation")
    )

    bad_checker = deepcopy(complete)
    bad_checker["exact_clause_checker_contract"]["roundtrip_checker_passed"] = False
    more_validations.append((bad_checker, "exact checker roundtrip failed"))

    bad_protected = deepcopy(complete)
    bad_protected["protected_files_unchanged"]["all_unchanged"] = False
    more_validations.append((bad_protected, "protected files changed"))

    bad_gate_summary = deepcopy(complete)
    bad_gate_summary["gate_check_summary"]["failed_checks"] = ["manual_failure"]
    more_validations.append((bad_gate_summary, "ready score cannot open with failed checks"))

    complete_without_score = deepcopy(complete)
    complete_without_score["constraint_saturation_fixture_ready_score"] = 0.0
    complete_without_score["aggregate_row_recomputation"]["recomputed_ready_score"] = 0.0
    more_validations.append((complete_without_score, "complete status requires ready score 1.0"))

    bad_blocked = deepcopy(complete)
    bad_blocked["status"] = "blocked_proof_preserving_constraint_saturation_fixture"
    bad_blocked["verdict_class"] = "partial"
    more_validations.append((bad_blocked, "blocked status requires blocked verdict_class"))

    for payload, expected in more_validations:
        payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
        assert any(expected in error for error in mod.validate_artifact(payload)), expected
