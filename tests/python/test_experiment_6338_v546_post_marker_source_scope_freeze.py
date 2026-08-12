"""Tests for Exp6338 V546 post-marker source freeze.

Spec refs: REQ-INFRA-6338, SCENARIO-INFRA-6338-1,
SCENARIO-INFRA-6338-2, SCENARIO-INFRA-6338-3,
SCENARIO-INFRA-6338-4, SCENARIO-INFRA-6338-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6338_v546_post_marker_source_scope_freeze as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _references() -> str:
    return (
        "## V545 Planner Refresh (2026-08-12, after milestone 2026.08.544)\n\n"
        "- old exact guard source.\n"
        "<!-- V545-PLANNER-REFRESH-20260812-END -->\n\n"
        "## V546 Planner Refresh (2026-08-12, after milestone 2026.08.545)\n\n"
        "- **The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding** - "
        "arXiv:2608.10137, https://arxiv.org/abs/2608.10137.\n"
        "- **LeJIT: Just-in-Time Logic Enforcement** - HotNets 2025, "
        "https://hhy.ee.princeton.edu/papers/2025_hotnets_lejit.pdf; code at "
        "https://github.com/HongyuHe/LeJIT.\n"
        "- **NxN E-valuation: Hypothesis Certification via a Conformal CRT Null** - "
        "arXiv:2608.06621, https://arxiv.org/abs/2608.06621.\n"
        "- **Why Does CLAUDE.md Keep Growing? Catastrophic Remembering in Agentic Coding** - "
        "arXiv:2608.11095, https://arxiv.org/abs/2608.11095.\n"
        "- **AI Evaluation Should Measure Verification Cost, Not Correctness Alone** - "
        "arXiv:2608.08709, https://arxiv.org/abs/2608.08709.\n"
        "<!-- V546-PLANNER-REFRESH-20260812-END -->\n"
    )


def _make_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel_path in mod.PROTECTED_RELATIVE_PATHS:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == mod.RESEARCH_REFERENCES_RELATIVE_PATH:
            path.write_text(_references(), encoding="utf-8")
        elif (REPO / rel_path).exists():
            shutil.copyfile(REPO / rel_path, path)
        else:
            path.write_text(f"{rel_path.as_posix()} fixture\n", encoding="utf-8")
    return root


def _accepted_candidate() -> mod.JsonDict:
    return {
        "stable_id": "arxiv:2608.99998",
        "paper_identity": "arxiv:2608.99998",
        "repository_identity": None,
        "mechanism": "fresh_prefix_certificate",
        "retired_scope": False,
        "title": "Fresh Post-Marker V546 Prefix Certificate",
        "url": "https://arxiv.org/abs/2608.99998",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-12T14:54:20Z",
        "date_evidence": "submitted one second after the V546 marker commit",
        "scope_effect": "would change the prefix canary only if accepted",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:" + "c" * 64,
    }


def test_req_infra_6338_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6338: OpenSpec records the V546 source freeze contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6338") :]

    for token in (
        "REQ-INFRA-6338",
        "SCENARIO-INFRA-6338-1",
        "SCENARIO-INFRA-6338-2",
        "SCENARIO-INFRA-6338-3",
        "SCENARIO-INFRA-6338-4",
        "SCENARIO-INFRA-6338-5",
        mod.PLANNER_MARKER,
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`llm_call_count=0`",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_6338_marker_boundaries_and_date_parsing() -> None:
    """SCENARIO-INFRA-6338-1: marker and date handling is exclusive."""

    marker = mod.v546_marker_snapshot(REPO)

    assert marker["marker_text"] == mod.PLANNER_MARKER
    assert marker["marker_line"] == 34019
    assert marker["marker_count"] == 1
    assert marker["marker_committed_at_utc"] == mod.MARKER_COMMITTED_AT_UTC

    accepted = _accepted_candidate()
    assert mod.classify_candidate(accepted, reference_text="")["disposition"] == "accepted"

    at_marker = deepcopy(accepted)
    at_marker["source_timestamp"] = mod.MARKER_COMMITTED_AT_UTC
    assert mod.classify_candidate(at_marker, reference_text="")["disposition"] == (
        "cutoff_confound"
    )

    bare_date = deepcopy(accepted)
    bare_date["source_timestamp"] = "2026-08-12"
    assert mod.classify_candidate(bare_date, reference_text="")["disposition"] == (
        "cutoff_confound"
    )

    unstable = deepcopy(accepted)
    unstable["url"] = "https://github.com/search?q=parser+state"
    assert mod.classify_candidate(unstable, reference_text="")["rejection_reason"] == (
        "candidate lacks a stable https URL"
    )


def test_scenario_6338_promoted_receipts_have_dates_and_local_consequences() -> None:
    """SCENARIO-INFRA-6338-2: promoted planner sources keep direct receipts."""

    report = mod.build_report(
        REPO,
        date="20260812",
        duration_s=2.0,
        search_completed_utc="2026-08-12T15:34:33Z",
    )

    promoted = report["promoted_findings"]
    assert [row["stable_id"] for row in promoted] == [
        "arxiv:2608.10137",
        "github:HongyuHe/LeJIT",
        "arxiv:2608.06621",
        "arxiv:2608.11095",
        "arxiv:2608.08709",
    ]
    for row in promoted:
        assert row["direct_url"].startswith("https://")
        assert row["first_publication_date"]
        assert row["accessed_at_utc"] == "2026-08-12T15:34:33Z"
        assert row["local_executable_consequence"]
        assert row["planner_promoted"] is True

    assert report["parser_bias_receipt"]["stable_id"] == "arxiv:2608.10137"
    assert report["lejit_receipt"]["code_url"] == "https://github.com/HongyuHe/LeJIT"
    assert report["nxn_evalue_receipt"]["first_publication_utc"].startswith("2026-08-06")
    assert report["catastrophic_remembering_receipt"]["local_consequence"].startswith(
        "evidence-carrying"
    )
    assert "verification-cost" in report["verification_cost_receipt"]["local_consequence"]


def test_scenario_6338_duplicate_watch_inaccessible_and_scope_hashes(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6338-3: dedupe and protected-file checks fail closed."""

    root = _make_repo(tmp_path / "repo")
    accepted = _accepted_candidate()

    duplicate_paper = deepcopy(accepted)
    duplicate_paper["stable_id"] = "fresh-paper-repeat"
    duplicate_paper["paper_identity"] = "arxiv:2608.10137"

    duplicate_repo = deepcopy(accepted)
    duplicate_repo["stable_id"] = "github:repeat"
    duplicate_repo["paper_identity"] = "github:repeat"
    duplicate_repo["repository_identity"] = "github:HongyuHe/LeJIT"

    duplicate_mechanism = deepcopy(accepted)
    duplicate_mechanism["stable_id"] = "mechanism:repeat"
    duplicate_mechanism["paper_identity"] = "mechanism:repeat"
    duplicate_mechanism["mechanism"] = "parser_state_bias_correction"

    watch_only = deepcopy(accepted)
    watch_only["stable_id"] = "github:MVPandey/Enso"
    watch_only["paper_identity"] = "github:MVPandey/Enso"
    watch_only["repository_identity"] = "github:MVPandey/Enso"
    watch_only["mechanism"] = "third_party_kona_replication"
    watch_only["watch_only"] = True
    watch_only["content_hash"] = "sha256:" + "e" * 64

    inaccessible = deepcopy(accepted)
    inaccessible["stable_id"] = "arxiv:window-query"
    inaccessible["paper_identity"] = "arxiv:window-query"
    inaccessible["inaccessible"] = True
    inaccessible["content_hash"] = "sha256:" + "f" * 64

    retired = deepcopy(accepted)
    retired["stable_id"] = "retired:hidden-state"
    retired["paper_identity"] = "retired:hidden-state"
    retired["mechanism"] = "hidden_state_probe"
    retired["retired_scope"] = True
    retired["content_hash"] = "sha256:" + "a" * 64

    no_consequence = deepcopy(accepted)
    no_consequence["stable_id"] = "no:consequence"
    no_consequence["paper_identity"] = "no:consequence"
    no_consequence["local_executable_consequence"] = False
    no_consequence["content_hash"] = "sha256:" + "b" * 64

    partitions = mod.partition_candidates(
        [
            duplicate_paper,
            duplicate_repo,
            duplicate_mechanism,
            accepted,
            watch_only,
            inaccessible,
            retired,
            no_consequence,
        ],
        reference_text=(root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8"),
    )

    assert [row["stable_id"] for row in partitions["accepted"]] == [accepted["stable_id"]]
    assert [row["stable_id"] for row in partitions["duplicate_findings"]] == [
        "fresh-paper-repeat",
        "github:repeat",
        "mechanism:repeat",
    ]
    assert partitions["watch_only_findings"][0]["rejection_reason"].startswith("watch-only")
    assert partitions["inaccessible_sources"][0]["rejection_reason"].startswith("source endpoint")
    assert [row["stable_id"] for row in partitions["excluded_findings_and_reasons"]] == [
        "retired:hidden-state",
        "no:consequence",
    ]

    before = mod.protected_hashes(root)
    (root / "CODEX.md").write_text("changed\n", encoding="utf-8")
    changed = mod.protected_unchanged(root, before)
    assert changed["all_unchanged"] is False
    assert changed["paths"]["CODEX.md"]["unchanged"] is False


def test_scenario_6338_frozen_contracts_and_zero_delta_report() -> None:
    """SCENARIO-INFRA-6338-4: V546 lanes and model policy stay frozen."""

    report = mod.build_report(
        REPO,
        date="20260812",
        duration_s=2.0,
        search_completed_utc="2026-08-12T15:34:33Z",
    )

    assert report["status"] == "complete_null"
    assert report["accepted_count"] == 0
    assert isinstance(report["accepted_count"], int)
    assert report["llm_call_count"] == 0
    assert isinstance(report["llm_call_count"], int)
    assert report["roadmap_scope_delta"]["delta_kind"] == "zero_source_delta"
    assert report["roadmap_scope_delta"]["new_lane_count"] == 0

    prefix = report["frozen_prefix_generation_contract"]
    assert prefix["version"] == mod.CONTRACT_VERSION
    assert prefix["allowed_methods"] == ["parser_state_correction", "jit_smt_prefix_feasibility"]
    assert prefix["post_hoc_energy_search_allowed"] is False

    certified = report["frozen_certified_learning_contract"]
    assert certified["anytime_evalue_ledger_required"] is True
    assert certified["gguf_weight_update_allowed"] is False

    arc = report["frozen_arc_influence_contract"]
    assert arc["solve_credit_allowed"] is False
    assert arc["default_off"] is True

    model_policy = report["frozen_model_policy"]
    assert set(model_policy["mandatory_hf_ids"]) == set(mod.MANDATED_GGUF_IDS)
    assert model_policy["llm_call_count_for_exp6338"] == 0

    hardware = report["frozen_hardware_nonuse_contract"]
    assert hardware["board_execution_authorized"] is False
    assert hardware["gatemate_command_count"] == 0
    assert hardware["excluded_hardware"] == ["GateMate", "KV260", "TSU", "Kona"]


def test_scenario_6338_schema_write_and_validation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6338-5: output is checksummed, typed, and atomic."""

    root = _make_repo(tmp_path / "repo")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    before_text = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes()

    report = mod.write_freeze(
        root,
        date="20260812",
        duration_s=2.0,
        search_completed_utc="2026-08-12T15:34:33Z",
        env={ARTIFACT_ROOT_ENV: str(artifact_root)},
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )

    target = artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes() == before_text
    assert mod.validate_report(report) == []
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["source_queries_by_channel"]) == set(mod.REQUIRED_SOURCE_CHANNELS)
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["honest_verdict"].startswith("complete_null:")

    for mutator, error in (
        (lambda data: data.pop("status"), "missing required field: status"),
        (lambda data: data.update({"accepted_count": {"value": 0}}), "accepted_count"),
        (lambda data: data.update({"llm_call_count": 1}), "llm_call_count"),
        (lambda data: data.update({"field_principles": {}}), "missing field_principles"),
        (
            lambda data: data.update({"source_queries_by_channel": {}}),
            "source_queries_by_channel",
        ),
        (
            lambda data: data.update({"search_window_start_utc": "2026-08-12T14:54:18Z"}),
            "search_window_start_utc",
        ),
        (
            lambda data: data["frozen_hardware_nonuse_contract"].update(
                {"board_execution_authorized": True}
            ),
            "frozen_hardware_nonuse_contract",
        ),
        (lambda data: data.update({"honest_verdict": "ok"}), "honest_verdict"),
        (lambda data: data.update({"status": "complete_delta"}), "status"),
    ):
        bad = deepcopy(report)
        mutator(bad)
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert any(error in err for err in mod.validate_report(bad))

    bad = deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad)

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=mod.REPO_ROOT, write=True, command_receipts=None: {
            "status": f"complete-{date}"
        },
    )
    assert mod.main(["--date", "20260812"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out

    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", tmp_path / "missing.json")
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    receipt_path.write_text(json.dumps({"focused": 0, "full": 3}), encoding="utf-8")
    assert mod.read_external_test_receipts() == [
        {"command": "focused", "exit_code": 0},
        {"command": "full", "exit_code": 3},
    ]

    receipt_path.write_text("{bad", encoding="utf-8")
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    with pytest.raises(ValueError, match="invalid Exp6338 freeze"):
        mod.write_report({"status": "complete"}, root, env={ARTIFACT_ROOT_ENV: str(artifact_root)})


def test_scenario_6338_helper_edges_and_run_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6338: helper edges stay explicit and non-mutating."""

    assert mod._read_text(tmp_path / "missing.md") == ""
    assert mod._parse_timestamp("not-a-time") is None
    assert mod._parse_timestamp("2026-08-12T14:54:20").tzinfo is not None
    assert mod._is_stable_url("ftp://example.com/file") is False

    accepted = _accepted_candidate()
    for field, value, reason in (
        ("reproducible_evidence", False, "candidate lacks reproducible evidence"),
        ("primary_or_first_party", False, "candidate is not primary or first-party"),
    ):
        row = deepcopy(accepted)
        row[field] = value
        assert mod.classify_candidate(row, reference_text="")["rejection_reason"] == reason

    for seen_key, seen_value, reason in (
        ("seen_papers", {accepted["paper_identity"]}, "paper identity repeated in this sweep"),
        (
            "seen_repositories",
            {"github:fresh/repo"},
            "repository identity repeated in this sweep",
        ),
        (
            "seen_mechanisms",
            {accepted["mechanism"]},
            "mechanism repeated in this sweep",
        ),
        ("seen_hashes", {accepted["content_hash"]}, "content hash repeated in this sweep"),
    ):
        row = deepcopy(accepted)
        if seen_key == "seen_repositories":
            row["repository_identity"] = "github:fresh/repo"
        kwargs = {
            "reference_text": "",
            "seen_papers": set(),
            "seen_repositories": set(),
            "seen_mechanisms": set(),
            "seen_hashes": set(),
            seen_key: seen_value,
        }
        assert mod.classify_candidate(row, **kwargs)["rejection_reason"] == reason

    for field, value, message in (
        ("url", None, "missing fields"),
        ("url", "ftp://example.com/file", "stable URL"),
        ("content_hash", "sha256:bad", "content hash"),
        ("source_timestamp", mod.MARKER_COMMITTED_AT_UTC, "strictly after"),
        ("reproducible_evidence", False, "reproducible"),
        ("primary_or_first_party", False, "primary"),
        ("local_executable_consequence", False, "local executable"),
        ("watch_only", True, "watch-only"),
        ("retired_scope", True, "retired scope"),
    ):
        row = deepcopy(accepted)
        row[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_accepted_candidate(row)

    repo_candidate = deepcopy(accepted)
    repo_candidate["repository_identity"] = "github:fresh/repo"
    partitions = mod.partition_candidates([repo_candidate], reference_text="")
    assert partitions["accepted"][0]["repository_identity"] == "github:fresh/repo"

    delta_report = mod.build_report(
        _make_repo(tmp_path / "delta-repo"),
        date="20260812",
        candidates=[accepted],
        duration_s=2.0,
        search_completed_utc="2026-08-12T15:34:33Z",
    )
    assert delta_report["status"] == "complete_delta"
    assert delta_report["roadmap_scope_delta"]["delta_kind"] == "accepted_source_delta"

    report = mod.build_report(
        REPO,
        date="20260812",
        duration_s=2.0,
        search_completed_utc="2026-08-12T15:34:33Z",
    )
    for mutator, error in (
        (lambda data: data.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda data: data.update({"verifier_is_oracle": True}), "verifier_is_oracle"),
        (lambda data: data.update({"field_provenance": {}}), "missing field_provenance"),
        (
            lambda data: data.update({"search_completed_utc": "2026-08-12T14:54:19Z"}),
            "search_completed_utc",
        ),
        (
            lambda data: data["protected_files_unchanged"].update({"all_unchanged": False}),
            "protected_files_unchanged",
        ),
        (
            lambda data: data["frozen_prefix_generation_contract"].update({"version": "bad"}),
            "frozen_prefix_generation_contract",
        ),
    ):
        bad = deepcopy(report)
        mutator(bad)
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert any(error in err for err in mod.validate_report(bad))

    receipt_path = tmp_path / "list-receipts.json"
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    receipt_path.write_text(json.dumps([{"command": "focused", "exit_code": 0}]), encoding="utf-8")
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    writes: list[dict[str, object]] = []

    def fake_write_report(
        report: dict[str, object], root: Path = REPO, *, env: object = None
    ) -> Path:
        writes.append(report)
        return tmp_path / mod.RESULT_RELATIVE_PATH.name

    monkeypatch.setattr(mod, "write_report", fake_write_report)
    run_report = mod.run(
        date="20260812",
        root=REPO,
        write=True,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert writes and run_report["status"] == "complete_null"

    no_write_report = mod.run(
        date="20260812",
        root=REPO,
        write=False,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert no_write_report["status"] == "complete_null"
