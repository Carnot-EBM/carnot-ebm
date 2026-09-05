"""Tests for REQ-REPORT-7011 and SCENARIO-REPORT-7011-* contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess
import sys
from urllib.error import HTTPError, URLError

from carnot import experiment_7011_v614_sota_ingestion as mod


ROOT = Path(__file__).resolve().parents[2]


def _passing_commands() -> list[dict[str, object]]:
    """Build terminal receipts without running child commands in unit tests."""

    return [
        {
            "name": name,
            "command": name,
            "exit_code": 0,
            "outcome": "pass",
            "terminal": True,
            "passed": True,
        }
        for name in mod.VALIDATION_COMMAND_NAMES
    ]


def _artifact(tmp_path: Path) -> dict[str, object]:
    """Build the complete deterministic receipt against the real read-only inputs."""

    return mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "artifact.json",
        network_available=True,
        command_rows=_passing_commands(),
        duration_s=1.0,
    )


def _refresh_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def _fake_root(tmp_path: Path, *, marker: bool = True) -> Path:
    """Create the smallest isolated tree accepted by the preflight."""

    root = tmp_path / "repo"
    for relative in mod.REQUIRED_LOCAL_PATHS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = "readable target\n"
        if relative == mod.REFERENCE_PATH:
            text = f"{mod.REFERENCE_MARKER}\n" if marker else "older marker\n"
        target.write_text(text, encoding="utf-8")
    return root


def test_req_report_7011_spec_precedes_implementation() -> None:
    """REQ-REPORT-7011 owns all source, map, boundary, and artifact scenarios."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7011") :]
    for scenario in (
        "PREFLIGHT",
        "SOURCES",
        "MAP",
        "NONCLAIMS",
        "SECONDARY",
        "NOCHANGE",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-7011-{scenario}" in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_7011_sources_and_release_identities_are_terminal() -> None:
    """SCENARIO-REPORT-7011-SOURCES preserves direct access and immutable code IDs."""

    queries = mod.source_query_rows()
    primaries = mod.primary_source_rows()
    repositories = mod.repository_rows()
    identities = mod.release_identity_rows()

    assert {row["source_id"] for row in primaries} == set(mod.SOURCE_IDS)
    assert {row["source_id"] for row in repositories} == set(mod.SOURCE_IDS)
    assert all(row["url"].startswith("https://") for row in queries + primaries + repositories)
    assert all(row["publication_or_update_date"] for row in primaries)
    assert all(row["access_outcome"] and row["terminal"] is True for row in queries)
    assert all(row["access_outcome"] and row["terminal"] is True for row in primaries)
    assert all(row["method_extraction"] for row in primaries)
    assert all(row["terminal"] is True for row in repositories + identities)
    assert all(
        row["identity"] and row["identity_type"] in {"git_commit", "hf_revision"}
        for row in identities
        if row["available"]
    )
    assert (
        next(row for row in repositories if row["source_id"] == "introconformal")["access_outcome"]
        == "empty_repository_no_commit"
    )
    assert (
        next(row for row in queries if row["query_id"] == "batchsum_openreview")["access_outcome"]
        == "challenge_required"
    )


def test_scenario_report_7011_method_map_has_inputs_boundaries_controls_and_falsifiers() -> None:
    """SCENARIO-REPORT-7011-MAP maps only accepted software mechanisms."""

    methods = mod.method_rows()
    mappings = mod.method_to_module_rows()
    accepted = {row["method_id"] for row in methods if row["accepted"]}

    assert accepted == set(mod.ACCEPTED_METHOD_IDS)
    assert {row["classification"] for row in methods} == {
        "direct_adaptation",
        "architectural_analogy",
        "watch_only_dependency",
        "unsupported_hardware_claim",
    }
    assert {row["method_id"] for row in mappings} == accepted
    assert {row["method_id"] for row in mod.leakage_boundary_rows()} == accepted
    assert {row["method_id"] for row in mod.control_rows()} == accepted
    assert {row["method_id"] for row in mod.falsification_rows()} == accepted
    for row in mappings:
        assert row["required_inputs"]
        assert all((ROOT / path).is_file() for path in row["target_modules"])
    for row in mod.leakage_boundary_rows():
        assert row["learner_allowed_inputs"]
        assert row["learner_excluded_inputs"]
        assert row["exact_authority_external"] is True
    assert all(row["control_arm"] for row in mod.control_rows())
    assert all(row["failure_condition"] for row in mod.falsification_rows())


def test_scenario_report_7011_dependencies_nonclaims_and_secondary_checks_are_explicit() -> None:
    """SCENARIO-REPORT-7011-NONCLAIMS and SECONDARY stop source-to-local drift."""

    dependencies = mod.unsupported_dependency_rows()
    non_claims = mod.non_claim_rows()
    secondary = mod.secondary_check_rows()

    assert {row["source_id"] for row in dependencies} >= {
        "bbwm",
        "introconformal",
        "z1t",
    }
    assert all(row["dependency"] and row["local_effect"] for row in dependencies)
    assert {row["claim_id"] for row in non_claims} >= {
        "no_arc_solve_claim",
        "no_oracle_distinct_claim",
        "no_z1_execution_claim",
        "no_z1_efficiency_claim",
    }
    assert all(row["prohibited_claim"] and row["reason"] for row in non_claims)
    assert {row["check_id"] for row in secondary} == {"ebt_citations", "arm_ebm_citations", "kona"}
    assert all(row["terminal"] is True and row["access_outcome"] for row in secondary)
    assert all(row["citation_count_claimed"] is False for row in secondary[:2])
    assert (
        next(row for row in secondary if row["check_id"] == "kona")["public_implementation_claimed"]
        is False
    )


def test_scenario_report_7011_nochange_preserves_the_reference_ledger(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7011-NOCHANGE emits no_change without writing the ledger."""

    before = (ROOT / mod.REFERENCE_PATH).read_bytes()
    rows = mod.reference_append_rows()

    assert rows == [
        {
            "action": "no_change",
            "marker": mod.PLANNER_MARKER,
            "appended": False,
            "reason": "No relevant primary or first-party artifact change was proved after the V614 marker.",
            "terminal": True,
        }
    ]
    artifact = _artifact(tmp_path)
    assert artifact["reference_append_rows"] == rows
    assert (ROOT / mod.REFERENCE_PATH).read_bytes() == before


def test_scenario_report_7011_preflight_blocks_with_exact_diagnostics(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7011-PREFLIGHT uses a complete blocked artifact."""

    root = _fake_root(tmp_path)
    checks = mod.check_preconditions(
        root,
        tmp_path / "ok" / "artifact.json",
        network_available=True,
    )
    assert all(row["passed"] for row in checks)

    missing_marker = _fake_root(tmp_path / "missing", marker=False)
    artifact = mod.build_artifact(
        missing_marker,
        "20260905",
        output_path=tmp_path / "blocked.json",
        network_available=True,
        command_rows=_passing_commands(),
        duration_s=0.1,
    )
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["v614_sota_ingestion_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_v614_sota_ingestion"
    assert artifact["gate_check_summary"] == {
        "failed_check": "v614_reference_marker",
        "expected_value": mod.REFERENCE_MARKER,
        "observed_value": "missing",
        "passed": False,
    }
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7011_artifact_recomputes_score_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7011-ARTIFACT validates coverage, score, class, and checksum."""

    artifact = _artifact(tmp_path)
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["v614_sota_ingestion_complete_score"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive_")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    mutations = {
        "field_principles": {},
        "primary_source_rows": artifact["primary_source_rows"][:-1],
        "release_identity_rows": artifact["release_identity_rows"][:-1],
        "method_to_module_rows": artifact["method_to_module_rows"][:-1],
        "control_rows": artifact["control_rows"][:-1],
        "v614_sota_ingestion_complete_score": 0,
        "inference_substrate": "wrong",
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_unfinished",
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert mod.validate_artifact(_refresh_checksum(changed)), field

    missing = deepcopy(artifact)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_fields:rows"]
    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(bad_checksum)


def test_req_report_7011_writer_validator_and_cli(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-7011 writes one artifact and exposes a validation CLI."""

    artifact = _artifact(tmp_path)
    path = tmp_path / "nested" / "result.json"
    assert mod.write_artifact(artifact, path) == path
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(path) == []
    assert mod.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]

    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: deepcopy(artifact))
    monkeypatch.setattr(mod, "run_validation_commands", lambda *_args: _passing_commands()[:-1])
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    writes: list[Path] = []
    monkeypatch.setattr(
        mod, "write_artifact", lambda _artifact, target: writes.append(target) or target
    )
    assert mod.main(["--date", "20260905", "--output", str(path)]) == 0
    assert writes == [path, path]
    assert mod.main(["--date", "bad", "--output", str(path)]) == 2

    monkeypatch.setattr(sys, "argv", [mod.MODULE_PATH.as_posix(), "--validate", str(path)])
    with __import__("pytest").raises(SystemExit, match="0"):
        runpy.run_path(ROOT / mod.MODULE_PATH, run_name="__main__")


def test_req_report_7011_precondition_probe_error_paths(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-REPORT-7011-PREFLIGHT distinguishes network and file failures."""

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(mod, "urlopen", lambda *_args, **_kwargs: Response())
    assert mod._network_available() is True
    monkeypatch.setattr(
        mod,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            HTTPError("https://arxiv.org", 403, "challenge", {}, None)
        ),
    )
    assert mod._network_available() is True
    monkeypatch.setattr(
        mod,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    assert mod._network_available() is False

    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod._writable(tmp_path / "artifact.json") is False

    class BrokenFile:
        def is_file(self) -> bool:
            return True

        def read_bytes(self) -> bytes:
            raise OSError("unreadable")

    assert mod._readable_nonempty(BrokenFile()) is False  # type: ignore[arg-type]

    root = _fake_root(tmp_path / "marker-error")
    original = Path.read_text

    def fail_marker(path: Path, *args: object, **kwargs: object) -> str:
        if path == root / mod.REFERENCE_PATH:
            raise OSError("unreadable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_marker)
    rows = mod.check_preconditions(
        root,
        tmp_path / "marker-error.json",
        network_available=False,
    )
    assert next(row for row in rows if row["check"] == "v614_reference_marker")["passed"] is False


def test_req_report_7011_validator_type_and_io_failures(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7011-ARTIFACT fails closed on malformed stored shapes."""

    artifact = _artifact(tmp_path)
    assert mod._command_state({"command_receipt_rows": []}) == "not_recorded"
    assert mod._command_state({"command_receipt_rows": ()}) == "blocked"
    assert mod._command_state({"command_receipt_rows": _passing_commands()[:-1]}) == "blocked"
    cases = (
        ("method_rows", ()),
        ("method_rows", []),
        ("method_to_module_rows", ()),
        ("primary_source_rows", ()),
        ("repository_rows", ()),
    )
    for field, value in cases:
        changed = deepcopy(artifact)
        changed[field] = value
        assert mod.completion_score(changed) == 0

    empty_principle = deepcopy(artifact)
    empty_principle["field_principles"]["rows"] = ""
    assert "field_principles_empty" in mod.validate_artifact(_refresh_checksum(empty_principle))

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.validate_artifact(malformed) == ["artifact_unreadable"]
    not_object = tmp_path / "list.json"
    not_object.write_text("[]\n", encoding="utf-8")
    assert mod.validate_artifact(not_object) == ["artifact_not_object"]
    with __import__("pytest").raises(ValueError, match="missing_required_fields"):
        mod.write_artifact({}, tmp_path / "invalid.json")


def test_req_report_7011_command_receipts_cover_success_defect_and_tool_failures(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-REPORT-7011 keeps every validation process outcome explicit."""

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["check"], 0, "ok", ""),
    )
    assert mod._run_command(ROOT, "ok", ["check"])["outcome"] == "pass"
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["check"], 1, "finding", "details"),
    )
    assert mod._run_command(ROOT, "defect", ["check"])["outcome"] == "contract_defect"
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["adversarial"], 1, '{"reports":[{"max_severity":1}]}', ""
        ),
    )
    warning = mod._run_command(ROOT, "adversarial_verification", ["adversarial"])
    assert warning["outcome"] == "pass_with_warning" and warning["passed"] is True
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["adversarial"], 1, '{"reports":[{"max_severity":2}]}', ""
        ),
    )
    assert (
        mod._run_command(ROOT, "adversarial_verification", ["adversarial"])["outcome"]
        == "contract_defect"
    )
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["adversarial"], 1, "not json", ""),
    )
    assert (
        mod._run_command(ROOT, "adversarial_verification", ["adversarial"])["outcome"]
        == "contract_defect"
    )
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["check"], 1, output="partial", stderr="late")
        ),
    )
    assert mod._run_command(ROOT, "timeout", ["check"])["outcome"] == "tool_timeout"
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    assert mod._run_command(ROOT, "missing", ["check"])["terminal"] is False

    names: list[str] = []
    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda _root, name, argv: (
            names.append(name)
            or {
                "name": name,
                "command": " ".join(argv),
                "exit_code": 0,
                "outcome": "pass",
                "terminal": True,
                "passed": True,
            }
        ),
    )
    rows = mod.run_validation_commands(ROOT, tmp_path / "artifact.json")
    assert names == list(mod.VALIDATION_COMMAND_NAMES[:-1])
    assert len(rows) == len(mod.VALIDATION_COMMAND_NAMES) - 1


def test_req_report_7011_validation_receipts_control_the_final_verdict(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7011 does not report positive when a required command failed."""

    failed = _passing_commands()
    failed[0] = failed[0] | {"exit_code": 1, "outcome": "contract_defect", "passed": False}
    disqualified = mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "disqualified.json",
        network_available=True,
        command_rows=failed,
        duration_s=1.0,
    )
    assert disqualified["v614_sota_ingestion_complete_score"] == 1
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["gate_check_summary"]["failed_check"] == "validation_commands"
    assert mod.validate_artifact(disqualified) == []

    nonterminal = _passing_commands()
    nonterminal[-1] = nonterminal[-1] | {"terminal": False, "passed": False}
    blocked = mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "blocked-tool.json",
        network_available=True,
        command_rows=nonterminal,
        duration_s=1.0,
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["observed_value"] == "nonterminal_or_missing"
    assert mod.validate_artifact(blocked) == []
