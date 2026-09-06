"""Tests for REQ-REPORT-7077 and SCENARIO-REPORT-7077-* contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError, URLError

import pytest

from carnot import experiment_7077_v620_sota_ingestion as mod


ROOT = Path(__file__).resolve().parents[2]


def _artifact(tmp_path: Path) -> dict[str, object]:
    """Build one positive receipt without making network calls in unit tests."""

    return mod.build_artifact(
        ROOT,
        "20260906",
        output_path=tmp_path / "artifact.json",
        network_available=True,
        source_capture_utc=mod.SOURCE_CAPTURE_UTC,
        duration_s=1.0,
    )


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def _fake_root(tmp_path: Path, *, marker: bool = True) -> Path:
    """Create the minimal isolated input tree accepted by preflight."""

    root = tmp_path / "repo"
    for relative in mod.REQUIRED_LOCAL_PATHS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = "readable input\n"
        if relative == mod.REFERENCE_PATH:
            text = f"{mod.REFERENCE_MARKER}\n" if marker else "older planner marker\n"
        target.write_text(text, encoding="utf-8")
    return root


def test_req_report_7077_spec_precedes_implementation() -> None:
    """REQ-REPORT-7077 owns the preflight, identity, boundary, and mapping rules."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7077") :]
    for scenario in ("PREFLIGHT", "IDENTITY", "INACCESSIBLE", "VENDOR", "MAPPING", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7077-{scenario}" in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_7077_identity_freezes_required_arxiv_sources() -> None:
    """SCENARIO-REPORT-7077-IDENTITY fixes the two primary paper and code identities."""

    rows = mod.primary_source_rows()
    required = {"arxiv:2605.02915", "arxiv:2605.03534"}
    selected = [row for row in rows if row["source_id"] in required]

    assert {row["source_id"] for row in selected} == required
    assert len(selected) == 2
    for row in selected:
        assert row["title"]
        assert row["authors"]
        assert row["source_url"].startswith("https://arxiv.org/abs/")
        assert row["version_or_revision"].startswith("v")
        assert row["capture_utc"] == mod.SOURCE_CAPTURE_UTC
        assert len(row["metadata_receipt_sha256"]) == 64
    self_verify = next(row for row in selected if row["source_id"] == "arxiv:2605.02915")
    sure_rag = next(row for row in selected if row["source_id"] == "arxiv:2605.03534")
    assert self_verify["code_url"] == "https://github.com/phalod-aditya/slm-confidence-signals"
    assert sure_rag["code_url"] is None


def test_scenario_report_7077_identity_rejects_duplicate_and_missing_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7077-IDENTITY fails duplicate IDs and absent identity fields."""

    artifact = _artifact(tmp_path)
    duplicate = deepcopy(artifact)
    duplicate["primary_source_rows"].append(deepcopy(duplicate["primary_source_rows"][0]))
    duplicate["rows"] = mod.combined_rows(duplicate)
    assert any(
        "duplicate primary source_id" in error
        for error in mod.validate_artifact(_refresh(duplicate))
    )

    missing = deepcopy(artifact)
    missing["primary_source_rows"][0]["title"] = ""
    missing["rows"] = mod.combined_rows(missing)
    assert any(
        "missing source identity" in error for error in mod.validate_artifact(_refresh(missing))
    )


def test_scenario_report_7077_queries_record_all_dispositions() -> None:
    """REQ-REPORT-7077 records exact topic families and closed candidate dispositions."""

    queries = mod.query_rows()
    decisions = mod.decision_rows()

    assert {row["query_family"] for row in queries if row["route"] == "arxiv"} == set(
        mod.REQUIRED_QUERY_FAMILIES
    )
    assert all(
        row["exact_query"] and row["capture_utc"] == mod.SOURCE_CAPTURE_UTC for row in queries
    )
    assert {row["disposition"] for row in decisions} >= {"selected", "watch_only", "rejected"}
    assert all(row["decision_basis"] != "search_rank" for row in decisions)
    assert all(row["disposition"] == "duplicate" for row in mod.duplicate_rows())


def test_scenario_report_7077_inaccessible_pages_are_mirrored_nonclaims(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7077-INACCESSIBLE preserves challenged pages without promotion."""

    artifact = _artifact(tmp_path)
    challenged = [row for row in artifact["openreview_rows"] if not row["content_verified"]]

    assert challenged
    assert {row["source_id"] for row in challenged} == {
        row["source_id"] for row in artifact["inaccessible_rows"]
    }
    assert all(row["access_outcome"] == "browser_challenge" for row in challenged)
    assert all(row["claim_allowed"] is False and row["terminal"] is True for row in challenged)

    forged = deepcopy(artifact)
    forged["inaccessible_rows"] = []
    forged["rows"] = mod.combined_rows(forged)
    assert any("inaccessible" in error for error in mod.validate_artifact(_refresh(forged)))


def test_scenario_report_7077_vendor_claims_are_bounded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7077-VENDOR keeps estimates and proprietary code non-scientific."""

    artifact = _artifact(tmp_path)
    extropic = artifact["extropic_rows"][0]
    kona = artifact["kona_rows"][0]

    assert extropic["evidence_class"] == "public_software_and_vendor_estimated_hardware"
    for key in ("runtime_claimed", "power_claimed", "availability_claimed", "speed_claimed"):
        assert extropic[key] is False
    assert kona["classification"] == "proprietary_watch_only"
    assert kona["public_weights_observed"] is False
    assert kona["local_runner_observed"] is False
    assert kona["public_implementation_promoted"] is False

    forged = deepcopy(artifact)
    forged["extropic_rows"][0]["speed_claimed"] = True
    forged["rows"] = mod.combined_rows(forged)
    assert any("vendor-only claim" in error for error in mod.validate_artifact(_refresh(forged)))


def test_scenario_report_7077_mapping_rejects_unsupported_promotion(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7077-MAPPING rejects vendor and repository readiness promotion."""

    artifact = _artifact(tmp_path)
    promoted = {row["hook_id"] for row in artifact["decision_rows"] if row["promoted"]}
    mapped = {row["hook_id"] for row in artifact["task_mapping_rows"]}
    assert promoted == mapped
    assert all(row["task_ids"] or row["defer_decision"] for row in artifact["task_mapping_rows"])

    forged = deepcopy(artifact)
    forged["kona_rows"][0]["public_implementation_promoted"] = True
    forged["rows"] = mod.combined_rows(forged)
    assert any(
        "unsupported implementation promotion" in error
        for error in mod.validate_artifact(_refresh(forged))
    )

    unmapped = deepcopy(artifact)
    unmapped["task_mapping_rows"] = unmapped["task_mapping_rows"][:-1]
    unmapped["rows"] = mod.combined_rows(unmapped)
    assert any(
        "promoted hook mapping" in error for error in mod.validate_artifact(_refresh(unmapped))
    )


def test_scenario_report_7077_preflight_blocks_stale_capture_and_missing_inputs(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7077-PREFLIGHT emits exact current-UTC and input diagnostics."""

    root = _fake_root(tmp_path)
    checks = mod.check_preconditions(
        root,
        tmp_path / "artifact.json",
        "20260906",
        network_available=True,
        source_capture_utc=mod.SOURCE_CAPTURE_UTC,
    )
    assert all(row["passed"] for row in checks)

    stale = mod.build_artifact(
        root,
        "20260906",
        output_path=tmp_path / "stale.json",
        network_available=True,
        source_capture_utc="2026-09-05T23:59:59Z",
        duration_s=0.1,
    )
    assert stale["verdict_class"] == "blocked"
    assert stale["honest_verdict"] == "blocked_v620_sota_ingestion"
    assert stale["gate_check_summary"] == {
        "failed_check": "source_capture_utc",
        "expected_value": "UTC timestamp dated 2026-09-06",
        "observed_value": "2026-09-05T23:59:59Z",
        "passed": False,
    }
    assert mod.validate_artifact(stale) == []

    missing = _fake_root(tmp_path / "missing", marker=False)
    blocked = mod.build_artifact(
        missing,
        "20260906",
        output_path=tmp_path / "blocked.json",
        network_available=True,
        source_capture_utc=mod.SOURCE_CAPTURE_UTC,
        duration_s=0.1,
    )
    assert blocked["gate_check_summary"]["failed_check"] == "v620_reference_marker"
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7077_artifact_recomputes_score_verdict_and_checksum(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7077-ARTIFACT independently rejects forged terminal fields."""

    artifact = _artifact(tmp_path)
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == "web_bibliographic_search_only_no_llm"
    assert artifact["sota_ingestion_complete_score"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"] == "complete_positive_v620_sota_ingestion"
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    mutations = {
        "field_principles": {},
        "query_rows": artifact["query_rows"][:-1],
        "source_capture_utc": "2026-09-05T23:59:59Z",
        "sota_ingestion_complete_score": 0,
        "inference_substrate": "live_llm",
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_v620_sota_ingestion",
    }
    for field, value in mutations.items():
        forged = deepcopy(artifact)
        forged[field] = value
        if field in mod.ROW_COLLECTION_FIELDS:
            forged["rows"] = mod.combined_rows(forged)
        assert mod.validate_artifact(_refresh(forged)), field


def test_write_validate_and_cli_round_trip(tmp_path: Path) -> None:
    """REQ-REPORT-7077 validates before atomic write and exposes the required CLI."""

    artifact = _artifact(tmp_path)
    output = tmp_path / "written.json"
    mod.write_artifact(artifact, output)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    invalid = deepcopy(artifact)
    invalid["reproducibility_checksum"] = "forged"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.write_artifact(invalid, tmp_path / "invalid.json")

    cli_output = tmp_path / "cli.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / mod.SCRIPT_PATH),
            "--date",
            "20260906",
            "--output",
            str(cli_output),
            "--source-capture-utc",
            mod.SOURCE_CAPTURE_UTC,
            "--network-available",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    written = json.loads(cli_output.read_text(encoding="utf-8"))
    assert written["verdict_class"] == "positive"
    assert mod.validate_artifact(written) == []


def test_network_and_filesystem_precondition_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7077-PREFLIGHT covers HTTP and local I/O terminal states."""

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(mod, "urlopen", lambda *_args, **_kwargs: Response())
    assert mod._network_available() is True

    def raise_http(*_args: object, **_kwargs: object) -> None:
        raise HTTPError("https://arxiv.org", 503, "busy", {}, None)

    monkeypatch.setattr(mod, "urlopen", raise_http)
    assert mod._network_available() is True

    def raise_url(*_args: object, **_kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr(mod, "urlopen", raise_url)
    assert mod._network_available() is False

    source = tmp_path / "source.txt"
    source.write_text("x", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def unreadable(path: Path) -> bytes:
        if path == source:
            raise OSError("denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    assert mod._readable_nonempty(source) is False
    assert mod._writable(tmp_path / "missing" / "artifact.json") is False

    def unwritable(*_args: object, **_kwargs: object) -> None:
        raise OSError("denied")

    monkeypatch.setattr(mod.tempfile, "NamedTemporaryFile", unwritable)
    assert mod._writable(tmp_path / "artifact.json") is False
    assert mod._capture_is_current("not-a-time", "20260906") is False


def test_preflight_marker_read_error_and_live_probe_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7077 records a marker read failure and uses the declared network probe."""

    root = _fake_root(tmp_path)
    original_read_text = Path.read_text

    def marker_denied(path: Path, *args: object, **kwargs: object) -> str:
        if path == root / mod.REFERENCE_PATH:
            raise OSError("denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", marker_denied)
    monkeypatch.setattr(mod, "_network_available", lambda: True)
    checks = mod.check_preconditions(
        root,
        tmp_path / "artifact.json",
        "20260906",
        source_capture_utc=mod.SOURCE_CAPTURE_UTC,
    )
    assert next(row for row in checks if row["check"] == "v620_reference_marker")["passed"] is False


def test_coverage_rules_fail_each_evidence_boundary(tmp_path: Path) -> None:
    """REQ-REPORT-7077 recomputes source, identity, decision, and code coverage."""

    artifact = _artifact(tmp_path)
    mutations: list[tuple[str, object]] = []

    undated = deepcopy(artifact["huggingface_rows"])
    undated[0]["capture_utc"] = "2026-09-05T23:59:59Z"
    mutations.append(("huggingface_rows", undated))

    missing_family = [
        row
        for row in artifact["query_rows"]
        if row.get("query_family") != mod.REQUIRED_QUERY_FAMILIES[0]
    ]
    mutations.append(("query_rows", missing_family))

    bad_receipt = deepcopy(artifact["primary_source_rows"])
    bad_receipt[0]["metadata_receipt_sha256"] = "0" * 64
    mutations.append(("primary_source_rows", bad_receipt))

    missing_required = [
        row for row in artifact["primary_source_rows"] if row["source_id"] != "arxiv:2605.02915"
    ]
    mutations.append(("primary_source_rows", missing_required))
    mutations.append(("decision_rows", []))

    bad_duplicates = deepcopy(artifact["duplicate_rows"])
    bad_duplicates[0]["disposition"] = "selected"
    mutations.append(("duplicate_rows", bad_duplicates))
    mutations.append(("code_identity_rows", artifact["code_identity_rows"][:-2]))

    for field, value in mutations:
        forged = deepcopy(artifact)
        forged[field] = value
        forged["rows"] = mod.combined_rows(forged)
        assert mod.completion_score(forged) == 0, field


def test_all_unsupported_vendor_and_code_promotions_are_rejected(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7077-VENDOR exercises every prohibited claim family."""

    artifact = _artifact(tmp_path)

    github = deepcopy(artifact)
    github["github_rows"][0]["local_dependency_promoted"] = True
    assert "unsupported implementation promotion: GitHub repository" in mod._unsupported_promotions(
        github
    )

    extropic_hardware = deepcopy(artifact)
    extropic_hardware["extropic_rows"][0]["hardware_execution_promoted"] = True
    assert "unsupported implementation promotion: Extropic hardware" in mod._unsupported_promotions(
        extropic_hardware
    )

    for collection, source in (("extropic_rows", "Extropic"), ("kona_rows", "Kona")):
        for claim in ("runtime", "power", "availability", "speed"):
            forged = deepcopy(artifact)
            forged[collection][0][f"{claim}_claimed"] = True
            assert f"vendor-only claim: {source} {claim}" in mod._unsupported_promotions(forged)

    malformed = deepcopy(artifact)
    malformed["extropic_rows"].insert(0, "bad row")
    malformed["kona_rows"].insert(0, "bad row")
    assert mod._unsupported_promotions(malformed) == []


def test_build_relative_output_and_disqualified_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7077 uses a terminal disqualification for contract failure after preflight."""

    root = _fake_root(tmp_path)
    monkeypatch.setattr(mod, "completion_score", lambda _artifact: 0)
    artifact = mod.build_artifact(
        root,
        "20260906",
        output_path=Path("results/artifact.json"),
        network_available=True,
        source_capture_utc=mod.SOURCE_CAPTURE_UTC,
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["gate_check_summary"]["failed_check"] == "source_ingestion_contract"
    assert artifact["duration_s"] >= 0


def test_validator_rejects_malformed_files_shape_and_terminal_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7077-ARTIFACT covers malformed storage and terminal forgeries."""

    artifact = _artifact(tmp_path)
    assert mod.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.validate_artifact(malformed) == ["artifact_unreadable"]

    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert mod.validate_artifact(array) == ["artifact_not_object"]
    assert mod.validate_artifact(42) == ["artifact_not_object"]

    missing = deepcopy(artifact)
    missing.pop("rows")
    assert "artifact fields mismatch" in mod.validate_artifact(missing)[0]
    extra = deepcopy(artifact)
    extra["extra"] = True
    assert "artifact fields mismatch" in mod.validate_artifact(extra)[0]

    mutations = {
        "verdict_class": "unknown",
        "duration_s": -1,
        "random_seed": 0,
        "rows": [],
        "preconditions_checked": [],
    }
    for field, value in mutations.items():
        forged = deepcopy(artifact)
        forged[field] = value
        assert mod.validate_artifact(_refresh(forged)), field


def test_validator_rejects_forged_blocked_and_disqualified_receipts(tmp_path: Path) -> None:
    """REQ-REPORT-7077 holds each terminal verdict to its structural evidence."""

    root = _fake_root(tmp_path)
    blocked = mod.build_artifact(
        root,
        "20260906",
        output_path=tmp_path / "blocked.json",
        network_available=False,
        source_capture_utc=mod.SOURCE_CAPTURE_UTC,
        duration_s=0.1,
    )
    for field, value in (
        ("gate_check_summary", {}),
        ("sota_ingestion_complete_score", 1),
        ("verdict_class", "partial"),
        ("honest_verdict", "partial_v620"),
    ):
        forged = deepcopy(blocked)
        forged[field] = value
        assert mod.validate_artifact(_refresh(forged)), field

    disqualified = deepcopy(_artifact(tmp_path))
    disqualified["query_rows"] = disqualified["query_rows"][:-1]
    disqualified["rows"] = mod.combined_rows(disqualified)
    disqualified["sota_ingestion_complete_score"] = 0
    disqualified["verdict_class"] = "disqualified"
    disqualified["honest_verdict"] = "disqualified_v620_sota_ingestion"
    disqualified["gate_check_summary"] = {
        "failed_check": "source_ingestion_contract",
        "expected_value": 1,
        "observed_value": 0,
        "passed": False,
    }
    assert mod.validate_artifact(_refresh(disqualified))
    for field, value in (
        ("sota_ingestion_complete_score", 1),
        ("verdict_class", "positive"),
        ("honest_verdict", "complete_positive_v620"),
        ("gate_check_summary", {}),
    ):
        forged = deepcopy(disqualified)
        forged[field] = value
        assert mod.validate_artifact(_refresh(forged)), field


def test_main_validate_bad_date_and_internal_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7077 CLI exits distinctly for valid, invalid, and malformed requests."""

    artifact = _artifact(tmp_path)
    valid = tmp_path / "valid.json"
    mod.write_artifact(artifact, valid)
    assert mod.main(["--validate", str(valid)]) == 0

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert mod.main(["--validate", str(invalid)]) == 1
    assert mod.main(["--date", "2026-09-06"]) == 2
    direct_output = tmp_path / "direct.json"
    assert (
        mod.main(
            [
                "--date",
                "20260906",
                "--output",
                str(direct_output),
                "--network-available",
            ]
        )
        == 0
    )
    assert direct_output.is_file()

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced"])
    assert (
        mod.main(
            [
                "--date",
                "20260906",
                "--output",
                str(tmp_path / "never-written.json"),
                "--network-available",
            ]
        )
        == 1
    )
