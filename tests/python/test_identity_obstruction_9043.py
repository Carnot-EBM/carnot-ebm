"""REQ-REPORT-9043-V772: private metadata controls never substitute for measurements."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting import identity_obstruction_9043 as e
from carnot.reporting import identity_obstruction_runner_9043 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def receipt(tmp_path: Path, body: Any, **extra: Any) -> dict[str, Any]:
    """Keep private request and response bytes independently authenticated."""
    path = tmp_path / "response.body"
    path.write_bytes(body if isinstance(body, bytes) else json.dumps(body).encode())
    url = e.old.request_url("en.wikipedia.org", ["Alias_name"])
    return (
        dict(e.old.ref(path), url=url, request_sha256=canonical_hash(url), status=200, error=None)
        | extra
    )


@pytest.mark.parametrize(
    ("page", "category"),
    [
        ({"title": "Alias name", "ns": 0, "missing": ""}, "missing_page"),
        ({"title": "Alias name", "ns": 10, "pageid": 3}, "existing_page_without_qid"),
        ({"title": "Alias name", "invalid": "", "invalidreason": "private"}, "unresolved_other"),
        (
            {"title": "Alias name", "pageid": 3, "pageprops": {"wikibase_item": "bad"}},
            "unresolved_other",
        ),
    ],
)
def test_page_shapes(tmp_path: Path, page: dict[str, Any], category: str) -> None:
    """SCENARIO-REPORT-9043-CLASSIFICATION: successful HTTP is not an identity."""
    rec = receipt(
        tmp_path,
        {
            "query": {
                "normalized": [{"from": "Alias_name", "to": "Alias name"}],
                "pages": {"3": page},
            }
        },
    )
    row = e.observe(rec, "en.wikipedia.org/Alias name")
    assert row["category"] == category
    assert row["pageid_present"] == ("pageid" in page)
    assert row["wikibase_item_present"] == ("wikibase_item" in page.get("pageprops", {}))
    assert row["parser_result"]["wikidata_id"] is None


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"query":{"pages":[]}}'])
def test_malformed_and_missing_receipts(tmp_path: Path, body: bytes) -> None:
    """SCENARIO-REPORT-9043-CUSTODY: malformed bytes cannot become an absent property."""
    rec = receipt(tmp_path, body)
    assert (
        e.observe(rec, "en.wikipedia.org/Alias name")["category"] == "malformed_or_missing_receipt"
    )
    Path(rec["path"]).unlink()
    assert (
        e.observe(rec, "en.wikipedia.org/Alias name")["category"] == "malformed_or_missing_receipt"
    )


def test_alias_omission_is_only_a_private_control(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9043-CLASSIFICATION: an independently visible QID proves omission."""
    data = {
        "query": {
            "converted": [{"from": "Alias_name", "to": "Canonical"}],
            "pages": {
                "4": {
                    "title": "Canonical",
                    "ns": 0,
                    "pageid": 4,
                    "pageprops": {"wikibase_item": "Q4"},
                }
            },
        }
    }
    rec = receipt(tmp_path, data)
    row = e.observe(rec, "en.wikipedia.org/Alias name")
    assert row["category"] == "alias_or_parser_omission"
    assert row["wikibase_item"] == "Q4" and row["parser_result"]["wikidata_id"] is None
    data["query"]["redirects"] = data["query"].pop("converted")
    rec = receipt(tmp_path, data)
    assert e.observe(rec, "en.wikipedia.org/Alias name")["category"] == "unresolved_other"
    rec["status"], rec["error"] = 429, "HTTP429"
    assert e.observe(rec, "en.wikipedia.org/Alias name")["category"] == "unresolved_other"
    rec["request_sha256"] = "changed"
    assert (
        e.observe(rec, "en.wikipedia.org/Alias name")["category"] == "malformed_or_missing_receipt"
    )


def test_actual_retained_obstructions() -> None:
    """SCENARIO-REPORT-9043-CLASSIFICATION: retained bytes explain all original units."""
    data = e.load(e.UPSTREAM)
    diagnosis = e.diagnose(data["metadata"])
    assert diagnosis["category_counts"] == dict.fromkeys(e.CATEGORIES, 0) | {
        "missing_page": 206,
        "existing_page_without_qid": 9,
    }
    assert len(diagnosis["obstruction_rows"]) == 215
    assert len(diagnosis["release_group_rows"]) == 2
    assert sum(row["historical"] for row in diagnosis["obstruction_rows"]) == 213
    assert len(diagnosis["request_rows"]) == 98
    assert sum(row["status"] == 429 for row in diagnosis["request_rows"]) == 1
    assert diagnosis["identity_diagnosis_ready_score"] == 1
    assert diagnosis["source_identity_ready_score"] == 0
    assert not diagnosis["parser_regression_evidence"]["proven_omissions"]
    assert {row["namespace"] for row in diagnosis["obstruction_rows"]} == {0, 10, 118}


def test_changed_authority_and_missing_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9043-CUSTODY: missing and changed inputs name the actual operand."""
    for path, observed in [(tmp_path / "missing", "missing_file"), (tmp_path / "changed", "bad")]:
        if observed == "bad":
            path.write_text("{}")
        with pytest.raises(e.Blocked) as exc:
            e.load(path)
        assert exc.value.operand["path"] == str(path)
        assert exc.value.operand["observed"] == (
            observed if observed == "missing_file" else e.old.ref(path)["sha256"]
        )


def test_measure_build_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9043-CUSTODY: rehashed aggregates cannot replace original bytes."""
    scope = r.scope_manifest([])
    reference = r.measure(e.UPSTREAM, tmp_path / (e.NAME + ".json"), "20261112", scope)
    value = r.build(reference, {})
    assert value["verdict_class"] == "disqualified"
    assert value["source_identity_ready_score"] == value["identity_diagnosis_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    primitive = json.loads(Path(reference["path"]).read_bytes())
    primitive["reduction"]["category_counts"]["missing_page"] -= 1
    atomic_json(Path(reference["path"]), primitive)
    atomic_json(candidate, r.build(e.old.ref(Path(reference["path"])), {}))
    with pytest.raises(ValueError, match="diagnosis_drift"):
        r.replay(candidate)
    primitive["preconditions_checked"]["authenticated_inputs"] = False
    atomic_json(Path(reference["path"]), primitive)
    atomic_json(candidate, r.build(e.old.ref(Path(reference["path"])), {}))
    with pytest.raises(ValueError, match="diagnosis_drift"):
        r.replay(candidate)


def test_blocked_gates_and_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9043-CUSTODY: absence and resource operands stay terminal blocked."""
    monkeypatch.setattr(r, "receipts_pass", lambda *a: True)
    for source in [tmp_path / "missing", e.UPSTREAM]:
        if source == e.UPSTREAM:
            monkeypatch.setattr(r.sys, "prefix", str(tmp_path))
        ref = r.measure(source, tmp_path / (e.NAME + ".json"), "20261112", r.scope_manifest([]))
        value = r.build(ref, {})
        assert value["verdict_class"] == "blocked"
        assert value["identity_diagnosis_ready_score"] == 0
        assert next(g for g in value["gate_check_summary"] if not g["passed"])["observed"] in [
            "missing_file",
            str(tmp_path),
        ]


def test_receipts_are_bound_to_commands_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9043-VALIDATION: claimed normal exits require frozen receipts."""
    log = tmp_path / "log"
    log.write_text("private control")
    scope = r.scope_manifest([dict(name=name, argv=[name]) for name in r.CHECKS])
    checks = [
        dict(
            name=c["name"],
            argv=c["argv"],
            passed=True,
            normal_exit=True,
            exit_code=0,
            timed_out=False,
            started_monotonic_ns=scope["frozen_ns"],
            ended_monotonic_ns=scope["frozen_ns"] + 1,
            scope_sha256=canonical_hash(scope),
            stdout_path=str(log),
            stderr_path=str(log),
            stdout_sha256=e.old.ref(log)["sha256"],
            stderr_sha256=e.old.ref(log)["sha256"],
        )
        for c in scope["commands"]
    ]
    monkeypatch.setattr(r, "verify_measured", lambda *a: True)
    valid = dict(checks=checks, owned_statement_coverage={})
    assert r.receipts_pass(scope, valid)
    assert not r.receipts_pass(scope, {})
    for field, value in [
        ("normal_exit", False),
        ("argv", []),
        ("scope_sha256", "changed"),
        ("stdout_sha256", "changed"),
        ("timed_out", True),
        ("started_monotonic_ns", 0),
        ("exit_code", 1),
    ]:
        altered = deepcopy(valid)
        altered["checks"][0][field] = value
        assert not r.receipts_pass(scope, altered)


def cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """The real child must import and replay from outside the checkout without PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    if env.get("COVERAGE_RCFILE"):
        command[1:2] = ["-m", "coverage", "run", "--parallel-mode"]
    e.progress("before_private_cli", 0, 1)
    process = subprocess.Popen(
        command + list(args),
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while True:
        try:
            stdout, stderr = process.communicate(timeout=30)
            break
        except subprocess.TimeoutExpired:
            e.progress("private_cli_wait", 0, 1)
    e.progress("after_private_cli", 1, 0)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def test_private_cli_rejection_replay_and_scanners(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9043-VALIDATION: real child and unchanged validators execute."""
    assert cli(tmp_path, "--date", "wrong").returncode == 1
    output = tmp_path / (e.NAME + ".json")
    assert (
        cli(tmp_path, "--upstream", str(tmp_path / "missing"), "--output", str(output)).returncode
        == 1
    )
    assert not output.exists()
    raw = tmp_path / "raw" / e.NAME
    rejected = raw / "rejected_primary.json"
    assert json.loads(rejected.read_bytes())["verdict_class"] == "disqualified"
    assert cli(tmp_path, "--cold-replay", str(rejected)).returncode == 0
    result = json.loads(rejected.read_bytes())
    result["completed_count"] += 1
    result["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in result.items() if k != "reproducibility_checksum"}
    )
    atomic_json(rejected, result)
    assert cli(tmp_path, "--cold-replay", str(rejected)).returncode == 1


def test_publication_and_qualified_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9043-VALIDATION: successful owned checks qualify diagnosis alone."""
    output = tmp_path / (e.NAME + ".json")
    reference = r.measure(e.UPSTREAM, output, "20261112", r.scope_manifest([]))
    value = r.build(reference, {})
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.validate(candidate, tmp_path / "scanners")["passed"]
    monkeypatch.setattr(r, "receipts_pass", lambda *a: True)
    value = r.build(reference, {})
    assert value["verdict_class"] == "blocked" and value["identity_diagnosis_ready_score"] == 1
    monkeypatch.setattr(r, "publish_primary", lambda path, value, validator: {"passed": True})
    assert r.main(["--output", str(output)]) == 0


def test_reconstruction_refuses_changed_denominators_and_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9043-CUSTODY: independent reconstruction catches rehashed drift."""
    data = e.load(e.UPSTREAM)
    altered = deepcopy(data["metadata"])
    altered["unresolved_requests"].pop()
    with pytest.raises(ValueError, match="unresolved_denominator_drift"):
        e.diagnose(altered)
    altered = deepcopy(data["metadata"])
    altered["source_identity_ready_score"] = 1
    with pytest.raises(ValueError, match="strict_closure_drift"):
        e.diagnose(altered)
    ref = r.measure(e.UPSTREAM, tmp_path / (e.NAME + ".json"), "20261112", r.scope_manifest([]))
    primitive = json.loads(Path(ref["path"]).read_bytes())
    primitive["reduction"]["gate_check_summary"][-1]["passed"] = False
    atomic_json(Path(ref["path"]), primitive)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, r.build(e.old.ref(Path(ref["path"])), {}))
    with pytest.raises(ValueError, match="gate_drift"):
        r.replay(candidate)
    value = json.loads(candidate.read_bytes())
    value["code_config_hashes"][0]["sha256"] = "changed"
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="operand_hash"):
        r.replay(candidate)
    rec = receipt(tmp_path, {"error": {"code": "maxlag"}})
    assert e.observe(rec, "en.wikipedia.org/Alias name")["category"] == "unresolved_other"
