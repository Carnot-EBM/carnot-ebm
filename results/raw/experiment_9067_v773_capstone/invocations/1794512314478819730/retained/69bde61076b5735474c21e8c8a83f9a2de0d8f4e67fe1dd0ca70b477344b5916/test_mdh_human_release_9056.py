"""REQ-REPORT-9056-AUTHORITY: private examples test custody, never supply measurements."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import pytest

from carnot.reporting import mdh_human_release_9056 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def workbook(path: Path, shared: bool = False, extra: str = "") -> Path:
    """Keep poisoned targets private so tests detect accidental planning leakage."""
    ns = e.NS
    values = [
        ["Category", "Prompt ID", "Prompts", "Model Responses", "Correctness", extra],
        ["History", "1", "Who wrote the book?", "An author", "SECRET_TARGET", "SECRET_MODEL"],
        ["", "2", "Who wrote the book?", "", "SECRET_TARGET", "SECRET_MODEL"],
        ["", "", "", "", "", ""],
        ["Average", "0.9", "SECRET_SCORE", "SECRET_SUMMARY", "SECRET_TARGET", "SECRET_MODEL"],
    ]
    root = ET.Element(f"{{{ns}}}worksheet")
    data = ET.SubElement(root, f"{{{ns}}}sheetData")
    strings = ET.Element(f"{{{ns}}}sst")
    for i, row in enumerate(values, 1):
        element = ET.SubElement(data, f"{{{ns}}}row", r=str(i))
        for j, text in enumerate(row):
            cell = ET.SubElement(element, f"{{{ns}}}c", r=chr(65 + j) + str(i))
            if shared:
                cell.set("t", "s")
                ET.SubElement(cell, f"{{{ns}}}v").text = str((i - 1) * 6 + j)
                si = ET.SubElement(strings, f"{{{ns}}}si")
                ET.SubElement(si, f"{{{ns}}}t").text = text
            else:
                cell.set("t", "inlineStr")
                inline = ET.SubElement(cell, f"{{{ns}}}is")
                ET.SubElement(inline, f"{{{ns}}}t").text = text
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(root))
        archive.writestr("xl/worksheets/sheet2.xml", '<worksheet xmlns="' + ns + '"/>')
        if shared:
            archive.writestr("xl/sharedStrings.xml", ET.tostring(strings))
    return path


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("extra", ["Model", "Annotator 1", "R-Score"])
def test_allowlist_and_original_authority(tmp_path: Path, shared: bool, extra: str) -> None:
    """SCENARIO-REPORT-9056-SCHEMA: targets and duplicated names never become authority."""
    result = e.workbook(workbook(tmp_path / "private.xlsx", shared, extra))
    assert len(result["records"]) == 2
    assert len(result["sheets"]) == 2
    assert result["records"][1]["response"] == ""
    assert result["records"][1]["category"] == "History"
    assert "SECRET" not in json.dumps(result)
    assert all(r["original_judgment_authority"] == "unknown" for r in result["records"])
    assert result["members"] and result["sheets"][0]["protected_columns"]


def test_grouping_is_transitive_and_outcome_blind() -> None:
    """SCENARIO-REPORT-9056-CUSTODY: text and metadata merge before protected labels."""
    records = [
        dict(unit_id="a", prompt="One fact", prompt_id="P1", category="a"),
        dict(unit_id="b", prompt=" ONE  FACT ", prompt_id="P2", category="a"),
        dict(unit_id="c", prompt="Different fact", prompt_id="P2", category="a"),
        dict(unit_id="d", prompt="Unrelated text", prompt_id="P3", category="b"),
    ]
    groups = e.groups(records)
    assert len(groups) == 2
    assert sorted(len(g["unit_ids"]) for g in groups) == [1, 3]
    assert all(g["role"] == "UNASSIGNED" for g in groups)
    assert groups == e.groups(list(reversed(records)))


def test_workbook_safety_and_numeric_cells(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9056-SCHEMA: unsafe ZIP members and formulas cannot execute."""
    path = workbook(tmp_path / "unsafe.xlsx")
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("xl/vbaProject.bin", b"never execute")
    with pytest.raises(ValueError, match="unsafe_workbook"):
        e.workbook(path)
    cell = ET.fromstring('<c xmlns="' + e.NS + '"><v>42</v></c>')
    assert e.cell_text(cell, []) == "42"
    assert e.cell_text(ET.fromstring('<c xmlns="' + e.NS + '"/>'), []) == ""


def test_missing_and_changed_release(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9056-CUSTODY: missing bytes differ from an observed zero."""
    value = e.inspect(tmp_path)
    assert value["gate_check_summary"][0]["observed"] == "missing_file"
    assert value["human_source_ready_score"] == 0
    path = tmp_path / "metadata.json"
    path.write_text("{}")
    value = e.inspect(tmp_path)
    assert value["gate_check_summary"][0]["observed"] != "missing_file"
    assert not value["gate_check_summary"][0]["passed"]


def test_gate_operands() -> None:
    """SCENARIO-REPORT-9056-CUSTODY: missing fields are never coerced to zero."""
    gate = e.gate("test", 1, 0, dict(path="private", sha256=None))
    assert gate["observed"] == 0 and gate["passed"] is False
    assert gate["artifact_field"] == "test"


def cli(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Use the actual script from a foreign directory without ambient imports."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *arguments],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_private_cli_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9056-QUALIFICATION: actual children reject missing and rehashed inputs."""
    output = tmp_path / "results" / (e.NAME + ".json")
    result = cli(
        ["--date", "20261112", "--inputs", str(tmp_path / "missing"), "--output", str(output)],
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["human_source_ready_score"] == 0
    assert cli(["--cold-replay", str(output)], tmp_path).returncode == 0
    value["completed_count"] += 1
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert cli(["--cold-replay", str(output)], tmp_path).returncode == 1
    assert cli(["--date", "bad"], tmp_path).returncode == 1


def test_build_controls_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9056-QUALIFICATION: receipts cannot bless changed original operands."""
    from carnot.reporting import mdh_human_release_runner_9056 as runner

    monkeypatch.setattr(runner, "receipts_pass", lambda scope, validation: True)
    reference = runner.measure(
        tmp_path / "missing", tmp_path / "output", "20261112", runner.scope_manifest([])
    )
    value = runner.build(reference, {})
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_release_bytes")
    output = tmp_path / "value.json"
    atomic_json(output, value)
    assert runner.replay(output)["passed"] is True
    value["human_source_ready_score"] = 1
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        runner.replay(output)
    primitive = json.loads(Path(reference["path"]).read_bytes())
    primitive["reduction"]["completed_count"] = 77
    atomic_json(Path(reference["path"]), primitive)
    value["primitive_reference"] = e.ref(Path(reference["path"]))
    value["raw_shard_hashes"][0] = value["primitive_reference"]
    atomic_json(output, value)
    with pytest.raises(ValueError, match="source_reduction_drift"):
        runner.replay(output)


def test_receipt_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9056-QUALIFICATION: failed, stale or empty checks disqualify."""
    from carnot.reporting import mdh_human_release_runner_9056 as runner

    assert not runner.receipts_pass({}, {})
    assert not runner.receipts_pass(runner.scope_manifest([]), {})
    commands = [dict(name=n, argv=["private"]) for n in runner.CHECKS]
    scope = runner.scope_manifest(commands)
    log = tmp_path / "log"
    log.write_text("actual private receipt")
    checks = [
        dict(
            name=n,
            argv=["private"],
            passed=True,
            normal_exit=True,
            timed_out=False,
            exit_code=0,
            started_monotonic_ns=scope["frozen_ns"],
            ended_monotonic_ns=scope["frozen_ns"] + 1,
            scope_sha256=canonical_hash(scope),
            stdout_path=str(log),
            stdout_sha256=e.ref(log)["sha256"],
            stderr_path=str(log),
            stderr_sha256=e.ref(log)["sha256"],
        )
        for n in runner.CHECKS
    ]
    validation = dict(checks=checks, owned_statement_coverage={})
    monkeypatch.setattr(runner, "verify_measured", lambda report, owned: True)
    assert runner.receipts_pass(scope, validation)
    checks[0]["passed"] = False
    assert not runner.receipts_pass(scope, validation)
    checks[0]["passed"] = True
    log.write_text("changed bytes")
    assert not runner.receipts_pass(scope, validation)


def test_validation_uses_real_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9056-QUALIFICATION: original scanners and foreign replay stay required."""
    from carnot.reporting import mdh_human_release_runner_9056 as runner

    commands = []

    def child(name: str, argv: list[str], logs: Path, **kwargs: object) -> dict:
        commands.append(argv)
        return dict(passed=True, normal_exit=True)

    monkeypatch.setattr(runner, "child", child)
    assert runner.validate(tmp_path / "candidate", tmp_path / "logs")["passed"]
    assert any("-u" in cmd and "PYTHONPATH" in cmd for cmd in commands)
    assert any("--strict" in cmd for cmd in commands)


def private_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Private hashes exercise structure without granting author authority or measured credit."""
    path = workbook(tmp_path / "private.xlsx")
    atomic_json(tmp_path / "metadata.json", dict(sha=e.REVISION, cardData=dict(license="mit")))
    (tmp_path / "README.md").write_text(
        "Private MIT declaration; not original annotation instructions"
    )
    atomic_json(
        tmp_path / "inventory.json",
        dict(exposed_keys=[], rows=[], cross_language_exposure_resolution_complete=False),
    )
    monkeypatch.setattr(
        e,
        "PINS",
        {
            p.name: e.ref(p)["sha256"].split(":")[1]
            for p in [path, tmp_path / "metadata.json", tmp_path / "README.md"]
        },
    )
    monkeypatch.setattr(e, "HISTORY", {})
    monkeypatch.setattr(e, "ROOT", tmp_path)
    monkeypatch.setattr(e, "INVENTORY", "inventory.json")
    monkeypatch.setattr(
        e, "INVENTORY_PIN", e.ref(tmp_path / "inventory.json")["sha256"].split(":")[1]
    )
    return tmp_path


def test_private_complete_audit_and_changed_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9056-CUSTODY: intact structure stays blocked; self-rehashed source swaps fail."""
    source = private_release(tmp_path, monkeypatch)
    value = e.inspect(source)
    assert value["completed_count"] == 2
    assert value["human_source_ready_score"] == 0
    assert value["candidate_prompt_group_count"] == 1
    assert value["exposure_join_rows"][0]["status"] == "unknown_source_family_overlap"
    assert "SECRET" not in json.dumps(value)
    inventory = source / "inventory.json"
    atomic_json(inventory, dict(exposed_keys=[value["group_rows"][0]["prompt_keys"][0]], rows=[]))
    monkeypatch.setattr(e, "INVENTORY_PIN", e.ref(inventory)["sha256"].split(":")[1])
    assert e.inspect(source)["exposure_join_rows"][0]["status"] == "exposed"
    inventory.write_text("changed")
    assert any(not g["passed"] for g in e.inspect(source)["gate_check_summary"])
    with zipfile.ZipFile(source / "private.xlsx", "a") as archive:
        archive.writestr("swapped_authority", b"new human claims")
    assert e.inspect(source)["gate_check_summary"][0]["passed"] is False


def test_parsing_failure_and_empty_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9056-SCHEMA: malformed workbook structure stays terminal blocked."""
    source = private_release(tmp_path, monkeypatch)
    (source / "private.xlsx").write_bytes(b"not zip")
    pins = dict(e.PINS)
    pins["private.xlsx"] = e.ref(source / "private.xlsx")["sha256"].split(":")[1]
    monkeypatch.setattr(e, "PINS", pins)
    assert e.inspect(source)["gate_check_summary"][-1]["check"] == "release_structure"
    assert len(e.groups([dict(unit_id="empty", prompt="", prompt_id="", category="")])) == 1


def test_resource_block_and_unhashed_operand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9056-QUALIFICATION: resource failure and changed code cannot qualify."""
    from carnot.reporting import mdh_human_release_runner_9056 as runner

    monkeypatch.setattr(runner.os, "sysconf", lambda name: 1)
    reference = runner.measure(
        tmp_path / "missing", tmp_path / "output", "20261112", runner.scope_manifest([])
    )
    value = runner.build(reference, {})
    assert value["preconditions_checked"]["available_memory_bytes"] == 1
    assert not value["gate_check_summary"][0]["passed"]
    output = tmp_path / "artifact.json"
    atomic_json(output, value)
    assert runner.replay(output)["passed"] is True
    value["code_config_hashes"][0]["sha256"] = "sha256:changed"
    atomic_json(output, value)
    with pytest.raises(ValueError, match="operand_hash"):
        runner.replay(output)


def test_pinned_author_inventory() -> None:
    """SCENARIO-REPORT-9056-SCHEMA: original bytes enumerate all 4800 slots without summary leakage."""
    value = e.inspect(e.DEFAULT_INPUT)
    assert value["completed_count"] == 4800
    assert value["human_source_ready_score"] == 0
    assert len(value["historical_verdicts"]) == 3
    assert len(value["workbook_schema_rows"]) == 16
    assert sum(not r.get("response") for r in value["records"]) == 2
    assert all(s["response_slot_count"] in {0, 400} for s in value["workbook_schema_rows"])
    assert value["candidate_prompt_group_count"] < 400
    assert value["protected_outcome_values_projected"] is False
    assert all(
        set(reference) == {"path", "sha256"} for reference in value["source_artifact_hashes"]
    )


def test_empty_prompt_and_parent_compression(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9056-SCHEMA: incomplete text never enters an eligible group."""
    path = workbook(tmp_path / "empty.xlsx")
    with zipfile.ZipFile(path) as archive:
        data = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(data["xl/worksheets/sheet1.xml"])
    root.find(e.TAG + "sheetData")[1][2].find(e.TAG + "is")[0].text = ""
    data["xl/worksheets/sheet1.xml"] = ET.tostring(root)
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in data.items():
            archive.writestr(name, raw)
    assert len(e.workbook(path)["records"]) == 1
    rows = [
        dict(unit_id=str(i), prompt="a b c", prompt_id="one", category="same") for i in range(5)
    ]
    assert len(e.groups(rows)) == 1


def test_rehashed_raw_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9056-CUSTODY: independent row recomputation rejects a repaired envelope."""
    from carnot.reporting import mdh_human_release_runner_9056 as runner

    monkeypatch.setattr(runner, "receipts_pass", lambda scope, validation: True)
    reference = runner.measure(
        tmp_path / "missing", tmp_path / "output", "20261112", runner.scope_manifest([])
    )
    primitive = json.loads(Path(reference["path"]).read_bytes())
    rows = Path(primitive["raw_rows"]["path"])
    atomic_json(rows, dict(records=[dict(unit_id="invented")]))
    primitive["raw_rows"] = e.ref(rows)
    atomic_json(Path(reference["path"]), primitive)
    value = runner.build(e.ref(Path(reference["path"])), {})
    output = tmp_path / "artifact.json"
    atomic_json(output, value)
    with pytest.raises(ValueError, match="primitive_rows_drift"):
        runner.replay(output)
