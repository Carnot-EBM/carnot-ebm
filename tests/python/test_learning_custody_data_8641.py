"""REQ-REPORT-8641: custody requires actual child bytes and source contexts."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import learning_custody_data_8641 as c


@pytest.fixture
def sealed(tmp_path):
    """Use real interpreter execution so fake line lists cannot satisfy custody."""
    source = tmp_path / "operand.py"
    source.write_text("value = 1\nprint(value, flush=True)\n")
    root = tmp_path / "child"
    root.mkdir()
    config = root / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ninclude=" + str(source) + "\n[report]\nexclude_lines=\n"
    )
    receipt = c.invoke(
        "fixture",
        ["-m", "coverage", "run", "--rcfile=" + str(config), "--context=fixture", str(source)],
        root,
        root / "coverage",
        deadline=30,
    )
    manifest = c.seal(root, receipt, ["fixture::test"], [c.ref(source)], "fixture")
    return root, source, manifest


def test_actual_seal_and_repeated_combine(sealed, tmp_path):
    """SCENARIO-REPORT-8641-CUSTODY: copies preserve original and aggregate bytes."""
    root, source, manifest = sealed
    assert manifest["pid"] > 0 and manifest["exit_code"] == 0
    assert manifest["data"] and c.verify(manifest)
    report = c.aggregate([manifest], tmp_path / "aggregate")
    assert report["files"][str(source)]["missing_lines"] == []
    before = c.ref(Path(report["aggregate_reference"]["path"]))
    assert c.aggregate([manifest], tmp_path / "aggregate") == report
    assert c.ref(Path(before["path"])) == before and c.verify(manifest)
    assert (root / "sealed" / "receipt.json").is_file()
    unused = tmp_path / "unused.py"
    unused.write_text("not_executed = 1\n")
    changed = deepcopy(manifest)
    changed["sources"].append(c.ref(unused))
    from carnot.reporting.current_work_receipt import atomic_json

    commitment = Path(changed["commitment"]["path"])
    atomic_json(commitment, {k: v for k, v in changed.items() if k != "commitment"})
    changed["commitment"] = c.ref(commitment)
    unused.write_text("not_executed = 2\n")
    assert c.verify(changed)


@pytest.mark.parametrize(
    "attack",
    [
        "deleted",
        "overwritten",
        "empty",
        "duplicate",
        "foreign",
        "substitution",
        "exit",
        "contexts",
        "source",
        "zero",
    ],
)
def test_rejections_even_after_rehashing(sealed, tmp_path, attack):
    """SCENARIO-REPORT-8641-CUSTODY: repaired hashes do not replace child authority."""
    _, source, original = sealed
    changed = deepcopy(original)
    data = Path(changed["data"][0]["reference"]["path"])
    if attack == "deleted":
        data.unlink()
    elif attack in {"overwritten", "empty"}:
        data.chmod(0o644)
        data.write_bytes(b"changed" if attack == "overwritten" else b"")
    elif attack == "duplicate":
        changed["data"].append(deepcopy(changed["data"][0]))
    elif attack in {"foreign", "substitution"}:
        substitute = tmp_path / "substitute"
        substitute.write_bytes(data.read_bytes())
        changed["data"][0]["reference"] = c.ref(substitute)
        if attack == "foreign":
            changed["data"][0]["files"] = {"foreign.py": [1]}
    elif attack == "exit":
        changed["exit_code"] = -9
    elif attack == "contexts":
        changed["context"] = "forged"
    elif attack == "source":
        source.write_text("value = 2\n")
        changed["sources"] = [c.ref(source)]
    else:
        changed["data"] = []
    assert not c.verify(changed)
    with pytest.raises(ValueError, match="custody"):
        c.aggregate([changed], tmp_path / "bad")


def test_missing_empty_unsuffixed_and_killed(tmp_path, sealed):
    """REQ-VERIFY-8641: absence and killed actual children cannot qualify."""
    root, _, manifest = sealed
    with pytest.raises(ValueError, match="duplicate"):
        c.aggregate([manifest, manifest], tmp_path / "duplicate")
    path = tmp_path / "empty"
    path.mkdir()
    with pytest.raises(ValueError, match="zero"):
        c.seal(path, dict(passed=True, pid=1, exit_code=0), [], [], "empty")
    killed = c.invoke(
        "killed", ["-c", "import time; time.sleep(5)"], root / "kill", root / "none", deadline=0.1
    )
    assert killed["timed_out"] and killed["exit_code"] == -9
    assert not c.verify(dict(manifest, receipt=killed))
    assert c.inspect_data(tmp_path / "absent") is None
    (path / "coverage").write_bytes(b"")
    assert c.inspect_data(path / "coverage") is None
    # Unsuffixed Coverage.py output receives the same immediate sealing rules.
    source = tmp_path / "one.py"
    source.write_text("x = 1\n")
    receipt = c.invoke(
        "plain",
        [
            "-m",
            "coverage",
            "run",
            "--rcfile=/dev/null",
            "--include=" + str(source),
            "--context=plain",
            str(source),
        ],
        path,
        path / "coverage",
    )
    c.invoke(
        "empty_auxiliary",
        [
            "-m",
            "coverage",
            "run",
            "--rcfile=/dev/null",
            "--include=" + str(tmp_path / "never.py"),
            str(source),
        ],
        path,
        path / "empty_auxiliary",
    )
    assert c.seal(path, receipt, ["one"], [c.ref(source)], "plain")["data"]
    empty_only = tmp_path / "empty-only"
    empty_only.mkdir()
    zero = c.invoke(
        "empty_only",
        [
            "-m",
            "coverage",
            "run",
            "--rcfile=/dev/null",
            "--include=" + str(tmp_path / "never.py"),
            str(source),
        ],
        empty_only,
        empty_only / "coverage",
    )
    with pytest.raises(ValueError, match="invalid_child"):
        c.seal(empty_only, zero, [], [c.ref(source)], "empty")


def test_old_failure_is_executed(tmp_path):
    """SCENARIO-REPORT-8641-CUSTODY: destructive old combines lose actual shards."""
    rows = c.controls(tmp_path)
    assert all(row["rejected"] for row in rows)
    assert {r["mutation"] for r in rows} >= {
        "lost_shard_second_combine",
        "deleted",
        "overwritten",
        "empty",
        "duplicate",
        "foreign_source",
        "rehashed_substitution",
    }


def test_deep_rejection_paths(sealed, tmp_path, monkeypatch):
    """REQ-REPORT-8641: directly test inner checks apart from the outer byte commitment."""
    from coverage import CoverageData
    from carnot.reporting.current_work_receipt import atomic_json
    import shutil

    root, _, manifest = sealed
    original = deepcopy(manifest)
    commitment_path = Path(original["commitment"]["path"])
    for attack in (
        "directory",
        "killed",
        "no_sources",
        "no_data",
        "duplicate",
        "foreign_path",
        "contexts",
        "junit",
        "log",
        "absent",
    ):
        changed = deepcopy(original)
        if attack == "directory":
            changed["receipt"]["coverage_file"] = str(tmp_path / "wrong")
        elif attack == "killed":
            changed["receipt"]["passed"] = False
        elif attack == "no_sources":
            changed["sources"][0]["sha256"] = "wrong"
        elif attack == "no_data":
            changed["data"] = []
        elif attack == "duplicate":
            changed["data"] *= 2
        elif attack == "foreign_path":
            changed["data"][0]["reference"]["path"] = str(tmp_path / "foreign")
        elif attack == "contexts":
            changed["context"] = "forged"
        elif attack == "junit":
            junit = tmp_path / "changed.xml"
            junit.write_text("<testsuite/>")
            changed["junit"] = dict(path=str(junit), sha256="wrong")
        elif attack == "log":
            changed["receipt"]["stdout_sha256"] = "wrong"
        else:
            changed["data"][0]["reference"]["path"] = str(commitment_path.parent / "absent")
        atomic_json(commitment_path, {k: v for k, v in changed.items() if k != "commitment"})
        changed["commitment"] = c.ref(commitment_path)
        assert not c.verify(changed), attack
    database = tmp_path / "corrupt"
    database.write_bytes(b"SQLite format 3\x00broken")
    assert c.inspect_data(database) is None
    data = CoverageData(basename=str(tmp_path / "valid_empty"))
    data.add_lines({"/empty.py": []})
    data.write()
    assert c.inspect_data(tmp_path / "valid_empty") is not None
    fresh = tmp_path / "rejection"
    fresh.mkdir()
    shutil.copyfile(original["data"][0]["reference"]["path"], fresh / "coverage")
    with pytest.raises(ValueError, match="invalid_child"):
        c.seal(
            fresh,
            dict(original["receipt"], coverage_file=str(fresh / "coverage")),
            [],
            [],
            "fixture",
        )
    monkeypatch.setattr(
        c, "sha256_file", lambda path: "changed" if path.name == "data-0" else "original"
    )
    second = tmp_path / "copy-race"
    second.mkdir()
    shutil.copyfile(original["data"][0]["reference"]["path"], second / "coverage")
    with pytest.raises(ValueError, match="copy_changed"):
        c.seal(second, original["receipt"], [], [], "fixture")
