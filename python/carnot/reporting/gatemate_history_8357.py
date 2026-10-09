"""REQ-VERIFY-8357: recover exact producer bytes without repairing old evidence.

Content-addressed custody and Git may recover a missing source. A current file
with different bytes remains a diagnostic and cannot replace the producer seal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import gatemate_ledger_execution_8330 as supervisor
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
SEALS = {
    "sha256:6f79a6d2374c41baec7bf5ecb792b8534bc59b91e8b0c8cbe5720305d8545512": (
        "sha256:11ad525107a42f9017513b49207fb20f18c412a9dc0482c743e6aac770b7d8f8",
        "sha256:1a954d66f6202d5487dfd4b784ceeb0b8aac07d88df63227bec03485fbfb8e65",
    ),
    "sha256:48eb3ad7bccb2d869d561c3e021b50348fed3339bddbce2d493442ce7108f73b": (
        "sha256:0aaf24b1780d6580d9e71efccbaf3e9f3142bec140e183597dc0d4f42bb72e56",
        "sha256:5cbaee62cce2150a755e3ac78684389aef51d49ef0dbf1d59149455723ab077b",
    ),
}


def recover(
    ref: Json, root: Path, raw: Path, index: dict[str, list[Path]], receipts: list[Json]
) -> Json:
    """Only exact bytes qualify; preserve the actual changed hash when recovery fails."""
    source, expected = Path(ref["path"]), ref["sha256"]
    observed = sha256_file(source) if source.is_file() else None
    choices = [source, *index.get(expected, [])]
    if source.is_relative_to(root) and (root / ".git").exists() and observed != expected:
        relative = str(source.relative_to(root))
        plan = [
            CommandSpec(
                "git_history", ("git", "log", "--all", "--format=%H", "--", relative), "custody", 30
            )
        ]
        found = supervisor.execute(plan, raw / "git-log")
        receipts.extend(found)
        for i, commit in enumerate(Path(found[0]["stdout_path"]).read_text().splitlines()):
            plan = [
                CommandSpec("git_source", ("git", "show", commit + ":" + relative), "custody", 15)
            ]
            found = supervisor.execute(plan, raw / ("git-source-" + str(i)))
            receipts.extend(found)
            choices.append(Path(found[0]["stdout_path"]))
    selected = next((p for p in choices if p.is_file() and sha256_file(p) == expected), None)
    result = dict(
        original_path=str(source),
        expected_sha256=expected,
        observed_sha256=observed,
        available=selected is not None,
        reference=None,
        observed_reference=None,
    )
    raw.mkdir(parents=True, exist_ok=True)
    if selected:
        output = raw / (expected[7:] + source.suffix)
        output.write_bytes(selected.read_bytes())
        output.chmod(0o400)
        result["reference"] = reference(output)
    if observed is not None and observed != expected:
        output = raw / (observed[7:] + ".observed")
        output.write_bytes(source.read_bytes())
        output.chmod(0o400)
        result["observed_reference"] = reference(output)
    return result


def requirements(value: Json) -> list[Json]:
    """The producer primary defines all required source, configuration and raw seals."""
    refs = [
        value["replay_input_reference"],
        *value["source_artifact_hashes"],
        *value["code_config_hashes"],
        *value["raw_shard_hashes"],
    ]
    return list({(r["path"], r["sha256"]): r for r in refs}.values())


def verify(rows: list[Json]) -> None:
    """A rehashed manifest cannot change the expected seal of a recovered operand."""
    for row in rows:
        if type(row["available"]) is not bool:
            raise ValueError("closure_availability_drift")
        if row["available"]:
            if row["reference"]["sha256"] != row["expected_sha256"]:
                raise ValueError("closure_seal_drift")
            checked(row["reference"])
        elif row["reference"] is not None:
            raise ValueError("missing_closure_drift")
        if row["observed_reference"]:
            if row["observed_reference"]["sha256"] != row["observed_sha256"]:
                raise ValueError("observed_closure_drift")
            checked(row["observed_reference"])


def authenticate(path: Path, pin: str, raw: Path, index: dict[str, list[Path]]) -> Json:
    """Retain terminal custody and every missing source, even for an old blocked outcome."""
    if sha256_file(path) != pin:
        raise ValueError("historical_primary_drift")
    value = json.loads(path.read_bytes())
    side = path.parent / "raw" / path.stem / "validators" / (pin[7:] + ".json")
    bound = read_bound_sidecar(path, side)
    terminal_path = Path(value["terminal_validation_sidecar_path"])
    terminal = json.loads(terminal_path.read_bytes())
    if pin in SEALS and (sha256_file(side), sha256_file(terminal_path)) != SEALS[pin]:
        raise ValueError("historical_sidecar_seal_drift")
    if (
        bound["report"]["passed"] is not True
        or terminal["publication"]["primary_sha256"] != pin
        or terminal["publication"]["primary_path"] != str(path.absolute())
        or terminal["private_candidate_validation"]["passed"] is not True
    ):
        raise ValueError("historical_terminal_drift")
    receipts: list[Json] = []
    refs = [reference(path), reference(side), reference(terminal_path), *requirements(value)]
    rows = [recover(r, path.parents[1], raw / str(i), index, receipts) for i, r in enumerate(refs)]
    verify(rows)
    return dict(
        primary=rows[0]["reference"],
        primary_sha256=pin,
        rows=rows,
        receipts=receipts,
        passed=all(r["available"] for r in rows),
        historical_honest_verdict=value["honest_verdict"],
        historical_verdict_class=value["verdict_class"],
    )


def replay_bundle(bundle: Json, raw: Path) -> Json:
    """Run the unchanged historical validator against copied producer-bound source.

    Location changes preserve expected hashes. The original primary and checksum
    remain untouched, and no historical comparison consults current authority.
    """
    import sys
    from tempfile import mkdtemp

    if bundle["passed"] is not True:
        raise ValueError("incomplete_history_closure")
    private = Path(mkdtemp(prefix="exp8357-immutable-"))
    private.chmod(0o700)
    mapping = {}
    for row in bundle["rows"]:
        if row["available"]:
            mapping[row["original_path"]] = row["reference"]["path"]
            source = Path(row["original_path"])
            if source.suffix == ".py":
                root = Path(__file__).resolve().parents[3]
                target = private / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(checked(row["reference"]).read_bytes())
                target.chmod(0o400)
    script = (
        "import json,sys,pathlib,importlib;import carnot.reporting,scripts,scripts.experiments;"
        "p=pathlib.Path(sys.argv[1]);"
        "carnot.reporting.__path__.insert(0,str(p/'python/carnot/reporting'));"
        "scripts.__path__.insert(0,str(p/'scripts'));"
        "scripts.experiments.__path__.insert(0,str(p/'scripts/experiments'));"
        "m=importlib.import_module(sys.argv[4]);"
        "mapping=json.loads(sys.argv[3]);original=m.checked;"
        "resolver=lambda r:original(dict(r,path=mapping.get(r['path'],r['path'])));"
        "[(setattr(v,'checked',resolver)) for k,v in list(sys.modules.items()) "
        "if k.startswith('carnot.') and getattr(v,'checked',None) is original];"
        "print(json.dumps(m.replay(pathlib.Path(sys.argv[2]))),flush=True)"
    )
    identity = json.loads(checked(bundle["primary"]).read_bytes())["experiment_id"]
    module = {8330: "gatemate_ledger_execution_8330", 8344: "gatemate_ledger_execution_8344"}[
        identity
    ]
    spec = CommandSpec(
        "immutable_history_replay",
        (
            sys.executable,
            "-u",
            "-c",
            script,
            str(private),
            bundle["primary"]["path"],
            json.dumps(mapping),
            "carnot.reporting." + module,
        ),
        "historical",
        60,
    )
    receipt = supervisor.execute([spec], raw / "replay")[0]
    bundle["receipts"].append(receipt)
    return dict(passed=receipt["passed"] and receipt["normal_exit"], receipt=receipt)
