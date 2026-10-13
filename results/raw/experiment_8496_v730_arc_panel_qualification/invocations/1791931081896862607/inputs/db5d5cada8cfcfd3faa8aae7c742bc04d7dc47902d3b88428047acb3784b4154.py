"""REQ-VERIFY-8496: a successful historical replay qualifies the precise missing rejection.

Original test nodes and independent producer seals remain unchanged. Coverage is
measured in real processes rather than inferred from a fixture's expected result.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import arc_generalization_panel_8468 as old
from carnot.reporting import arc_panel_qualification_8496 as p
from carnot.reporting import arc_panel_validation_8482 as shipped
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
UNION = "results/raw/experiment_8482_v729_arc_panel_qualification/invocations/1791875345048975426/coverage/coverage_shard_union.json"

BASELINE = [
    *shipped.p.OWNED,
    "python/carnot/reporting/arc_generalization_panel_8468.py",
    "scripts/experiments/experiment_8468_v728_arc_generalization_panel.py",
]


def rejection(raw: Path) -> Json:
    """Change only panel data after real successful original replay under its sealed authority."""
    raw.mkdir(parents=True, exist_ok=True)
    if sha256_file(p.ROOT / p.PRIOR) != p.PRIOR_PIN:
        raise ValueError("historical_primary_pin")
    primary = json.loads((p.ROOT / p.PRIOR).read_bytes())
    ref = primary["work_reference"]
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("historical_work_pin")
    work = json.loads(Path(ref["path"]).read_bytes())
    refs = work["source_input_receipts"]
    if any(sha256_file(Path(r["snapshot_path"])) != r["sha256"] for r in refs):
        raise ValueError("historical_snapshot_pin")
    with frozen_inputs(refs, raw / "writes"):
        contract = old.authority.authority(old.ROOT, raw / "authority")
        assert contract["activated"]
        assert old.replay(primary), "successful original sealed replay required"
        registry = yaml.safe_load((old.ROOT / "ops/arc_solve_registry.yaml").read_bytes())["games"]
        assert old.freeze(work["panel"]["installed_roster"], registry) == work["panel"]
        assert work["preconditions_checked"]["task_sha256"] == old.TASK_PIN
        private = deepcopy(work)
        # Historical log files stay pinned too; the private reconstruction changes no observations.
        value = old.build(private, primary["validation_receipts"], raw, raw / (old.NAME + ".json"))
        assert old.replay(value), "successful private replay required before mutation"
        atomic_json(raw / "successful_candidate.json", value)
        changed = deepcopy(private)
        game = changed["panel"]["selected_games"][0]
        changed["panel"]["registry_precheck"][game]["levels_reproduced"] += 1
        assert [k for k in private if changed[k] != private[k]] == ["panel"]
        candidate = old.build(
            changed, primary["validation_receipts"], raw / "mutated", raw / (old.NAME + ".json")
        )
        with old.bindings():
            assert old._REPLAY(candidate), "outer replay must still accept recomputed seals"
        assert not old.replay(candidate), "original independent comparison must reject"
        atomic_json(raw / "mutated_candidate.json", candidate)
    receipt = dict(
        successful_replay=True,
        outer_replay_after_mutation=True,
        original_rejection=True,
        original_line=350,
        changed_fields=["panel"],
        historical_task_sha256=old.TASK_PIN,
        historical_primary_sha256=p.PRIOR_PIN,
        source_input_receipts=refs,
        registry_operand="independently_sealed_original_registry",
        fixture_is_observation=False,
    )
    atomic_json(raw / "rejection_receipt.json", receipt)
    atomic_json(
        raw.parent / (raw.name + "_rejection_receipt.json"),
        dict(
            receipt,
            successful_candidate=value,
            mutated_candidate=candidate,
            successful_work=private,
            mutated_work=changed,
        ),
    )
    p.progress("successful_historical_replay_then_original_line350_rejection", 1, 0)
    return receipt


def manifest(private: Path) -> list[Json]:
    """Reuse every original bounded shard, adding only the new tests and exact rejection child."""
    suites = [p.TEST, *shipped.SUITES]
    with patch.multiple(shipped, p=p, SUITES=suites):
        specs = list(shipped.manifest(private))
    baseline = BASELINE
    config = private / "coverage.ini"
    text = config.read_text().replace(
        "[report]", "".join("    " + str(p.ROOT / f) + "\n" for f in baseline) + "[report]"
    )
    config.write_text(text)
    for name in ["ruff_check", "ruff_format", "strict_mypy"]:
        spec = next(s for s in specs if s["name"] == name)
        spec["argv"] = [a for a in spec["argv"] if a not in baseline]
    assert (
        sha256_file(p.ROOT / UNION)
        == "sha256:47aa00f3e691dc1947925b91af0ce5d1576cdf29e7dbb02ab033a7c3630562c0"
    )
    original = json.loads((p.ROOT / UNION).read_bytes())
    current = json.loads((private / "coverage_shard_union.json").read_bytes())
    assert all(current[k]["original"] == v["original"] for k, v in original.items())
    atomic_json(
        private / "coverage_baseline_union_receipt.json",
        dict(
            passed=True,
            original_reference=dict(path=str(p.ROOT / UNION), sha256=sha256_file(p.ROOT / UNION)),
            exact_original_union=True,
            baseline_statements_required=392,
        ),
    )
    first = next(i for i, s in enumerate(specs) if s["name"] == "coverage_combine")
    specs.insert(
        first,
        dict(
            name="original_rejection_350",
            argv=[
                str(p.ROOT / ".venv/bin/coverage"),
                "run",
                "--rcfile=" + str(config),
                str(p.ROOT / p.CLI),
                "--qualify-rejection",
                str(private / "coverage_rejection"),
            ],
            expected=0,
            deadline=60,
            scope="owned",
        ),
    )
    return specs
