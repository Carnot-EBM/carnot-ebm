"""REQ-REPORT-8663: reproduce the sealed failure before a small live SDK witness."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting import arc_reset_8663 as e
from carnot.reporting.arc_observer_8624 import checked, checked_bytes
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
OLD = e.ROOT / "results/raw/experiment_8650_v742_arc_reset_transfer/invocations/1792359279150240328"
RECORDER_PIN = "sha256:f56fa524bf124a8c93e7ad76e253a902b556b351316c8681846335ceea4c2cf3"


def authenticate(root: Path) -> Json:
    """Original producer hashes authenticate observations independently of corrections."""
    current, fixture = [checked(root / p, h) for p, h in e.PINS.items()]
    ref = current["work_reference"]
    work = checked(Path(ref["path"]), ref["sha256"])
    episodes = [checked(Path(r["path"]), r["sha256"]) for r in work["references"]]
    for path, digest in work["source_artifact_hashes"].items():
        checked_bytes(Path(path), digest)
    checked_bytes(
        root / "tests/python/test_arc_reset_provenance_8649.py",
        "sha256:72c2217c6a68120a092c4ebad5e78b4d319e4ebf7c7143cad12369640e8e3ea8",
    )
    assert current["verdict_class"] == "disqualified"
    assert fixture["reset_provenance_ready_score"] == 1
    requests = sum(len(ep["reset_provenance"]["reset_request_rows"]) for ep in episodes)
    assert requests == len(current["reset_request_rows"]) == 688
    assert len(current["missing_cause_rows"]) == 236
    assert current["censored_count"] == 16
    assert len(current["action_parity_rows"]) == 8
    assert all(row["passed"] for row in current["action_parity_rows"])
    errors = [
        dict(episode_id=ep["unit"]["episode_id"], errors=ep["recorder_errors"])
        for ep in episodes
        if ep["unit"]["reset_provenance"] == "on"
    ]
    assert len(errors) == 8 and all("TypeError" in row["errors"] for row in errors)
    return dict(
        missing_causes=236,
        request_rows=requests,
        censored_redirects=16,
        action_equal_pairs=8,
        complete_horizons=0,
        provenance_verdict="disqualified",
        fixture_ready_score=1,
        quarantines_unchanged=True,
        references=work["references"],
        error_rows=errors,
        historical_model_provenance=work["historical_model_provenance"],
    )


def reproduce(history: Json, raw: Path) -> Json:
    """Capture the exact old TypeError through four real recorder/wrapper entry paths."""
    from arcengine import FrameDataRaw, GameState
    import carnot.agentic as package
    from carnot.agentic import arc_competition_agent as live
    from carnot.agentic.arc_generalization_runtime import withheld_policy_inputs
    from carnot.agentic.arc_live_panel_runtime_8384 import no_models

    episode = checked(Path(history["references"][1]["path"]), history["references"][1]["sha256"])
    public = episode["reset_observations"][0]["public_observation"]
    frame = FrameDataRaw(
        game_id=episode["unit"]["game"],
        state=GameState.NOT_FINISHED,
        levels_completed=public["level"],
        full_reset=public["full_reset"],
    )
    frame.frame = [np.asarray(grid, dtype=np.int8) for grid in public["pixels"]]
    fixture = raw / "real_frame_fixture.json"
    atomic_json(
        fixture,
        dict(
            public=public,
            source=history["references"][1],
            runtime_class=type(frame).__module__ + "." + type(frame).__name__,
            shape=[list(item.shape) for item in frame.frame],
            value_types=[
                type(frame.frame).__name__,
                type(frame.frame[0]).__name__,
                type(frame.state).__name__,
            ],
            historical_dtype_retained=False,
        ),
    )
    path = OLD / "current_source_closure/python/carnot/agentic/arc_reset_provenance.py"
    checked_bytes(path, RECORDER_PIN)
    spec = importlib.util.spec_from_file_location("sealed_reset_8663", path)
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    hashing = old.public_hash
    rows: list[Json] = []
    label = "public_hash"

    def traced(value: Any) -> Any:
        try:
            return hashing(value)
        except TypeError as exc:
            site = traceback.extract_tb(exc.__traceback__)[-1]
            if not any(row["path"] == label for row in rows):
                rows.append(
                    dict(
                        path=label,
                        error="TypeError",
                        message=str(exc)[:240],
                        site=dict(path=site.filename, line=site.lineno, function=site.name),
                        value_types=[
                            type(value).__name__,
                            type(getattr(value, "frame", None)).__name__,
                            "numpy.ndarray",
                        ],
                    )
                )
            raise

    old.public_hash = traced
    recorder = old.ResetProvenance()
    for label, call in [
        ("public_hash", lambda: old.public_hash(frame)),
        ("begin", lambda: (recorder.begin(None), recorder.begin(frame))),
    ]:
        try:
            call()
        except TypeError:
            pass
    stale_context_reproduced = recorder.index == 2 and recorder.context["action_index"] == 1
    label = "sdk_call"
    policy = SimpleNamespace(_arc_reset_provenance=old.ResetProvenance())
    old.sdk_call(policy, ("RESET", None), lambda: frame)

    class Base:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

        def do_action_request(self, action: Any) -> Any:
            return frame

    label = "make_carnot_agent"
    with (
        withheld_policy_inputs(),
        no_models(),
        patch.object(live, "_load_submitted_frame_change_scorer", return_value=None),
        patch.object(package, "arc_reset_provenance", old, create=True),
        patch.dict(sys.modules, {"carnot.agentic.arc_reset_provenance": old}),
        patch.dict(
            os.environ,
            {"CARNOT_ARC_DISABLE_INDUCTION": "1", "CARNOT_ARC_TRAJECTORY_SUPERVISOR": "0"},
        ),
    ):
        agent = live.make_carnot_agent(Base)(game_id="private-fixture")
        agent._policy._arc_reset_provenance = old.ResetProvenance()
        agent._policy.explorer._arc_reset_provenance = agent._policy._arc_reset_provenance
        action = agent.choose_action([], frame)
        agent.do_action_request(action)
    result = dict(
        passed=len(rows) == 4 and stale_context_reproduced,
        exception_reproduction_rows=rows,
        stale_context_reproduced=stale_context_reproduced,
        real_frame_fixture_reference=dict(path=str(fixture), sha256=sha256_file(fixture)),
        original_recorder=dict(path=str(path), sha256=RECORDER_PIN),
    )
    atomic_json(raw / "reproduction.json", result)
    return result


def episode(path: Path) -> int:
    """Use a fresh official SDK for one small arm, with all policy shortcuts withheld."""
    from carnot.reporting.arc_reset_live_8649 import collect
    from carnot.reporting.arc_supervisor_live_panel_8384 import sdk

    unit = json.loads(path.read_bytes())
    raw = Path(unit["directory"])
    e.progress("official_SDK_before")
    arcade = sdk()
    implementation = Path(inspect.getfile(type(arcade)))
    result = collect(unit, arcade, raw)
    result["sdk_identity"] = dict(
        official=type(arcade).__module__.startswith("arc_agi."),
        path=str(implementation),
        sha256=sha256_file(implementation),
    )
    observations = []
    for line in (raw / "steps.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["action"]["name"] == "RESET":
            public = dict(
                result["observations"][row["index"] - 1], pixels=row["observation"]["pixels"]
            )
            observations.append(
                dict(action_index=row["index"], observation_sha256=canonical_hash(public))
            )
    result["reset_observations"] = observations
    result["legacy"] = canonical_hash(result["legacy"])
    atomic_json(raw / "witness.json", result)
    e.progress("official_SDK_after", len(result["actions"]), 16 - len(result["actions"]))
    return int(bool(result["termination"]["error"] or result["model_tripwire_attempts"]))
