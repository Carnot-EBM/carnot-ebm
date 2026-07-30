"""REQ-HARNESS-6051: a VERIFIED-INERT drift acknowledgement, expensive enough not to abuse.

WHY THIS EXISTS (2026-07-30). Three registered artifacts declare a code dependency but NO
`rebuild_command`. When that dependency legitimately changes, the author had no sanctioned way to
clear `artifact_freshness_lint`: they could not rebuild, so the only remaining moves were to edit the
recorded sha256 silently or to pass `--no-verify`. The lint's own docstring names the second as the
failure mode to avoid ("blocking on 'I cannot check' would train people to pass --no-verify, which is
worse than the gap it closes") -- and the first is worse still, because it launders an unverified
change into a verified-looking provenance block.

The third move is an acknowledgement PINNED TO THE EXACT NEW HASH, which is what makes it safe: the
next edit to the same file produces a different hash and the acknowledgement stops applying, so it
can never become a standing exemption for a file. It also REQUIRES a reason and evidence, because an
acknowledgement that does not say why is indistinguishable from a silent hash bump.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_afl", os.path.join(REPO, "scripts", "artifact_freshness_lint.py")
)
assert _spec is not None and _spec.loader is not None
afl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(afl)


def _ack(**over) -> dict:
    base = {
        "path": "python/carnot/agentic/arc_executable_world_model.py",
        "sha256_was": "a" * 64,
        "sha256_now": "b" * 64,
        "reason": "the guard early-returns outside pytest, so it cannot alter any recorded value",
        "evidence": "three sibling artifacts rebuilt and deep-diffed: 75,755 values, 0 diffs",
    }
    base.update(over)
    return base


def test_a_complete_acknowledgement_is_honoured() -> None:
    out = afl._acknowledged_inert_drift({"freshness_acknowledgements": [_ack()]})
    assert out == {"python/carnot/agentic/arc_executable_world_model.py": "b" * 64}


def test_an_acknowledgement_without_a_reason_or_evidence_is_IGNORED() -> None:
    """The property that stops this becoming a silent hash bump wearing a structured field."""
    assert afl._acknowledged_inert_drift({"freshness_acknowledgements": [_ack(reason="")]}) == {}
    assert afl._acknowledged_inert_drift({"freshness_acknowledgements": [_ack(evidence="")]}) == {}
    assert afl._acknowledged_inert_drift({"freshness_acknowledgements": [_ack(reason="   ")]}) == {}


def test_an_acknowledgement_is_pinned_to_one_exact_hash() -> None:
    """It must NOT be a standing exemption: a later edit produces a new hash and it lapses."""
    ack = afl._acknowledged_inert_drift({"freshness_acknowledgements": [_ack()]})
    path = "python/carnot/agentic/arc_executable_world_model.py"
    assert ack[path] == "b" * 64
    assert ack[path] != "c" * 64  # a subsequent edit is NOT covered


def test_a_malformed_hash_is_rejected() -> None:
    assert (
        afl._acknowledged_inert_drift({"freshness_acknowledgements": [_ack(sha256_now="")]}) == {}
    )
    assert (
        afl._acknowledged_inert_drift({"freshness_acknowledgements": [_ack(sha256_now="deadbeef")]})
        == {}
    )


def test_absent_or_malformed_block_is_simply_empty_not_an_error() -> None:
    """A crash in the freshness layer blocks every commit while reporting nothing -- worse than a miss."""
    assert afl._acknowledged_inert_drift({}) == {}
    assert afl._acknowledged_inert_drift({"freshness_acknowledgements": "nonsense"}) == {}
    assert (
        afl._acknowledged_inert_drift({"freshness_acknowledgements": ["nonsense", 7, None]}) == {}
    )


def test_check_artifact_reports_fresh_with_the_acknowledgement_and_stale_without(tmp_path) -> None:
    """End-to-end through `check_artifact`, which is what the hook actually calls."""
    dep = tmp_path / "dep.py"
    dep.write_text("# version 2\n")
    now = hashlib.sha256(dep.read_bytes()).hexdigest()

    artifact = tmp_path / "a.json"

    def write(acks: list) -> None:
        artifact.write_text(
            json.dumps(
                {
                    "provenance": {
                        "code": [{"path": str(dep), "sha256": "0" * 64}],
                        "freshness_acknowledgements": acks,
                    }
                }
            )
        )

    write([])
    status, detail, _cmd = afl.check_artifact(artifact)
    assert status == "stale", detail

    write([_ack(path=str(dep), sha256_was="0" * 64, sha256_now=now)])
    status, detail, _cmd = afl.check_artifact(artifact)
    assert status == "fresh"
    assert any("ACKNOWLEDGED as verified-inert" in line for line in detail)

    # And an acknowledgement for the WRONG hash does not clear it -- the pin is enforced here too.
    write([_ack(path=str(dep), sha256_was="0" * 64, sha256_now="f" * 64)])
    assert afl.check_artifact(artifact)[0] == "stale"


def test_the_three_real_acknowledgements_are_complete_and_current() -> None:
    """The acknowledgements committed on 2026-07-30 must be well-formed and still apply.

    If a later commit edits `arc_executable_world_model.py` this test keeps passing (it only checks
    well-formedness), but the LINT will correctly go stale again -- which is the intended behaviour
    and why the pin is the safety property.
    """
    targets = [
        "results/outer_loop_arc_first_win_llm_on_eval_concurrency_20260727.json",
        "results/outer_loop_arc_generator_concurrency_fix_20260727.json",
        "results/outer_loop_arc_llm_on_wallclock_envelope_20260726.json",
    ]
    for rel in targets:
        d = json.loads((REPO / rel).read_text())
        acks = d["provenance"]["freshness_acknowledgements"]
        assert acks, rel
        for a in acks:
            assert len(a["sha256_now"]) == 64 and len(a["sha256_was"]) == 64, rel
            assert a["sha256_now"] != a["sha256_was"], rel
            assert len(a["reason"]) > 80, rel
            assert len(a["evidence"]) > 80, rel
            # The evidence must name where a reader can check it, not just assert inertness.
            assert "6011" in a["evidence"] or "changelog" in a["evidence"], rel
