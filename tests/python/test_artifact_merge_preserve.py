"""Tests for scripts/artifact_merge_preserve.py — analyzer rebuilds carry
hand-authored keys forward.

REQ-OPS-REBUILD-PRESERVE-1: a freshness-mandated rebuild must not delete
the record left by the previous one. Generated keys win; old-only
top-level keys carry; `provenance.freshness_acknowledgements` survives a
regenerated provenance; an unreadable existing artifact refuses the
merge rather than being overwritten.

The incident regression reads the REAL artifact the 2026-08-21 incident
destroyed keys on (`results/outer_loop_arc_early_stop_grace_sweep_
20260726.json`, tracked in git so always present — asserted, never
skipped), copies it under tmp_path, and proves every `rebuild_note_*`
key and all acknowledgements survive a simulated rebuild byte-identical.
All writes go to tmp_path only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_INCIDENT_ARTIFACT = _REPO / "results" / "outer_loop_arc_early_stop_grace_sweep_20260726.json"


def _load():
    spec = importlib.util.spec_from_file_location(
        "artifact_merge_preserve", _REPO / "scripts" / "artifact_merge_preserve.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load()


def _old() -> dict:
    return {
        "headline": {"delta": 0.1},
        "provenance": {
            "code": [{"path": "scripts/x.py", "sha256": "a" * 64}],
            "git_head": "old",
            "freshness_acknowledgements": [
                {
                    "path": "scripts/dep.py",
                    "sha256_was": "b" * 64,
                    "sha256_now": "c" * 64,
                    "reason": "comment-only",
                    "evidence": "diff shows docstring change only",
                }
            ],
        },
        "rebuild_note_20260731_split_enable": {"why_rebuilt": "split enable flag"},
        "duration_s": 1.0,
    }


def _new() -> dict:
    return {
        "headline": {"delta": 0.2},
        "provenance": {
            "code": [{"path": "scripts/x.py", "sha256": "d" * 64}],
            "git_head": "new",
        },
        "duration_s": 2.0,
    }


def test_rebuild_note_carries_byte_identical():
    """SCENARIO-OPS-REBUILD-PRESERVE-1-REBUILD-NOTE."""
    merged = M.merge_preserve(_old(), _new())
    assert merged["rebuild_note_20260731_split_enable"] == {"why_rebuilt": "split enable flag"}


def test_acks_carry_into_regenerated_provenance():
    """SCENARIO-OPS-REBUILD-PRESERVE-1-ACKS: the acks survive; the
    current-build provenance facts are the NEW ones."""
    merged = M.merge_preserve(_old(), _new())
    prov = merged["provenance"]
    assert prov["git_head"] == "new"
    assert prov["code"][0]["sha256"] == "d" * 64
    assert prov["freshness_acknowledgements"] == _old()["provenance"]["freshness_acknowledgements"]


def test_generated_wins_on_conflict():
    """SCENARIO-OPS-REBUILD-PRESERVE-1-GENERATED-WINS."""
    merged = M.merge_preserve(_old(), _new())
    assert merged["headline"] == {"delta": 0.2}
    assert merged["duration_s"] == 2.0


def test_retired_key_drops():
    """SCENARIO-OPS-REBUILD-PRESERVE-1-RETIRED-DROPS: deliberate drops go
    through the explicit argument, never through silence."""
    merged = M.merge_preserve(_old(), _new(), retired_keys=("rebuild_note_20260731_split_enable",))
    assert "rebuild_note_20260731_split_enable" not in merged


def test_new_acks_win_when_generated_provides_them():
    """The generated payload owns what it emits — including acks, when a
    rebuild deliberately rewrites them."""
    new = _new()
    new["provenance"]["freshness_acknowledgements"] = []
    merged = M.merge_preserve(_old(), new)
    assert merged["provenance"]["freshness_acknowledgements"] == []


def test_dropped_provenance_subkey_is_stated_not_silent(capsys):
    """Rule 4: deleting an old sub-key inside a regenerated dict may be
    correct; doing it silently is not."""
    old = _old()
    old["provenance"]["hand_added_context"] = "the note a future author nests here"
    M.merge_preserve(old, _new())
    err = capsys.readouterr().err
    assert "hand_added_context" in err
    assert "provenance" in err


def test_unreadable_existing_refuses_and_leaves_file(tmp_path):
    """SCENARIO-OPS-REBUILD-PRESERVE-1-UNREADABLE-REFUSES: fail closed —
    the rebuild is re-runnable, the keys under the unreadable file are
    not recoverable after an overwrite."""
    out = tmp_path / "artifact.json"
    out.write_text("{corrupt")
    with pytest.raises(M.MergeRefusedError):
        M.merge_preserve_with_file(out, _new())
    assert out.read_text() == "{corrupt"


def test_non_dict_existing_refuses(tmp_path):
    out = tmp_path / "artifact.json"
    out.write_text("[1, 2, 3]")
    with pytest.raises(M.MergeRefusedError):
        M.merge_preserve_with_file(out, _new())


def test_missing_file_is_first_build(tmp_path):
    merged = M.merge_preserve_with_file(tmp_path / "absent.json", _new())
    assert merged == _new()


def test_incident_artifact_survives_simulated_rebuild(tmp_path):
    """Regression on the ACTUAL incident inputs (2026-08-21): the real
    early-stop-sweep artifact's rebuild_note_* keys and all 25
    acknowledgements survive a simulated wholesale rebuild
    byte-identical. The artifact is tracked in git; its absence is a
    broken checkout, asserted rather than skipped."""
    assert _INCIDENT_ARTIFACT.exists(), "tracked incident artifact missing from checkout"
    real = json.loads(_INCIDENT_ARTIFACT.read_text())
    note_keys = [k for k in real if k.startswith("rebuild_note_")]
    acks = real["provenance"]["freshness_acknowledgements"]
    assert note_keys and acks  # the incident keys are really there

    # A wholesale rebuild payload: everything the analyzer would emit —
    # every key EXCEPT the hand-authored notes — with a fresh provenance
    # that (like the real analyzers) does not emit acknowledgements.
    generated = {k: v for k, v in real.items() if not k.startswith("rebuild_note_")}
    generated["provenance"] = {
        k: v for k, v in real["provenance"].items() if k != "freshness_acknowledgements"
    }

    out = tmp_path / "artifact.json"
    out.write_text(json.dumps(real))
    merged = M.merge_preserve_with_file(out, generated)

    for key in note_keys:
        assert json.dumps(merged[key], sort_keys=True) == json.dumps(real[key], sort_keys=True)
    assert json.dumps(merged["provenance"]["freshness_acknowledgements"], sort_keys=True) == (
        json.dumps(acks, sort_keys=True)
    )


def _calls_merge_preserve(path: Path) -> bool:
    """AST-verified: the file contains a real merge_preserve_with_file CALL.

    String presence was the first draft and it counted docstrings and
    comments (adversarial-review note, 2026-08-22); a Call node cannot be
    satisfied by prose."""
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "merge_preserve_with_file":
                return True
    return False


def test_every_registered_rebuilder_calls_merge_preserve():
    """Wiring, scoped by CONCEPT not by filename glob: every rebuilder the
    freshness system itself registers (`ops/analyzer_artifact_index.json`
    analyzer fields) must merge-preserve before writing — the glob-only
    first draft missed arc_gateway_card_ground_truth.py and four others
    (adversarial-review finding 1, 2026-08-22).

    Exempt, with the reason stated: builders resident under `results/`
    are one-shot session scripts kept as evidence next to their rows;
    they are not re-run by the freshness lint's remedy path, and editing
    files under results/ is avoided on principle (evidence, read-only).
    The interim habit (diff-before-staging) still applies if one is ever
    re-run by hand."""
    index = json.loads((_REPO / "ops" / "analyzer_artifact_index.json").read_text())
    rebuilders = sorted({meta["analyzer"] for meta in index.values()})
    assert rebuilders, "empty analyzer index — cannot verify wiring"
    missing = []
    for rel in rebuilders:
        if rel.startswith("results/"):
            continue  # stated exemption above
        path = _REPO / rel
        assert path.exists(), f"registered rebuilder missing from checkout: {rel}"
        if not _calls_merge_preserve(path):
            missing.append(rel)
    assert missing == [], f"registered rebuilders writing without merge-preserve: {missing}"


def test_every_analyze_script_calls_merge_preserve():
    """The analyze_*.py family is also checked by glob, so a NEW analyzer
    that has not yet registered its first artifact is still caught."""
    analyzers = sorted((_REPO / "scripts").glob("analyze_*.py"))
    assert len(analyzers) >= 10
    missing = [p.name for p in analyzers if not _calls_merge_preserve(p)]
    assert missing == [], f"analyzers writing without merge-preserve: {missing}"
