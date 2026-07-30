"""REQ-ARC-WMTE-6048: the tracked engine-evidence store is not writable from inside a test.

THE INCIDENT (2026-07-30 review). `results/arc_e3/` holds the induced world-model engines that
several published artifacts cite as their origin fixtures. The project treats it as READ-ONLY
evidence. But `tests/python/test_codeonly_induce_scoping.py` drove `LocalGGUFProposer.induce` with
a stubbed `urlopen` and never redirected `E3_DIR`, so running the suite WROTE
`results/arc_e3/g/world_model.py`.

It was caught only by luck: the stub's canned body was byte-identical to the committed content, so
`git status` stayed clean and nothing was lost. Change one character of that stub and the suite
would have silently clobbered committed evidence, and the next `git add -A` would have committed
the clobber. A 2026-07-29 note in the module had already identified this hazard ("a process that
imports first and sets the variable afterwards keeps writing to the real evidence store") and
deliberately left it unfixed because changing write routing risks the live path.

THE FIX, and why it is scoped the way it is. The LIVE agent writes to this directory by design --
that is the store's purpose -- so a blanket write refusal would break production. A write from
inside a TEST, however, is never legitimate. `PYTEST_CURRENT_TEST` identifies exactly that case,
so the guard fires there and nowhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from carnot.agentic import arc_executable_world_model as awm


REPO = Path(__file__).resolve().parents[2]


def test_guard_refuses_a_write_into_the_tracked_evidence_store() -> None:
    """The exact path the incident wrote to must raise, not succeed."""
    with pytest.raises(RuntimeError) as exc:
        awm._guard_engine_write(REPO / "results" / "arc_e3" / "g")
    assert "TRACKED" in str(exc.value)
    assert "CARNOT_ARC_E3_DIR" in str(exc.value), "the message must say how to fix it"


def test_guard_refuses_the_store_root_and_any_game_under_it() -> None:
    for target in (
        REPO / "results" / "arc_e3",
        REPO / "results" / "arc_e3" / "ka59",
        REPO / "results" / "arc_e3" / "ft09" / "nested",
    ):
        with pytest.raises(RuntimeError):
            awm._guard_engine_write(target)


def test_guard_allows_a_redirected_store(tmp_path: Path) -> None:
    """A test that redirects the store must be entirely unaffected -- no false positives."""
    awm._guard_engine_write(tmp_path / "e3" / "g")
    # A sibling directory under results/ that is NOT the engine store is also fine.
    awm._guard_engine_write(REPO / "results" / "arc_e3_origin_fixtures" / "g")


def test_guard_is_inert_outside_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LIVE agent must keep writing here, so the guard must not fire without pytest set.

    This is the property that makes the guard safe to ship: it is scoped to the situation where a
    write is definitionally wrong, not to the directory.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    awm._guard_engine_write(REPO / "results" / "arc_e3" / "g")


def test_explicit_opt_in_reenables_the_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test whose PURPOSE is the default-path write can say so out loud."""
    monkeypatch.setenv("CARNOT_ARC_E3_ALLOW_EVIDENCE_WRITE", "1")
    awm._guard_engine_write(REPO / "results" / "arc_e3" / "g")


def test_proposer_write_paths_are_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must be wired into the WRITERS, not merely available as a helper.

    Asserted through `_write_world_model` because that is the shortest path to a real write; the
    same guard call sits at the top of `_gen_to_file` and the codex induce path.
    """
    monkeypatch.setattr(awm, "E3_DIR", REPO / "results" / "arc_e3")
    p = awm.LocalGGUFProposer(
        repo_substr="X", model_path="/x.gguf", port=59998, max_tokens=8, tries=1
    )
    with pytest.raises(RuntimeError):
        p._write_world_model("g", "import numpy as np\n")
    # And the file was not touched.
    assert os.path.exists(REPO / "results" / "arc_e3" / "g" / "world_model.py")
