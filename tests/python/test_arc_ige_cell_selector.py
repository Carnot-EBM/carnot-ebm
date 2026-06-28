"""Tests for IGE-style LLM-guided Go-Explore cell selection (arXiv:2405.15143).

The contract under test (built 2026-06-28, operator-directed "let's try IGE-style LLM-guided
Go-Explore"): an IGECellSelector RANKS already-archived Go-Explore cells via an LLM promisingness
judgement, returns the chosen cell index, and returns None (so the archive keeps its heuristic) on any
failure — no GPU server, an unparseable reply, fewer than two cells, or disabled. The archive delegates
cell choice to the selector when present and falls back to its min() heuristic otherwise. The selector is
never an oracle (verifier_is_oracle=False) and cannot fabricate a solve. Spec: the IGE first-win lever.
"""

from __future__ import annotations

from carnot.agentic.arc_go_explore import (
    GoExploreReplayArchive,
    coerce_go_explore_archive,
)
from carnot.agentic.arc_ige_cell_selector import (
    IGECellSelector,
    _build_prompt,
    _parse_index,
    _render_signature,
    coerce_ige_cell_selector,
)


class _StubProposer:
    """Stub LocalGGUFProposer: returns a scripted (ok, text) for complete_text, records the prompt."""

    def __init__(self, ok: bool = True, text: str = "0") -> None:
        self._ok = ok
        self._text = text
        self.prompts: list[str] = []

    def complete_text(self, prompt: str, **_kwargs) -> tuple[bool, str]:
        self.prompts.append(prompt)
        return self._ok, self._text


def _descriptors(n: int) -> list[dict]:
    return [
        {"index": i, "level": 0, "depth": i + 1, "visits": 0, "seen": 1, "signature": (0, 1, i % 3, 2)}
        for i in range(n)
    ]


# --- helper rendering / parsing -------------------------------------------------------------------

def test_parse_index_takes_first_in_range():
    assert _parse_index("I would pick 2 here", 5) == 2
    assert _parse_index("cell 3.", 5) == 3


def test_parse_index_rejects_out_of_range_and_nonnumeric():
    assert _parse_index("99", 3) is None
    assert _parse_index("none of them", 4) is None
    assert _parse_index("", 4) is None


def test_render_signature_lays_out_square_grid():
    # a 4-element signature renders as a 2x2 grid of single-char colours
    assert _render_signature((0, 1, 2, 3)) == "01\n23"


def test_build_prompt_lists_cells_and_demands_single_integer():
    prompt = _build_prompt(_descriptors(3))
    assert "CELL 0:" in prompt and "CELL 2:" in prompt
    assert "integer index" in prompt.lower()


# --- selector firing / fallback -------------------------------------------------------------------

def test_selector_fires_and_returns_mapped_index():
    sel = IGECellSelector(proposer=_StubProposer(ok=True, text="1"))
    choice = sel(_descriptors(3))
    assert choice == 1
    diag = sel.diagnostics()
    assert diag["calls"] == 1 and diag["llm_choices"] == 1 and diag["fallbacks"] == 0
    assert diag["verifier_is_oracle"] is False


def test_selector_returns_none_below_two_cells():
    sel = IGECellSelector(proposer=_StubProposer(ok=True, text="0"))
    assert sel(_descriptors(1)) is None
    # no LLM call should be spent on a trivial single-cell choice
    assert sel.diagnostics()["calls"] == 0


def test_selector_counts_server_unavailable_and_falls_back():
    sel = IGECellSelector(proposer=_StubProposer(ok=False, text="no server"))
    assert sel(_descriptors(3)) is None
    diag = sel.diagnostics()
    assert diag["server_unavailable"] == 1 and diag["fallbacks"] == 1 and diag["llm_choices"] == 0


def test_selector_counts_parse_failure_and_falls_back():
    sel = IGECellSelector(proposer=_StubProposer(ok=True, text="I cannot decide"))
    assert sel(_descriptors(3)) is None
    diag = sel.diagnostics()
    assert diag["parse_failures"] == 1 and diag["fallbacks"] == 1


def test_selector_disabled_returns_none_without_calling_llm():
    stub = _StubProposer(ok=True, text="0")
    sel = IGECellSelector(proposer=stub, enabled=False)
    assert sel(_descriptors(3)) is None
    assert stub.prompts == []


def test_selector_truncates_to_max_cells_and_maps_back_to_original_index():
    # 5 cells but only 3 shown to the LLM; the model says "2" -> the 3rd SHOWN cell, whose original
    # descriptor index is also 2 here. Confirms the prompt is bounded and the mapping uses the index field.
    stub = _StubProposer(ok=True, text="2")
    sel = IGECellSelector(proposer=stub, max_cells_in_prompt=3)
    choice = sel(_descriptors(5))
    assert choice == 2
    # the prompt must contain only the first 3 cells (bounded prompt)
    assert "CELL 3:" not in stub.prompts[0] and "CELL 2:" in stub.prompts[0]


# --- coerce -------------------------------------------------------------------------------------

def test_coerce_ige_variants():
    assert coerce_ige_cell_selector(None) is None
    assert coerce_ige_cell_selector(False) is None
    assert coerce_ige_cell_selector("") is None
    inst = IGECellSelector(proposer=_StubProposer())
    assert coerce_ige_cell_selector(inst) is inst
    assert isinstance(coerce_ige_cell_selector(True), IGECellSelector)
    assert isinstance(coerce_ige_cell_selector("ige"), IGECellSelector)
    assert coerce_ige_cell_selector({"enabled": False}) is None
    cfg = coerce_ige_cell_selector({"enabled": True, "max_cells_in_prompt": 5})
    assert isinstance(cfg, IGECellSelector) and cfg.max_cells_in_prompt == 5


# --- archive delegation --------------------------------------------------------------------------

def _populate(arch: GoExploreReplayArchive) -> None:
    # two eligible cells with distinct prefixes; insertion order is the eligible order.
    arch._cells[(0, (0, 0, 1, 1))] = {"prefix": [{"action": 1, "data": None}], "visits": 0, "depth": 5, "seen": 1}
    arch._cells[(0, (1, 2, 3, 4))] = {"prefix": [{"action": 2, "data": None}], "visits": 0, "depth": 2, "seen": 1}


def test_archive_delegates_cell_choice_to_selector():
    # selector returns index 1 -> the SECOND cell (action 2) must be returned, NOT the heuristic's pick
    # (the heuristic would prefer the deeper depth=5 first cell).
    arch = GoExploreReplayArchive(selector=lambda descs: 1)
    _populate(arch)
    prefix = arch.select_prefix()
    assert prefix == [{"action": 2, "data": None}]
    diag = arch.diagnostics()
    assert diag["selector_enabled"] is True and diag["selector_used"] == 1 and diag["selector_fallbacks"] == 0


def test_archive_falls_back_to_heuristic_when_selector_declines():
    # selector returns None -> heuristic picks min(visits, -depth, ...) = the DEEPER cell (depth 5, action 1).
    arch = GoExploreReplayArchive(selector=lambda descs: None)
    _populate(arch)
    prefix = arch.select_prefix()
    assert prefix == [{"action": 1, "data": None}]
    assert arch.diagnostics()["selector_fallbacks"] == 1


def test_archive_falls_back_on_out_of_range_selector_index():
    arch = GoExploreReplayArchive(selector=lambda descs: 99)
    _populate(arch)
    prefix = arch.select_prefix()
    assert prefix == [{"action": 1, "data": None}]  # heuristic deeper-cell pick
    assert arch.diagnostics()["selector_fallbacks"] == 1


def test_archive_selector_exception_falls_back():
    def _boom(descs):
        raise RuntimeError("selector blew up")

    arch = GoExploreReplayArchive(selector=_boom)
    _populate(arch)
    prefix = arch.select_prefix()
    assert prefix == [{"action": 1, "data": None}]
    assert arch.diagnostics()["selector_fallbacks"] == 1


def test_coerce_go_explore_archive_accepts_callable_and_ige_string():
    called = {}

    def _sel(descs):
        called["hit"] = True
        return 0

    arch = coerce_go_explore_archive({"enabled": True, "selector": _sel})
    assert arch is not None and arch.selector is _sel
    arch_ige = coerce_go_explore_archive({"enabled": True, "selector": "ige"})
    assert isinstance(arch_ige.selector, IGECellSelector)
