"""Tests for REQ-LEARN-3958: cross-game DSL transfer."""

from carnot.agentic.arc_world_model_synth import extract_library_fragments


def test_extract_library_fragments_extracts_helpers_only():
    """SCENARIO-LEARN-3958: extracts only helper functions, not predict."""
    code = '''
def _helper_func(grid):
    return grid + 1

def predict(grid, action):
    return _helper_func(grid)
'''
    fragments = extract_library_fragments(code)
    assert len(fragments) == 1
    assert "def _helper_func(grid):" in fragments[0]
    assert "def predict" not in fragments[0]


def test_extract_library_fragments_handles_syntax_errors():
    fragments = extract_library_fragments("def predict(grid, action): return")
    assert len(fragments) == 0

    fragments = extract_library_fragments("def _bad_syntax() ->")
    assert len(fragments) == 0
