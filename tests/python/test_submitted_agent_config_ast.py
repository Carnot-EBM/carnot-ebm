"""Tests for the shared submitted-agent-config source reader.

Spec refs: REQ-ARC-WMTE-4548, SCENARIO-ARC-WMTE-4548, REQ-ARC-WMTE-4560,
SCENARIO-ARC-WMTE-4560 -- both integration gates read the submitted agent's configuration
through this module, and both assert that the config they measured is the config that ships.

WHY THIS FILE EXISTS, AND WHY THE FIRST TEST IS THE IMPORTANT ONE.
The reader parses ``arc_competition_agent.py`` with ``ast`` instead of importing it, to avoid
dragging the whole agentic stack (and an optional ``llama_cpp``) into a gate that only wants to
read ~96 flags. That tradeoff buys speed and robustness but introduces a specific hazard: a
hand-written evaluator can return a dict that is *plausible but wrong*, and nothing downstream
would notice, because the gates only ever asserted that the read SUCCEEDED.

That gap was measured, not assumed. Deliberately corrupting the reader's comparison operator --
so ``frozen_generator.mtp`` flipped True->False and ``spec_type`` flipped "draft-mtp"->None --
left all 15 tests across both gate suites GREEN. A gate whose entire purpose is "the agent we
measured is the agent we submit" cannot tell that it read the wrong configuration. So the first
test below pins the parse against the real import, which is the only ground truth there is.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from carnot.submitted_agent_config_ast import (
    UnresolvableConfigValue,
    parse_submitted_agent_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_SOURCE = REPO_ROOT / "python" / "carnot" / "agentic" / "arc_competition_agent.py"


def test_parsed_config_is_identical_to_the_real_import() -> None:
    """The parse must equal what Python itself produces -- same keys, same values, same types.

    This is the assertion that catches a reader which resolves a name to the wrong value. It is
    also the reason this test imports the heavy module even though the whole point of the reader
    is to avoid doing so: a test may pay a cost the production path refuses to, and there is no
    other way to know the cheap path agrees with the real one.
    """
    parsed = parse_submitted_agent_config(AGENT_SOURCE)

    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    truth = dict(SUBMITTED_AGENT_CONFIG)

    assert set(parsed) == set(truth), (
        "parsed key set differs from the imported one; a config entry is being dropped or invented"
    )
    # Compared key-by-key rather than with a single dict == so a failure names the offender.
    mismatched = {key: (parsed[key], truth[key]) for key in truth if parsed[key] != truth[key]}
    assert not mismatched, f"parsed values disagree with the real import: {mismatched}"
    # Type identity matters as much as equality here: True == 1 in Python, so a reader that
    # returned 1 where the config says True would pass an == check while writing a wrong type
    # into the artifact.
    wrong_type = {
        key: (type(parsed[key]).__name__, type(truth[key]).__name__)
        for key in truth
        if type(parsed[key]) is not type(truth[key])
    }
    assert not wrong_type, f"parsed values have the wrong types: {wrong_type}"


def test_the_frozen_generator_nested_dict_resolves_its_import_aliases() -> None:
    """A regression pin on the concrete shapes that broke the original reader.

    ``frozen_generator`` is a nested dict whose values are ``ARC_LIVE_GENERATOR_*`` names
    imported from another module, plus one comparison and one conditional. Each of those is a
    form the original flat ``literal_eval`` reader could not evaluate.
    """
    config = parse_submitted_agent_config(AGENT_SOURCE)
    frozen = config["frozen_generator"]

    assert isinstance(frozen["model_id"], str) and frozen["model_id"], (
        "model_id came from a `from ... import` alias; an unresolved alias would leave it empty"
    )
    assert isinstance(frozen["mtp"], bool), (
        'mtp is defined as `X != "0"` -- a Compare node, which literal_eval cannot evaluate'
    )
    assert frozen["spec_type"] is None or isinstance(frozen["spec_type"], str), (
        "spec_type is a conditional expression (IfExp)"
    )
    assert isinstance(frozen["required_shared_libraries"], list)

    # The two top-level names that produced the original bare KeyError / silent drop.
    assert isinstance(config["frontier_tier_count"], int), (
        "frontier_tier_count is assigned from an imported TIER_COUNT alias -- the exact "
        "construct that used to raise KeyError('SUBMITTED_FRONTIER_TIER_COUNT')"
    )
    assert isinstance(config["goal_energy_source"], str) and config["goal_energy_source"], (
        "goal_energy_source is a bare imported name, the second latent instance of the same bug"
    )


def _parse_snippet(tmp_path: Path, source: str) -> dict[str, object]:
    """Write a synthetic agent module and parse it, so the unit cases below stay hermetic."""
    path = tmp_path / "fake_agent.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return parse_submitted_agent_config(path)


def test_resolves_a_local_constant_reference_chain(tmp_path: Path) -> None:
    """``A = 5; B = A; config = {"k": B}`` must resolve to 5, not be dropped."""
    config = _parse_snippet(
        tmp_path,
        """
        FIRST = 5
        SECOND = FIRST
        SUBMITTED_AGENT_CONFIG = {"k": SECOND}
        """,
    )
    assert config == {"k": 5}


def test_resolves_a_from_import_alias(tmp_path: Path) -> None:
    """The origin bug: a constant assigned from an imported alias.

    ``math.pi`` stands in for the real ``TIER_COUNT`` -- a genuine import from a real module, so
    this exercises the actual import-and-getattr path rather than a stub of it.
    """
    config = _parse_snippet(
        tmp_path,
        """
        from math import pi as PI_ALIAS
        SUBMITTED_PI = PI_ALIAS
        SUBMITTED_AGENT_CONFIG = {"pi": SUBMITTED_PI, "direct": PI_ALIAS}
        """,
    )
    assert config["pi"] == pytest.approx(3.14159, abs=1e-4)
    assert config["direct"] == pytest.approx(3.14159, abs=1e-4)


def test_a_later_local_assignment_shadows_an_import_of_the_same_name(tmp_path: Path) -> None:
    """Resolution order must match Python's: the assignment seen first wins at that point."""
    config = _parse_snippet(
        tmp_path,
        """
        from math import pi as VALUE
        VALUE = 42
        SUBMITTED_AGENT_CONFIG = {"v": VALUE}
        """,
    )
    assert config == {"v": 42}


def test_resolves_names_nested_inside_containers(tmp_path: Path) -> None:
    """Names inside a dict/list/tuple must resolve at any depth, not just at the top level."""
    config = _parse_snippet(
        tmp_path,
        """
        INNER = "deep"
        NUM = 7
        SUBMITTED_AGENT_CONFIG = {
            "nested": {"a": INNER, "b": [NUM, {"c": INNER}]},
            "tup": (NUM, INNER),
        }
        """,
    )
    assert config["nested"] == {"a": "deep", "b": [7, {"c": "deep"}]}
    assert config["tup"] == (7, "deep")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ('FLAG != "0"', True),
        ('FLAG == "0"', False),
        ('"on" if FLAG != "0" else None', "on"),
        ('"on" if FLAG == "0" else None', None),
        ("not TRUTHY", False),
        ("TRUTHY and FLAG", "1"),
        ("FALSY or FLAG", "1"),
    ],
)
def test_evaluates_the_bounded_pure_expression_set(
    tmp_path: Path, expression: str, expected: object
) -> None:
    """The config really does derive flags from a pinned string, so these must work.

    Each case is side-effect-free by construction -- no call, no attribute access -- which is the
    boundary the reader draws.
    """
    config = _parse_snippet(
        tmp_path,
        f"""
        FLAG = "1"
        TRUTHY = True
        FALSY = False
        SUBMITTED_AGENT_CONFIG = {{"v": {expression}}}
        """,
    )
    assert config["v"] == expected


def test_a_computed_value_refuses_loudly_and_names_the_key(tmp_path: Path) -> None:
    """The whole point of the rewrite: fail where the cause is, naming the key.

    The original reader silently dropped what it could not evaluate and then died with a bare
    ``KeyError`` from an unrelated lookup, sending the reader of the traceback to the wrong place.
    """
    with pytest.raises(UnresolvableConfigValue) as excinfo:
        _parse_snippet(
            tmp_path,
            """
            SUBMITTED_AGENT_CONFIG = {"computed": len("abc")}
            """,
        )
    message = str(excinfo.value)
    assert "'computed'" in message, "the failure must name the offending config key"
    assert "Call" in message, "the failure must say what kind of expression it refused"


def test_an_unresolvable_name_refuses_loudly_rather_than_keyerror(tmp_path: Path) -> None:
    """A name that is neither an earlier constant nor an import must not surface as KeyError."""
    with pytest.raises(UnresolvableConfigValue) as excinfo:
        _parse_snippet(
            tmp_path,
            """
            SUBMITTED_AGENT_CONFIG = {"k": NEVER_DEFINED}
            """,
        )
    message = str(excinfo.value)
    assert "NEVER_DEFINED" in message
    assert "'k'" in message


def test_a_forward_reference_is_not_resolvable(tmp_path: Path) -> None:
    """A constant defined AFTER the config cannot be in scope, and must not be silently invented.

    Python would raise NameError building the dict, so resolving it here would make the reader
    disagree with the module it claims to describe.
    """
    with pytest.raises(UnresolvableConfigValue):
        _parse_snippet(
            tmp_path,
            """
            SUBMITTED_AGENT_CONFIG = {"k": DEFINED_LATER}
            DEFINED_LATER = 1
            """,
        )


def test_a_module_without_the_config_returns_empty(tmp_path: Path) -> None:
    """No config assignment at all is an empty dict, not an exception."""
    assert _parse_snippet(tmp_path, "OTHER = 1\n") == {}


def test_unresolvable_unrelated_constants_do_not_abort_the_parse(tmp_path: Path) -> None:
    """Most module-level assignments are computed and irrelevant; skipping THOSE is correct.

    The distinction this pins: an unresolvable constant the config never references is skipped
    quietly, while one the config DOES reference fails loudly (previous two tests). Getting this
    backwards in either direction is the bug -- abort on everything and the reader is useless;
    skip on everything and we are back to the silent drop.
    """
    config = _parse_snippet(
        tmp_path,
        """
        IRRELEVANT = len("abc")
        WANTED = 3
        SUBMITTED_AGENT_CONFIG = {"k": WANTED}
        """,
    )
    assert config == {"k": 3}
