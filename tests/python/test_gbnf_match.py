"""Model-free acceptance checks for the ARC induction tool grammar.

Spec: REQ-ARC-WMTE-7046, SCENARIO-ARC-WMTE-7046-A, REQ-ARC-WMTE-7044.

The verdict tables below were recorded 2026-09-05 from llama.cpp build b9606's own
`test-gbnf-validator` (tests/test-gbnf-validator.cpp), fed the same grammar text and
the same candidate strings. They pin `carnot.testing.gbnf_match` to the reference
engine: a disagreement is a bug in the Python reader, never in llama.cpp. The old
grammar is reproduced byte for byte (its sha256 is asserted) because the finding
this module records is about that exact text.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from carnot.agentic.arc_induction_tool_loop import _tool_grammar
from carnot.agentic.arc_induction_tools import TOOL_NAMES, TOOL_SCHEMAS
from carnot.testing.gbnf_match import accepts, parse_gbnf

CODE = "import numpy as np\ndef engine(grid, action, data):\n    return grid"
COMPACT = {"separators": (",", ":")}
FULL = json.dumps({"name": "run_engine_on_transitions", "arguments": {"code": CODE}}, **COMPACT)
FULL_EXTRA = json.dumps(
    {"name": "run_engine_on_transitions", "arguments": {"code": CODE, "note": "x"}}, **COMPACT
)
FIND = (
    '{"name":"find_objects","arguments":{"t":0,"which":"%s",'
    '"predicate_code":"def accept(obj): return True","max_objects":5}}'
)

CASES = {
    "A_empty_shell_run_engine": '{"name":"run_engine_on_transitions","arguments":{}}',
    "B_full_run_engine": FULL,
    "C_unknown_name": '{"name":"bogus","arguments":{}}',
    "D_no_arguments_key": '{"name":"run_engine_on_transitions"}',
    "E_wrong_key": '{"name":"run_engine_on_transitions","arguments":{"cod":"x"}}',
    "F_diff_grids_missing_t": '{"name":"diff_grids","arguments":{}}',
    "G_list_transitions_empty_ok": '{"name":"list_transitions","arguments":{}}',
    "H_empty_code_string": '{"name":"run_engine_on_transitions","arguments":{"code":""}}',
    "I_code_wrong_type": '{"name":"run_engine_on_transitions","arguments":{"code":5}}',
    "J_query_region_partial": '{"name":"query_region","arguments":{"t":0}}',
    "K_query_region_full": '{"name":"query_region","arguments":{"t":0,"r0":0,"r1":2,"c0":0,"c1":2}}',
    "L_find_objects_full": FIND % "before",
    "M_find_objects_bad_enum": FIND % "sideways",
    "N_full_run_engine_plus_extra_key": FULL_EXTRA,
    "O_query_region_full_with_which": (
        '{"name":"query_region","arguments":{"t":0,"r0":0,"r1":2,"c0":0,"c1":2,"which":"after"}}'
    ),
    "P_full_run_engine_SPACED": json.dumps(
        {"name": "run_engine_on_transitions", "arguments": {"code": CODE}}
    ),
    "Q_run_engine_code_after_extra": (
        '{"name":"run_engine_on_transitions","arguments":{"note":"x","code":"import numpy as np"}}'
    ),
    "R_diff_grids_t_string": '{"name":"diff_grids","arguments":{"t":"0"}}',
}

# The envelope-only grammar shipped in commit d5d72141ca, reproduced from its builder.
_OLD_TERMINALS = " | ".join(json.dumps(json.dumps(n)) for n in TOOL_NAMES)
OLD_GRAMMAR = (
    'root ::= "{\\"name\\":" tool-name ",\\"arguments\\":" object "}"\n'
    + "tool-name ::= "
    + _OLD_TERMINALS
    + "\n"
    + r"""
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array ::= "[" ws (value ("," ws value)*)? "]" ws
value ::= object | array | string | number | ("true" | "false" | "null") ws
string ::= "\"" char* "\"" ws
char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= [ \t\n\r]*
"""
)
OLD_GRAMMAR_SHA256_PREFIX = "6a09d16de9e9deec"

# test-gbnf-validator verdicts, 2026-09-05, on the grammar above (V = valid, I = invalid).
OLD_VERDICTS = dict(
    A_empty_shell_run_engine="V",
    B_full_run_engine="V",
    C_unknown_name="I",
    D_no_arguments_key="I",
    E_wrong_key="V",
    F_diff_grids_missing_t="V",
    G_list_transitions_empty_ok="V",
    H_empty_code_string="V",
    I_code_wrong_type="V",
    J_query_region_partial="V",
    K_query_region_full="V",
    L_find_objects_full="V",
    M_find_objects_bad_enum="V",
    N_full_run_engine_plus_extra_key="V",
    O_query_region_full_with_which="V",
    P_full_run_engine_SPACED="I",
    Q_run_engine_code_after_extra="V",
    R_diff_grids_t_string="V",
)
# test-gbnf-validator verdicts, 2026-09-05, on _tool_grammar(TOOL_SCHEMAS).
NEW_VERDICTS = dict(
    A_empty_shell_run_engine="I",
    B_full_run_engine="V",
    C_unknown_name="I",
    D_no_arguments_key="I",
    E_wrong_key="I",
    F_diff_grids_missing_t="I",
    G_list_transitions_empty_ok="V",
    H_empty_code_string="I",
    I_code_wrong_type="I",
    J_query_region_partial="I",
    K_query_region_full="V",
    L_find_objects_full="V",
    M_find_objects_bad_enum="I",
    N_full_run_engine_plus_extra_key="V",
    O_query_region_full_with_which="V",
    P_full_run_engine_SPACED="I",
    Q_run_engine_code_after_extra="I",
    R_diff_grids_t_string="I",
)


def test_old_grammar_text_is_the_recorded_one():
    """REQ-ARC-WMTE-7046: the pinned verdicts are about these exact bytes."""
    assert hashlib.sha256(OLD_GRAMMAR.encode()).hexdigest().startswith(OLD_GRAMMAR_SHA256_PREFIX)
    assert len(OLD_GRAMMAR.encode()) == 614
    assert set(CASES) == set(OLD_VERDICTS) == set(NEW_VERDICTS)


@pytest.mark.parametrize("case", sorted(CASES))
def test_reader_matches_validator_on_the_envelope_only_grammar(case):
    """REQ-ARC-WMTE-7046: the Python reader agrees with llama.cpp on the old grammar."""
    assert accepts(OLD_GRAMMAR, CASES[case]) is (OLD_VERDICTS[case] == "V")


@pytest.mark.parametrize("case", sorted(CASES))
def test_reader_matches_validator_on_the_required_arguments_grammar(case):
    """REQ-ARC-WMTE-7046: the Python reader agrees with llama.cpp on the new grammar."""
    assert accepts(_tool_grammar(TOOL_SCHEMAS), CASES[case]) is (NEW_VERDICTS[case] == "V")


def test_empty_shell_was_grammatical_and_is_not_now():
    """SCENARIO-ARC-WMTE-7046-A: the finding, and the fix, in one place."""
    empty = CASES["A_empty_shell_run_engine"]
    assert accepts(OLD_GRAMMAR, empty)
    new = _tool_grammar(TOOL_SCHEMAS)
    assert not accepts(new, empty)
    assert accepts(new, FULL)
    assert not accepts(new, CASES["H_empty_code_string"])
    assert accepts(new, CASES["G_list_transitions_empty_ok"])


def test_required_arguments_grammar_covers_every_session_tool():
    """REQ-ARC-WMTE-7046: one call rule per schema, required keys in schema order."""
    grammar = _tool_grammar(TOOL_SCHEMAS)
    rules = parse_gbnf(grammar)
    # root ::= call-0 | call-1 | ...: one single-item alternative per schema.
    assert [alt[0][1] for alt in rules["root"]] == [f"call-{i}" for i in range(len(TOOL_SCHEMAS))]
    for i, schema in enumerate(TOOL_SCHEMAS):
        required = schema["function"].get("parameters", {}).get("required") or []
        text = grammar.splitlines()[2 + 2 * i]
        assert text.startswith(f"args-{i} ::= ")
        positions = [text.find(json.dumps(json.dumps(k))) for k in required]
        assert all(p >= 0 for p in positions)
        assert positions == sorted(positions)
        if not required:
            assert text == f"args-{i} ::= object"


# ---- reader self-tests: each construct of the supported subset ---------------------


def test_literal_and_char_class():
    """REQ-ARC-WMTE-7046: literals match exactly; classes honour ranges and negation."""
    assert accepts('root ::= "ab" [0-9]', "ab7")
    assert not accepts('root ::= "ab" [0-9]', "abx")
    assert not accepts('root ::= "ab" [0-9]', "ab77")
    assert accepts("root ::= [^a-c]", "d")
    assert not accepts("root ::= [^a-c]", "b")
    assert accepts("root ::= [a-cx]", "x")


@pytest.mark.parametrize(
    "grammar,ok,bad",
    [
        ('root ::= "a"*', ["", "a", "aaa"], ["b"]),
        ('root ::= "a"+', ["a", "aa"], [""]),
        ('root ::= "a"?', ["", "a"], ["aa"]),
        ('root ::= "a"{2}', ["aa"], ["a", "aaa"]),
        ('root ::= "a"{1,2}', ["a", "aa"], ["", "aaa"]),
        ('root ::= "a"{2,}', ["aa", "aaaa"], ["a"]),
        ('root ::= "ab"*', ["", "abab"], ["aba"]),
    ],
)
def test_repetition_forms(grammar, ok, bad):
    """REQ-ARC-WMTE-7046: every suffix, and a suffix binds the whole literal."""
    for s in ok:
        assert accepts(grammar, s), (grammar, s)
    for s in bad:
        assert not accepts(grammar, s), (grammar, s)


def test_escapes_in_literals_and_classes():
    """REQ-ARC-WMTE-7046: the escapes llama.cpp's parse_char accepts."""
    assert accepts(r'root ::= "\"" [^"\\] "\""', '"x"')
    assert not accepts(r'root ::= "\"" [^"\\] "\""', '"\\"')
    assert accepts(r'root ::= "\x41B\n"', "AB\n")
    assert accepts(r"root ::= [\]\[]", "]")
    with pytest.raises(ValueError):
        accepts(r'root ::= "\q"', "q")


def test_groups_alternation_nesting_and_comments():
    """REQ-ARC-WMTE-7046: groups may span lines; comments are ignored."""
    grammar = 'root ::= ("a" | "b") inner+  # trailing comment\ninner ::= (\n  "c" |\n  "d"\n)\n'
    assert accepts(grammar, "acd")
    assert accepts(grammar, "bc")
    assert not accepts(grammar, "a")
    assert not accepts(grammar, "ca")


def test_nullable_repetition_terminates():
    """REQ-ARC-WMTE-7046: a repeated item that can match empty must not loop."""
    grammar = 'root ::= (ws)* "x"\nws ::= [ ]*\n'
    assert accepts(grammar, "x")
    assert accepts(grammar, "  x")
    assert not accepts(grammar, "  ")


def test_undefined_rule_missing_root_and_unsupported_forms_raise():
    """REQ-ARC-WMTE-7046: the reader fails loudly outside its subset."""
    with pytest.raises(ValueError):
        parse_gbnf('root ::= "a" other\n')
    with pytest.raises(ValueError):
        accepts('other ::= "a"\n', "a")
    with pytest.raises(NotImplementedError):
        accepts("root ::= <[1]>", "a")
    with pytest.raises(NotImplementedError):
        accepts('root ::= "a" .', "ab")
    with pytest.raises(ValueError):
        accepts("root ::= *", "")
