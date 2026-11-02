"""Match a string against a GBNF grammar with no model and no llama.cpp binary.

WHY THIS EXISTS (2026-09-05). The live induction tool grammar was believed to
require a code payload. It did not: `{"name":"run_engine_on_transitions",
"arguments":{}}` was grammatical, and a CPU trial returned exactly that twice.
Nothing in the test suite could say what the grammar accepted, because the
reference engine lives inside llama.cpp and needs a compiled binary. This module
is a small reader for the GBNF subset our grammars use, so a test can assert
acceptance and rejection of exact strings without a model.

WHAT IT MIRRORS. Parsing follows `src/llama-grammar.cpp` (build b9606,
`parse_sequence`): literals with the escapes `parse_char` accepts, character
classes with ranges and `^` negation, rule references, parenthesised groups,
`|` alternation, and the repetition suffixes `* + ? {n} {n,} {n,m}`. A suffix
applies to the whole preceding literal, class, reference or group. Token rules
(`<...>`, `!<...>`) and the any-char `.` are not supported and raise.

CROSS-CHECK. `tests/python/test_gbnf_match.py` pins this reader to verdicts
recorded from llama.cpp's own `test-gbnf-validator` on the same strings.
Treat a disagreement there as a bug in this file, never in llama.cpp.
"""

from __future__ import annotations

from typing import Union

# One grammar item. Shapes:
#   ("lit", text)                       exact characters
#   ("chars", negate, ((lo, hi), ...))  one character in (or not in) the ranges
#   ("ref", name)                       another rule
#   ("group", alternatives)             parenthesised alternation
#   ("rep", item, min, max_or_None)     repetition of one item
Item = tuple  # kept loose on purpose; the shapes above are the contract
Sequence_ = list  # a sequence of items
Alternatives = list  # a list of sequences


def _is_word_char(c: str) -> bool:
    return c.isascii() and (c.isalnum() or c == "-")


class _Parser:
    """Turn GBNF text into {rule name: alternatives}. Raises ValueError on bad text."""

    def __init__(self, src: str) -> None:
        self.s = src
        self.i = 0

    def _peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def _space(self, newline_ok: bool) -> None:
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "#":
                while self.i < len(self.s) and self.s[self.i] not in "\r\n":
                    self.i += 1
            elif c in " \t" or (newline_ok and c in "\r\n"):
                self.i += 1
            else:
                break

    def _name(self) -> str:
        start = self.i
        while self.i < len(self.s) and _is_word_char(self.s[self.i]):
            self.i += 1
        if self.i == start:
            raise ValueError(f"expecting name at offset {start}")
        return self.s[start : self.i]

    def _int(self) -> int:
        start = self.i
        while self.i < len(self.s) and self.s[self.i].isdigit():
            self.i += 1
        if self.i == start:
            raise ValueError(f"expecting integer at offset {start}")
        return int(self.s[start : self.i])

    def _hex(self, size: int) -> int:
        chunk = self.s[self.i : self.i + size]
        if len(chunk) != size or any(ch not in "0123456789abcdefABCDEF" for ch in chunk):
            raise ValueError(f"expecting {size} hex chars at offset {self.i}")
        self.i += size
        return int(chunk, 16)

    def _char(self) -> int:
        """One code point, honouring the same escapes as llama.cpp's parse_char."""
        if self.i >= len(self.s):
            raise ValueError("unexpected end of input")
        c = self.s[self.i]
        if c != "\\":
            self.i += 1
            return ord(c)
        esc = self.s[self.i + 1] if self.i + 1 < len(self.s) else ""
        self.i += 2
        if esc == "x":
            return self._hex(2)
        if esc == "u":
            return self._hex(4)
        if esc == "U":
            return self._hex(8)
        simple = {"t": 9, "r": 13, "n": 10, "\\": 92, '"': 34, "[": 91, "]": 93}
        if esc in simple:
            return simple[esc]
        raise ValueError(f"unknown escape at offset {self.i - 2}")

    def parse(self) -> dict[str, Alternatives]:
        rules: dict[str, Alternatives] = {}
        self._space(True)
        while self.i < len(self.s):
            name = self._name()
            self._space(False)
            if not self.s.startswith("::=", self.i):
                raise ValueError(f"expecting ::= at offset {self.i}")
            self.i += 3
            self._space(True)
            rules[name] = self._alternates(nested=False)
            if self.i < len(self.s) and self.s[self.i] not in "\r\n":
                raise ValueError(f"expecting newline or end at offset {self.i}")
            self._space(True)
        refs = {
            item[1]
            for alts in rules.values()
            for seq in alts
            for item in _walk(seq)
            if item[0] == "ref"
        }
        missing = sorted(refs - set(rules))
        if missing:
            raise ValueError(f"undefined rule(s): {missing}")
        return rules

    def _alternates(self, nested: bool) -> Alternatives:
        alts = [self._sequence(nested)]
        while self._peek() == "|":
            self.i += 1
            self._space(True)
            alts.append(self._sequence(nested))
        return alts

    def _sequence(self, nested: bool) -> Sequence_:
        items: Sequence_ = []
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == '"':
                self.i += 1
                text = []
                while self._peek() != '"':
                    text.append(chr(self._char()))
                self.i += 1
                items.append(("lit", "".join(text)))
                self._space(nested)
            elif c == "[":
                self.i += 1
                negate = self._peek() == "^"
                if negate:
                    self.i += 1
                ranges: list[tuple[int, int]] = []
                while self._peek() != "]":
                    lo = self._char()
                    hi = lo
                    if self._peek() == "-" and self.s[self.i + 1 : self.i + 2] not in ("]", ""):
                        self.i += 1
                        hi = self._char()
                    ranges.append((lo, hi))
                self.i += 1
                items.append(("chars", negate, tuple(ranges)))
                self._space(nested)
            elif _is_word_char(c):
                items.append(("ref", self._name()))
                self._space(nested)
            elif c == "(":
                self.i += 1
                self._space(True)
                alts = self._alternates(nested=True)
                if self._peek() != ")":
                    raise ValueError(f"expecting ) at offset {self.i}")
                self.i += 1
                items.append(("group", alts))
                self._space(nested)
            elif c in "*+?":
                self.i += 1
                self._space(nested)
                lo, hi = {"*": (0, None), "+": (1, None), "?": (0, 1)}[c]
                self._wrap(items, lo, hi)
            elif c == "{":
                self.i += 1
                self._space(nested)
                lo = self._int()
                self._space(nested)
                hi: Union[int, None]
                if self._peek() == "}":
                    hi = lo
                elif self._peek() == ",":
                    self.i += 1
                    self._space(nested)
                    hi = self._int() if self._peek().isdigit() else None
                    self._space(nested)
                    if self._peek() != "}":
                        raise ValueError(f"expecting }} at offset {self.i}")
                else:
                    raise ValueError(f"expecting , or }} at offset {self.i}")
                self.i += 1
                self._space(nested)
                self._wrap(items, lo, hi)
            elif c in "<!.":
                raise NotImplementedError(
                    f"token rules and '.' are not supported (offset {self.i})"
                )
            else:
                break
        return items

    @staticmethod
    def _wrap(items: Sequence_, lo: int, hi: Union[int, None]) -> None:
        if not items:
            raise ValueError("expecting preceding item to */+/?/{")
        items[-1] = ("rep", items[-1], lo, hi)


def _walk(seq: Sequence_):  # type: ignore[no-untyped-def]
    for item in seq:
        yield item
        if item[0] == "rep":
            yield from _walk([item[1]])
        elif item[0] == "group":
            for alt in item[1]:
                yield from _walk(alt)


class _Matcher:
    """Set-of-end-positions matcher. `rule_ends(name, pos)` is every index the rule
    can finish at when it starts at `pos`. Full acceptance means len(text) is in
    the root's set from 0."""

    def __init__(self, rules: dict[str, Alternatives], text: str) -> None:
        self.rules = rules
        self.text = text
        self.memo: dict[tuple[str, int], frozenset[int]] = {}
        self.active: set[tuple[str, int]] = set()

    def rule_ends(self, name: str, pos: int) -> frozenset[int]:
        key = (name, pos)
        if key in self.memo:
            return self.memo[key]
        if key in self.active:
            # Left recursion would loop forever; our JSON grammars have none.
            return frozenset()
        self.active.add(key)
        out = self.alts_ends(self.rules[name], pos)
        self.active.discard(key)
        self.memo[key] = out
        return out

    def alts_ends(self, alts: Alternatives, pos: int) -> frozenset[int]:
        out: set[int] = set()
        for seq in alts:
            out |= self.seq_ends(seq, pos)
        return frozenset(out)

    def seq_ends(self, seq: Sequence_, pos: int) -> frozenset[int]:
        positions: frozenset[int] = frozenset({pos})
        for item in seq:
            nxt: set[int] = set()
            for p in positions:
                nxt |= self.item_ends(item, p)
            positions = frozenset(nxt)
            if not positions:
                break
        return positions

    def item_ends(self, item: Item, pos: int) -> frozenset[int]:
        kind = item[0]
        if kind == "lit":
            text = item[1]
            return frozenset({pos + len(text)}) if self.text.startswith(text, pos) else frozenset()
        if kind == "chars":
            if pos >= len(self.text):
                return frozenset()
            cp = ord(self.text[pos])
            inside = any(lo <= cp <= hi for lo, hi in item[2])
            return frozenset({pos + 1}) if inside != item[1] else frozenset()
        if kind == "ref":
            return self.rule_ends(item[1], pos)
        if kind == "group":
            return self.alts_ends(item[1], pos)
        if kind == "rep":
            return self._rep_ends(item[1], item[2], item[3], pos)
        raise ValueError(f"unknown item kind {kind!r}")

    def _rep_ends(self, inner: Item, lo: int, hi: Union[int, None], pos: int) -> frozenset[int]:
        result: set[int] = set()
        frontier: frozenset[int] = frozenset({pos})
        count = 0
        if lo == 0:
            result.add(pos)
        while frontier and (hi is None or count < hi):
            nxt: set[int] = set()
            for p in frontier:
                nxt |= self.item_ends(inner, p)
            count += 1
            if count >= lo:
                result |= nxt
            if frozenset(nxt) == frontier:
                # Fixed point: the inner item matched empty. Every later
                # iteration yields this same set, so add it and stop.
                result |= nxt
                break
            frontier = frozenset(nxt)
        return frozenset(result)


def parse_gbnf(grammar: str) -> dict[str, Alternatives]:
    """Parse GBNF text. Raises ValueError on malformed text or an undefined rule."""
    return _Parser(grammar).parse()


def accepts(grammar: str, text: str, root: str = "root") -> bool:
    """True when `text` is exactly one complete derivation of `root` in `grammar`."""
    rules = parse_gbnf(grammar)
    if root not in rules:
        raise ValueError(f"root rule {root!r} is not defined")
    return len(text) in _Matcher(rules, text).rule_ends(root, 0)
