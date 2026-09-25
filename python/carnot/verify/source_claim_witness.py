"""Conservative witnesses for explicit claims about numbered Python source.

The input is treated as untrusted text. Python's parser inspects it; nothing in
the supplied source is imported, evaluated or executed. A witness concerns one
small proposition and never certifies the surrounding prose.

Spec: REQ-REPORT-7644, SCENARIO-REPORT-7644-STRUCTURE/ABSTAIN.
"""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Any


SCHEMA_VERSION = "carnot.source_claim_witness.v1"
FEATURE_NAMES = ("symbol_defined", "definition_line", "call_line", "raise_line", "literal_line")
_OPEN = re.compile(r"^```(?:python|py) file=([^\s`]+)\s*$")
_NUMBERED = re.compile(r"^(\d+) \| ?(.*)$")
_CLAIM = re.compile(
    r"^In `(?P<file>[^`]+)`, `(?P<symbol>[^`]+)` "
    r"(?P<verb>exists|is defined at line (?P<defined>\d+)"
    r"|is called at line (?P<called>\d+)"
    r"|is raised at line (?P<raised>\d+)"
    r"|appears at line (?P<literal>\d+))(?P<tail>.*)$"
)
_QUALIFIED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z")


def _byte_offset(source: str, character_offset: int) -> int:
    return len(source[:character_offset].encode("utf-8"))


def parse_numbered_blocks(source: str) -> list[dict[str, Any]]:
    """Return code and exact original UTF-8 spans for closed numbered blocks.

    A malformed line makes its block incomplete. A closing fence is required;
    omitted text is never silently treated as an empty or complete file.
    """

    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    cursor = 0
    for raw in source.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        if current is None:
            opening = _OPEN.fullmatch(line)
            if opening:
                current = {"file": opening.group(1), "lines": [], "complete": True, "closed": False}
        elif line == "```":
            current["closed"] = True
            blocks.append(current)
            current = None
        else:
            numbered = _NUMBERED.fullmatch(line)
            if numbered is None:
                current["complete"] = False
            else:
                number = int(numbered.group(1))
                code = numbered.group(2)
                if number != len(current["lines"]) + 1:
                    current["complete"] = False
                start = _byte_offset(source, cursor + numbered.start(2))
                current["lines"].append(
                    {
                        "number": number,
                        "code": code,
                        "byte_start": start,
                        "exact_bytes": code.encode().hex(),
                    }
                )
        cursor += len(raw)
    if current is not None:
        current["complete"] = False
        blocks.append(current)
    for block in blocks:
        if not block["lines"]:
            block["complete"] = False
            continue
        try:
            block["tree"] = ast.parse("\n".join(line["code"] for line in block["lines"]))
        except SyntaxError:
            block["complete"] = False
    return blocks


def _path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _path(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _facts(block: dict[str, Any]) -> dict[str, list[tuple[str, int, int]]]:
    """Index definitions, explicit calls and raises without resolving aliases."""

    found: dict[str, list[tuple[str, int, int]]] = {"defined": [], "called": [], "raised": []}
    tree = block["tree"]

    def walk(node: ast.AST, scope: str = "") -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol = f"{scope}.{node.name}" if scope else node.name
            found["defined"].append((symbol, node.lineno, node.col_offset))
            scope = symbol
        if isinstance(node, ast.Call):
            name = _path(node.func)
            if name:
                found["called"].append((name, node.lineno, node.col_offset))
        if isinstance(node, ast.Raise) and node.exc is not None:
            name = _path(node.exc.func if isinstance(node.exc, ast.Call) else node.exc)
            if name:
                found["raised"].append((name, node.lineno, node.col_offset))
        for child in ast.iter_child_nodes(node):
            walk(child, scope)

    walk(tree)
    return found


def verify_claim(
    source: str, claim: str, *, closed_files: list[str] | tuple[str, ...] = ()
) -> dict[str, Any]:
    """Check one explicit proposition and keep unsupported language visible.

    ``closed_files`` is caller supplied authority that these named files are
    complete. The verifier cannot infer a closed world from a Markdown fence.
    Only complete, parseable, single-block files can yield contradiction.
    """

    digest = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    result: dict[str, Any] = {
        "status": "unknown",
        "proposition_checked": None,
        "source_offset": None,
        "source_sha256": digest,
        "reason": "claim_outside_structural_grammar",
        "parser_completeness": "missing",
        "residual_unverified_span": bool(claim.strip()),
        "schema_version": SCHEMA_VERSION,
    }
    match = _CLAIM.match(claim)
    if match is None or not match.group("tail") or match.group("tail")[0] not in ".,":
        return result
    symbol = match.group("symbol")
    kind = (
        "literal"
        if match.group("literal")
        else next((name for name in ("defined", "called", "raised") if match.group(name)), "exists")
    )
    if kind != "literal" and not _QUALIFIED.fullmatch(symbol):
        return result
    if kind == "literal" and not (symbol.startswith('"') and symbol.endswith('"')):
        return result
    file = match.group("file")
    proposition_end = match.end("verb") + 1
    result["proposition_checked"] = claim[:proposition_end]
    result["residual_unverified_span"] = bool(claim[proposition_end:].strip())
    blocks = [block for block in parse_numbered_blocks(source) if block["file"] == file]
    if not blocks:
        result["reason"] = "file_not_supplied"
        return result
    complete = len(blocks) == 1 and blocks[0]["complete"] and blocks[0]["closed"]
    result["parser_completeness"] = "complete" if complete else "incomplete"
    if not complete:
        result["reason"] = "source_incomplete_or_ambiguous"
        return result
    block = blocks[0]
    tree: ast.AST = block["tree"]
    aliases = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        for name in (alias.name.split(".")[0], alias.asname or alias.name.split(".")[0])
    }
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    if symbol.split(".")[0] in aliases or symbol in assigned:
        result["reason"] = "alias_or_assignment_unresolved"
        return result
    facts = _facts(block)
    if kind == "literal":
        code_literal = symbol[1:-1]
        candidates = [
            (
                line["number"],
                line["byte_start"] + line["code"].encode().index(code_literal.encode()),
            )
            for line in block["lines"]
            if code_literal.encode() in line["code"].encode()
        ]
    else:
        rows = facts["defined" if kind == "exists" else kind]
        candidates = [
            (line, block["lines"][line - 1]["byte_start"] + column)
            for name, line, column in rows
            if name == symbol
        ]
    target_line = (
        None if kind == "exists" else int(match.group(kind if kind != "literal" else "literal"))
    )
    hits = [offset for line, offset in candidates if target_line is None or line == target_line]
    if hits:
        result.update(
            status="supported", source_offset=hits[0], reason="exact_ast_or_literal_match"
        )
    elif file in closed_files:
        result.update(status="contradicted", reason="absent_in_complete_closed_scope")
        if candidates:
            result["source_offset"] = candidates[0][1]
    else:
        result["reason"] = "open_scope_cannot_prove_absence"
    return result
