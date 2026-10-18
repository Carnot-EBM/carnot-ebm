"""Typed, byte-replayable atoms from native tool output.

Only visible source membership or an AST definition in a visible line can be
checked. A closed Markdown fence is not authority that the whole file exists.
Spec: REQ-REPORT-7658 and SCENARIO-REPORT-7658-DIALECTS/SCOPE/MUTATIONS.
"""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Any


FENCE = re.compile(r"```[^\n]*\n(?P<body>.*?)\n```", re.DOTALL)
NUMBERED = re.compile(r"(?P<number>\d+): ?(?P<code>.*)")
SEARCH = re.compile(r"(?P<path>[^\s:]+\.[\w]+):(?P<number>\d+):(?P<code>.*)")
STACK = re.compile(
    r"\bat [^\n]*?\(?(?P<path>(?:/|[\w.-]+/)[^\s():]+\.[\w]+):(?P<number>\d+):(?P<column>\d+)\)?"
)
PATH_LINE = re.compile(
    r"`(?P<path>[^`\s]+\.[\w]+):(?P<number>\d+)(?::\d+)?`|`(?P<file>[^`\s]+\.[\w]+)`[^.]{0,65}?\b(?:at )?lines? (?P<at>\d+)(?:-\d+)?"
)
IDENT_LINE = re.compile(
    r"`(?P<name>[A-Za-z_]\w*)`[^.]{0,120}?\b(?:defined|located)[^.]{0,100}?\blines? (?P<number>\d+)(?:-\d+)?"
)
SCOPE_LINE = re.compile(
    r"\blines? (?P<number>\d+)(?:-\d+)?[^.]{0,90}?`(?P<name>[A-Za-z_]\w*)` (?:function|class|method)"
)


def digest(value: str) -> str:
    """Bind the exact UTF-8 bytes used by the reader."""
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _byte(source: str, position: int) -> int:
    return len(source[:position].encode())


def _definition_facts(lines: list[dict[str, Any]]) -> None:
    """Attach AST definition names and nested scopes to visible Python lines."""
    try:
        tree = ast.parse("\n".join(line["text"] for line in lines))
    except SyntaxError:
        return

    def visit(node: ast.AST, scope: str = "") -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{scope}.{node.name}" if scope else node.name
            lines[node.lineno - 1]["definitions"].append(name)
            for line in lines[node.lineno - 1 : node.end_lineno]:
                line["scopes"].append(name)
            scope = name
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree)


def parse_atoms(source: str) -> list[dict[str, Any]]:
    """Parse visible records and retain original byte offsets and scope."""
    match = FENCE.search(source)
    if match is None:
        return []
    body = match.group("body")
    rows = body.splitlines(keepends=True)
    starts: list[int] = []
    cursor = match.start("body")
    for row in rows:
        starts.append(cursor)
        cursor += len(row)
    stripped = [row.rstrip("\r\n") for row in rows]
    meaningful = [row for row in stripped if row.strip()]
    if not meaningful:
        return []
    if all(NUMBERED.fullmatch(row) for row in meaningful):
        dialect = "numbered"
    elif all(SEARCH.fullmatch(row) for row in meaningful):
        dialect = "grep"
    elif any(STACK.search(row) for row in meaningful):
        dialect = "stack"
    else:
        try:
            ast.parse(body)
        except SyntaxError:
            return []
        dialect = "plain"
    atoms: list[dict[str, Any]] = []
    identity = f"anonymous:{digest(body)}"
    for raw, start in zip(stripped, starts, strict=True):
        if not raw.strip():
            continue
        record = NUMBERED.fullmatch(raw) if dialect == "numbered" else None
        record = SEARCH.fullmatch(raw) if dialect == "grep" else record
        record = STACK.search(raw) if dialect == "stack" else record
        if dialect == "stack" and record is None:
            continue
        position = record.start("code") if record and "code" in record.groupdict() else 0
        value = record.group("code") if record and "code" in record.groupdict() else raw
        number = (
            int(record.group("number"))
            if record and "number" in record.groupdict()
            else len(atoms) + 1
        )
        path = record.group("path") if record and "path" in record.groupdict() else identity
        atoms.append(
            {
                "dialect": dialect,
                "source_id": path,
                "line": number,
                "byte_start": _byte(source, start + position),
                "byte_end": _byte(source, start + position + len(value)),
                "text": value,
                "complete": dialect in {"plain", "numbered"},
                "definitions": [],
                "scopes": [],
            }
        )
    if dialect in {"plain", "numbered"}:
        _definition_facts(atoms)
    return atoms


def extract_claims(answer: str) -> list[dict[str, Any]]:
    """Keep only explicit original-answer path/line or definition spans."""
    candidates: list[dict[str, Any]] = []
    for match in PATH_LINE.finditer(answer):
        candidates.append(
            {
                "kind": "path_line",
                "source_id": match.group("path") or match.group("file"),
                "line": int(match.group("number") or match.group("at")),
                "byte_start": _byte(answer, match.start()),
                "byte_end": _byte(answer, match.end()),
                "text": match.group(),
            }
        )
    for match in IDENT_LINE.finditer(answer):
        if re.search(r"\b(?:not|never|didn't)\b", match.group(), re.IGNORECASE):
            continue
        candidates.append(
            {
                "kind": "definition",
                "name": match.group("name"),
                "line": int(match.group("number")),
                "byte_start": _byte(answer, match.start()),
                "byte_end": _byte(answer, match.end()),
                "text": match.group(),
            }
        )
    for match in SCOPE_LINE.finditer(answer):
        candidates.append(
            {
                "kind": "scope_line",
                "name": match.group("name"),
                "line": int(match.group("number")),
                "byte_start": _byte(answer, match.start()),
                "byte_end": _byte(answer, match.end()),
                "text": match.group(),
            }
        )
    return sorted(candidates, key=lambda claim: (claim["byte_start"], claim["byte_end"]))


def verify_answer(source: str, answer: str) -> dict[str, Any]:
    """Check typed propositions while leaving all other prose unverified."""
    atoms = parse_atoms(source)
    claims = extract_claims(answer)
    witnesses: list[dict[str, Any]] = []
    for claim in claims:
        matching = [atom for atom in atoms if atom["line"] == claim["line"]]
        if claim["kind"] == "path_line":
            matching = [atom for atom in matching if atom["source_id"] == claim["source_id"]]
            status = "observed" if matching else "unknown"
        elif claim["kind"] == "scope_line":
            scoped = [
                atom
                for atom in matching
                if any(name.split(".")[-1] == claim["name"] for name in atom["scopes"])
            ]
            status = "observed" if scoped else "unknown"
            matching = scoped
        else:
            defining = [
                atom
                for atom in matching
                if any(name.split(".")[-1] == claim["name"] for name in atom["definitions"])
            ]
            status = (
                "observed"
                if defining
                else "scoped_contradiction"
                if matching and all(atom["complete"] for atom in matching)
                else "unknown"
            )
            matching = defining or matching if status != "unknown" else []
        witness = dict(claim)
        witness["status"] = status
        witness["evidence_span"] = (
            [matching[0]["byte_start"], matching[0]["byte_end"]] if matching else None
        )
        witness["completeness"] = matching[0]["complete"] if matching else False
        witnesses.append(witness)
    statuses = {row["status"] for row in witnesses}
    status = (
        "observed"
        if "observed" in statuses
        else ("scoped_contradiction" if "scoped_contradiction" in statuses else "unknown")
    )
    return {
        "status": status,
        "witnesses": witnesses,
        "checked_propositions": sum(row["status"] != "unknown" for row in witnesses),
        "residual_unverified_text": answer,
        "whole_answer_certified": False,
    }


def replay(
    source: str,
    answer: str,
    source_sha256: str,
    answer_sha256: str,
    atoms: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    evaluator_sidecar: Any = None,
) -> None:
    """Independently recompute every byte span and reject label access."""
    if evaluator_sidecar is not None:
        raise ValueError("evaluator_sidecar_access")
    if digest(source) != source_sha256:
        raise ValueError("source_hash_mismatch")
    if digest(answer) != answer_sha256:
        raise ValueError("answer_hash_mismatch")
    encoded = source.encode()
    for atom in atoms:
        start, end = atom["byte_start"], atom["byte_end"]
        if not 0 <= start <= end <= len(encoded) or encoded[start:end] != atom["text"].encode():
            raise ValueError("source_offset_mismatch")
    if atoms != parse_atoms(source):
        raise ValueError("atom_replay_mismatch")
    if result != verify_answer(source, answer):
        raise ValueError("claim_replay_mismatch")
