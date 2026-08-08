"""Transition object-delta perception for ARC world-model induction.

Spec refs: REQ-ARC-WMTE-6213,
SCENARIO-ARC-WMTE-6213-TRANSLATION,
SCENARIO-ARC-WMTE-6213-HUD-REJECTION,
SCENARIO-ARC-WMTE-6213-FAIL-OPEN,
SCENARIO-ARC-WMTE-6213-PROMPT-WIRING.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from typing import Any

import numpy as np


SCHEMA_VERSION = "carnot.arc.object_delta_perception.v1"
FLAG_NAME = "CARNOT_ARC_OBJECT_DELTA_PERCEPTION"
_FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def object_delta_perception_on(environ: Mapping[str, str] | None = None) -> bool:
    """Return true only for the explicit treatment arm."""

    env = os.environ if environ is None else environ
    return str(env.get(FLAG_NAME, "0")).strip() == "1"


def component_schema() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION + ".component",
        "fields": {
            "id": "stable row-local id after deterministic component ordering",
            "color": "integer cell value",
            "area": "4-connected component cell count",
            "bbox": "[row0, col0, row1, col1]",
            "centroid": "[row, col] rounded to 3 decimals",
            "shape_id": "translation-invariant hash of normalized cells",
            "identity_signature": "color-aware shape identity used for matching",
            "normalized_cell_sample": "first 16 normalized cells for inspection",
        },
    }


def transition_delta_schema() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION + ".transition_delta",
        "fields": {
            "action": "chosen visible action id",
            "data": "chosen visible action data, JSON-normalized",
            "before_components": "components from the before grid after HUD rejection",
            "after_components": "components from the after grid after HUD rejection",
            "matches": "unique translation-invariant before/after identity matches",
            "relations": "pairwise relative-centroid deltas among matched components",
            "ambiguous_matches": "fail-open receipts for non-unique identities",
            "created_after_components": "after components with no admitted match",
            "removed_before_components": "before components with no admitted match",
        },
    }


def _grid(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D visible grid, got shape {arr.shape}")
    return arr


def _round3(value: float) -> float:
    return round(float(value), 3)


def _sha16(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _shape_id(cells: Sequence[tuple[int, int]]) -> str:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = tuple(sorted((int(r) - min_r, int(c) - min_c) for r, c in cells))
    return _sha16(normalized)[:12]


def _component_fully_masked(
    cells: Sequence[tuple[int, int]],
    mask: np.ndarray | None,
) -> bool:
    if mask is None:
        return False
    return all(bool(mask[r, c]) for r, c in cells)


def _mask_or_none(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    return arr if arr.shape == shape and bool(arr.any()) else None


def _extract_components_with_receipt(
    value: Any,
    *,
    hud_mask: Any = None,
    prefix: str = "o",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grid = _grid(value)
    height, width = grid.shape
    mask = _mask_or_none(hud_mask, (height, width))
    visited = np.zeros((height, width), dtype=bool)
    rows: list[dict[str, Any]] = []
    rejected = 0
    for row in range(height):
        for col in range(width):
            if visited[row, col]:
                continue
            color = int(grid[row, col])
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((int(r), int(c)))
                for dr, dc in _FOUR_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            if _component_fully_masked(cells, mask):
                rejected += 1
                continue
            ys = [r for r, _ in cells]
            xs = [c for _, c in cells]
            min_r, max_r = min(ys), max(ys)
            min_c, max_c = min(xs), max(xs)
            normalized = sorted((int(r) - min_r, int(c) - min_c) for r, c in cells)
            shape_id = _shape_id(cells)
            identity = _sha16({"color": color, "area": len(cells), "shape_id": shape_id})
            rows.append(
                {
                    "id": "",
                    "color": color,
                    "area": int(len(cells)),
                    "bbox": [int(min_r), int(min_c), int(max_r), int(max_c)],
                    "centroid": [_round3(float(np.mean(ys))), _round3(float(np.mean(xs)))],
                    "shape_id": shape_id,
                    "identity_signature": identity,
                    "normalized_cell_sample": [list(cell) for cell in normalized[:16]],
                }
            )
    rows.sort(
        key=lambda item: (
            item["bbox"][0],
            item["bbox"][1],
            item["bbox"][2],
            item["bbox"][3],
            item["color"],
            item["area"],
            item["shape_id"],
        )
    )
    for index, item in enumerate(rows):
        item["id"] = f"{prefix}{index}"
    return rows, {"rejected_hud_components": rejected}


def extract_components(value: Any, *, hud_mask: Any = None) -> list[dict[str, Any]]:
    """Return 4-connected components from one agent-visible grid."""

    rows, _receipt = _extract_components_with_receipt(value, hud_mask=hud_mask)
    return rows


def _edge_mask(shape: tuple[int, int], edge: str, thickness: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if edge == "top":
        mask[:thickness, :] = True
    elif edge == "bottom":
        mask[height - thickness :, :] = True
    elif edge == "left":
        mask[:, :thickness] = True
    elif edge == "right":
        mask[:, width - thickness :] = True
    return mask


def _dominant_fraction(values: np.ndarray) -> float:
    flat = [int(value) for value in np.asarray(values).flatten().tolist()]
    if not flat:
        return 0.0
    return float(max(Counter(flat).values())) / float(len(flat))


def _hud_candidates(shape: tuple[int, int]) -> list[tuple[str, int, np.ndarray]]:
    height, width = shape
    max_thickness = max(1, min(2, min(height, width) // 8))
    out: list[tuple[str, int, np.ndarray]] = []
    for thickness in range(1, max_thickness + 1):
        for edge in ("top", "bottom", "left", "right"):
            out.append((edge, thickness, _edge_mask(shape, edge, thickness)))
    return out


def hud_rejection_rules() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION + ".hud_rules",
        "edge_band_max_thickness": "min(2, min(height, width)//8), at least 1",
        "dominant_strip_color_fraction_min": 0.75,
        "admission_requires": [
            "all transition grids have one 2-D shape",
            "candidate is a thin frame-edge strip",
            "some before/after strip has a dominant color covering at least 75%",
            "at least one observed transition changes cells only inside that strip",
        ],
        "rejection_scope": "only components wholly inside the admitted strip are removed",
        "fail_open": "no strip is admitted when these conditions are not all met",
    }


def admitted_hud_strip(transitions: Sequence[Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Infer a conservative HUD strip from visible before/after grids only."""

    rows = list(transitions)
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION + ".hud_receipt",
        "admitted": False,
        "reason": "no_transitions",
        "edge": None,
        "thickness": 0,
        "dominant_color_fraction": 0.0,
        "strip_only_motion_transitions": 0,
        "mixed_motion_transitions": 0,
        "candidate_count": 0,
    }
    if not rows:
        return None, receipt
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        first = _grid(rows[0].grid)
        shape = first.shape
        for transition in rows:
            before = _grid(transition.grid)
            after = _grid(transition.next_grid)
            if before.shape != shape or after.shape != shape:
                receipt["reason"] = "shape_mismatch"
                return None, receipt
            pairs.append((before, after))
    except Exception:
        receipt["reason"] = "malformed_transition_grid"
        return None, receipt
    best: tuple[int, float, int, str, np.ndarray, int] | None = None
    candidates = _hud_candidates(shape)
    receipt["candidate_count"] = len(candidates)
    for edge_index, (edge, thickness, mask) in enumerate(candidates):
        dominant = max(_dominant_fraction(grid[mask]) for pair in pairs for grid in pair)
        if dominant < 0.75:
            continue
        strip_only = mixed = 0
        for before, after in pairs:
            diff = before != after
            if not bool(diff.any()):
                continue
            inside = int((diff & mask).sum())
            outside = int((diff & ~mask).sum())
            if inside > 0 and outside == 0:
                strip_only += 1
            elif inside > 0 and outside > 0:
                mixed += 1
        if strip_only == 0:
            continue
        rank = (strip_only, dominant, -thickness, -edge_index)
        if best is None or rank > best[:4]:
            best = (strip_only, dominant, -thickness, -edge_index, mask, mixed)
            receipt.update(
                {
                    "edge": edge,
                    "thickness": int(thickness),
                    "dominant_color_fraction": _round3(dominant),
                    "strip_only_motion_transitions": int(strip_only),
                    "mixed_motion_transitions": int(mixed),
                }
            )
    if best is None:
        receipt["reason"] = "no_admitted_strip"
        return None, receipt
    receipt["admitted"] = True
    receipt["reason"] = "strip_only_motion_with_broad_edge_band"
    return best[4], receipt


def _canonical_data(value: Any) -> Any:
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except TypeError:
        return str(value)


def _match_components(
    before: Sequence[dict[str, Any]],
    after: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    before_by_sig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    after_by_sig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in before:
        before_by_sig[str(item["identity_signature"])].append(dict(item))
    for item in after:
        after_by_sig[str(item["identity_signature"])].append(dict(item))
    matched_before: set[str] = set()
    matched_after: set[str] = set()
    matches: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for signature in sorted(set(before_by_sig) | set(after_by_sig)):
        before_rows = sorted(before_by_sig.get(signature, []), key=lambda item: item["id"])
        after_rows = sorted(after_by_sig.get(signature, []), key=lambda item: item["id"])
        if len(before_rows) == 1 and len(after_rows) == 1:
            b = before_rows[0]
            a = after_rows[0]
            matched_before.add(str(b["id"]))
            matched_after.add(str(a["id"]))
            dy = _round3(float(a["centroid"][0]) - float(b["centroid"][0]))
            dx = _round3(float(a["centroid"][1]) - float(b["centroid"][1]))
            matches.append(
                {
                    "identity_signature": signature,
                    "before_id": b["id"],
                    "after_id": a["id"],
                    "before_component": b,
                    "after_component": a,
                    "centroid_delta": [dy, dx],
                    "delta_kind": "stable" if dy == 0.0 and dx == 0.0 else "translated",
                }
            )
        elif before_rows and after_rows:
            ambiguous.append(
                {
                    "identity_signature": signature,
                    "before_ids": [row["id"] for row in before_rows],
                    "after_ids": [row["id"] for row in after_rows],
                    "before_count": len(before_rows),
                    "after_count": len(after_rows),
                    "fail_open": True,
                    "reason": "ambiguous_identity_signature",
                }
            )
    removed = [dict(row) for row in before if str(row["id"]) not in matched_before]
    created = [dict(row) for row in after if str(row["id"]) not in matched_after]
    return {
        "matches": sorted(matches, key=lambda item: (item["before_id"], item["after_id"])),
        "ambiguous_matches": ambiguous,
        "removed_before_components": removed,
        "created_after_components": created,
    }


def _relations(matches: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(matches, key=lambda item: str(item["before_id"]))
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            lb = left["before_component"]
            rb = right["before_component"]
            la = left["after_component"]
            ra = right["after_component"]
            before_delta = [
                _round3(float(rb["centroid"][0]) - float(lb["centroid"][0])),
                _round3(float(rb["centroid"][1]) - float(lb["centroid"][1])),
            ]
            after_delta = [
                _round3(float(ra["centroid"][0]) - float(la["centroid"][0])),
                _round3(float(ra["centroid"][1]) - float(la["centroid"][1])),
            ]
            relation_delta = [
                _round3(after_delta[0] - before_delta[0]),
                _round3(after_delta[1] - before_delta[1]),
            ]
            rows.append(
                {
                    "before_pair": [left["before_id"], right["before_id"]],
                    "after_pair": [left["after_id"], right["after_id"]],
                    "before_pair_colors": [
                        int(lb["color"]),
                        int(rb["color"]),
                    ],
                    "before_delta": before_delta,
                    "after_delta": after_delta,
                    "relation_delta": relation_delta,
                    "relation_invariant": relation_delta == [0.0, 0.0],
                }
            )
    return rows


def build_object_delta_table(
    transitions: Sequence[Any],
    *,
    max_transitions: int | None = None,
) -> dict[str, Any]:
    """Build the versioned table from visible consecutive grids and actions."""

    rows = list(transitions)
    if max_transitions is not None:
        rows = rows[: max(0, int(max_transitions))]
    hud_mask, hud_receipt = admitted_hud_strip(rows)
    transition_rows: list[dict[str, Any]] = []
    for index, transition in enumerate(rows):
        before_grid = _grid(transition.grid)
        after_grid = _grid(transition.next_grid)
        before, before_receipt = _extract_components_with_receipt(
            before_grid,
            hud_mask=hud_mask,
            prefix="b",
        )
        after, after_receipt = _extract_components_with_receipt(
            after_grid,
            hud_mask=hud_mask,
            prefix="a",
        )
        match_receipt = _match_components(before, after)
        matches = match_receipt["matches"]
        changed = (
            int((before_grid != after_grid).sum()) if before_grid.shape == after_grid.shape else 0
        )
        transition_rows.append(
            {
                "schema": SCHEMA_VERSION + ".transition",
                "index": int(index),
                "action": int(transition.action),
                "data": _canonical_data(getattr(transition, "data", None)),
                "changed_cell_count": changed,
                "hud_rejection": dict(hud_receipt),
                "hud_rejected_component_counts": {
                    "before": int(before_receipt["rejected_hud_components"]),
                    "after": int(after_receipt["rejected_hud_components"]),
                },
                "before_components": before,
                "after_components": after,
                "matches": matches,
                "relations": _relations(matches),
                "ambiguous_matches": match_receipt["ambiguous_matches"],
                "removed_before_components": match_receipt["removed_before_components"],
                "created_after_components": match_receipt["created_after_components"],
            }
        )
    return {
        "schema": SCHEMA_VERSION,
        "component_schema": component_schema(),
        "transition_delta_schema": transition_delta_schema(),
        "hud_rejection_rules": hud_rejection_rules(),
        "hud_rejection": dict(hud_receipt),
        "transition_count": len(transition_rows),
        "transitions": transition_rows,
        "forbidden_access_counts": forbidden_access_counts(),
    }


def object_delta_table_json(table: Mapping[str, Any]) -> str:
    return json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _component_line(component: Mapping[str, Any]) -> str:
    return (
        f"{component['id']}:c={component['color']} area={component['area']} "
        f"bbox={component['bbox']} centroid={component['centroid']} "
        f"shape={component['shape_id']} sig={component['identity_signature']}"
    )


def serialize_object_delta_table(table: Mapping[str, Any]) -> str:
    lines = [
        "OBJECT DELTA PERCEPTION "
        f"({SCHEMA_VERSION}; visible before/after grids and chosen actions only)",
        f"HUD rejection: {json.dumps(table.get('hud_rejection', {}), sort_keys=True)}",
    ]
    for row in table.get("transitions", []):
        lines.append(
            f"--- TRANSITION {row['index']} ACTION{row['action']} data="
            f"{json.dumps(row['data'], sort_keys=True)} changed_cells={row['changed_cell_count']}"
        )
        lines.append("  before components:")
        for component in row["before_components"]:
            lines.append("    " + _component_line(component))
        lines.append("  after components:")
        for component in row["after_components"]:
            lines.append("    " + _component_line(component))
        lines.append("  matches:")
        for match in row["matches"]:
            lines.append(
                f"    {match['before_id']}->{match['after_id']} "
                f"delta={match['centroid_delta']} kind={match['delta_kind']} "
                f"sig={match['identity_signature']}"
            )
        lines.append(
            "  ambiguous_fail_open: " + json.dumps(row["ambiguous_matches"], sort_keys=True)
        )
        lines.append(
            "  created_after_ids: "
            + json.dumps([item["id"] for item in row["created_after_components"]])
        )
        lines.append(
            "  removed_before_ids: "
            + json.dumps([item["id"] for item in row["removed_before_components"]])
        )
        lines.append("  relations: " + json.dumps(row["relations"], sort_keys=True))
    if not table.get("transitions"):
        lines.append("  (no transitions)")
    return "\n".join(lines)


def object_delta_block(transitions: Sequence[Any]) -> str:
    """Serialize the prompt block, returning empty text on any failure."""

    try:
        return serialize_object_delta_table(build_object_delta_table(transitions))
    except Exception:
        return ""


def forbidden_access_counts() -> dict[str, int]:
    return {
        "game_source_access_count": 0,
        "bfs_access_count": 0,
        "adapter_access_count": 0,
        "registry_trajectory_access_count": 0,
        "hidden_state_access_count": 0,
    }
