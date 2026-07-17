#!/usr/bin/env python3
"""Development-proxy planner for the public lf52 levels 8--10.

The level layouts and mechanics in this file were recovered from the permitted
PUBLIC source ``environment_files/lf52/271a04aa/lf52.py``.  A plan emitted here
is never treated as a solve by itself: it must be translated into and verified
through ordinary offline ``env.step`` calls.
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from dataclasses import dataclass
from itertools import count
from typing import Literal, NamedTuple

Point = tuple[int, int]
Payload = Literal["E", "T", "P", "B"]
Carrier = tuple[int, int, Payload]


LEVEL_GRIDS: dict[int, list[str]] = {
    8: [
        "       ",
        "",
        " ........",
        " xp...p..",
        " ......p.",
        "<-p...p..",
        "|...b....",
        "|...b...x",
        "|       |",
        "L-,>   <3",
        "   |   ;",
        " ......bb",
        " L--P,P-3",
    ],
    9: [
        "       ",
        "           x..p.p......",
        "  ..b..    .........bb.",
        "  ...b.    .p.....p.p..",
        "  .....              ..",
        "  ....,--------------..",
        "  xb..   ",
        "  .b..x  ",
    ],
    10: [
        "   .x. ",
        "<-T-T-T->",
        "| ; ; ; |",
        "| L-t-3 |",
        "|       |",
        "| ...bb |",
        "L--b... |",
        "  ..b.. |",
        "  b.... |",
        "  ....x 7",
        "        7",
        "        7",
        "        7",
        "        7",
    ],
}

RAIL_CHARS = frozenset("-|L3<>Tt,;?PD7")
CARRIER_PAYLOAD: dict[str, Payload] = {
    ",": "E",
    ";": "E",
    "?": "E",
    "P": "T",
    "D": "T",
    "7": "B",
}
DIRECTIONS: tuple[Point, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
CARRIER_ACTIONS: tuple[tuple[int, Point], ...] = (
    (1, (0, -1)),
    (2, (0, 1)),
    (3, (-1, 0)),
    (4, (1, 0)),
)


class State(NamedTuple):
    carriers: tuple[Carrier, ...]
    plains: tuple[Point, ...]
    blues: tuple[Point, ...]


@dataclass(frozen=True)
class Layout:
    level: int
    rails: frozenset[Point]
    sockets: frozenset[Point]
    static_torches: frozenset[Point]
    initial: State


def canonical(
    carriers: list[Carrier] | tuple[Carrier, ...],
    plains: set[Point] | list[Point] | tuple[Point, ...],
    blues: set[Point] | list[Point] | tuple[Point, ...],
) -> State:
    return State(tuple(sorted(carriers)), tuple(sorted(plains)), tuple(sorted(blues)))


def parse_layout(level: int) -> Layout:
    rails: set[Point] = set()
    sockets: set[Point] = set()
    torches: set[Point] = set()
    carriers: list[Carrier] = []
    plains: set[Point] = set()
    blues: set[Point] = set()
    for y, row in enumerate(LEVEL_GRIDS[level]):
        for x, char in enumerate(row):
            point = (x, y)
            if char in RAIL_CHARS:
                rails.add(point)
            if char in ".xbp":
                sockets.add(point)
            if char == "p":
                torches.add(point)
            elif char == "x":
                plains.add(point)
            elif char == "b":
                blues.add(point)
            if char in CARRIER_PAYLOAD:
                payload = CARRIER_PAYLOAD[char]
                carriers.append((x, y, payload))
    return Layout(
        level=level,
        rails=frozenset(rails),
        sockets=frozenset(sockets),
        static_torches=frozenset(torches),
        initial=canonical(carriers, plains, blues),
    )


def carrier_move(layout: Layout, state: State, delta: Point) -> State:
    dx, dy = delta
    carriers = list(state.carriers)
    order = list(range(len(carriers)))
    if dx:
        order.sort(key=lambda i: carriers[i][0], reverse=dx > 0)
    else:
        order.sort(key=lambda i: carriers[i][1], reverse=dy > 0)
    occupied = {(x, y) for x, y, _ in carriers}
    for index in order:
        x, y, payload = carriers[index]
        destination = (x + dx, y + dy)
        if destination in occupied or destination not in layout.rails:
            continue
        occupied.remove((x, y))
        occupied.add(destination)
        carriers[index] = (destination[0], destination[1], payload)
    return canonical(carriers, state.plains, state.blues)


def carrier_payload_at(state: State, point: Point) -> tuple[int, Payload] | None:
    for index, (x, y, payload) in enumerate(state.carriers):
        if (x, y) == point:
            return index, payload
    return None


def pieces(state: State) -> list[tuple[Literal["P", "B"], Point, int | None]]:
    result: list[tuple[Literal["P", "B"], Point, int | None]] = []
    result.extend(("P", point, None) for point in state.plains)
    result.extend(("B", point, None) for point in state.blues)
    for index, (x, y, payload) in enumerate(state.carriers):
        if payload in ("P", "B"):
            result.append((payload, (x, y), index))
    return result


def occupant(layout: Layout, state: State, point: Point) -> tuple[str, int | None] | None:
    if point in state.plains:
        return "P", None
    if point in state.blues:
        return "B", None
    if point in layout.static_torches:
        return "T", None
    carrier = carrier_payload_at(state, point)
    if carrier is not None and carrier[1] != "E":
        return carrier[1], carrier[0]
    return None


def landing_kind(layout: Layout, state: State, point: Point) -> tuple[str, int | None] | None:
    if occupant(layout, state, point) is not None:
        return None
    carrier = carrier_payload_at(state, point)
    if carrier is not None and carrier[1] == "E":
        return "carrier", carrier[0]
    if point in layout.sockets:
        return "socket", None
    return None


def legal_jumps(layout: Layout, state: State) -> list[tuple[Point, Point, State]]:
    successors: list[tuple[Point, Point, State]] = []
    for kind, source, source_carrier in pieces(state):
        for dx, dy in DIRECTIONS:
            crossed = (source[0] + dx, source[1] + dy)
            destination = (source[0] + 2 * dx, source[1] + 2 * dy)
            crossed_occupant = occupant(layout, state, crossed)
            destination_kind = landing_kind(layout, state, destination)
            if crossed_occupant is None or destination_kind is None:
                continue

            carriers = list(state.carriers)
            plains = set(state.plains)
            blues = set(state.blues)
            if source_carrier is None:
                (plains if kind == "P" else blues).remove(source)
            else:
                x, y, _ = carriers[source_carrier]
                carriers[source_carrier] = (x, y, "E")

            crossed_kind, crossed_carrier = crossed_occupant
            if kind == "P" and crossed_kind == "P":
                if crossed_carrier is None:
                    plains.remove(crossed)
                else:
                    x, y, _ = carriers[crossed_carrier]
                    carriers[crossed_carrier] = (x, y, "E")

            landing_type, landing_carrier = destination_kind
            if landing_type == "socket":
                (plains if kind == "P" else blues).add(destination)
            else:
                assert landing_carrier is not None
                x, y, payload = carriers[landing_carrier]
                assert payload == "E"
                carriers[landing_carrier] = (x, y, kind)
            successors.append((source, destination, canonical(carriers, plains, blues)))
    return successors


def plain_positions(state: State) -> tuple[Point, ...]:
    result = list(state.plains)
    result.extend((x, y) for x, y, payload in state.carriers if payload == "P")
    return tuple(sorted(result))


def blue_positions(state: State) -> tuple[Point, ...]:
    result = list(state.blues)
    result.extend((x, y) for x, y, payload in state.carriers if payload == "B")
    return tuple(sorted(result))


def heuristic(layout: Layout, state: State) -> float:
    plains = plain_positions(state)
    if len(plains) <= 1:
        return -10_000.0
    if layout.level == 10 and len(plains) == 2:
        # The lower socket chamber touches the carrier rail only at (2,6).
        # A green unloaded rightward over a blue at (3,6) lands at (4,6),
        # immediately below a second green staged at (4,5); that second green
        # can then remove it by jumping down into the empty (4,7) socket.
        available_jumps = legal_jumps(layout, state)
        if any(len(plain_positions(successor)) == 1 for _, _, successor in available_jumps):
            return -9_000.0
        free_plain_distance = min(
            (abs(point[0] - 4) + abs(point[1] - 5) for point in state.plains),
            default=20,
        )
        loaded_plain_positions = [
            (x, y) for x, y, payload in state.carriers if payload == "P"
        ]
        loaded_plain_distance = min(
            (abs(point[0] - 2) + abs(point[1] - 6) for point in loaded_plain_positions),
            default=15,
        )
        relay_distance = min(
            abs(point[0] - 3) + abs(point[1] - 6) for point in blue_positions(state)
        )
        blockers = sum(occupant(layout, state, point) is not None for point in ((4, 6), (4, 7)))
        mobility = len(available_jumps)
        return (
            18.0 * free_plain_distance
            + 12.0 * loaded_plain_distance
            + 5.0 * relay_distance
            + 8.0 * blockers
            - 0.75 * mobility
        )
    pair_distance = min(
        abs(a[0] - b[0]) + abs(a[1] - b[1])
        for index, a in enumerate(plains)
        for b in plains[index + 1 :]
    )
    mobility = len(legal_jumps(layout, state))
    # Weighted best-first: prioritize bringing the two removable pegs together,
    # while preferring states with more oracle-visible relay options.
    return 8.0 * pair_distance - 0.75 * mobility


def solve(level: int, state_limit: int, time_limit_s: float) -> dict[str, object]:
    layout = parse_layout(level)
    start = layout.initial
    serial = count()
    frontier: list[tuple[float, int, int, State]] = [(heuristic(layout, start), 0, next(serial), start)]
    best_cost: dict[State, int] = {start: 0}
    parent: dict[State, tuple[State, tuple[str, object]]] = {}
    expanded = 0
    generated = 1
    began = time.monotonic()
    best_plain_distance = 10**9
    goal: State | None = None

    while frontier and expanded < state_limit and time.monotonic() - began < time_limit_s:
        _, cost, _, state = heapq.heappop(frontier)
        if cost != best_cost.get(state):
            continue
        expanded += 1
        plains = plain_positions(state)
        if len(plains) == 1:
            goal = state
            break
        distance = min(
            abs(a[0] - b[0]) + abs(a[1] - b[1])
            for index, a in enumerate(plains)
            for b in plains[index + 1 :]
        )
        if distance < best_plain_distance:
            best_plain_distance = distance
            print(
                "best",
                json.dumps(
                    {
                        "expanded": expanded,
                        "generated": generated,
                        "cost": cost,
                        "plain_distance": distance,
                        "plains": plains,
                        "blues": blue_positions(state),
                        "carriers": state.carriers,
                        "elapsed_s": round(time.monotonic() - began, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if expanded % 100_000 == 0:
            print(
                "progress",
                json.dumps(
                    {
                        "expanded": expanded,
                        "generated": generated,
                        "frontier": len(frontier),
                        "cost": cost,
                        "elapsed_s": round(time.monotonic() - began, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        successors: list[tuple[tuple[str, object], State]] = []
        for source, destination, successor in legal_jumps(layout, state):
            successors.append((("J", (source, destination)), successor))
        for action, delta in CARRIER_ACTIONS:
            successor = carrier_move(layout, state, delta)
            if successor != state:
                successors.append((("C", action), successor))

        for action, successor in successors:
            new_cost = cost + 1
            if new_cost >= best_cost.get(successor, 10**18):
                continue
            best_cost[successor] = new_cost
            parent[successor] = (state, action)
            priority = new_cost + heuristic(layout, successor)
            heapq.heappush(frontier, (priority, new_cost, next(serial), successor))
            generated += 1

    plan: list[tuple[str, object]] = []
    if goal is not None:
        cursor = goal
        while cursor != start:
            previous, action = parent[cursor]
            plan.append(action)
            cursor = previous
        plan.reverse()
    elapsed = time.monotonic() - began
    return {
        "level": level,
        "found": goal is not None,
        "plan": plan,
        "plan_steps": len(plan),
        "carrier_moves": sum(action[0] == "C" for action in plan),
        "jumps": sum(action[0] == "J" for action in plan),
        "expanded": expanded,
        "generated": generated,
        "frontier": len(frontier),
        "elapsed_s": round(elapsed, 3),
        "best_plain_distance": best_plain_distance,
        "state_limit": state_limit,
        "time_limit_s": time_limit_s,
        "initial": {
            "carriers": start.carriers,
            "plains": plain_positions(start),
            "blues": blue_positions(start),
        },
        "final": None
        if goal is None
        else {
            "carriers": goal.carriers,
            "plains": plain_positions(goal),
            "blues": blue_positions(goal),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(8, 9, 10), required=True)
    parser.add_argument("--states", type=int, default=5_000_000)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = solve(args.level, args.states, args.seconds)
    rendered = json.dumps(result, indent=2)
    print("RESULT", rendered, flush=True)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
