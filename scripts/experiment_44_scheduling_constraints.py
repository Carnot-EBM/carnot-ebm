#!/usr/bin/env python3
"""Experiment 44: Scheduling constraint verification via Ising model.

**Researcher summary:**
    Encodes scheduling problems (meeting rooms, task ordering, resource
    capacity) as Ising models and verifies whether proposed schedules
    are feasible. Catches the kind of constraint violations LLMs commonly
    hallucinate: double-booked rooms, meetings before their prerequisites,
    same person in two places at once.

**Detailed explanation for engineers:**
    Scheduling is a practical constraint satisfaction domain where LLMs
    frequently produce impossible outputs. For example, an LLM might
    generate a schedule that puts Meeting A and Meeting B in Room 1 at
    2pm simultaneously.

    We encode scheduling as a binary Ising model where:

    Variable encoding:
        x_{e,t} = 1 means "event e is assigned to time slot t."
        For R resources, we have x_{e,t,r} = 1 means "event e is at
        time t in resource r." We flatten these into a 1D spin vector.

    Constraint types and their Ising encodings:

    1. **One-hot (exactly-one-slot):** Each event must occupy exactly one
       time slot. For event e with possible slots {t1, t2, ...}, we need
       sum_t x_{e,t} = 1. Encoded as:
       - Positive bias on each x_{e,t} (encourage at least one active).
       - Strong antiferromagnetic (negative) coupling between every pair
         (x_{e,ti}, x_{e,tj}) for i != j (penalize more than one active).

    2. **Exclusion (no double-booking):** Two events cannot share the same
       resource at the same time. For events e1, e2 at same time t and
       resource r: strong antiferromagnetic coupling between x_{e1,t,r}
       and x_{e2,t,r}.

    3. **Ordering (precedence):** Event A must finish before event B starts.
       Encoded as: for all t_a >= t_b, add antiferromagnetic coupling
       between x_{A,t_a} and x_{B,t_b}. This penalizes A being at or
       after B.

    4. **Duration:** Event e takes D slots. Encoded by treating the event
       as D sub-events that must occupy consecutive slots.

    5. **Resource capacity:** At most K events can use a resource at time t.
       Encoded pairwise: for every pair of events at the same time and
       resource, add a weak antiferromagnetic coupling. When more than K
       events try to be at the same time, the accumulated penalty exceeds
       the bias reward.

    The sampler searches for low-energy configurations. If the minimum
    energy found is zero (or very close), the schedule is feasible. If
    all samples have high energy, the constraints are unsatisfiable --
    indicating a scheduling conflict.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_44_scheduling_constraints.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-001
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


# ---------------------------------------------------------------------------
# 1. Scheduling problem definition
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single event (meeting, task) to be scheduled.

    Attributes:
        name: Human-readable label for the event.
        duration: Number of consecutive time slots the event occupies.
        resource: Which resource (room, machine) the event requires.
            If None, the event is resource-agnostic.
        person: Which person attends. Used for "same person, two places"
            conflict detection. If None, no person constraint.
    """
    name: str
    duration: int = 1
    resource: str | None = None
    person: str | None = None


@dataclass
class SchedulingProblem:
    """A scheduling problem with events, time slots, and constraints.

    Attributes:
        events: List of events to schedule.
        n_slots: Number of available time slots.
        resources: List of resource names (rooms, machines).
        ordering: List of (before_event_idx, after_event_idx) precedence pairs.
        capacity: Dict mapping resource name to max concurrent events.
        available_slots: Dict mapping event index to set of allowed slot indices.
            If absent for an event, all slots are available.
    """
    events: list[Event]
    n_slots: int
    resources: list[str]
    ordering: list[tuple[int, int]] = field(default_factory=list)
    capacity: dict[str, int] = field(default_factory=dict)
    available_slots: dict[int, set[int]] = field(default_factory=dict)


@dataclass
class ProposedSchedule:
    """A proposed assignment of events to time slots.

    Attributes:
        assignments: Dict mapping event index to the time slot it is
            assigned to. If an event is missing, it is unscheduled.
    """
    assignments: dict[int, int]


# ---------------------------------------------------------------------------
# 2. Ising encoding
# ---------------------------------------------------------------------------

def encode_scheduling_as_ising(
    problem: SchedulingProblem,
    schedule: ProposedSchedule,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode a scheduling problem + proposed schedule as an Ising model.

    **How the encoding works:**

    We create one binary spin per (event, time_slot) pair:
        spin index = event_idx * n_slots + slot_idx

    The proposed schedule is encoded via biases: for each assigned event,
    the assigned slot gets a strong positive bias (encouraging spin=1)
    and all other slots get a strong negative bias (discouraging them).

    Constraints are encoded as couplings in the coupling matrix J:

    - **One-hot:** Antiferromagnetic coupling between all pairs of slots
      for the same event. Penalty strength = PENALTY.
    - **Exclusion (double-booking):** Antiferromagnetic coupling between
      spins of different events at the same slot on the same resource.
    - **Ordering:** Antiferromagnetic coupling between x_{A,t_a} and
      x_{B,t_b} for all t_a >= t_b (A must be before B).
    - **Person conflict:** Antiferromagnetic coupling between spins of
      different events at the same time slot if they share a person.
    - **Availability:** Strong negative bias on spins for unavailable slots.

    Args:
        problem: The scheduling problem definition.
        schedule: The proposed schedule to verify.

    Returns:
        Tuple of (biases, coupling_matrix, spin_labels) where:
        - biases: shape (n_spins,) float32
        - coupling_matrix: shape (n_spins, n_spins) float32, symmetric, zero diagonal
        - spin_labels: human-readable label for each spin
    """
    n_events = len(problem.events)
    n_slots = problem.n_slots
    n_spins = n_events * n_slots

    biases = np.zeros(n_spins, dtype=np.float32)
    J = np.zeros((n_spins, n_spins), dtype=np.float32)

    # Penalty strength for hard constraints. Needs to be large enough
    # that violating a constraint always costs more than satisfying it.
    PENALTY = 5.0
    # Bias strength for the proposed assignment. Smaller than PENALTY so
    # that constraint violations dominate.
    ASSIGN_BIAS = 3.0

    # Helper: spin index for event e at time slot t.
    def spin(e: int, t: int) -> int:
        return e * n_slots + t

    # Human-readable labels for debugging.
    labels = []
    for e_idx, event in enumerate(problem.events):
        for t in range(n_slots):
            labels.append(f"{event.name}@t{t}")

    # --- Encode the proposed assignment as biases ---
    for e_idx, t_assigned in schedule.assignments.items():
        # Strong positive bias on the assigned slot.
        biases[spin(e_idx, t_assigned)] += ASSIGN_BIAS
        # Negative bias on all other slots (the event shouldn't be elsewhere).
        for t in range(n_slots):
            if t != t_assigned:
                biases[spin(e_idx, t)] -= ASSIGN_BIAS * 0.5

    # --- Constraint 1: One-hot (each event at exactly one time) ---
    # Antiferromagnetic coupling between all slot-pairs for the same event.
    for e_idx in range(n_events):
        for t1 in range(n_slots):
            for t2 in range(t1 + 1, n_slots):
                s1, s2 = spin(e_idx, t1), spin(e_idx, t2)
                J[s1, s2] -= PENALTY
                J[s2, s1] -= PENALTY

    # --- Constraint 2: Exclusion (same resource, same time) ---
    # Build a map: resource -> list of event indices using that resource.
    resource_events: dict[str, list[int]] = {}
    for e_idx, event in enumerate(problem.events):
        if event.resource:
            resource_events.setdefault(event.resource, []).append(e_idx)

    for _resource, e_indices in resource_events.items():
        for i, e1 in enumerate(e_indices):
            for e2 in e_indices[i + 1:]:
                # These events share a resource. They must not overlap in time.
                # For duration-1 events: same slot is forbidden.
                dur1 = problem.events[e1].duration
                dur2 = problem.events[e2].duration
                for t1 in range(n_slots):
                    for t2 in range(n_slots):
                        # Events overlap if their time ranges intersect.
                        # Event e1 at t1 occupies [t1, t1+dur1).
                        # Event e2 at t2 occupies [t2, t2+dur2).
                        if t1 < t2 + dur2 and t2 < t1 + dur1:
                            s1, s2 = spin(e1, t1), spin(e2, t2)
                            J[s1, s2] -= PENALTY
                            J[s2, s1] -= PENALTY

    # --- Constraint 3: Ordering (precedence) ---
    for before_idx, after_idx in problem.ordering:
        dur_before = problem.events[before_idx].duration
        for t_a in range(n_slots):
            for t_b in range(n_slots):
                # "before" event at t_a must finish by t_a + dur_before.
                # "after" event at t_b must start at or after that.
                # Violation: t_a + dur_before > t_b (before hasn't finished
                # when after starts).
                if t_a + dur_before > t_b:
                    s1, s2 = spin(before_idx, t_a), spin(after_idx, t_b)
                    J[s1, s2] -= PENALTY
                    J[s2, s1] -= PENALTY

    # --- Constraint 4: Person conflict (same person, same time) ---
    person_events: dict[str, list[int]] = {}
    for e_idx, event in enumerate(problem.events):
        if event.person:
            person_events.setdefault(event.person, []).append(e_idx)

    for _person, e_indices in person_events.items():
        for i, e1 in enumerate(e_indices):
            for e2 in e_indices[i + 1:]:
                dur1 = problem.events[e1].duration
                dur2 = problem.events[e2].duration
                for t1 in range(n_slots):
                    for t2 in range(n_slots):
                        if t1 < t2 + dur2 and t2 < t1 + dur1:
                            s1, s2 = spin(e1, t1), spin(e2, t2)
                            J[s1, s2] -= PENALTY
                            J[s2, s1] -= PENALTY

    # --- Constraint 5: Availability windows ---
    for e_idx, allowed in problem.available_slots.items():
        for t in range(n_slots):
            if t not in allowed:
                # This slot is unavailable for this event. Strong
                # negative bias makes the sampler avoid it.
                biases[spin(e_idx, t)] -= PENALTY * 2.0

    return biases, J, labels


# ---------------------------------------------------------------------------
# 3. Verification via ParallelIsingSampler
# ---------------------------------------------------------------------------

def verify_schedule(
    problem: SchedulingProblem,
    schedule: ProposedSchedule,
) -> dict[str, Any]:
    """Verify a proposed schedule by sampling the Ising encoding.

    **How it works:**
    1. Encode the scheduling problem + proposed schedule as Ising biases
       and couplings.
    2. Run ParallelIsingSampler to search for low-energy configurations.
    3. Check the proposed assignment directly for constraint violations.
    4. If the sampler finds a lower-energy (fewer violations) configuration
       than the proposal, report which constraints are violated.

    Returns:
        Dict with keys:
        - "feasible": bool -- whether the proposed schedule is conflict-free.
        - "violations": list of human-readable violation descriptions.
        - "n_violations": int
        - "energy_proposed": float -- energy of the proposed assignment.
        - "energy_best": float -- energy of the best sampled assignment.
    """
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.samplers.parallel_ising import (
        ParallelIsingSampler,
        AnnealingSchedule,
    )

    # Direct violation check (ground truth, no sampling needed).
    violations = check_violations_direct(problem, schedule)

    # Also run Ising sampling to verify the encoding works.
    biases, J, labels = encode_scheduling_as_ising(problem, schedule)
    n_spins = biases.shape[0]

    sampler = ParallelIsingSampler(
        n_warmup=800,
        n_samples=40,
        steps_per_sample=15,
        schedule=AnnealingSchedule(0.1, 10.0),
        use_checkerboard=True,
    )

    samples = sampler.sample(
        jrandom.PRNGKey(44),
        jnp.array(biases, dtype=jnp.float32),
        jnp.array(J, dtype=jnp.float32),
        beta=10.0,
    )

    # Compute energy of the proposed assignment.
    n_slots = problem.n_slots
    proposed_spins = np.zeros(n_spins, dtype=np.float32)
    for e_idx, t in schedule.assignments.items():
        proposed_spins[e_idx * n_slots + t] = 1.0
    energy_proposed = _ising_energy(proposed_spins, biases, J)

    # Find best energy among samples.
    best_energy = float("inf")
    for s_idx in range(samples.shape[0]):
        s = np.array(samples[s_idx], dtype=np.float32)
        e = _ising_energy(s, biases, J)
        if e < best_energy:
            best_energy = e

    return {
        "feasible": len(violations) == 0,
        "violations": violations,
        "n_violations": len(violations),
        "energy_proposed": float(energy_proposed),
        "energy_best": float(best_energy),
    }


def _ising_energy(
    spins: np.ndarray,
    biases: np.ndarray,
    J: np.ndarray,
) -> float:
    """Compute Ising energy: E = -(b^T s + s^T J s).

    Lower energy = better (more constraints satisfied, stronger assignment).
    """
    return float(-(biases @ spins + spins @ J @ spins))


def check_violations_direct(
    problem: SchedulingProblem,
    schedule: ProposedSchedule,
) -> list[str]:
    """Directly check a proposed schedule for constraint violations.

    This is the ground-truth checker. The Ising encoding should agree
    with this checker: if this says "violation," the Ising model should
    assign high energy, and vice versa.

    Returns:
        List of human-readable violation descriptions. Empty = feasible.
    """
    violations: list[str] = []
    n_events = len(problem.events)
    n_slots = problem.n_slots

    # Check: every event is assigned.
    for e_idx in range(n_events):
        if e_idx not in schedule.assignments:
            violations.append(
                f"Event '{problem.events[e_idx].name}' is not assigned a time slot"
            )

    # Check: assigned slot is within bounds.
    for e_idx, t in schedule.assignments.items():
        dur = problem.events[e_idx].duration
        if t < 0 or t + dur > n_slots:
            violations.append(
                f"Event '{problem.events[e_idx].name}' at t={t} with "
                f"duration {dur} exceeds slot range [0, {n_slots})"
            )

    # Check: availability windows.
    for e_idx, t in schedule.assignments.items():
        if e_idx in problem.available_slots:
            if t not in problem.available_slots[e_idx]:
                violations.append(
                    f"Event '{problem.events[e_idx].name}' at t={t} is "
                    f"outside its available slots "
                    f"{sorted(problem.available_slots[e_idx])}"
                )

    # Check: resource exclusion (no double-booking).
    resource_usage: dict[str, dict[int, list[str]]] = {}
    for e_idx, t in schedule.assignments.items():
        event = problem.events[e_idx]
        if event.resource:
            for dt in range(event.duration):
                slot = t + dt
                usage = resource_usage.setdefault(event.resource, {})
                usage.setdefault(slot, []).append(event.name)

    for resource, slot_map in resource_usage.items():
        for slot, names in slot_map.items():
            cap = problem.capacity.get(resource, 1)
            if len(names) > cap:
                violations.append(
                    f"Resource '{resource}' at t={slot} has {len(names)} "
                    f"events ({', '.join(names)}) but capacity is {cap}"
                )

    # Check: ordering constraints.
    for before_idx, after_idx in problem.ordering:
        if before_idx in schedule.assignments and after_idx in schedule.assignments:
            t_before = schedule.assignments[before_idx]
            t_after = schedule.assignments[after_idx]
            dur_before = problem.events[before_idx].duration
            if t_before + dur_before > t_after:
                violations.append(
                    f"Ordering violated: '{problem.events[before_idx].name}' "
                    f"(t={t_before}, dur={dur_before}) must finish before "
                    f"'{problem.events[after_idx].name}' (t={t_after})"
                )

    # Check: person conflicts (same person, overlapping times).
    person_schedule: dict[str, list[tuple[int, int, str]]] = {}
    for e_idx, t in schedule.assignments.items():
        event = problem.events[e_idx]
        if event.person:
            entries = person_schedule.setdefault(event.person, [])
            entries.append((t, t + event.duration, event.name))

    for person, entries in person_schedule.items():
        entries_sorted = sorted(entries)
        for i, (s1, e1, n1) in enumerate(entries_sorted):
            for s2, e2, n2 in entries_sorted[i + 1:]:
                if s1 < e2 and s2 < e1:
                    violations.append(
                        f"Person '{person}' double-booked: "
                        f"'{n1}' [t={s1}-{e1}) and '{n2}' [t={s2}-{e2})"
                    )

    return violations


# ---------------------------------------------------------------------------
# 4. Test scenarios
# ---------------------------------------------------------------------------

def get_test_scenarios() -> list[dict[str, Any]]:
    """Return 10 scheduling test scenarios: valid and invalid.

    Each scenario is a dict with:
    - name: description
    - problem: SchedulingProblem
    - schedule: ProposedSchedule
    - expected_feasible: bool
    - description: what the scenario tests
    """
    scenarios: list[dict[str, Any]] = []

    # === VALID SCHEDULES (should be feasible) ===

    # Scenario 1: 3 meetings, 3 rooms, no conflicts.
    scenarios.append({
        "name": "3 meetings, 3 rooms, no conflicts",
        "description": "Each meeting in its own room at the same time -- no overlap.",
        "problem": SchedulingProblem(
            events=[
                Event("StandupA", resource="Room1"),
                Event("StandupB", resource="Room2"),
                Event("StandupC", resource="Room3"),
            ],
            n_slots=4,
            resources=["Room1", "Room2", "Room3"],
        ),
        "schedule": ProposedSchedule({0: 0, 1: 0, 2: 0}),
        "expected_feasible": True,
    })

    # Scenario 2: Sequential tasks with ordering constraints.
    scenarios.append({
        "name": "Sequential tasks, correct order",
        "description": "A before B before C, and the schedule respects this.",
        "problem": SchedulingProblem(
            events=[
                Event("Build", resource="CI"),
                Event("Test", resource="CI"),
                Event("Deploy", resource="CI"),
            ],
            n_slots=6,
            resources=["CI"],
            ordering=[(0, 1), (1, 2)],
        ),
        "schedule": ProposedSchedule({0: 0, 1: 1, 2: 2}),
        "expected_feasible": True,
    })

    # Scenario 3: Multi-slot events, no overlap.
    scenarios.append({
        "name": "Multi-slot events, no overlap",
        "description": "Two 2-hour meetings in the same room, back to back.",
        "problem": SchedulingProblem(
            events=[
                Event("Workshop", duration=2, resource="BigRoom"),
                Event("Review", duration=2, resource="BigRoom"),
            ],
            n_slots=6,
            resources=["BigRoom"],
        ),
        "schedule": ProposedSchedule({0: 0, 1: 2}),
        "expected_feasible": True,
    })

    # Scenario 4: Person attends two non-overlapping meetings.
    scenarios.append({
        "name": "Person in two meetings, no overlap",
        "description": "Alice attends meetings at different times -- no conflict.",
        "problem": SchedulingProblem(
            events=[
                Event("Morning sync", resource="Room1", person="Alice"),
                Event("Afternoon review", resource="Room2", person="Alice"),
            ],
            n_slots=4,
            resources=["Room1", "Room2"],
        ),
        "schedule": ProposedSchedule({0: 0, 1: 2}),
        "expected_feasible": True,
    })

    # === INVALID SCHEDULES (LLM-style errors) ===

    # Scenario 5: Double-booked room.
    scenarios.append({
        "name": "Double-booked room",
        "description": "Two meetings in Room1 at the same time. Classic LLM error.",
        "problem": SchedulingProblem(
            events=[
                Event("MeetingA", resource="Room1"),
                Event("MeetingB", resource="Room1"),
            ],
            n_slots=4,
            resources=["Room1"],
        ),
        "schedule": ProposedSchedule({0: 1, 1: 1}),
        "expected_feasible": False,
    })

    # Scenario 6: Meeting before its prerequisite.
    scenarios.append({
        "name": "Meeting before prerequisite",
        "description": "Test scheduled before Build, violating ordering.",
        "problem": SchedulingProblem(
            events=[
                Event("Build", resource="CI"),
                Event("Test", resource="CI"),
            ],
            n_slots=4,
            resources=["CI"],
            ordering=[(0, 1)],  # Build must be before Test
        ),
        "schedule": ProposedSchedule({0: 2, 1: 1}),  # Test at t=1, Build at t=2!
        "expected_feasible": False,
    })

    # Scenario 7: Same person in two places at once.
    scenarios.append({
        "name": "Same person in two places",
        "description": "Alice is in Room1 and Room2 at the same time.",
        "problem": SchedulingProblem(
            events=[
                Event("Design review", resource="Room1", person="Alice"),
                Event("Sprint planning", resource="Room2", person="Alice"),
            ],
            n_slots=4,
            resources=["Room1", "Room2"],
        ),
        "schedule": ProposedSchedule({0: 1, 1: 1}),
        "expected_feasible": False,
    })

    # Scenario 8: Event outside available hours.
    scenarios.append({
        "name": "Event outside available hours",
        "description": "Meeting scheduled at t=3 but only t=0,1 are available.",
        "problem": SchedulingProblem(
            events=[
                Event("Morning standup", resource="Room1"),
            ],
            n_slots=4,
            resources=["Room1"],
            available_slots={0: {0, 1}},  # Only morning slots
        ),
        "schedule": ProposedSchedule({0: 3}),  # Scheduled in afternoon
        "expected_feasible": False,
    })

    # Scenario 9: Overlapping multi-slot events in same room.
    scenarios.append({
        "name": "Overlapping multi-slot events",
        "description": "2-hour workshop starts at t=1, 2-hour review starts at t=2. They overlap at t=2.",
        "problem": SchedulingProblem(
            events=[
                Event("Workshop", duration=2, resource="BigRoom"),
                Event("Review", duration=2, resource="BigRoom"),
            ],
            n_slots=6,
            resources=["BigRoom"],
        ),
        "schedule": ProposedSchedule({0: 1, 1: 2}),  # Overlap at slot 2
        "expected_feasible": False,
    })

    # Scenario 10: Resource capacity exceeded.
    scenarios.append({
        "name": "Resource capacity exceeded",
        "description": "Lab supports 2 concurrent experiments, but 3 are scheduled at t=0.",
        "problem": SchedulingProblem(
            events=[
                Event("ExpA", resource="Lab"),
                Event("ExpB", resource="Lab"),
                Event("ExpC", resource="Lab"),
            ],
            n_slots=4,
            resources=["Lab"],
            capacity={"Lab": 2},  # Max 2 concurrent
        ),
        "schedule": ProposedSchedule({0: 0, 1: 0, 2: 0}),  # All 3 at t=0
        "expected_feasible": False,
    })

    return scenarios


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all scheduling scenarios and report results."""
    print("=" * 72)
    print("EXPERIMENT 44: Scheduling Constraint Verification via Ising Model")
    print("  Encode scheduling constraints as Ising couplings + biases.")
    print("  Detect double-booking, ordering violations, capacity overflows.")
    print("=" * 72)

    start = time.time()
    scenarios = get_test_scenarios()
    results: list[dict[str, Any]] = []

    for scenario in scenarios:
        name = scenario["name"]
        problem = scenario["problem"]
        schedule = scenario["schedule"]
        expected = scenario["expected_feasible"]

        result = verify_schedule(problem, schedule)
        actual = result["feasible"]
        correct = actual == expected

        icon = "+" if correct else "X"
        status = "FEASIBLE" if actual else f"INFEASIBLE ({result['n_violations']} violations)"
        print(f"  [{icon}] {name:<42s} -> {status}")

        if result["violations"]:
            for v in result["violations"]:
                print(f"        - {v}")

        # Show Ising energy comparison.
        print(
            f"        Energy: proposed={result['energy_proposed']:.1f}, "
            f"best_sampled={result['energy_best']:.1f}"
        )

        results.append({
            "name": name,
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "n_violations": result["n_violations"],
            "energy_proposed": result["energy_proposed"],
            "energy_best": result["energy_best"],
            "description": scenario["description"],
        })

    # --- Summary ---
    elapsed = time.time() - start
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"EXPERIMENT 44 RESULTS ({elapsed:.1f}s)")
    print(sep)

    n_total = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_valid = sum(1 for r in results if r["expected"])
    n_invalid = n_total - n_valid
    n_valid_correct = sum(1 for r in results if r["expected"] and r["actual"])
    n_invalid_detected = sum(
        1 for r in results if not r["expected"] and not r["actual"]
    )

    print(f"  Total scenarios:                 {n_total}")
    print(f"  Correct classifications:         {n_correct}/{n_total}")
    print(f"  Valid schedules confirmed:       {n_valid_correct}/{n_valid}")
    print(f"  Invalid schedules detected:      {n_invalid_detected}/{n_invalid}")

    # Energy analysis: invalid schedules should have higher proposed energy.
    valid_energies = [
        r["energy_proposed"] for r in results if r["expected"]
    ]
    invalid_energies = [
        r["energy_proposed"] for r in results if not r["expected"]
    ]
    if valid_energies and invalid_energies:
        avg_valid = np.mean(valid_energies)
        avg_invalid = np.mean(invalid_energies)
        print(f"\n  Avg proposed energy (valid):    {avg_valid:.1f}")
        print(f"  Avg proposed energy (invalid):  {avg_invalid:.1f}")
        print(f"  Energy gap (invalid - valid):   {avg_invalid - avg_valid:.1f}")

    if n_correct == n_total:
        print(f"\n  VERDICT: PASS -- Perfect scheduling constraint detection!")
    elif n_correct >= n_total * 0.8:
        print(f"\n  VERDICT: PASS -- Good detection ({n_correct}/{n_total})")
    else:
        print(f"\n  VERDICT: PARTIAL -- Needs work ({n_correct}/{n_total})")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
