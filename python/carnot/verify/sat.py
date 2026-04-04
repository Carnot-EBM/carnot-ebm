"""SAT constraint satisfaction -- JAX implementation.

**Researcher summary:**
    Encodes CNF SAT clauses as differentiable energy terms using product
    relaxation. Variables are continuous in [0,1] (0=False, 1=True). Each
    clause's energy is the product of (1 - literal_value), which is 0 when
    any literal is true and 1 when all are false. A binary penalty pushes
    variables toward {0,1} after repair.

**Detailed explanation for engineers:**
    SAT (Boolean Satisfiability) is the canonical constraint satisfaction
    problem. A CNF formula has variables x_1..x_n and clauses like
    (x_1 OR NOT x_2 OR x_3). An assignment satisfies the formula when
    every clause has at least one true literal.

    EBMs need differentiable energy, so we use continuous relaxation:

    1. Variables x_i in [0,1] where 0=False, 1=True
    2. Positive literal x_i has value x_i; negated literal NOT x_i has
       value (1 - x_i)
    3. Clause energy: E = prod(1 - literal_value)
       - If any literal is ~1 (true), the product is ~0 (satisfied)
       - If all literals are ~0 (false), the product is ~1 (violated)
    4. Binary penalty: E = sum(x_i * (1 - x_i))
       - Pushes variables toward {0,1} — 0 at binary, max at 0.5

    This is the SAT analog of the Sudoku uniqueness constraint: both use
    continuous relaxation of discrete problems with smooth penalty functions.

    **Why product relaxation?** For typical 3-SAT clauses (3 literals),
    the product is simple and all literals contribute gradient signal.
    The alternative sum relaxation max(0, 1-sum(lits))^2 works too but
    has less uniform gradient distribution.

    **DIMACS format**: The standard CNF file format used in SAT competitions.
    Lines starting with 'c' are comments, 'p cnf N M' declares N variables
    and M clauses, and clause lines list literal indices (negative = negated)
    terminated by 0.

Spec: REQ-INFER-001, SCENARIO-INFER-001, SCENARIO-INFER-002, SCENARIO-INFER-006
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint, ComposedEnergy


@dataclass
class SATClause:
    """A single CNF clause: a disjunction of literals.

    **Researcher summary:**
        Represents (l_1 OR l_2 OR ... OR l_k) where each literal is a
        variable index plus a negation flag.

    **Detailed explanation for engineers:**
        Each literal is a tuple (variable_index, is_negated). Variable
        indices are 0-based. For example, the clause (x1 OR NOT x2 OR x3)
        with 0-based indexing is: [(0, False), (1, True), (2, False)].

    Spec: REQ-INFER-001
    """

    literals: list[tuple[int, bool]] = field(default_factory=list)


class SATClauseConstraint(BaseConstraint):
    """Energy for a single SAT clause using product relaxation.

    **Researcher summary:**
        E(x) = prod_{l in clause} (1 - val(l)), where val(l) is x_i for
        positive literal, (1 - x_i) for negated. Zero when satisfied.

    **Detailed explanation for engineers:**
        The product relaxation maps a discrete OR to a smooth function:

        - For clause (l_1 OR l_2 OR l_3):
          E = (1 - val(l_1)) * (1 - val(l_2)) * (1 - val(l_3))
        - If any literal has value ~1 (true): some factor is ~0, product ~0
        - If all literals have value ~0 (false): all factors ~1, product ~1

        The gradient flows through all literals, which is important for
        repair: the optimizer knows which variables to flip. This is
        smoother than the max-based alternative and works well for
        typical 3-SAT instances.

    For example::

        # Clause (x0 OR NOT x1 OR x2)
        clause = SATClause([(0, False), (1, True), (2, False)])
        constraint = SATClauseConstraint("clause_0", clause)
        x = jnp.array([1.0, 1.0, 0.0])  # x0=T, x1=T, x2=F
        # val(x0)=1, val(NOT x1)=0, val(x2)=0
        # E = (1-1)*(1-0)*(1-0) = 0 (satisfied by x0)

    Spec: REQ-INFER-001, SCENARIO-INFER-001
    """

    def __init__(self, name: str, clause: SATClause) -> None:
        self._name = name
        self._clause = clause

    @property
    def name(self) -> str:
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        return 0.01

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute product relaxation energy for this clause.

        Spec: REQ-INFER-001
        """
        product = jnp.float32(1.0)
        for var_idx, is_negated in self._clause.literals:
            val = x[var_idx]
            if is_negated:
                val = 1.0 - val
            product = product * (1.0 - val)
        return product


class SATBinaryConstraint(BaseConstraint):
    """Soft penalty pushing SAT variables toward binary {0, 1}.

    **Researcher summary:**
        E(x) = sum_i x_i * (1 - x_i). Zero at binary values, maximal
        at x_i = 0.5. Acts as a regularizer during repair.

    **Detailed explanation for engineers:**
        After gradient repair on clause constraints, variables may settle
        at fractional values (e.g., x_i = 0.7) that satisfy the continuous
        relaxation but don't correspond to a clean binary assignment. This
        penalty term nudges them toward {0, 1}:

        - At x_i = 0 or x_i = 1: penalty = 0
        - At x_i = 0.5: penalty = 0.25 (maximum)

        The weight on this constraint (default 0.1) is intentionally lower
        than clause weights (1.0) so clause satisfaction takes priority.
        After repair, the final rounding step snaps to binary.

    Spec: REQ-INFER-001, SCENARIO-INFER-006
    """

    def __init__(self, name: str, var_indices: list[int]) -> None:
        self._name = name
        self._var_indices = var_indices

    @property
    def name(self) -> str:
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        return 0.01

    def energy(self, x: jax.Array) -> jax.Array:
        """Sum of x_i * (1 - x_i) over all variables.

        Spec: SCENARIO-INFER-006
        """
        vals = x[jnp.array(self._var_indices)]
        return jnp.sum(vals * (1.0 - vals))


def build_sat_energy(
    clauses: list[SATClause],
    n_vars: int,
    clause_weight: float = 1.0,
    binary_weight: float = 0.1,
) -> ComposedEnergy:
    """Build a ComposedEnergy for a CNF SAT instance.

    **Researcher summary:**
        One constraint per clause (weight 1.0) plus one binary penalty
        (weight 0.1). Returns a ComposedEnergy ready for verify/repair.

    **Detailed explanation for engineers:**
        Mirrors ``build_sudoku_energy``: creates a ComposedEnergy with
        input_dim=n_vars, adds one SATClauseConstraint per clause and
        one SATBinaryConstraint for all variables. The binary penalty
        weight (0.1) is lower than clause weight (1.0) so the optimizer
        prioritizes satisfying clauses, then pushes toward clean binary.

    Args:
        clauses: List of SATClause objects defining the CNF formula.
        n_vars: Number of boolean variables.
        clause_weight: Weight for each clause constraint.
        binary_weight: Weight for the binary penalty.

    Returns:
        ComposedEnergy with input_dim=n_vars.

    Spec: REQ-INFER-001
    """
    composed = ComposedEnergy(input_dim=n_vars)

    for i, clause in enumerate(clauses):
        composed.add_constraint(
            SATClauseConstraint(f"clause_{i}", clause),
            clause_weight,
        )

    composed.add_constraint(
        SATBinaryConstraint("binary_penalty", list(range(n_vars))),
        binary_weight,
    )

    return composed


def parse_dimacs(text: str) -> tuple[list[SATClause], int]:
    """Parse DIMACS CNF format into clauses and variable count.

    **Researcher summary:**
        Standard SAT competition format. Lines: 'c' = comment,
        'p cnf N M' = header, literal lines end with 0.

    **Detailed explanation for engineers:**
        DIMACS CNF format:
        - Comment lines start with 'c'
        - Problem line: 'p cnf <n_vars> <n_clauses>'
        - Clause lines: space-separated integers ending with 0
          - Positive integer i means variable x_i (1-based!)
          - Negative integer -i means NOT x_i
          - 0 terminates the clause

        Variables are 1-based in DIMACS but 0-based internally, so we
        subtract 1 from each variable index.

    Args:
        text: DIMACS CNF format string.

    Returns:
        Tuple of (list of SATClause, number of variables).

    Spec: SCENARIO-INFER-002
    """
    clauses: list[SATClause] = []
    n_vars = 0
    current_literals: list[tuple[int, bool]] = []

    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            parts = line.split()
            n_vars = int(parts[2])
            continue

        # Clause line: integers ending with 0
        for token in line.split():
            lit = int(token)
            if lit == 0:
                if current_literals:
                    clauses.append(SATClause(list(current_literals)))
                    current_literals = []
            else:
                var_idx = abs(lit) - 1  # Convert 1-based to 0-based
                is_negated = lit < 0
                current_literals.append((var_idx, is_negated))

    # Handle case where last clause doesn't end with 0
    if current_literals:
        clauses.append(SATClause(list(current_literals)))

    return clauses, n_vars


def assignment_to_array(assignment: list[bool]) -> jax.Array:
    """Convert boolean assignment to JAX float array.

    Spec: REQ-INFER-001
    """
    return jnp.array([1.0 if v else 0.0 for v in assignment])


def array_to_assignment(x: jax.Array, threshold: float = 0.5) -> list[bool]:
    """Round continuous array to boolean assignment.

    Spec: REQ-INFER-001
    """
    return [bool(float(v) >= threshold) for v in x]
