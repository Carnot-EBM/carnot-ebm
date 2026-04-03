//! # Verifiable Reasoning: Constraints as Energy Terms
//!
//! **For EBM researchers:** This module implements constraint-as-energy verification
//! where each constraint is a differentiable energy term (0 when satisfied, >0 when
//! violated). Composed constraints form a joint energy landscape. Gradient-based repair
//! descends only on violated terms, preserving satisfied regions.
//!
//! **For software engineers new to EBMs:**
//!
//! This module is Carnot's anti-hallucination primitive. It answers the question:
//! "How do I PROVE that an AI's output is correct, rather than just hoping it is?"
//!
//! The key idea is surprisingly simple:
//!
//! 1. **A constraint is a function that returns a number.** If the answer is correct,
//!    the function returns 0. If the answer is wrong, it returns a positive number
//!    that tells you HOW WRONG it is.
//!
//! 2. **You can combine many constraints together.** Each constraint checks one rule.
//!    The total "wrongness" is the sum of all individual wrongness scores.
//!
//! 3. **Because these functions are differentiable (smooth), you can use calculus
//!    to FIX wrong answers.** The gradient tells you which direction to nudge each
//!    value to reduce the wrongness. This is "gradient-based repair."
//!
//! ## Why This Is Fundamentally Different From LLMs
//!
//! An LLM generates text by predicting "what word comes next?" — it has no mechanism
//! to CHECK whether its output satisfies hard constraints. It might generate a Sudoku
//! solution that looks plausible but has duplicate numbers in a row.
//!
//! This module flips the script: instead of generating-and-hoping, you define what
//! "correct" means as energy functions, and then the system can:
//! - **Verify** any proposed solution (is total energy ~0?)
//! - **Diagnose** which specific rules are broken (per-constraint decomposition)
//! - **Repair** broken solutions by following the gradient downhill
//!
//! ## Concrete Example: Sudoku
//!
//! Imagine verifying a Sudoku solution. You'd have constraints like:
//! - "Each row has unique digits" — energy = count of duplicates in each row
//! - "Each column has unique digits" — energy = count of duplicates in each column
//! - "Each 3x3 box has unique digits" — energy = count of duplicates in each box
//!
//! If someone gives you a wrong solution, you don't just get "wrong" — you get
//! "row 3 has a duplicate, column 7 has a duplicate, box 5 is fine." And the gradient
//! tells you exactly which cells to change and in which direction.
//!
//! Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004

pub mod sudoku;

use ndarray::{Array1, ArrayView1};

use crate::{EnergyFunction, Float};

/// A single verifiable constraint, expressed as an energy term.
///
/// **For EBM researchers:** A differentiable constraint function E_c(x) where
/// E_c(x) = 0 iff x satisfies the constraint, and E_c(x) > 0 proportional to
/// violation magnitude. Must provide both energy and gradient for use in
/// gradient-based repair.
///
/// **For software engineers:**
///
/// Think of a `ConstraintTerm` as a "rule checker with superpowers." A normal rule
/// checker returns true/false. This one returns a NUMBER that tells you:
/// - **0** means "rule is satisfied" (everything is fine)
/// - **A small positive number** means "rule is slightly violated" (almost correct)
/// - **A large positive number** means "rule is badly violated" (very wrong)
///
/// The superpower is that it also provides a GRADIENT — a set of directions telling
/// you "if you nudge these values in these directions, the violation will decrease."
/// This is what makes automatic repair possible.
///
/// For example, imagine a constraint "x must be at least 5":
/// - If x = 7, energy = 0 (satisfied, 7 >= 5)
/// - If x = 4, energy = 1 (violated by 1)
/// - If x = 2, energy = 9 (violated by 3, squared to 9)
/// - The gradient at x = 2 says "increase x" — which is exactly right!
///
/// For engineers coming from testing: this is like an assertion that, instead of
/// just failing, tells you exactly how far off you are and which direction to go.
///
/// Spec: REQ-VERIFY-001
pub trait ConstraintTerm: Send + Sync {
    /// Human-readable name for this constraint.
    ///
    /// Used in verification reports to identify which constraints passed or failed.
    /// For example: "row_3_unique", "column_sum_equals_45", "temperature_in_range".
    fn name(&self) -> &str;

    /// Compute constraint energy for configuration `x`. Returns 0.0 if satisfied.
    ///
    /// **For EBM researchers:** E_c(x) >= 0, with E_c(x) = 0 iff constraint satisfied.
    ///
    /// **For software engineers:** This is the "how wrong is it?" function. You pass in
    /// a candidate solution as a vector of numbers, and it tells you how badly this
    /// constraint is violated. Zero means perfect. Bigger means worse.
    ///
    /// For example, if the constraint is "values must sum to 10":
    /// - x = [3, 3, 4] -> energy = 0 (sum is 10, perfect)
    /// - x = [3, 3, 3] -> energy = 1 (sum is 9, off by 1, squared = 1)
    /// - x = [1, 1, 1] -> energy = 49 (sum is 3, off by 7, squared = 49)
    fn energy(&self, x: &ArrayView1<Float>) -> Float;

    /// Gradient of constraint energy with respect to x.
    ///
    /// **For EBM researchers:** dE_c/dx, used for gradient-based repair. Must be
    /// consistent with `energy()` (i.e., following -grad reduces energy).
    ///
    /// **For software engineers:** The gradient is a vector of the same size as `x`,
    /// where each element tells you "how much does the energy change if I increase
    /// this particular value by a tiny amount?"
    ///
    /// - A positive gradient at position i means: increasing x[i] increases the energy
    ///   (makes things worse), so you should DECREASE x[i] to fix things.
    /// - A negative gradient at position i means: increasing x[i] decreases the energy
    ///   (makes things better), so you should INCREASE x[i] to fix things.
    /// - A zero gradient at position i means: x[i] doesn't affect this constraint.
    ///
    /// For example, if the constraint is "x[0] must be >= 5" and x[0] = 2:
    /// - The gradient at index 0 would be negative (increasing x[0] reduces violation)
    /// - The gradient at index 1 would be zero (x[1] doesn't affect this constraint)
    ///
    /// The repair algorithm uses this by going in the OPPOSITE direction of the gradient
    /// (gradient descent): new_x = old_x - step_size * gradient.
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float>;

    /// Threshold below which the constraint is considered satisfied.
    ///
    /// Due to floating-point arithmetic, energy might never be exactly 0.0.
    /// This threshold (default: 1e-6) provides a tolerance. Energy values at
    /// or below this threshold are treated as "close enough to zero" = satisfied.
    ///
    /// You can override this for constraints that need tighter or looser tolerance.
    fn satisfaction_threshold(&self) -> Float {
        1e-6
    }

    /// Is this constraint satisfied for configuration x?
    ///
    /// Convenience method: returns true if energy(x) <= satisfaction_threshold().
    /// You rarely need to override this — just override `satisfaction_threshold()`
    /// if the default tolerance doesn't work for your constraint.
    fn is_satisfied(&self, x: &ArrayView1<Float>) -> bool {
        self.energy(x) <= self.satisfaction_threshold()
    }
}

/// Report for a single constraint's evaluation.
///
/// **For EBM researchers:** Per-term decomposition of the energy landscape, providing
/// both raw and weighted energies for interpretability and diagnosis.
///
/// **For software engineers:** This is like a line item on a report card. For each
/// constraint (rule), it tells you:
/// - What rule was checked (`name`)
/// - How badly it was violated (`energy`) — 0 means satisfied
/// - How much it contributed to the total score after weighting (`weighted_energy`)
/// - Whether it passed or failed (`satisfied`)
///
/// For example, in a Sudoku verification:
/// ```text
/// ConstraintReport { name: "row_1_unique", energy: 0.0, weighted_energy: 0.0, satisfied: true }
/// ConstraintReport { name: "row_2_unique", energy: 2.0, weighted_energy: 2.0, satisfied: false }
/// ```
/// This tells you row 1 is fine but row 2 has duplicates.
///
/// Spec: REQ-VERIFY-002, REQ-VERIFY-003
#[derive(Debug, Clone)]
pub struct ConstraintReport {
    /// The human-readable name of the constraint that was evaluated.
    pub name: String,
    /// Raw energy from the constraint (before weighting). 0.0 = satisfied.
    pub energy: Float,
    /// Energy after multiplying by the constraint's weight.
    /// This is what actually contributes to the total composed energy.
    /// Higher weights mean this constraint matters more in the total score.
    pub weighted_energy: Float,
    /// Whether the raw energy is below the satisfaction threshold.
    pub satisfied: bool,
}

/// Verification verdict: the final yes/no answer with details on failures.
///
/// **For EBM researchers:** Binary verdict with explicit enumeration of violated terms
/// for interpretable diagnostics.
///
/// **For software engineers:** This is the bottom line — did everything pass, or did
/// something fail? If something failed, it tells you EXACTLY which constraints failed
/// by name, so you can diagnose the problem.
///
/// For engineers coming from testing frameworks: this is like a test suite result —
/// either all tests pass (Verified) or you get a list of which tests failed (Violated).
///
/// Spec: REQ-VERIFY-003
#[derive(Debug, Clone)]
pub enum Verdict {
    /// All constraints satisfied — the configuration is verified correct.
    Verified,
    /// One or more constraints violated. `failing` contains the names of each
    /// violated constraint, enabling targeted diagnosis and repair.
    Violated { failing: Vec<String> },
}

/// Complete verification result for a configuration.
///
/// **For EBM researchers:** Full decomposition of the energy landscape at a point,
/// providing total energy, per-term reports, and a binary verdict. Useful for
/// monitoring convergence during training and for post-hoc verification.
///
/// **For software engineers:** This is the complete "verification report" you get
/// when you check whether a candidate solution satisfies all your rules. It bundles:
/// - The total energy (overall "wrongness score")
/// - Individual reports for each constraint
/// - A pass/fail verdict with the names of any failing constraints
///
/// For example, checking a Sudoku solution might produce:
/// ```text
/// VerificationResult {
///     total_energy: 4.0,
///     constraints: [row_1: ok, row_2: failed(2.0), col_5: failed(2.0), ...],
///     verdict: Violated { failing: ["row_2_unique", "col_5_unique"] }
/// }
/// ```
///
/// Spec: REQ-VERIFY-003
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Sum of all weighted constraint energies. 0.0 (or near-zero) means all
    /// constraints are satisfied. Useful for tracking progress during repair.
    pub total_energy: Float,
    /// Per-constraint breakdown. Lets you see exactly which rules passed/failed
    /// and by how much.
    pub constraints: Vec<ConstraintReport>,
    /// The final verdict: Verified (all pass) or Violated (with failing names).
    pub verdict: Verdict,
}

impl VerificationResult {
    /// Returns true if all constraints are satisfied.
    ///
    /// This is the primary "is this answer correct?" check. If this returns true,
    /// the configuration has been verified to satisfy all constraints.
    pub fn is_verified(&self) -> bool {
        matches!(self.verdict, Verdict::Verified)
    }

    /// Returns the names of failing constraints, if any.
    ///
    /// Useful for error messages and targeted repair. Returns an empty vec if
    /// everything is satisfied.
    ///
    /// For example: `["row_3_unique", "col_7_unique"]` tells you exactly which
    /// Sudoku rules are broken.
    pub fn failing_constraints(&self) -> Vec<&str> {
        match &self.verdict {
            Verdict::Verified => vec![],
            Verdict::Violated { failing } => failing.iter().map(|s| s.as_str()).collect(),
        }
    }
}

/// An energy function composed of weighted constraint terms.
///
/// **For EBM researchers:** A factored energy function E(x) = sum_i w_i * E_i(x) where
/// each E_i is a `ConstraintTerm`. Supports per-term decomposition for interpretability,
/// selective gradient computation (violated terms only) for repair, and implements
/// `EnergyFunction` for use with standard samplers (MCMC, Langevin, etc.).
///
/// **For software engineers:**
///
/// `ComposedEnergy` is the central data structure for verifiable reasoning. It lets you
/// build complex verification logic by combining simple, independent constraints — much
/// like building a complex query from simple WHERE clauses in SQL.
///
/// ## How It Works
///
/// You create a `ComposedEnergy` and add constraint terms to it, each with a weight:
///
/// ```text
/// let mut verifier = ComposedEnergy::new(81);  // 81 cells in a Sudoku
/// verifier.add_constraint(Box::new(RowUniqueConstraint::new(0)), 1.0);  // row 0
/// verifier.add_constraint(Box::new(RowUniqueConstraint::new(1)), 1.0);  // row 1
/// verifier.add_constraint(Box::new(ColUniqueConstraint::new(0)), 1.0);  // col 0
/// // ... etc
/// ```
///
/// The weight controls how much each constraint matters relative to others. A weight
/// of 2.0 means violations of that constraint count twice as much in the total energy.
///
/// ## Why Composition Is Powerful
///
/// Each constraint is simple and independently testable. But composed together, they
/// can express complex requirements:
/// - A Sudoku verifier is just 27 "uniqueness" constraints (9 rows + 9 cols + 9 boxes)
/// - A scheduling verifier might combine "no time conflicts" + "room capacity" + "teacher availability"
/// - A physics simulation verifier might combine "energy conservation" + "momentum conservation"
///
/// Because each constraint is independent, you get automatic diagnosis: if the Sudoku
/// fails, you know WHICH rows/columns/boxes have problems. With an LLM, you'd just
/// know "it's wrong."
///
/// ## The Three Superpowers
///
/// 1. **Verify**: Check if a solution satisfies all constraints at once
/// 2. **Decompose**: See which specific constraints pass/fail (diagnosis)
/// 3. **Repair**: Use gradients from violated constraints to fix a broken solution
///
/// For engineers coming from constraint solvers (like Z3 or OR-tools): this is similar
/// in spirit, but uses continuous optimization (gradient descent) instead of discrete
/// search. The trade-off is that it works with continuous/differentiable domains and
/// can handle soft constraints naturally via weights.
///
/// Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004
pub struct ComposedEnergy {
    /// Each entry is a (constraint, weight) pair. The weight scales the constraint's
    /// energy contribution to the total. Higher weight = more important constraint.
    terms: Vec<(Box<dyn ConstraintTerm>, Float)>,
    /// Dimensionality of the input vector. All constraints must operate on vectors
    /// of this size. For a Sudoku, this would be 81 (9x9 grid flattened).
    input_dim: usize,
}

impl ComposedEnergy {
    /// Create a new composed energy function with no constraints.
    ///
    /// `input_dim` is the size of the input vector that constraints will operate on.
    /// For example, a 9x9 Sudoku flattened to a vector would have input_dim = 81.
    pub fn new(input_dim: usize) -> Self {
        Self {
            terms: Vec::new(),
            input_dim,
        }
    }

    /// Add a constraint term with a weight.
    ///
    /// **For EBM researchers:** Adds w_i * E_i(x) to the composed energy.
    ///
    /// **For software engineers:** Register a new rule to check, with a weight that
    /// controls how important this rule is relative to others.
    ///
    /// - `weight = 1.0` is the default/normal importance
    /// - `weight = 2.0` means violations of this constraint count double
    /// - `weight = 0.5` means violations count half as much
    ///
    /// For example, in a scheduling system, you might weight "no double-booking"
    /// at 10.0 (hard constraint, must not violate) and "preferred room" at 0.1
    /// (soft constraint, nice to have).
    ///
    /// Spec: REQ-VERIFY-004
    pub fn add_constraint(&mut self, term: Box<dyn ConstraintTerm>, weight: Float) {
        self.terms.push((term, weight));
    }

    /// Number of constraint terms currently registered.
    pub fn num_constraints(&self) -> usize {
        self.terms.len()
    }

    /// Per-constraint energy decomposition: evaluate each constraint independently.
    ///
    /// **For EBM researchers:** Returns per-term E_i(x), w_i * E_i(x), and satisfaction
    /// status. The sum of weighted_energy values equals the total composed energy.
    ///
    /// **For software engineers:** This is the "detailed report" — instead of just
    /// getting a single pass/fail, you get a breakdown showing how each individual
    /// constraint performed. This is the key to INTERPRETABLE verification.
    ///
    /// For example, imagine checking a Sudoku solution:
    /// ```text
    /// decompose(x) -> [
    ///   { name: "row_0", energy: 0.0, satisfied: true },   // Row 0 is fine
    ///   { name: "row_1", energy: 4.0, satisfied: false },  // Row 1 has duplicates!
    ///   { name: "col_0", energy: 0.0, satisfied: true },   // Column 0 is fine
    ///   ...
    /// ]
    /// ```
    ///
    /// This decomposability is what makes EBM-based verification fundamentally more
    /// useful than a black-box "correct/incorrect" check.
    ///
    /// Spec: REQ-VERIFY-002
    pub fn decompose(&self, x: &ArrayView1<Float>) -> Vec<ConstraintReport> {
        self.terms
            .iter()
            .map(|(term, weight)| {
                let raw_energy = term.energy(x);
                ConstraintReport {
                    name: term.name().to_string(),
                    energy: raw_energy,
                    // The weighted energy is what actually counts toward the total.
                    // This lets constraints with higher weights dominate the optimization.
                    weighted_energy: weight * raw_energy,
                    satisfied: term.is_satisfied(x),
                }
            })
            .collect()
    }

    /// Produce a full verification result: total energy, per-constraint reports, and verdict.
    ///
    /// **For EBM researchers:** Evaluates E(x) = sum_i w_i * E_i(x), decomposes per term,
    /// and produces a binary verdict based on per-term satisfaction thresholds.
    ///
    /// **For software engineers:** This is the main "check everything" method. It:
    /// 1. Evaluates every constraint against the input
    /// 2. Sums up the weighted energies into a total score
    /// 3. Identifies which constraints failed
    /// 4. Returns a verdict: Verified (all pass) or Violated (with failure list)
    ///
    /// For example:
    /// ```text
    /// let result = verifier.verify(&sudoku_solution.view());
    /// if result.is_verified() {
    ///     println!("Valid Sudoku!");
    /// } else {
    ///     println!("Invalid! Broken rules: {:?}", result.failing_constraints());
    /// }
    /// ```
    ///
    /// Spec: REQ-VERIFY-003
    pub fn verify(&self, x: &ArrayView1<Float>) -> VerificationResult {
        // Step 1: Get the detailed per-constraint breakdown
        let reports = self.decompose(x);

        // Step 2: Sum up the total weighted energy across all constraints.
        // This single number summarizes "how wrong" the solution is overall.
        let total_energy: Float = reports.iter().map(|r| r.weighted_energy).sum();

        // Step 3: Collect the names of all violated (unsatisfied) constraints.
        // This enables targeted diagnosis — you know WHICH rules are broken.
        let failing: Vec<String> = reports
            .iter()
            .filter(|r| !r.satisfied)
            .map(|r| r.name.clone())
            .collect();

        // Step 4: Produce the verdict. If no constraints are failing, we're verified.
        let verdict = if failing.is_empty() {
            Verdict::Verified
        } else {
            Verdict::Violated { failing }
        };

        VerificationResult {
            total_energy,
            constraints: reports,
            verdict,
        }
    }

    /// Compute gradient from ONLY the violated constraints, ignoring satisfied ones.
    ///
    /// **For EBM researchers:** Computes sum_i [w_i * dE_i/dx] where the sum is
    /// restricted to terms where E_i(x) > threshold. This selective gradient is
    /// the key primitive for repair: it drives optimization toward satisfying violated
    /// constraints without disturbing already-satisfied regions of the energy landscape.
    ///
    /// **For software engineers:**
    ///
    /// This is the secret weapon of gradient-based repair. Normal gradient descent
    /// would compute gradients from ALL constraints, even the ones that are already
    /// satisfied. That's a problem because:
    ///
    /// - Imagine x[0] = 10 satisfies "x[0] >= 1" (satisfied) but x[1] = 0 violates
    ///   "x[1] >= 5" (violated).
    /// - If you computed gradients from ALL constraints, the satisfied constraint
    ///   might have a gradient that pushes x[0] around unnecessarily.
    /// - By only computing gradients from VIOLATED constraints, x[0] stays at 10
    ///   (gradient = 0 for satisfied constraints) while x[1] gets pushed toward 5.
    ///
    /// This is analogous to how a good teacher only corrects the mistakes in your
    /// homework — they don't change the answers you already got right.
    ///
    /// For example, in a Sudoku repair scenario:
    /// - Row 1 is correct (satisfied) -> its gradient is excluded, so row 1 stays intact
    /// - Row 3 has a duplicate (violated) -> its gradient is included, guiding the fix
    ///
    /// Spec: REQ-VERIFY-005
    pub fn grad_violated_only(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        for (term, weight) in &self.terms {
            // Only include gradient contributions from VIOLATED constraints.
            // Satisfied constraints contribute zero gradient, leaving those
            // dimensions of the solution undisturbed during repair.
            if !term.is_satisfied(x) {
                // Accumulate weighted gradient: w_i * dE_i/dx
                grad = grad + *weight * term.grad_energy(x);
            }
        }
        grad
    }
}

/// Implementation of the `EnergyFunction` trait for `ComposedEnergy`.
///
/// **For EBM researchers:** This allows `ComposedEnergy` to be used with any sampler
/// (MCMC, Langevin dynamics, etc.) as a standard energy function. The composed energy
/// E(x) = sum_i w_i * E_i(x) and its gradient are computed by aggregating all terms.
///
/// **For software engineers:** By implementing `EnergyFunction`, `ComposedEnergy` can
/// be plugged into Carnot's sampling infrastructure. This means you can not only VERIFY
/// solutions but also GENERATE solutions that satisfy all constraints — the sampler
/// will naturally find low-energy configurations where all constraints are satisfied.
///
/// This is the bridge between "checking answers" and "generating correct answers."
impl EnergyFunction for ComposedEnergy {
    /// Total energy: sum of all weighted constraint energies.
    /// E(x) = w_1 * E_1(x) + w_2 * E_2(x) + ... + w_n * E_n(x)
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        self.terms
            .iter()
            .map(|(term, weight)| weight * term.energy(x))
            .sum()
    }

    /// Total gradient: sum of all weighted constraint gradients.
    /// dE/dx = w_1 * dE_1/dx + w_2 * dE_2/dx + ... + w_n * dE_n/dx
    ///
    /// Note: unlike `grad_violated_only`, this includes ALL constraints — even
    /// satisfied ones. This is the standard gradient used by samplers; for repair,
    /// use `grad_violated_only` instead.
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        for (term, weight) in &self.terms {
            // Accumulate gradient from every constraint, regardless of satisfaction.
            grad = grad + *weight * term.grad_energy(x);
        }
        grad
    }

    /// The dimensionality of the input space.
    fn input_dim(&self) -> usize {
        self.input_dim
    }
}

/// Gradient-based repair: iteratively fix a broken solution by descending on violated constraints.
///
/// **For EBM researchers:** Performs projected gradient descent using only the gradient
/// from violated constraint terms: x_{t+1} = x_t - lr * sum_{violated i} [w_i * dE_i/dx].
/// Terminates when all constraints are satisfied or `max_steps` is reached. Returns the
/// full verification history for convergence analysis.
///
/// **For software engineers:**
///
/// This function takes a broken solution and tries to FIX it, step by step. Here's the
/// intuition:
///
/// 1. You start with a candidate solution `x` that violates some constraints.
/// 2. At each step, the function asks: "Which constraints are currently violated?"
/// 3. For those violated constraints, it computes a gradient — a direction that says
///    "nudge the values THIS way to reduce the violations."
/// 4. It takes a small step in that direction: `x_new = x_old - step_size * gradient`
/// 5. Repeat until everything is satisfied, or we run out of steps.
///
/// The crucial property: **satisfied constraints are left alone.** If row 1 of your
/// Sudoku is already correct, the repair process won't touch it — it only fixes the
/// broken rows. This is because `grad_violated_only` returns zero gradient for
/// satisfied constraints.
///
/// ## Parameters
///
/// - `composed`: The set of weighted constraints to satisfy
/// - `x`: The initial (possibly broken) solution to repair
/// - `step_size`: How big a step to take each iteration (like a learning rate in ML).
///   Too large = overshoot and oscillate. Too small = converge slowly. Typical: 0.01-0.1.
/// - `max_steps`: Safety limit on iterations. Prevents infinite loops if constraints
///   are unsatisfiable or step_size is too small.
///
/// ## Returns
///
/// A tuple of:
/// - The repaired solution vector (may or may not fully satisfy all constraints)
/// - A history of `VerificationResult` at each step, useful for monitoring convergence
///   (you should see total_energy decreasing over time)
///
/// ## Example Scenario
///
/// Imagine repairing a scheduling conflict:
/// ```text
/// Initial:  [Meeting A at 9am, Meeting B at 9am]  -- conflict! energy > 0
/// Step 1:   [Meeting A at 9am, Meeting B at 9:30am]  -- less conflict, energy decreased
/// Step 2:   [Meeting A at 9am, Meeting B at 10am]  -- no conflict! energy ≈ 0, done
/// ```
///
/// Spec: REQ-VERIFY-005
pub fn repair(
    composed: &ComposedEnergy,
    x: &Array1<Float>,
    step_size: Float,
    max_steps: usize,
) -> (Array1<Float>, Vec<VerificationResult>) {
    let mut x = x.clone();
    let mut history = Vec::new();

    for _ in 0..max_steps {
        // Check the current state: which constraints are satisfied/violated?
        let result = composed.verify(&x.view());

        // Remember whether we're done BEFORE moving the result into history.
        let done = result.is_verified();
        history.push(result);

        // If all constraints are satisfied, we're done — no need to repair further.
        if done {
            break;
        }

        // Compute gradient from ONLY the violated constraints.
        // Satisfied constraints contribute zero gradient, so they stay untouched.
        let grad = composed.grad_violated_only(&x.view());

        // Take a step in the negative gradient direction (gradient DESCENT).
        // This nudges x toward lower energy = fewer/smaller constraint violations.
        // The step_size controls how aggressive each step is.
        x = &x - step_size * &grad;
    }

    (x, history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Constraint: x[0] should be >= 1.0
    /// Energy = max(0, 1.0 - x[0])^2
    struct MinValueConstraint {
        index: usize,
        min_val: Float,
    }

    impl ConstraintTerm for MinValueConstraint {
        fn name(&self) -> &str {
            "min_value"
        }

        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            let violation = (self.min_val - x[self.index]).max(0.0);
            violation * violation
        }

        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            let mut grad = Array1::zeros(x.len());
            let violation = self.min_val - x[self.index];
            if violation > 0.0 {
                grad[self.index] = -2.0 * violation;
            }
            grad
        }
    }

    /// Constraint: sum of x should equal target
    /// Energy = (sum(x) - target)^2
    struct SumConstraint {
        target: Float,
    }

    impl ConstraintTerm for SumConstraint {
        fn name(&self) -> &str {
            "sum_equals_target"
        }

        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            let diff = x.sum() - self.target;
            diff * diff
        }

        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            let diff = x.sum() - self.target;
            Array1::from_elem(x.len(), 2.0 * diff)
        }
    }

    #[test]
    fn test_constraint_term_satisfied() {
        // REQ-VERIFY-001: satisfied constraint returns ~0 energy
        let c = MinValueConstraint {
            index: 0,
            min_val: 1.0,
        };
        let x = array![2.0, 0.0];
        assert!(c.energy(&x.view()) < 1e-6);
        assert!(c.is_satisfied(&x.view()));
    }

    #[test]
    fn test_constraint_term_violated() {
        // REQ-VERIFY-001: violated constraint returns > 0 energy
        let c = MinValueConstraint {
            index: 0,
            min_val: 1.0,
        };
        let x = array![0.5, 0.0];
        let e = c.energy(&x.view());
        assert!(e > 0.0);
        assert!(!c.is_satisfied(&x.view()));
        // Energy proportional to violation: (1.0 - 0.5)^2 = 0.25
        assert!((e - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_composed_energy_sum() {
        // REQ-VERIFY-004: composed energy is sum of weighted terms
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 1.0,
            }),
            1.0,
        );
        composed.add_constraint(Box::new(SumConstraint { target: 3.0 }), 2.0);

        let x = array![0.5, 1.0];
        let total = composed.energy(&x.view());

        // min_value: (1.0 - 0.5)^2 = 0.25, weight 1.0 -> 0.25
        // sum: (1.5 - 3.0)^2 = 2.25, weight 2.0 -> 4.5
        // total: 4.75
        assert!((total - 4.75).abs() < 1e-4);
    }

    #[test]
    fn test_decomposition_sums_to_total() {
        // SCENARIO-VERIFY-003: sum of decomposed energies = total
        let mut composed = ComposedEnergy::new(3);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 1.0,
            }),
            1.0,
        );
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 1,
                min_val: 2.0,
            }),
            1.5,
        );
        composed.add_constraint(Box::new(SumConstraint { target: 6.0 }), 1.0);

        let x = array![0.5, 1.0, 2.0];
        let reports = composed.decompose(&x.view());
        let total = composed.energy(&x.view());

        let decomposed_sum: Float = reports.iter().map(|r| r.weighted_energy).sum();
        assert!(
            (total - decomposed_sum).abs() < 1e-6,
            "Decomposition sum {decomposed_sum} != total {total}"
        );
        // Each constraint is named
        assert!(reports.iter().all(|r| !r.name.is_empty()));
    }

    #[test]
    fn test_verification_all_satisfied() {
        // REQ-VERIFY-003: VERIFIED when all constraints satisfied
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 1.0,
            }),
            1.0,
        );
        composed.add_constraint(Box::new(SumConstraint { target: 5.0 }), 1.0);

        let x = array![2.0, 3.0]; // min satisfied (2>=1), sum satisfied (5==5)
        let result = composed.verify(&x.view());
        assert!(result.is_verified());
        assert!(result.failing_constraints().is_empty());
        assert!(result.total_energy < 1e-6);
    }

    #[test]
    fn test_verification_with_violations() {
        // REQ-VERIFY-003: VIOLATED with specific failing constraint names
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 5.0,
            }),
            1.0,
        );
        composed.add_constraint(Box::new(SumConstraint { target: 10.0 }), 1.0);

        let x = array![1.0, 2.0]; // min violated (1<5), sum violated (3!=10)
        let result = composed.verify(&x.view());
        assert!(!result.is_verified());
        let failing = result.failing_constraints();
        assert_eq!(failing.len(), 2);
        assert!(failing.contains(&"min_value"));
        assert!(failing.contains(&"sum_equals_target"));
    }

    #[test]
    fn test_composition_add_constraints() {
        // REQ-VERIFY-004: constraints can be composed independently
        let mut composed = ComposedEnergy::new(2);
        assert_eq!(composed.num_constraints(), 0);

        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 0.0,
            }),
            1.0,
        );
        assert_eq!(composed.num_constraints(), 1);

        composed.add_constraint(Box::new(SumConstraint { target: 0.0 }), 2.0);
        assert_eq!(composed.num_constraints(), 2);
    }

    #[test]
    fn test_composed_as_energy_function() {
        // REQ-VERIFY-004: ComposedEnergy implements EnergyFunction
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(Box::new(SumConstraint { target: 0.0 }), 1.0);

        // Use through the EnergyFunction trait
        let ef: &dyn EnergyFunction = &composed;
        let x = array![1.0, -1.0]; // sum = 0, satisfied
        let e = ef.energy(&x.view());
        assert!(e < 1e-6);
        assert_eq!(ef.input_dim(), 2);
    }

    #[test]
    fn test_grad_violated_only() {
        // REQ-VERIFY-005: gradient from only violated constraints
        let mut composed = ComposedEnergy::new(2);
        // Constraint 1: x[0] >= 5.0 (will be violated)
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 5.0,
            }),
            1.0,
        );
        // Constraint 2: sum = 10.0 (will also be violated at x=[1,2], sum=3)
        // But we use x=[1,9] so sum=10 is satisfied while min_value is violated
        composed.add_constraint(Box::new(SumConstraint { target: 10.0 }), 1.0);

        // x=[1,9]: min_value violated (1<5), sum satisfied (1+9=10)
        let x = array![1.0, 9.0];

        // Sum constraint is satisfied, so its gradient contribution is non-zero
        // in the total but excluded from violated-only
        let grad_violated = composed.grad_violated_only(&x.view());
        let grad_all = composed.grad_energy(&x.view());

        // Total gradient includes sum constraint gradient (which is 0 here since
        // energy=0 means grad=0 too for sum constraint). Actually for a satisfied
        // quadratic constraint at exactly the target, grad IS 0. So let's verify
        // the violated-only gradient is correct on its own.

        // Violated-only gradient should only come from min_value constraint
        // min_value grad at x[0]=1, min=5: -2*(5-1) = -8 at index 0, 0 at index 1
        assert!(
            (grad_violated[0] - (-8.0)).abs() < 1e-4,
            "Expected grad[0]=-8.0, got {}",
            grad_violated[0]
        );
        assert!(
            grad_violated[1].abs() < 1e-6,
            "Expected grad[1]=0, got {}",
            grad_violated[1]
        );

        // Now test a case where both are violated so the gradient includes both
        let x2 = array![1.0, 2.0]; // min violated (1<5), sum violated (3!=10)
        let grad_both = composed.grad_violated_only(&x2.view());
        // min_value grad: [-8, 0], sum grad: [2*(3-10), 2*(3-10)] = [-14, -14]
        // total violated: [-22, -14]
        assert!(
            (grad_both[0] - (-22.0)).abs() < 1e-3,
            "Expected grad[0]=-22, got {}",
            grad_both[0]
        );
        assert!(
            (grad_both[1] - (-14.0)).abs() < 1e-3,
            "Expected grad[1]=-14, got {}",
            grad_both[1]
        );
    }

    #[test]
    fn test_repair_fixes_violations() {
        // REQ-VERIFY-005, SCENARIO-VERIFY-002: repair reduces violations
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 3.0,
            }),
            1.0,
        );
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 1,
                min_val: 3.0,
            }),
            1.0,
        );

        let x = array![0.0, 0.0]; // both violated
        let (repaired, history) = repair(&composed, &x, 0.1, 200);

        // Energy should decrease
        let initial_energy = history.first().unwrap().total_energy;
        let final_energy = history.last().unwrap().total_energy;
        assert!(
            final_energy < initial_energy,
            "Repair should reduce energy: {initial_energy} -> {final_energy}"
        );

        // Repaired values should be closer to the minimum
        assert!(
            repaired[0] > 0.0 && repaired[1] > 0.0,
            "Repair should push values toward min: {:?}",
            repaired
        );
    }

    #[test]
    fn test_repair_preserves_satisfied() {
        // SCENARIO-VERIFY-002: previously satisfied constraints remain satisfied
        let mut composed = ComposedEnergy::new(2);
        // x[0] >= 1.0 (will be satisfied from the start)
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 1.0,
            }),
            1.0,
        );
        // x[1] >= 5.0 (will be violated)
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 1,
                min_val: 5.0,
            }),
            1.0,
        );

        let x = array![10.0, 0.0]; // x[0] satisfied, x[1] violated
        let (repaired, _) = repair(&composed, &x, 0.1, 100);

        // x[0] should still be >= 1.0 (was satisfied, repair shouldn't touch it)
        // Because grad_violated_only only uses the violated x[1] constraint,
        // the gradient for x[0] is 0, so x[0] stays at 10.0
        assert!(
            repaired[0] >= 1.0,
            "Satisfied constraint should be preserved: x[0]={}",
            repaired[0]
        );
        // x[1] should have moved toward 5.0
        assert!(
            repaired[1] > 0.0,
            "Violated constraint should improve: x[1]={}",
            repaired[1]
        );
    }

    #[test]
    fn test_deterministic_reproducibility() {
        // SCENARIO-VERIFY-006: identical inputs produce identical results
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(Box::new(SumConstraint { target: 5.0 }), 1.0);

        let x = array![1.0, 2.0];
        let e1 = composed.energy(&x.view());
        let e2 = composed.energy(&x.view());
        let r1 = composed.verify(&x.view());
        let r2 = composed.verify(&x.view());

        assert_eq!(e1, e2, "Energy must be deterministic");
        assert_eq!(
            r1.total_energy, r2.total_energy,
            "Verification must be deterministic"
        );
    }
}
