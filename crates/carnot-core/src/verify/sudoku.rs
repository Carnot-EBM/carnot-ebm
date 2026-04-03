//! Sudoku constraint satisfaction as an energy-based verification example.
//!
//! Demonstrates the full verifiable reasoning pipeline:
//! - Encode row, column, box, and clue constraints as ConstraintTerms
//! - Compose them into a single energy function
//! - Verify a solution (energy=0 ⟺ valid Sudoku)
//! - Repair an invalid solution via gradient descent on violated constraints
//!
//! Spec: SCENARIO-VERIFY-001, SCENARIO-VERIFY-002

use ndarray::{Array1, ArrayView1};

use crate::verify::{ComposedEnergy, ConstraintTerm};
use crate::Float;

/// Constraint: all values in a group of 9 cells should be distinct (1-9).
///
/// Energy = sum over all pairs (i,j) of max(0, 1 - |x[i] - x[j]|)^2
/// This is 0 when all values are at least 1 apart (distinct integers),
/// and > 0 when any two values are close (duplicates).
struct UniquenessConstraint {
    name: String,
    /// Indices into the 81-element grid.
    indices: [usize; 9],
}

impl ConstraintTerm for UniquenessConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let mut energy: Float = 0.0;
        for i in 0..9 {
            for j in (i + 1)..9 {
                let diff = (x[self.indices[i]] - x[self.indices[j]]).abs();
                let violation = (1.0 - diff).max(0.0);
                energy += violation * violation;
            }
        }
        energy
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        for i in 0..9 {
            for j in (i + 1)..9 {
                let xi = x[self.indices[i]];
                let xj = x[self.indices[j]];
                let diff = xi - xj;
                let abs_diff = diff.abs();
                if abs_diff < 1.0 {
                    // d/dx_i [(1 - |xi - xj|)^2] = -2(1 - |d|) * sign(d)
                    let sign = if diff >= 0.0 { 1.0 } else { -1.0 };
                    let factor = -2.0 * (1.0 - abs_diff) * sign;
                    grad[self.indices[i]] += factor;
                    grad[self.indices[j]] -= factor;
                }
            }
        }
        grad
    }

    fn satisfaction_threshold(&self) -> Float {
        0.01 // slightly relaxed for continuous optimization
    }
}

/// Constraint: a cell should equal a given clue value.
///
/// Energy = (x[index] - value)^2
struct ClueConstraint {
    name: String,
    index: usize,
    value: Float,
}

impl ConstraintTerm for ClueConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let diff = x[self.index] - self.value;
        diff * diff
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        grad[self.index] = 2.0 * (x[self.index] - self.value);
        grad
    }

    fn satisfaction_threshold(&self) -> Float {
        0.01
    }
}

/// Build a Sudoku energy function from a puzzle.
///
/// `clues` is a 9x9 grid where 0 means empty and 1-9 are given values.
///
/// Returns a ComposedEnergy with:
/// - 9 row uniqueness constraints
/// - 9 column uniqueness constraints
/// - 9 box uniqueness constraints
/// - N clue constraints (one per given value)
///
/// Spec: SCENARIO-VERIFY-001
pub fn build_sudoku_energy(clues: &[[u8; 9]; 9]) -> ComposedEnergy {
    let mut composed = ComposedEnergy::new(81);

    // Row constraints
    for row in 0..9 {
        let indices: [usize; 9] = std::array::from_fn(|col| row * 9 + col);
        composed.add_constraint(
            Box::new(UniquenessConstraint {
                name: format!("row_{row}"),
                indices,
            }),
            1.0,
        );
    }

    // Column constraints
    for col in 0..9 {
        let indices: [usize; 9] = std::array::from_fn(|row| row * 9 + col);
        composed.add_constraint(
            Box::new(UniquenessConstraint {
                name: format!("col_{col}"),
                indices,
            }),
            1.0,
        );
    }

    // 3x3 box constraints
    for box_row in 0..3 {
        for box_col in 0..3 {
            let mut indices = [0usize; 9];
            let mut k = 0;
            for r in 0..3 {
                for c in 0..3 {
                    indices[k] = (box_row * 3 + r) * 9 + (box_col * 3 + c);
                    k += 1;
                }
            }
            composed.add_constraint(
                Box::new(UniquenessConstraint {
                    name: format!("box_{box_row}_{box_col}"),
                    indices,
                }),
                1.0,
            );
        }
    }

    // Clue constraints
    for row in 0..9 {
        for col in 0..9 {
            if clues[row][col] != 0 {
                composed.add_constraint(
                    Box::new(ClueConstraint {
                        name: format!("clue_r{row}c{col}"),
                        index: row * 9 + col,
                        value: clues[row][col] as Float,
                    }),
                    10.0, // high weight to strongly enforce clues
                );
            }
        }
    }

    composed
}

/// Convert a 9x9 grid to a flat 81-element array.
pub fn grid_to_array(grid: &[[u8; 9]; 9]) -> Array1<Float> {
    let mut arr = Array1::zeros(81);
    for row in 0..9 {
        for col in 0..9 {
            arr[row * 9 + col] = grid[row][col] as Float;
        }
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::repair;
    use crate::EnergyFunction;

    // A known valid Sudoku solution
    const VALID_SOLUTION: [[u8; 9]; 9] = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ];

    // Puzzle with some clues (zeros = empty)
    const PUZZLE: [[u8; 9]; 9] = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ];

    #[test]
    fn test_valid_solution_energy_zero() {
        // SCENARIO-VERIFY-001: correct solution has total energy = 0
        let energy_fn = build_sudoku_energy(&PUZZLE);
        let x = grid_to_array(&VALID_SOLUTION);
        let result = energy_fn.verify(&x.view());

        assert!(
            result.is_verified(),
            "Valid Sudoku should verify. Failing: {:?}",
            result.failing_constraints()
        );
        assert!(
            result.total_energy < 0.1,
            "Valid Sudoku energy should be ~0, got {}",
            result.total_energy
        );
    }

    #[test]
    fn test_invalid_solution_has_violations() {
        // SCENARIO-VERIFY-001: incorrect solution has energy > 0
        let energy_fn = build_sudoku_energy(&PUZZLE);

        // Create an invalid solution: swap two values in row 0
        let mut bad = VALID_SOLUTION;
        bad[0][0] = 3; // was 5
        bad[0][1] = 5; // was 3 — now row has duplicate-free but clues violated
        let x = grid_to_array(&bad);
        let result = energy_fn.verify(&x.view());

        assert!(!result.is_verified());
        assert!(result.total_energy > 0.1);

        // Should specifically identify the violated clue constraints
        let failing = result.failing_constraints();
        assert!(
            !failing.is_empty(),
            "Should have failing constraints"
        );
    }

    #[test]
    fn test_per_constraint_decomposition() {
        // SCENARIO-VERIFY-001, SCENARIO-VERIFY-003: per-constraint energy
        let energy_fn = build_sudoku_energy(&PUZZLE);
        let x = grid_to_array(&VALID_SOLUTION);
        let reports = energy_fn.decompose(&x.view());

        // Should have 9 rows + 9 cols + 9 boxes + N clues
        assert!(reports.len() >= 27);

        // All row/col/box constraints should be satisfied
        for r in &reports {
            if r.name.starts_with("row_") || r.name.starts_with("col_") || r.name.starts_with("box_") {
                assert!(
                    r.satisfied,
                    "Constraint {} should be satisfied, energy={}",
                    r.name, r.energy
                );
            }
        }

        // Sum of weighted energies should equal total
        let total = energy_fn.energy(&x.view());
        let decomposed_sum: Float = reports.iter().map(|r| r.weighted_energy).sum();
        assert!(
            (total - decomposed_sum).abs() < 1e-4,
            "Decomposition sum {} != total {}",
            decomposed_sum, total
        );
    }

    #[test]
    fn test_repair_improves_invalid_solution() {
        // SCENARIO-VERIFY-002: gradient-based repair reduces violations
        let energy_fn = build_sudoku_energy(&PUZZLE);

        // Start with a bad configuration: all 5s
        let x = Array1::from_elem(81, 5.0 as Float);
        let initial_result = energy_fn.verify(&x.view());
        assert!(!initial_result.is_verified());

        // Repair
        let (repaired, history) = repair(&energy_fn, &x, 0.01, 500);

        // Energy should decrease
        let initial_energy = history.first().unwrap().total_energy;
        let final_energy = history.last().unwrap().total_energy;
        assert!(
            final_energy < initial_energy,
            "Repair should reduce energy: {} -> {}",
            initial_energy, final_energy
        );

        // Number of violations should decrease (or at least not increase)
        let initial_violations = history.first().unwrap().failing_constraints().len();
        let final_violations = history.last().unwrap().failing_constraints().len();
        assert!(
            final_violations <= initial_violations,
            "Violations should not increase: {} -> {}",
            initial_violations, final_violations
        );
    }

    #[test]
    fn test_clue_constraints_high_weight() {
        // SCENARIO-VERIFY-001: clue constraints have high weight
        let energy_fn = build_sudoku_energy(&PUZZLE);

        // Solution that violates a clue
        let mut bad = VALID_SOLUTION;
        bad[0][0] = 1; // should be 5 (clue)
        let x = grid_to_array(&bad);
        let reports = energy_fn.decompose(&x.view());

        // Find the violated clue constraint
        let clue_report = reports.iter().find(|r| r.name == "clue_r0c0").unwrap();
        assert!(!clue_report.satisfied);
        // Weighted energy should be high (weight=10)
        assert!(
            clue_report.weighted_energy > 1.0,
            "Clue violation should have high weighted energy: {}",
            clue_report.weighted_energy
        );
    }
}
