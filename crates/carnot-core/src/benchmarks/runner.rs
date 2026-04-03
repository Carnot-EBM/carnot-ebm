//! # Benchmark Runner: Execute, Measure, and Record Baselines
//!
//! **For researchers:** Runs benchmark energy functions through gradient descent, records
//! convergence metrics (energy gap, wall-clock time, convergence steps), and persists
//! baseline records as JSON for regression tracking. Feeds into the autoresearch loop
//! (REQ-AUTO-001, REQ-AUTO-002).
//!
//! **For engineers coming from neural networks:**
//!
//! This module is the "test harness" for the benchmark energy functions. It does three things:
//!
//! 1. **Runs benchmarks** — Takes a benchmark energy function, optimizes it via gradient descent,
//!    and measures how well the optimizer did (final energy, number of steps to converge, wall
//!    clock time).
//!
//! 2. **Records metrics** — Captures the results in a [`BenchmarkMetrics`] struct that includes
//!    the "energy gap" (how far from the known optimal the optimizer got). An energy gap of 0
//!    means perfect convergence.
//!
//! 3. **Saves baselines** — Persists results as JSON files ([`BaselineRecord`]) so that future
//!    runs can compare against them. If a code change makes the Rosenbrock benchmark converge
//!    in 1000 steps instead of 5000, that's a measurable improvement. If it regresses from
//!    energy gap 0.001 to 0.5, something broke.
//!
//! **How this feeds into the autoresearch loop:**
//!
//! The autoresearch system (REQ-AUTO-001) works like this:
//! 1. Run all benchmarks with the current algorithm -> get baseline metrics
//! 2. Propose a new algorithm variant (e.g., different step size schedule, new sampler)
//! 3. Run all benchmarks with the new variant -> get new metrics
//! 4. Compare: did the new variant beat the baseline on any benchmark?
//! 5. If yes, adopt it as the new baseline. If no, discard it.
//!
//! This is essentially automated hyperparameter tuning and algorithm selection, but driven
//! by rigorous benchmarks with known answers rather than noisy real-world metrics.
//!
//! Spec: REQ-AUTO-001, REQ-AUTO-002

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::{EnergyFunction, Float};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Metrics captured from a single benchmark run.
///
/// **For researchers:** Quantitative results from optimizing a benchmark function:
/// final energy, convergence speed, and the gap to the known global optimum.
///
/// **For engineers:** After running a benchmark, this struct tells you everything about
/// how well the optimizer performed. The most important field is `energy_gap` — the
/// difference between the energy the optimizer achieved and the known global minimum.
/// An energy gap of 0.0 means the optimizer found the exact minimum. An energy gap of
/// 5.0 means it got stuck somewhere significantly suboptimal.
///
/// For example, running the DoubleWell benchmark might produce:
/// ```text
/// BenchmarkMetrics {
///     benchmark_name: "double_well_2d",
///     final_energy: 0.0001,        // very close to the known minimum of 0.0
///     convergence_steps: 347,       // converged after 347 gradient steps
///     wall_clock_seconds: 0.012,    // took 12 milliseconds
///     final_position: [0.9998, 0.001], // very close to the known minimum [1, 0]
///     energy_gap: 0.0001,           // 0.0001 - 0.0 = 0.0001 (excellent!)
/// }
/// ```
///
/// Spec: REQ-AUTO-001
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Name identifying which benchmark was run (e.g., "double_well_2d", "rosenbrock_2d").
    pub benchmark_name: String,
    /// The energy value at the optimizer's final position. Lower is better.
    /// Compare against the benchmark's `global_min_energy` to assess quality.
    pub final_energy: Float,
    /// Number of gradient descent steps taken before convergence (or max_steps if it didn't converge).
    /// Convergence is detected when the gradient norm drops below 1e-6.
    pub convergence_steps: usize,
    /// Wall-clock time for the entire optimization run, in seconds.
    /// Useful for comparing computational cost of different approaches.
    pub wall_clock_seconds: f64,
    /// The position (in input space) where the optimizer stopped.
    /// Compare against the benchmark's `global_min_location`.
    pub final_position: Vec<Float>,
    /// The gap between the achieved energy and the known global minimum:
    /// `energy_gap = final_energy - known_optimal_energy`.
    /// Zero means perfect optimization. This is the primary metric for the autoresearch loop.
    pub energy_gap: Float,
}

/// A complete baseline record: a snapshot of benchmark results at a specific code version.
///
/// **For researchers:** Versioned collection of benchmark metrics for regression tracking.
/// Serializable to JSON for persistence across sessions and CI runs.
///
/// **For engineers:** Think of this as a "report card" for a particular version of the code.
/// It stores the results of running ALL benchmarks, tagged with the version and git commit.
/// When you make a change and re-run benchmarks, you compare the new `BaselineRecord` against
/// the previous one to detect improvements or regressions.
///
/// For example, you might have:
/// - `baseline_v0.1.0.json` with energy_gap 0.05 on Rosenbrock
/// - After improving the optimizer: `baseline_v0.2.0.json` with energy_gap 0.001 on Rosenbrock
/// - This proves the improvement is real and quantifiable
///
/// The baselines are stored as JSON files and can be loaded back for comparison.
///
/// Spec: REQ-AUTO-002
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineRecord {
    /// Semantic version of the code that produced these results (e.g., "0.1.0").
    pub version: String,
    /// Git commit hash for exact reproducibility.
    pub commit: String,
    /// ISO 8601 timestamp of when the benchmarks were run.
    pub timestamp: String,
    /// Map from benchmark name to its metrics. Keys are strings like "double_well_2d",
    /// "rosenbrock_2d", etc.
    pub benchmarks: HashMap<String, BenchmarkMetrics>,
}

impl Default for BaselineRecord {
    fn default() -> Self {
        Self {
            version: "0.1.0".to_string(),
            commit: String::new(),
            timestamp: String::new(),
            benchmarks: HashMap::new(),
        }
    }
}

impl BaselineRecord {
    /// Create a new empty baseline record with default version "0.1.0".
    pub fn new() -> Self {
        Self::default()
    }

    /// Save this baseline record to a JSON file at the given path.
    ///
    /// **For engineers:** The JSON is pretty-printed for human readability. You can
    /// inspect the file directly to see benchmark results, diff it in git to track
    /// changes over time, or load it programmatically for automated comparison.
    ///
    /// Spec: REQ-AUTO-002
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load a baseline record from a JSON file.
    ///
    /// **For engineers:** Use this to load a previous baseline for comparison. The typical
    /// workflow is: load old baseline, run new benchmarks, compare energy gaps.
    ///
    /// Spec: REQ-AUTO-002
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&json).map_err(|e| e.to_string())
    }
}

/// Run a single benchmark: minimize an energy function via gradient descent and record metrics.
///
/// **For researchers:** Simple steepest-descent optimizer with fixed step size and gradient-norm
/// convergence criterion (threshold 1e-6). Intentionally basic — the autoresearch loop tests
/// more sophisticated approaches against these gradient-descent baselines.
///
/// **For engineers coming from neural networks:**
///
/// This function is the core benchmark executor. It takes an energy function and tries to
/// minimize it using the simplest possible optimizer: vanilla gradient descent with a fixed
/// learning rate. The algorithm is intentionally simple because it serves as a *baseline* —
/// the autoresearch loop will try fancier approaches (adaptive step sizes, momentum, different
/// MCMC samplers) and compare them against this baseline.
///
/// The optimization loop works exactly like training a neural network with SGD:
/// 1. Compute the gradient of the energy at the current position
/// 2. Check if the gradient is small enough to declare convergence (norm < 1e-6)
/// 3. If not converged, update: `x = x - step_size * gradient`
/// 4. Repeat until converged or max_steps reached
///
/// For example:
/// ```ignore
/// let dw = DoubleWell::new(2);
/// let init = Array1::from_vec(vec![0.5, 0.5]);
/// let metrics = run_benchmark("double_well_2d", &dw, &init, 0.01, 10000, 0.0);
/// println!("Converged in {} steps, energy gap = {}", metrics.convergence_steps, metrics.energy_gap);
/// ```
///
/// Spec: REQ-AUTO-001
pub fn run_benchmark(
    name: &str,
    energy_fn: &dyn EnergyFunction,
    initial: &Array1<Float>,
    step_size: Float,
    max_steps: usize,
    known_optimal_energy: Float,
) -> BenchmarkMetrics {
    // Start the wall-clock timer for performance measurement.
    let start = Instant::now();
    let mut x = initial.clone();
    // Default to max_steps; will be overwritten if we converge early.
    let mut convergence_steps = max_steps;

    for step in 0..max_steps {
        // Compute the gradient of the energy function at the current position.
        // This tells us the direction of steepest ascent; we'll move in the opposite direction.
        let grad = energy_fn.grad_energy(&x.view());

        // Compute the L2 norm (magnitude) of the gradient vector.
        // When this is very small, the optimizer has reached a (local) minimum or saddle point.
        let grad_norm: Float = grad.dot(&grad).sqrt();

        // Convergence check: if the gradient is essentially zero, we've found a stationary point.
        // The threshold 1e-6 is standard for single-precision (f32) optimization.
        if grad_norm < 1e-6 {
            convergence_steps = step;
            break;
        }

        // Gradient descent update: move in the direction of steepest descent (negative gradient).
        // This is identical to SGD in neural network training, but without minibatches.
        x = &x - step_size * &grad;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let final_energy = energy_fn.energy(&x.view());

    BenchmarkMetrics {
        benchmark_name: name.to_string(),
        final_energy,
        convergence_steps,
        wall_clock_seconds: elapsed,
        final_position: x.to_vec(),
        // The energy gap is the key metric: how far from optimal did we end up?
        // For a perfect optimizer on a convex problem, this should be ~0.
        // For multimodal problems (like Rastrigin), gradient descent often gets stuck
        // in a local minimum, resulting in a positive energy gap.
        energy_gap: final_energy - known_optimal_energy,
    }
}

/// Run the standard benchmark suite and return a complete baseline record.
///
/// **For researchers:** Executes DoubleWell-2D, Rosenbrock-2D, Rastrigin-2D, and Ackley-2D
/// with hand-tuned step sizes and iteration budgets. Returns a `BaselineRecord` suitable for
/// persistence and regression comparison.
///
/// **For engineers:** This is the "run all benchmarks" convenience function. It creates each
/// benchmark with sensible parameters, runs gradient descent on each one, and packages all
/// the results into a single `BaselineRecord`. The step sizes and max_steps are hand-tuned
/// per benchmark:
///
/// | Benchmark | Step size | Max steps | Why |
/// |-----------|-----------|-----------|-----|
/// | DoubleWell 2D | 0.01 | 10,000 | Moderate landscape, converges easily |
/// | Rosenbrock 2D | 0.001 | 50,000 | Narrow valley needs small steps and patience |
/// | Rastrigin 2D | 0.001 | 10,000 | Small steps to avoid overshooting local minima |
/// | Ackley 2D | 0.01 | 10,000 | Moderate step size, relies on starting near origin |
///
/// The returned record does NOT have `commit` or `timestamp` filled in — the caller should
/// populate those before saving.
///
/// Spec: REQ-AUTO-001, REQ-AUTO-002
pub fn run_standard_benchmarks() -> BaselineRecord {
    use super::{Ackley, DoubleWell, Rastrigin, Rosenbrock};

    let mut record = BaselineRecord::new();

    // --- DoubleWell 2D ---
    // Starting at [0.5, 0.5], which is between the two minima at x[0]=+/-1.
    // Gradient descent will converge to whichever minimum the gradient points toward
    // from the starting position (in this case, x[0]=+1 since 0.5 > 0).
    let dw = DoubleWell::new(2);
    let info = dw.info();
    let init = Array1::from_vec(vec![0.5, 0.5]);
    let metrics = run_benchmark(
        "double_well_2d",
        &dw,
        &init,
        0.01,
        10000,
        info.global_min_energy,
    );
    record
        .benchmarks
        .insert(metrics.benchmark_name.clone(), metrics);

    // --- Rosenbrock 2D ---
    // Starting at [0, 0], which is inside the valley but far from the minimum at [1, 1].
    // The small step size (0.001) and high iteration budget (50,000) are needed because
    // the Rosenbrock valley floor has very gentle slope — progress is slow.
    let rb = Rosenbrock::new(2);
    let info = rb.info();
    let init = Array1::from_vec(vec![0.0, 0.0]);
    let metrics = run_benchmark(
        "rosenbrock_2d",
        &rb,
        &init,
        0.001,
        50000,
        info.global_min_energy,
    );
    record
        .benchmarks
        .insert(metrics.benchmark_name.clone(), metrics);

    // --- Rastrigin 2D ---
    // Starting at [0.1, 0.1], very close to the global minimum at the origin.
    // This tests whether gradient descent can reach the exact minimum without
    // being pulled into the nearest local minimum at [1, 0] or [0, 1].
    let ras = Rastrigin::new(2);
    let info = ras.info();
    let init = Array1::from_vec(vec![0.1, 0.1]);
    let metrics = run_benchmark(
        "rastrigin_2d",
        &ras,
        &init,
        0.001,
        10000,
        info.global_min_energy,
    );
    record
        .benchmarks
        .insert(metrics.benchmark_name.clone(), metrics);

    // --- Ackley 2D ---
    // Starting at [0.5, 0.5], which is near but not at the global minimum at the origin.
    // The bumpy surface means gradient descent may not reach exactly zero, but should get close.
    let ack = Ackley::new(2);
    let info = ack.info();
    let init = Array1::from_vec(vec![0.5, 0.5]);
    let metrics = run_benchmark(
        "ackley_2d",
        &ack,
        &init,
        0.01,
        10000,
        info.global_min_energy,
    );
    record
        .benchmarks
        .insert(metrics.benchmark_name.clone(), metrics);

    record
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::DoubleWell;

    #[test]
    fn test_run_benchmark_converges() {
        // REQ-AUTO-001: benchmark runner finds minimum
        let dw = DoubleWell::new(2);
        // Start near a minimum to ensure convergence within budget
        let init = Array1::from_vec(vec![0.8, 0.1]);
        let metrics = run_benchmark("test_dw", &dw, &init, 0.01, 10000, 0.0);

        assert!(
            metrics.final_energy < 0.1,
            "Should converge near minimum, got energy={}",
            metrics.final_energy
        );
        assert!(metrics.wall_clock_seconds >= 0.0);
    }

    #[test]
    fn test_run_standard_benchmarks() {
        // REQ-AUTO-001, REQ-AUTO-002: standard suite runs and produces baselines
        let record = run_standard_benchmarks();
        assert!(record.benchmarks.len() >= 4);
        assert!(record.benchmarks.contains_key("double_well_2d"));
        assert!(record.benchmarks.contains_key("rosenbrock_2d"));
        assert!(record.benchmarks.contains_key("rastrigin_2d"));
        assert!(record.benchmarks.contains_key("ackley_2d"));

        // DoubleWell should converge close to 0
        let dw_metrics = &record.benchmarks["double_well_2d"];
        assert!(
            dw_metrics.final_energy < 0.1,
            "DoubleWell should converge, got energy={}",
            dw_metrics.final_energy
        );
    }

    #[test]
    fn test_baseline_save_load_roundtrip() {
        // REQ-AUTO-002: baselines can be persisted and loaded
        let record = run_standard_benchmarks();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("baseline.json");

        record.save(&path).unwrap();
        let loaded = BaselineRecord::load(&path).unwrap();

        assert_eq!(loaded.benchmarks.len(), record.benchmarks.len());
        for (name, metrics) in &record.benchmarks {
            let loaded_metrics = &loaded.benchmarks[name];
            assert!(
                (metrics.final_energy - loaded_metrics.final_energy).abs() < 1e-6,
                "Energy mismatch for {name}"
            );
        }
    }
}
