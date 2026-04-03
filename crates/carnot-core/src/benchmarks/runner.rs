//! Benchmark runner: execute benchmarks, measure metrics, record baselines.
//!
//! Spec: REQ-AUTO-001, REQ-AUTO-002

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::{EnergyFunction, Float};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Metrics from a single benchmark run.
///
/// Spec: REQ-AUTO-001
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub benchmark_name: String,
    pub final_energy: Float,
    pub convergence_steps: usize,
    pub wall_clock_seconds: f64,
    pub final_position: Vec<Float>,
    pub energy_gap: Float, // final_energy - known_optimal
}

/// A complete baseline record.
///
/// Spec: REQ-AUTO-002
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineRecord {
    pub version: String,
    pub commit: String,
    pub timestamp: String,
    pub benchmarks: HashMap<String, BenchmarkMetrics>,
}

impl BaselineRecord {
    pub fn new() -> Self {
        Self {
            version: "0.1.0".to_string(),
            commit: String::new(),
            timestamp: String::new(),
            benchmarks: HashMap::new(),
        }
    }

    /// Save baseline to JSON file.
    ///
    /// Spec: REQ-AUTO-002
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load baseline from JSON file.
    ///
    /// Spec: REQ-AUTO-002
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&json).map_err(|e| e.to_string())
    }
}

/// Run a benchmark: minimize an energy function via simple gradient descent
/// and record metrics.
///
/// This is a basic optimizer for benchmarking; the autoresearch loop will
/// test more sophisticated approaches against these baselines.
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
    let start = Instant::now();
    let mut x = initial.clone();
    let mut convergence_steps = max_steps;

    for step in 0..max_steps {
        let grad = energy_fn.grad_energy(&x.view());
        let grad_norm: Float = grad.dot(&grad).sqrt();

        // Convergence check
        if grad_norm < 1e-6 {
            convergence_steps = step;
            break;
        }

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
        energy_gap: final_energy - known_optimal_energy,
    }
}

/// Run all standard benchmarks and return a baseline record.
///
/// Spec: REQ-AUTO-001, REQ-AUTO-002
pub fn run_standard_benchmarks() -> BaselineRecord {
    use super::{Ackley, DoubleWell, GaussianMixture, Rastrigin, Rosenbrock};

    let mut record = BaselineRecord::new();

    // DoubleWell 2D
    let dw = DoubleWell::new(2);
    let info = dw.info();
    let init = Array1::from_vec(vec![0.5, 0.5]);
    let metrics = run_benchmark("double_well_2d", &dw, &init, 0.01, 10000, info.global_min_energy);
    record.benchmarks.insert(metrics.benchmark_name.clone(), metrics);

    // Rosenbrock 2D
    let rb = Rosenbrock::new(2);
    let info = rb.info();
    let init = Array1::from_vec(vec![0.0, 0.0]);
    let metrics = run_benchmark("rosenbrock_2d", &rb, &init, 0.001, 50000, info.global_min_energy);
    record.benchmarks.insert(metrics.benchmark_name.clone(), metrics);

    // Rastrigin 2D
    let ras = Rastrigin::new(2);
    let info = ras.info();
    let init = Array1::from_vec(vec![0.1, 0.1]);
    let metrics = run_benchmark("rastrigin_2d", &ras, &init, 0.001, 10000, info.global_min_energy);
    record.benchmarks.insert(metrics.benchmark_name.clone(), metrics);

    // Ackley 2D
    let ack = Ackley::new(2);
    let info = ack.info();
    let init = Array1::from_vec(vec![0.5, 0.5]);
    let metrics = run_benchmark("ackley_2d", &ack, &init, 0.01, 10000, info.global_min_energy);
    record.benchmarks.insert(metrics.benchmark_name.clone(), metrics);

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
