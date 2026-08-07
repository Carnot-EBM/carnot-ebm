//! carnot-python: PyO3 bindings for the Carnot EBM framework.
//!
//! Exposes Rust EBM implementations to Python with zero-copy numpy
//! array transfer where possible.
//!
//! Spec: REQ-CORE-005, SCENARIO-CORE-005, FR-08

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

mod adaptive_state;
mod kv260;
mod mode_jump;
mod one_axis_tempering;
mod pipeline;
mod s2kan;
mod verification_learning;

use carnot_boltzmann::{
    soft_bellman_solve as rust_soft_bellman_solve,
    soft_bellman_solve_path as rust_soft_bellman_solve_path, BoltzmannConfig, BoltzmannModel,
    SoftBellmanSolution,
};
use carnot_core::{math::gec::project_gradient, EnergyFunction, Float};
use carnot_gibbs::{Activation, GibbsConfig, GibbsModel};
use carnot_ising::{IsingConfig, IsingModel};
use carnot_samplers::{HmcSampler, LangevinSampler, Sampler};

fn soft_bellman_solution_to_dict<'py>(
    py: Python<'py>,
    solution: SoftBellmanSolution,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("soft_values", solution.soft_values)?;
    d.set_item("immediate_rewards", solution.immediate_rewards)?;
    d.set_item("token_energies", solution.token_energies)?;
    d.set_item("sequence_affinity", solution.sequence_affinity)?;
    d.set_item("sequence_energy", solution.sequence_energy)?;
    d.set_item("log_partition", solution.log_partition)?;
    d.set_item("log_probability", solution.log_probability)?;
    d.set_item(
        "max_abs_bellman_residual",
        solution.max_abs_bellman_residual,
    )?;
    Ok(d)
}

/// Solve the Soft Bellman ARM-to-EBM mapping for token logprobs.
///
/// Pass either a flat list of chosen token logprobs, or normalized logprob rows
/// plus `token_ids` to select the generated path.
///
/// Spec: REQ-INFER-2056
#[pyfunction(name = "soft_bellman_solve")]
#[pyo3(signature = (logprobs, token_ids=None))]
fn py_soft_bellman_solve<'py>(
    py: Python<'py>,
    logprobs: &Bound<'_, PyAny>,
    token_ids: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyDict>> {
    let solution = match token_ids {
        Some(token_ids) => {
            let rows = logprobs.extract::<Vec<Vec<Float>>>().map_err(|_| {
                PyValueError::new_err(
                    "logprobs must be a list of logprob rows when token_ids is provided",
                )
            })?;
            rust_soft_bellman_solve_path(&rows, &token_ids)
        }
        None => {
            let flat = logprobs.extract::<Vec<Float>>().map_err(|_| {
                PyValueError::new_err("logprobs must be a flat list unless token_ids is provided")
            })?;
            rust_soft_bellman_solve(&flat)
        }
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    soft_bellman_solution_to_dict(py, solution)
}

// ---------------------------------------------------------------------------
// Ising Model
// ---------------------------------------------------------------------------

/// Ising (small tier) Energy Based Model — Rust implementation.
///
/// E(x) = -0.5 * x^T J x - b^T x
///
/// Spec: REQ-CORE-005, REQ-TIER-001
#[pyclass(name = "RustIsingModel")]
struct PyIsingModel {
    inner: IsingModel,
}

#[pymethods]
impl PyIsingModel {
    #[new]
    #[pyo3(signature = (input_dim=784, coupling_init="xavier_uniform"))]
    fn new(input_dim: usize, coupling_init: &str) -> PyResult<Self> {
        let config = IsingConfig {
            input_dim,
            hidden_dim: None,
            coupling_init: coupling_init.to_string(),
        };
        let model = IsingModel::new(config)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: model })
    }

    /// Compute scalar energy for input x.
    fn energy(&self, x: PyReadonlyArray1<Float>) -> Float {
        self.inner.energy(&x.as_array())
    }

    /// Compute energy for a batch of inputs.
    fn energy_batch<'py>(
        &self,
        py: Python<'py>,
        xs: PyReadonlyArray2<Float>,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.energy_batch(&xs.as_array());
        PyArray1::from_owned_array(py, result)
    }

    /// Compute gradient of energy w.r.t. x.
    fn grad_energy<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<Float>,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.grad_energy(&x.as_array());
        PyArray1::from_owned_array(py, result)
    }

    /// Number of input dimensions.
    fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }

    /// Parameter memory in bytes.
    fn parameter_memory_bytes(&self) -> usize {
        self.inner.parameter_memory_bytes()
    }
}

// ---------------------------------------------------------------------------
// Gibbs Model
// ---------------------------------------------------------------------------

/// Gibbs (medium tier) Energy Based Model — Rust implementation.
///
/// Multi-layer energy network with analytical backprop.
///
/// Spec: REQ-CORE-005, REQ-TIER-002
#[pyclass(name = "RustGibbsModel")]
struct PyGibbsModel {
    inner: GibbsModel,
}

#[pymethods]
impl PyGibbsModel {
    #[new]
    #[pyo3(signature = (input_dim=784, hidden_dims=vec![512, 256], activation="silu", dropout=0.0))]
    fn new(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        activation: &str,
        dropout: f64,
    ) -> PyResult<Self> {
        let act = match activation {
            "silu" => Activation::SiLU,
            "relu" => Activation::ReLU,
            "tanh" => Activation::Tanh,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown activation: {other}. Use 'silu', 'relu', or 'tanh'."
                )))
            }
        };
        let config = GibbsConfig {
            input_dim,
            hidden_dims,
            activation: act,
            dropout,
        };
        let model = GibbsModel::new(config)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: model })
    }

    fn energy(&self, x: PyReadonlyArray1<Float>) -> Float {
        self.inner.energy(&x.as_array())
    }

    fn energy_batch<'py>(
        &self,
        py: Python<'py>,
        xs: PyReadonlyArray2<Float>,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.energy_batch(&xs.as_array());
        PyArray1::from_owned_array(py, result)
    }

    fn grad_energy<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<Float>,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.grad_energy(&x.as_array());
        PyArray1::from_owned_array(py, result)
    }

    fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }
}

// ---------------------------------------------------------------------------
// Boltzmann Model
// ---------------------------------------------------------------------------

/// Boltzmann (large tier) Energy Based Model — Rust implementation.
///
/// Deep residual energy network with analytical backprop.
///
/// Spec: REQ-CORE-005, REQ-TIER-003
#[pyclass(name = "RustBoltzmannModel")]
struct PyBoltzmannModel {
    inner: BoltzmannModel,
}

#[pymethods]
impl PyBoltzmannModel {
    #[new]
    #[pyo3(signature = (input_dim=784, hidden_dims=vec![1024, 512, 256, 128], num_heads=4, residual=true))]
    fn new(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        num_heads: usize,
        residual: bool,
    ) -> PyResult<Self> {
        let config = BoltzmannConfig {
            input_dim,
            hidden_dims,
            num_heads,
            residual,
            layer_norm: false,
        };
        let model = BoltzmannModel::new(config)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: model })
    }

    fn energy(&self, x: PyReadonlyArray1<Float>) -> Float {
        self.inner.energy(&x.as_array())
    }

    fn energy_batch<'py>(
        &self,
        py: Python<'py>,
        xs: PyReadonlyArray2<Float>,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.energy_batch(&xs.as_array());
        PyArray1::from_owned_array(py, result)
    }

    fn grad_energy<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<Float>,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.grad_energy(&x.as_array());
        PyArray1::from_owned_array(py, result)
    }

    fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }
}

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------

/// Langevin Dynamics sampler — Rust implementation.
///
/// Spec: REQ-CORE-005, REQ-SAMPLE-001
#[pyclass(name = "RustLangevinSampler")]
struct PyLangevinSampler {
    inner: LangevinSampler,
}

#[pymethods]
impl PyLangevinSampler {
    #[new]
    #[pyo3(signature = (step_size=0.01))]
    fn new(step_size: Float) -> Self {
        Self {
            inner: LangevinSampler::new(step_size),
        }
    }

    /// Sample from an energy model. Model must be a Rust model (RustIsingModel, etc).
    fn sample_ising<'py>(
        &self,
        py: Python<'py>,
        model: &PyIsingModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self
            .inner
            .sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }

    fn sample_gibbs<'py>(
        &self,
        py: Python<'py>,
        model: &PyGibbsModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self
            .inner
            .sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }

    fn sample_boltzmann<'py>(
        &self,
        py: Python<'py>,
        model: &PyBoltzmannModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self
            .inner
            .sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }
}

/// HMC sampler — Rust implementation.
///
/// Spec: REQ-CORE-005, REQ-SAMPLE-002
#[pyclass(name = "RustHMCSampler")]
struct PyHmcSampler {
    inner: HmcSampler,
}

#[pymethods]
impl PyHmcSampler {
    #[new]
    #[pyo3(signature = (step_size=0.1, num_leapfrog_steps=10))]
    fn new(step_size: Float, num_leapfrog_steps: usize) -> Self {
        Self {
            inner: HmcSampler::new(step_size, num_leapfrog_steps),
        }
    }

    fn sample_ising<'py>(
        &self,
        py: Python<'py>,
        model: &PyIsingModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self
            .inner
            .sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }

    fn sample_gibbs<'py>(
        &self,
        py: Python<'py>,
        model: &PyGibbsModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self
            .inner
            .sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }

    fn sample_boltzmann<'py>(
        &self,
        py: Python<'py>,
        model: &PyBoltzmannModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self
            .inner
            .sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }
}

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

/// Projects a gradient onto the feasible region defined by a reference gradient and epsilon.
///
/// Spec: REQ-FR11-1683
#[pyfunction(name = "project_gradient")]
#[pyo3(signature = (grad, ref_grad, epsilon=0.0))]
fn py_project_gradient<'py>(
    py: Python<'py>,
    grad: PyReadonlyArray1<Float>,
    ref_grad: PyReadonlyArray1<Float>,
    epsilon: Float,
) -> Bound<'py, PyArray1<Float>> {
    let result = project_gradient(grad.as_array(), ref_grad.as_array(), epsilon);
    PyArray1::from_owned_array(py, result)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Carnot EBM framework — Python bindings.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Math
    m.add_function(wrap_pyfunction!(py_project_gradient, m)?)?;

    // Models
    m.add_function(wrap_pyfunction!(py_soft_bellman_solve, m)?)?;
    m.add_class::<PyIsingModel>()?;
    m.add_class::<PyGibbsModel>()?;
    m.add_class::<PyBoltzmannModel>()?;

    // Samplers
    m.add_class::<PyLangevinSampler>()?;
    m.add_class::<PyHmcSampler>()?;

    // Pipeline (verification)
    pipeline::register_pipeline_module(m)?;

    // S2KAN
    s2kan::register_s2kan_module(m)?;

    // Verification learning
    verification_learning::register_verification_learning_module(m)?;

    // Adaptive-state microkernel
    adaptive_state::register_adaptive_state_module(m)?;

    // KV260 Potts hardware sampler
    kv260::register_kv260_module(m)?;

    // One-axis corrected-cDLS replica exchange
    one_axis_tempering::register_one_axis_tempering_module(m)?;

    // Fixed Exp6166/Exp6180 mode-jump sampler
    mode_jump::register_mode_jump_module(m)?;

    Ok(())
}
