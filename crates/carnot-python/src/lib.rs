//! carnot-python: PyO3 bindings for the Carnot EBM framework.
//!
//! Exposes Rust EBM implementations to Python with zero-copy numpy
//! array transfer where possible.
//!
//! Spec: REQ-CORE-005, SCENARIO-CORE-005, FR-08

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use carnot_core::{EnergyFunction, Float};
use carnot_ising::{IsingConfig, IsingModel};
use carnot_gibbs::{Activation, GibbsConfig, GibbsModel};
use carnot_boltzmann::{BoltzmannConfig, BoltzmannModel};
use carnot_samplers::{HmcSampler, LangevinSampler, Sampler};

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
    fn energy_batch<'py>(&self, py: Python<'py>, xs: PyReadonlyArray2<Float>) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.energy_batch(&xs.as_array());
        PyArray1::from_owned_array(py, result)
    }

    /// Compute gradient of energy w.r.t. x.
    fn grad_energy<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<Float>) -> Bound<'py, PyArray1<Float>> {
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
    fn new(input_dim: usize, hidden_dims: Vec<usize>, activation: &str, dropout: f64) -> PyResult<Self> {
        let act = match activation {
            "silu" => Activation::SiLU,
            "relu" => Activation::ReLU,
            "tanh" => Activation::Tanh,
            other => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown activation: {other}. Use 'silu', 'relu', or 'tanh'.")
            )),
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

    fn energy_batch<'py>(&self, py: Python<'py>, xs: PyReadonlyArray2<Float>) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.energy_batch(&xs.as_array());
        PyArray1::from_owned_array(py, result)
    }

    fn grad_energy<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<Float>) -> Bound<'py, PyArray1<Float>> {
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
    fn new(input_dim: usize, hidden_dims: Vec<usize>, num_heads: usize, residual: bool) -> PyResult<Self> {
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

    fn energy_batch<'py>(&self, py: Python<'py>, xs: PyReadonlyArray2<Float>) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.energy_batch(&xs.as_array());
        PyArray1::from_owned_array(py, result)
    }

    fn grad_energy<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<Float>) -> Bound<'py, PyArray1<Float>> {
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
        let result = self.inner.sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }

    fn sample_gibbs<'py>(
        &self,
        py: Python<'py>,
        model: &PyGibbsModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }

    fn sample_boltzmann<'py>(
        &self,
        py: Python<'py>,
        model: &PyBoltzmannModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.sample(&model.inner, &init.as_array().to_owned(), n_steps);
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
        let result = self.inner.sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }

    fn sample_gibbs<'py>(
        &self,
        py: Python<'py>,
        model: &PyGibbsModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }

    fn sample_boltzmann<'py>(
        &self,
        py: Python<'py>,
        model: &PyBoltzmannModel,
        init: PyReadonlyArray1<Float>,
        n_steps: usize,
    ) -> Bound<'py, PyArray1<Float>> {
        let result = self.inner.sample(&model.inner, &init.as_array().to_owned(), n_steps);
        PyArray1::from_owned_array(py, result)
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Carnot EBM framework — Python bindings.
#[pymodule]
fn carnot_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Models
    m.add_class::<PyIsingModel>()?;
    m.add_class::<PyGibbsModel>()?;
    m.add_class::<PyBoltzmannModel>()?;

    // Samplers
    m.add_class::<PyLangevinSampler>()?;
    m.add_class::<PyHmcSampler>()?;

    Ok(())
}
