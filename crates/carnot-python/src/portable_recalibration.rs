//! Narrow PyO3 adapter for the shared durable recalibration service.
//!
//! This file contains no service algorithm. It only converts Python values to
//! the typed Rust core and converts the typed results back to Python tuples.
//! Spec: REQ-REPORT-7626 and SCENARIO-REPORT-7626-COLD.

use carnot_core::portable_recalibration::PortableRecalibrationService;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::path::PathBuf;

#[pyclass(name = "RustPortableRecalibrationService")]
pub struct PyPortableRecalibrationService {
    inner: PortableRecalibrationService,
}

#[pymethods]
impl PyPortableRecalibrationService {
    #[new]
    fn new(state_path: String) -> PyResult<Self> {
        let inner = PortableRecalibrationService::open(PathBuf::from(state_path))
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    fn predict(&mut self, event_id: String, probability: f64) -> PyResult<(String, f64, String)> {
        let decision = self
            .inner
            .predict(event_id, probability)
            .map_err(PyValueError::new_err)?;
        Ok((decision.event_id, decision.probability, decision.action))
    }

    fn release_feedback(
        &mut self,
        event_id: String,
        label: i32,
    ) -> (String, bool, bool, Option<String>) {
        let acknowledgment = self.inner.release_feedback(event_id, label);
        (
            acknowledgment.event_id,
            acknowledgment.acknowledged,
            acknowledgment.durable,
            acknowledgment.error,
        )
    }

    fn state_summary(&self) -> PyResult<(usize, Vec<String>, String)> {
        let state = self.inner.state().map_err(PyValueError::new_err)?;
        Ok((state.sample_count, state.processed_event_ids, state.schema))
    }
}

pub fn register_portable_recalibration_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPortableRecalibrationService>()?;
    Ok(())
}
