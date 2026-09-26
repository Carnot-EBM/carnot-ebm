//! Narrow PyO3 adapter for the shared durable recalibration service.
//!
//! This file contains no service algorithm. It only converts Python values to
//! the typed Rust core and converts the typed results back to Python tuples.
//! Spec: REQ-REPORT-7626 and SCENARIO-REPORT-7626-COLD.

use carnot_core::portable_recalibration::PortableRecalibrationService;
use carnot_core::record_decision::RecordDecisionService;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::path::PathBuf;

#[pyclass(name = "RustPortableRecalibrationService")]
pub struct PyPortableRecalibrationService {
    inner: PortableRecalibrationService,
    record: Option<RecordDecisionService>,
    record_path: PathBuf,
}

#[pymethods]
impl PyPortableRecalibrationService {
    #[new]
    fn new(state_path: String) -> PyResult<Self> {
        let path = PathBuf::from(state_path);
        let inner =
            PortableRecalibrationService::open(path.clone()).map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            record: None,
            record_path: path,
        })
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

    fn predict_record(
        &mut self,
        event_id: String,
        payload_json: String,
        parameters_json: String,
    ) -> PyResult<(String, f64, String)> {
        if self.record.is_none() {
            self.record = Some(
                RecordDecisionService::open(self.record_path.clone())
                    .map_err(PyValueError::new_err)?,
            );
        }
        self.record
            .as_mut()
            .expect("record opened above")
            .predict(event_id, &payload_json, &parameters_json)
            .map_err(PyValueError::new_err)
    }

    fn release_record_feedback(
        &mut self,
        event_id: String,
        label: i32,
    ) -> (String, bool, bool, Option<String>) {
        if self.record.is_none() {
            match RecordDecisionService::open(self.record_path.clone()) {
                Ok(service) => self.record = Some(service),
                Err(error) => return (event_id, false, false, Some(error)),
            }
        }
        match self.record.as_mut() {
            Some(record) => record.release_feedback(event_id, label),
            None => unreachable!("record opened above"),
        }
    }

    fn record_state_summary(&mut self) -> PyResult<(usize, Vec<String>, Vec<String>, String)> {
        if self.record.is_none() {
            self.record = Some(
                RecordDecisionService::open(self.record_path.clone())
                    .map_err(PyValueError::new_err)?,
            );
        }
        self.record
            .as_ref()
            .map(RecordDecisionService::summary)
            .ok_or_else(|| PyValueError::new_err("record_state_unavailable"))
    }
}

pub fn register_portable_recalibration_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPortableRecalibrationService>()?;
    Ok(())
}
