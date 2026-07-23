use carnot_core::adaptive_state::{
    AdaptiveEvent, AdaptiveStateKernel, OperationResult, MAX_EVENT_ID_LEN, MAX_REASON_LEN,
    MAX_REPLAY_LIMIT,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

/// Bounded adaptive-state microkernel exposed through PyO3.
///
/// Spec: REQ-LEARN-5859, SCENARIO-LEARN-5859-OPERATION-PARITY.
#[pyclass(name = "RustAdaptiveStateKernel")]
pub struct PyAdaptiveStateKernel {
    inner: AdaptiveStateKernel,
}

#[pymethods]
impl PyAdaptiveStateKernel {
    #[new]
    #[pyo3(signature = (capacity=8, history_capacity=32))]
    fn new(capacity: u32, history_capacity: u32) -> PyResult<Self> {
        let inner =
            AdaptiveStateKernel::new(capacity, history_capacity).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn restore(checkpoint: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bytes = checkpoint.extract::<Vec<u8>>()?;
        let inner = AdaptiveStateKernel::restore(&bytes).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    fn apply_event<'py>(
        &mut self,
        py: Python<'py>,
        event: &Bound<'_, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = match parse_event(event, &self.inner) {
            Ok(event) => self.inner.apply_event(event),
            Err(result) => result,
        };
        result_to_dict(py, result)
    }

    fn acquire_core<'py>(
        &mut self,
        py: Python<'py>,
        event_id: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        result_to_dict(py, self.inner.acquire_core(&event_id))
    }

    fn quarantine<'py>(
        &mut self,
        py: Python<'py>,
        event_id: String,
        reason_code: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        if reason_code.len() > MAX_REASON_LEN {
            return result_to_dict(py, reject(&self.inner, "INVALID_REASON"));
        }
        result_to_dict(py, self.inner.quarantine(&event_id, &reason_code))
    }

    fn promote<'py>(&mut self, py: Python<'py>, event_id: String) -> PyResult<Bound<'py, PyDict>> {
        result_to_dict(py, self.inner.promote(&event_id))
    }

    fn select_replay<'py>(&self, py: Python<'py>, limit: i64) -> PyResult<Bound<'py, PyDict>> {
        if limit < 0 {
            return result_to_dict(py, reject(&self.inner, "INVALID_REPLAY_LIMIT"));
        }
        if limit > i64::from(MAX_REPLAY_LIMIT) {
            return result_to_dict(py, reject(&self.inner, "REPLAY_LIMIT_EXCEEDED"));
        }
        result_to_dict(py, self.inner.select_replay(limit as u32))
    }

    fn roll_back<'py>(&mut self, py: Python<'py>, version_id: i64) -> PyResult<Bound<'py, PyDict>> {
        result_to_dict(py, self.inner.roll_back(version_id))
    }

    fn serialize<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.serialize())
    }

    fn canonical_state_json(&self) -> String {
        self.inner.canonical_state_json()
    }

    fn canonical_state_hash(&self) -> String {
        self.inner.canonical_state_hash()
    }

    fn version_id(&self) -> u32 {
        self.inner.version_id()
    }
}

fn result_to_dict(py: Python<'_>, result: OperationResult) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("accepted", result.accepted)?;
    dict.set_item("code", result.code)?;
    dict.set_item("state_hash", result.state_hash)?;
    dict.set_item("version_id", result.version_id)?;
    if let Some(selected) = result.selected_replay {
        dict.set_item("selected_replay", selected)?;
    }
    Ok(dict)
}

fn parse_event(
    event: &Bound<'_, PyDict>,
    kernel: &AdaptiveStateKernel,
) -> Result<AdaptiveEvent, OperationResult> {
    if event.len() != 6 {
        return Err(reject(kernel, "MALFORMED_EVENT"));
    }
    let event_id =
        extract_string(event, "event_id").map_err(|_| reject(kernel, "MALFORMED_EVENT"))?;
    if event_id.is_empty() || event_id.len() > MAX_EVENT_ID_LEN {
        return Err(reject(kernel, "INVALID_EVENT_ID"));
    }
    let change = extract_string(event, "change").map_err(|_| reject(kernel, "MALFORMED_EVENT"))?;
    let signature_hash =
        extract_string(event, "signature_hash").map_err(|_| reject(kernel, "MALFORMED_EVENT"))?;
    let payload_hash =
        extract_string(event, "payload_hash").map_err(|_| reject(kernel, "MALFORMED_EVENT"))?;
    let chronology_index =
        extract_i64(event, "chronology_index").map_err(|_| reject(kernel, "MALFORMED_EVENT"))?;
    let confidence_q16 =
        extract_i64(event, "confidence_q16").map_err(|_| reject(kernel, "MALFORMED_EVENT"))?;
    if chronology_index < 0
        || chronology_index > i64::from(u32::MAX)
        || confidence_q16 < 0
        || confidence_q16 > i64::from(u16::MAX)
    {
        return Err(reject(kernel, "FIXED_WIDTH_OVERFLOW"));
    }
    Ok(AdaptiveEvent {
        change,
        chronology_index: chronology_index as u32,
        confidence_q16: confidence_q16 as u16,
        event_id,
        payload_hash,
        signature_hash,
    })
}

fn extract_string(event: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    event
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract::<String>()
}

fn extract_i64(event: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    event
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract::<i64>()
}

fn reject(kernel: &AdaptiveStateKernel, code: &str) -> OperationResult {
    OperationResult {
        accepted: false,
        code: code.to_string(),
        state_hash: kernel.canonical_state_hash(),
        version_id: kernel.version_id(),
        selected_replay: None,
    }
}

pub fn register_adaptive_state_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyAdaptiveStateKernel>()?;
    Ok(())
}
