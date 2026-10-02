//! REQ-PYBIND-8027: explicit construction keeps this prototype opt-in.
use carnot_core::numerical_update_8027::State;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;

#[pyclass(name = "RustNumericalUpdate8027")]
struct NumericalUpdate {
    state: State,
}

#[pymethods]
impl NumericalUpdate {
    #[new]
    fn new(encoded: &str) -> PyResult<Self> {
        let state: State =
            serde_json::from_str(encoded).map_err(|e| PyValueError::new_err(e.to_string()))?;
        state.validate().map_err(PyValueError::new_err)?;
        Ok(Self { state })
    }

    fn design(&self, raw: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        raw.iter()
            .map(|r| self.state.design(r).map_err(PyValueError::new_err))
            .collect()
    }

    /// Timing begins after Python conversion; caller measures the full binding.
    fn update_batch(
        &mut self,
        rows: Vec<Vec<f64>>,
        labels: Vec<Option<u8>>,
    ) -> PyResult<(Vec<f64>, u64, u64)> {
        if rows.len() != labels.len()
            || rows
                .iter()
                .any(|r| r.len() != 110 || r.iter().any(|v| !v.is_finite()))
            || labels.iter().flatten().any(|y| *y > 1)
        {
            return Err(PyValueError::new_err("batch_contract"));
        }
        let began = Instant::now();
        let mut hot = 0;
        let mut probabilities = Vec::with_capacity(rows.len());
        for (x, y) in rows.iter().zip(labels) {
            let p = self.state.probability(x);
            probabilities.push(p);
            if let Some(y) = y {
                let start = Instant::now();
                self.state.step(x, y, p);
                hot += start.elapsed().as_nanos() as u64;
            }
        }
        Ok((probabilities, began.elapsed().as_nanos() as u64, hot))
    }

    fn effective(&self) -> Vec<f64> {
        self.state.effective()
    }

    fn state_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.state).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn numerical_echo_8027(rows: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    rows
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NumericalUpdate>()?;
    m.add_function(wrap_pyfunction!(numerical_echo_8027, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_and_batch_contract() {
        // REQ-PYBIND-8027: typed malformed boundaries never mutate state.
        assert!(NumericalUpdate::new("[]").is_err());
        assert_eq!(numerical_echo_8027(vec![vec![1.0]]), vec![vec![1.0]]);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let m = PyModule::new(py, "_rust").unwrap();
            register(&m).unwrap();
        });
        let encoded = serde_json::json!({
            "arm":"conditioned_energy", "parameters":vec![0.0;110],
            "decay_scale":1.0, "calibration":[0.0,1.0],
            "geometry":{"scaler":{"minimum":vec![0.0;9],"maximum":vec![1.0;9]},
                "logit_center":0.0,"logit_scale":1.0,
                "knots":[0.0,0.0,0.0,0.0,1.0/9.0,2.0/9.0,3.0/9.0,4.0/9.0,
                    5.0/9.0,6.0/9.0,7.0/9.0,8.0/9.0,1.0,1.0,1.0,1.0]}
        })
        .to_string();
        Python::with_gil(|py| {
            let m = PyModule::new(py, "_rust").unwrap();
            register(&m).unwrap();
            let object = m
                .getattr("RustNumericalUpdate8027")
                .unwrap()
                .call1((encoded.as_str(),))
                .unwrap();
            object
                .call_method1("design", (vec![vec![0.5; 9]],))
                .unwrap();
            object
                .call_method1("update_batch", (vec![vec![0.0; 110]], vec![Some(1u8)]))
                .unwrap();
            object.call_method0("effective").unwrap();
            object.call_method0("state_json").unwrap();
        });
        let mut n = NumericalUpdate::new(&encoded).unwrap();
        let before = n.state_json().unwrap();
        assert!(n
            .update_batch(vec![vec![f64::NAN; 110]], vec![Some(1)])
            .is_err());
        assert_eq!(before, n.state_json().unwrap());
        let design = n.design(vec![vec![0.5; 9]]).unwrap();
        assert_eq!(design[0].len(), 110);
        let (p, _, _) = n.update_batch(design, vec![Some(1)]).unwrap();
        assert_eq!(p, vec![0.5]);
        assert_eq!(n.effective().len(), 110);
    }
}
