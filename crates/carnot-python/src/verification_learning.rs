use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use carnot_core::verification_learning::{UnlabelledData, VerificationLearningProxy, VlConstraint};

#[pyclass(name = "RustVerificationLearningProxy")]
pub struct PyVerificationLearningProxy {
    inner: VerificationLearningProxy,
}

#[pymethods]
impl PyVerificationLearningProxy {
    #[new]
    #[pyo3(signature = (constraints=None))]
    fn new(constraints: Option<&Bound<'_, PyList>>) -> PyResult<Self> {
        let mut rs_constraints = Vec::new();

        if let Some(py_constraints) = constraints {
            for item in py_constraints.iter() {
                if let Ok(dict) = item.downcast::<PyDict>() {
                    let c_type = dict
                        .get_item("type")?
                        .and_then(|v| v.extract::<String>().ok());

                    let c_value = dict
                        .get_item("value")?
                        .and_then(|v| v.extract::<String>().ok())
                        .unwrap_or_else(|| "".to_string());

                    rs_constraints.push(VlConstraint { c_type, c_value });
                }
            }
        }

        Ok(Self {
            inner: VerificationLearningProxy::new(rs_constraints),
        })
    }

    fn score_constraint_satisfaction<'py>(
        &self,
        py: Python<'py>,
        unlabelled_data: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rs_data = parse_unlabelled_data(unlabelled_data)?;
        let scores = self.inner.score_constraint_satisfaction(&rs_data);

        let py_dict = PyDict::new(py);
        for (k, v) in scores {
            py_dict.set_item(k, v)?;
        }

        Ok(py_dict)
    }

    fn compute_proxy_loss(&self, unlabelled_data: &Bound<'_, PyList>) -> PyResult<f64> {
        let rs_data = parse_unlabelled_data(unlabelled_data)?;
        Ok(self.inner.compute_proxy_loss(&rs_data))
    }
}

fn parse_unlabelled_data(py_list: &Bound<'_, PyList>) -> PyResult<Vec<UnlabelledData>> {
    let mut rs_data = Vec::new();
    for item in py_list.iter() {
        if let Ok(dict) = item.downcast::<PyDict>() {
            let id = dict
                .get_item("id")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_else(|| "unknown".to_string());

            let text = dict
                .get_item("text")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_else(|| "".to_string());

            rs_data.push(UnlabelledData { id, text });
        }
    }
    Ok(rs_data)
}

pub fn register_verification_learning_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyVerificationLearningProxy>()?;
    Ok(())
}
