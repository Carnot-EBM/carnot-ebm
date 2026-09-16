//! In-process schedule evaluation for the retained acquired constraints.
//!
//! Python owns request dictionaries. This module copies constraint terms once,
//! converts each batch directly, and calls the existing exact Rust evaluator.
//! It does not serialize JSON or start a helper process inside the batch call.
//!
//! Spec: REQ-PYBIND-7339 and SCENARIO-PYBIND-7339-*.

use carnot_constraints::{
    evaluate_schedule, evaluate_schedule_batch, ScheduleAssignment, ScheduleConstraint,
    ScheduleEvaluation, ScheduleRequest, ScheduleTermEnergy,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyList};

fn invalid(error: impl Into<String>) -> ScheduleEvaluation {
    ScheduleEvaluation {
        valid_input: false,
        feasible: false,
        complete_oracle_certificate: false,
        total_energy: None,
        terms: Vec::new(),
        error: Some(error.into()),
    }
}

fn item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    dict.get_item(key)
}

fn optional_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    Ok(item(dict, key)?.and_then(|value| value.extract::<String>().ok()))
}

fn optional_i64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i64>> {
    Ok(item(dict, key)?.and_then(|value| {
        if value.is_instance_of::<PyBool>() {
            None
        } else {
            value.extract::<i64>().ok()
        }
    }))
}

fn compile_constraints(
    value: &Bound<'_, PyAny>,
) -> PyResult<(Vec<ScheduleConstraint>, Option<String>)> {
    let Ok(items) = value.downcast::<PyList>() else {
        return Ok((Vec::new(), Some("invalid_constraints".to_string())));
    };
    let mut constraints = Vec::with_capacity(items.len());
    for value in items.iter() {
        let Ok(term) = value.downcast::<PyDict>() else {
            return Ok((Vec::new(), Some("invalid_constraint".to_string())));
        };
        constraints.push(ScheduleConstraint {
            kind: optional_string(term, "kind")?.unwrap_or_default(),
            constraint_id: optional_string(term, "constraint_id")?.unwrap_or_default(),
            version: optional_string(term, "version")?,
            left: optional_string(term, "left")?,
            right: optional_string(term, "right")?,
            minimum: optional_i64(term, "minimum")?,
            window_size: optional_i64(term, "window_size")?,
            maximum: optional_i64(term, "maximum")?,
        });
    }
    Ok((constraints, None))
}

fn required_i64(
    dict: &Bound<'_, PyDict>,
    key: &str,
    error_field: &str,
) -> PyResult<Result<i64, ScheduleEvaluation>> {
    Ok(match optional_i64(dict, key)? {
        Some(value) => Ok(value),
        None => Err(invalid(format!("invalid_integer:{error_field}"))),
    })
}

fn request_from_python(
    value: &Bound<'_, PyAny>,
    constraints: &[ScheduleConstraint],
) -> PyResult<Result<ScheduleRequest, ScheduleEvaluation>> {
    let Ok(request) = value.downcast::<PyDict>() else {
        return Ok(Err(invalid("invalid_request")));
    };
    let schema = optional_string(request, "schema")?.unwrap_or_default();
    if schema != carnot_constraints::SCHEDULE_REQUEST_SCHEMA {
        return Ok(Err(invalid("invalid_schema")));
    }
    let executor_version = optional_string(request, "executor_version")?.unwrap_or_default();
    if executor_version.is_empty() {
        return Ok(Err(invalid("empty_executor_version")));
    }
    let slot_min = match required_i64(request, "slot_min", "slot_min")? {
        Ok(value) => value,
        Err(error) => return Ok(Err(error)),
    };
    let slot_max = match required_i64(request, "slot_max", "slot_max")? {
        Ok(value) => value,
        Err(error) => return Ok(Err(error)),
    };
    if slot_min > slot_max {
        return Ok(Err(invalid("invalid_slot_domain")));
    }
    let Some(schedule_value) = item(request, "schedule")? else {
        return Ok(Err(invalid("empty_schedule")));
    };
    let Ok(schedule_items) = schedule_value.downcast::<PyList>() else {
        return Ok(Err(invalid("empty_schedule")));
    };
    if schedule_items.is_empty() {
        return Ok(Err(invalid("empty_schedule")));
    }
    let mut schedule = Vec::with_capacity(schedule_items.len());
    for value in schedule_items.iter() {
        let Ok(assignment) = value.downcast::<PyDict>() else {
            return Ok(Err(invalid("invalid_assignment")));
        };
        let activity = optional_string(assignment, "activity")?.unwrap_or_default();
        if activity.is_empty() {
            return Ok(Err(invalid("empty_activity")));
        }
        let slot = match required_i64(assignment, "slot", &format!("schedule:{activity}"))? {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        schedule.push(ScheduleAssignment { activity, slot });
    }
    Ok(Ok(ScheduleRequest {
        schema,
        executor_version,
        slot_min,
        slot_max,
        schedule,
        constraints: constraints.to_vec(),
    }))
}

fn term_to_dict<'py>(py: Python<'py>, term: &ScheduleTermEnergy) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("constraint_id", &term.constraint_id)?;
    result.set_item("kind", &term.kind)?;
    result.set_item("energy", term.energy)?;
    result.set_item("satisfied", term.satisfied)?;
    Ok(result)
}

fn evaluation_to_dict<'py>(
    py: Python<'py>,
    evaluation: &ScheduleEvaluation,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("valid_input", evaluation.valid_input)?;
    result.set_item("feasible", evaluation.feasible)?;
    result.set_item(
        "complete_oracle_certificate",
        evaluation.complete_oracle_certificate,
    )?;
    result.set_item("total_energy", evaluation.total_energy)?;
    let terms = evaluation
        .terms
        .iter()
        .map(|term| term_to_dict(py, term))
        .collect::<PyResult<Vec<_>>>()?;
    result.set_item("terms", terms)?;
    result.set_item("error", &evaluation.error)?;
    Ok(result)
}

/// Immutable acquired constraints with a direct ordered batch boundary.
///
/// The constructor copies each term into Rust-owned storage. A later change to
/// the caller's list cannot change evaluation. Each call allocates detached
/// result dictionaries so Python can safely mutate or release them.
#[pyclass(name = "RustCompiledScheduleEvaluator", frozen)]
pub(crate) struct PyCompiledScheduleEvaluator {
    constraints: Vec<ScheduleConstraint>,
    compile_error: Option<String>,
}

#[pymethods]
impl PyCompiledScheduleEvaluator {
    #[new]
    fn new(constraints: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (constraints, compile_error) = compile_constraints(constraints)?;
        Ok(Self {
            constraints,
            compile_error,
        })
    }

    /// Evaluate one ordered batch without JSON, IPC, or Python fallback.
    fn evaluate_batch<'py>(
        &self,
        py: Python<'py>,
        requests: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let request_items = requests
            .downcast::<PyList>()
            .map_err(|_| PyValueError::new_err("invalid_requests"))?;
        let mut parsed = Vec::with_capacity(request_items.len());
        let mut positions = Vec::with_capacity(request_items.len());
        let mut evaluations = vec![None; request_items.len()];
        for (index, value) in request_items.iter().enumerate() {
            match request_from_python(&value, &self.constraints)? {
                Ok(request) => {
                    if let Some(error) = &self.compile_error {
                        let mut probe = request.clone();
                        probe.constraints.clear();
                        let dynamic = evaluate_schedule(&probe);
                        evaluations[index] = Some(if dynamic.valid_input {
                            invalid(error.clone())
                        } else {
                            dynamic
                        });
                    } else {
                        positions.push(index);
                        parsed.push(request);
                    }
                }
                Err(error) => evaluations[index] = Some(error),
            }
        }
        for (index, evaluation) in positions.into_iter().zip(evaluate_schedule_batch(&parsed)) {
            evaluations[index] = Some(evaluation);
        }
        evaluations
            .iter()
            .map(|evaluation| {
                evaluation_to_dict(
                    py,
                    evaluation
                        .as_ref()
                        .expect("every batch position receives one evaluation"),
                )
            })
            .collect()
    }
}

pub(crate) fn register_schedule_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCompiledScheduleEvaluator>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::invalid;

    // REQ-PYBIND-7339 / SCENARIO-PYBIND-7339-ROUNDTRIP
    #[test]
    fn invalid_boundary_result_never_looks_feasible() {
        let result = invalid("invalid_integer:slot_min");
        assert!(!result.valid_input);
        assert!(!result.feasible);
        assert!(!result.complete_oracle_certificate);
        assert_eq!(result.total_energy, None);
        assert!(result.terms.is_empty());
        assert_eq!(result.error.as_deref(), Some("invalid_integer:slot_min"));
    }
}
