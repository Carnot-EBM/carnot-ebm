//! REQ-REPORT-8731: retain state while reusing the checked integer kernel.
use super::*;
use pyo3::exceptions::PyOverflowError;

type Audit = (
    Vec<i64>,
    i64,
    BTreeMap<String, u64>,
    BTreeMap<String, u64>,
    u64,
);
type Telemetry = (BTreeMap<String, u64>, BTreeMap<String, u64>, u64);

fn error(message: &'static str) -> PyErr {
    if message == "signed64" {
        PyOverflowError::new_err(message)
    } else {
        PyValueError::new_err(message)
    }
}

#[pyclass]
struct ResidentSparse8731 {
    r: Request,
    pending: bool,
    token: u64,
}

#[pymethods]
impl ResidentSparse8731 {
    #[new]
    fn new(
        parameters: Vec<i64>,
        decay_scale: i64,
        temperature: i64,
        context: &str,
    ) -> PyResult<Self> {
        let f = serde_json::from_str(context).map_err(|_| error("context_contract"))?;
        let mut r = Request {
            f,
            parameters,
            decay_scale,
            temperature,
            ..Request::default()
        };
        execute("serialize", &mut r).map_err(error)?;
        Ok(Self {
            r,
            pending: false,
            token: 0,
        })
    }

    fn quantized(&mut self, value: i32, saturated: bool) {
        *self.r.f.operations.get_mut("saturation_checks").unwrap() += 1;
        *self.r.f.counts.get_mut("saturation").unwrap() += u64::from(saturated);
        let _ = value;
    }

    fn prepare(
        &mut self,
        knots: Vec<i64>,
        scaled: Vec<i64>,
    ) -> PyResult<(Vec<usize>, Vec<i64>, i64, u64)> {
        self.pending = false;
        if knots.iter().any(|x| !(0..=UNIT).contains(x)) || knots.windows(2).any(|w| w[0] > w[1]) {
            return Err(error("supported_domain"));
        }
        let (ids, weights) = basis(&mut self.r.f, &knots, &scaled).map_err(error)?;
        let value = residual(
            &mut self.r.f,
            &self.r.parameters,
            self.r.decay_scale,
            &ids,
            &weights,
        )
        .map_err(error)?;
        self.r.ids = ids.clone();
        self.r.weights = weights.clone();
        self.token = self
            .token
            .checked_add(1)
            .ok_or_else(|| error("token_overflow"))?;
        self.pending = true;
        Ok((ids, weights, value, self.token))
    }

    fn apply(&mut self, probability: i64, label: i64, token: u64) -> PyResult<(i64, usize, usize)> {
        if !(0..=UNIT).contains(&probability) || !(0..=1).contains(&label) {
            return Err(error("supported_domain"));
        }
        if !self.pending {
            return Err(error("pending_probability"));
        }
        if token != self.token {
            return Err(error("late_probability"));
        }
        let (parameters, scale, writes) = update(
            &mut self.r.f,
            &self.r.parameters,
            self.r.decay_scale,
            self.r.temperature,
            probability,
            label,
            &self.r.ids,
            &self.r.weights,
        )
        .map_err(error)?;
        self.r.parameters = parameters;
        self.r.decay_scale = scale;
        self.pending = false;
        Ok((scale, self.r.ids.len(), writes))
    }

    fn telemetry(&self, token: u64) -> PyResult<Telemetry> {
        if token != self.token {
            return Err(error("diagnostic_token"));
        }
        Ok((
            self.r.f.counts.clone(),
            self.r.f.operations.clone(),
            self.r.f.maximum,
        ))
    }

    fn audit(&self, token: u64) -> PyResult<Audit> {
        // Qualification reads state after each update; this diagnostic copy is counted separately.
        let (counts, operations, maximum) = self.telemetry(token)?;
        Ok((
            self.r.parameters.clone(),
            self.r.decay_scale,
            counts,
            operations,
            maximum,
        ))
    }

    fn serialize(&mut self, token: u64) -> PyResult<Vec<u8>> {
        if token != self.token {
            return Err(error("diagnostic_token"));
        }
        let value = execute("serialize", &mut self.r).map_err(error)?;
        Ok(serde_json::from_value(value).unwrap())
    }

    fn control(&mut self, operation: &str, a: i64, b: i64) -> PyResult<i64> {
        // Error probes share the resident counters and the original dispatcher.
        self.r.a = a;
        self.r.b = b;
        let value = execute(operation, &mut self.r).map_err(error)?;
        value.as_i64().ok_or_else(|| error("control_operation"))
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ResidentSparse8731>()
}

#[cfg(test)]
#[path = "spline_resident_8731_tests.rs"]
mod tests;
