//! REQ-REPORT-8719: checked CPU arithmetic; host code retains policy and storage.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const UNIT: i64 = 1 << 24;
type Result<T> = std::result::Result<T, &'static str>;

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct Fixed {
    counts: BTreeMap<String, u64>,
    operations: BTreeMap<String, u64>,
    maximum: u64,
}
impl Default for Fixed {
    fn default() -> Self {
        Self {
            counts: [
                "saturation",
                "overflow",
                "decay_underflow",
                "renormalization",
            ]
            .map(|k| (k.to_string(), 0))
            .into(),
            operations: [
                "multiply",
                "divide",
                "saturation_checks",
                "accumulator_checks",
            ]
            .map(|k| (k.to_string(), 0))
            .into(),
            maximum: 0,
        }
    }
}
impl Fixed {
    fn checked(&mut self, n: i128) -> Result<i64> {
        *self.operations.get_mut("accumulator_checks").unwrap() += 1;
        self.maximum = self
            .maximum
            .max(n.unsigned_abs().min(u64::MAX as u128) as u64);
        i64::try_from(n).map_err(|_| {
            *self.counts.get_mut("overflow").unwrap() += 1;
            "signed64"
        })
    }
    fn ratio(&mut self, n: i64, d: i64) -> Result<i64> {
        *self.operations.get_mut("divide").unwrap() += 1;
        if d <= 0 {
            return Err("positive_denominator");
        }
        let n = self.checked(n as i128)?;
        // Euclidean division preserves Python's floor quotient for negative ties.
        let (q, r) = (n.div_euclid(d), n.rem_euclid(d));
        Ok(q + i64::from(2 * r as i128 > d as i128 || (2 * r as i128 == d as i128 && q % 2 != 0)))
    }
    fn sat(&mut self, n: i64) -> i64 {
        *self.operations.get_mut("saturation_checks").unwrap() += 1;
        *self.counts.get_mut("saturation").unwrap() +=
            u64::from(n < i32::MIN as i64 || n > i32::MAX as i64);
        n.clamp(i32::MIN as i64, i32::MAX as i64)
    }
    fn product(&mut self, a: i64, b: i64) -> Result<i64> {
        *self.operations.get_mut("multiply").unwrap() += 1;
        self.checked(a as i128 * b as i128)
    }
    fn mul(&mut self, a: i64, b: i64) -> Result<i64> {
        let product = self.product(a, b)?;
        let rounded = self.ratio(product, UNIT)?;
        Ok(self.sat(rounded))
    }
    fn div(&mut self, a: i64, b: i64) -> Result<i64> {
        let product = self.product(a, UNIT)?;
        let rounded = self.ratio(product, b)?;
        Ok(self.sat(rounded))
    }
}

fn basis(f: &mut Fixed, knots: &[i64], scaled: &[i64]) -> Result<(Vec<usize>, Vec<i64>)> {
    if knots.len() != 16 || scaled.len() != 9 || scaled.iter().any(|x| !(0..=UNIT).contains(x)) {
        return Err("supported_domain");
    }
    let (mut ids, mut weights) = (vec![108], vec![UNIT]);
    for (j, &x) in scaled.iter().enumerate() {
        let span = (4..16)
            .find(|&i| knots[i] > x)
            .map_or(11, |i| i - 1)
            .clamp(3, 11);
        let mut previous = BTreeMap::from([(span, UNIT)]);
        for degree in 1..4 {
            let mut current = BTreeMap::new();
            for i in span - degree..=span {
                let mut n = 0;
                for (operand, width, prior) in [
                    (
                        x - knots[i],
                        knots[i + degree] - knots[i],
                        *previous.get(&i).unwrap_or(&0),
                    ),
                    (
                        knots[i + degree + 1] - x,
                        knots[i + degree + 1] - knots[i + 1],
                        *previous.get(&(i + 1)).unwrap_or(&0),
                    ),
                ] {
                    if width != 0 && prior != 0 {
                        let product = f.product(operand, prior)?;
                        n += f.ratio(product, width)?;
                    }
                }
                current.insert(i, f.sat(n));
            }
            previous = current;
        }
        for (i, weight) in previous {
            if weight != 0 {
                ids.push(j * 12 + i);
                weights.push(weight);
            }
        }
    }
    Ok((ids, weights))
}

fn residual(f: &mut Fixed, p: &[i64], scale: i64, ids: &[usize], weights: &[i64]) -> Result<i64> {
    let mut total = 0;
    for (&i, &w) in ids.iter().zip(weights) {
        let product = f.product(p[i], w)?;
        total = f.checked(total as i128 + product as i128)?;
    }
    let rounded = f.ratio(total, UNIT)?;
    let saturated = f.sat(rounded);
    f.mul(saturated, scale)
}

#[allow(clippy::too_many_arguments)]
fn update(
    f: &mut Fixed,
    p: &[i64],
    old_scale: i64,
    temperature: i64,
    q: i64,
    label: i64,
    ids: &[usize],
    weights: &[i64],
) -> Result<(Vec<i64>, i64, usize)> {
    let factor = f.sat(((1.0_f64 - 0.002 * 0.01) * UNIT as f64).round_ties_even() as i64);
    let mut scale = f.mul(old_scale, factor)?;
    *f.counts.get_mut("decay_underflow").unwrap() += u64::from(scale == old_scale);
    let mut parameters = p.to_vec();
    let mut writes = 0;
    if scale < UNIT / 256 {
        *f.counts.get_mut("renormalization").unwrap() += 1;
        *f.counts.get_mut("decay_underflow").unwrap() += u64::from(scale == 0);
        for v in &mut parameters {
            *v = f.mul(*v, old_scale)?;
        }
        scale = f.sat(factor);
        writes = 109;
    }
    let q = f.sat(q);
    let d1 = f.mul(2 * (q - label * UNIT), q)?;
    let d2 = f.mul(d1, UNIT - q)?;
    let derivative = f.div(d2, temperature)?;
    for (&i, &w) in ids.iter().zip(weights) {
        let rate = f.sat((0.01 * UNIT as f64).round_ties_even() as i64);
        let dw = f.mul(derivative, w)?;
        let lr = f.mul(rate, dw)?;
        let delta = f.div(lr, scale)?;
        parameters[i] = f.sat(parameters[i] - delta);
    }
    Ok((parameters, scale, writes))
}

#[derive(Deserialize)]
#[serde(default)]
struct Request {
    f: Fixed,
    a: i64,
    b: i64,
    parameters: Vec<i64>,
    decay_scale: i64,
    temperature: i64,
    ids: Vec<usize>,
    weights: Vec<i64>,
    knots: Vec<i64>,
    scaled: Vec<i64>,
    probability: i64,
    label: i64,
}
impl Default for Request {
    fn default() -> Self {
        Self {
            f: Fixed::default(),
            a: 0,
            b: 0,
            parameters: vec![],
            decay_scale: UNIT,
            temperature: UNIT,
            ids: vec![],
            weights: vec![],
            knots: vec![],
            scaled: vec![],
            probability: 0,
            label: 0,
        }
    }
}
fn execute(operation: &str, r: &mut Request) -> Result<Value> {
    if r.f.counts.keys().cloned().collect::<Vec<_>>()
        != Fixed::default().counts.keys().cloned().collect::<Vec<_>>()
        || r.f.operations.keys().cloned().collect::<Vec<_>>()
            != Fixed::default()
                .operations
                .keys()
                .cloned()
                .collect::<Vec<_>>()
    {
        return Err("context_contract");
    }
    if ["residual", "update", "serialize"].contains(&operation)
        && (r.parameters.len() != 109
            || r.parameters.iter().any(|&x| i32::try_from(x).is_err())
            || !(2..=UNIT).contains(&r.decay_scale)
            || !(UNIT / 2..=2 * UNIT).contains(&r.temperature)
            || r.ids.len() != r.weights.len()
            || r.ids.len() > 37
            || r.ids.iter().any(|&i| i > 108)
            || r.weights
                .iter()
                .any(|&w| w.unsigned_abs() > (UNIT + 16) as u64))
    {
        return Err("supported_domain");
    }
    let out = match operation {
        "ratio" => json!(r.f.ratio(r.a, r.b)?),
        "product" => json!(r.f.product(r.a, r.b)?),
        "sat" => json!(r.f.sat(r.a)),
        "mul" => json!(r.f.mul(r.a, r.b)?),
        "div" => json!(r.f.div(r.a, r.b)?),
        "basis" => {
            if r.knots.iter().any(|x| !(0..=UNIT).contains(x))
                || r.knots.windows(2).any(|w| w[0] > w[1])
            {
                return Err("supported_domain");
            }
            json!(basis(&mut r.f, &r.knots, &r.scaled)?)
        }
        "residual" => json!(residual(
            &mut r.f,
            &r.parameters,
            r.decay_scale,
            &r.ids,
            &r.weights
        )?),
        "update" => {
            if !(0..=UNIT).contains(&r.probability) || !(0..=1).contains(&r.label) {
                return Err("supported_domain");
            }
            json!(update(
                &mut r.f,
                &r.parameters,
                r.decay_scale,
                r.temperature,
                r.probability,
                r.label,
                &r.ids,
                &r.weights
            )?)
        }
        "serialize" => {
            let mut bytes = Vec::with_capacity(444);
            for &v in r.parameters.iter().chain([&r.decay_scale, &r.temperature]) {
                bytes.extend_from_slice(&(v as i32).to_le_bytes());
            }
            json!(bytes)
        }
        _ => return Err("operation"),
    };
    Ok(out)
}

#[pyfunction]
fn fixed_sparse_8719(operation: &str, payload: &str) -> PyResult<String> {
    let mut r: Request =
        serde_json::from_str(payload).map_err(|_| PyValueError::new_err("request_contract"))?;
    let outcome = execute(operation, &mut r);
    Ok(match outcome {
        Ok(output) => json!({"output": output, "f": r.f, "error": null}),
        Err(error) => json!({"output": null, "f": r.f, "error": error}),
    }
    .to_string())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fixed_sparse_8719, m)?)?;
    #[cfg(feature = "spline-resident-8731")]
    resident::register(m)?;
    Ok(())
}

#[cfg(feature = "spline-resident-8731")]
#[path = "spline_resident_8731.rs"]
mod resident;

#[cfg(test)]
#[path = "spline_fixed_8719_tests.rs"]
mod tests;
