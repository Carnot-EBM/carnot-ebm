//! REQ-VERIFY-8105: bounded arithmetic preserves the public state's provenance.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Geometry {
    mean: Vec<f64>,
    std: Vec<f64>,
    sigma: f64,
}
#[derive(Deserialize)]
struct Center {
    x: Vec<f64>,
}
#[derive(Deserialize)]
struct State {
    geometry: Geometry,
    centers: Vec<Center>,
    coefficients: Vec<f64>,
}

#[pyclass(name = "RustRadial8105")]
struct Radial {
    original: Value,
    state: State,
}

fn invalid() -> PyErr {
    PyValueError::new_err("radial_contract")
}
fn probability(logit: f64) -> f64 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let weight = logit.exp();
        weight / (1.0 + weight)
    }
}

#[pymethods]
impl Radial {
    #[new]
    fn new(encoded: &str) -> PyResult<Self> {
        let original: Value = serde_json::from_str(encoded).map_err(|_| invalid())?;
        let state: State = serde_json::from_value(original.clone()).map_err(|_| invalid())?;
        let g = &state.geometry;
        let d = g.mean.len();
        if ![1, 9].contains(&d)
            || ![16, 20, 24, 28].contains(&state.centers.len())
            || g.std.len() != d
            || g.mean.iter().any(|v| !v.is_finite())
            || g.std.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || !g.sigma.is_finite()
            || g.sigma <= 0.0
            || state
                .centers
                .iter()
                .any(|c| c.x.len() != d || c.x.iter().any(|v| !v.is_finite()))
            || state.coefficients.len() != state.centers.len() + 1
            || state.coefficients.iter().any(|v| !v.is_finite())
        {
            return Err(invalid());
        }
        Ok(Self { original, state })
    }

    fn design(&self, rows: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let g = &self.state.geometry;
        rows.iter()
            .map(|r| {
                if r.len() != g.mean.len() || r.iter().any(|v| !v.is_finite()) {
                    return Err(invalid());
                }
                let z: Vec<f64> = r
                    .iter()
                    .zip(&g.mean)
                    .zip(&g.std)
                    .map(|((x, m), s)| (x - m) / s)
                    .collect();
                if z.iter().any(|v| !v.is_finite()) {
                    return Err(invalid());
                }
                let mut phi = vec![1.0];
                for c in &self.state.centers {
                    // A huge finite separation gives a zero Gaussian without squaring overflow.
                    let norm = z
                        .iter()
                        .zip(&c.x)
                        .fold(0.0_f64, |n, (x, c)| n.hypot((x - c) / g.sigma));
                    phi.push((-0.5 * norm * norm).exp());
                }
                Ok(phi)
            })
            .collect()
    }

    fn predict(&self, rows: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.design(rows)?
            .iter()
            .map(|r| {
                let logit: f64 = r
                    .iter()
                    .zip(&self.state.coefficients)
                    .map(|(x, w)| x * w)
                    .sum();
                if !logit.is_finite() {
                    return Err(invalid());
                }
                Ok(probability(logit))
            })
            .collect()
    }

    fn gradient(&self, rows: Vec<Vec<f64>>, labels: Vec<u8>, ridge: f64) -> PyResult<Vec<f64>> {
        if rows.is_empty()
            || rows.len() != labels.len()
            || labels.iter().any(|y| *y > 1)
            || !ridge.is_finite()
            || ridge <= 0.0
        {
            return Err(invalid());
        }
        let phi = self.design(rows.clone())?;
        let p = self.predict(rows)?;
        let mut grad: Vec<f64> = self.state.coefficients.iter().map(|w| ridge * w).collect();
        for ((r, p), y) in phi.iter().zip(p).zip(labels) {
            for (g, x) in grad.iter_mut().zip(r) {
                *g += x * (p - f64::from(y)) / phi.len() as f64;
            }
        }
        if grad.iter().any(|v| !v.is_finite()) {
            return Err(invalid());
        }
        Ok(grad)
    }

    fn state_json(&self) -> String {
        self.original.to_string()
    }

    fn checkpoint(&self) -> String {
        let payload = self.state_json();
        let sha256 = format!("{:x}", Sha256::digest(payload.as_bytes()));
        serde_json::json!({"payload":payload,"sha256":sha256}).to_string()
    }

    #[staticmethod]
    fn restore(encoded: &str) -> PyResult<Self> {
        let v: Value = serde_json::from_str(encoded).map_err(|_| invalid())?;
        let payload = v["payload"].as_str().ok_or_else(invalid)?;
        if v["sha256"].as_str()
            != Some(format!("{:x}", Sha256::digest(payload.as_bytes())).as_str())
        {
            return Err(invalid());
        }
        Self::new(payload)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Radial>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn req_verify_8105_arithmetic_and_checkpoint() {
        // REQ-VERIFY-8105: nonzero state and invalid operands exercise the safe port.
        pyo3::prepare_freethreaded_python();
        let encoded = serde_json::json!({"geometry":{"mean":[0.0],"std":[1.0],"sigma":1.0},
            "centers":(0..16).map(|i| serde_json::json!({"x":[0.0],"source_id":i})).collect::<Vec<_>>(),
            "coefficients":vec![0.1;17]}).to_string();
        let n = Radial::new(&encoded).unwrap();
        assert_eq!(n.design(vec![vec![0.0]]).unwrap()[0], vec![1.0; 17]);
        assert!(n.predict(vec![vec![0.0]]).unwrap()[0] > 0.5);
        assert_eq!(
            n.gradient(vec![vec![0.0]], vec![1], 0.01).unwrap().len(),
            17
        );
        assert_eq!(
            Radial::restore(&n.checkpoint()).unwrap().state_json(),
            n.state_json()
        );
        assert!(Radial::new("[]").is_err());
        let mut invalid_geometry: Value = serde_json::from_str(&encoded).unwrap();
        invalid_geometry["geometry"]["std"][0] = serde_json::json!(0.0);
        assert!(Radial::new(&invalid_geometry.to_string()).is_err());
        assert!(Radial::restore("{").is_err());
        assert!(Radial::restore("{}").is_err());
        assert!(Radial::restore(r#"{"payload":"{}","sha256":"wrong"}"#).is_err());
        assert!(n.predict(vec![vec![f64::NAN]]).is_err());
        assert!(n.predict(vec![vec![]]).is_err());
        assert!(n.gradient(vec![], vec![], 0.01).is_err());
        let mut v: Value = serde_json::from_str(&encoded).unwrap();
        v["geometry"]["mean"][0] = serde_json::json!(-1e308);
        assert!(Radial::new(&v.to_string())
            .unwrap()
            .design(vec![vec![1e308]])
            .is_err());
        v["geometry"]["mean"][0] = serde_json::json!(0.0);
        v["coefficients"] = serde_json::json!(vec![1e308; 17]);
        let huge = Radial::new(&v.to_string()).unwrap();
        assert!(huge.predict(vec![vec![0.0]]).is_err());
        v["coefficients"] = serde_json::json!([
            1e308, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]);
        assert!(Radial::new(&v.to_string())
            .unwrap()
            .gradient(vec![vec![0.0]], vec![0], 10.0)
            .is_err());
        assert!(probability(-10.0) < 0.5);
        Python::with_gil(|py| {
            let m = PyModule::new(py, "_rust").unwrap();
            register(&m).unwrap();
            let bound = m
                .getattr("RustRadial8105")
                .unwrap()
                .call1((encoded.as_str(),))
                .unwrap();
            bound.call_method1("design", (vec![vec![0.0]],)).unwrap();
            bound.call_method1("predict", (vec![vec![0.0]],)).unwrap();
            bound
                .call_method1("gradient", (vec![vec![0.0]], vec![1u8], 0.01))
                .unwrap();
            bound.call_method0("state_json").unwrap();
            let cp = bound.call_method0("checkpoint").unwrap();
            m.getattr("RustRadial8105")
                .unwrap()
                .call_method1("restore", (cp,))
                .unwrap();
        });
    }
}
