//! REQ-PYBIND-8027: tests precede the calibrated numerical primitive.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_update_restart() {
        // SCENARIO-PYBIND-8027-PARITY: decay survives JSON round trip.
        let mut s = State::fixture();
        let mut x = vec![0.0; 110];
        x[0] = 1.0;
        let p = s.probability(&x);
        s.step(&x, 1, p);
        let restored: State = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(s.effective(), restored.effective());
        assert!((s.decay_scale - 0.99998).abs() < 1e-15);
    }

    #[test]
    fn test_valid_state_and_negative_probability() {
        // REQ-PYBIND-8027: valid state and negative stable logits execute fully.
        let mut s = State::fixture();
        s.validate().unwrap();
        s.calibration = [-20.0, 1.0];
        let x = s.design(&[0.0; 9]).unwrap();
        assert!(s.probability(&x) > 0.0);
        assert!(s.probability(&x) < 1e-7);
        s.geometry.scaler.maximum[0] = 0.0;
        assert!(s.design(&[0.0; 9]).unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_basis_boundaries_and_nonfinite() {
        // REQ-PYBIND-8027: endpoints retain partition of unity; NaN fails.
        let s = State::fixture();
        for value in [0.0, 1.0 / 9.0, 0.5, 1.0] {
            let x = s.design(&[value; 9]).unwrap();
            for j in 0..9 {
                assert!((x[2 + j * 12..2 + (j + 1) * 12].iter().sum::<f64>() - 1.0).abs() < 1e-12);
            }
        }
        assert!(s.design(&[f64::NAN; 9]).is_err());
        let mut invalid = s;
        invalid.decay_scale = 0.0;
        assert!(invalid.validate().is_err());
    }
}

use serde::{Deserialize, Serialize};

/// Float64 state retains the multiplier so a restart does not alter regularization.
#[derive(Clone, Serialize, Deserialize)]
pub struct State {
    pub arm: String,
    pub parameters: Vec<f64>,
    pub decay_scale: f64,
    pub calibration: [f64; 2],
    pub geometry: Geometry,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Geometry {
    pub scaler: Scaler,
    pub logit_center: f64,
    pub logit_scale: f64,
    pub knots: Vec<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Scaler {
    pub minimum: Vec<f64>,
    pub maximum: Vec<f64>,
}

impl State {
    /// Reject malformed state before any arithmetic can mutate coefficients.
    pub fn validate(&self) -> Result<(), String> {
        let g = &self.geometry;
        let finite = self
            .parameters
            .iter()
            .chain(&self.calibration)
            .chain(&g.knots)
            .chain(&g.scaler.minimum)
            .chain(&g.scaler.maximum)
            .all(|v| v.is_finite());
        if !finite
            || self.arm != "conditioned_energy"
            || self.parameters.len() != 110
            || !self.decay_scale.is_finite()
            || self.decay_scale <= 0.0
            || !g.logit_center.is_finite()
            || !g.logit_scale.is_finite()
            || g.logit_scale <= 0.0
            || g.scaler.minimum.len() != 9
            || g.scaler.maximum.len() != 9
            || g.knots != knots()
        {
            return Err("state_contract".into());
        }
        Ok(())
    }

    pub fn effective(&self) -> Vec<f64> {
        self.parameters
            .iter()
            .map(|v| v * self.decay_scale)
            .collect()
    }

    /// Cox recursion uses the same clamped cubic knots as the Python design.
    pub fn design(&self, raw: &[f64]) -> Result<Vec<f64>, String> {
        if raw.len() != 9 || raw.iter().any(|v| !v.is_finite()) {
            return Err("feature_contract".into());
        }
        let q = raw[0].clamp(1e-4, 1.0 - 1e-4);
        let g = &self.geometry;
        let mut out = vec![1.0, ((q / (1.0 - q)).ln() - g.logit_center) / g.logit_scale];
        for (j, value) in raw.iter().enumerate() {
            let lo = g.scaler.minimum[j];
            let hi = g.scaler.maximum[j];
            let span = if hi > lo { hi - lo } else { 1.0 };
            let x = ((value - lo) / span).clamp(0.0, 1.0);
            let mut b = [0.0; 15];
            for (i, cell) in b.iter_mut().enumerate() {
                *cell = f64::from(g.knots[i] <= x && x < g.knots[i + 1]);
            }
            for degree in 1..=3 {
                for i in 0..15 - degree {
                    let left = g.knots[i + degree] - g.knots[i];
                    let right = g.knots[i + degree + 1] - g.knots[i + 1];
                    b[i] = if left > 0.0 {
                        (x - g.knots[i]) / left * b[i]
                    } else {
                        0.0
                    } + if right > 0.0 {
                        (g.knots[i + degree + 1] - x) / right * b[i + 1]
                    } else {
                        0.0
                    };
                }
            }
            if x == 1.0 {
                b[..12].fill(0.0);
                b[11] = 1.0;
            }
            out.extend_from_slice(&b[..12]);
        }
        Ok(out)
    }

    pub fn probability(&self, x: &[f64]) -> f64 {
        let z = self.calibration[0]
            + self.calibration[1]
                * x.iter()
                    .zip(&self.parameters)
                    .map(|(x, p)| x * p * self.decay_scale)
                    .sum::<f64>();
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let v = z.exp();
            v / (1.0 + v)
        }
    }

    /// One scalar decay covers every coefficient; data writes stay local.
    pub fn step(&mut self, x: &[f64], y: u8, p: f64) {
        let scale = self.decay_scale * (1.0 - 2.0 * 0.001 * 0.01);
        for (value, coefficient) in x.iter().zip(&mut self.parameters) {
            if *value != 0.0 {
                *coefficient -= 0.01 * (p - f64::from(y)) * self.calibration[1] * value / scale;
            }
        }
        self.decay_scale = scale;
    }

    #[cfg(test)]
    fn fixture() -> Self {
        Self {
            arm: "conditioned_energy".into(),
            parameters: vec![0.01; 110],
            decay_scale: 1.0,
            calibration: [0.2, 1.3],
            geometry: Geometry {
                scaler: Scaler {
                    minimum: vec![0.0; 9],
                    maximum: vec![1.0; 9],
                },
                logit_center: 0.0,
                logit_scale: 1.0,
                knots: knots(),
            },
        }
    }
}

fn knots() -> Vec<f64> {
    let mut k = vec![0.0; 4];
    k.extend((1..9).map(|i| f64::from(i) / 9.0));
    k.extend([1.0; 4]);
    k
}
