//! REQ-VERIFY-8379: ordered binary64 arithmetic leaves probabilities in Python.
//!
//! These loops retain multiplication, division and addition as separate steps.
//! The build forbids fast-math because changed rounding can change an action.

const KNOTS: [f64; 12] = [0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0];

fn coefficients(c: &[f64]) -> Result<(), &'static str> {
    if c.len() != 34 || c.iter().any(|v| !v.is_finite() || v.abs() > 4.0) {
        return Err("coefficients");
    }
    Ok(())
}

/// Match the original basis stage order so endpoint handling cannot drift.
pub fn design(x: &[f64]) -> Result<Vec<f64>, &'static str> {
    if x.len() != 5
        || x.iter().any(|v| !v.is_finite())
        || x[1..].iter().any(|v| !(0.0..=1.0).contains(v))
    {
        return Err("features");
    }
    let mut output = vec![x[0], 1.0];
    for &v in &x[1..] {
        let mut current = vec![0.0; 11];
        if v == 0.0 || v == 1.0 {
            let mut endpoint = vec![0.0; 8];
            endpoint[if v == 0.0 { 0 } else { 7 }] = 1.0;
            output.extend(endpoint);
            continue;
        }
        for i in 0..11 {
            current[i] = f64::from(KNOTS[i] <= v && v < KNOTS[i + 1]);
        }
        for degree in 1..=3 {
            let mut next = vec![0.0; 11 - degree];
            for i in 0..next.len() {
                let lw = KNOTS[i + degree] - KNOTS[i];
                let rw = KNOTS[i + degree + 1] - KNOTS[i + 1];
                let left = if lw == 0.0 {
                    0.0
                } else {
                    (v - KNOTS[i]) * current[i] / lw
                };
                let right = if rw == 0.0 {
                    0.0
                } else {
                    (KNOTS[i + degree + 1] - v) * current[i + 1] / rw
                };
                next[i] = left + right;
            }
            current = next;
        }
        output.extend(current);
    }
    Ok(output)
}

/// Keep each local dot before the four ordered global additions.
pub fn logits(c: &[f64], rows: &[Vec<f64>], temperature: f64) -> Result<Vec<f64>, &'static str> {
    coefficients(c)?;
    if rows.is_empty() {
        return Err("features");
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("temperature");
    }
    rows.iter()
        .map(|x| {
            let phi = design(x)?;
            let mut z = c[0] * x[0] + c[1];
            for feature in 0..4 {
                let mut local = 0.0;
                for i in 2 + 8 * feature..10 + 8 * feature {
                    local += phi[i] * c[i];
                }
                z += local;
            }
            Ok(z / temperature)
        })
        .collect()
}

/// Python supplies the frozen optimizer residual; Rust never computes expit.
pub fn update(c: &[f64], x: &[f64], residual: f64) -> Result<Vec<f64>, &'static str> {
    coefficients(c)?;
    let phi = design(x)?;
    if !residual.is_finite() {
        return Err("residual");
    }
    let mut gradient = vec![0.0; 34];
    for i in 2..34 {
        gradient[i] = residual * phi[i];
    }
    let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
    let scale = if norm == 0.0 {
        1.0
    } else {
        (1.0 / norm).min(1.0)
    };
    let mut output = c.to_vec();
    for i in 2..34 {
        if gradient[i] != 0.0 {
            output[i] = (c[i] - 0.01 * gradient[i] * scale).clamp(-4.0, 4.0);
        }
    }
    Ok(output)
}
