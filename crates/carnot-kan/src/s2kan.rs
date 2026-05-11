use carnot_core::Float;
use ndarray::{Array2, ArrayView2};

#[derive(Debug, Clone)]
pub struct S2KANConfig {
    pub input_dim: usize,
    pub temperature: Float,
}

#[derive(Debug, Clone)]
pub struct S2KANParams {
    /// Shape: (input_dim, 3)
    pub gate_logits: Array2<Float>,
}

#[derive(Debug, Clone)]
pub struct S2KANLayer {
    pub config: S2KANConfig,
    pub params: S2KANParams,
}

impl S2KANLayer {
    pub fn new(config: S2KANConfig, params: S2KANParams) -> Self {
        Self { config, params }
    }

    pub fn evaluate_primitives_single(val: Float) -> [Float; 3] {
        [val.sin(), val.exp(), 1.0 / (1.0 + (-10.0 * val).exp())]
    }

    pub fn gates(&self) -> Array2<Float> {
        let mut gates = self.params.gate_logits.clone();
        gates.mapv_inplace(|v| v / self.config.temperature);
        // softmax along axis=1
        for mut row in gates.rows_mut() {
            let max_val = row.iter().copied().fold(Float::NEG_INFINITY, Float::max);
            let mut sum_exp = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum_exp += *v;
            }
            for v in row.iter_mut() {
                *v /= sum_exp;
            }
        }
        gates
    }

    pub fn forward(&self, x: &ArrayView2<Float>) -> Array2<Float> {
        let gates = self.gates(); // shape: (input_dim, 3)
        let (batch_size, input_dim) = (x.nrows(), x.ncols());
        let mut y = Array2::<Float>::zeros((batch_size, input_dim));
        for b in 0..batch_size {
            for i in 0..input_dim {
                let val = x[[b, i]];
                let prims = Self::evaluate_primitives_single(val);
                let out = prims[0] * gates[[i, 0]] + prims[1] * gates[[i, 1]] + prims[2] * gates[[i, 2]];
                y[[b, i]] = out;
            }
        }
        y
    }
}
