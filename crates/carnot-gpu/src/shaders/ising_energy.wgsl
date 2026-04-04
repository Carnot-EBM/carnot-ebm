// WGSL compute shader: batch Ising model energy computation
//
// The Ising energy function is:
//   E(x) = -0.5 * x^T J x - b^T x
//
// where J is the coupling matrix (input_dim x input_dim) and b is the bias vector.
// This shader computes E(x_i) for each sample x_i in a batch, in parallel.
//
// Binding layout:
//   @group(0) @binding(0) - coupling matrix J (flattened row-major, dim*dim f32)
//   @group(0) @binding(1) - bias vector b (dim f32)
//   @group(0) @binding(2) - input batch X (flattened, batch_size * dim f32)
//   @group(0) @binding(3) - output energies (batch_size f32)
//   @group(0) @binding(4) - params: [dim, batch_size]

@group(0) @binding(0) var<storage, read> coupling: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> inputs: array<f32>;
@group(0) @binding(3) var<storage, read_write> energies: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_idx = global_id.x;
    let dim = params[0];
    let batch_size = params[1];

    if (sample_idx >= batch_size) {
        return;
    }

    let x_offset = sample_idx * dim;

    // Compute -0.5 * x^T J x
    var quadratic: f32 = 0.0;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        var jx_i: f32 = 0.0;
        for (var j: u32 = 0u; j < dim; j = j + 1u) {
            jx_i = jx_i + coupling[i * dim + j] * inputs[x_offset + j];
        }
        quadratic = quadratic + inputs[x_offset + i] * jx_i;
    }

    // Compute -b^T x
    var linear: f32 = 0.0;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        linear = linear + bias[i] * inputs[x_offset + i];
    }

    // E(x) = -0.5 * x^T J x - b^T x
    energies[sample_idx] = -0.5 * quadratic - linear;
}
