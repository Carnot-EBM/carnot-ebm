//! Benchmark: GPU vs CPU for Ising batch energy computation.
//!
//! Run with: cargo run -p carnot-gpu --example bench_gpu_vs_cpu --release

use carnot_gpu::{GpuContext, IsingGpu};
use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() {
    let ctx = GpuContext::new().expect("No GPU available");
    println!("GPU: {} ({:?})", ctx.device_name(), ctx.backend());
    println!();

    for dim in [10, 50, 100, 500] {
        for batch_size in [10, 100, 1000, 10000] {
            bench(dim, batch_size, &ctx);
        }
        println!();
    }
}

fn bench(dim: usize, batch_size: usize, ctx: &GpuContext) {
    // Create random-ish coupling matrix and bias
    let coupling: Vec<f32> = (0..dim * dim)
        .map(|i| ((i * 17 + 3) % 100) as f32 / 100.0 - 0.5)
        .collect();
    let bias: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();

    // Create random inputs
    let inputs: Vec<f32> = (0..batch_size * dim)
        .map(|i| ((i * 13 + 7) % 100) as f32 / 100.0)
        .collect();

    // GPU benchmark
    let ising_gpu = IsingGpu::new(ctx, &coupling, &bias, dim);

    // Warm up GPU
    let _ = ising_gpu.energy_batch(ctx, &inputs);

    let start = Instant::now();
    let gpu_energies = ising_gpu.energy_batch(ctx, &inputs).unwrap();
    let gpu_time = start.elapsed();

    // CPU benchmark
    let j = Array2::from_shape_vec((dim, dim), coupling).unwrap();
    let b = Array1::from_vec(bias);

    let start = Instant::now();
    let mut cpu_energies = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let x = ndarray::ArrayView1::from(&inputs[i * dim..(i + 1) * dim]);
        let jx = j.dot(&x);
        let e = -0.5 * x.dot(&jx) - b.dot(&x);
        cpu_energies.push(e);
    }
    let cpu_time = start.elapsed();

    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

    println!(
        "dim={:4}, batch={:5}: GPU {:8.2}ms, CPU {:8.2}ms, speedup {:.1}x",
        dim,
        batch_size,
        gpu_time.as_secs_f64() * 1000.0,
        cpu_time.as_secs_f64() * 1000.0,
        speedup,
    );

    // Verify correctness
    let max_err: f32 = gpu_energies
        .iter()
        .zip(cpu_energies.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.1, "GPU/CPU mismatch: max_err={}", max_err);
}
