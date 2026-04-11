//! GPU-accelerated Ising model energy computation.
//!
//! Runs the Ising energy E(x) = -0.5 * x^T J x - b^T x on the GPU
//! for batches of input configurations. Each configuration is evaluated
//! in parallel by a separate GPU thread.
//!
//! For batch sizes > ~100, this is significantly faster than CPU (ndarray).
//! For small batches, CPU may be faster due to GPU launch overhead.

use crate::context::{GpuContext, GpuError};
use bytemuck::{Pod, Zeroable};
use carnot_core::Float;

/// GPU-accelerated Ising model.
///
/// Holds the coupling matrix and bias on the GPU. Call `energy_batch()`
/// to evaluate many configurations in a single GPU dispatch.
pub struct IsingGpu {
    coupling_buffer: wgpu::Buffer,
    bias_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    dim: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    dim: u32,
    batch_size: u32,
}

impl IsingGpu {
    /// Create a GPU Ising model from coupling matrix and bias vector.
    ///
    /// # Arguments
    /// * `ctx` - GPU context (device + queue)
    /// * `coupling` - Flattened coupling matrix J, row-major, dim*dim elements
    /// * `bias` - Bias vector b, dim elements
    /// * `dim` - Input dimensionality
    pub fn new(ctx: &GpuContext, coupling: &[Float], bias: &[Float], dim: usize) -> Self {
        use wgpu::util::DeviceExt;

        assert_eq!(coupling.len(), dim * dim, "coupling must be dim*dim");
        assert_eq!(bias.len(), dim, "bias must be dim");

        // Upload coupling matrix to GPU
        let coupling_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("coupling"),
                contents: bytemuck::cast_slice(coupling),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Upload bias vector to GPU
        let bias_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bias"),
                contents: bytemuck::cast_slice(bias),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Params buffer (dim, batch_size) — updated per dispatch
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&Params {
                    dim: dim as u32,
                    batch_size: 0,
                }),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Load WGSL shader
        let shader_source = include_str!("shaders/ising_energy.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ising_energy"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Bind group layout: coupling, bias, inputs, outputs, params
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ising_layout"),
                    entries: &[
                        // coupling matrix
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // bias vector
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // input batch
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // output energies
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // params
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ising_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ising_energy_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            coupling_buffer,
            bias_buffer,
            params_buffer,
            pipeline,
            bind_group_layout,
            dim: dim as u32,
        }
    }

    /// Compute energy for a batch of configurations on the GPU.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `inputs` - Flattened input batch, batch_size * dim elements (row-major)
    ///
    /// # Returns
    /// Vector of energies, one per input configuration.
    pub fn energy_batch(&self, ctx: &GpuContext, inputs: &[Float]) -> Result<Vec<Float>, GpuError> {
        let batch_size = inputs.len() / self.dim as usize;
        assert_eq!(
            inputs.len(),
            batch_size * self.dim as usize,
            "inputs length must be batch_size * dim"
        );

        use wgpu::util::DeviceExt;

        // Upload inputs
        let input_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("inputs"),
                contents: bytemuck::cast_slice(inputs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output buffer
        let output_size = (batch_size * std::mem::size_of::<Float>()) as u64;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_energies"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for readback
        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Update params
        let params = Params {
            dim: self.dim,
            batch_size: batch_size as u32,
        };
        ctx.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ising_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.coupling_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ising_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ising_compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch enough workgroups to cover all samples
            // Workgroup size is 64 (defined in WGSL shader)
            let workgroups = (batch_size as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        ctx.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().map_err(|_| GpuError::BufferMap)?;

        let data = slice.get_mapped_range();
        let energies: Vec<Float> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(energies)
    }

    /// Input dimensionality.
    pub fn dim(&self) -> usize {
        self.dim as usize
    }
}
