use std::borrow::Cow;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use enum_extract::extract;
use itertools::Itertools;

use crate::tensor::*;
use crate::device::GPU;

pub fn forward<'a, const N_a: usize, const N_b: usize>(
    gpu: &'a GPU,
    tensor_a: &Tensor<'a, GPU, Strided<N_a>, f32, N_a>,
    tensor_b: &Tensor<'a, GPU, Strided<N_b>, f32, N_b>,
    ) -> Tensor<'a, GPU, Strided<{N_a + N_b}>, f32, {N_a + N_b}> {
        //Unpack tensors
        let Tensor {
            device: gpu_a,
            tensor_layout: Strided {
                strides: tensor_a_strides,
            },
            shape: tensor_a_shape,
            data: tensor_a_data,
        } = tensor_a;

        let Tensor {
            device: gpu_b,
            tensor_layout: Strided {
                strides: tensor_b_strides,
            },
            shape: tensor_b_shape,
            data: tensor_b_data,
        } = tensor_b;

        //Check if tensor devices match
        assert!(!(gpu as *const _ != *gpu_a as *const _), "Tensor device mismatch");
        assert!(!(gpu as *const _ != *gpu_b as *const _), "Tensor device mismatch");

        //Calculate important values for internal use
        let a_len = tensor_a_shape.iter().count();
        let type_size = std::mem::size_of::<f32>();

        //Create meta data values
        let output_tensor_shape = {
            let mut shape = [0 as usize; N_a + N_b];
            for (location, value) in shape.iter_mut().zip(tensor_a_shape.iter().cloned().chain(tensor_b_shape.iter().cloned())) {
                *location = value;
            }
            shape
        };

        let size: usize = output_tensor_shape.iter().product();
        
        let output_tensor_strides = {
            let mut strides = [0 as usize; N_a + N_b];
            let mut current_stride = 1;
            for (location, value) in strides.iter_mut().zip(output_tensor_shape.iter().cloned()).rev() {
                *location = current_stride;
                current_stride = current_stride * value;
            }
            strides
        };
        
        let execution_indexes = {
            let mut labeled_dims: Vec<(usize, usize)> = output_tensor_shape.iter().copied().enumerate().collect();
            labeled_dims.sort_by(|(_, a), (_, b)| b.cmp(a));
            labeled_dims.into_iter().map(|(x, _)| x).take(3).collect::<Vec<usize>>()
        };
        
        let (execution_sizes, execution_strides, execution_a_strides, execution_b_strides) = {
            let mut sizes: [u32; 3] = [1, 1, 1];
            let mut strides: [u32; 3] = [0, 0, 0];
            let mut a_strides: [u32; 3] = [0, 0, 0];
            let mut b_strides: [u32; 3] = [0, 0, 0];

            for (index, key) in execution_indexes.iter().copied().enumerate() {
                sizes[index] = output_tensor_shape[key] as u32;
                strides[index] = output_tensor_strides[key] as u32;
                if key < a_len {
                    a_strides[index] = tensor_a_strides[key] as u32;
                } else {
                    b_strides[index] = tensor_b_strides[key - a_len] as u32;
                }
            }
            
            (sizes, strides, a_strides, b_strides)
        };

        //Load wgsl shader
        let cs_module = gpu.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        //Create bind group layout
        let bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },],
            }
        );

        //Create compute pipeline from shader
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }
        );

        let compute_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: "main",
        });

        //Create command buffer encoder
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Create meta data buffer
        let meta_buffer = {
            let meta_data = OperationMetaData {
                output_strides: execution_strides,
                pad_0: 0,
                tensor_a_strides: execution_a_strides,
                pad_1: 0,
                tensor_b_strides: execution_b_strides,
                pad_2: 0,
            };
            gpu.device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Meta Data Buffer"),
                    contents: bytemuck::bytes_of(&meta_data),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            )
        };

        //Unwrap tensor buffers
        let buffer_a = extract!(TensorData::GPUStrided(_), tensor_a_data).unwrap();
        let buffer_b = extract!(TensorData::GPUStrided(_), tensor_b_data).unwrap();

        //Create buffer for use in output tensor
        let output_buffer = gpu.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Output buffer"),
                size: (type_size * size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );

        //Add instructions to encoder
        let offsets: Vec<(usize, usize, usize)> = {
            let mut info: Vec<(usize, (usize, usize, usize))> = vec![];
            for (index, size) in output_tensor_shape.iter().copied().enumerate() {
                if execution_indexes.contains(&index) {
                    continue;
                }
                if index < a_len {
                    info.push((size, (output_tensor_strides[index], tensor_a_strides[index], 0)));
                }
                else {
                     info.push((size, (output_tensor_strides[index], 0, tensor_b_strides[index])));
                }
            }
            let mut offsets: Vec<(usize, usize, usize)> = info.into_iter()
                .map(|(size, (output_stride, stride_a, stride_b))| (0..size).map(|x| (x * output_stride,x * stride_a, x * stride_b)).collect::<Vec<(usize, usize, usize)>>())
                .multi_cartesian_product().map(|x| x.into_iter().fold((0, 0, 0), |(acc, acc_a, acc_b), (x, x_a, x_b)| (acc + x,acc_a + x_a, acc_b + x_b)))
                .collect();
            if offsets.is_empty() {
                offsets.push((0, 0, 0));
            }
            offsets
        };

        for start_positions in offsets {
            //Create buffer for start position
            let (start_position, start_position_a, start_position_b) = start_positions;
            let offset = Offset {
                offset: start_position as u32,
                offset_a: start_position_a as u32,
                offset_b: start_position_b as u32,
            };
            let start_buffer = gpu.device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Offset buffer"),
                    contents: bytemuck::bytes_of(&offset),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );

            //Create bind group
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: start_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                }],
            });

            //Create compute pass
            let mut compute_pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("Tensor Product"),
                }
            );
            
            //Run compute pass
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch(execution_sizes[0], execution_sizes[1], execution_sizes[2]);
        }

        //Submit operations to gpu
        gpu.queue.submit(Some(encoder.finish()));
        
        //Return
        Tensor {
            device: gpu,
            tensor_layout: Strided {
                strides: output_tensor_strides,
            },
            shape: output_tensor_shape,
            data: TensorData::GPUStrided::<f32>(output_buffer),
        }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OperationMetaData {
    output_strides: [u32; 3],
    pad_0: u32,
    tensor_a_strides: [u32; 3],
    pad_1: u32,
    tensor_b_strides: [u32; 3],
    pad_2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Offset {
    offset: u32,
    offset_a: u32,
    offset_b: u32,
}

