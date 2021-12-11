use std::borrow::Cow;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use enum_extract::extract;
use itertools::Itertools;

use crate::tensor::*;
use crate::device::GPU;

pub fn forward<'a, const N: usize>(tensor_a: &Tensor<'a, GPU, Strided<N>, f32, N>, tensor_b: &Tensor<'a, GPU, Strided<N>, f32, N>) -> Tensor<'a, GPU, Strided<N>, f32, N> {
        //Unpack tensors
        let Tensor {
            device: gpu,
            tensor_layout: Strided {
                strides: tensor_a_strides,
            },
            shape: tensor_a_shape,
            data: tensor_a_data,
        } = tensor_a;

        let Tensor {
            device: _,
            tensor_layout: Strided {
                strides: tensor_b_strides,
            },
            shape: tensor_b_shape,
            data: tensor_b_data,
        } = tensor_b;

        //Check if tensor shapes match
        assert!(!(tensor_a_shape != tensor_b_shape), "Tensor shape mismatch");

        //Create meta data values
        let size: usize = tensor_a_shape.iter().product();
        let type_size = std::mem::size_of::<f32>();

        let output_tensor_shape = tensor_a_shape.clone();
        let output_tensor_strides = tensor_a_strides.clone();

        let execution_indexes = {
            let mut labeled_dims: Vec<(usize, usize)> = tensor_a_shape.iter().copied().enumerate().collect();
            labeled_dims.sort_by(|(_, a), (_, b)| b.cmp(a));
            labeled_dims.into_iter().map(|(x, _)| x).take(3).collect::<Vec<usize>>()
        };
        
        let (execution_sizes, execution_strides, execution_a_strides, execution_b_strides) = {
            let mut sizes: [u32; 3] = [1, 1, 1,];
            let mut strides: [u32; 3] = [0,0,0];
            let mut a_strides: [u32; 3] = [0,0,0];
            let mut b_strides: [u32; 3] = [0,0,0];

            for (index, key) in execution_indexes.iter().copied().enumerate() {
                sizes[index] = output_tensor_shape[key] as u32;
                strides[index] = output_tensor_strides[key] as u32;
                a_strides[index] = tensor_a_strides[key] as u32;
                b_strides[index] = tensor_b_strides[key] as u32;
            }

            println!("{:?}", sizes);
            println!("{:?}", strides);
            println!("{:?}", a_strides);
            println!("{:?}", b_strides);
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
        let offsets: Vec<usize> = {
            let mut info: Vec<(usize, usize)> = vec![];
            for (index, size) in output_tensor_shape.iter().copied().enumerate() {
                if execution_indexes.contains(&index) {
                    continue;
                }
                info.push((size, output_tensor_strides[index]));
            }
            let mut offsets: Vec<usize> = info.into_iter()
                .map(|(size, stride)| (0..size).map(|x| x * stride).collect::<Vec<usize>>())
                .multi_cartesian_product().map(|x| x.into_iter().sum()).collect();
            if offsets.is_empty() {
                offsets.push(0);
            }
            offsets
        };

        for start_position in offsets {
            //Create buffer for start position
            println!("{}", start_position);
            let offset = Offset {
                offset: start_position as u32,
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
                    label: Some("Elementwise Addition"),
                }
            );
            
            //Run compute pass
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            println!("{:?}", execution_sizes);
            compute_pass.dispatch(execution_sizes[0], execution_sizes[1], execution_sizes[2]);
        }

        //Submit operations to gpu
        gpu.queue.submit(Some(encoder.finish()));
        
        //Return
        Tensor {
            device: <&GPU>::clone(gpu),
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
}

