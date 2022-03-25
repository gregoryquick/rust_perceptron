use wgpu::util::{BufferInitDescriptor, DeviceExt};
use enum_extract::extract;
use itertools::Itertools;

use crate::tensor::*;
use crate::device::GPU;

mod pipelines;

pub fn forward<'a, const N: usize>(
    gpu: &'a GPU,
    tensor_a: &Tensor<'a, GPU, Strided<N>, f32, N>,
    tensor_b: &Tensor<'a, GPU, Strided<N>, f32, N>
    ) -> Tensor<'a, GPU, Strided<N>, f32, N> {
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
        assert!(gpu as *const _ == *gpu_a as *const _, "Tensor device mismatch");
        assert!(gpu as *const _ == *gpu_b as *const _, "Tensor device mismatch");

        //Check if tensor shapes are compatable 
        assert!(tensor_a_shape == tensor_b_shape, "Tensor shape mismatch");

        //Calculate important values for internal use
        let type_size = std::mem::size_of::<f32>();

        //Create meta data values
        let output_tensor_shape = tensor_a_shape.clone();

        let size: usize = output_tensor_shape.iter().product();
        
        let output_tensor_strides = tensor_a_strides.clone();

        let execution_indexes = {
            let mut labeled_dims: Vec<(usize, usize)> = output_tensor_shape.iter().copied().enumerate().collect();
            labeled_dims.sort_by(|(_, a), (_, b)| b.cmp(a));
            labeled_dims.into_iter().map(|(x, _)| x).take(3).collect::<Vec<usize>>()
        };
        
        let (execution_sizes, execution_strides, execution_a_strides, execution_b_strides) = {
            let mut sizes: [u32; 3] = [1, 1, 1,];
            let mut strides: [u32; 3] = [0, 0, 0];
            let mut a_strides: [u32; 3] = [0, 0, 0];
            let mut b_strides: [u32; 3] = [0, 0, 0];

            for (index, key) in execution_indexes.iter().copied().enumerate() {
                sizes[index] = output_tensor_shape[key] as u32;
                strides[index] = output_tensor_strides[key] as u32;
                a_strides[index] = tensor_a_strides[key] as u32;
                b_strides[index] = tensor_b_strides[key] as u32;
            }

            (sizes, strides, a_strides, b_strides)
        };

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
                info.push((size, (output_tensor_strides[index], tensor_a_strides[index], tensor_b_strides[index])));
            }
            let mut offsets: Vec<(usize, usize, usize)> = info.into_iter()
                .map(|(size, (output_stride, stride_a, stride_b))| (0..size).map(|x| (x * output_stride, x * stride_a, x * stride_b)).collect::<Vec<(usize, usize, usize)>>())
                .multi_cartesian_product().map(|x| x.into_iter().fold((0, 0, 0), |(acc, acc_a, acc_b), (x, x_a, x_b)| (acc + x,acc_a + x_a, acc_b + x_b)))
                .collect();
            if offsets.is_empty() {
                offsets.push((0, 0, 0));
            }
            offsets
        };

        for start_positions in offsets {
            //Create buffer for start positions
            let (start_position, start_position_a, start_position_b) = start_positions;
            let offset = Offset {
                offset: start_position as u32,
                offset_a: start_position as u32,
                offset_b: start_position as u32,
            };
            let start_buffer = gpu.device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Offset buffer"),
                    contents: bytemuck::bytes_of(&offset),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            );

            //Create pipeline and binding layout
            let (bind_group_layout, compute_pipeline) = pipelines::forward_pipeline(gpu);

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

pub fn backward<'a, const N: usize>(
    gpu: &'a GPU,
    tangent: &Tensor<'a, GPU, Strided<N>, f32, N>,
    tensor_a: &Tensor<'a, GPU, Strided<N>, f32, N>,
    tensor_b: &Tensor<'a, GPU, Strided<N>, f32, N>
    ) -> (Tensor<'a, GPU, Strided<N>, f32, N>, Tensor<'a, GPU, Strided<N>, f32, N>) {
        //Unpack tensors
        let Tensor {
            device: tangent_gpu,
            tensor_layout: Strided {
                strides: tangent_strides,
            },
            shape: tangent_shape,
            data: tangent_data,
        } = tangent;
        let Tensor {
            device: tensor_a_gpu,
            tensor_layout: Strided {
                strides: tensor_a_strides,
            },
            shape: tensor_a_shape,
            data: tensor_a_data,
        } = tensor_a;
        let Tensor {
            device: tensor_b_gpu,
            tensor_layout: Strided {
                strides: tensor_b_strides,
            },
            shape: tensor_b_shape,
            data: tensor_b_data,
        } = tensor_b;


        //Check if tensor devices match
        assert!(gpu as *const _ == *tangent_gpu as *const _, "Tensor device mismatch");
        assert!(gpu as *const _ == *tensor_a_gpu as *const _, "Tensor device mismatch");
        assert!(gpu as *const _ == *tensor_b_gpu as *const _, "Tensor device mismatch");

        //Check if tensor shapes are compatable 
        assert!(tangent_shape == tensor_a_shape, "Tensor shape mismatch");
        assert!(tangent_shape == tensor_b_shape, "Tensor shape mismatch");

        //Create command buffer encoder
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Calculate important values for internal use
        let type_size = std::mem::size_of::<f32>();

        //Create meta data values
        let output_tensor_shape = tensor_a_shape.clone();

        let size: usize = output_tensor_shape.iter().product();
        
        let output_tensor_strides = tangent_strides.clone();

        let execution_indexes = {
            let mut labeled_dims: Vec<(usize, usize)> = output_tensor_shape.iter().copied().enumerate().collect();
            labeled_dims.sort_by(|(_, a), (_, b)| b.cmp(a));
            labeled_dims.into_iter().map(|(x, _)| x).take(3).collect::<Vec<usize>>()
        };
        
        let (execution_sizes, execution_strides, execution_tangent_strides) = {
            let mut sizes: [u32; 3] = [1, 1, 1,];
            let mut strides: [u32; 3] = [0, 0, 0];
            let mut t_strides: [u32; 3] = [0, 0, 0];
  
            for (index, key) in execution_indexes.iter().copied().enumerate() {
                sizes[index] = output_tensor_shape[key] as u32;
                strides[index] = output_tensor_strides[key] as u32;
                t_strides[index] = tangent_strides[key] as u32;
            }

            (sizes, strides, t_strides)
        };

        //Shared
            //Unwrap tangent buffer
            let tangent_buffer = extract!(TensorData::GPUStrided(_), tangent_data).unwrap();
            //Create meta data buffer
            let meta_buffer = {
                let meta_data = PullbackMetaData {
                    output_strides: execution_strides,
                    pad_0: 0,
                    tensor_strides: execution_tangent_strides,
                    pad_1: 0,
                };
                gpu.device.create_buffer_init(
                    &BufferInitDescriptor {
                        label: Some("Meta Data Buffer"),
                        contents: bytemuck::bytes_of(&meta_data),
                        usage: wgpu::BufferUsages::STORAGE,
                    }
                )
            };


        //Tensor a
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
            let offsets: Vec<(usize, usize)> = {
                let mut info: Vec<(usize, (usize, usize))> = vec![];
                for (index, size) in output_tensor_shape.iter().copied().enumerate() {
                    if execution_indexes.contains(&index) {
                        continue;
                    }
                    info.push((size, (output_tensor_strides[index], tangent_strides[index])));
                }
                let mut offsets: Vec<(usize, usize)> = info.into_iter()
                    .map(|(size, (output_stride, tangent_stride))| (0..size).map(|x| (x * output_stride, x * tangent_stride)).collect::<Vec<(usize, usize)>>())
                    .multi_cartesian_product().map(|x| x.into_iter().fold((0, 0), |(acc, tangent_acc), (x, tangent_x)| (acc + x, tangent_acc + tangent_x)))
                    .collect();
                if offsets.is_empty() {
                    offsets.push((0, 0));
                }
                offsets
            };

            for start_positions in offsets {
                //Create buffer for start positions
                let (output_start_position, tensor_start_position) = start_positions;
                let offset = PullbackOffset {
                    output_offset: output_start_position as u32,
                    tensor_offset: tensor_start_position as u32,
                };
                let start_buffer = gpu.device.create_buffer_init(
                    &BufferInitDescriptor {
                        label: Some("Offset buffer"),
                        contents: bytemuck::bytes_of(&offset),
                        usage: wgpu::BufferUsages::STORAGE,
                    }
                );

                //Create pipeling and binding layout
                let (bind_group_layout, compute_pipeline) = pipelines::backward_pipeline(gpu);

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
                        resource: tangent_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    }],
                });

                //Create compute pass
                let mut compute_pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor {
                        label: Some("Backwards Elementwise Addition"),
                    }
                );
                
                //Run compute pass
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch(execution_sizes[0], execution_sizes[1], execution_sizes[2]);
            }

            let tensor_a_tangent = Tensor {
                device: gpu,
                tensor_layout: Strided {
                    strides: output_tensor_strides,
                },
                shape: output_tensor_shape,
                data: TensorData::GPUStrided::<f32>(output_buffer),
            };

        //Tensor b
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
            let offsets: Vec<(usize, usize)> = {
                let mut info: Vec<(usize, (usize, usize))> = vec![];
                for (index, size) in output_tensor_shape.iter().copied().enumerate() {
                    if execution_indexes.contains(&index) {
                        continue;
                    }
                    info.push((size, (output_tensor_strides[index], tangent_strides[index])));
                }
                let mut offsets: Vec<(usize, usize)> = info.into_iter()
                    .map(|(size, (output_stride, tangent_stride))| (0..size).map(|x| (x * output_stride, x * tangent_stride)).collect::<Vec<(usize, usize)>>())
                    .multi_cartesian_product().map(|x| x.into_iter().fold((0, 0), |(acc, tangent_acc), (x, tangent_x)| (acc + x, tangent_acc + tangent_x)))
                    .collect();
                if offsets.is_empty() {
                    offsets.push((0, 0));
                }
                offsets
            };

            for start_positions in offsets {
                //Create buffer for start positions
                let (output_start_position, tensor_start_position) = start_positions;
                let offset = PullbackOffset {
                    output_offset: output_start_position as u32,
                    tensor_offset: tensor_start_position as u32,
                };
                let start_buffer = gpu.device.create_buffer_init(
                    &BufferInitDescriptor {
                        label: Some("Offset buffer"),
                        contents: bytemuck::bytes_of(&offset),
                        usage: wgpu::BufferUsages::STORAGE,
                    }
                );

                //Create pipeling and binding layout
                let (bind_group_layout, compute_pipeline) = pipelines::backward_pipeline(gpu);

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
                        resource: tangent_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    }],
                });

                //Create compute pass
                let mut compute_pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor {
                        label: Some("Backwards Elementwise Addition"),
                    }
                );
                
                //Run compute pass
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch(execution_sizes[0], execution_sizes[1], execution_sizes[2]);
            }

            let tensor_b_tangent = Tensor {
                device: gpu,
                tensor_layout: Strided {
                    strides: output_tensor_strides,
                },
                shape: output_tensor_shape,
                data: TensorData::GPUStrided::<f32>(output_buffer),
            };

        //Submit operations to gpu
        gpu.queue.submit(Some(encoder.finish()));
        
        //Return
        (tensor_a_tangent, tensor_b_tangent)
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PullbackMetaData {
    output_strides: [u32; 3],
    pad_0: u32,
    tensor_strides: [u32; 3],
    pad_1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PullbackOffset {
    output_offset: u32,
    tensor_offset: u32,  
}


