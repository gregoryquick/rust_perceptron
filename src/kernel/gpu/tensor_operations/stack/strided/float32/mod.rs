use wgpu::util::{BufferInitDescriptor, DeviceExt};
use enum_extract::extract;
use itertools::Itertools;

use crate::tensor::*;
use crate::device::GPU;

mod pipelines;

pub fn forward<'a, I: Clone, const N: usize>(
    gpu: &'a GPU,
    tensors: I,
    ) -> Tensor<'a, GPU, Strided<{N + 1}>, f32, {N + 1}>
    where I: Iterator<Item = &'a Tensor<'a, GPU, Strided<N>, f32, N>>, [(); N + 1]: {
        //Check if tensor devices match
        for tensor in tensors.clone() {
            let Tensor {
                device: tensor_gpu,
                tensor_layout: _,
                shape: _,
                data: _,
            } = tensor;

            assert!(gpu as *const _ == *tensor_gpu as *const _, "Tensor device mismatch");
        }

        //Check if tensor shapes are compatable
        let Tensor {
            device: _,
            tensor_layout: _,
            shape: concat_shape,
            data: _,
        } = tensors.clone().next().unwrap();

        for tensor in tensors.clone() {
            let Tensor {
                device: _,
                tensor_layout: _,
                shape: tensor_shape,
                data: _,
            } = tensor;

            assert!(tensor_shape == concat_shape, "Tensor shape mismatch");
        }
        
        //Calculate important values for internal use
        let iterator_len = tensors.clone().count();
        let type_size = std::mem::size_of::<f32>();
        
        //Create meta data values
        let output_tensor_shape = {
            let mut shape = [0 as usize; N + 1];
            for (location, value) in shape.iter_mut().zip(Some(iterator_len).into_iter().chain(concat_shape.iter().cloned())) {
                *location = value;
            }
            shape
        };

        let size: usize = output_tensor_shape.iter().product();
        
        let output_tensor_strides = {
            let mut strides = [0 as usize; N + 1];
            let mut current_stride = 1;
            for (location, value) in strides.iter_mut().zip(output_tensor_shape.iter().cloned()).rev() {
                *location = current_stride;
                current_stride = current_stride * value;
            }
            strides
        };
        
        let execution_indexes = {
            let mut labeled_dims: Vec<(usize, usize)> = output_tensor_shape.iter().copied().enumerate().skip(1).collect();
            labeled_dims.sort_by(|(_, a), (_, b)| b.cmp(a));
            labeled_dims.into_iter().map(|(x, _)| x).take(3).collect::<Vec<usize>>()
        };
        
        let (execution_sizes, execution_strides) = {
            let mut sizes: [u32; 3] = [1, 1, 1];
            let mut strides: [u32; 3] = [0, 0, 0];

            for (index, key) in execution_indexes.iter().copied().enumerate() {
                sizes[index] = output_tensor_shape[key] as u32;
                strides[index] = output_tensor_strides[key] as u32;
            }
            
            (sizes, strides)
        };
         
        //Create command buffer encoder
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );
        
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
        for (i, tensor) in tensors.clone().enumerate() {
            let Tensor {
                device: _,
                tensor_layout: Strided {
                    strides: tensor_strides,
                },
                shape: _,
                data: tensor_data,
            } = tensor;

            //Deterimine strides on gpu executed indices
            let execution_tensor_strides = {
                let mut t_strides: [u32; 3] = [0, 0, 0];

                for (index, key) in execution_indexes.iter().copied().enumerate() {
                    t_strides[index] = tensor_strides[key - 1] as u32;
                }
            
                t_strides
            };

            //Create meta data buffer
            let meta_buffer = {
                let meta_data = OperationMetaData {
                    output_strides: execution_strides,
                    pad_0: 0,
                    tensor_strides: execution_tensor_strides,
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

            //Unwrap tensor buffers
            let tensor_buffer = extract!(TensorData::GPUStrided(_), tensor_data).unwrap();

            //Unroll along extra indices
            let offsets: Vec<(usize, usize)> = {
                let mut info: Vec<(usize, (usize, usize))> = vec![];
                for (index, size) in output_tensor_shape.iter().copied().enumerate() {
                    if execution_indexes.contains(&index) {
                        continue;
                    }
                    info.push((size, (output_tensor_strides[index], tensor_strides[index])));
                }
                let mut offsets: Vec<(usize, usize)> = info.into_iter()
                    .map(|(size, (output_stride, tensor_stride))| (0..size).map(|x| (x * output_stride, x * tensor_stride)).collect::<Vec<(usize, usize)>>())
                    .multi_cartesian_product().map(|x| x.into_iter().fold((0, 0), |(output_acc, tensor_acc), (output_x, tensor_x)| (output_acc + output_x, tensor_acc + tensor_x)))
                    .collect();
                if offsets.is_empty() {
                    offsets.push((0, 0));
                }
                offsets
            };

            for start_positions in offsets {
                //Create buffer for start position
                let (output_start_position, tensor_start_position) = start_positions;
                let offset = Offset {
                    output_offset: (output_start_position + output_tensor_strides[0] * i) as u32,
                    tensor_offset: tensor_start_position as u32,
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
                        resource: tensor_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    }],
                });

                //Create compute pass
                let mut compute_pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor {
                        label: Some("Stacking"),
                    }
                );
            
                //Run compute pass
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch(execution_sizes[0], execution_sizes[1], execution_sizes[2]);
            }
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
    tensor_strides: [u32; 3],
    pad_1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Offset {
    output_offset: u32,
    tensor_offset: u32,
}

pub fn backward<'a, I: Clone, const N: usize>(
    gpu: &'a GPU,
    tangent: &'a Tensor<'a, GPU, Strided<{N + 1}>, f32, {N + 1}>,
    tensors: I,
    ) -> Vec<Tensor<'a, GPU, Strided<N>, f32, N>>
    where I: Iterator<Item = &'a Tensor<'a, GPU, Strided<N>, f32, N>>, [(); N + 1]: {
        //Unpack backprop_grad tensor
        let Tensor {
            device: tangent_gpu,
            tensor_layout: Strided {
                    strides: tangent_strides,
            },
            shape: tangent_shape,
            data: tangent_data,
        } = tangent.clone();
    
        assert!(gpu as *const _ == tangent_gpu as *const _, "Tensor device mismatch");
    
        //Check if tensor devices match
        for tensor in tensors.clone() {
            let Tensor {
                device: tensor_gpu,
                tensor_layout: _,
                shape: _,
                data: _,
            } = tensor;

            assert!(gpu as *const _ == *tensor_gpu as *const _, "Tensor device mismatch");
        }

        //Check if tensor shapes are compatable
        for tensor in tensors.clone() {
            let Tensor {
                device: _,
                tensor_layout: _,
                shape: tensor_shape,
                data: _,
            } = tensor;

            assert!(tensor_shape == &tangent_shape[1..], "Tensor shape mismatch");
        }

        //Calculate important values for internal use
        let iterator_len = tensors.clone().count();
        let type_size = std::mem::size_of::<f32>();

        //Create meta data values
        let output_tensor_shape = {
            let mut shape = [0 as usize; N];
            for (location, value) in shape.iter_mut().zip(tangent_shape.iter().skip(1).cloned()) {
                *location = value;
            }
            shape
        };

        let size: usize = output_tensor_shape.iter().product();
        
        let output_tensor_strides = {
            let mut strides = [0 as usize; N];
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
        
        let (execution_sizes, execution_strides) = {
            let mut sizes: [u32; 3] = [1, 1, 1];
            let mut strides: [u32; 3] = [0, 0, 0];

            for (index, key) in execution_indexes.iter().copied().enumerate() {
                sizes[index] = output_tensor_shape[key] as u32;
                strides[index] = output_tensor_strides[key] as u32;
            }
            
            (sizes, strides)
        };

        //Create command buffer encoder
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Create vec for output tensors
        let mut grads = vec![];

        //Unwrap grad tensor buffer
        let backprop_grad_buffer = extract!(TensorData::GPUStrided(_),tangent_data).unwrap();

        //create tensors and que operations to encoder
        for (i, tensor) in tensors.clone().enumerate() {
            let Tensor {
                device: _,
                tensor_layout: Strided {
                    strides: tensor_strides,
                },
                shape: tensor_shape,
                data: tensor_data,
            } = tensor;
            
            //Deterimine strides on gpu executed indices
            let execution_tensor_strides = {
                let mut t_strides: [u32; 3] = [0, 0, 0];

                for (index, key) in execution_indexes.iter().copied().enumerate() {
                    t_strides[index] = tensor_strides[key] as u32;
                }
            
                t_strides
            };

            //Create buffer for use in output tensor
            let output_buffer = gpu.device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Output buffer"),
                    size: (type_size * size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }
            );

            //Create meta data buffer
            let meta_buffer = {
                let meta_data = OperationMetaData {
                    output_strides: execution_strides,
                    pad_0: 0,
                    tensor_strides: execution_tensor_strides,
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

            //Unroll along  extra indices
            let offsets: Vec<(usize, usize)> = {
                let mut info: Vec<(usize, (usize, usize))> = vec![];
                for (index, size) in output_tensor_shape.iter().copied().enumerate() {
                    if execution_indexes.contains(&index) {
                        continue;
                    }
                    info.push((size, (output_tensor_strides[index], tangent_strides[index + 1])));
                }
                let mut offsets: Vec<(usize, usize)> = info.into_iter()
                    .map(|(size, (output_stride,  backprop_grad_stride))| (0..size).map(|x| (x * output_stride, x *  backprop_grad_stride)).collect::<Vec<(usize, usize)>>())
                    .multi_cartesian_product().map(|x| x.into_iter().fold((0, 0), |(output_acc, tensor_acc), (output_x, tensor_x)| (output_acc + output_x, tensor_acc + tensor_x)))
                    .collect();
                if offsets.is_empty() {
                    offsets.push((0, 0));
                }
                offsets
            };
            
            for start_positions in offsets {
                //Create buffer for start position
                let (output_start_position, tensor_start_position) = start_positions;
                let offset = Offset {
                    output_offset: output_start_position as u32,
                    tensor_offset: (tensor_start_position + size * i) as u32,
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
                        resource: backprop_grad_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    }],
                });

                //Create compute pass
                let mut compute_pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor {
                        label: Some("Backwards Stacking"),
                    }
                );
            
                //Run compute pass
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch(execution_sizes[0], execution_sizes[1], execution_sizes[2]);
            }

            //Add tensor to output
            grads.push(Tensor {
                device: gpu,
                tensor_layout: Strided {
                    strides: output_tensor_strides,
                },
                shape: output_tensor_shape,
                data: TensorData::GPUStrided::<f32>(output_buffer),
            });
        }
        
        //Submit operations to gpu
        gpu.queue.submit(Some(encoder.finish()));


        //Return
        grads
}
