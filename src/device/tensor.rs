//! Tensors bundle raw data with structure information
use anyhow::{Result, anyhow};

use crate::device::Device;

///Types for shape and stride
pub type Shape = (usize, usize);
pub type Stride = (usize, usize);

///Type for tensor data on any device
pub struct Tensor<'a> {
    pub device: &'a Device,
    pub interior_data: TensorData,
    pub shape: Shape,
    pub stride: Stride,
}

/// Actual interior tensor data
pub enum TensorData {
    GPUData {
        data: wgpu::Buffer,
    },
    CPUData {
        data: Vec<f32>,
    },
}

impl<'a> Tensor<'a> {
    ///Returns max number of elements in tensor
    pub fn size(&self) -> usize {
        //Return
        self.shape.0 * self.shape.1
    }

    ///Transfers Tensor to new device
    pub async fn to<'b>(self, target_device: &'b Device) -> Result<Tensor<'b>> {
        let size = self.size();
        
        let Tensor {
            device: src_device,
            interior_data,
            shape,
            stride,
        } = self;

        match interior_data {
            TensorData::CPUData {data,} => {
                match src_device {
                    Device::Cpu => {
                        match target_device {
                            Device::Gpu{device, ..} => {
                                use wgpu::util::{BufferInitDescriptor, DeviceExt};

                                //Load data to gpu
                                let gpu_data = device.create_buffer_init(
                                    &BufferInitDescriptor {
                                    label: None,
                                    contents: bytemuck::cast_slice(&data[..]),
                                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                                    }
                                );

                                //Return
                                Ok(
                                    Tensor {
                                        device: target_device,
                                        interior_data: TensorData::GPUData {
                                            data: gpu_data,
                                        },
                                        shape,
                                        stride,
                                    }                                
                                )
                            },
                            _ => {
                                Err(anyhow!("Transfer betwean these devices is not suported"))
                            },
                        }
                    },
                    _ => {
                        Err(anyhow!("Tensor format does not match device!"))
                    },
                }
            },
            TensorData::GPUData {data,} => {
                match src_device {
                    Device::Gpu{device, queue, ..} => {
                        match target_device {
                            Device::Cpu => {
                                let type_size = std::mem::size_of::<f32>();
                                
                                //Create command buffer encoder
                                let mut encoder = device.create_command_encoder(
                                    &wgpu::CommandEncoderDescriptor {
                                        label: None
                                    }
                                );

                                //Copy data to readable buffer
                                let size = (type_size * size) as wgpu::BufferAddress;

                                let staging_buffer = device.create_buffer(
                                    &wgpu::BufferDescriptor {
                                        label: Some("Staging buffer"),
                                        size,
                                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                        mapped_at_creation: false,
                                    }
                                );
                                encoder.copy_buffer_to_buffer(
                                    &data, 0,
                                    &staging_buffer, 0,
                                    size,
                                );

                                //Submit commands to gpu
                                queue.submit(Some(encoder.finish()));

                                //Create future of the computation
                                let buffer_slice = staging_buffer.slice(..);
                                let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

                                //Register mapping callbacks
                                device.poll(wgpu::Maintain::Wait);

                                //Check if error occured
                                buffer_future.await?;

                                //Pull from gpu
                                let data = buffer_slice.get_mapped_range();
                                let result = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<f32>(b)).collect();

                                //Drop mapped view and unmap buffer
                                drop(data);
                                staging_buffer.unmap();

                                //Return
                                Ok(
                                    Tensor {
                                        device: target_device,
                                        interior_data: TensorData::CPUData {
                                            data: result,
                                        },
                                        shape,
                                        stride,
                                    }
                                )
                            },
                            _ => {
                                Err(anyhow!("Transfer betwean these devices is not suported"))
                            },
                        }
                    },
                    _ => {
                        Err(anyhow!("Tensor format does not match device!"))
                    },
                }
            },
        }
        //End
    }
}

