//! Tensors bundle raw data with structure information
use anyhow::{Result, anyhow};

use crate::device::Device;

///Type for tensor data on any device
pub enum Tensor {
    GPUMatrix {
        data: wgpu::Buffer,
        shape: (usize, usize),
        stride: (usize, usize),
    },
    CPUMatrix {
        data: Vec<f32>,
        shape: (usize, usize),
        stride: (usize, usize),
    },
}

impl Tensor {
    ///Returns total elements of the Tensor
    pub fn size(&self) -> usize {
        match self {
             Tensor::CPUMatrix{shape, ..} | Tensor::GPUMatrix{shape, ..} => {
                //Return
                shape.0 * shape.1
             },
        }
    }
    
    ///Transfers the Tensor from one device to another
    pub async fn transfer_device(self, src_device: &Device, target_device: &Device) -> Result<Tensor> {
        match self {
            Tensor::CPUMatrix{data, shape, stride} => {
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
                            Tensor::GPUMatrix {
                                data: gpu_data,
                                shape,
                                stride,
                            }
                        )
                    },
                    Device::Cpu{..} => {
                        //Return
                        Err(anyhow!("Tensor already on cpu!"))
                    },
                }
            },
            Tensor::GPUMatrix{data, shape, stride} => {
                match src_device {
                    Device::Gpu{device, queue, ..} => {
                        let type_size = std::mem::size_of::<f32>();
                        
                        //Create command buffer encoder
                        let mut encoder = device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: None,
                            }
                        );

                        //Copy data to readable buffer
                        let size = (type_size * shape.0 * shape.1) as wgpu::BufferAddress;
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

                        //Drop mapped view
                        drop(data);
                        //Unmap buffer
                        staging_buffer.unmap();
                        
                        //Return
                        Ok(
                            Tensor::CPUMatrix {
                                data: result,
                                shape,
                                stride,
                            }
                        )
                    },
                    Device::Cpu{..} => {
                        //Return
                        Err(anyhow!("Tensor not on cpu"))
                    },
                }
            },
        }
    }
}
