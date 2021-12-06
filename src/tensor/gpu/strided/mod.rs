//! Implementation of basic operations for strided gpu tensors
use std::error::Error;

use enum_extract::extract;

use crate::device::{GPU, CPU};
use crate::tensor::{Tensor, TensorData, Strided};

impl<'a, T: Clone +  bytemuck::Pod, const N: usize> Tensor<'a, GPU, Strided<N>, T, N> {
    pub async fn to_device<'b>(self, target_device: &'b CPU) -> Result<Tensor<'b, CPU, Strided<N>, T, N>, Box<dyn Error>> {
        let Tensor {
            device: gpu,
            tensor_layout,
            shape,
            data,
        } = self;
        let src_buffer = extract!(TensorData::GPUStrided(_), data).unwrap();
        let type_size = std::mem::size_of::<T>();
        let size: usize = self.shape.iter().product();

        //Create command buffer encoder
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );
        
        //Create Staging buffer to read data to cpu
        let staging_buffer = gpu.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        //Copy src_buffer to staging_buffer
        encoder.copy_buffer_to_buffer(
            &src_buffer, 0,
            &staging_buffer, 0,
            (type_size * size) as wgpu::BufferAddress,
        );

        //Submit commands to gpu
        gpu.queue.submit(Some(encoder.finish()));

        //Create future of the computation
        let buffer_slice = staging_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        //Register mapping callbacks
        gpu.device.poll(wgpu::Maintain::Wait);

        //Check if error occured
        buffer_future.await?;

        //Pull data from gpu
        let data = buffer_slice.get_mapped_range();
        let trg_vec = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<T>(b)).collect();

        
        //Drop mapped view and unmap buffer
        drop(data);
        staging_buffer.unmap();

        //Return
        Ok(Tensor {
            device: target_device,
            tensor_layout: tensor_layout,
            shape: shape,
            data: TensorData::CPUStrided::<T>(trg_vec),
        })
    }
}

impl<'a, T: Clone + bytemuck::Pod, const N: usize> Clone for Tensor<'a, GPU, Strided<N>, T, N> {
    fn clone(&self) -> Self {
        let gpu = self.device;
        let src_buffer = extract!(TensorData::GPUStrided(_), &self.data).unwrap();
        let type_size = std::mem::size_of::<T>();
        let size: usize = self.shape.iter().product();

        //Create buffer for new tensor
        let trg_buffer = gpu.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: None,
                size: (type_size * size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        //Create command buffer encoder
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Copy src_buffer to trg_buffer
        encoder.copy_buffer_to_buffer(
            src_buffer, 0,
            &trg_buffer, 0,
            (type_size * size) as wgpu::BufferAddress,
        );

        //Submit commands to gpu
        gpu.queue.submit(Some(encoder.finish()));

        //Return
        Tensor {
            device: <&GPU>::clone(&self.device),
            tensor_layout: self.tensor_layout.clone(),
            shape: self.shape,
            data: TensorData::GPUStrided::<T>(trg_buffer),
        }
    }
}

