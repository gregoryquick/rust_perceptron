//! Implementation of basic operations for strided cpu tensors
use std::error::Error;

use enum_extract::extract;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::device::{CPU, GPU};
use crate::tensor::{Tensor, TensorData, Strided};

impl<'a, T: Clone +  bytemuck::Pod, const N: usize> Tensor<'a, CPU, Strided<N>, T, N> {
    pub async fn to_device<'b>(self, target_device: &'b GPU) -> Result<Tensor<'b, GPU, Strided<N>, T, N>, Box<dyn Error>> {
        let Tensor {
            device: cpu,
            tensor_layout,
            shape,
            data,
        } = self;
        let src_vec = extract!(TensorData::CPUStrided(_), data).unwrap();

        //Load data to gpu
        let trg_buffer = target_device.device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&src_vec[..]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }
        );

        //Return
        Ok(Tensor {
            device: target_device,
            tensor_layout: tensor_layout,
            shape: shape,
            data: TensorData::GPUStrided::<T>(trg_buffer),
        })
    }
}

impl<'a, T: Clone +  bytemuck::Pod, const N: usize> Clone for Tensor<'a, CPU, Strided<N>, T, N> {
    fn clone(&self) -> Self {
        let src_vec = extract!(TensorData::CPUStrided(_), &self.data).unwrap();
        Tensor {
            device: <&CPU>::clone(&self.device),
            tensor_layout: self.tensor_layout.clone(),
            shape: self.shape,
            data: TensorData::CPUStrided::<T>(src_vec.clone()),
        }
    }
}

