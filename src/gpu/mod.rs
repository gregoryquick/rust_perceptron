use anyhow::{Result, anyhow};
use crate::device::tensor::{Tensor, TensorData, Shape, Stride};

pub mod matrix_operations;

pub fn unpack_borrow<'a>(tensor_borrow: &'a Tensor) -> Result<(&'a wgpu::Buffer, &'a Shape, &'a Stride)> {
    let (tensor_data, tensor_shape, tensor_stride) = {
        let Tensor {
            interior_data,
            shape,
            stride,
            ..
        } = tensor_borrow;

        match interior_data {
            TensorData::GPUData {data,} => {
                Ok((data, shape, stride))
            },
            _ => {
                Err(anyhow!("Tensor on wrong device"))
            }
        }
        //End
    }?;

    Ok((tensor_data, tensor_shape, tensor_stride))
}
