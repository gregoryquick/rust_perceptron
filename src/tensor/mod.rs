//! Module contains implementation of tensors with there basic operations
//!
use crate::device::Device;

mod cpu;
mod gpu;

pub struct Tensor<'a, D: Device, L: Layout, T: Clone + bytemuck::Pod, const N: usize> {
    pub device: &'a D,
    pub tensor_layout: L,
    pub shape: [usize; N],
    pub data: TensorData<T>,
}

///Trait indicates that a struct contains layout information
pub trait Layout {
}

#[derive(Clone)]
pub struct Strided<const N: usize> {
    pub strides: [usize; N],
}

impl<const N: usize> Layout for Strided<N> {
}

///Enum for raw tensor data
pub enum TensorData<T> {
    CPUStrided(Vec<T>),
    GPUStrided(wgpu::Buffer),
}


