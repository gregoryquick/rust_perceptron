use anyhow::{Result, anyhow};

use crate::device::{Device, tensor::Tensor};
use crate::autograd::{Operation, Idx};
use crate::gpu;

/// Struct for tensor output
pub struct MatrixAdd<'a> {
    output_used: Vec<bool>,    
    input_binding: Vec<Option<(Idx, usize)>>,
    last_output: Option<Vec<Tensor<'a>>>,
    grad: Option<Tensor<'a>>,
    backprop_grads: Option<Vec<Option<Tensor<'a>>>>,
}

impl<'a>  MatrixAdd<'a> {
    pub fn new() -> Self {
        Self {
            output_used: vec![false],
            input_binding: vec![None, None],
            last_output: None,
            grad: None,
            backprop_grads: None,
        }
    }
}

impl<'a> Operation<'a> for MatrixAdd<'a> {
    /// Return clone of binding info
    fn input_binds(&self)-> Vec<Option<(Idx, usize)>> {
        self.input_binding.clone()
    }

    /// Set input binding
    fn set_bind(&mut self, socket: usize, bind_src: (Idx, usize)) -> Result<()> {
        match self.input_binding.get_mut(socket) {
            None => {
                return Err(anyhow!("Socket {} is invalid", socket))
            },
            Some(data) => {
                *data = Some(bind_src);
                //Return
                Ok(())
            },
        }
        //End
    }

    /// Cannot attach to input sockets in graph
    fn output_used(&self) -> Vec<bool> {
        Vec::new()
    }

    /// No output sockets, throw error
    fn set_used(&mut self, socket: usize, is_usued: bool) -> Result<()> {
        match self.output_used.get_mut(socket) {
            None => {
                return Err(anyhow!("Socket {} is invalid", socket))
            },
            Some(data) => {
                *data = is_usued;
                //Return
                Ok(())
            },
        }
        //End
    }

    /// TODO
    fn forward(&self, input_data: &[&Tensor], graph_device: &'a Device) -> Result<Option<Vec<Tensor<'a>>>> {
        let mut input_data = input_data.into_iter();
        let tensor_a = input_data.next().unwrap();
        let tensor_b = input_data.next().unwrap();

        match graph_device {
            Device::Gpu{device, queue, ..} => {
                //Create command buffer encoder
                let mut encoder = device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: None,
                    }
                );

                let output = gpu::matrix_operations::matrixadd::forward(tensor_a, tensor_b, &mut encoder, device, graph_device)?;

                queue.submit(Some(encoder.finish()));

                Ok(Some(output))
            },
            _ => {
                Err(anyhow!("Device not supported for operation"))
            },
        }
        //End
    }

    /// Save data to last output
    fn write_forward(&mut self, new_data: Vec<Tensor<'a>>) {
        self.last_output = Some(new_data);
    }

    /// No output sockets, throw error
    fn read_forward(&self) -> Result<Vec<&Tensor>> {
        Err(anyhow!("No output sockets to read"))
    }

    ///TODO
    fn grad_enabled(&self) -> bool {
        false
    }

    /// TODO
    /// Calculate the backpropagating gradient on operation
    fn acumulate_grads(&mut self,grads: &[(&Tensor, usize)], device: &Device,) {
        //
    }

    /// TODO
    /// Attempt to read the grad
    fn read_grad(&self) -> Result<Vec<&Tensor>> {
        Err(anyhow!("TODO"))
    }

    /// TODO
    /// No cost function so how would this be calculated?
    fn backprop(&mut self, input_data: &[&Tensor], needs_grad: &[bool], device: &Device) -> Result<()> {
        Ok(())
    }
    
    ///TODO
    fn read_backprop(&self) -> Result<Vec<Option<&Tensor>>> {
        Err(anyhow!("TODO"))
    }
}

