use anyhow::{Result, anyhow};

use crate::device::{Device, tensor::Tensor};
use crate::autograd::{Operation, Idx};
use crate::gpu;

/// Struct for tensor output
pub struct CrossEntropy<'a> {
    input_binding: Vec<Option<(Idx, usize)>>,
    last_output: Option<Vec<Tensor<'a>>>,
    grad: Option<Tensor<'a>>,
    backprop_grads: Option<Vec<Option<Tensor<'a>>>>,
    uses: usize,
}

impl<'a> CrossEntropy<'a> {
    pub fn new() -> Self {
        Self {
            input_binding: vec![None, None],
            last_output: None,
            grad: None,
            backprop_grads: None,
            uses: 0,
        }
    }
}

impl<'a> Operation<'a> for CrossEntropy<'a> {
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

    ///Returns how many times each output socket is used
    fn output_uses(&self) -> Vec<usize> {
        vec![self.uses]
    }

    /// Set number of uses to a new number
    fn set_uses(&mut self, socket: usize, use_num: usize) -> Result<()> {
        if socket != 0 {
            return Err(anyhow!("Socket {} is invalid", socket))
        }
        
        //Set use number
        self.uses = use_num;
        
        //Return
        Ok(())
    }


    /// Compute cross entropy betwean the inputs
    fn forward(&self, input_data: &[&Tensor], graph_device: &'a Device) -> Result<Option<Vec<Tensor<'a>>>> {
        if input_data.len() != 2 {
            return Err(anyhow!("Malformed input"))
        }

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

                let output = gpu::cost_functions::crossentropy::forward(tensor_a, tensor_b, &mut encoder, device, graph_device)?;

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
    fn write_forward(&mut self, new_data: Vec<Tensor<'a>>) -> Result<()> {
        self.last_output = Some(new_data);
        Ok(())
    }

    /// Return result if it is saved
    fn read_forward(&self) -> Result<Vec<&Tensor<'a>>> {
        match &self.last_output {
            None => {
                Err(anyhow!("No forward pass saved"))
            },
            Some(data) => {
                let mut output = Vec::new();
                for tensor in data.iter() {
                    output.push(tensor);
                }
                //Return
                Ok(output)
            },
        }
        //End
    }

    ///TODO
    fn grads_enabled(&self) -> Vec<bool> {
        vec![true]
    }

    ///Write grad if not malformed
    fn write_grads(&mut self, new_data: Vec<Option<Tensor<'a>>>) -> Result<()> {
        if new_data.len() != 1 {
             return Err(anyhow!("Malformed grads"))
        }
        
        self.grad = new_data.into_iter().next().unwrap();

        //Return
        Ok(())
    }
    
    ///Return reference to the grad if it exists
    fn read_grads(&self) -> Result<Vec<Option<&Tensor<'a>>>> {
        match &self.grad {
            None => {
                //Return
                Err(anyhow!("Operation does not have grad enabled"))
            },
            Some(tensor) => {
                //Return
                Ok(vec![Some(tensor)])
            }
        }
        //End
    }

    /// TODO
    /// No cost function so how would this be calculated?
    fn backprop(&self, input_data: &[&Tensor], needs_grad: &[bool], backprop_grads: &[Option<&Tensor>], graph_device: &Device) -> Result<Vec<Option<Tensor<'a>>>> {
        let mut input_data = input_data.into_iter();
        let tensor_a = input_data.next().unwrap();
        let tensor_b = input_data.next().unwrap();

        let mut needs_grad = needs_grad.into_iter();
        let grad_need_a = needs_grad.next().unwrap();
        let grad_need_b = needs_grad.next().unwrap();

        let mut backprop_grads = backprop_grads.into_iter();
        let backprop_a = backprop_grads.next().unwrap();

        match graph_device {
             Device::Gpu{device, queue, ..} => {
                 Err(anyhow!("TODO"))
             },
             _ => {
                Err(anyhow!("Device not supported for operation"))
            },
        }
    }

    ///TODO
    fn write_backprop(&mut self, new_data: Vec<Option<Tensor<'a>>>) -> Result<()> {
        Err(anyhow!("TODO"))
    }
    
    ///TODO
    fn read_backprop(&self) -> Result<Vec<Option<&Tensor<'a>>>> {
        Err(anyhow!("TODO"))
    }
}

