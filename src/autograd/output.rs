use anyhow::{Result, anyhow};

use crate::device::{Device, tensor::Tensor};
use crate::autograd::{Operation, Idx};


/// Struct for tensor output
pub struct OutputProbe<'a> {
    input_binding: Vec<Option<(Idx, usize)>>,
    last_output: Option<Tensor<'a>>,
}

impl<'a>  OutputProbe<'a> {
    pub fn new() -> Self {
        Self {
            input_binding: vec![None],
            last_output: None,
        }
    }
}

impl<'a> Operation<'a> for OutputProbe<'a> {
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
        Err(anyhow!("No output sockets to set"))
    }

    /// TODO
    fn forward(&self, input_data: &[&Tensor], device: &'a Device) -> Result<Option<Vec<Tensor<'a>>>> {
        Ok(None)
    }

    /// TODO
    fn write_forward(&mut self, new_data: Vec<Tensor<'a>>) {
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
    fn acumulate_grads(&mut self, grads: &[(&Tensor, usize)], device: &Device,) {
        //
    }

    /// TODO
    /// Attempt to read the grad
    fn read_grad(&self) -> Result<Vec<&Tensor>> {
        Err(anyhow!("TODO"))
    }

    /// TODO
    /// No cost function so how would this be calculated?
    fn backprop(&mut self,input_data: &[&Tensor], needs_grad: &[bool], device: &Device) -> Result<()> {
        //
        Ok(())
    }
    
    ///TODO
    fn read_backprop(&self) -> Result<Vec<Option<&Tensor>>> {
        Err(anyhow!("TODO"))
    }
}
