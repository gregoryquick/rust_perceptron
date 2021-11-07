use anyhow::{Result, anyhow};

use crate::device::{Device, tensor::Tensor};
use crate::autograd::{Operation, Idx};

/// Struct for tensor variables with no inputs
pub struct GraphTensor<'a> {
    tensor: Tensor<'a>,
    grad: Option<Tensor<'a>>
}

impl<'a> GraphTensor<'a> {
    pub fn new(tensor: Tensor<'a>, grad: Option<Tensor<'a>>) -> Self {
        Self {
            tensor,
            grad,
        }
    }
}

impl<'a> Operation<'a> for GraphTensor<'a> {
    /// No inputs, return empty vec
    fn input_binds(&self)-> Vec<Option<(Idx, usize)>> {
        Vec::new()
    }

    /// No input sockets, throw error
    fn set_bind(&mut self, socket: usize, bind_src: (Idx, usize)) -> Result<()> {
        Err(anyhow!("No input sockets to set"))
    }

    /// Ignore input, value is fixed
    fn forward(&self, input_data: &[&Tensor], device: &'a Device) -> Result<Option<Vec<Tensor<'a>>>> {
        //Do nothing
        Ok(None)
    }

    /// Ignore data, do nothing
    fn write_forward(&mut self, new_data: Vec<Tensor<'a>>) {
    }

    /// Return reference to tensor
    fn read_forward(&self) ->  Result<Vec<&Tensor>> {
        Ok(vec![&self.tensor])
    }

    /// Return if there is a grad
    fn grad_enabled(&self) -> bool {
        match &self.grad {
            None => {
                false
            },
            Some(..) => {
                true
            }
        }
        //End
    }

    /// Calculate the backpropagating gradient on operation
    fn acumulate_grads(&mut self, grads: &[(&Tensor, usize)], device: &Device,) {
        //TODO
    }

    fn read_grad(&self) -> Result<Vec<&Tensor>> {
        match &self.grad {
            None => {
                //Return
                Err(anyhow!("Operation does not have grad enabled"))
            },
            Some(tensor) => {
                //Return
                Ok(vec![tensor])
            }
        }
        //End
    }

    /// No inputs to backprop to ignore
    fn backprop(&mut self, input_data: &[&Tensor], needs_grad: &[bool], device: &Device) -> Result<()> {
        //Nothing to do
        Ok(())
    }

    /// No input socket, throw error
    fn read_backprop(&self) -> Result<Vec<Option<&Tensor>>> {
        Err(anyhow!("No input sockets to read"))
    }

}
