use anyhow::{Result, anyhow};

use crate::device::{Device, tensor::Tensor};
use crate::autograd::{Operation, Idx};

/// Struct for tensor variables with no inputs
pub struct GraphTensor<'a> {
    tensor: Tensor<'a>,
    grad: Option<Tensor<'a>>,
    uses: usize,
}

impl<'a> GraphTensor<'a> {
    pub fn new(tensor: Tensor<'a>, grad: Option<Tensor<'a>>) -> Self {
        Self {
            tensor,
            grad,
            uses: 0,
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

    /// Ignore input, value is fixed
    fn forward(&self, input_data: &[&Tensor], device: &'a Device) -> Result<Option<Vec<Tensor<'a>>>> {
        if !input_data.is_empty() {
            return Err(anyhow!("Does not take inputs"))
        }
        
        //Do nothing
        Ok(None)
    }

    /// Ignore data, do nothing
    fn write_forward(&mut self, new_data: Vec<Tensor<'a>>) -> Result<()> {
        if !new_data.is_empty() {
            return Err(anyhow!("Cannot write to tensor output"))
        }
        Ok(())
    }

    /// Return reference to tensor
    fn read_forward(&self) ->  Result<Vec<&Tensor<'a>>> {
        Ok(vec![&self.tensor])
    }

    /// Return if there is a grad
    fn grads_enabled(&self) -> Vec<bool> {
        match &self.grad {
            None => {
                vec![false]
            },
            Some(..) => {
                vec![true]
            }
        }
        //End
    }
    
    ///Write grad it input is not malformed
    fn write_grads(&mut self, new_data: Vec<Option<Tensor<'a>>>) -> Result<()> {
        if new_data.len() != 1 {
             return Err(anyhow!("Malformed grads"))
        }
        
        self.grad = new_data.into_iter().next().unwrap();

        //Return
        Ok(())
    }

    ///
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

    /// No inputs to backprop to ignore
    fn backprop(&self, input_data: &[&Tensor], needs_grad: &[bool], backprop_grads: &[Option<&Tensor>], device: &Device) -> Result<Vec<Option<Tensor<'a>>>> {
        //Nothing to do
        Ok(vec![])
    }

     ///TODO
    fn write_backprop(&mut self, new_data: Vec<Option<Tensor<'a>>>) -> Result<()> {
        Ok(())
    }

    /// No input socket, throw error
    fn read_backprop(&self) -> Result<Vec<Option<&Tensor<'a>>>> {
        Err(anyhow!("No input sockets to read"))
    }

}
