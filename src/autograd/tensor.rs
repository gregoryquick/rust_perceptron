use anyhow::Result;
use crate::device::{Device, tensor::Tensor};

/// Struct for tensor variables with no inputs
struct GraphTensor<'a> {
    output_used: Vec<bool>,
    tensor: Tensor<'a>,
    grad: Option<Tensor<'a>>
}

impl<'a> super::Operation for GraphTensor<'a> {
    /// No inputs, return empy vec
    fn input_shape(&self) -> Vec<Option<usize>> {
        Vec::new()
    }

    /// No inputs, return empty vec
    fn input_bound(&self)-> Vec<bool> {
        Vec::new()
    }

    /// Return shape of tensor
    fn output_shape(&self) -> Vec<Option<usize>> {
        let (x, y) = self.tensor.shape;
        vec!(Some(x), Some(y))
    }
    
    /// Return clone of internal tracker
    fn output_used(&self) -> Vec<bool> {
        self.output_used.clone()
    }

    /// Ignore input and return reference to tensor
    fn forward(&mut self,
               input_data: &[&Tensor],
               encoder: &mut wgpu::CommandEncoder,
               device: &Device) -> Result<&Tensor> {
        Ok(&self.tensor)
    }

    /// No inputs to backprop to, return empy vec
    fn backprop(&self,
                //backprop_grad: Tensor,
                input_data: &[&Tensor],
                needs_grad: &[bool],
                encoder: &mut wgpu::CommandEncoder,
                device: &Device
                ) -> Result<Vec<Option<Tensor>>> {
        Ok(Vec::new())
    }

    fn grad(&self) -> Option<&Tensor> {
        match &self.grad {
            None => {
                //Return
                None
            },
            Some(tensor) => {
                //Return
                Some(tensor)
            }
        }
        //End
    }
}
