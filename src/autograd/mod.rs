use anyhow::Result;

use crate::device::{Device, tensor::Tensor};

pub mod graph;
pub mod matrix_operations;
pub mod tensor;
pub mod output;

/// The ID for a node in the graph
pub type Idx = usize;

/// Trait for operations in the compute graph
pub trait Operation<'a> {
    /// Returns bound operation with output socket for each input
    fn input_binds(&self)-> Vec<Option<(Idx, usize)>>;

    /// Bind an input socket to a chosen output socket
    fn set_bind(&mut self, socket: usize, bind_src: (Idx, usize)) -> Result<()>;

    /// Returns bools to indicate which outputs are used
    fn output_used(&self) -> Vec<bool>;

    /// Set if a given socket is used or not
    fn set_used(&mut self, socket: usize, is_usued: bool) -> Result<()>;

    /// Takes slice of references to input operations output tensors
    /// to perform forward propogation and return new values if aplicable
    fn forward(&self, input_data: &[&Tensor], device: &'a Device) -> Result<Option<Vec<Tensor<'a>>>>;

    ///Replace stored forward prop values
    fn write_forward(&mut self, new_data: Vec<Tensor<'a>>);

    /// Attempts to read stored forward prop
    fn read_forward(&self) ->  Result<Vec<&Tensor>>;

    ///Get it the operation is keeping a grad
    fn grad_enabled(&self) -> bool;
    
    /// Calculate the backpropagating gradient on operation
    fn acumulate_grads(&mut self, grads: &[(&Tensor, usize)], device: &Device,);
        
    /// Return reference to internal tensor used for tracking
    /// the backpropigating grad the operation recieves
    fn read_grad(&self) -> Result<Vec<&Tensor>>;
    
    /// Return grads with respect to all inputs flagged with `needs_grad`
    /// Requres tensor contaning the backpropogating grad to operation
    fn backprop(&mut self, input_data: &[&Tensor], needs_grad: &[bool], device: &Device) -> Result<()>;
    
    /// Attemps to read stored backpropagating grads to inputs
    fn read_backprop(&self) -> Result<Vec<Option<&Tensor>>>;
}
