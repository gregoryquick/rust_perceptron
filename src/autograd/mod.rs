use anyhow::{Result, anyhow};

use crate::device::{Device, tensor::Tensor};
use crate::gpu;

pub mod cost_functions;
pub mod graph;
pub mod matrix_operations;
pub mod tensor;

/// The ID for a node in the graph
pub type Idx = usize;

/// Trait for operations in the compute graph
pub trait Operation<'a> {
    /// Returns bound operation with output socket for each input
    fn input_binds(&self)-> Vec<Option<(Idx, usize)>>;

    /// Bind an input socket to a chosen output socket
    fn set_bind(&mut self, socket: usize, bind_src: (Idx, usize)) -> Result<()>;

    ///Returns how many times each output socket is used
    fn output_uses(&self) -> Vec<usize>;

    /// Set number of uses to a new number
    fn set_uses(&mut self, socket: usize, use_num: usize) -> Result<()>;

    /// Takes slice of references to input operations output tensors
    /// to perform forward propogation and return new values if aplicable
    fn forward(&self, input_data: &[&Tensor], device: &'a Device) -> Result<Option<Vec<Tensor<'a>>>>;

    /// Replace stored forward prop values
    fn write_forward(&mut self, new_data: Vec<Tensor<'a>>) -> Result<()>;

    /// Attempts to read stored forward prop
    fn read_forward(&self) ->  Result<Vec<&Tensor<'a>>>;

    ///Get it the operation is keeping a grad
    fn grads_enabled(&self) -> Vec<bool>;
    
    /// Replace stored backpropagating gradient values
    fn write_grads(&mut self, new_data: Vec<Option<Tensor<'a>>>) -> Result<()>;
        
    /// Return reference to internal tensor used for tracking
    /// the backpropigating grad the operation recieves
    fn read_grads(&self) -> Result<Vec<Option<&Tensor<'a>>>>;
    
    /// Return grads with respect to all inputs flagged with `needs_grad`
    /// Requres tensor contaning the backpropogating grad to operation
    fn backprop(&self, input_data: &[&Tensor], needs_grad: &[bool], backprop_grads: &[Option<&Tensor>], device: &Device) -> Result<Vec<Option<Tensor<'a>>>>;

    /// Replace stored backprop output values
    fn write_backprop(&mut self, new_data: Vec<Option<Tensor<'a>>>) -> Result<()>;
    
    /// Attemps to read stored backpropagating grads to inputs
    fn read_backprop(&self) -> Result<Vec<Option<&Tensor<'a>>>>;
}

pub fn acumulate_grads<'a>(grads: &[(Option<&Tensor<'a>>, usize)], grads_enabled: &[bool], graph_device: &'a Device,) -> Result<Vec<Option<Tensor<'a>>>> {
    let mut output_grads : Vec<Option<Tensor<'a>>> = Vec::new();
    let mut grad_vec = grads.to_vec();
    grad_vec.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    let grads_by_socket: Vec<(Vec<(&Tensor, usize)>, bool)> = grad_vec.group_by(|(_, a), (_, b)| a == b).map(|slice| {
        let vec: Vec<(&Tensor, usize)> = slice.to_vec().into_iter().filter_map(|(maybe_tensor, socket)| maybe_tensor.map(|tensor| (tensor, socket))).collect();
        vec
    }).zip(grads_enabled.to_vec().into_iter()).collect();
        
    for (grads_of_socket, compute_grad) in grads_by_socket {
        if compute_grad {
            match grads_of_socket.len() {
                0 => {
                    output_grads.push(None);
                },
                1 => {
                    let mut data = grads_of_socket.into_iter();
                    let (grad_0, _) = data.next().unwrap();
                    output_grads.push(Some(grad_0.duplicate()?));
                },
                _ => {
                    let mut data = grads_of_socket.into_iter();
                    let (grad_0, _) = data.next().unwrap();
                    let (grad_1, _) = data.next().unwrap();
                    match graph_device {
                        Device::Gpu{device, queue, ..} => {
                            //Create command buffer encoder
                            let mut encoder = device.create_command_encoder(
                                &wgpu::CommandEncoderDescriptor {
                                    label: None,
                                }
                            );

                            let mut accumulation = gpu::matrix_operations::matrixadd::forward(grad_0, grad_1, &mut encoder, device, graph_device)?.into_iter().next().unwrap();

                            for (grad_n, _) in data {
                                accumulation = gpu::matrix_operations::matrixadd::forward(&accumulation, grad_n, &mut encoder, device, graph_device)?.into_iter().next().unwrap();
                            }

                            
                            queue.submit(Some(encoder.finish()));

                            output_grads.push(Some(accumulation));
                        },
                        _ => {
                            return Err(anyhow!("Device not supported for operation"))
                        },
                    }
                },
            }
            //End
        } else {
            output_grads.push(None);
        }
    }
        
    //Return
    Ok(output_grads)
}
