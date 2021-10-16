use crate::pipelines;

use rand::prelude::*;

pub mod fullyconnected;
pub mod batchnorm;
pub mod relu;
pub mod softmax;

#[typetag::serde(tag = "type")]
pub trait NetworkLayer {
    fn get_topology(&self) -> Vec<(usize, usize)>;

    fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<wgpu::Buffer>;

    fn save_from_gpu(&mut self, anchor: &pipelines::Device, data: &Vec<wgpu::Buffer>);

    fn forward(&self,
               input: &wgpu::Buffer,
               layer_data: &Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> wgpu::Buffer;

    fn forward_for_backprop(&self,
               input: &wgpu::Buffer,
               layer_data: &mut Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> (wgpu::Buffer, Vec<wgpu::Buffer>);
    
    fn backprop(&self,
                backprop_grad: &wgpu::Buffer,
                layer_data: &Vec<wgpu::Buffer>, 
                backprop_data: &Vec<wgpu::Buffer>,
                anchor: &pipelines::Device,
                encoder: &mut wgpu::CommandEncoder,
                batch_size: usize,) -> (wgpu::Buffer, Vec<Option<wgpu::Buffer>>);
}

pub fn generate_layer(input_size: usize, layer_type: super::LayerType) -> (usize, Box<dyn NetworkLayer>) {
    use super::LayerType::*;
    use rand_distr::*;
    match layer_type {
        FullyConnected(output_size) => {
            let mut rng = rand::thread_rng();
            let avg: f32 = ((input_size + output_size) as f32) / 2.0;
            let dist = Normal::new(0.0, 1.0 / avg.sqrt()).unwrap();
            let layer = Box::new(fullyconnected::FullyConnected {
                weights:{
                    let vector: Vec<f32> = (0..input_size * output_size).map(|_i| {rng.sample(dist)}).collect();
                    vector
                },
                output_dimension: output_size,
                input_dimension: input_size,
            });
            
            //Return
            (output_size, layer)
        },
        Batchnorm => {
            let mut rng = rand::thread_rng();
            let dist_var = Normal::new(1.0,0.1).unwrap();
            let dist_mean = Normal::new(0.0,1.0).unwrap();
            let layer = Box::new(batchnorm::Batchnorm {
                gamma:{
                    let vector: Vec<f32> = (0..input_size).map(|_i| {rng.sample(dist_var)}).collect();
                    vector
                },
                beta:{
                    let vector: Vec<f32> = (0..input_size).map(|_i| {rng.sample(dist_mean)}).collect();
                    vector
                },
                data_var:{
                    let vector: Vec<f32> = (0..input_size).map(|_i| {1.0}).collect();
                    vector
                },
                data_mean:{
                    let vector: Vec<f32> = (0..input_size).map(|_i| {1.0}).collect();
                    vector
                },
                batches_sampled: 0,
                dimension: input_size,
            });

            //Return
            (input_size, layer)
        },
        Relu => {
            let layer = Box::new(relu::Relu {
                dimension: input_size,
            });

            //Return
            (input_size, layer)
        },
        Softmax => {
            let layer = Box::new(softmax::Softmax {
                dimension: input_size,
            });
            
            //Return
            (input_size, layer)
        },
    }
}
