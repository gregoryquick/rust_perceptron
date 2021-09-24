use crate::pipelines;

use rand::prelude::*;

pub mod denselayer;
pub mod batchnorm;
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
    match layer_type {
        DenseLayer(output_size) => {
            let mut rng = rand::thread_rng();
            use rand::distributions::Uniform;
            let dist = Uniform::new(-1.0,1.0);
            let layer = Box::new(denselayer::Denselayer {
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
            use rand::distributions::Uniform;
            let dist_var = Uniform::new(0.5,1.5);
            let dist_mean = Uniform::new(-1.0,1.0);
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
        Softmax(output_size) => {
            let mut rng = rand::thread_rng();
            use rand::distributions::Uniform;
            let dist = Uniform::new(-1.0,1.0);
            let layer = Box::new(softmax::Softmax {
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
    }
}
