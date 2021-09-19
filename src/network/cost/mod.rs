use crate::pipelines;

pub mod squarederror;

#[typetag::serde(tag = "type")]
pub trait CostFunction {
    //Should include summation
    fn cost(&self,
            prediction: &wgpu::Buffer,
            target: &wgpu::Buffer,
            anchor: &pipelines::Device,
            encoder: &mut wgpu::CommandEncoder,
            batch_size: usize,) -> wgpu::Buffer;

    //Should not include summation
    fn cost_prime(&self,
                  prediction: &wgpu::Buffer,
                  target: &wgpu::Buffer,
                  anchor: &pipelines::Device,
                  encoder: &mut wgpu::CommandEncoder,
                  batch_size: usize,) -> wgpu::Buffer;
}

pub fn generate_cost(input_size: usize, cost_function: super::CostFunction) -> Box<dyn CostFunction> {
    use super::CostFunction::*;
    match cost_function {
        SquaredError => {
            Box::new(squarederror::SquaredError {
                dimension: input_size,
            })
        },
    }
}
