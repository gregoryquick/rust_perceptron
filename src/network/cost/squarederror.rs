use crate::pipelines;

use serde::{Serialize, Deserialize};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[derive(Serialize, Deserialize, Debug)]
pub struct SquaredError {
    pub dimension: usize,
}

#[typetag::serde]
impl super::CostFunction for SquaredError {
    fn cost(&self,
            prediction: &wgpu::Buffer,
            target: &wgpu::Buffer,
            anchor: &pipelines::Device,
            encoder: &mut wgpu::CommandEncoder,
            batch_size: usize,) -> wgpu::Buffer {
        let device = &anchor.device;

        //Create loss pipeline
        let loss_uniforms = {
            let uniform_data = [self.dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };

        let loss_pipeline = pipelines::elementsubtract::Pipeline::new::<f32>(anchor, (
                &loss_uniforms,
                prediction,
                target,
            ),
            self.dimension,
            batch_size,
        );

        //Run loss pipeline
        loss_pipeline.run(encoder, self.dimension, batch_size);
        
        //Create error pipeline
        let error_pipeline = pipelines::squarederror::Pipeline::new::<f32>(anchor, (
                &loss_uniforms,
                &loss_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run loss pipeline
        error_pipeline.run(encoder, self.dimension, batch_size);

        //Return
        error_pipeline.output_buffer
    }

    fn cost_prime(&self,
                  prediction: &wgpu::Buffer,
                  target: &wgpu::Buffer,
                  anchor: &pipelines::Device,
                  encoder: &mut wgpu::CommandEncoder,
                  batch_size: usize,) -> wgpu::Buffer {
        let device = &anchor.device;

        //Create loss pipeline
        let loss_uniforms = {
            let uniform_data = [self.dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };

        let loss_pipeline = pipelines::elementsubtract::Pipeline::new::<f32>(anchor, (
                &loss_uniforms,
                prediction,
                target,
            ),
            self.dimension,
            batch_size,
        );

        //Run loss pipeline
        loss_pipeline.run(encoder, self.dimension, batch_size);
        
        //Return
        loss_pipeline.output_buffer
    }
}
