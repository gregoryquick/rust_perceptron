use crate::pipelines;

pub struct Stochasticgradientdescent {
    learning_rate: f32,
}

impl Stochasticgradientdescent {
    pub fn new(learning_rate: f32) -> Self{
        Stochasticgradientdescent {
            learning_rate,
        }
    }

    pub fn partial_step(&mut self, anchor: &pipelines::Device, encoder: &mut wgpu::CommandEncoder, 
                current_value: wgpu::Buffer, gradient: wgpu::Buffer,
                output_dimension: usize, data_dimension: usize,) -> wgpu::Buffer {
        let device = &anchor.device;
        
        //Load learning rate into gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let scalar_buffer: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[self.learning_rate]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        //Create learning rate application pipeline
        let learning_pipeline = pipelines::scalarmultiply::Pipeline::new::<f32>(anchor, (
                None,
                Some(scalar_buffer),
                Some(gradient),
                None,
            ),
            output_dimension,
            data_dimension,
        );

        //Run learning pipeline
        learning_pipeline.run(anchor, encoder, output_dimension, data_dimension);

        //Create descent application pipeline
        let descent_pipeline = pipelines::elementsubtract::Pipeline::new::<f32>(anchor, (
                Some(learning_pipeline.uniform_buffer),
                Some(current_value),
                Some(learning_pipeline.output_buffer),
                None,
            ),
            output_dimension,
            data_dimension,
        );

        //Run descent pipeline
        descent_pipeline.run(anchor, encoder, output_dimension, data_dimension);

        //Return
        descent_pipeline.output_buffer
    }

    //This is for the ending of the backpropagation phase
    pub fn step(&mut self){
    }
}
