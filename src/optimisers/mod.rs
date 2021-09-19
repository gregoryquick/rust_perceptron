use crate::pipelines;

use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub struct Stochasticgradientdescent {
    learning_rate: f32,
}

impl Stochasticgradientdescent {
    pub fn new(learning_rate: f32) -> Self{
        Stochasticgradientdescent {
            learning_rate,
        }
    }

    pub fn step(&mut self,
                network_data: &mut Vec<Vec<wgpu::Buffer>>,
                network_grad: &Vec<Vec<Option<wgpu::Buffer>>>,
                anchor: &pipelines::Device,
                network_topology: &Vec<Vec<(usize, usize)>>,) {
        let queue = &anchor.queue;
        let device = &anchor.device;
        
        //Load data to gpu
        let learning_rate: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[self.learning_rate]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }
        );

        //Create command buffer encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Iterate though network to update data
        let network_iterator = network_data.into_iter()
            .zip(network_grad.into_iter())
            .zip(network_topology.into_iter());
        for ((layer_data, layer_grads), layer_topology) in network_iterator {
            let layer_iterator = layer_data.into_iter()
                .zip(layer_grads.into_iter())
                .zip(layer_topology.into_iter());
            for ((data, grad), data_topology) in layer_iterator {
                match grad {
                    Some(grad_buffer) => {
                        //Do stuff and things
                        let &(output_size, input_size) = data_topology;

                        //Create learning_rate pipeline
                        let learning_rate_uniforms = {
                            let uniform_data = [output_size as u32, input_size as u32];
                            device.create_buffer_init(
                                &BufferInitDescriptor {
                                    label: Some("Uniform Buffer"),
                                    contents: bytemuck::bytes_of(&uniform_data),
                                    usage: wgpu::BufferUsages::UNIFORM,
                                }
                            )
                        };

                        let learning_rate_pipeline = pipelines::scalarmultiply::Pipeline::new::<f32>(anchor, (
                                &learning_rate_uniforms,
                                &learning_rate,
                                grad_buffer,
                            ),
                            output_size,
                            input_size,
                        );

                        //Create learning_rate pipeline
                        learning_rate_pipeline.run(&mut encoder, output_size, input_size);

                        //Create update pipeline
                        let update_pipeline = pipelines::addvectortobatch::Pipeline::new::<f32>(anchor, (
                                &learning_rate_uniforms,
                                data,
                                &learning_rate_pipeline.output_buffer,
                            ),
                            output_size,
                            input_size,
                        );

                        //Create update pipeline
                        update_pipeline.run(&mut encoder, output_size, input_size);

                        //Update network values
                        *data = update_pipeline.output_buffer
                    }
                    None => {
                        //Do nothing
                    }
                }
            }
        }

        //Submit encoder
        queue.submit(Some(encoder.finish()));
    }
}
