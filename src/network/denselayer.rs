use crate::pipelines;

struct Denselayer {
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl Denselayer {
    fn load_buffers(self) -> Vec<wgpu::Buffer> {
        return Vec::new();
    }

    fn forward<T: bytemuck::Pod>(self, input: wgpu::Buffer,
                                 anchor: &pipelines::Device,
                                 encoder: &mut wgpu::CommandEncoder,
                                 output_dimension: usize,
                                 data_dimension: usize,
                                 batch_size:usize,) -> wgpu::Buffer {
        let device = &anchor.device;
        //Load data to gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let layer_weights: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.weights[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        let layer_biases: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.biases[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        let weight_pipeline = pipelines::matrixmultiply::Pipeline::new::<T>(anchor, (
                None,
                Some(layer_weights),
                Some(input),
                None,
            ),
            output_dimension,
            data_dimension,
            batch_size,
        );
        

        //Return
        weight_pipeline.output_buffer
    }
}

