mod matrixdot;
mod leakyrelu;

pub struct PipelineAnchor {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    input_size: usize,
    output_size: usize,
}

impl PipelineAnchor {
    pub async fn new(input_size: usize, output_size: usize,) -> Self {
        //Get device
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
            },
        ).await.unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        ).await.unwrap();
        
        //Return
        PipelineAnchor {
            adapter,
            device,
            queue,
            input_size,
            output_size,
        }
    }
}

pub async fn run_forward_pass<T: bytemuck::Pod>(anchor: &PipelineAnchor, network_weights: &Vec<T>, input_data: &Vec<T>, batch_size: usize) -> Option<Vec<T>>{
    let type_size = std::mem::size_of::<T>();
    let device = &anchor.device;

    //Create command encoder
    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: None 
        }
    );

    //Create first pipeline
    let matrix_pipeline = matrixdot::Pipeline::new::<T>(anchor, (None, None, None), batch_size);

    //Load data into pipeline
    use wgpu::util::{BufferInitDescriptor, DeviceExt};
    let weight_data_buffer = device.create_buffer_init(
        &BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&network_weights[..]),
            usage: wgpu::BufferUsage::COPY_SRC,
        }
    );
    encoder.copy_buffer_to_buffer(
        &weight_data_buffer, 0,
        &matrix_pipeline.weight_buffer, 0,
        (type_size * anchor.input_size * anchor.output_size) as wgpu::BufferAddress,
    );

    let input_data_buffer = device.create_buffer_init(
        &BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input_data[..]),
            usage: wgpu::BufferUsage::COPY_SRC,
        }
    );
    encoder.copy_buffer_to_buffer(
        &input_data_buffer, 0,
        &matrix_pipeline.input_buffer, 0,
        (type_size * anchor.input_size * batch_size) as wgpu::BufferAddress,
    );

    //Load info for running pipeline into encoder
    matrix_pipeline.run(anchor, &mut encoder, batch_size);

    //Create second pipeline
    let activation_pipeline = leakyrelu::Pipeline::new::<T>(anchor, (Some(matrix_pipeline.output_buffer), None), batch_size);

    //No data loading needed
    
    //Run pipeline
    activation_pipeline.run(anchor, &mut encoder, batch_size);

    //Copy data out of gpu
    let staging_buffer = device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            size: (type_size * anchor.output_size * batch_size) as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        }
    );

    encoder.copy_buffer_to_buffer(
        &activation_pipeline.output_buffer, 0,
        &staging_buffer, 0,
        (type_size * anchor.output_size * batch_size) as wgpu::BufferAddress,
    );

    let queue = &anchor.queue;

    queue.submit(Some(encoder.finish()));

    //Create future of the computation
    let buffer_slice = staging_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
    
    //Wait for computation to complete
    device.poll(wgpu::Maintain::Wait);

    match buffer_future.await {
        Ok(()) => {
            //Get buffer contents
            let data = buffer_slice.get_mapped_range();
            //Convert to T and apply activation function
            let result: Vec<T> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<T>(b)).collect();
                
            //Drop mapped view
            drop(data);
            //Unmap buffer
            staging_buffer.unmap();

            Some(result)
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            None
        }
    }
}

