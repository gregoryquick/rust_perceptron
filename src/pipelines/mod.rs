pub mod applyweights;
pub mod leakyrelu;
pub mod leakyreluprime;
pub mod loss;
pub mod backprop;
pub mod descendgrad;
pub mod error;


pub struct PipelineAnchor {
    pub _adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub input_size: usize,
    pub output_size: usize,
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
            _adapter: adapter,
            device,
            queue,
            input_size,
            output_size,
        }
    }
}

pub async fn run_batch_error<T: bytemuck::Pod>(anchor: &PipelineAnchor, network_weights: wgpu::Buffer, input_data: &Vec<T>, label_data: &Vec<T>, batch_size: usize) -> Option<(T, wgpu::Buffer)> {
    let type_size = std::mem::size_of::<T>();
    let device = &anchor.device;

    //Create command encoder
    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: None 
        }
    );
    
    //Load data for pipeline
    use wgpu::util::{BufferInitDescriptor, DeviceExt};
    let weight_data_buffer = network_weights;
    
    let input_data_buffer = device.create_buffer_init(
        &BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input_data[..]),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        }
    );

    let label_data_buffer = device.create_buffer_init(
        &BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&label_data[..]),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        }
    );

    //Create first pipeline
    let matrix_pipeline = applyweights::Pipeline::new::<T>(anchor, (Some(weight_data_buffer), Some(input_data_buffer), None), batch_size);

    //Load info for running pipeline into encoder
    matrix_pipeline.run(anchor, &mut encoder, batch_size);

    //Create second pipeline
    let activation_pipeline = leakyrelu::Pipeline::new::<T>(anchor, (Some(matrix_pipeline.output_buffer), None), batch_size);
    
    //Run pipeline
    activation_pipeline.run(anchor, &mut encoder, batch_size);

    //Create loss pipeline
    let loss_pipeline = loss::Pipeline::new::<T>(anchor, (Some(activation_pipeline.output_buffer), Some(label_data_buffer), None), batch_size);
    
    //Run loss pipeline
    loss_pipeline.run(anchor, &mut encoder, batch_size);

    //Create batch error pipeline
    let error_pipeline = error::Pipeline::new::<T>(anchor, (Some(loss_pipeline.output_buffer), None), batch_size);
    
    //Run error pipeline
    error_pipeline.run(anchor, &mut encoder, batch_size);

    //Copy data out of gpu
    let staging_buffer = device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            size: (type_size) as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        }
    );
    encoder.copy_buffer_to_buffer(
        &error_pipeline.output_buffer, 0,
        &staging_buffer, 0,
        (type_size) as wgpu::BufferAddress,
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
            let result: T = *bytemuck::from_bytes::<T>(&data);
                
            //Drop mapped view
            drop(data);
            //Unmap buffer
            staging_buffer.unmap();

            Some((result, matrix_pipeline.weight_buffer))
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            None
        }
    }
}
