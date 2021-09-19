mod data;
mod pipelines;
mod network;
mod optimisers;

use futures::executor::block_on;

fn main() {
    //Global vars
    let output_size: usize = 10;
    let batch_size: usize = 32;
    
    //Load data
    let data_set = data::mnist::load_data("train").unwrap();

    //Create/Load network
    use network::LayerType::*;
    let generator_topology = vec![Batchnorm, DenseLayer(1024), Batchnorm, DenseLayer(1024), Batchnorm, DenseLayer(output_size), Batchnorm];
    let mut my_network = network::perceptron::Network::new(28*28, generator_topology);
    //my_network.save_to_file("weights/network.bin");
    //let mut my_network = network::perceptron::Network::load_from_file("weights/network.bin");

    let mut optimiser = optimisers::Stochasticgradientdescent::new(0.01);

    let network_topology = my_network.get_topology();

    //Connect to device
    let anchor = block_on(pipelines::Device::new());

    //Load network data to gpu
    let mut network_data = my_network.load_to_gpu(&anchor);

    //Get batch
    let batch = data_set.generate_batch(batch_size);
    let batch_images = data::DataSet::<f32>::get_data(&batch);
    let batch_labels = data::DataSet::<f32>::get_labels(&batch);

    //Load batch data into vectors
    let input_data = {
        let mut vector: Vec<f32> = Vec::with_capacity(28*28 * batch_size);
        for data in batch_images.into_iter() {
            vector.push(*data);
        }
        vector
    };

    let label_data = {
        let mut vector: Vec<f32> = Vec::with_capacity(28*28 * batch_size);
        for data in batch_labels.into_iter() {
            vector.push(*data);
        }
        vector
    };

    //Initial prediction with weights (and make sure that mean and var estimates are initilized)
    my_network.backprop::<f32>(&input_data, &label_data, &mut network_data, &anchor, batch_size);
    println!("Prediction 0:");
    let prediction = my_network.feedforward::<f32>(&input_data, &network_data, &anchor, batch_size);
    let cost = my_network.cost::<f32>(&prediction, &label_data, &anchor, batch_size);
    println!("{:?}", from_gpu::<f32>(&cost, &anchor, batch_size).unwrap());
    
    //Save network
    my_network.save_from_gpu(&anchor, &network_data);
    my_network.save_to_file("weights/network.bin");

    //Run training loop
    for i in 1..20 {
        let network_grads = my_network.backprop::<f32>(&input_data, &label_data, &mut network_data, &anchor, batch_size);
        optimiser.step(&mut network_data, &network_grads, &anchor, &network_topology);
        println!("Prediction {}:", i);
        let prediction = my_network.feedforward::<f32>(&input_data, &network_data, &anchor, batch_size);
        let cost = my_network.cost::<f32>(&prediction, &label_data, &anchor, batch_size);
        println!("{:?}", from_gpu::<f32>(&cost, &anchor, batch_size).unwrap());
    }

    //Save network
    //my_network.save_from_gpu(&anchor, &network_data);
    //my_network.save_to_file("weights/network.bin");
}

fn to_gpu<T: bytemuck::Pod>(input: &Vec<T>, anchor: &pipelines::Device) -> wgpu::Buffer {
    let device = &anchor.device;

    //Load data to gpu
    use wgpu::util::{BufferInitDescriptor, DeviceExt};
    let input_buffer = device.create_buffer_init(
        &BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input[..]),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        }
    );

    //Return
    input_buffer
}

fn from_gpu<T: bytemuck::Pod>(buffer: &wgpu::Buffer, anchor: &pipelines::Device, size: usize) -> Option<Vec<T>> {
    let queue = &anchor.queue;
    let device = &anchor.device;
    let type_size = std::mem::size_of::<T>();

    //Create command buffer encoder
    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: None,
        }
    );

    //Copy to readable buffer
    let staging_buffer = device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            size: (type_size * size) as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        }
    );
    encoder.copy_buffer_to_buffer(
        buffer, 0,
        &staging_buffer, 0,
        (type_size * size) as wgpu::BufferAddress,
    );

    //Submit commands to gpu
    queue.submit(Some(encoder.finish()));

    //Create future of the computation
    let buffer_slice = staging_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        
    //Register mapping callbacks
    device.poll(wgpu::Maintain::Wait);

    //Wait for computation to complete
    block_on(async {
        match buffer_future.await {
            Ok(()) => {
                //Get buffer contents
                let data = buffer_slice.get_mapped_range();
                //Convert to T
                let result: Vec<T> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<T>(b)).collect();
                //Drop mapped view
                drop(data);
                //Unmap buffer
                staging_buffer.unmap();

                //Return
                return Some(result)
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                None
            }
        }
    })
}
