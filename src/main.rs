mod data;
mod pipelines;
mod network;
mod optimisers;

use futures::executor::block_on;

fn main() {
    //Logging
    use std::env;
    env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();

    //Global vars
    let output_size: usize = 10;
    let batch_size: usize = 128;
    
    //Load data
    let training_data = data::mnist::load_data("train").unwrap();
    let test_data = data::mnist::load_data("t10k").unwrap();

    //Create/Load network
    use network::LayerType::*;
    use network::CostFunction::*;
    let generator_topology = vec![FullyConnected(128), Relu, FullyConnected(output_size), Softmax];
    let mut my_network = network::perceptron::Network::new(28*28, generator_topology, CrossEntropy);
    //let mut my_network = network::perceptron::Network::load_from_file("weights/network.bin");

    let mut optimiser = optimisers::Stochasticgradientdescent::new(0.001);

    let network_topology = my_network.get_topology();

    //Connect to device
    let anchor = block_on(pipelines::Device::new());

    //Load network data to gpu
    let mut network_data = my_network.load_to_gpu(&anchor);

    //Run training loop
    for i in 0..1 {
        let mut j = 0;
        //Break epoc into batches
        for batch in training_data.generate_epoc(batch_size).into_iter() {
            println!("Epoc: {}, Batch: {}", i, j);
            j += 1;
            let batch_images = batch.get_data();
            let batch_labels = batch.get_labels();
            
            //Step optimization
            let network_grads = my_network.backprop::<f32>(&batch_images, &batch_labels, &mut network_data, &anchor, batch.get_size());
            optimiser.step(&mut network_data, &network_grads, &anchor, &network_topology);

            //Get test batch
            let test_batch = test_data.generate_batch(batch_size);
            let batch_images = test_batch.get_data();
            let batch_labels = test_batch.get_labels();

            //Compute cost
            let prediction = my_network.feedforward::<f32>(&batch_images, &network_data, &anchor, batch_size);
            let cost = my_network.cost::<f32>(&prediction, &batch_labels, &anchor, batch_size, true);
            println!("Cost: {:?}", from_gpu::<f32>(&cost, &anchor, 1).unwrap());
        
            //Show sample prediction with ground truth for it
            println!("{:?}", from_gpu::<f32>(&prediction, &anchor, output_size).unwrap());
            println!("{:?}", batch_labels.get(0..output_size));

        }
        //End epoc
    }

    //Save network
    my_network.save_from_gpu(&anchor, &network_data);
    my_network.save_to_file("weights/network.bin");
}

//fn to_gpu<T: bytemuck::Pod>(input: &Vec<T>, anchor: &pipelines::Device) -> wgpu::Buffer {
//    let device = &anchor.device;
//
//    //Load data to gpu
//    use wgpu::util::{BufferInitDescriptor, DeviceExt};
//    let input_buffer = device.create_buffer_init(
//        &BufferInitDescriptor {
//            label: None,
//            contents: bytemuck::cast_slice(&input[..]),
//            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//        }
//    );
//
//    //Return
//    input_buffer
//}

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
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
