mod data;
mod pipelines;
mod network;
mod optimisers;

use futures::executor::block_on;

fn main() {
    let batch_size: usize = 32;
    
    let data_set = data::mnist::load_data("t10k").unwrap();

    let batch = data_set.generate_batch(batch_size);
    let batch_images = data::DataSet::<f32>::get_data(&batch);
    let batch_labels = data::DataSet::<f32>::get_labels(&batch);

    println!("Label:");
    println!("{:?}", batch_labels);

    let generator_topology: Vec<usize> = vec![28*28, 1024, 1024, 10];
    let my_network = network::NeuralNetwork::new(generator_topology);
    my_network.save_to_file("weights/network.bin");
    let mut my_network = network::NeuralNetwork::load_from_file("weights/network.bin");

    let mut optimiser = optimisers::Stochasticgradientdescent::new(0.1);

    let network_topology = my_network.get_topology();

    //Dereference data into vectors
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

    //Connect to device
    let anchor = block_on(pipelines::Device::new());

    //Load network to gpu
    let mut network_data = my_network.load_to_gpu(&anchor);

    //Run training loop
    println!("Prediction 0:");
    println!("{:?}", my_network.feedforward::<f32>(&input_data, &network_data, &anchor, batch_size).unwrap());
    for i in 1..100 {
        let network_grads =  my_network.backprop::<f32>(&input_data, &label_data, &mut network_data, &anchor, batch_size);
        optimiser.step(&mut network_data, &network_grads, &anchor, &network_topology);
        println!("Prediction {}:", i);
        println!("{:?}", my_network.feedforward::<f32>(&input_data, &network_data, &anchor, batch_size).unwrap());
    }

    //Save network
    //my_network.save_from_gpu(&anchor, &network_data);
    //my_network.save_to_file("weights/network.bin");
}

