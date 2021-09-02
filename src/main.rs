mod data;
mod pipelines;
mod network;
//mod optimisers;

use futures::executor::block_on;

fn main() {
    let batch_size: usize = 32;
    
    let data_set = data::mnist::load_data("t10k").unwrap();

    let batch = data_set.generate_batch(batch_size);
    let batch_images = data::DataSet::<f32>::get_data(&batch);
    let batch_labels = data::DataSet::<f32>::get_labels(&batch);

    println!("Label:");
    println!("{:?}", batch_labels);

    let topology: Vec<usize> = vec![28*28, 1024, 1024, 512, 32, 10];
    let my_network = network::NeuralNetwork::new(topology);
    my_network.save_to_file("weights/network.bin");
    let my_network = network::NeuralNetwork::load_from_file("weights/network.bin");

    //Dereference data into vectors
    let input_data = {
        let mut vector: Vec<f32> = vec![0f32; 28*28 * batch_size];
        for (loc, data) in vector.iter_mut().zip(batch_images.iter()) {
            *loc = **data;
        }
        vector
    };
    let _label_data = {
        let mut vector: Vec<f32> = vec![0f32; 28*28 * batch_size];
        for (loc, data) in vector.iter_mut().zip(batch_labels.iter()) {
            *loc = **data;
        }
        vector
    };

    //Connect to device
    let anchor = block_on(pipelines::Device::new());

    //Load network to gpu
    let network_data = my_network.load_to_gpu(&anchor);

    //Run feedforward
    println!("Prediction 0:");
    println!("{:?}", my_network.feedforward::<f32>(&input_data, &network_data, &anchor, batch_size).unwrap());
    println!("Prediction 1:");
    println!("{:?}", my_network.feedforward::<f32>(&input_data, &network_data, &anchor, batch_size).unwrap());

}

