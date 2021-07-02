mod data;
mod pipelines;
mod network;

fn main() {
    let batch_size: usize = 1;
    
    let data_set = data::mnist::load_data("t10k").unwrap();

    let batch = data_set.generate_batch(batch_size);
    let batch_images = data::DataSet::<f32>::get_data(&batch);
    let batch_labels = data::DataSet::<f32>::get_labels(&batch);

    println!("Label:");
    println!("{:?}", batch_labels);

    let my_network = network::NeuralNetwork::new(28*28, 10);

    println!("Prediction:");
    let input_data = {
        let mut vector: Vec<f32> = vec![0f32; 28*28 * batch_size];
        for (loc, data) in vector.iter_mut().zip(batch_images.into_iter()) {
            *loc = *data;
        }
        vector
    };
    println!("{:?}", my_network.feedforward::<f32>(input_data, batch_size).unwrap());
}

