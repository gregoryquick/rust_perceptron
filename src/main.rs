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

    //let topology: Vec<usize> = vec![28*28, 10];
    //let my_network = network::NeuralNetwork::new(topology);
    //my_network.save("weights/network.bin");
    let my_network = network::NeuralNetwork::load("weights/network.bin");

    println!("Prediction:");
    let input_data = {
        let mut vector: Vec<f32> = vec![0f32; 28*28 * batch_size];
        for (loc, data) in vector.iter_mut().zip(batch_images.iter()) {
            *loc = **data;
        }
        vector
    };
    println!("{:?}", my_network.feedforward::<f32>(input_data, batch_size).unwrap());

    //This is to make sure I have not moved anything I shouldn't have
    println!("Prediction again");
    //I could probly make it so feedforwad only borrows the input
    let input_data = {
        let mut vector: Vec<f32> = vec![0f32; 28*28 * batch_size];
        for (loc, data) in vector.iter_mut().zip(batch_images.iter()) {
            *loc = **data;
        }
        vector
    };
    println!("{:?}", my_network.feedforward::<f32>(input_data, batch_size).unwrap());

}

