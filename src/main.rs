mod data;
mod pipelines;
mod network;

fn main() {
    let batch_size: usize = 1;
    
    let data_set = data::mnist::load_data("t10k").unwrap();

    let batch = data_set.generate_batch(batch_size);
    //let batch_images = data::DataSet::<f32>::get_data(&batch);
    let batch_labels = data::DataSet::<f32>::get_labels(&batch);

    println!("Label:");
    println!("{:?}", batch_labels)
}

