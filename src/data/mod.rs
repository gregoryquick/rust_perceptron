pub mod mnist;

use rand::prelude::*;

pub struct DataSet<T: LabeledData<f32,u8>> {
    data: Vec<T>,
}

impl DataSet<mnist::Data> {
    pub fn new(dataset_filename: &str) -> Self {
        DataSet::<mnist::Data>{
            data: mnist::load_data(dataset_filename).unwrap()
        }
    }
    
    pub fn generate_batch(&self, batch_size: usize) -> Vec<&mnist::Data> {
        let mut rng = rand::thread_rng();
        let batch_data: Vec<&mnist::Data> = self.data.iter().choose_multiple(&mut rng, batch_size);
        return batch_data;
    }

    pub fn get_data<'a>(batch: &Vec<&'a mnist::Data>) -> Vec<&'a Vec<f32>> {
        let batch_data: Vec<&Vec<f32>> = batch.iter().map(|item| item.get_data()).collect();
        batch_data
    }
    pub fn get_labels<'a>(batch: &Vec<&'a mnist::Data>) -> Vec<&'a u8> {
        let batch_labels: Vec<&u8> = batch.iter().map(|item| item.get_label()).collect();
        batch_labels
    }
}

pub trait LabeledData<Data: bytemuck::Pod, Label: bytemuck::Pod> {
    fn get_data(&self) -> &Vec<Data>;

    fn get_label(&self) -> &Label;
}

