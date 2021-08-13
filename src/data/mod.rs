pub mod mnist;

use rand::prelude::*;

pub struct LabeledData<Data: bytemuck::Pod> {
    data: Vec<Data>,
    label: Vec<Data>,
}

impl<Data: bytemuck::Pod> LabeledData<Data> {
    fn get_data(&self) -> &Vec<Data> {
        &self.data
    }

    fn get_label(&self) -> &Vec<Data> {
        &self.label
    }
}

pub struct DataSet<Data: bytemuck::Pod> {
    data: Vec<LabeledData<Data>>,
}

impl<Data: bytemuck::Pod> DataSet<Data> {
    pub fn generate_batch(&self, batch_size: usize) -> Vec<&LabeledData<Data>> {
        let mut rng = rand::thread_rng();
        let batch_data: Vec<&LabeledData<Data>> = self.data.iter().choose_multiple(&mut rng, batch_size);
        return batch_data;
    }

    pub fn get_data<'a>(batch: &Vec<&'a LabeledData<Data>>) -> Vec<&'a Data> {
        let batch_data: Vec<&Data> = batch.iter().map(|item| item.get_data().iter()).flatten().collect();
        batch_data
    }
    
    pub fn get_labels<'a>(batch: &Vec<&'a LabeledData<Data>>) -> Vec<&'a Data> {
        let batch_labels: Vec<&Data> = batch.iter().map(|item| item.get_label().iter()).flatten().collect();
        batch_labels
    }
}
