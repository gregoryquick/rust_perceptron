pub mod mnist;

use rand::prelude::*;

#[derive(Clone)]
pub struct LabeledData<Data: Clone> {
    data: Vec<Data>,
    labels: Vec<Data>,
}

impl<Data: Clone> LabeledData<Data> {
    fn get_data(&self) -> Vec<Data> {
        self.data.clone()
    }

    fn get_labels(&self) -> Vec<Data> {
        self.labels.clone()
    }
}

pub struct DataSet<Data: Clone> {
    data: Vec<LabeledData<Data>>,
}

impl<Data: Clone> DataSet<Data> {
    pub fn generate_epoc(&self, batch_size: usize) -> Vec<Self> {
        let mut rng = rand::thread_rng();
        let mut batch_data: Vec<LabeledData<Data>> = self.data.clone();
        batch_data.shuffle(&mut rng);

        //Return
        batch_data.chunks(batch_size).map(|batch_data| {
            DataSet::<Data> {
                data: batch_data.to_vec(),
            }
        }).collect()
    }

    pub fn generate_batch(&self, batch_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut batch_data: Vec<LabeledData<Data>> = self.data[..].choose_multiple(&mut rng, batch_size).cloned().collect();
        batch_data.shuffle(&mut rng);

        //Return
        DataSet::<Data> {
            data: batch_data,
        }
    }

    pub fn get_data(&self) -> Vec<Data> {
        let batch_data: Vec<Data> = self.data.iter().map(|item| item.clone().get_data().into_iter()).flatten().collect();
        batch_data
    }
    
    pub fn get_labels(&self) -> Vec<Data> {
        let batch_labels: Vec<Data> = self.data.iter().map(|item| item.clone().get_labels().into_iter()).flatten().collect();
        batch_labels
    }

    pub fn get_size(&self) -> usize {
        self.data.len()
    }
}
