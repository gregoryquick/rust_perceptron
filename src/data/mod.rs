//! For handling labeled data that gets ingested into neural network

pub mod mnist;

use rand::prelude::*;

/// Single piece of data with label
#[derive(Clone)]
pub struct LabeledData<Data: Clone> {
    data: Vec<Data>,
    labels: Vec<Data>,
}

impl<Data: Clone> LabeledData<Data> {
    /// Returns clone of data
    fn get_data(&self) -> Vec<Data> {
        self.data.clone()
    }

    /// Returns clone of label
    fn get_labels(&self) -> Vec<Data> {
        self.labels.clone()
    }
}

/// Structure for managing whole collection of `LabeledData`
pub struct DataSet<Data: Clone> {
    data: Vec<LabeledData<Data>>,
}

impl<Data: Clone> DataSet<Data> {
    /// Creates vec of `DataSet` that contain batches of max size `batch_size`
    /// which contain clones of a random element of generating `DataSet`
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

    /// Creates `DataSet` of size `batch_size` of clones of random elements of `DataSet`
    pub fn generate_batch(&self, batch_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut batch_data: Vec<LabeledData<Data>> = self.data[..].choose_multiple(&mut rng, batch_size).cloned().collect();
        batch_data.shuffle(&mut rng);

        //Return
        DataSet::<Data> {
            data: batch_data,
        }
    }


    /// Return vec of clones of data from `DataSet`
    pub fn get_data(&self) -> Vec<Data> {
        let batch_data: Vec<Data> = self.data.iter().flat_map(|item| item.clone().get_data().into_iter()).collect();
        batch_data
    }
    
    /// Return vec of clones of labels from `DataSet`
    pub fn get_labels(&self) -> Vec<Data> {
        let batch_labels: Vec<Data> = self.data.iter().flat_map(|item| item.clone().get_labels().into_iter()).collect();
        batch_labels
    }

    /// Return size of `DataSet`
    pub fn get_size(&self) -> usize {
        self.data.len()
    }
}
