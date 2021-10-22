//! Used for imporant mnist data from file

use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{Cursor, Read};
use crate::data::LabeledData;
use crate::data::DataSet;
use anyhow::Result;


/// Loads mnist data from location and converts to `DataSet<f32>`
pub fn load_data(dataset_name: &str) -> Result<DataSet<f32>> {
     let filename = format!("mnist/{}-labels-idx1-ubyte", dataset_name);
     let label_data = &FileData::new(&mut (File::open(filename))?)?;
     let filename = format!("mnist/{}-images-idx3-ubyte", dataset_name);
     let image_data = &FileData::new(&mut (File::open(filename))?)?;
     let mut images: Vec<Vec<f32>> = Vec::new();
     let image_shape = (image_data.sizes[1] * image_data.sizes[2]) as usize;
     
     for i in 0..image_data.sizes[0] as usize {
         let start = i * image_shape;
         let image_vec = image_data.data[start.. start + image_shape].to_vec();
         let image_data: Vec<f32> = image_vec.into_iter().map(|x| f32::from(x) / 255.).collect();
         images.push(image_data);
     }

     let classifications = label_data.data.clone().into_iter().map(|x| {
        let mut vec: Vec<f32> = vec![0_f32; 10];
        vec[x as usize] = 1.0;
        vec
     });
     
     let mut ret: Vec<LabeledData<f32>> = Vec::new();
     for (image, classification) in images.into_iter().zip(classifications) {
        ret.push(LabeledData::<f32> {
            data: image,
            labels: classification,
        });
    }
    Ok(DataSet::<f32> {
       data: ret
    })
}

/// Struct for the data contained in file
#[derive(Debug)]
struct FileData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl FileData {
    /// Reads `FileData` from `file`
    fn new(file: &mut File) -> Result<FileData> {
        let mut contents: Vec<u8> = Vec::new();
        file.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);
        let magic_number = r.read_i32::<BigEndian>()?;        

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        
        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;
        
        Ok(FileData{
            sizes,
            data,
        })
    }
}
