use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{Cursor, Read};
use crate::data::LabeledData;

pub struct Data {
    pub image: Vec<f32>,
    pub classification: u8,
}

impl LabeledData<f32,u8> for Data {
    fn get_data(&self) -> &Vec<f32> {
        &self.image
    }
    
    fn get_label(&self) -> &u8 {
        &self.classification
    }

    fn from_label(&self) -> Vec<f32> {
        let mut vec: Vec<f32> = vec![0f32; 10];
        vec[self.classification as usize] = 1.0;
        vec
    }
}

pub(super) fn load_data(dataset_name: &str) -> Result<Vec<Data>, std::io::Error> {
     let filename = format!("mnist/{}-labels-idx1-ubyte", dataset_name);
     let label_data = &FileData::new(&mut (File::open(filename))?)?;
     let filename = format!("mnist/{}-images-idx3-ubyte", dataset_name);
     let image_data = &FileData::new(&mut (File::open(filename))?)?;
     let mut images: Vec<Vec<f32>> = Vec::new();
     let image_shape = (image_data.sizes[1] * image_data.sizes[2]) as usize;
     
     for i in 0..image_data.sizes[0] as usize {
         let start = i * image_shape;
         let image_vec = image_data.data[start.. start + image_shape].to_vec();
         let image_data: Vec<f32> = image_vec.into_iter().map(|x| x as f32 / 255.).collect();
         images.push(image_data);
     }

     let classifications: Vec<u8> = label_data.data.clone();
     
     let mut ret: Vec<Data> = Vec::new();
     for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(Data {
            image,
            classification,

        })
    }
     Ok(ret)
}

#[derive(Debug)]
struct FileData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl FileData {
    fn new(file: &mut File) -> Result<FileData, std::io::Error> {
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
            sizes: sizes,
            data: data,
        })
    }
}
