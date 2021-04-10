#![feature(const_generics)]
mod pipelines;
mod data;

extern crate serde;
#[macro_use]
extern crate serde_derive;
mod arrays {
    use std::{convert::TryInto, marker::PhantomData};

    use serde::{
        de::{SeqAccess, Visitor},
        ser::SerializeTuple,
        Deserialize, Deserializer, Serialize, Serializer,
    };
    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        data: &[T; N],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_tuple(N)?;
        for item in data {
            s.serialize_element(item)?;
        }
        s.end()
    }

    struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
    where
        T: Deserialize<'de>,
    {
        type Value = [T; N];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(&format!("an array of length {}", N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // can be optimized using MaybeUninit
            let mut data = Vec::with_capacity(N);
            for _ in 0..N {
                match (seq.next_element())? {
                    Some(val) => data.push(val),
                    None => return Err(serde::de::Error::invalid_length(N, &self)),
                }
            }
            match data.try_into() {
                Ok(arr) => Ok(arr),
                Err(_) => unreachable!(),
            }
        }
    }
    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
    }
}

use rand::prelude::*;

use futures::executor::block_on;
use std::thread;
use std::fs::File;


fn main() {
    //Default stack size is 8 * 1024 * 1024
    const STACK_SIZE: usize = 8 * 1024 * 1024;    
    //Spawn thread with explicit stack size
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {run(false)})
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}

fn run(generate_new_weights: bool) {
    //Network parameters
    const DATA_DIM: usize = 28 * 28;
    const OUTPUT_DIM: usize = 10;

    //~ ~ C O D E ~ ~
    use rand::distributions::Uniform;
    let pipeline_manager = block_on(PipelineManager::new(DATA_DIM, OUTPUT_DIM));
    let mut rng = rand::thread_rng();

    
    //Get network weights
    const WEIGHT_SIZE: usize = DATA_DIM * OUTPUT_DIM;
    let dist = Uniform::new(-1.0,1.0);
    #[derive(Serialize, Deserialize, Debug)]
    struct Weights<const N: usize> {
        #[serde(with = "arrays")]
        arr: [f32; N],
    }

    let network_weights = {
        let mut vector: [f32; WEIGHT_SIZE] = [0f32; WEIGHT_SIZE];
        
        if generate_new_weights {
            for num in vector.iter_mut() {
                    *num = rng.sample(dist);
            }
            let file = File::create("weights/network.bin").unwrap();
            bincode::serialize_into(&file, &Weights::<WEIGHT_SIZE>{arr: vector}).unwrap();
        } else {
            let file = File::open("weights/network.bin").unwrap();
            let weight_data: Weights::<WEIGHT_SIZE> = bincode::deserialize_from(&file).unwrap();
            for (loc, data) in vector.iter_mut().zip(weight_data.arr.iter()) {
                *loc = *data;
            }
        }
        
        vector
    };
    
    //Load data
    let training_data = data::load_data("train").unwrap();
    const BATCH_SIZE: usize = 1;
    let batch_data: Vec<data::MnistImage> = training_data.into_iter().choose_multiple(&mut rng, BATCH_SIZE);
    let batch_labels: Vec<u8> = batch_data.iter().map(|x| x.classification).collect();
    let batch_images: Vec<Vec<f32>> = batch_data.into_iter().map(|x| x.image).collect(); 

    let input_vector = {
        const DATA_SIZE: usize = DATA_DIM * BATCH_SIZE;
        let mut vector: [f32; DATA_SIZE] = [0f32; DATA_SIZE];
        for (loc, data) in vector.iter_mut().zip(batch_images.into_iter().flatten()) {
            *loc = data;
        }
        vector
    };

    //Compute forward pass result
    let forward_pipeline = pipeline_manager.new_pipeline::<pipelines::ForwardPass, f32>(BATCH_SIZE);
    let backward_pipeline = pipeline_manager.new_pipeline::<pipelines::BackwardPass, f32>(BATCH_SIZE);

    let result = block_on(pipeline_manager.run_forward_pass::<f32>(forward_pipeline, &network_weights, &input_vector)).unwrap();
    let results: Vec<&[f32]> = result.chunks(OUTPUT_DIM).collect();
    println!("Result:");
    println!("{:?}", results);
}

#[allow(dead_code)]
struct PipelineManager {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    network_shape: (usize, usize),
}

impl PipelineManager{
    async fn new(input_size: usize, output_size: usize,) -> Self{
        //Get device
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
            },
        ).await.unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        ).await.unwrap();
        
        PipelineManager{
            adapter: adapter,
            device: device,
            queue: queue,
            network_shape: (input_size, output_size),
        }
    }

    fn new_pipeline<P: pipelines::Pipeline, T: bytemuck::Pod>(&self, batch_size: usize,) -> P {
        P::new::<T>(&self.device, self.network_shape.0, self.network_shape.1, batch_size)
    }

    async fn run_forward_pass<T: bytemuck::Pod>(&self, pipeline: pipelines::ForwardPass, network_weights: &[T], input_vector: &[T]) -> Option<Vec<T>> {
        use pipelines::*;
        let type_size = std::mem::size_of::<T>();

        //Create command encoder
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None 
            }
        );

        //Load data into gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let buffers = pipeline.get_buffers();
        let batch_size: usize = pipeline.get_batch_size();


        let weight_data_buffer = self.device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(network_weights),
                usage: wgpu::BufferUsage::COPY_SRC,
            }
        );
        encoder.copy_buffer_to_buffer(
            &weight_data_buffer, 0,
            &buffers[1], 0,
            (type_size * self.network_shape.0 * self.network_shape.1) as wgpu::BufferAddress,
        );
        
        let input_data_buffer = self.device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(input_vector),
                usage: wgpu::BufferUsage::COPY_SRC,
            }
        );
        encoder.copy_buffer_to_buffer(
            &input_data_buffer, 0,
            &buffers[2], 0,
            (type_size * self.network_shape.0 * batch_size) as wgpu::BufferAddress,
        );

        //Create the compute pass (Mutably borrows encoder)
        let mut compute_pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor { 
                label: None 
            }
        );
        
        //Add compute pipeline to compute pass
        compute_pass.set_pipeline(pipeline.get_compute_pipeline());
        let bind_groups = pipeline.get_bind_groups();
        compute_pass.set_bind_group(0, &bind_groups[0], &[]);
        //Work groups of x=output_size, Y = batch_size, Z = 1
        compute_pass.dispatch(self.network_shape.1 as u32, batch_size as u32, 1);
        
        //Encoder borrow is gone now!
        drop(compute_pass);

        //Copy output from gpu buffer to staging buffer on cpu
        let staging_buffer = self.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.network_shape.1 * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(
            &buffers[3], 0,
            &staging_buffer, 0,
            (type_size * self.network_shape.1 * batch_size) as wgpu::BufferAddress,
        );

        //Finish building command encoder and submit
        self.queue.submit(Some(encoder.finish()));

        //Create future of the computation
        let buffer_slice = staging_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        //Wait for computation to complete
        self.device.poll(wgpu::Maintain::Wait);

        //Handle computation result and return
        match buffer_future.await {
            Ok(()) => {
                //Get buffer contents
                let data = buffer_slice.get_mapped_range();
                let result: Vec<T> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<T>(b)).collect();
                
                //Drop mapped view
                drop(data);
                //Unmap buffer
                staging_buffer.unmap();

                Some(result)
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                None
            }
        }
    }
}
