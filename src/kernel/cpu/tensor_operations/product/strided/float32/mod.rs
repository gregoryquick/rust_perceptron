use enum_extract::extract;
use itertools::Itertools;

use crate::tensor::*;
use crate::device::CPU;

pub fn forward<'a, const N_a: usize, const N_b: usize>(
    cpu: &'a CPU,
    tensor_a: &Tensor<'a, CPU, Strided<N_a>, f32, N_a>,
    tensor_b: &Tensor<'a, CPU, Strided<N_b>, f32, N_b>,
    ) -> Tensor<'a, CPU, Strided<{N_a + N_b}>, f32, {N_a + N_b}> {
        //Unpack tensors
        let Tensor {
            device: cpu_a,
            tensor_layout: Strided {
                strides: tensor_a_strides,
            },
            shape: tensor_a_shape,
            data: tensor_a_data,
        } = tensor_a;

        let Tensor {
            device: cpu_b,
            tensor_layout: Strided {
                strides: tensor_b_strides,
            },
            shape: tensor_b_shape,
            data: tensor_b_data,
        } = tensor_b;

        //Check if tensor devices match
        assert!(!(cpu as *const _ != *cpu_a as *const _), "Tensor device mismatch");
        assert!(!(cpu as *const _ != *cpu_b as *const _), "Tensor device mismatch");

        //Calculate important values for internal use
        let a_len = tensor_a_shape.iter().count();
        let type_size = std::mem::size_of::<f32>();

        //Create meta data values
        let output_tensor_shape = {
            let mut shape = [0 as usize; N_a + N_b];
            for (location, value) in shape.iter_mut().zip(tensor_a_shape.iter().cloned().chain(tensor_b_shape.iter().cloned())) {
                *location = value;
            }
            shape
        };

        let size: usize = output_tensor_shape.iter().product();
        let stride_factor: usize = tensor_b_shape.iter().product();
        
        let output_tensor_strides = {
            let mut strides = [0 as usize; N_a + N_b];
            for (location, value) in strides.iter_mut().zip(tensor_a_strides.iter().cloned().map(|v| stride_factor * v).chain(tensor_b_strides.iter().cloned())) {
                *location = value;
            }
            strides
        };

        //Unwrap tensor data
        let vec_a = extract!(TensorData::CPUStrided(_), tensor_a_data).unwrap();
        let vec_b = extract!(TensorData::CPUStrided(_), tensor_b_data).unwrap();

        //Create vec for use in output tensor
        let output_vec: Vec<f32> = vec_a.iter().cartesian_product(vec_b.iter()).map(|(a, b)| a * b).collect();

        //Return
        Tensor {
            device: cpu,
            tensor_layout: Strided {
                strides: output_tensor_strides,
            },
            shape: output_tensor_shape,
            data: TensorData::CPUStrided::<f32>(output_vec),
        }
}
