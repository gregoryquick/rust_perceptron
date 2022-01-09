use enum_extract::extract;

use crate::tensor::*;
use crate::device::CPU;

pub fn forward<'a, const N: usize>(
    cpu: &'a CPU,
    tensor_a: &Tensor<'a, CPU, Strided<N>, f32, N>,
    tensor_b: &Tensor<'a, CPU, Strided<N>, f32, N>
    ) -> Tensor<'a, CPU, Strided<N>, f32, N> {
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

        //Check if tensor shapes match
        assert!(!(tensor_a_shape != tensor_b_shape), "Tensor shape mismatch");

        //Calculate important values for internal use
        let type_size = std::mem::size_of::<f32>();

        //Create meta data values
        let output_tensor_shape = tensor_a_shape.clone();

        let size: usize = output_tensor_shape.iter().product();
        
        let output_tensor_strides = tensor_a_strides.clone();
        
        //Unwrap tensor data
        let vec_a = extract!(TensorData::CPUStrided(_), tensor_a_data).unwrap();
        let vec_b = extract!(TensorData::CPUStrided(_), tensor_b_data).unwrap();

        //Create vec for use in output tensor
        let output_vec: Vec<f32> = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a + b).collect();

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


