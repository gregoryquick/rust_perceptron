use std::borrow::Cow;

use anyhow::{Result, anyhow};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::device::Device;
use crate::device::tensor::Tensor;
use crate::gpu::unpack_borrow;

pub fn forward<'a>(tensor_a: &Tensor,
                   tensor_b: &Tensor,
                   encoder: &mut wgpu::CommandEncoder,
                   wgpu_device: &wgpu::Device,
                   device: &'a Device,
                   ) -> Result<Vec<Tensor<'a>>> {
    //Unpack tensors
    let (tensor_a_data, &tensor_a_shape, &tensor_a_stride) = unpack_borrow(tensor_a)?;
    let (tensor_b_data, &tensor_b_shape, &tensor_b_stride) = unpack_borrow(tensor_b)?;

    //Verify tensor shapes
    if tensor_a_shape != tensor_b_shape || tensor_a_stride != tensor_b_stride {
        return Err(anyhow!("Tensors do no match in shape and stride"))
    }

    //Metadata for output_tensor
    let data_size = tensor_a_shape.0 * tensor_a_shape.1;
    let type_size = std::mem::size_of::<f32>();

    let output_tensor_shape = (tensor_a_shape.0, tensor_a_shape.1);
    let output_tensor_stride = (tensor_a_stride.0, tensor_a_stride.1);

    //Create buffer for tensor metadata
    #[allow(clippy::cast_possible_truncation)]
    let tensor_meta_buffer = {
        let data = [
            tensor_a_shape.0 as u32, tensor_a_shape.1 as u32,
            tensor_a_stride.0 as u32, tensor_a_stride.1 as u32,
            tensor_b_shape.0 as u32, tensor_b_shape.1 as u32,
            tensor_b_stride.0 as u32, tensor_b_stride.1 as u32,
            output_tensor_shape.0 as u32, output_tensor_shape.1 as u32,
            output_tensor_stride.0 as u32, output_tensor_stride.1 as u32,
        ];
        wgpu_device.create_buffer_init(
            &BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&data),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        )
    };

    //Create buffer for use in output tensor
    let output_buffer = wgpu_device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("Output buffer"),
            size: (type_size * data_size) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }
    );

    //Load wgsl shager
    let cs_module = wgpu_device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("matrixadd.wgsl"))),
    });

    //Create compute pipline from shader
    let compute_pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });
    
    //Create bind group
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: tensor_meta_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: tensor_a_data.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: tensor_b_data.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: output_buffer.as_entire_binding(),
        }],
    });

    //Create compute pass
    let mut compute_pass = encoder.begin_compute_pass(
        &wgpu::ComputePassDescriptor {
            label: Some("Elementwise Addition"),
        }
    );

    //Run compute pass
    compute_pass.set_pipeline(&compute_pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    compute_pass.dispatch(output_tensor_shape.0 as u32, output_tensor_shape.1 as u32, 1);
    
    //Return
    Err(anyhow!("matrix_add"))
}
