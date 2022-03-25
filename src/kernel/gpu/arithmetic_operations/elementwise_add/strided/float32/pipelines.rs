use std::borrow::Cow;

use wgpu::{BindGroupLayout, ComputePipeline};

use crate::device::GPU;

pub fn forward_pipeline<'a>(gpu: &'a GPU,) -> (BindGroupLayout, ComputePipeline) {
    //Create bind group layout
    let bind_group_layout = gpu.device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: false,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            },],
        }
    );

    let pipeline_layout = gpu.device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        }
    );

    //Load wgsl shader
    let cs_module = gpu.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("forward.wgsl"))),
    });


    //Create compute pipeline from shader
    let compute_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });

    //Return
    (bind_group_layout, compute_pipeline)
}

pub fn backward_pipeline<'a>(gpu: &'a GPU,) -> (BindGroupLayout, ComputePipeline) {
    //Create bind group layout
    let bind_group_layout = gpu.device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: false,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            },],
        }
    );

    let pipeline_layout = gpu.device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        }
    );

    //Load wgsl shader
    let cs_module = gpu.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("backward.wgsl"))),
    });


    //Create compute pipeline from shader
    let compute_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });

    //Return
    (bind_group_layout, compute_pipeline)
}

