pub struct Pipeline {
    pub uniform_buffer: wgpu::Buffer,
    pub weight_buffer: wgpu::Buffer,
    pub input_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    bind_group_0: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
}

impl Pipeline {
    // Contract a m x n matrix with a n x k matrix to make a m x k matrix
    pub fn new<T: bytemuck::Pod>(anchor: &super::Device,
                                 buffers: (Option<wgpu::Buffer>, // uniform buffer
                                           Option<wgpu::Buffer>, // m x n matrix
                                           Option<wgpu::Buffer>, // n x k matrix
                                           Option<wgpu::Buffer>),// output
                                 input_size: usize,
                                 output_size: usize,
                                 contract_size: usize,){
        let type_size = std::mem::size_of::<T>();
        let device = &anchor.device;
        //Create/load buffers
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        
        let uniform_buffer = buffers.0.unwrap_or({
            let uniform_data = [contract_size as u32, output_size as u32, input_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsage::UNIFORM,
                }
            )
        });
        //0-0

        let weight_buffer = buffers.1.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Network Weights"),
                    size: (type_size * output_size * contract_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        );
        //0-1

        let input_buffer = buffers.2.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Input Buffer"),
                    size: (type_size * contract_size * input_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        );
        //0-2
        
        let output_buffer = buffers.3.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Output buffer"),
                    size: (type_size * output_size * input_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
                    mapped_at_creation: false,
                }
            )
        );
        //0-3
        
        //Create bind group(s)
        let bind_group_layout_0 = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Matrix multiply bind group layout 0"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
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
                    visibility: wgpu::ShaderStage::COMPUTE,
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
                    visibility: wgpu::ShaderStage::COMPUTE,
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
        let bind_group_0 = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label:  Some("Matrix multiply bind group 0"),
                layout: &bind_group_layout_0,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },],
            }
        );

        //Create compute pipeline
    }
}
