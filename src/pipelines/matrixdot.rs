pub struct Pipeline {
    uniform_buffer: wgpu::Buffer,
    weight_buffer: wgpu::Buffer,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    bind_group_0: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
}
impl Pipeline {
    pub fn new<T: bytemuck::Pod>(anchor: &super::PipelineAnchor, batch_size: usize,) -> Self {
        let type_size = std::mem::size_of::<T>();
        let input_size = anchor.input_size;
        let output_size =  anchor.output_size;
        let device = &anchor.device;
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        //Create buffers
        let uniform_data = [input_size as u32, output_size as u32, batch_size as u32];
        let uniform_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniform_data),
                usage: wgpu::BufferUsage::UNIFORM,
            }
        );
        //0-0

        let weight_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Network Weights"),
                size: (type_size * input_size * output_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        //0-1

        let input_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Input Buffer"),
                size: (type_size * input_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        //0-2
        
        let output_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Output buffer"),
                size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        //0-3
        
        //Create bind group(s)
        let bind_group_layout_0 = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Matrix dot product bind group layout 0"),
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
                label:  Some("Matrix dot product bind group 0"),
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
        let cs_src = include_str!("shaders/matrixdot.comp");
        let mut compiler = shaderc::Compiler::new().unwrap();
        let cs_spirv = compiler.compile_into_spirv(cs_src, shaderc::ShaderKind::Compute, "shader.comp", "main", None).unwrap();
        let cs_module = device.create_shader_module(
            &wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::util::make_spirv(cs_spirv.as_binary_u8()),
                flags: wgpu::ShaderFlags::empty(),
            }
        );

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout_0],
                push_constant_ranges: &[],
            }
        );

        let compute_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Matrix dot product pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );
        

        //Return
        Pipeline {
            uniform_buffer,
            weight_buffer,
            input_buffer,
            output_buffer,
            bind_group_0,
            compute_pipeline,
        }
    }
}
