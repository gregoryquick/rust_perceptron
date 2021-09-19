pub struct Pipeline {
    pub output_buffer: wgpu::Buffer,
    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
}

impl Pipeline {
    //Take an 2 m-length vectors of mean and variance of batch and use to compute batchnorm of an m x n matrix
    pub fn new<T: bytemuck::Pod>(anchor: &super::Device,
                                 buffers: (&wgpu::Buffer, // uniform buffer
                                           &wgpu::Buffer, // m x n matrix
                                           &wgpu::Buffer, // m-length vector
                                           &wgpu::Buffer,),//m-length vector
                                m_size: usize,
                                n_size: usize,) -> Self {
        let type_size = std::mem::size_of::<T>();
        let device = &anchor.device;

        //Create/load buffers
        
        let uniform_buffer = buffers.0;
        //0-0

        let matrix_buffer = buffers.1;
        //0-1

        let batch_mean = buffers.2;
        //0-2

        let batch_var = buffers.3;
        //0-3
        
        let output_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Output buffer"),
                size: (type_size * m_size * n_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        //1-0
        
        //Create bind group(s)
        let bind_group_layout_0 = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Batch Normilization bind group layout 0"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
                },],
            }
        );
        let bind_group_0 = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label:  Some("Batch Normilization bind group 0"),
                layout: &bind_group_layout_0,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: batch_mean.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: batch_var.as_entire_binding(),
                },],
            }
        );

        let bind_group_layout_1 = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Batch Normilization bind group layout 1"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
        let bind_group_1 = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label:  Some("Batch Normilization bind group 1"),
                layout: &bind_group_layout_1,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },],
            }
        );

        //Create compute pipeline
        let cs_src = include_str!("shader.comp");
        let mut compiler = shaderc::Compiler::new().unwrap();
        let cs_spirv = compiler.compile_into_spirv(cs_src, shaderc::ShaderKind::Compute, "batchnorm.comp", "main", None).unwrap();
        let cs_module = device.create_shader_module(
            &wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::util::make_spirv(cs_spirv.as_binary_u8()),
            }
        );
        
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
                push_constant_ranges: &[],
            }
        );

        let compute_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Batch Normilization pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );

         Pipeline {
            output_buffer,
            bind_group_0,
            bind_group_1,
            compute_pipeline,
        }
    }

    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, m_size: usize, n_size: usize,) {
        //Create compute pass
        let mut compute_pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: Some("Batch Normilization"),
            }
        );

        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.bind_group_0, &[]);
        compute_pass.set_bind_group(1, &self.bind_group_1, &[]);
        //Work groups of X = m_size, Y = n_size, Z = 1
        compute_pass.dispatch(m_size as u32, n_size as u32, 1);
    }
}
