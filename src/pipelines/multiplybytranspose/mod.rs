pub struct Pipeline {
    pub uniform_buffer: wgpu::Buffer,
    pub matrix_a_buffer: wgpu::Buffer,
    pub matrix_b_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    bind_group_0: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
}

impl Pipeline {
    // Contract a m x n matrix with the transpose of a k x n matrix to make a m x k matrix
    pub fn new<T: bytemuck::Pod>(anchor: &super::Device,
                                 buffers: (Option<wgpu::Buffer>, // uniform buffer
                                           Option<wgpu::Buffer>, // m x n matrix
                                           Option<wgpu::Buffer>, // k x n matrix
                                           Option<wgpu::Buffer>),// output
                                 m_size: usize,
                                 n_size: usize,
                                 k_size: usize,) -> Self {
        let type_size = std::mem::size_of::<T>();
        let device = &anchor.device;
        //Create/load buffers
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        
        let uniform_buffer = buffers.0.unwrap_or({
            let uniform_data = [m_size as u32, n_size as u32, k_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsage::UNIFORM,
                }
            )
        });
        //0-0

        let matrix_a_buffer = buffers.1.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Matrix A"),
                    size: (type_size * m_size * n_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        );
        //0-1

        let matrix_b_buffer = buffers.2.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Matrix B"),
                    size: (type_size * k_size * n_size) as wgpu::BufferAddress,
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
                    size: (type_size * m_size * k_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
                    mapped_at_creation: false,
                }
            )
        );
        //0-3
        
        //Create bind group(s)
        let bind_group_layout_0 = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Multiply By Transpose bind group layout 0"),
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
                label:  Some("Multiply By Transpose bind group 0"),
                layout: &bind_group_layout_0,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matrix_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },],
            }
        );

        //Create compute pipeline
        let cs_src = include_str!("shader.comp");
        let mut compiler = shaderc::Compiler::new().unwrap();
        let cs_spirv = compiler.compile_into_spirv(cs_src, shaderc::ShaderKind::Compute, "multiplybytranspose.comp", "main", None).unwrap();
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
                label: Some("Multiply By Transpose pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );

         Pipeline {
            uniform_buffer,
            matrix_a_buffer,
            matrix_b_buffer,
            output_buffer,
            bind_group_0,
            compute_pipeline,
        }
    }

    pub fn run(&self, anchor: &super::Device, encoder: &mut wgpu::CommandEncoder, m_size: usize, n_size: usize, k_size: usize,) {
        //Create compute pass
        let mut compute_pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: Some("Multiply By Transpose"),
            }
        );

        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.bind_group_0, &[]);
        //Work groups of X = m_size, Y = k_size, Z = 1
        compute_pass.dispatch(m_size as u32, k_size as u32, 1);
    }
}
