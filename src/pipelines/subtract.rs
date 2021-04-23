pub struct Pipeline {
    pub uniform_buffer: wgpu::Buffer,
    pub input_buffer_a: wgpu::Buffer,
    pub input_buffer_b: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub bind_group_0: wgpu::BindGroup,
    pub compute_pipeline: wgpu::ComputePipeline,
}
impl Pipeline {
    pub fn new<T: bytemuck::Pod>(anchor: &super::PipelineAnchor,
                                 buffers: (Option<wgpu::Buffer>,
                                           Option<wgpu::Buffer>,
                                           Option<wgpu::Buffer>)
                                 , batch_size: usize,) -> Self {
        let type_size = std::mem::size_of::<T>();
        let output_size = anchor.output_size;
        let device = &anchor.device;
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        //Create buffers
        let uniform_data = [output_size as u32, batch_size as u32];
        let uniform_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniform_data),
                usage: wgpu::BufferUsage::UNIFORM,
            }
        );
        //0-0

        let input_buffer_a = buffers.0.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Input Buffer A"),
                    size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        );
        //0-1
        
        let input_buffer_b = buffers.1.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Input Buffer B"),
                    size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        );
        //0-2

        let output_buffer = buffers.2.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Output buffer"),
                    size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
                    mapped_at_creation: false,
                }
            )
        );
        //0-3
        
        //Create bind group(s)
        let bind_group_layout_0 = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Leaky relu bind group layout 0"),
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
                label:  Some("Leaky relu bind group 0"),
                layout: &bind_group_layout_0,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer_a.as_entire_binding(),
                },
                                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buffer_b.as_entire_binding(),
                },

                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },],
            }
        );

        //Create compute pipeline
        let cs_src = include_str!("shaders/subtract.comp");
        let mut compiler = shaderc::Compiler::new().unwrap();
        let cs_spirv = compiler.compile_into_spirv(cs_src, shaderc::ShaderKind::Compute, "subtract.comp", "main", None).unwrap();
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
                label: Some("Leaky relu pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );
        

        //Return
        Pipeline {
            uniform_buffer,
            input_buffer_a,
            input_buffer_b,
            output_buffer,
            bind_group_0,
            compute_pipeline,
        }
    }
    
    pub fn run(&self, anchor: &super::PipelineAnchor, encoder: &mut wgpu::CommandEncoder, batch_size: usize) {
        //Create compute pass
        let mut compute_pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor { 
                label: None 
            }
        );

        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.bind_group_0, &[]);
        //Work groups of X = batch_size, Y = output_size, Z = 1
        compute_pass.dispatch(batch_size as u32, anchor.output_size as u32, 1);
    }
}
