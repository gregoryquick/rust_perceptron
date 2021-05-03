pub struct Pipeline {
    pub uniform_buffer: wgpu::Buffer,
    pub loss_buffer: wgpu::Buffer,
    pub sensitivity_buffer: wgpu::Buffer,
    pub input_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    bind_group_0: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
}

impl Pipeline {
    pub fn new<T: bytemuck::Pod>(anchor: &super::PipelineAnchor,
                                 buffers: (Option<wgpu::Buffer>,
                                           Option<wgpu::Buffer>,
                                           Option<wgpu::Buffer>,
                                           Option<wgpu::Buffer>)
                                 , batch_size: usize,) -> Self {
        let type_size = std::mem::size_of::<T>();
        let input_size = anchor.input_size;
        let output_size = anchor.output_size;
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

        let loss_buffer = buffers.0.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Prediction loss"),
                    size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        );
        //0-1

        let sensitivity_buffer = buffers.1.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Prediction Sensitivity"),
                    size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        );
        //0-2
        
        let input_buffer = buffers.2.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Input Data"),
                    size: (type_size * input_size * batch_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        );
        //0-3
        
        let output_buffer = buffers.3.unwrap_or(
            device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Output buffer"),
                    size: (type_size * input_size * output_size) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
                    mapped_at_creation: false,
                }
            )
        );
        //0-4
        
        //Create bind group(s)
        let bind_group_layout_0 = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Error backprop layout 0"),
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
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
                label:  Some("Error backprop group 0"),
                layout: &bind_group_layout_0,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: loss_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sensitivity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },],
            }
        );

        //Create compute pipeline
        let cs_src = include_str!("shaders/backprop.comp");
        let mut compiler = shaderc::Compiler::new().unwrap();
        let cs_spirv = compiler.compile_into_spirv(cs_src, shaderc::ShaderKind::Compute, "backprop.comp", "main", None).unwrap();
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
                label: Some("Error backprop pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );
        

        //Return
        Pipeline {
            uniform_buffer,
            loss_buffer,
            sensitivity_buffer,
            input_buffer,
            output_buffer,
            bind_group_0,
            compute_pipeline,
        }
    }

    pub fn run(&self, anchor: &super::PipelineAnchor, encoder: &mut wgpu::CommandEncoder, _batch_size: usize) {
        //Create compute pass
        let mut compute_pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor { 
                label: None 
            }
        );

        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.bind_group_0, &[]);
        //Work groups of X = output_size, Y = input_size, Z = 1
        compute_pass.dispatch(anchor.output_size as u32, anchor.input_size as u32, 1);
    }
}
