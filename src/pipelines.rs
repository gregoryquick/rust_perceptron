pub trait Pipeline {
    fn new<T: bytemuck::Pod>(device: &wgpu::Device, input_size: usize, output_size: usize, number_of_inputs: usize,) -> Self;

    fn get_buffers(&self) -> &Vec<wgpu::Buffer>;

    fn get_batch_size(&self) -> usize;

    fn get_bind_group(&self) -> &wgpu::BindGroup;

    fn get_compute_pipeline(&self) -> &wgpu::ComputePipeline;
}

pub struct ForwardPass {
    buffers: Vec<wgpu::Buffer>,
    batch_size: usize,
    bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
}

impl Pipeline for ForwardPass {
    fn new<T: bytemuck::Pod>(device: &wgpu::Device, input_size: usize, output_size: usize, batch_size: usize,) -> Self{
        //Create buffers
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let mut buffers: Vec<wgpu::Buffer> = Vec::new();
        let type_size = std::mem::size_of::<T>();

        let uniform_data = [input_size as u32, output_size as u32, batch_size as u32];
        let uniform_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniform_data),
                usage: wgpu::BufferUsage::UNIFORM,
            }
        );
        buffers.push(uniform_buffer);

        let weight_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Network Weights"),
                size: (type_size * input_size * output_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        buffers.push(weight_buffer);

        let input_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Input Buffer"),
                size: (type_size * input_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        buffers.push(input_buffer);
        
        let output_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Output buffer"),
                size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        buffers.push(output_buffer);
        
        //Create buffer bind group for pipeline
        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
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
        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("Forward Pass bind group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers[3].as_entire_binding(),
                },],
            }
        );

        //Create compute pipeline
        let cs_src = include_str!("shaders/forward.comp");
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
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }
        );
        
        let compute_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Forward pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );
        ForwardPass{
            buffers,
            batch_size,
            bind_group,
            compute_pipeline,
        }
    }

    fn get_buffers(&self) -> &Vec<wgpu::Buffer>{
        &self.buffers
    }

    fn get_batch_size(&self) -> usize {
        self.batch_size
    }


    fn get_bind_group(&self) -> &wgpu::BindGroup{
        &self.bind_group
    }

    fn get_compute_pipeline(&self) -> &wgpu::ComputePipeline{
        &self.compute_pipeline
    }
}

pub struct BackwardPass {
    buffers: Vec<wgpu::Buffer>,
    batch_size: usize,
    bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
}

impl Pipeline for BackwardPass {
    fn new<T: bytemuck::Pod>(device: &wgpu::Device, input_size: usize, output_size: usize, batch_size: usize,) -> Self{
        //Create buffers
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let mut buffers: Vec<wgpu::Buffer> = Vec::new();
        let type_size = std::mem::size_of::<T>();

        let uniform_data = [input_size as u32, output_size as u32, batch_size as u32];
        let uniform_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniform_data),
                usage: wgpu::BufferUsage::UNIFORM,
            }
        );
        buffers.push(uniform_buffer); //0
        
        let weight_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Network Weights"),
                size: (type_size * input_size * output_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        buffers.push(weight_buffer); //1

        let input_data = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Input Data"),
                size: (type_size * input_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        buffers.push(input_data); //2

        let label_data = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Label Data"),
                size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        buffers.push(label_data); //3

        let prediction_data = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Prediction Data"),
                size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        buffers.push(prediction_data); //4

        let output_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Output buffer"),
                size: (type_size * input_size * output_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        buffers.push(output_buffer); //5


        //Create buffer bind group for pipeline
        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
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
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("Backward Pass bind group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers[3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers[4].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers[5].as_entire_binding(),
                },],
            }
        );

        //Create compute pipeline
        let cs_src = include_str!("shaders/backward.comp");
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
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }
        );
        
        let compute_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Backwards pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );

        BackwardPass{
            buffers,
            batch_size,
            bind_group,
            compute_pipeline,
        }
    }
    
    fn get_buffers(&self) -> &Vec<wgpu::Buffer>{
        &self.buffers
    }

    fn get_batch_size(&self) -> usize {
        self.batch_size
    }


    fn get_bind_group(&self) -> &wgpu::BindGroup{
        &self.bind_group
    }

    fn get_compute_pipeline(&self) -> &wgpu::ComputePipeline{
        &self.compute_pipeline
    }
}
