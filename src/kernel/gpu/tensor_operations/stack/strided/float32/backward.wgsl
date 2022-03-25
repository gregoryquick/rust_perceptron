[[block]] struct OperationMetaData {
	[[align(8), size(16)]] output_strides: vec3<u32>;
	[[align(8), size(16)]] tensor_strides: vec3<u32>;
};

[[block]] struct Offset {
	output_offset: u32;
   	tensor_offset: u32;
};

[[block]] struct Data {
   	data: [[stride(4)]] array<f32>;
};

[[group(0), binding(0)]] var<storage, read> meta_data: OperationMetaData;
[[group(0), binding(1)]] var<storage, read> start_position: Offset;
[[group(0), binding(2)]] var<storage, read> tensor: Data;
[[group(0), binding(3)]] var<storage, read_write> target: Data;

fn toIndex(indices: vec3<u32>) -> u32 {
	return indices.x + indices.y + indices.z;
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main ([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let target_index: u32 = start_position.output_offset + toIndex(global_id * meta_data.output_strides);
	let tensor_index: u32 = start_position.tensor_offset + toIndex(global_id * meta_data.tensor_strides);
	target.data[target_index] = tensor.data[tensor_index];
}