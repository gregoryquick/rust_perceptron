[[block]] struct TensorMetaData {
	tensor_a_shape_0: u32;
	tensor_a_shape_1: u32;
	tensor_a_stride_0: u32;
	tensor_a_stride_1: u32;
	tensor_b_shape_0: u32;
	tensor_b_shape_1: u32;
	tensor_b_stride_0: u32;
	tensor_b_stride_1: u32;
	target_shape_0: u32;
	target_shape_1: u32;
	target_stride_0: u32;
	target_stride_1: u32;
};

[[block]] struct Data {
   	data: [[stride(4)]] array<f32>;
};

[[group(0), binding(0)]] var<storage, read> meta_data : TensorMetaData;
[[group(0), binding(1)]] var<storage, read> tensor_a : Data;
[[group(0), binding(2)]] var<storage, read> tensor_b : Data;
[[group(0), binding(3)]] var<storage, read_write> target : Data;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main ([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let index_a : u32 = global_id.x * meta_data.tensor_a_stride_0 + global_id.y * meta_data.tensor_a_stride_1;
	let index_b : u32 = global_id.x * meta_data.tensor_b_stride_0 + global_id.y * meta_data.tensor_b_stride_1;
	let result = tensor_a.data[index_a] + tensor_b.data[index_b];

	let target_index : u32 = global_id.x * meta_data.target_stride_0 + global_id.y * meta_data.target_stride_1;
	target.data[target_index] = result;
}
