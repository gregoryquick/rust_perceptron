use std::collections::HashMap;
//use std::ops::Index;

use anyhow::Result;

use crate::device::{Device, tensor::Tensor};

mod tensor;

/// The ID for a node in the graph
pub type Idx = usize;

/// Trait for operations in the compute graph
pub trait Operation {
    /// Retutns info on the shape restriction for inputs
    fn input_shape(&self) -> Vec<Option<usize>>;

    /// Returns bools to indicate which outputs are bound
    fn input_bound(&self)-> Vec<bool>;

    /// Retutns info on the shape restriction for output
    fn output_shape(&self) -> Vec<Option<usize>>;

    /// Returns bools to indicate which outputs are used
    fn output_used(&self) -> Vec<bool>;

    /// Returns reference to internaly stored output tensor
    /// Takes slice of references to input's output tensors
    fn forward(&mut self,
               input_data: &[&Tensor],
               encoder: &mut wgpu::CommandEncoder,
               device: &Device) -> Result<&Tensor>;
    
    /// Return grads with respect to all inputs flagged with `needs_grad`
    /// Requres tensor contaning the backpropogating grad to operation
    fn backprop(&self,
                //backprop_grad: Tensor,
                input_data: &[&Tensor],
                needs_grad: &[bool],
                encoder: &mut wgpu::CommandEncoder,
                device: &Device
                ) -> Result<Vec<Option<Tensor>>>;

    /// Return reference to internal tensor used for tracking
    /// the backpropigating grad the operation recieves
    fn grad(&self) -> Option<&Tensor>;
}


/// Graph of the actual computation
pub struct ComputeGraph<'a> {
    device: &'a Device,
    graph: HashMap<Idx, Vec<Idx>>,
    order: Vec<Idx>,
    operations: Vec<Box<dyn Operation + Send>>,
}

/// TODO
impl<'a> ComputeGraph<'a> {
    ///Add an operation to the compute graph and return a handle to it
    pub fn add_operation(&mut self, operation: Box<dyn Operation + Send>) -> Idx {
        let index = self.operations.len();
        self.operations.push(operation);
        self.graph.entry(index).or_insert_with(Vec::new);
        self.order.push(index);
        //Return
        index
    }
    
    pub fn dummy_new(device: &'a Device, graph_info: Vec<(Idx, Idx)>) -> Self {
        let mut graph: HashMap<Idx, Vec<Idx>> = HashMap::new();
        for value in graph_info {
            let source_vertex = graph.entry(value.0).or_insert_with(Vec::new);
            source_vertex.push(value.1);
        }
        
        let order = ComputeGraph::get_topological_order(&graph);

        //Return
        ComputeGraph {
            device,
            graph,
            order,
            operations: Vec::new(),
        }
    }
    
    pub fn dummy_test(&self) {
        println!("Order: {:?}", self.order);
        println!("Map: {:?}", self.graph);
    }

    /// Returns vec of indexes in topological order
    fn get_topological_order(graph: &HashMap<Idx, Vec<Idx>>) -> Vec<Idx> {
        let source_nodes = graph.keys();
        let mut stack: Vec<Idx> = vec![];
        for node in source_nodes {
            ComputeGraph::get_order(graph, node, &mut stack);
        }
        stack.reverse();

        //Return
        stack
    }

    fn get_order(graph: &HashMap<Idx, Vec<Idx>>, node: &Idx, stack: &mut Vec<Idx>) {
        let receiving_nodes = graph.get(node);
        match receiving_nodes {
            None => {
            },
            Some(receiving_nodes) => {
                for value in receiving_nodes {
                    ComputeGraph::get_order(graph, value, stack);
                }

            },
        }
        
        if !stack.contains(node) {
            stack.push(*node);
        }

        //Return
    }
}
