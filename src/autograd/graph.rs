use std::collections::HashMap;

use anyhow::{Result, anyhow};

use crate::device::Device;
use crate::autograd::{Operation, Idx};

/// Graph of the actual computation
pub struct ComputeGraph<'a> {
    pub device: &'a Device,
    graph: HashMap<Idx, Vec<Idx>>,
    order: Vec<Idx>,
    operations: Vec<Box<dyn Operation<'a> + Send + 'a>>,
}

/// TODO
impl<'a> ComputeGraph<'a> {
    ///Create empy compute graph
    pub fn new(device: &'a Device,) -> Self {
        let graph: HashMap<Idx, Vec<Idx>> = HashMap::new();
        let order = Vec::new();
        let operations = Vec::new();

        //Return
        ComputeGraph {
            device,
            graph,
            order,
            operations,
        }
    }
    
    ///Add an operation to the compute graph and return a handle to it
    pub fn add_operation(&mut self, operation: Box<dyn Operation<'a> + Send + 'a>) -> Idx {
        let index = self.operations.len();
        self.operations.push(operation);
        self.graph.entry(index).or_insert_with(Vec::new);
        self.order.push(index);
        //Return
        index
    }

    /// Return reference to an operation by index
    pub fn operation(&self, index: Idx) -> &dyn Operation<'a> {
        self.operations[index].as_ref()
    }

    ///Return mutable reference to operation
    pub fn operation_mut(&mut self, index: Idx) -> &mut dyn Operation<'a> {
         self.operations[index].as_mut()
    }

    #[allow(clippy::type_complexity)]
    /// Get info on all bindings of inputs to outputs in graph
    pub fn binds(&self) -> Vec<(Idx, Vec<Option<(Idx, usize)>>)> {
        let mut bind_info = Vec::new();
        for index in self.order.clone() {
            let operation = &self.operations[index];
            bind_info.push((index, operation.input_binds()));
        }
        bind_info
    }

    /// Attempt to link given socets on identified operations
    pub fn link(&mut self, src_id: Idx, trg_id: Idx, link: (usize, usize),) -> Result<()> {
        let (src_socket ,trg_socket) = link;

        //Check if trg sockets are free
        let input_binds =  &self.operations[trg_id].input_binds();
        if let Some(..) = input_binds.get(trg_socket).unwrap() {
            return Err(anyhow!("Socket {} on operation {} already bound", trg_socket, trg_id))
        }
        //Set src info
        let src = &mut self.operations[src_id];
        src.set_used(src_socket, true)?;

        //Set binding info
        let trg = &mut self.operations[trg_id];
        trg.set_bind(trg_socket, (src_id, src_socket))?;

        //Add link to graph
        let source_vertex = self.graph.entry(src_id).or_insert_with(Vec::new);
        source_vertex.push(trg_id);

        //Rebuild graph order
        self.order = ComputeGraph::get_topological_order(&self.graph);
        
        //Return
        Ok(())
    }

    /// TMP execution function
    pub fn test(&mut self) -> Result<()> {
        for id in self.order.clone() {
            //Get inputs for operation
            let mut input_values = Vec::new();
            for bind in self.operation(id).input_binds() {
                match bind {
                    None => {
                        return Err(anyhow!("Operation {} has unbound inputs", id))
                    },
                    Some((src_id, src_socket)) => {
                        let input = self.operation(src_id).read_forward()?[src_socket];
                        input_values.push(input);
                    },
                }
            }
            //Get outputs for operation
            let operation_output = self.operation(id).forward(input_values.as_slice(), self.device,)?;
            match operation_output {
                None => {
                    //Continue without writing
                },
                Some(output_data) => {
                    //Write data
                    self.operation_mut(id).write_forward(output_data);
                },
            }
            //End
        }
        Ok(())
    }
    
    pub fn print_graph(&self) {
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

    /// Finds all nodes before chosen node in graph and registers unregister nodes in order
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
