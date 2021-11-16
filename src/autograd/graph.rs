use std::collections::HashMap;

use anyhow::{Result, anyhow};

use crate::device::{Device, tensor::Tensor};
use crate::autograd::{Operation, Idx, acumulate_grads};

/// Graph of the actual computation
pub struct ComputeGraph<'a> {
    pub device: &'a Device,
    pub forward_graph: HashMap<Idx, Vec<(Idx, usize)>>,
    pub reverse_graph: HashMap<Idx, Vec<(Idx, usize)>>,
    pub order: Vec<Idx>,
    pub operations: Vec<Box<dyn Operation<'a> + Send + 'a>>,
}

/// TODO
impl<'a> ComputeGraph<'a> {
    ///Create empy compute graph
    pub fn new(device: &'a Device,) -> Self {
        let forward_graph: HashMap<Idx, Vec<(Idx, usize)>> = HashMap::new();
        let reverse_graph: HashMap<Idx, Vec<(Idx, usize)>> = HashMap::new();
        let order = Vec::new();
        let operations = Vec::new();

        //Return
        ComputeGraph {
            device,
            forward_graph,
            reverse_graph,
            order,
            operations,
        }
    }
    
    ///Add an operation to the compute graph and return a handle to it
    pub fn add_operation(&mut self, operation: Box<dyn Operation<'a> + Send + 'a>) -> Idx {
        let index = self.operations.len();
        self.operations.push(operation);
        self.forward_graph.entry(index).or_insert_with(Vec::new);
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
    pub fn link(&mut self, src_info: (Idx, usize), trg_info: (Idx, usize)) -> Result<()> {
        let (src_id, src_socket) = src_info;
        let (trg_id ,trg_socket) = trg_info;

        //Check if trg sockets are free
        let input_binds =  &self.operations[trg_id].input_binds();
        if let Some(..) = input_binds.get(trg_socket).unwrap() {
            return Err(anyhow!("Socket {} on operation {} already bound", trg_socket, trg_id))
        }
        
        //Set binding info
        let trg = &mut self.operations[trg_id];
        trg.set_bind(trg_socket, (src_id, src_socket))?;

        let src = &mut self.operations[src_id];
        src.set_uses(src_socket, src.output_uses()[src_socket] + 1)?;

        //Add link to graph
        let source_vertex = self.forward_graph.entry(src_id).or_insert_with(Vec::new);
        source_vertex.push((trg_id, trg_socket));

        let target_vertex = self.reverse_graph.entry(trg_id).or_insert_with(Vec::new);
        target_vertex.push((src_id, src_socket));


        //Rebuild graph order
        self.order = ComputeGraph::get_topological_order(&self.forward_graph);
        
        //Return
        Ok(())
    }

    /// Run network in forward mode
    pub fn forward(&mut self) -> Result<()> {
        for id in self.order.clone() {
            //Get data needed to run forwardprop
            let input_values = self.get_input_values(id)?;
            //Get outputs for operation
            let operation_output = self.operation(id).forward(input_values.as_slice(), self.device,)?;
            match operation_output {
                None => {
                    //Continue without writing
                },
                Some(output_data) => {
                    //Write data
                    self.operation_mut(id).write_forward(output_data)?;
                },
            }
            //End
        }
        Ok(())
    }

    ///TMP backward function
    #[allow(clippy::needless_collect)]
    pub fn backward(&mut self, start_grads_with: &[Idx]) -> Result<()> {
        let mut operations = self.order.clone();
        let intitial_operations: Vec<Idx> = operations.drain_filter(|operation| start_grads_with.contains(operation)).collect();
        for id in intitial_operations.into_iter().rev() {
            //Get data needed to run backprop
            let input_values = self.get_input_values(id)?;
            let grad_needs = self.get_input_grad_info(id)?;
            let backprop_grads: Vec<Option<&Tensor>> = self.operation(id).read_forward()?.into_iter().map(Some).collect();
            
            //Run backprop
            let input_grads = self.operation(id).backprop(input_values.as_slice(), grad_needs.as_slice(), backprop_grads.as_slice(), self.device,)?;

            //Save grads
            self.operation_mut(id).write_backprop(input_grads)?;
        }
        for id in operations.into_iter().rev() {
            //Accumulate grads on operation
            let grads = self.get_backpropigating_grads(id)?;
            let grad_needs = self.operation(id).grads_enabled();
            let accumulated_grads = acumulate_grads(grads.as_slice(), grad_needs.as_slice(), self.device,)?;

            //Update grads received by operation
            self.operation_mut(id).write_grads(accumulated_grads)?;

            //Get data needed to run backprop
            let input_values = self.get_input_values(id)?;
            let grad_needs = self.get_input_grad_info(id)?;
            let backprop_grads = self.operation(id).read_grads()?;

            //Run backprop
            let input_grads = self.operation(id).backprop(input_values.as_slice(), grad_needs.as_slice(), backprop_grads.as_slice(), self.device,)?;
            
            //Save grads
            self.operation_mut(id).write_backprop(input_grads)?;
        }
        Ok(())
    }

    ///Get values that are input into a given operation
    fn get_input_values(&self, operation: Idx) -> Result<Vec<&Tensor>> {
        let mut input_values = Vec::new();
        for bind in self.operation(operation).input_binds() {
            match bind {
                None => {
                    return Err(anyhow!("Operation {} has unbound inputs", operation))
                },
                Some((input_id, input_socket)) => {
                    let input = self.operation(input_id).read_forward()?[input_socket];
                    input_values.push(input);
                },
            }
            //End
        }

        //Return
        Ok(input_values)
    }

    ///Get if grads are enabled for inputs of a given operation
    fn get_input_grad_info(&self, operation: Idx) ->  Result<Vec<bool>> {
        let mut grad_needs = Vec::new();
        for bind in self.operation(operation).input_binds() {
            match bind {
                None => {
                    return Err(anyhow!("Operation {} has unbound inputs", operation))
                },
                Some((input_id, input_socket)) => {
                    let needs_grad = self.operation(input_id).grads_enabled()[input_socket];
                    grad_needs.push(needs_grad);
                },
            }
            //End
        }

        //Return
        Ok(grad_needs)
    }

    ///Get all of the grads for the inputs this operation feeds into
    fn get_backpropigating_grads(&self, operation: Idx) -> Result<Vec<(Option<&Tensor<'a>>, usize)>> {
        let mut grads = Vec::new();
        for (upstream_id, upstream_socket) in self.reverse_graph.get(&operation).unwrap().clone() {
            let grad = self.operation(upstream_id).read_backprop()?[upstream_socket];
            let (_ ,downstream_socket) = self.operation(upstream_id).input_binds()[upstream_socket].unwrap();
            grads.push((grad, downstream_socket));
        }

        //Return
        Ok(grads)
    }
    
    ///Returns clones of values of designated output sockets in order
    pub fn get_outputs(&self, outputs_to_read: &[(Idx, usize)]) -> Result<Vec<Tensor>> {
        let to_read = outputs_to_read.into_iter();

        let mut output = Vec::new();

        //Get data
        for &(id, socket) in to_read {
            let data = self.operation(id).read_forward()?[socket].duplicate()?;
            output.push(data);
        }
        
        //Return
        Ok(output)
    }
    
    pub fn print_graph(&self) {
        println!("Order: {:?}", self.order);
        println!("Forward Map: {:?}", self.forward_graph);
        println!("Reverse Map: {:?}", self.reverse_graph);
    }

    /// Returns vec of indexes in topological order
    fn get_topological_order(graph: &HashMap<Idx, Vec<(Idx, usize)>>) -> Vec<Idx> {
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
    fn get_order(graph: &HashMap<Idx, Vec<(Idx, usize)>>, node: &Idx, stack: &mut Vec<Idx>) {
        let receiving_nodes = graph.get(node);
        match receiving_nodes {
            None => {
            },
            Some(receiving_nodes) => {
                for (value, _) in receiving_nodes {
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
