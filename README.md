# Project
Basic implementation of autograd in rust with wgpu based gpu acceleration

## Structure

Path    |                             Usage                             |
--------|---------------------------------------------------------------|
autograd|Containts augrad graph code including hooks to call operations|
device  |Contains code for managing device finding and work distribution|
dispatch|Used to determine kernels to use for an operation based on the tensor information|
kernel  |Contains actual code used to execute operations for specific tensor formats|
tensor  |Contains basic tensor implementation and simple non-graph tensor operations|

## Usage Instructions
Currently non-functional
