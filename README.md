# Micrograd

A tiny autograd engine. As you do math on values, it records them in a graph, then walks that graph backwards to compute gradients — that's backpropagation.

On top of it is a small neural network library with a PyTorch-like API.
Both are small: ~100 lines for the engine, ~50 for the nets.

It only works on single numbers (scalars), so a neuron gets split into its individual adds and multiplies. That's still enough to build and train real deep nets

Mainly useful for learning how backprop actually works.

(ref: [karpathy/micrograd](https://github.com/karpathy/micrograd))
