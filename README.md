### 📌 It is the official implementation of TINNs

### 📄 Paper
- [TINNs in arXiv](https://arxiv.org/abs/2601.20361)


#### Time-Induced Neural Networks (TINNs) aims to solve various time-dependent partial differential equations (PDEs). It  parameterizes the network weights as a learned function of time, allowing the effective spatial representation to evolve over time while maintaining shared structure.

#### TINNs on benchmark time-dependent PDEs, showing improved accuracy and stability over other baselines.



![Structure Comparison](figure/structure_comparison.png)


### 🚀 Code

This repository implements TINNs for solving time-dependent PDEs using JAX.
#### Features
- Solves 1D/2D PDEs using TINNs
- Supports Burgers / Allen-Cahn / Klein-Gordon / Korteweg-De Vries / Wave equations
- Implemented in JAX with automatic differentiation
- Optimizer: LM

#### Requirements
- 🧩 List for [Required Packages](requirement.txt)

In each PDEs, there is a .py file to train the model, and the .mat file is the reference (exact) solution if needed.

