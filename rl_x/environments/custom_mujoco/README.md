# Custom MuJoCo Environments

Contains two examples for custom MuJoCo environments.

The examples can be used as a template for other custom MuJoCo environments. They contain:
- All necessary handling of the MuJoCo physics engine directly via its Python bindings to form a stand-alone environment class
- Implementation of a GLFW viewer for rendering (non-MJX version)

The version with the MJX suffix uses the new MuJoCo XLA backend that enables running the environment on a GPU (similar to Isaac Gym / Sim).
A modern NVIDIA GPU can easily handle 4000 of those environments in parallel.
This gives a significant speedup compared to using normal MuJoCo.  
MJX-based environments break the typical Gym interface and can currently only be used with the ```flax_full_jit``` versions of the ```PPO``` and ```SAC``` algorithms.

More specifically, the example uses the Ant robot and defines as the task to track a given velocity command.

| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Ant | Flat value | Continuous | Numpy |
| Ant MJX | Flat value | Continuous | JAX |