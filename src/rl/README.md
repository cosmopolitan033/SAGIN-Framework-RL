# Reinforcement Learning for SAGIN Optimization

This module implements the hierarchical Reinforcement Learning (RL) approach for the Space-Air-Ground Integrated Network (SAGIN) as described in the research paper. The implementation follows the Markov Decision Process (MDP) formulation with hierarchical agents controlling dynamic UAV allocation and task offloading.

## Implementation Details

### MDP Formulation

The implementation follows the exact MDP formulation from the paper:
- **State Space**: Global and local states for central and local agents
- **Action Space**: Dynamic UAV allocation and task offloading decisions
- **Reward Function**: Based on task completion, load balancing, and energy constraints
- **Transition Dynamics**: Handled by the SAGIN network simulation

### Hierarchical RL Structure

The system consists of:
1. **Central Agent**: Controls dynamic UAV allocation across regions
2. **Local Agents**: Make per-task offloading decisions at each static UAV

## Components

### Environment (`environment.py`)
- `SAGINRLEnvironment`: Interfaces with the SAGIN network to provide states, process actions, and calculate rewards

### Agents (`agents.py`)
- `CentralAgent`: Implements the central controller with an actor-critic architecture
- `LocalAgent`: Implements the local agents for task offloading with policy networks

### Neural Network Models (`models.py`)
- `ActorCriticNetwork`: Used by the central agent for dynamic UAV allocation
- `PolicyNetwork`: Used by local agents for task offloading decisions

### Training Coordination (`trainers.py`)
- `HierarchicalRLTrainer`: Coordinates the training of central and local agents

## State and Action Spaces

### Global State (Central Agent)
As defined in the paper:
```
s^global_t = ({λ_r(t)}, {L_{v^stat_r}(t)}, {E_{v^stat_r}(t)}, {A_n(t)}, {x_{v^dyn_n}(t)})
```

Where:
- `λ_r(t)`: Task arrival rate in region r
- `L_{v^stat_r}(t)`: Queue length at static UAV in region r
- `E_{v^stat_r}(t)`: Residual energy of static UAV in region r
- `A_n(t)`: Availability (0 or 1) of dynamic UAV n
- `x_{v^dyn_n}(t)`: Position of dynamic UAV n

### Local State (Local Agents)
As defined in the paper:
```
s^local_{r,t} = (Q_r(t), E_{v^stat_r}(t), N^dyn_r(t), Λ_r(t))
```

Where:
- `Q_r(t)`: Current task queue at static UAV in region r
- `E_{v^stat_r}(t)`: Residual energy of static UAV in region r
- `N^dyn_r(t)`: Number of available dynamic UAVs in region r
- `Λ_r(t)`: Current spatio-temporal task intensity in region r

### Action Spaces

- **Central Agent Actions**: `A^dyn(t) = {A^dyn_n(t) ∈ R | n=1,...,N}`
  - Assigns each dynamic UAV to a target region

- **Local Agent Actions**: `D_j(t) ∈ {local, dynamic, satellite}`
  - Determines whether to process task locally, forward it to a dynamic UAV, or escalate it to a satellite

## Reward Function

The reward function follows the paper's definition:
```
r_t = ∑_j I(T_total,j ≤ τ_j) - α_1 * ΔL_t - α_2 * ∑_v I(E_v(t) < E_min)
```

Where:
- First term rewards tasks completed within deadlines
- Second term penalizes load imbalance
- Third term penalizes UAVs whose energy drops below threshold

## Usage

To train the RL system:
```bash
python examples/rl_sagin_optimization.py --train --episodes 1000
```

To evaluate trained agents:
```bash
python examples/rl_sagin_optimization.py --eval --model-path results/final_model
```

## Dependencies

- PyTorch: For neural network implementation
- NumPy: For numerical operations
- Matplotlib: For visualization of training progress
