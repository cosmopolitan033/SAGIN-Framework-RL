# SAGIN:  Content Caching and Task Offloading Framework in Space-Air-Ground Integrated Networks

This project implements a **Deep Reinforcement Learning-based framework** for **joint content caching and task offloading** in **SAGIN (Space-Air-Ground Integrated Network)** environments. It simulates UAVs (Unmanned Aerial Vehicles), satellites, and ground-based IoT devices collaborating for efficient data processing, content caching, and task execution under resource constraints.

---

## 🚀 Overview of SAGIN Architecture

```
                 +----------------------------+
                 |        Satellite Tier      |
                 |   (Global Coverage, High   |
                 |     Storage & Computing)   |
                 +----------------------------+
                            ↑   ↑   ↑
                            ↓   ↓   ↓
       +---------------------------------------------+
       |              Aerial Tier (UAVs)             |
       |    (Mobile Nodes, Cache/Compute Enabled)    |
       +---------------------------------------------+
            ↑     ↑       ↑       ↑        ↑
            ↓     ↓       ↓       ↓        ↓
+------------------------------------------------------------+
| Ground Tier (IoT Regions + Ground Vehicles + Sensors)      |
| (Local Data Generation, Mobility, V2V/V2U Communication)   |
+------------------------------------------------------------+
```

- **Ground Vehicles** move according to mobility models, communicate with each other (V2V) and UAVs (V2U), and generate/process tasks.
- **IoT Devices** generate content periodically.
- **UAVs** aggregate, cache, and process tasks or offload them to neighbors, ground vehicles, or satellites.
- **Satellites** act as powerful compute/cache nodes with global coverage.

---

## ⚙️ Code Structure and Modules

| File                          | Purpose |
|------------------------------|---------|
| `main.py`                    | Runs PPO-driven simulation (learning agent) |
| `compare_policies.py`        | Baseline simulation (popularity-based caching, greedy offloading) |
| `sagin_env.py`               | Core environment managing all three tiers (vehicles, UAVs, satellites) |
| `ground_vehicle.py`          | Ground Vehicle class: mobility, V2V/V2U communication, task execution |
| `uav.py`                     | UAV class: caching, task execution, energy handling |
| `satellite.py`               | Satellite class: task queueing, execution, TTL-based eviction |
| `iot_region.py`              | Models IoT regions, activation, content generation |
| `ppo_gru_agent.py`           | PPO + GRU agent for intelligent caching decisions |
| `content_popularity_predictor.py` | GRU-based popularity predictor (for future use) |
| `communication_model.py`     | Channel modeling between all network entities |
| `visualize_network.py`       | Visualization of the three-tier architecture and metrics |

---

## 🔄 Simulation Flow

1. **Ground Vehicle Mobility**  
   Ground vehicles move according to mobility models (random waypoint or route following).

2. **V2V and V2U Link Updates**  
   Vehicle-to-vehicle and vehicle-to-UAV communication links are updated based on positions.

3. **Content Generation**  
   - IoT devices generate content based on Zipf-distributed popularity
   - Ground vehicles generate content with different types (sensor, image, video, map)

4. **Content Aggregation**  
   UAVs receive content from IoT devices and ground vehicles in their coverage area.

5. **OFDM Slot Allocation**  
   UAVs get randomly assigned slots to communicate with satellites.

6. **Satellite Caching**  
   Satellites receive and store content with TTL constraints.

7. **Task Generation**  
   - UAVs generate computational tasks that need associated content
   - Ground vehicles generate tasks with different priorities and deadlines

8. **Offloading Decisions**  
   - For UAV tasks: Try local, then neighbor UAVs, then ground vehicles, then satellite
   - For vehicle tasks: Try local, then V2V neighbors, then connected UAV, then satellite
   - If not found, task is dropped

9. **Task Execution**  
   Ground vehicles, UAVs, and satellites execute queued tasks if delays are within bounds.

10. **Caching Decision via PPO**  
    A GRU-based PPO agent decides which content to cache in UAVs and vehicles based on observed metrics.

11. **Eviction & Energy Update**  
    Expired content is evicted. Energy is updated for all entities—simulation halts if any UAV's energy is depleted.

12. **Visualization**  
    Network structure and metrics are visualized periodically to track system performance.

---

## ▶️ How to Run

### A. Baseline Policy (no RL):
```bash
python compare_policies.py
```
This runs the simulation using:
- Popularity-based caching
- Greedy offloading
- 5 episodes of 5 time slots each

### B. PPO Policy (learning-based):
```bash
python main.py
```
This:
- Loads a PPO+GRU agent
- Executes 500 time slots
- Learns caching decisions per UAV

---

## 📦 Dependencies

- Python 3.8+
- NumPy
- PyTorch
- Matplotlib

Install with:
```bash
pip install numpy torch matplotlib
```

---

## 📊 Output

- Task logs and success summaries printed after each run
- Cache hit rates and delay-bound statistics per UAV
- Optional plots in `compare_policies.py`

---

## ⚠️ Notes

- The simulation stops **immediately** if **any UAV's energy becomes zero**, simulating real-world failure.
- Satellites have **universal coverage**; they do not move.
- UAVs cache content within limited memory and update their cache per time slot.
- Content and tasks have TTLs and delay bounds.

---

## 🧠 Future Extensions

- Multi-agent PPO
- Federated caching across UAVs
- Energy-aware path planning for UAV movement
- Satellite handovers and mobility

---

## 👨‍💻 Authors

This codebase was developed as part of research in **Intelligent Communication Networks** and **AI-Driven SAGIN Optimization**. For questions or collaboration, reach out to the project maintainer.

---