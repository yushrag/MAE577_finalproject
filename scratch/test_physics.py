import numpy as np
import trimesh
from physics import SimulationEngine
from simulation import get_default_parameters

config = get_default_parameters()
engine = SimulationEngine(config)

print("Starting test step...")
state = engine.step(0.01)
print("Step completed.")
print(f"Number of agents: {len(state['positions'])}")
print(f"First agent pos: {state['positions'][0]}")
