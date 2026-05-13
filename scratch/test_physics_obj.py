import sys
import os
sys.path.append('.')
from simulation import get_default_parameters
from physics import SimulationEngine
import numpy as np

config = get_default_parameters()
engine = SimulationEngine(config)
print("Engine initialized.")

# Simulate a step
try:
    state = engine.step(0.016)
    print("Step 1 successful.")
    # Simulate many steps
    for i in range(10):
        engine.step(0.016)
    print("10 steps successful.")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
