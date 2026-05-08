def get_default_parameters():
    """
    Returns the initial configuration for the simulation.
    Synchronized with the Three.js visualization.
    """
    return {
        'num_agents': 30,
        'box_size': 40,
        'agent_radius': 0.5,
        'air_density': 0.05, # Adjusted for better drifting visualization
        'drag_coeff': 0.47,  # Sphere
        'agent_mass': 1.0,
        'max_thrust': 10.0,
        'gain': 2.0,
        'restitution': 1.0,
        'microgravity': [0, 0, -0.005] # Z is up/down
    }
