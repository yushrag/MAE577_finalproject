def get_default_parameters():
    """
    Returns the initial configuration for the simulation.
    This is the single source of truth for all physics constants.
    """
    return {
        'num_agents': 10,
        'box_size': 40,
        'agent_radius': 0.5,
        'air_density': 0.05,
        'drag_coeff': 0.47,  # Sphere
        'agent_mass': 1.0,
        'max_thrust': 10.0,
        'gain': 2.0,
        'restitution': 0.6,
        'microgravity': [0, 0, -0.005], # Z is up/down
        'mesh_path': 'static/Tesserae_SimpleShell.obj',
        'lj_epsilon': 10.0,
        'lj_sigma': 10.0,  # agent_radius * 2
        'milling_enabled': False,
        'milling_fov_angle': 30.0, # theta half-angle in degrees
        'milling_speed': 5.0,
        'milling_turn_rate': 5.0
    }
