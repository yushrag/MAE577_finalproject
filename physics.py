import numpy as np

class SimulationEngine:
    def __init__(self, config):
        """
        The Engine only cares about the config it is given.
        All parameters are pulled from this single source of truth.
        """
        self.config = config
        
        # Initialize state based on the config
        self.reset()

    def reset(self):
        """Resets agent positions and velocities using config parameters."""
        num_agents = self.config['num_agents']
        box_size = self.config['box_size']

        # Positions: centered in box
        self.pos = (np.random.rand(num_agents, 3) - 0.5) * box_size * 0.8
        self.vel = np.zeros((num_agents, 3))
        self.acc = np.zeros((num_agents, 3))
        self.command_timers = np.zeros(num_agents)
        self.target_vel = np.zeros(3)

    def apply_command(self, target_vel, duration):
        """Sets the target velocity and duration for all agents."""
        self.target_vel = np.array(target_vel)
        self.command_timers[:] = duration

    def step(self, dt, rho=None):
        """Advances the simulation by dt seconds using config parameters."""
        # Pull everything from self.config (Source of Truth)
        num_agents = self.config['num_agents']
        box_size = self.config['box_size']
        radius = self.config['agent_radius']
        mass = self.config['agent_mass']
        cd = self.config['drag_coeff']
        gain = self.config['gain']
        max_thrust = self.config['max_thrust']
        restitution = self.config['restitution']
        gravity = np.array(self.config['microgravity'])
        
        # Override rho if the UI slider moved, otherwise use config
        rho = rho if rho is not None else self.config['air_density']
        
        area = np.pi * (radius ** 2)

        # 1. Reset acceleration with gravity
        self.acc = np.tile(gravity, (num_agents, 1))

        # 2. Control Logic (Velocity Tracking)
        active_mask = self.command_timers > 0
        if np.any(active_mask):
            error = self.target_vel - self.vel[active_mask]
            thrust = error * gain
            
            # Clamp thrust magnitude
            thrust_mags = np.linalg.norm(thrust, axis=1, keepdims=True)
            over_limit = (thrust_mags > max_thrust).flatten()
            if np.any(over_limit):
                thrust[over_limit] *= (max_thrust / thrust_mags[over_limit])
            
            self.acc[active_mask] += thrust / mass
            self.command_timers[active_mask] -= dt

        # 3. Drag Logic
        speeds = np.linalg.norm(self.vel, axis=1, keepdims=True)
        moving_mask = (speeds > 0.001).flatten()
        if np.any(moving_mask):
            drag_mag = 0.5 * rho * (speeds[moving_mask]**2) * cd * area
            drag_force = -(self.vel[moving_mask] / speeds[moving_mask]) * drag_mag
            self.acc[moving_mask] += drag_force / mass

        # 4. Numerical Integration (Euler)
        self.vel += self.acc * dt
        self.pos += self.vel * dt

        # 5. Boundary Collisions
        limit = (box_size / 2) - radius
        for i in range(3): # x, y, z
            # Positive wall
            pos_wall = self.pos[:, i] > limit
            self.pos[pos_wall, i] = limit
            self.vel[pos_wall, i] *= -restitution
            
            # Negative wall
            neg_wall = self.pos[:, i] < -limit
            self.pos[neg_wall, i] = -limit
            self.vel[neg_wall, i] *= -restitution

        return {
            'positions': self.pos.tolist(),
            'velocities': self.vel.tolist(),
            'avg_vel': float(np.mean(np.linalg.norm(self.vel, axis=1)))
        }
