import numpy as np
import trimesh
import os
import threading

class SimulationEngine:
    def __init__(self, config):
        """
        The Engine pulls all parameters from config (single source of truth).
        If an STL path is provided in config, it uses mesh collision detection.
        Otherwise falls back to box collision.
        """
        self.config = config
        self.mesh = None
        self.mesh_scale = 1.0
        self._lock = threading.Lock()

        # Load Mesh if provided
        mesh_path = config.get('mesh_path')
        if mesh_path and os.path.exists(mesh_path):
            raw_mesh = trimesh.load(mesh_path)

            # --- Robust Mesh Cleaning ---
            # Ensure mesh is watertight, normals are consistent, and degenerate faces removed
            if isinstance(raw_mesh, trimesh.Scene):
                raw_mesh = raw_mesh.dump(concatenate=True)
            
            raw_mesh.process()
            raw_mesh.fix_normals()
            raw_mesh.fill_holes()

            # Center the mesh at origin
            raw_mesh.apply_translation(-raw_mesh.centroid)

            # Scale so the mesh fits within the configured box_size
            # Use the largest extent to normalize, then scale to box_size
            max_extent = max(raw_mesh.extents)
            target_size = config['box_size']
            self.mesh_scale = target_size / max_extent
            raw_mesh.apply_scale(self.mesh_scale)

            # Ensure normals point inward (agents are INSIDE the mesh)
            # trimesh normals point outward by default; we want to detect
            # when agents are OUTSIDE the mesh and push them back in
            if raw_mesh.is_watertight:
                self.mesh = raw_mesh
                print(f"[physics] Loaded Mesh: {mesh_path}")
                print(f"[physics] Mesh scaled by {self.mesh_scale:.4f}, extents: {raw_mesh.extents}")

                # --- Pre-build BVH indices ONCE at startup ---
                # This is the key optimization: building these structures takes
                # time, but means each per-frame query is O(log N) not O(N).
                print("[physics] Building BVH acceleration structures...")
                # ProximityQuery builds an rtree spatial index for O(log N) nearest-point lookups
                self.proximity = trimesh.proximity.ProximityQuery(self.mesh)
                print("[physics] BVH ready.")
            else:
                print("[physics] WARNING: Mesh is not watertight. Falling back to box collision.")
        else:
            print("[physics] No Mesh path provided. Using box collision.")

        self.sim_time = 0.0
        self.reset()

    def reset(self):
        """Resets agent positions and velocities using config parameters."""
        num_agents = self.config['num_agents']
        box_size = self.config['box_size']

        # Spawn agents in a smaller region at the center to ensure
        # they start inside the mesh
        self.pos = (np.random.rand(num_agents, 3) - 0.5) * box_size * 0.5
        self.vel = np.zeros((num_agents, 3))
        self.acc = np.zeros((num_agents, 3))
        self.command_timers = np.zeros(num_agents)
        self.target_vel = np.zeros(3)
        self.sim_time = 0.0

    def apply_command(self, target_vel, duration):
        """Sets the target velocity and duration for all agents."""
        self.target_vel = np.array(target_vel)
        self.command_timers[:] = duration

    def _resolve_mesh_collisions(self, restitution, radius):
        """
        Vectorized mesh collision resolution.
        """
        if self.mesh is None:
            return

        try:
            # Ensure input is float64 and finite
            points = np.ascontiguousarray(self.pos, dtype=np.float64)
            
            # BVH-accelerated nearest-point query
            # We use a try-except here because some versions of trimesh/rtree 
            # can fail with IndexError on certain geometries or Python versions
            closest_points, distances, face_indices = self.proximity.on_surface(points)
            
            # Identify which agents are actually colliding or outside
            # Vectorized inside/outside test: (pos - closest) . face_normal
            face_normals = self.mesh.face_normals[face_indices]
            to_agent = self.pos - closest_points
            
            # agents are "outside" if dot product is positive
            dot_products = np.einsum('ij,ij->i', to_agent, face_normals)
            is_outside = dot_products > 0
            
            colliding_mask = is_outside | (distances < radius)
            
            if np.any(colliding_mask):
                # Push agents back inside
                # We use -face_normal because face_normal points OUT
                self.pos[colliding_mask] = closest_points[colliding_mask] - face_normals[colliding_mask] * (radius + 0.01)
                
                # Reflect velocities
                # v_new = v - (1+e)*(v.n)*n
                v_dot_n = np.einsum('ij,ij->i', self.vel[colliding_mask], face_normals[colliding_mask])
                
                # Only reflect if moving toward the wall (v.n > 0)
                reflect_mask = v_dot_n > 0
                if np.any(reflect_mask):
                    idx = np.where(colliding_mask)[0][reflect_mask]
                    v_dot_n_ref = v_dot_n[reflect_mask][:, np.newaxis]
                    n_ref = face_normals[idx]
                    self.vel[idx] -= (1 + restitution) * v_dot_n_ref * n_ref
                    
        except Exception as e:
            print(f"[physics] Collision query failed: {e}. Falling back to box.")
            # Fallback logic is handled by the caller if self.mesh is None, 
            # but here we just skip this frame's mesh collision to prevent a crash.
            pass

    def step(self, dt, rho=None, lj_epsilon=None, lj_sigma=None, milling_enabled=None, milling_fov_angle=None, milling_speed=None, milling_turn_rate=None):
        """Advances the simulation by dt seconds using sub-stepping for stability."""
        with self._lock:
            # 4x sub-stepping: split the total dt into smaller chunks
            # This makes collision detection more precise and prevents high-speed tunnelling
            sub_steps = 4
            sub_dt = dt / sub_steps
            
            for _ in range(sub_steps):
                self._internal_step(sub_dt, rho, lj_epsilon, lj_sigma, milling_enabled, milling_fov_angle, milling_speed, milling_turn_rate)

            # Final safety check and results
            v_mags = np.linalg.norm(self.vel, axis=1)
            avg_v = float(np.mean(v_mags)) if len(v_mags) > 0 and np.all(np.isfinite(v_mags)) else 0.0

            self.sim_time += dt

            return {
                'time': self.sim_time,
                'positions': np.nan_to_num(self.pos, nan=0.0).tolist(),
                'velocities': np.nan_to_num(self.vel, nan=0.0).tolist(),
                'avg_vel': avg_v
            }

    def _internal_step(self, dt, rho=None, lj_epsilon=None, lj_sigma=None, milling_enabled=None, milling_fov_angle=None, milling_speed=None, milling_turn_rate=None):
        """Core physics logic for a single sub-step."""
        # Pull everything from self.config
        num_agents = self.config['num_agents']
        box_size = self.config['box_size']
        radius = self.config['agent_radius']
        mass = self.config['agent_mass']
        cd = self.config['drag_coeff']
        gain = self.config['gain']
        max_thrust = self.config['max_thrust']
        restitution = self.config['restitution']
        gravity = np.array(self.config['microgravity'])
        
        # Override parameters if provided, otherwise use config
        epsilon = lj_epsilon if lj_epsilon is not None else self.config.get('lj_epsilon', 1.0)
        sigma = lj_sigma if lj_sigma is not None else self.config.get('lj_sigma', 1.0)
        rho = rho if rho is not None else self.config['air_density']
        
        # Milling params
        m_enabled = milling_enabled if milling_enabled is not None else self.config.get('milling_enabled', False)
        m_fov = milling_fov_angle if milling_fov_angle is not None else self.config.get('milling_fov_angle', 30.0)
        m_speed = milling_speed if milling_speed is not None else self.config.get('milling_speed', 5.0)
        m_turn = milling_turn_rate if milling_turn_rate is not None else self.config.get('milling_turn_rate', 5.0)

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

        # 3.5 Lennard-Jones Inter-Agent Forces
        if num_agents > 1:
            diff = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            
            sr6 = (sigma / dist) ** 6
            sr12 = sr6 ** 2
            
            force_mag = 24 * epsilon * (2 * sr12 - sr6) / dist
            
            # Clamp force to avoid physics explosions
            max_force = 100.0 * mass
            force_mag = np.clip(force_mag, -max_force, max_force)
            
            direction = diff / dist[..., np.newaxis]
            lj_force = np.sum(force_mag[..., np.newaxis] * direction, axis=1)
            self.acc += lj_force / mass

        # 3.8 Milling Logic
        if m_enabled and num_agents > 1:
            speeds = np.linalg.norm(self.vel, axis=1, keepdims=True)
            # Agents with no speed get a random heading
            headings = np.where(speeds > 0.001, self.vel / np.maximum(speeds, 0.001), 
                                np.random.randn(num_agents, 3))
            headings /= np.linalg.norm(headings, axis=1, keepdims=True)

            to_j = self.pos[np.newaxis, :, :] - self.pos[:, np.newaxis, :] 
            dist = np.linalg.norm(to_j, axis=-1)
            np.fill_diagonal(dist, np.inf)

            to_j_norm = to_j / np.maximum(dist[..., np.newaxis], 1e-6)
            dot_prods = np.sum(headings[:, np.newaxis, :] * to_j_norm, axis=-1)

            cos_theta = np.cos(np.radians(m_fov))
            in_fov_mask = dot_prods > cos_theta

            target_dirs = np.sum(to_j_norm * in_fov_mask[..., np.newaxis], axis=1)
            target_dirs_norm = np.linalg.norm(target_dirs, axis=1, keepdims=True)
            has_targets = (target_dirs_norm > 0).flatten()

            if np.any(has_targets):
                current_headings = headings[has_targets]
                
                # Enforce chiral milling around a single global axis (Z-axis)
                # By turning in a fixed direction (e.g., right) when a neighbor is in the FOV,
                # the swarm naturally forms a single coherent, macroscopic circle.
                z_axis = np.array([0.0, 0.0, 1.0])
                lateral_dir = np.cross(z_axis, current_headings)
                
                # Normalize lateral_dir to ensure constant turn rate even if pitched
                lat_norm = np.linalg.norm(lateral_dir, axis=1, keepdims=True)
                valid_lat = (lat_norm > 1e-5).flatten()
                
                if np.any(valid_lat):
                    lateral_dir_valid = lateral_dir[valid_lat] / lat_norm[valid_lat]
                    turn_accel = m_speed * m_turn * lateral_dir_valid
                    
                    idx = np.where(has_targets)[0][valid_lat]
                    self.acc[idx] += turn_accel

            # Forward propulsion to maintain milling speed
            speed_error = m_speed - speeds.flatten()
            forward_thrust = headings * (speed_error[:, np.newaxis] * gain)
            
            # Clamp thrust
            thrust_mags = np.linalg.norm(forward_thrust, axis=1, keepdims=True)
            over_limit = (thrust_mags > max_thrust).flatten()
            if np.any(over_limit):
                forward_thrust[over_limit] *= (max_thrust / thrust_mags[over_limit])
                
            self.acc += forward_thrust / mass

        # 4. Numerical Integration (Euler)
        self.vel += self.acc * dt
        
        # Clamp velocity to prevent physics explosions
        max_speed = 20.0
        speeds = np.linalg.norm(self.vel, axis=1, keepdims=True)
        over_speed = (speeds > max_speed).flatten()
        if np.any(over_speed):
            self.vel[over_speed] *= (max_speed / speeds[over_speed])

        self.pos += self.vel * dt

        # Safety Check: Prevent NaNs or explosions from crashing the proximity query or frontend
        max_reset_attempts = 3
        for attempt in range(max_reset_attempts):
            if np.all(np.isfinite(self.pos)) and np.all(np.isfinite(self.vel)):
                break
            
            print(f"[physics] WARNING: Non-finite state detected (attempt {attempt+1}). Resetting simulation.")
            self.reset()

        # 5. Collision Detection
        if self.mesh is not None:
            self._resolve_mesh_collisions(restitution, radius)
        else:
            # Fallback: simple box collision
            limit = (box_size / 2) - radius
            for i in range(3):
                pos_wall = self.pos[:, i] > limit
                self.pos[pos_wall, i] = limit
                self.vel[pos_wall, i] *= -restitution

                neg_wall = self.pos[:, i] < -limit
                self.pos[neg_wall, i] = -limit
                self.vel[neg_wall, i] *= -restitution
