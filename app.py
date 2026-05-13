from flask import Flask, render_template, jsonify, request
import numpy as np
import logging
import csv
import time

# Temporarily re-enable logs for debugging
log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)

from simulation import get_default_parameters
from physics import SimulationEngine

app = Flask(__name__)

# Initialize Simulation Engine
config = get_default_parameters()
engine = SimulationEngine(config)

recording_active = False
csv_file = None
csv_writer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(config)

@app.route('/api/step', methods=['POST'])
def step():
    try:
        data = request.json or {}
        dt = data.get('dt', 0.016)
        rho = data.get('rho')
        lj_epsilon = data.get('lj_epsilon')
        lj_sigma = data.get('lj_sigma')
        milling_enabled = data.get('milling_enabled')
        milling_fov_angle = data.get('milling_fov_angle')
        milling_speed = data.get('milling_speed')
        milling_turn_rate = data.get('milling_turn_rate')
        
        state = engine.step(
            dt, 
            rho=rho, 
            lj_epsilon=lj_epsilon, 
            lj_sigma=lj_sigma,
            milling_enabled=milling_enabled,
            milling_fov_angle=milling_fov_angle,
            milling_speed=milling_speed,
            milling_turn_rate=milling_turn_rate
        )
        
        global recording_active, csv_writer, csv_file
        if recording_active and csv_writer:
            row = [state.get('time', 0.0)]
            for i in range(len(state['positions'])):
                row.extend(state['positions'][i])
                row.extend(state['velocities'][i])
            csv_writer.writerow(row)
            # Flush frequently so data isn't lost if stopped abruptly
            csv_file.flush() 
            
        return jsonify(state)
    except Exception as e:
        import traceback
        print(f"Error in /api/step: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/toggle_recording', methods=['POST'])
def toggle_recording():
    global recording_active, csv_file, csv_writer
    data = request.json or {}
    record = data.get('record', False)
    
    if record and not recording_active:
        filename = f"simulation_data_{int(time.time())}.csv"
        csv_file = open(filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        # Write header
        header = ['Time']
        num_agents = engine.config['num_agents']
        for i in range(num_agents):
            header.extend([
                f'Agent{i}_X', f'Agent{i}_Y', f'Agent{i}_Z',
                f'Agent{i}_VX', f'Agent{i}_VY', f'Agent{i}_VZ'
            ])
        csv_writer.writerow(header)
        
        recording_active = True
        print(f"Started recording to {filename}")
        return jsonify({'status': 'recording_started', 'filename': filename})
        
    elif not record and recording_active:
        recording_active = False
        if csv_file:
            csv_file.close()
            csv_file = None
            csv_writer = None
        print("Stopped recording")
        return jsonify({'status': 'recording_stopped'})
        
    return jsonify({'status': 'no_change'})

@app.route('/api/command', methods=['POST'])
def command():
    data = request.json or {}
    target_vel = data.get('target_vel', [0, 0, 0])
    duration = data.get('duration', 2.0)
    engine.apply_command(target_vel, duration)
    return jsonify({'status': 'command_applied'})

@app.route('/api/reset', methods=['POST'])
def reset():
    engine.reset()
    return jsonify({'status': 'reset_complete'})

if __name__ == '__main__':
    app.run(debug=False, port=5001)
