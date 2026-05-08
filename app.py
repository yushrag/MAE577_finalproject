from flask import Flask, render_template, jsonify, request
import numpy as np

from simulation import get_default_parameters
from physics import SimulationEngine

app = Flask(__name__)

# Initialize Simulation Engine
config = get_default_parameters()
engine = SimulationEngine(config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(config)

@app.route('/api/step', methods=['POST'])
def step():
    data = request.json or {}
    dt = data.get('dt', 0.016)
    rho = data.get('rho')
    state = engine.step(dt, rho=rho)
    return jsonify(state)

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
    app.run(debug=True, port=5001)
