from flask import Flask, render_template, jsonify, request
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    # Return initial configuration for the simulation
    return jsonify({
        'num_agents': 10,
        'box_size': 40,
        'air_density': 1.225, # kg/m^3
        'drag_coeff': 0.47, # Sphere
        'agent_mass': 0.1, # kg
        'thrust_power': 5.0,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
