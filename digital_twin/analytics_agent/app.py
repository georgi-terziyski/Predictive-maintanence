from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
from datetime import datetime
import numpy as np
import statistics

load_dotenv()

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        data = request.json
        
        # Validate input data
        if not data or 'simulations' not in data:
            return jsonify({'error': 'Missing simulations data'}), 400
            
        simulations = data['simulations']
        
        # Calculate statistics for each simulation
        results = []
        for sim in simulations:
            if not all(k in sim for k in ['temperatures', 'vibrations', 'loads']):
                continue
                
            # Temperature analysis
            temps = sim['temperatures']
            temp_stats = {
                'mean': np.mean(temps),
                'median': statistics.median(temps),
                'stdev': statistics.stdev(temps) if len(temps) > 1 else 0,
                'min': min(temps),
                'max': max(temps)
            }
            
            # Vibration analysis
            vibs = sim['vibrations']
            vib_stats = {
                'mean': np.mean(vibs),
                'median': statistics.median(vibs),
                'stdev': statistics.stdev(vibs) if len(vibs) > 1 else 0,
                'min': min(vibs),
                'max': max(vibs)
            }
            
            # Load analysis
            loads = sim['loads']
            load_stats = {
                'mean': np.mean(loads),
                'median': statistics.median(loads),
                'stdev': statistics.stdev(loads) if len(loads) > 1 else 0,
                'min': min(loads),
                'max': max(loads)
            }
            
            results.append({
                'simulation_id': sim.get('simulation_id', 'unknown'),
                'temperature_stats': temp_stats,
                'vibration_stats': vib_stats,
                'load_stats': load_stats
            })
            
        return jsonify({
            'analysis_id': f'analysis_{datetime.now().strftime("%Y%m%d%H%M%S")}',
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_simulations():
    try:
        data = request.json
        
        # Validate input data
        if not data or 'simulations' not in data or len(data['simulations']) < 2:
            return jsonify({'error': 'At least 2 simulations required for comparison'}), 400
            
        simulations = data['simulations']
        
        # Calculate comparison metrics
        comparison = {
            'temperature_diffs': [],
            'vibration_diffs': [],
            'load_diffs': []
        }
        
        # Compare each simulation to the first one
        base_sim = simulations[0]
        for sim in simulations[1:]:
            # Temperature comparison
            temp_diff = np.mean(sim['temperatures']) - np.mean(base_sim['temperatures'])
            
            # Vibration comparison
            vib_diff = np.mean(sim['vibrations']) - np.mean(base_sim['vibrations'])
            
            # Load comparison
            load_diff = np.mean(sim['loads']) - np.mean(base_sim['loads'])
            
            comparison['temperature_diffs'].append(temp_diff)
            comparison['vibration_diffs'].append(vib_diff)
            comparison['load_diffs'].append(load_diff)
            
        return jsonify({
            'comparison_id': f'compare_{datetime.now().strftime("%Y%m%d%H%M%S")}',
            'base_simulation': base_sim.get('simulation_id', 'unknown'),
            'comparison': comparison
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=int(os.getenv('ANALYTICS_AGENT_PORT')))
