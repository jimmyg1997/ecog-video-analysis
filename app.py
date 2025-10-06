#!/usr/bin/env python3
"""
ECoG Brain-Computer Interface Flask Web Application
IEEE-SMC-2025 ECoG Video Analysis Competition

A comprehensive Flask web application to showcase ECoG research with:
- Interactive data visualizations
- Real-time brain activity display
- Video annotation synchronization
- Preprocessing pipeline visualization
- Results and analysis dashboard
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append('src')

# Import our ECoG modules
from utils.data_loader import DataLoader
from utils.config import AnalysisConfig
from visualization.pipeline_visualizer import PipelineVisualizer
from visualization.brain_atlas import BrainAtlas

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'ecog-bci-research-2025'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global data storage (will be loaded on startup)
data_loader = None
config = None
visualizer = None
brain_atlas = None
raw_data = None
preprocessed_data = None
video_annotations = None

def initialize_data():
    """Initialize data loading and processing modules."""
    global data_loader, config, visualizer, brain_atlas, raw_data, preprocessed_data, video_annotations
    
    print("🔄 Initializing ECoG data modules...")
    
    # Initialize modules with error handling
    try:
        config = AnalysisConfig()
        data_loader = DataLoader()
        visualizer = PipelineVisualizer(config)
        brain_atlas = BrainAtlas(config)
        print("✅ Core modules initialized")
    except Exception as e:
        print(f"⚠️  Could not initialize core modules: {e}")
        # Create mock data for demonstration
        raw_data = {
            'ecog_data': np.random.randn(160, 10000),
            'sampling_rate': 1200,
            'duration': 252,
            'good_channels': 156,
            'bad_channels': 4,
            'stimcode': np.random.randint(1, 7, 252)
        }
        preprocessed_data = {
            'epochs': np.random.randn(252, 156, 840),
            'time_vector': np.linspace(-0.3, 0.4, 840),
            'stimcode': np.random.randint(1, 7, 252),
            'experiment_id': 'demo_experiment'
        }
        video_annotations = {
            'video_info': {
                'duration': 252,
                'fps': 30,
                'total_frames': 7560
            },
            'annotations': [
                {'time_start': 10, 'time_end': 16, 'category': 'face', 'label': 'Face 1'},
                {'time_start': 27, 'time_end': 35, 'category': 'object', 'label': 'Object 1'},
                {'time_start': 48, 'time_end': 53, 'category': 'kanji', 'label': 'Kanji 1'}
            ]
        }
        print("✅ Mock data created for demonstration")
        return
    
    # Load raw data
    print("📊 Loading raw ECoG data...")
    try:
        raw_data = data_loader.load_raw_data()
        print(f"✅ Loaded raw data: {raw_data['ecog_data'].shape}")
    except Exception as e:
        print(f"⚠️  Could not load raw data: {e}")
        raw_data = None
    
    # Load video annotations
    print("📹 Loading video annotations...")
    try:
        annotation_file = Path('results/annotations/video_annotation_data.json')
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                video_annotations = json.load(f)
            print(f"✅ Loaded {len(video_annotations['annotations'])} video annotations")
        else:
            print("⚠️  Video annotations not found")
            video_annotations = None
    except Exception as e:
        print(f"⚠️  Could not load video annotations: {e}")
        video_annotations = None
    
    # Try to load preprocessed data from latest experiment
    print("🔧 Loading preprocessed data...")
    try:
        # Find the latest experiment directory
        results_dir = Path('results/pipeline')
        if results_dir.exists():
            experiments = [d for d in results_dir.iterdir() if d.is_dir()]
            if experiments:
                latest_exp = max(experiments, key=lambda x: x.stat().st_mtime)
                preprocessed_dir = Path(f'data/preprocessed/{latest_exp.name}')
                if preprocessed_dir.exists():
                    # Load preprocessed data files
                    epochs_file = preprocessed_dir / 'epochs.npy'
                    time_vector_file = preprocessed_dir / 'time_vector.npy'
                    stimcode_file = preprocessed_dir / 'stimcode.npy'
                    
                    if all(f.exists() for f in [epochs_file, time_vector_file, stimcode_file]):
                        preprocessed_data = {
                            'epochs': np.load(epochs_file),
                            'time_vector': np.load(time_vector_file),
                            'stimcode': np.load(stimcode_file),
                            'experiment_id': latest_exp.name
                        }
                        print(f"✅ Loaded preprocessed data from {latest_exp.name}")
                    else:
                        print("⚠️  Preprocessed data files incomplete")
                        preprocessed_data = None
                else:
                    print("⚠️  Preprocessed data directory not found")
                    preprocessed_data = None
            else:
                print("⚠️  No experiments found")
                preprocessed_data = None
        else:
            print("⚠️  Results directory not found")
            preprocessed_data = None
    except Exception as e:
        print(f"⚠️  Could not load preprocessed data: {e}")
        preprocessed_data = None
    
    print("🎉 Data initialization completed!")

# Routes
@app.route('/')
def home():
    """Home page with project overview."""
    return render_template('home.html', 
                         raw_data=raw_data,
                         video_annotations=video_annotations,
                         preprocessed_data=preprocessed_data)

@app.route('/data-overview')
def data_overview():
    """Data overview page with interactive visualizations."""
    return render_template('data_overview.html', 
                         raw_data=raw_data,
                         preprocessed_data=preprocessed_data)

@app.route('/preprocessing')
def preprocessing():
    """Preprocessing pipeline page."""
    return render_template('preprocessing.html', 
                         raw_data=raw_data,
                         preprocessed_data=preprocessed_data)

@app.route('/video-annotations')
def video_annotations_page():
    """Video annotations page with embedded player."""
    return render_template('video_annotations.html', 
                         video_annotations=video_annotations)

@app.route('/ecog-visualization')
def ecog_visualization():
    """Real-time ECoG visualization page."""
    return render_template('ecog_visualization.html', 
                         raw_data=raw_data,
                         preprocessed_data=preprocessed_data,
                         video_annotations=video_annotations)

@app.route('/results-analysis')
def results_analysis():
    """Results and analysis page."""
    return render_template('results_analysis.html', 
                         preprocessed_data=preprocessed_data)

@app.route('/methodology')
def methodology():
    """Methodology page."""
    return render_template('methodology.html')

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/feature-extraction')
def feature_extraction():
    """Feature extraction analysis page."""
    return render_template('feature_extraction.html')

@app.route('/modelling')
def modelling():
    """Machine learning modelling page."""
    return render_template('modelling.html')


# Static file routes for results (moved to static file serving section)

# API Routes
@app.route('/api/data/overview')
def api_data_overview():
    """API endpoint for data overview statistics."""
    if raw_data is None:
        return jsonify({'error': 'Raw data not loaded'}), 500
    
    overview = {
        'channels': int(raw_data['ecog_data'].shape[0]),
        'samples': int(raw_data['ecog_data'].shape[1]),
        'sampling_rate': float(raw_data['sampling_rate']),
        'duration': float(raw_data['duration']),
        'duration_minutes': float(raw_data['duration'] / 60),
        'good_channels': int(raw_data.get('good_channels', 156)),
        'bad_channels': int(raw_data.get('bad_channels', 4))
    }
    
    if preprocessed_data is not None:
        overview.update({
            'trials': int(preprocessed_data['epochs'].shape[0]),
            'timepoints': int(preprocessed_data['epochs'].shape[2]),
            'experiment_id': preprocessed_data['experiment_id']
        })
    
    return jsonify(overview)

@app.route('/api/data/channel-stats')
def api_channel_stats():
    """API endpoint for channel statistics."""
    if raw_data is None:
        return jsonify({'error': 'Raw data not loaded'}), 500
    
    ecog_data = raw_data['ecog_data']
    channel_stats = []
    
    for i in range(ecog_data.shape[0]):
        channel_data = ecog_data[i, :]
        stats = {
            'channel_id': i,
            'mean': float(np.mean(channel_data)),
            'std': float(np.std(channel_data)),
            'min': float(np.min(channel_data)),
            'max': float(np.max(channel_data)),
            'variance': float(np.var(channel_data))
        }
        channel_stats.append(stats)
    
    return jsonify(channel_stats)

@app.route('/api/data/signal-quality')
def api_signal_quality():
    """API endpoint for signal quality metrics."""
    if raw_data is None:
        return jsonify({'error': 'Raw data not loaded'}), 500
    
    ecog_data = raw_data['ecog_data']
    
    # Calculate quality metrics
    channel_vars = np.var(ecog_data, axis=1)
    channel_max_amps = np.max(np.abs(ecog_data), axis=1)
    
    quality_metrics = {
        'mean_variance': float(np.mean(channel_vars)),
        'std_variance': float(np.std(channel_vars)),
        'mean_amplitude': float(np.mean(channel_max_amps)),
        'std_amplitude': float(np.std(channel_max_amps)),
        'total_channels': int(ecog_data.shape[0]),
        'total_samples': int(ecog_data.shape[1])
    }
    
    return jsonify(quality_metrics)

@app.route('/api/annotations')
def api_annotations():
    """API endpoint for video annotations."""
    if video_annotations is None:
        return jsonify({'error': 'Video annotations not loaded'}), 500
    
    return jsonify(video_annotations)

@app.route('/api/annotations/categories')
def api_annotation_categories():
    """API endpoint for annotation category statistics."""
    if video_annotations is None:
        return jsonify({'error': 'Video annotations not loaded'}), 500
    
    categories = {}
    for annotation in video_annotations['annotations']:
        category = annotation['category']
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    return jsonify(categories)

@app.route('/api/ecog/signal/<int:channel_id>')
def api_ecog_signal(channel_id):
    """API endpoint for ECoG signal data for a specific channel."""
    if raw_data is None:
        return jsonify({'error': 'Raw data not loaded'}), 500
    
    if channel_id >= raw_data['ecog_data'].shape[0]:
        return jsonify({'error': 'Invalid channel ID'}), 400
    
    # Get signal data (downsample for performance)
    signal_data = raw_data['ecog_data'][channel_id, :]
    sampling_rate = raw_data['sampling_rate']
    
    # Downsample to 100Hz for visualization
    downsample_factor = int(sampling_rate / 100)
    downsampled_signal = signal_data[::downsample_factor]
    downsampled_time = np.arange(len(downsampled_signal)) / 100.0
    
    return jsonify({
        'time': downsampled_time.tolist(),
        'signal': downsampled_signal.tolist(),
        'sampling_rate': 100.0,
        'original_sampling_rate': float(sampling_rate)
    })

@app.route('/api/ecog/epochs/<int:trial_id>')
def api_ecog_epochs(trial_id):
    """API endpoint for ECoG epoch data for a specific trial."""
    if preprocessed_data is None:
        return jsonify({'error': 'Preprocessed data not loaded'}), 500
    
    epochs = preprocessed_data['epochs']
    if trial_id >= epochs.shape[0]:
        return jsonify({'error': 'Invalid trial ID'}), 400
    
    trial_data = epochs[trial_id, :, :]
    time_vector = preprocessed_data['time_vector']
    stimcode = preprocessed_data['stimcode'][trial_id]
    
    return jsonify({
        'trial_id': int(trial_id),
        'stimcode': int(stimcode),
        'time_vector': time_vector.tolist(),
        'channels': trial_data.shape[0],
        'timepoints': trial_data.shape[1],
        'data': trial_data.tolist()
    })

@app.route('/api/ecog/topography/<int:trial_id>/<float:time_point>')
def api_ecog_topography(trial_id, time_point):
    """API endpoint for topographic map data at a specific time point."""
    if preprocessed_data is None:
        return jsonify({'error': 'Preprocessed data not loaded'}), 500
    
    epochs = preprocessed_data['epochs']
    time_vector = preprocessed_data['time_vector']
    
    if trial_id >= epochs.shape[0]:
        return jsonify({'error': 'Invalid trial ID'}), 400
    
    # Find closest time point
    time_idx = np.argmin(np.abs(time_vector - time_point))
    topography_data = epochs[trial_id, :, time_idx]
    
    return jsonify({
        'trial_id': int(trial_id),
        'time_point': float(time_point),
        'time_index': int(time_idx),
        'topography': topography_data.tolist()
    })

# Static file serving
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

@app.route('/data/<path:filename>')
def data_files(filename):
    """Serve data files."""
    return send_from_directory('data', filename)

@app.route('/results/<path:filename>')
def results_files(filename):
    """Serve files from the results directory."""
    return send_from_directory('results', filename)


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

if __name__ == '__main__':
    print("🚀 Starting ECoG Brain-Computer Interface Web Application")
    print("=" * 60)
    print("IEEE-SMC-2025 ECoG Video Analysis Competition")
    print("=" * 60)
    
    # Initialize data
    initialize_data()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)
