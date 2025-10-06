#!/usr/bin/env python3
"""
Run Modeling Pipeline for ECoG Classification
IEEE-SMC-2025 ECoG Video Analysis Competition

This script runs the complete modeling pipeline including all 3 modeling approaches
and all 3 complex visualizations for the 05_modelling stage.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import our modeling pipeline
from modeling.modeling_pipeline import ModelingPipeline
from utils.config import AnalysisConfig

def get_latest_experiment_features():
    """Get the latest experiment's features directory."""
    features_base = Path('data/features')
    
    if not features_base.exists():
        raise ValueError("No features directory found. Please run the preprocessing pipeline first.")
    
    # Find the latest experiment
    experiments = []
    for item in features_base.iterdir():
        if item.is_dir() and item.name.startswith('experiment'):
            try:
                exp_num = int(item.name.replace('experiment', ''))
                experiments.append((exp_num, item))
            except ValueError:
                continue
    
    if not experiments:
        raise ValueError("No experiment directories found in features folder")
    
    # Get the latest experiment
    latest_exp_num, latest_exp_path = max(experiments, key=lambda x: x[0])
    
    print(f"📂 Found latest experiment: experiment{latest_exp_num}")
    return latest_exp_path, f"experiment{latest_exp_num}"

def get_latest_experiment_data():
    """Get the latest experiment's data directories."""
    data_base = Path('data')
    
    # Find the latest experiment in preprocessed data
    preprocessed_base = data_base / 'preprocessed'
    latest_preprocessed = None
    
    if preprocessed_base.exists():
        experiments = []
        for item in preprocessed_base.iterdir():
            if item.is_dir() and item.name.startswith('experiment'):
                try:
                    exp_num = int(item.name.replace('experiment', ''))
                    experiments.append((exp_num, item))
                except ValueError:
                    continue
        
        if experiments:
            latest_exp_num, latest_preprocessed = max(experiments, key=lambda x: x[0])
    
    # Find the latest experiment in raw data
    raw_base = data_base / 'raw'
    latest_raw = None
    
    if raw_base.exists():
        experiments = []
        for item in raw_base.iterdir():
            if item.is_dir() and item.name.startswith('experiment'):
                try:
                    exp_num = int(item.name.replace('experiment', ''))
                    experiments.append((exp_num, item))
                except ValueError:
                    continue
        
        if experiments:
            latest_exp_num, latest_raw = max(experiments, key=lambda x: x[0])
    
    return latest_raw, latest_preprocessed

def main():
    """Main function to run the modeling pipeline."""
    print("🚀 Starting ECoG Modeling Pipeline")
    print("IEEE-SMC-2025 ECoG Video Analysis Competition")
    print("=" * 70)
    
    try:
        # Get latest experiment data
        print("📂 Finding latest experiment data...")
        features_path, experiment_id = get_latest_experiment_features()
        raw_data_path, preprocessed_data_path = get_latest_experiment_data()
        
        print(f"📊 Features path: {features_path}")
        print(f"📊 Raw data path: {raw_data_path}")
        print(f"📊 Preprocessed data path: {preprocessed_data_path}")
        
        # Initialize configuration
        config = AnalysisConfig()
        
        # Initialize modeling pipeline
        modeling_pipeline = ModelingPipeline(config, experiment_id)
        
        # Run complete modeling pipeline
        results = modeling_pipeline.run_complete_modeling_pipeline(
            features_path=features_path,
            raw_data_path=raw_data_path,
            preprocessed_data_path=preprocessed_data_path
        )
        
        # Print summary
        print("\n📊 MODELING PIPELINE SUMMARY")
        print("=" * 70)
        print(modeling_pipeline.get_summary_report())
        
        print("\n🎉 Modeling pipeline completed successfully!")
        print("🎯 Check the results/05_modelling directory for all outputs!")
        
    except Exception as e:
        print(f"\n❌ Modeling pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
