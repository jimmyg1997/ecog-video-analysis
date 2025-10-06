#!/usr/bin/env python3
"""
Experiment Runner for ECoG Pipeline
IEEE-SMC-2025 ECoG Video Analysis Competition

This script allows you to run experiments with custom experiment IDs.
"""

import sys
import os
sys.path.append('src')

from datetime import datetime
from run_comprehensive_pipeline_v2 import run_experiment
from utils.config import AnalysisConfig

def main():
    """Main function to run experiments."""
    print("🧪 ECoG Experiment Runner")
    print("=" * 50)
    
    # Get experiment ID from command line or generate one
    if len(sys.argv) > 1:
        experiment_id = sys.argv[1]
        print(f"🧪 Running experiment: {experiment_id}")
    else:
        experiment_id = None  # Let pipeline auto-generate (experiment1, experiment2, etc.)
        print(f"🧪 Auto-generating experiment ID (experiment1, experiment2, etc.)")
    
    # Create custom config if needed
    config = AnalysisConfig()
    
    # Run the experiment
    try:
        pipeline = run_experiment(experiment_id, config)
        print(f"\n✅ Experiment '{experiment_id}' completed successfully!")
        print(f"📁 Results saved in experiment-specific directories")
        return pipeline
    except Exception as e:
        print(f"\n❌ Experiment '{experiment_id}' failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
