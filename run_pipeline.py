#!/usr/bin/env python3
"""
IEEE-SMC-2025 ECoG Video Analysis Pipeline
Main execution script for the complete analysis pipeline

This script orchestrates the entire pipeline from raw data to final visualizations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def run_notebook(notebook_path, timeout=3600):
    """Execute a Jupyter notebook."""
    print(f"\n📓 Executing: {notebook_path}")
    
    try:
        # Convert notebook to script and execute
        cmd = [
            "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--inplace",
            "--ExecutePreprocessor.timeout=" + str(timeout),
            str(notebook_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            print(f"✅ Successfully executed: {notebook_path}")
            return True
        else:
            print(f"❌ Error executing {notebook_path}:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout executing {notebook_path}")
        return False
    except Exception as e:
        print(f"❌ Exception executing {notebook_path}: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas', 
        'sklearn', 'mne', 'tqdm', 'jupyter'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required dependencies are installed")
    return True

def main():
    parser = argparse.ArgumentParser(description='IEEE-SMC-2025 ECoG Video Analysis Pipeline')
    parser.add_argument('--task', choices=['1', '2', '3', 'all'], default='all',
                       help='Which task to run (1=EDA, 2=Preprocessing&Modeling, 3=Visualization, all=complete pipeline)')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout for notebook execution (seconds)')
    
    args = parser.parse_args()
    
    print_header("IEEE-SMC-2025 ECoG VIDEO ANALYSIS PIPELINE")
    print(f"🚀 Starting pipeline execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Task: {args.task}")
    
    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            sys.exit(1)
    
    # Define pipeline stages
    pipeline_stages = {
        '1': {
            'name': 'Task 1 - Exploratory Data Analysis (EDA & Signal Understanding)',
            'notebooks': ['notebooks/00_preview.ipynb', 'notebooks/01_visualize_raw.ipynb', 'notebooks/02_preprocess.ipynb', 'notebooks/03_features.ipynb', 'notebooks/04_analysis.ipynb'],
            'description': 'Build intuition about the dataset and identify brain regions, frequencies, and events'
        },
        '2': {
            'name': 'Task 2 - Preprocessing, Feature Extraction & Algorithm Comparison',
            'notebook': 'notebooks/2_Preprocessing_FeatureExtraction_Modeling.ipynb',
            'description': 'Preprocess data, extract features, and compare multiple algorithms'
        },
        '3': {
            'name': 'Task 3 - Live Video Annotation & Brain Activation Visualization',
            'notebook': 'notebooks/3_Live_Annotation_And_Visualization.ipynb',
            'description': 'Create interactive visualization linking ECoG activity to visual stimuli'
        }
    }
    
    # Execute selected tasks
    if args.task == 'all':
        tasks_to_run = ['1', '2', '3']
    else:
        tasks_to_run = [args.task]
    
    success_count = 0
    total_tasks = len(tasks_to_run)
    
    for task_id in tasks_to_run:
        if task_id not in pipeline_stages:
            print(f"❌ Unknown task: {task_id}")
            continue
        
        stage = pipeline_stages[task_id]
        print_header(f"🔧 {stage['name']}")
        print(f"📝 {stage['description']}")
        
        # Handle multiple notebooks for Task 1
        if 'notebooks' in stage:
            notebooks = stage['notebooks']
            task_success = True
            for notebook_path in notebooks:
                notebook_path = Path(notebook_path)
                if not notebook_path.exists():
                    print(f"❌ Notebook not found: {notebook_path}")
                    task_success = False
                    continue
                
                if not run_notebook(notebook_path, args.timeout):
                    task_success = False
            
            if task_success:
                success_count += 1
                print(f"✅ Task {task_id} completed successfully")
            else:
                print(f"❌ Task {task_id} failed")
        else:
            # Handle single notebook for other tasks
            notebook_path = Path(stage['notebook'])
            if not notebook_path.exists():
                print(f"❌ Notebook not found: {notebook_path}")
                continue
            
            if run_notebook(notebook_path, args.timeout):
                success_count += 1
                print(f"✅ Task {task_id} completed successfully")
            else:
                print(f"❌ Task {task_id} failed")
    
    # Summary
    print_header("PIPELINE EXECUTION SUMMARY")
    print(f"📊 Completed: {success_count}/{total_tasks} tasks")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_tasks:
        print("🎉 All tasks completed successfully!")
        print("\n📁 Check the following directories for results:")
        print("  • data/preprocessed/ - Preprocessed ECoG data")
        print("  • data/features/ - Extracted features")
        print("  • data/models/ - Trained models")
        print("  • results/visualizations/ - Generated visualizations")
        print("  • results/reports/ - Analysis reports")
    else:
        print("⚠️  Some tasks failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()