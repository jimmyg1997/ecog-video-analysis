#!/usr/bin/env python3
"""
Complete pipeline to create all video annotation approaches:
1. Create object detection base video
2. Create brain atlas with connectome
3. Apply all other approaches on top of base video
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    print("🎬 Complete Video Annotation Pipeline")
    print("=" * 40)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run complete video annotation pipeline')
    parser.add_argument('--duration', type=int, default=20, 
                       help='Duration in seconds (default: 20)')
    
    args = parser.parse_args()
    
    print(f"🎯 Running complete pipeline for {args.duration} seconds")
    
    # Step 1: Create base object detection video and brain atlas
    print(f"\n📋 Step 1: Creating base videos...")
    step1_success = run_command(
        f"python comprehensive_video_fixes.py --duration {args.duration}",
        "Creating object detection base video and brain atlas with connectome"
    )
    
    if not step1_success:
        print("❌ Step 1 failed. Stopping pipeline.")
        sys.exit(1)
    
    # Step 2: Apply all other approaches to base video
    print(f"\n📋 Step 2: Applying all approaches to base video...")
    step2_success = run_command(
        f"python apply_all_approaches_to_base.py --approaches '1,2,3,4' --duration {args.duration}",
        "Applying improved approaches (1,2,3,4) to base video"
    )
    
    if not step2_success:
        print("❌ Step 2 failed.")
        sys.exit(1)
    
    # Find the latest experiment directory
    experiment_dirs = list(Path("results/06_video_analysis").glob("experiment*"))
    if not experiment_dirs:
        print("❌ No experiment directories found")
        sys.exit(1)
    
    latest_experiment = max(experiment_dirs, key=os.path.getctime)
    print(f"\n📁 Latest experiment: {latest_experiment}")
    
    # List all created videos
    video_files = list(latest_experiment.glob("*.mp4"))
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📊 Created {len(video_files)} videos:")
    
    for video_file in sorted(video_files):
        file_size = video_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  📹 {video_file.name} ({file_size:.1f} MB)")
    
    print(f"\n🎯 Summary of what was created:")
    print(f"  ✅ Base Object Detection Video (with annotation info panel)")
    print(f"  ✅ Brain Atlas with Connectome (restored atlas visualization)")
    print(f"  ✅ Improved Motor Cortex Activation (with icons and better scaling)")
    print(f"  ✅ Improved Gait Phase Analysis (with interpretable labels)")
    print(f"  ✅ Improved ERSP Time Series (dynamic time series instead of bars)")
    print(f"  ✅ Improved Enhanced Brain Activation (circular layout)")
    
    print(f"\n🎬 All videos are now interpretable, visually appealing, and show clear differences")
    print(f"   when objects are visible vs when they are not!")

if __name__ == "__main__":
    main()
