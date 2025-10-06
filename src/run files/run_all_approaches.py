#!/usr/bin/env python3
"""
Run all video annotation approaches together in a single experiment.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from run_video_annotation_experiments import (
    ExperimentManager, DataLoader,
    SpatialMotorCortexAnnotator, GaitPhaseNeuralAnnotator, ERSPVideoAnnotator,
    EnhancedBrainRegionAnnotator, RealTimeObjectDetectionAnnotator, BrainAtlasActivationAnnotator
)

def main():
    print("🎬 Real-Time Video Annotation Experiments - ALL APPROACHES")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run all video annotation approaches')
    parser.add_argument('--duration', type=int, default=20, 
                       help='Duration in seconds (default: 20)')
    parser.add_argument('--approaches', type=str, default='1,2,3,4,5,6',
                       help='Comma-separated list of approaches to run (default: 1,2,3,4,5,6)')
    
    args = parser.parse_args()
    
    # Parse approaches to run
    approaches_to_run = [int(x.strip()) for x in args.approaches.split(',')]
    print(f"🚀 Running approaches: {approaches_to_run} for {args.duration} seconds")
    
    # Initialize experiment manager
    exp_manager = ExperimentManager()
    data_loader = DataLoader()
    
    # Load data
    features_data = data_loader.load_features()
    preprocessed_data = data_loader.load_preprocessed()
    
    # Load annotation data
    annotation_file = 'results/annotations/video_annotation_data.json'
    if not os.path.exists(annotation_file):
        annotation_file = 'results/06_video_analysis/video_annotation_data.json'
    
    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)
    
    # Extract data
    high_gamma_envelope = preprocessed_data.get('filtered_data', None)
    video_start_time = annotation_data['video_info']['video_start_time']
    sampling_rate = 600.0
    
    if high_gamma_envelope is None:
        print("❌ Error: Could not load ECoG data")
        sys.exit(1)
    
    # Video path
    video_path = 'data/raw/walk.mp4'
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"📊 ECoG data shape: {high_gamma_envelope.shape}")
    print(f"📊 Video start time: {video_start_time}")
    print(f"📁 Experiment folder: experiment{exp_manager.experiment_number}")
    print(f"📝 Annotations: {len(annotation_data['annotations'])} items")
    print()
    
    # Run selected approaches
    results = []
    approach_names = {
        1: "Spatial Motor Cortex Activation Map",
        2: "Gait-Phase Neural Signature Timeline", 
        3: "Event-Related Spectral Perturbation (ERSP) Video Overlay",
        4: "Enhanced Brain Region Activation",
        5: "Real-Time Object Detection Annotation",
        6: "Brain Atlas Activation Overlay"
    }
    
    for approach_num in approaches_to_run:
        if approach_num not in approach_names:
            print(f"⚠️ Skipping unknown approach: {approach_num}")
            continue
            
        print(f"🎬 Running Approach {approach_num}: {approach_names[approach_num]}")
        
        try:
            if approach_num == 1:
                annotator = SpatialMotorCortexAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_motor_cortex_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
                result_path = annotator.create_annotated_video(
                    video_path=video_path,
                    output_path=output_path,
                    start_time=video_start_time,
                    duration=args.duration,
                    fps=30
                )
                
            elif approach_num == 2:
                annotator = GaitPhaseNeuralAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_gait_phase_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
                result_path = annotator.create_annotated_video(
                    video_path=video_path,
                    output_path=output_path,
                    start_time=video_start_time,
                    duration=args.duration,
                    fps=30
                )
                
            elif approach_num == 3:
                annotator = ERSPVideoAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_ersp_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
                result_path = annotator.create_annotated_video(
                    video_path=video_path,
                    output_path=output_path,
                    start_time=video_start_time,
                    duration=args.duration,
                    fps=30
                )
                
            elif approach_num == 4:
                annotator = EnhancedBrainRegionAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_enhanced_brain_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
                result_path = annotator.create_annotated_video(
                    video_path=video_path,
                    output_path=output_path,
                    start_time=video_start_time,
                    duration=args.duration,
                    fps=30
                )
                
            elif approach_num == 5:
                annotator = RealTimeObjectDetectionAnnotator(high_gamma_envelope, video_start_time, sampling_rate, annotation_data['annotations'])
                video_filename = f"walk_annotated_object_detection_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
                result_path = annotator.create_annotated_video(
                    video_path=video_path,
                    output_path=output_path,
                    start_time=video_start_time,
                    duration=args.duration,
                    fps=30
                )
                
            elif approach_num == 6:
                annotator = BrainAtlasActivationAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_brain_atlas_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
                result_path = annotator.create_annotated_video(
                    video_path=video_path,
                    output_path=output_path,
                    start_time=video_start_time,
                    duration=args.duration,
                    fps=30
                )
            
            print(f"✅ Approach {approach_num} completed: {result_path}")
            results.append((approach_num, approach_names[approach_num], result_path))
            
        except Exception as e:
            print(f"❌ Error in Approach {approach_num}: {str(e)}")
            continue
    
    # Save experiment metadata
    experiment_metadata = {
        'experiment_number': exp_manager.experiment_number,
        'timestamp': datetime.now().isoformat(),
        'approaches_run': approaches_to_run,
        'duration': args.duration,
        'data_source': data_loader.latest_experiment,
        'video_path': video_path,
        'results': [
            {
                'approach': result[0],
                'name': result[1],
                'output_path': str(result[2])
            }
            for result in results
        ]
    }
    
    exp_manager.save_experiment_info(experiment_metadata)
    
    # Print summary
    print(f"\n🎉 Experiment {exp_manager.experiment_number} completed!")
    print(f"📁 Files saved to: {exp_manager.experiment_dir}")
    print(f"📊 Successfully completed {len(results)} approaches:")
    
    for approach_num, name, result_path in results:
        file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
        print(f"  ✅ Approach {approach_num}: {name}")
        print(f"     📹 Video: {os.path.basename(result_path)} ({file_size:.1f} MB)")
    
    if len(results) < len(approaches_to_run):
        failed = len(approaches_to_run) - len(results)
        print(f"⚠️ {failed} approach(es) failed to complete")

if __name__ == "__main__":
    main()
