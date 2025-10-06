#!/usr/bin/env python3
"""
Run ALL fixed video annotation approaches with comprehensive fixes.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from run_video_annotation_experiments import (
    ExperimentManager, DataLoader,
    SpatialMotorCortexAnnotator, GaitPhaseNeuralAnnotator, ERSPVideoAnnotator,
    EnhancedBrainRegionAnnotator, RealTimeObjectDetectionAnnotator
)

class AnnotationInfoDrawer:
    """Helper class to draw annotation information in upper left corner."""
    
    def __init__(self, annotations):
        self.annotations = annotations
    
    def get_current_annotation(self, current_time):
        """Get current annotation for the given time."""
        for annotation in self.annotations:
            if annotation['time_start'] <= current_time <= annotation['time_end']:
                return annotation
        return None
    
    def draw_annotation_info(self, frame, current_time):
        """Draw annotation information in upper left corner."""
        height, width = frame.shape[:2]
        
        # Create annotation info area
        info_width = 300
        info_height = 120
        info_x = 10
        info_y = 10
        
        # Background with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y), (info_x + info_width, info_y + info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (info_x, info_y), (info_x + info_width, info_y + info_height), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "CURRENT ANNOTATION", (info_x + 8, info_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Get current annotation
        current_annotation = self.get_current_annotation(current_time)
        
        if current_annotation:
            # Category
            cv2.putText(frame, f"Category: {current_annotation['category']}", 
                       (info_x + 8, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            
            # Label (truncate if too long)
            label = current_annotation['label']
            if len(label) > 25:
                label = label[:22] + "..."
            cv2.putText(frame, f"Label: {label}", 
                       (info_x + 8, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Confidence
            cv2.putText(frame, f"Confidence: {current_annotation['confidence']:.2f}", 
                       (info_x + 8, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            
            # Time range
            cv2.putText(frame, f"Time: {current_annotation['time_start']:.1f}s - {current_annotation['time_end']:.1f}s", 
                       (info_x + 8, info_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No annotation active", 
                       (info_x + 8, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(frame, f"Current time: {current_time:.1f}s", 
                       (info_x + 8, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

class FixedMotorCortexAnnotator:
    """Fixed motor cortex approach with better visualization and annotation info."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate, annotations):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        self.annotation_drawer = AnnotationInfoDrawer(annotations)
        
        # Motor cortex regions
        self.motor_regions = {
            'Primary Motor': list(range(40, 60)),
            'Visual Cortex': list(range(80, 100)),
            'Fusiform Gyrus': list(range(100, 120)),
            'Frontal': list(range(20, 40))
        }
    
    def get_motor_activation(self, time_point, region_name):
        """Get activation for a specific motor region."""
        sample_idx = int((time_point - self.video_start_time) * self.sampling_rate)
        window_samples = int(0.5 * self.sampling_rate)
        start_idx = max(0, sample_idx - window_samples // 2)
        end_idx = min(self.ecog_data.shape[1], sample_idx + window_samples // 2)
        
        if start_idx >= end_idx:
            return 0.0
        
        if region_name in self.motor_regions:
            channels = self.motor_regions[region_name]
            valid_channels = [ch for ch in channels if ch < self.ecog_data.shape[0]]
            
            if valid_channels:
                region_data = self.ecog_data[valid_channels, start_idx:end_idx]
                power = np.mean(region_data ** 2)
                scaled_power = power * 1000  # Enhanced scaling
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_motor_visualization(self, frame, activations, current_time):
        """Draw improved motor cortex visualization."""
        height, width = frame.shape[:2]
        
        # Create compact visualization area (top-right)
        viz_width = 300
        viz_height = 200
        viz_x = width - viz_width - 10
        viz_y = 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "BRAIN REGION ACTIVATION", (viz_x + 8, viz_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Draw region bars
        y_offset = 30
        bar_width = 200
        bar_height = 15
        
        regions = ['Primary Motor', 'Visual Cortex', 'Fusiform Gyrus', 'Frontal']
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        for i, (region, color) in enumerate(zip(regions, colors)):
            activation = activations.get(region, 0.0)
            norm_activation = max(0, min(1, abs(activation) / 2000))  # Better scaling
            
            # Calculate position
            region_x = viz_x + 8
            region_y = viz_y + y_offset + i * (bar_height + 8)
            
            # Draw bar
            bar_width_scaled = int(norm_activation * bar_width)
            cv2.rectangle(frame, (region_x, region_y), 
                         (region_x + bar_width_scaled, region_y + bar_height), color, -1)
            cv2.rectangle(frame, (region_x, region_y), 
                         (region_x + bar_width, region_y + bar_height), (255, 255, 255), 1)
            
            # Label and value
            cv2.putText(frame, f"{region}: {activation:.0f}", 
                       (region_x + bar_width + 5, region_y + bar_height - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        
        # Interpretation
        cv2.putText(frame, "Red=Low, Yellow=Medium, Green=High", 
                   (viz_x + 8, viz_y + viz_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {current_time:.1f}s", (viz_x + 8, viz_y + viz_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, video_path, output_path, start_time=0, duration=20, fps=30):
        """Create annotated video with fixed motor cortex overlay."""
        print(f"🎬 Creating fixed motor cortex video: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int(start_time * original_fps)
        end_frame = int((start_time + duration) * original_fps)
        end_frame = min(end_frame, total_frames)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        # Process frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = start_time + frame_count / original_fps
            
            # Draw annotation info (upper left)
            self.annotation_drawer.draw_annotation_info(frame, current_time)
            
            # Get activations
            activations = {}
            for region in self.motor_regions.keys():
                activations[region] = self.get_motor_activation(current_time, region)
            
            # Draw motor visualization (top-right)
            self.draw_motor_visualization(frame, activations, current_time)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Fixed motor cortex video created: {output_path}")
        return output_path

class FixedGaitPhaseAnnotator:
    """Fixed gait phase approach with better scaling and interpretation."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate, annotations):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        self.annotation_drawer = AnnotationInfoDrawer(annotations)
        
        # Gait-related channels
        self.gait_channels = {
            'Heel Strike': list(range(40, 60)),
            'Stance': list(range(60, 80)),
            'Toe Off': list(range(80, 100)),
            'Swing': list(range(100, 120))
        }
    
    def get_gait_activation(self, time_point, phase_name):
        """Get activation for a specific gait phase."""
        sample_idx = int((time_point - self.video_start_time) * self.sampling_rate)
        window_samples = int(0.3 * self.sampling_rate)
        start_idx = max(0, sample_idx - window_samples // 2)
        end_idx = min(self.ecog_data.shape[1], sample_idx + window_samples // 2)
        
        if start_idx >= end_idx:
            return 0.0
        
        if phase_name in self.gait_channels:
            channels = self.gait_channels[phase_name]
            valid_channels = [ch for ch in channels if ch < self.ecog_data.shape[0]]
            
            if valid_channels:
                phase_data = self.ecog_data[valid_channels, start_idx:end_idx]
                power = np.mean(phase_data ** 2)
                scaled_power = power * 1000  # Enhanced scaling
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_gait_timeline(self, frame, activations, current_time):
        """Draw improved gait phase timeline."""
        height, width = frame.shape[:2]
        
        # Create timeline visualization area (bottom-right)
        timeline_width = 300
        timeline_height = 200
        timeline_x = width - timeline_width - 10
        timeline_y = height - timeline_height - 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (timeline_x, timeline_y), (timeline_x + timeline_width, timeline_y + timeline_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (timeline_x, timeline_y), (timeline_x + timeline_width, timeline_y + timeline_height), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "GAIT PHASE NEURAL SIGNATURE", (timeline_x + 8, timeline_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw phase bars with interpretation
        y_offset = 30
        bar_width = 200
        bar_height = 15
        
        phases = ['Heel Strike', 'Stance', 'Toe Off', 'Swing']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, (phase, color) in enumerate(zip(phases, colors)):
            activation = activations.get(phase, 0.0)
            norm_activation = max(0, min(1, abs(activation) / 1000))
            
            # Calculate position
            phase_x = timeline_x + 8
            phase_y = timeline_y + y_offset + i * (bar_height + 8)
            
            # Draw bar
            bar_width_scaled = int(norm_activation * bar_width)
            cv2.rectangle(frame, (phase_x, phase_y), 
                         (phase_x + bar_width_scaled, phase_y + bar_height), color, -1)
            cv2.rectangle(frame, (phase_x, phase_y), 
                         (phase_x + bar_width, phase_y + bar_height), (255, 255, 255), 1)
            
            # Interpret activation level
            if activation > 800:
                level = "HIGH"
                level_color = (0, 255, 0)
            elif activation > 400:
                level = "MED"
                level_color = (0, 255, 255)
            elif activation > 100:
                level = "LOW"
                level_color = (255, 255, 0)
            else:
                level = "MIN"
                level_color = (255, 255, 255)
            
            # Label and interpretation
            cv2.putText(frame, f"{phase}: {level}", 
                       (phase_x + bar_width + 5, phase_y + bar_height - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, level_color, 1)
        
        # Legend
        cv2.putText(frame, "HIGH/MED/LOW/MIN Activity", 
                   (timeline_x + 8, timeline_y + timeline_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {current_time:.1f}s", (timeline_x + 8, timeline_y + timeline_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, video_path, output_path, start_time=0, duration=20, fps=30):
        """Create annotated video with fixed gait phase overlay."""
        print(f"🎬 Creating fixed gait phase video: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int(start_time * original_fps)
        end_frame = int((start_time + duration) * original_fps)
        end_frame = min(end_frame, total_frames)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        # Process frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = start_time + frame_count / original_fps
            
            # Draw annotation info (upper left)
            self.annotation_drawer.draw_annotation_info(frame, current_time)
            
            # Get activations
            activations = {}
            for phase in self.gait_channels.keys():
                activations[phase] = self.get_gait_activation(current_time, phase)
            
            # Draw gait timeline (bottom-right)
            self.draw_gait_timeline(frame, activations, current_time)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Fixed gait phase video created: {output_path}")
        return output_path

def main():
    print("🚀 Running ALL Fixed Video Annotation Approaches")
    print("=" * 50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run all fixed video annotation approaches')
    parser.add_argument('--approaches', type=str, default='1,2,3,4,5,6',
                       help='Comma-separated list of approaches to run (default: 1,2,3,4,5,6)')
    parser.add_argument('--duration', type=int, default=20, 
                       help='Duration in seconds (default: 20)')
    
    args = parser.parse_args()
    
    # Parse approaches to run
    approaches_to_run = [int(x.strip()) for x in args.approaches.split(',')]
    print(f"🎯 Running approaches: {approaches_to_run} for {args.duration} seconds")
    
    # Initialize
    exp_manager = ExperimentManager()
    data_loader = DataLoader()
    
    # Load data
    preprocessed_data = data_loader.load_preprocessed()
    high_gamma_envelope = preprocessed_data.get('filtered_data', None)
    video_start_time = 0.0
    sampling_rate = 600.0
    
    # Load annotation data
    annotation_file = 'results/annotations/video_annotation_data.json'
    if not os.path.exists(annotation_file):
        annotation_file = 'results/06_video_analysis/video_annotation_data.json'
    
    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)
    
    if high_gamma_envelope is None:
        print("❌ Error: Could not load ECoG data")
        sys.exit(1)
    
    # Video path
    video_path = 'data/raw/walk.mp4'
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"📊 ECoG data shape: {high_gamma_envelope.shape}")
    print(f"📁 Experiment folder: experiment{exp_manager.experiment_number}")
    print(f"📝 Annotations: {len(annotation_data['annotations'])} items")
    
    # Run all approaches
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
            
        print(f"\n🎬 Running Approach {approach_num}: {approach_names[approach_num]}")
        
        try:
            if approach_num == 1:
                # Fixed Motor Cortex
                annotator = FixedMotorCortexAnnotator(high_gamma_envelope, video_start_time, sampling_rate, annotation_data['annotations'])
                video_filename = f"walk_annotated_motor_cortex_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 2:
                # Fixed Gait Phase
                annotator = FixedGaitPhaseAnnotator(high_gamma_envelope, video_start_time, sampling_rate, annotation_data['annotations'])
                video_filename = f"walk_annotated_gait_phase_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 3:
                # Use original ERSP (already fixed in main script)
                annotator = ERSPVideoAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_ersp_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 4:
                # Use original Enhanced Brain (already fixed in main script)
                annotator = EnhancedBrainRegionAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_enhanced_brain_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 5:
                # Use original Object Detection (already fixed in main script)
                annotator = RealTimeObjectDetectionAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_object_detection_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 6:
                # Use original Brain Atlas (already fixed in main script)
                from run_video_annotation_experiments import BrainAtlasActivationAnnotator
                annotator = BrainAtlasActivationAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_brain_atlas_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
            
            else:
                print(f"⚠️ Unknown approach: {approach_num}")
                continue
            
            # Create video
            result_path = annotator.create_annotated_video(
                video_path=video_path,
                output_path=output_path,
                start_time=video_start_time,
                duration=args.duration,
                fps=30
            )
            
            if result_path and os.path.exists(result_path):
                print(f"✅ Approach {approach_num} completed: {result_path}")
                results.append((approach_num, approach_names[approach_num], result_path))
            else:
                print(f"❌ Approach {approach_num} failed to create video")
            
        except Exception as e:
            print(f"❌ Error running Approach {approach_num}: {str(e)}")
            continue
    
    # Print summary
    print(f"\n🎉 All approaches completed!")
    print(f"📁 Files saved to: {exp_manager.experiment_dir}")
    print(f"📊 Successfully completed {len(results)} approaches:")
    
    for approach_num, name, result_path in results:
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
            print(f"  ✅ Approach {approach_num}: {name}")
            print(f"     📹 Video: {os.path.basename(result_path)} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()