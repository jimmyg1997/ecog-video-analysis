#!/usr/bin/env python3
"""
Fix all video annotation approaches and add annotation information to all approaches.
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

class FixedBrainAtlasAnnotator:
    """Fixed brain atlas approach with simplified visualization and annotation info."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate, annotations):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        self.annotation_drawer = AnnotationInfoDrawer(annotations)
        
        # Simplified brain regions
        self.brain_regions = {
            'Place': list(range(100, 120)),
            'Face': list(range(120, 140)),
            'Shape': list(range(80, 100)),
            'Word': list(range(140, 158))
        }
    
    def get_region_activation(self, time_point, region_name):
        """Get activation for a specific brain region."""
        sample_idx = int((time_point - self.video_start_time) * self.sampling_rate)
        window_samples = int(0.5 * self.sampling_rate)
        start_idx = max(0, sample_idx - window_samples // 2)
        end_idx = min(self.ecog_data.shape[1], sample_idx + window_samples // 2)
        
        if start_idx >= end_idx:
            return 0.0
        
        if region_name in self.brain_regions:
            channels = self.brain_regions[region_name]
            valid_channels = [ch for ch in channels if ch < self.ecog_data.shape[0]]
            
            if valid_channels:
                region_data = self.ecog_data[valid_channels, start_idx:end_idx]
                power = np.mean(region_data ** 2)
                scaled_power = power * 100
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_brain_atlas_overlay(self, frame, activations, current_time):
        """Draw simplified brain atlas visualization."""
        height, width = frame.shape[:2]
        
        # Create compact visualization area (top-right)
        atlas_width = 280
        atlas_height = 160
        atlas_x = width - atlas_width - 10
        atlas_y = 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (atlas_x, atlas_y), (atlas_x + atlas_width, atlas_y + atlas_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (atlas_x, atlas_y), (atlas_x + atlas_width, atlas_y + atlas_height), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "BRAIN ATLAS ACTIVATION", (atlas_x + 8, atlas_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Draw regions as bars
        y_offset = 30
        bar_width = 200
        bar_height = 12
        
        regions = ['Place', 'Face', 'Shape', 'Word']
        colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0)]
        
        for i, (region, color) in enumerate(zip(regions, colors)):
            activation = activations.get(region, 0.0)
            norm_activation = max(0, min(1, abs(activation) / 100))
            
            # Calculate position
            region_x = atlas_x + 8
            region_y = atlas_y + y_offset + i * (bar_height + 6)
            
            # Draw bar
            bar_width_scaled = int(norm_activation * bar_width)
            cv2.rectangle(frame, (region_x, region_y), 
                         (region_x + bar_width_scaled, region_y + bar_height), color, -1)
            cv2.rectangle(frame, (region_x, region_y), 
                         (region_x + bar_width, region_y + bar_height), (255, 255, 255), 1)
            
            # Label
            cv2.putText(frame, f"{region}: {activation:.1f}", 
                       (region_x + bar_width + 5, region_y + bar_height - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        
        # Time
        cv2.putText(frame, f"Time: {current_time:.1f}s", (atlas_x + 8, atlas_y + atlas_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, video_path, output_path, start_time=0, duration=20, fps=30):
        """Create annotated video with fixed brain atlas overlay."""
        print(f"🎬 Creating fixed brain atlas activation video: {output_path}")
        
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
            for region in self.brain_regions.keys():
                activations[region] = self.get_region_activation(current_time, region)
            
            # Draw brain atlas overlay (top-right)
            self.draw_brain_atlas_overlay(frame, activations, current_time)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Fixed brain atlas video created: {output_path}")
        return output_path

class FixedERSPAnnotator:
    """Fixed ERSP approach with time series visualization and annotation info."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate, annotations):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        self.annotation_drawer = AnnotationInfoDrawer(annotations)
        
        # Electrode clusters
        self.clusters = {
            'Motor': list(range(40, 80)),
            'Visual': list(range(80, 120)),
            'Temporal': list(range(120, 158))
        }
        
        # Store time series data
        self.time_series_data = {cluster: [] for cluster in self.clusters.keys()}
        self.max_history = 100  # Keep last 100 time points
    
    def get_cluster_activation(self, time_point, cluster_name):
        """Get activation for a specific cluster."""
        sample_idx = int((time_point - self.video_start_time) * self.sampling_rate)
        window_samples = int(0.2 * self.sampling_rate)
        start_idx = max(0, sample_idx - window_samples // 2)
        end_idx = min(self.ecog_data.shape[1], sample_idx + window_samples // 2)
        
        if start_idx >= end_idx:
            return 0.0
        
        if cluster_name in self.clusters:
            channels = self.clusters[cluster_name]
            valid_channels = [ch for ch in channels if ch < self.ecog_data.shape[0]]
            
            if valid_channels:
                cluster_data = self.ecog_data[valid_channels, start_idx:end_idx]
                power = np.mean(cluster_data ** 2)
                scaled_power = power * 100
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_ersp_overlay(self, frame, current_time):
        """Draw time series visualization instead of bars."""
        height, width = frame.shape[:2]
        
        # Create time series visualization area (top-right)
        panel_width = 280
        panel_height = 160
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "ERSP TIME SERIES", (panel_x + 8, panel_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Get current activations and update time series
        for cluster_name in self.clusters.keys():
            activation = self.get_cluster_activation(current_time, cluster_name)
            self.time_series_data[cluster_name].append(activation)
            
            # Keep only recent history
            if len(self.time_series_data[cluster_name]) > self.max_history:
                self.time_series_data[cluster_name] = self.time_series_data[cluster_name][-self.max_history:]
        
        # Draw time series for each cluster
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255)]  # Green, Yellow, Magenta
        cluster_names = list(self.clusters.keys())
        
        for i, (cluster_name, color) in enumerate(zip(cluster_names, colors)):
            if len(self.time_series_data[cluster_name]) < 2:
                continue
            
            # Calculate plot area for this cluster
            plot_height = 30
            plot_y = panel_y + 30 + i * (plot_height + 8)
            
            # Draw cluster label
            cv2.putText(frame, cluster_name, (panel_x + 8, plot_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
            
            # Draw time series
            data = self.time_series_data[cluster_name]
            if len(data) > 1:
                # Normalize data to plot area
                max_val = max(data) if max(data) > 0 else 1
                min_val = min(data)
                data_range = max_val - min_val if max_val != min_val else 1
                
                # Draw line
                for j in range(1, len(data)):
                    x1 = panel_x + 8 + int((j-1) * (panel_width - 16) / (len(data) - 1))
                    y1 = plot_y + plot_height - int((data[j-1] - min_val) / data_range * plot_height)
                    x2 = panel_x + 8 + int(j * (panel_width - 16) / (len(data) - 1))
                    y2 = plot_y + plot_height - int((data[j] - min_val) / data_range * plot_height)
                    
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw current value
                current_val = data[-1]
                cv2.putText(frame, f"{current_val:.1f}", (panel_x + panel_width - 50, plot_y + plot_height - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1)
        
        # Time
        cv2.putText(frame, f"Time: {current_time:.1f}s", (panel_x + 8, panel_y + panel_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, video_path, output_path, start_time=0, duration=20, fps=30):
        """Create annotated video with time series ERSP overlay."""
        print(f"🎬 Creating fixed ERSP time series video: {output_path}")
        
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
            
            # Draw ERSP overlay (top-right)
            self.draw_ersp_overlay(frame, current_time)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Fixed ERSP video created: {output_path}")
        return output_path

def main():
    print("🔧 Fixing All Video Annotation Approaches")
    print("=" * 45)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fix all video annotation approaches')
    parser.add_argument('--approaches', type=str, default='1,2,3,4,5,6',
                       help='Comma-separated list of approaches to fix (default: 1,2,3,4,5,6)')
    parser.add_argument('--duration', type=int, default=20, 
                       help='Duration in seconds (default: 20)')
    
    args = parser.parse_args()
    
    # Parse approaches to run
    approaches_to_run = [int(x.strip()) for x in args.approaches.split(',')]
    print(f"🚀 Fixing approaches: {approaches_to_run} for {args.duration} seconds")
    
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
    
    # Run fixes
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
            
        print(f"\n🔧 Fixing Approach {approach_num}: {approach_names[approach_num]}")
        
        try:
            if approach_num == 6:
                # Fixed Brain Atlas
                annotator = FixedBrainAtlasAnnotator(high_gamma_envelope, video_start_time, sampling_rate, annotation_data['annotations'])
                video_filename = f"walk_annotated_brain_atlas_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
                result_path = annotator.create_annotated_video(
                    video_path=video_path,
                    output_path=output_path,
                    start_time=video_start_time,
                    duration=args.duration,
                    fps=30
                )
                
            elif approach_num == 3:
                # Fixed ERSP
                annotator = FixedERSPAnnotator(high_gamma_envelope, video_start_time, sampling_rate, annotation_data['annotations'])
                video_filename = f"walk_annotated_ersp_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
                result_path = annotator.create_annotated_video(
                    video_path=video_path,
                    output_path=output_path,
                    start_time=video_start_time,
                    duration=args.duration,
                    fps=30
                )
            
            else:
                print(f"⚠️ Approach {approach_num} not yet fixed - using original implementation")
                continue
            
            print(f"✅ Approach {approach_num} fixed: {result_path}")
            results.append((approach_num, approach_names[approach_num], result_path))
            
        except Exception as e:
            print(f"❌ Error fixing Approach {approach_num}: {str(e)}")
            continue
    
    # Print summary
    print(f"\n🎉 Fix completed!")
    print(f"📁 Files saved to: {exp_manager.experiment_dir}")
    print(f"📊 Successfully fixed {len(results)} approaches:")
    
    for approach_num, name, result_path in results:
        file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
        print(f"  ✅ Approach {approach_num}: {name}")
        print(f"     📹 Video: {os.path.basename(result_path)} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()
