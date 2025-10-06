#!/usr/bin/env python3
"""
Fix all video annotation approaches based on user feedback.
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

class FixedBrainAtlasAnnotator:
    """Fixed brain atlas approach with simplified visualization."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate=600.0):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        
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
        
        # Create compact visualization area
        atlas_width = 300
        atlas_height = 180
        atlas_x = width - atlas_width - 10
        atlas_y = 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (atlas_x, atlas_y), (atlas_x + atlas_width, atlas_y + atlas_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (atlas_x, atlas_y), (atlas_x + atlas_width, atlas_y + atlas_height), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "BRAIN ATLAS ACTIVATION", (atlas_x + 8, atlas_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw regions as bars
        y_offset = 30
        bar_width = 250
        bar_height = 15
        
        regions = ['Place', 'Face', 'Shape', 'Word']
        colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0)]
        
        for i, (region, color) in enumerate(zip(regions, colors)):
            activation = activations.get(region, 0.0)
            norm_activation = max(0, min(1, abs(activation) / 100))
            
            # Calculate position
            region_x = atlas_x + 8
            region_y = atlas_y + y_offset + i * (bar_height + 8)
            
            # Draw bar
            bar_width_scaled = int(norm_activation * bar_width)
            cv2.rectangle(frame, (region_x, region_y), 
                         (region_x + bar_width_scaled, region_y + bar_height), color, -1)
            cv2.rectangle(frame, (region_x, region_y), 
                         (region_x + bar_width, region_y + bar_height), (255, 255, 255), 1)
            
            # Label
            cv2.putText(frame, f"{region}: {activation:.1f}", 
                       (region_x + bar_width + 5, region_y + bar_height - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Time
        cv2.putText(frame, f"Time: {current_time:.1f}s", (atlas_x + 8, atlas_y + atlas_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
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
            
            # Get activations
            activations = {}
            for region in self.brain_regions.keys():
                activations[region] = self.get_region_activation(current_time, region)
            
            # Draw overlay
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
    """Fixed ERSP approach with time series visualization instead of bars."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate=600.0):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        
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
        
        # Create time series visualization area
        panel_width = 300
        panel_height = 200
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "ERSP TIME SERIES", (panel_x + 8, panel_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
            plot_height = 40
            plot_y = panel_y + 30 + i * (plot_height + 10)
            
            # Draw cluster label
            cv2.putText(frame, cluster_name, (panel_x + 8, plot_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
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
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        
        # Time
        cv2.putText(frame, f"Time: {current_time:.1f}s", (panel_x + 8, panel_y + panel_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
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
            
            # Draw overlay
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
    print("🔧 Fixing Video Annotation Approaches")
    print("=" * 40)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fix video annotation approaches')
    parser.add_argument('--approach', type=int, choices=[6, 3], 
                       help='Approach to fix (6: Brain Atlas, 3: ERSP)')
    parser.add_argument('--duration', type=int, default=20, 
                       help='Duration in seconds (default: 20)')
    
    args = parser.parse_args()
    
    if not args.approach:
        print("❌ Please specify an approach to fix with --approach <number>")
        print("Available fixes: 6 (Brain Atlas), 3 (ERSP)")
        sys.exit(1)
    
    # Initialize
    exp_manager = ExperimentManager()
    data_loader = DataLoader()
    
    # Load data
    preprocessed_data = data_loader.load_preprocessed()
    high_gamma_envelope = preprocessed_data.get('filtered_data', None)
    video_start_time = 0.0
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
    print(f"📁 Experiment folder: experiment{exp_manager.experiment_number}")
    
    # Run fixes
    if args.approach == 6:
        print("🔧 Fixing Brain Atlas Approach...")
        annotator = FixedBrainAtlasAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
        video_filename = f"walk_annotated_brain_atlas_FIXED_exp{exp_manager.experiment_number}.mp4"
        output_path = exp_manager.get_video_path(video_filename)
        
        result_path = annotator.create_annotated_video(
            video_path=video_path,
            output_path=output_path,
            start_time=video_start_time,
            duration=args.duration,
            fps=30
        )
        
    elif args.approach == 3:
        print("🔧 Fixing ERSP Approach...")
        annotator = FixedERSPAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
        video_filename = f"walk_annotated_ersp_FIXED_exp{exp_manager.experiment_number}.mp4"
        output_path = exp_manager.get_video_path(video_filename)
        
        result_path = annotator.create_annotated_video(
            video_path=video_path,
            output_path=output_path,
            start_time=video_start_time,
            duration=args.duration,
            fps=30
        )
    
    print(f"\n🎉 Fix completed!")
    print(f"📁 Fixed video saved to: {result_path}")

if __name__ == "__main__":
    main()
