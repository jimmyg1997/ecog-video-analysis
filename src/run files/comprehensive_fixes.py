#!/usr/bin/env python3
"""
Comprehensive fixes for all video annotation approaches based on user feedback.
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

def main():
    print("🔧 Comprehensive Fixes for All Approaches")
    print("=" * 45)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Apply comprehensive fixes to all approaches')
    parser.add_argument('--approach', type=int, choices=[6], 
                       help='Approach to fix (6: Brain Atlas)')
    parser.add_argument('--duration', type=int, default=20, 
                       help='Duration in seconds (default: 20)')
    
    args = parser.parse_args()
    
    if not args.approach:
        print("❌ Please specify an approach to fix with --approach <number>")
        print("Available fixes: 6 (Brain Atlas)")
        sys.exit(1)
    
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
    if args.approach == 6:
        print("🔧 Fixing Brain Atlas Approach...")
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
        
        print(f"✅ Approach 6 fixed: {result_path}")
    
    print(f"\n🎉 Fix completed!")
    print(f"📁 Files saved to: {exp_manager.experiment_dir}")

if __name__ == "__main__":
    main()
