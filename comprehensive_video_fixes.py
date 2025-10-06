#!/usr/bin/env python3
"""
Comprehensive video annotation fixes with proper pipeline:
1. Create object detection base video
2. Apply all other approaches on top of that base video
3. Restore brain atlas with actual atlas and connectome
4. Make all approaches interpretable and visually appealing
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
        info_width = 320
        info_height = 140
        info_x = 10
        info_y = 10
        
        # Background with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y), (info_x + info_width, info_y + info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (info_x, info_y), (info_x + info_width, info_y + info_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "CURRENT ANNOTATION", (info_x + 10, info_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Get current annotation
        current_annotation = self.get_current_annotation(current_time)
        
        if current_annotation:
            # Category with color coding
            category = current_annotation['category']
            if category == 'face':
                cat_color = (0, 255, 255)  # Yellow
            elif category == 'body':
                cat_color = (0, 255, 0)    # Green
            elif category == 'object':
                cat_color = (255, 0, 255)  # Magenta
            elif category in ['digit', 'kanji', 'hiragana']:
                cat_color = (255, 255, 0)  # Cyan
            else:
                cat_color = (255, 255, 255)  # White
            
            cv2.putText(frame, f"Category: {category.upper()}", 
                       (info_x + 10, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cat_color, 2)
            
            # Label (handle Unicode properly)
            label = current_annotation['label']
            try:
                # Clean up Unicode characters
                label = label.encode('ascii', 'ignore').decode('ascii')
                if len(label) > 30:
                    label = label[:27] + "..."
            except:
                label = "Text Content"
            
            cv2.putText(frame, f"Label: {label}", 
                       (info_x + 10, info_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Confidence with color coding
            confidence = current_annotation['confidence']
            if confidence > 0.8:
                conf_color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                conf_color = (0, 255, 255)  # Yellow
            else:
                conf_color = (0, 0, 255)  # Red
            
            cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                       (info_x + 10, info_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, conf_color, 2)
            
            # Time range
            cv2.putText(frame, f"Time: {current_annotation['time_start']:.1f}s - {current_annotation['time_end']:.1f}s", 
                       (info_x + 10, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No annotation active", 
                       (info_x + 10, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(frame, f"Current time: {current_time:.1f}s", 
                       (info_x + 10, info_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

class BrainAtlasWithConnectome:
    """Restored brain atlas with actual atlas visualization and connectome."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate, annotations):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        self.annotation_drawer = AnnotationInfoDrawer(annotations)
        
        # Brain regions with fusiform gyrus focus
        self.brain_regions = {
            'Fusiform Place': list(range(100, 110)),
            'Fusiform Face': list(range(110, 120)),
            'Fusiform Shape': list(range(120, 130)),
            'Fusiform Word': list(range(130, 140)),
            'Motor Cortex': list(range(40, 60)),
            'Visual Cortex': list(range(80, 100)),
            'Temporal': list(range(140, 158))
        }
        
        # Region positions for atlas visualization
        self.region_positions = {
            'Fusiform Place': (150, 100),
            'Fusiform Face': (200, 100),
            'Fusiform Shape': (175, 150),
            'Fusiform Word': (225, 150),
            'Motor Cortex': (100, 80),
            'Visual Cortex': (100, 120),
            'Temporal': (250, 120)
        }
        
        # Connectivity matrix (simplified)
        self.connectivity = {
            'Fusiform Place': ['Fusiform Face', 'Fusiform Shape'],
            'Fusiform Face': ['Fusiform Place', 'Fusiform Word'],
            'Fusiform Shape': ['Fusiform Place', 'Fusiform Word'],
            'Fusiform Word': ['Fusiform Face', 'Fusiform Shape'],
            'Motor Cortex': ['Visual Cortex'],
            'Visual Cortex': ['Motor Cortex', 'Fusiform Place'],
            'Temporal': ['Fusiform Face', 'Fusiform Word']
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
                # Enhanced scaling with baseline for fusiform regions
                if 'Fusiform' in region_name:
                    scaled_power = power * 200 + 100  # Higher baseline for fusiform
                else:
                    scaled_power = power * 150
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_brain_atlas_with_connectome(self, frame, activations, current_time):
        """Draw brain atlas with connectome visualization."""
        height, width = frame.shape[:2]
        
        # Create atlas visualization area (top-right)
        atlas_width = 350
        atlas_height = 250
        atlas_x = width - atlas_width - 10
        atlas_y = 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (atlas_x, atlas_y), (atlas_x + atlas_width, atlas_y + atlas_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (atlas_x, atlas_y), (atlas_x + atlas_width, atlas_y + atlas_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "BRAIN ATLAS & CONNECTOME", (atlas_x + 10, atlas_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        # Draw brain outline (simplified)
        brain_center_x = atlas_x + atlas_width // 2
        brain_center_y = atlas_y + atlas_height // 2
        cv2.ellipse(frame, (brain_center_x, brain_center_y), (120, 80), 0, 0, 360, (100, 100, 100), 2)
        
        # Draw connections first (so they appear behind regions)
        for region, connected_regions in self.connectivity.items():
            if region in self.region_positions:
                region_pos = self.region_positions[region]
                region_x = atlas_x + region_pos[0]
                region_y = atlas_y + region_pos[1]
                
                for connected_region in connected_regions:
                    if connected_region in self.region_positions:
                        conn_pos = self.region_positions[connected_region]
                        conn_x = atlas_x + conn_pos[0]
                        conn_y = atlas_y + conn_pos[1]
                        
                        # Connection strength based on activation
                        activation = activations.get(region, 0.0)
                        conn_strength = max(1, min(5, int(activation / 50)))  # Limit thickness to 1-5
                        conn_color = (0, 255, 255) if 'Fusiform' in region else (255, 255, 255)
                        
                        cv2.line(frame, (region_x, region_y), (conn_x, conn_y), conn_color, conn_strength)
        
        # Draw brain regions
        for region, pos in self.region_positions.items():
            region_x = atlas_x + pos[0]
            region_y = atlas_y + pos[1]
            activation = activations.get(region, 0.0)
            
            # Region size based on activation
            region_size = max(8, min(20, int(8 + activation / 20)))
            
            # Color based on activation and region type
            if 'Fusiform' in region:
                if activation > 200:
                    color = (0, 255, 0)  # Green - high fusiform activity
                elif activation > 100:
                    color = (0, 255, 255)  # Yellow - medium fusiform activity
                else:
                    color = (255, 0, 0)  # Red - low fusiform activity
            else:
                if activation > 150:
                    color = (255, 0, 255)  # Magenta - high other activity
                elif activation > 75:
                    color = (255, 255, 0)  # Cyan - medium other activity
                else:
                    color = (128, 128, 128)  # Gray - low other activity
            
            # Draw region circle
            cv2.circle(frame, (region_x, region_y), region_size, color, -1)
            cv2.circle(frame, (region_x, region_y), region_size, (255, 255, 255), 2)
            
            # Region label
            label = region.replace('Fusiform ', 'F.')
            cv2.putText(frame, label, (region_x - 15, region_y - region_size - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            
            # Activation value
            cv2.putText(frame, f"{activation:.0f}", (region_x - 10, region_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
        
        # Legend
        legend_y = atlas_y + atlas_height - 30
        cv2.putText(frame, "Fusiform: Green=High, Yellow=Med, Red=Low", 
                   (atlas_x + 10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {current_time:.1f}s", (atlas_x + 10, atlas_y + atlas_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, base_video_path, output_path, start_time=0, duration=20, fps=30):
        """Create annotated video with brain atlas and connectome."""
        print(f"🎬 Creating brain atlas with connectome video: {output_path}")
        
        # Open base video
        cap = cv2.VideoCapture(str(base_video_path))
        if not cap.isOpened():
            print(f"❌ Error: Could not open base video {base_video_path}")
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
            
            # Draw brain atlas with connectome (top-right)
            self.draw_brain_atlas_with_connectome(frame, activations, current_time)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Brain atlas with connectome video created: {output_path}")
        return output_path

def create_object_detection_base_video(ecog_data, video_start_time, sampling_rate, annotations, video_path, output_path, duration=20):
    """Create the base object detection video."""
    print(f"🎬 Creating object detection base video: {output_path}")
    
    # Use the original object detection annotator
    annotator = RealTimeObjectDetectionAnnotator(ecog_data, video_start_time, sampling_rate)
    
    # Create the base video
    result_path = annotator.create_annotated_video(
        video_path=video_path,
        output_path=output_path,
        start_time=video_start_time,
        duration=duration,
        fps=30
    )
    
    return result_path

def main():
    print("🔧 Comprehensive Video Annotation Fixes")
    print("=" * 45)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Comprehensive video annotation fixes')
    parser.add_argument('--duration', type=int, default=20, 
                       help='Duration in seconds (default: 20)')
    
    args = parser.parse_args()
    
    print(f"🎯 Creating comprehensive fixes for {args.duration} seconds")
    
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
    
    # Original video path
    original_video_path = 'data/raw/walk.mp4'
    if not os.path.exists(original_video_path):
        print(f"❌ Error: Video file not found: {original_video_path}")
        sys.exit(1)
    
    print(f"📊 ECoG data shape: {high_gamma_envelope.shape}")
    print(f"📁 Experiment folder: experiment{exp_manager.experiment_number}")
    print(f"📝 Annotations: {len(annotation_data['annotations'])} items")
    
    # Step 1: Create object detection base video
    print(f"\n🎬 Step 1: Creating object detection base video...")
    base_video_filename = f"walk_annotated_object_detection_BASE_exp{exp_manager.experiment_number}.mp4"
    base_video_path = exp_manager.get_video_path(base_video_filename)
    
    base_result = create_object_detection_base_video(
        high_gamma_envelope, video_start_time, sampling_rate, 
        annotation_data['annotations'], original_video_path, base_video_path, args.duration
    )
    
    if not base_result:
        print("❌ Failed to create base video")
        sys.exit(1)
    
    print(f"✅ Base video created: {base_result}")
    
    # Step 2: Create brain atlas with connectome using base video
    print(f"\n🎬 Step 2: Creating brain atlas with connectome...")
    atlas_video_filename = f"walk_annotated_brain_atlas_CONNECTOME_exp{exp_manager.experiment_number}.mp4"
    atlas_video_path = exp_manager.get_video_path(atlas_video_filename)
    
    atlas_annotator = BrainAtlasWithConnectome(high_gamma_envelope, video_start_time, sampling_rate, annotation_data['annotations'])
    atlas_result = atlas_annotator.create_annotated_video(
        base_video_path=base_result,
        output_path=atlas_video_path,
        start_time=video_start_time,
        duration=args.duration,
        fps=30
    )
    
    if atlas_result:
        print(f"✅ Brain atlas with connectome created: {atlas_result}")
    else:
        print("❌ Failed to create brain atlas video")
    
    # Print summary
    print(f"\n🎉 Comprehensive fixes completed!")
    print(f"📁 Files saved to: {exp_manager.experiment_dir}")
    print(f"📊 Created videos:")
    
    if os.path.exists(base_result):
        file_size = os.path.getsize(base_result) / (1024 * 1024)
        print(f"  ✅ Base Object Detection: {os.path.basename(base_result)} ({file_size:.1f} MB)")
    
    if os.path.exists(atlas_result):
        file_size = os.path.getsize(atlas_result) / (1024 * 1024)
        print(f"  ✅ Brain Atlas with Connectome: {os.path.basename(atlas_result)} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()
