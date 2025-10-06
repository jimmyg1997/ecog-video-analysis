#!/usr/bin/env python3
"""
Fixed video annotation approaches with proper Unicode handling, 
dynamic visualizations, and all approaches in same experiment folder.
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
    ExperimentManager, DataLoader
)

class FixedObjectDetectionAnnotator:
    """Fixed object detection with proper Unicode handling."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate, annotations):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        self.annotations = annotations
        
        # Object detection channels
        self.object_channels = list(range(100, 140))  # Fusiform gyrus for object recognition
    
    def get_current_annotation(self, current_time):
        """Get current annotation for the given time."""
        for annotation in self.annotations:
            if annotation['time_start'] <= current_time <= annotation['time_end']:
                return annotation
        return None
    
    def get_object_activation(self, time_point):
        """Get object detection activation."""
        sample_idx = int((time_point - self.video_start_time) * self.sampling_rate)
        window_samples = int(0.5 * self.sampling_rate)
        start_idx = max(0, sample_idx - window_samples // 2)
        end_idx = min(self.ecog_data.shape[1], sample_idx + window_samples // 2)
        
        if start_idx >= end_idx:
            return 0.0
        
        valid_channels = [ch for ch in self.object_channels if ch < self.ecog_data.shape[0]]
        if valid_channels:
            object_data = self.ecog_data[valid_channels, start_idx:end_idx]
            power = np.mean(object_data ** 2)
            scaled_power = power * 1000
            return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def clean_label(self, label):
        """Clean label to avoid Unicode issues."""
        # Handle specific Japanese text cases
        if '湯呑' in label or 'yunomi' in label:
            return "Tea Cup"
        elif 'ねど' in label or 'nedoko' in label:
            return "Bed"
        elif '飛行機' in label or 'hikoki' in label:
            return "Airplane"
        elif '本' in label or 'hon' in label:
            return "Book"
        elif '犬' in label or 'inu' in label:
            return "Dog"
        elif '湯香' in label:
            return "Hot Water"
        elif 'ねどこ' in label:
            return "Bed (alt)"
        else:
            # For other cases, try to extract readable parts
            try:
                # Remove parentheses and their contents
                import re
                cleaned = re.sub(r'\([^)]*\)', '', label)
                cleaned = cleaned.strip()
                if len(cleaned) > 20:
                    cleaned = cleaned[:17] + "..."
                return cleaned if cleaned else "Text Content"
            except:
                return "Text Content"
    
    def draw_object_detection_info(self, frame, current_time):
        """Draw object detection information with proper Unicode handling."""
        height, width = frame.shape[:2]
        
        # Create info area (top-left)
        info_width = 320
        info_height = 180
        info_x = 10
        info_y = 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y), (info_x + info_width, info_y + info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (info_x, info_y), (info_x + info_width, info_y + info_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "OBJECT DETECTION", (info_x + 10, info_y + 20), 
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
            
            # Clean label
            clean_label = self.clean_label(current_annotation['label'])
            cv2.putText(frame, f"Label: {clean_label}", 
                       (info_x + 10, info_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Confidence
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
            
            # Brain activation
            brain_activation = self.get_object_activation(current_time)
            if brain_activation > 1500:
                brain_status = "HIGH ACTIVITY"
                brain_color = (0, 255, 0)
            elif brain_activation > 1000:
                brain_status = "MEDIUM ACTIVITY"
                brain_color = (0, 255, 255)
            elif brain_activation > 500:
                brain_status = "LOW ACTIVITY"
                brain_color = (255, 255, 0)
            else:
                brain_status = "MINIMAL ACTIVITY"
                brain_color = (255, 255, 255)
            
            cv2.putText(frame, f"Brain: {brain_status}", 
                       (info_x + 10, info_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.35, brain_color, 2)
            cv2.putText(frame, f"Power: {brain_activation:.0f}", 
                       (info_x + 10, info_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No object detected", 
                       (info_x + 10, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(frame, f"Current time: {current_time:.1f}s", 
                       (info_x + 10, info_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            brain_activation = self.get_object_activation(current_time)
            cv2.putText(frame, f"Brain Power: {brain_activation:.0f}", 
                       (info_x + 10, info_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
    
    def create_annotated_video(self, video_path, output_path, start_time=0, duration=20, fps=30):
        """Create object detection video."""
        print(f"🎬 Creating fixed object detection video: {output_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return None
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time * original_fps)
        end_frame = int((start_time + duration) * original_fps)
        end_frame = min(end_frame, total_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = start_time + frame_count / original_fps
            
            # Draw object detection info
            self.draw_object_detection_info(frame, current_time)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        print(f"✅ Fixed object detection video created: {output_path}")
        return output_path

class FixedBrainAtlasAnnotator:
    """Fixed brain atlas with useful connectome visualization."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate, annotations):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        self.annotations = annotations
        
        # Brain regions with fusiform focus
        self.brain_regions = {
            'Fusiform Place': list(range(100, 110)),
            'Fusiform Face': list(range(110, 120)),
            'Fusiform Shape': list(range(120, 130)),
            'Fusiform Word': list(range(130, 140)),
            'Motor Cortex': list(range(40, 60)),
            'Visual Cortex': list(range(80, 100)),
            'Temporal': list(range(140, 158))
        }
        
        # Region positions for brain atlas
        self.region_positions = {
            'Fusiform Place': (150, 100),
            'Fusiform Face': (200, 100),
            'Fusiform Shape': (175, 150),
            'Fusiform Word': (225, 150),
            'Motor Cortex': (100, 80),
            'Visual Cortex': (100, 120),
            'Temporal': (250, 120)
        }
        
        # Connectivity
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
                # Enhanced scaling for fusiform regions
                if 'Fusiform' in region_name:
                    scaled_power = power * 300 + 150  # Higher baseline
                else:
                    scaled_power = power * 200
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_brain_atlas(self, frame, activations, current_time):
        """Draw brain atlas with connectome."""
        height, width = frame.shape[:2]
        
        # Create atlas area (top-right)
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
        
        # Draw brain outline
        brain_center_x = atlas_x + atlas_width // 2
        brain_center_y = atlas_y + atlas_height // 2
        cv2.ellipse(frame, (brain_center_x, brain_center_y), (120, 80), 0, 0, 360, (100, 100, 100), 2)
        
        # Draw connections first
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
                        
                        # Connection strength
                        activation = activations.get(region, 0.0)
                        conn_strength = max(1, min(5, int(activation / 100)))
                        conn_color = (0, 255, 255) if 'Fusiform' in region else (255, 255, 255)
                        
                        cv2.line(frame, (region_x, region_y), (conn_x, conn_y), conn_color, conn_strength)
        
        # Draw brain regions
        for region, pos in self.region_positions.items():
            region_x = atlas_x + pos[0]
            region_y = atlas_y + pos[1]
            activation = activations.get(region, 0.0)
            
            # Region size based on activation
            region_size = max(12, min(25, int(12 + activation / 50)))
            
            # Color based on activation and region type
            if 'Fusiform' in region:
                if activation > 300:
                    color = (0, 255, 0)  # Green - high fusiform
                elif activation > 200:
                    color = (0, 255, 255)  # Yellow - medium fusiform
                else:
                    color = (255, 0, 0)  # Red - low fusiform
            else:
                if activation > 200:
                    color = (255, 0, 255)  # Magenta - high other
                elif activation > 100:
                    color = (255, 255, 0)  # Cyan - medium other
                else:
                    color = (128, 128, 128)  # Gray - low other
            
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
    
    def create_annotated_video(self, video_path, output_path, start_time=0, duration=20, fps=30):
        """Create brain atlas video."""
        print(f"🎬 Creating fixed brain atlas video: {output_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return None
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time * original_fps)
        end_frame = int((start_time + duration) * original_fps)
        end_frame = min(end_frame, total_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
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
            
            # Draw brain atlas
            self.draw_brain_atlas(frame, activations, current_time)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        print(f"✅ Fixed brain atlas video created: {output_path}")
        return output_path

class FixedGaitPhaseAnnotator:
    """Fixed gait phase with dynamic visualization."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate, annotations):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        self.annotations = annotations
        
        # Gait channels
        self.gait_channels = {
            'Heel Strike': list(range(40, 60)),
            'Stance': list(range(60, 80)),
            'Toe Off': list(range(80, 100)),
            'Swing': list(range(100, 120))
        }
        
        # Store history for dynamic visualization
        self.activation_history = {phase: [] for phase in self.gait_channels.keys()}
        self.max_history = 50
    
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
                scaled_power = power * 1500  # Enhanced scaling
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_gait_timeline(self, frame, current_time):
        """Draw dynamic gait phase timeline."""
        height, width = frame.shape[:2]
        
        # Create timeline area (bottom-right)
        timeline_width = 320
        timeline_height = 200
        timeline_x = width - timeline_width - 10
        timeline_y = height - timeline_height - 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (timeline_x, timeline_y), (timeline_x + timeline_width, timeline_y + timeline_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (timeline_x, timeline_y), (timeline_x + timeline_width, timeline_y + timeline_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "GAIT PHASE ANALYSIS", (timeline_x + 10, timeline_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        # Get current activations and update history
        current_activations = {}
        for phase in self.gait_channels.keys():
            activation = self.get_gait_activation(current_time, phase)
            current_activations[phase] = activation
            self.activation_history[phase].append(activation)
            
            # Keep only recent history
            if len(self.activation_history[phase]) > self.max_history:
                self.activation_history[phase] = self.activation_history[phase][-self.max_history:]
        
        # Draw dynamic bars
        y_offset = 40
        bar_width = 200
        bar_height = 15
        
        phases = ['Heel Strike', 'Stance', 'Toe Off', 'Swing']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        icons = ['👣', '🦵', '🦶', '🏃']
        
        for i, (phase, color, icon) in enumerate(zip(phases, colors, icons)):
            activation = current_activations.get(phase, 0.0)
            norm_activation = max(0, min(1, abs(activation) / 1500))
            
            # Calculate position
            phase_x = timeline_x + 10
            phase_y = timeline_y + y_offset + i * (bar_height + 8)
            
            # Draw dynamic bar
            bar_width_scaled = int(norm_activation * bar_width)
            if bar_width_scaled > 0:
                # Gradient effect
                for j in range(bar_width_scaled):
                    intensity = int(255 * (j / bar_width_scaled))
                    cv2.line(frame, (phase_x + j, phase_y), (phase_x + j, phase_y + bar_height), 
                            (int(color[0] * intensity/255), int(color[1] * intensity/255), int(color[2] * intensity/255)), 1)
            
            # Bar border
            cv2.rectangle(frame, (phase_x, phase_y), (phase_x + bar_width, phase_y + bar_height), (255, 255, 255), 1)
            
            # Icon and label
            cv2.putText(frame, icon, (phase_x - 25, phase_y + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Interpret activation level
            if activation > 1200:
                level = "HIGH"
                level_color = (0, 255, 0)
            elif activation > 800:
                level = "MED"
                level_color = (0, 255, 255)
            elif activation > 400:
                level = "LOW"
                level_color = (255, 255, 0)
            else:
                level = "MIN"
                level_color = (255, 255, 255)
            
            # Phase name and interpretation
            cv2.putText(frame, f"{phase}: {level}", 
                       (phase_x + bar_width + 5, phase_y + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, level_color, 1)
        
        # Draw time series for most active phase
        most_active_phase = max(current_activations.keys(), key=lambda k: current_activations[k])
        if len(self.activation_history[most_active_phase]) > 5:
            # Draw mini time series
            plot_x = timeline_x + 10
            plot_y = timeline_y + timeline_height - 40
            plot_width = timeline_width - 20
            plot_height = 25
            
            cv2.putText(frame, f"Most Active: {most_active_phase}", 
                       (plot_x, plot_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            
            data = self.activation_history[most_active_phase]
            if len(data) > 1:
                max_val = max(data) if max(data) > 0 else 1
                min_val = min(data)
                data_range = max_val - min_val if max_val != min_val else 1
                
                for j in range(1, len(data)):
                    x1 = plot_x + int((j-1) * plot_width / (len(data) - 1))
                    y1 = plot_y + plot_height - int((data[j-1] - min_val) / data_range * plot_height)
                    x2 = plot_x + int(j * plot_width / (len(data) - 1))
                    y2 = plot_y + plot_height - int((data[j] - min_val) / data_range * plot_height)
                    
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Time
        cv2.putText(frame, f"Time: {current_time:.1f}s", (timeline_x + 10, timeline_y + timeline_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, video_path, output_path, start_time=0, duration=20, fps=30):
        """Create gait phase video."""
        print(f"🎬 Creating fixed gait phase video: {output_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return None
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time * original_fps)
        end_frame = int((start_time + duration) * original_fps)
        end_frame = min(end_frame, total_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = start_time + frame_count / original_fps
            
            # Draw gait timeline
            self.draw_gait_timeline(frame, current_time)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        print(f"✅ Fixed gait phase video created: {output_path}")
        return output_path

def main():
    print("🔧 Fixed Video Annotation Approaches")
    print("=" * 40)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fixed video annotation approaches')
    parser.add_argument('--approaches', type=str, default='1,2,3,4,5,6',
                       help='Comma-separated list of approaches to run (default: 1,2,3,4,5,6)')
    parser.add_argument('--duration', type=int, default=20, 
                       help='Duration in seconds (default: 20)')
    
    args = parser.parse_args()
    
    # Parse approaches to run
    approaches_to_run = [int(x.strip()) for x in args.approaches.split(',')]
    print(f"🎯 Running approaches: {approaches_to_run} for {args.duration} seconds")
    
    # Initialize - use same experiment for all
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
    
    # Run all approaches in same experiment
    results = []
    approach_names = {
        1: "Fixed Motor Cortex Activation",
        2: "Fixed Gait Phase Analysis", 
        3: "Fixed ERSP Time Series",
        4: "Fixed Enhanced Brain Activation",
        5: "Fixed Object Detection",
        6: "Fixed Brain Atlas with Connectome"
    }
    
    for approach_num in approaches_to_run:
        if approach_num not in approach_names:
            print(f"⚠️ Skipping unknown approach: {approach_num}")
            continue
            
        print(f"\n🎬 Running Approach {approach_num}: {approach_names[approach_num]}")
        
        try:
            if approach_num == 5:
                # Fixed Object Detection
                annotator = FixedObjectDetectionAnnotator(high_gamma_envelope, video_start_time, sampling_rate, annotation_data['annotations'])
                video_filename = f"walk_annotated_object_detection_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 6:
                # Fixed Brain Atlas
                annotator = FixedBrainAtlasAnnotator(high_gamma_envelope, video_start_time, sampling_rate, annotation_data['annotations'])
                video_filename = f"walk_annotated_brain_atlas_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 2:
                # Fixed Gait Phase
                annotator = FixedGaitPhaseAnnotator(high_gamma_envelope, video_start_time, sampling_rate, annotation_data['annotations'])
                video_filename = f"walk_annotated_gait_phase_FIXED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
            
            else:
                print(f"⚠️ Approach {approach_num} not yet implemented in fixed version")
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
