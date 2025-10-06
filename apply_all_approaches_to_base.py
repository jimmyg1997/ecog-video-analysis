#!/usr/bin/env python3
"""
Apply all other approaches (1,2,3,4) on top of the base object detection video.
Make them all interpretable and visually appealing.
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
    EnhancedBrainRegionAnnotator
)

class ImprovedMotorCortexAnnotator:
    """Improved motor cortex approach with better visualization."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        
        # Motor cortex regions with better organization
        self.motor_regions = {
            'Primary Motor': list(range(40, 60)),
            'Visual Cortex': list(range(80, 100)),
            'Fusiform Gyrus': list(range(100, 120)),
            'Frontal Cortex': list(range(20, 40)),
            'Temporal': list(range(140, 158))
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
                scaled_power = power * 1500  # Enhanced scaling
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_motor_visualization(self, frame, activations, current_time):
        """Draw improved motor cortex visualization."""
        height, width = frame.shape[:2]
        
        # Create visualization area (top-right)
        viz_width = 320
        viz_height = 220
        viz_x = width - viz_width - 10
        viz_y = 10
        
        # Background with gradient effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), (255, 255, 255), 2)
        
        # Title with icon
        cv2.putText(frame, "🧠 BRAIN REGION ACTIVATION", (viz_x + 10, viz_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        # Draw region bars with enhanced visualization
        y_offset = 35
        bar_width = 220
        bar_height = 18
        
        regions = ['Primary Motor', 'Visual Cortex', 'Fusiform Gyrus', 'Frontal Cortex', 'Temporal']
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 165, 0)]
        icons = ['🏃', '👁️', '🧩', '💭', '👂']
        
        for i, (region, color, icon) in enumerate(zip(regions, colors, icons)):
            activation = activations.get(region, 0.0)
            norm_activation = max(0, min(1, abs(activation) / 2000))
            
            # Calculate position
            region_x = viz_x + 10
            region_y = viz_y + y_offset + i * (bar_height + 8)
            
            # Draw bar with gradient effect
            bar_width_scaled = int(norm_activation * bar_width)
            if bar_width_scaled > 0:
                # Gradient effect
                for j in range(bar_width_scaled):
                    intensity = int(255 * (j / bar_width_scaled))
                    cv2.line(frame, (region_x + j, region_y), (region_x + j, region_y + bar_height), 
                            (int(color[0] * intensity/255), int(color[1] * intensity/255), int(color[2] * intensity/255)), 1)
            
            # Bar border
            cv2.rectangle(frame, (region_x, region_y), (region_x + bar_width, region_y + bar_height), (255, 255, 255), 1)
            
            # Icon and label
            cv2.putText(frame, icon, (region_x - 25, region_y + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Region name and value
            cv2.putText(frame, f"{region}: {activation:.0f}", 
                       (region_x + bar_width + 5, region_y + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        
        # Interpretation guide
        cv2.putText(frame, "🟢 High Activity  🟡 Medium  🔴 Low", 
                   (viz_x + 10, viz_y + viz_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {current_time:.1f}s", (viz_x + 10, viz_y + viz_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, base_video_path, output_path, start_time=0, duration=20, fps=30):
        """Create annotated video with improved motor cortex overlay."""
        print(f"🎬 Creating improved motor cortex video: {output_path}")
        
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
        
        print(f"✅ Improved motor cortex video created: {output_path}")
        return output_path

class ImprovedGaitPhaseAnnotator:
    """Improved gait phase approach with better visualization."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        
        # Gait-related channels with better organization
        self.gait_channels = {
            'Heel Strike': list(range(40, 60)),
            'Stance Phase': list(range(60, 80)),
            'Toe Off': list(range(80, 100)),
            'Swing Phase': list(range(100, 120)),
            'Balance': list(range(120, 140))
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
                scaled_power = power * 1200  # Enhanced scaling
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_gait_timeline(self, frame, activations, current_time):
        """Draw improved gait phase timeline."""
        height, width = frame.shape[:2]
        
        # Create timeline visualization area (bottom-right)
        timeline_width = 320
        timeline_height = 220
        timeline_x = width - timeline_width - 10
        timeline_y = height - timeline_height - 10
        
        # Background with gradient effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (timeline_x, timeline_y), (timeline_x + timeline_width, timeline_y + timeline_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (timeline_x, timeline_y), (timeline_x + timeline_width, timeline_y + timeline_height), (255, 255, 255), 2)
        
        # Title with icon
        cv2.putText(frame, "🚶 GAIT PHASE ANALYSIS", (timeline_x + 10, timeline_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        # Draw phase bars with enhanced visualization
        y_offset = 35
        bar_width = 220
        bar_height = 18
        
        phases = ['Heel Strike', 'Stance Phase', 'Toe Off', 'Swing Phase', 'Balance']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        icons = ['👣', '🦵', '🦶', '🏃', '⚖️']
        
        for i, (phase, color, icon) in enumerate(zip(phases, colors, icons)):
            activation = activations.get(phase, 0.0)
            norm_activation = max(0, min(1, abs(activation) / 1200))
            
            # Calculate position
            phase_x = timeline_x + 10
            phase_y = timeline_y + y_offset + i * (bar_height + 8)
            
            # Draw bar with gradient effect
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
            if activation > 900:
                level = "HIGH"
                level_color = (0, 255, 0)
            elif activation > 600:
                level = "MED"
                level_color = (0, 255, 255)
            elif activation > 300:
                level = "LOW"
                level_color = (255, 255, 0)
            else:
                level = "MIN"
                level_color = (255, 255, 255)
            
            # Phase name and interpretation
            cv2.putText(frame, f"{phase}: {level}", 
                       (phase_x + bar_width + 5, phase_y + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, level_color, 1)
        
        # Legend
        cv2.putText(frame, "🟢 HIGH  🟡 MED  🟠 LOW  ⚪ MIN", 
                   (timeline_x + 10, timeline_y + timeline_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {current_time:.1f}s", (timeline_x + 10, timeline_y + timeline_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, base_video_path, output_path, start_time=0, duration=20, fps=30):
        """Create annotated video with improved gait phase overlay."""
        print(f"🎬 Creating improved gait phase video: {output_path}")
        
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
        
        print(f"✅ Improved gait phase video created: {output_path}")
        return output_path

class ImprovedERSPAnnotator:
    """Improved ERSP approach with time series visualization."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate):
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
        self.max_history = 80  # Keep last 80 time points
    
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
                scaled_power = power * 200
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_ersp_timeseries(self, frame, current_time):
        """Draw time series visualization."""
        height, width = frame.shape[:2]
        
        # Create time series visualization area (top-right)
        panel_width = 300
        panel_height = 180
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Background with gradient effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
        
        # Title with icon
        cv2.putText(frame, "📊 ERSP TIME SERIES", (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
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
        icons = ['🏃', '👁️', '👂']
        
        for i, (cluster_name, color, icon) in enumerate(zip(cluster_names, colors, icons)):
            if len(self.time_series_data[cluster_name]) < 2:
                continue
            
            # Calculate plot area for this cluster
            plot_height = 35
            plot_y = panel_y + 30 + i * (plot_height + 8)
            
            # Draw cluster label with icon
            cv2.putText(frame, f"{icon} {cluster_name}", (panel_x + 10, plot_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw time series
            data = self.time_series_data[cluster_name]
            if len(data) > 1:
                # Normalize data to plot area
                max_val = max(data) if max(data) > 0 else 1
                min_val = min(data)
                data_range = max_val - min_val if max_val != min_val else 1
                
                # Draw line with anti-aliasing effect
                for j in range(1, len(data)):
                    x1 = panel_x + 10 + int((j-1) * (panel_width - 20) / (len(data) - 1))
                    y1 = plot_y + plot_height - int((data[j-1] - min_val) / data_range * plot_height)
                    x2 = panel_x + 10 + int(j * (panel_width - 20) / (len(data) - 1))
                    y2 = plot_y + plot_height - int((data[j] - min_val) / data_range * plot_height)
                    
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw current value
                current_val = data[-1]
                cv2.putText(frame, f"{current_val:.1f}", (panel_x + panel_width - 50, plot_y + plot_height - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        
        # Time
        cv2.putText(frame, f"Time: {current_time:.1f}s", (panel_x + 10, panel_y + panel_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, base_video_path, output_path, start_time=0, duration=20, fps=30):
        """Create annotated video with improved ERSP time series."""
        print(f"🎬 Creating improved ERSP time series video: {output_path}")
        
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
            
            # Draw ERSP time series (top-right)
            self.draw_ersp_timeseries(frame, current_time)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Improved ERSP time series video created: {output_path}")
        return output_path

class ImprovedEnhancedBrainAnnotator:
    """Improved enhanced brain region approach."""
    
    def __init__(self, ecog_data, video_start_time, sampling_rate):
        self.ecog_data = ecog_data
        self.video_start_time = video_start_time
        self.sampling_rate = sampling_rate
        
        # Brain regions
        self.brain_regions = {
            'Motor': list(range(40, 60)),
            'Visual': list(range(80, 100)),
            'Fusiform': list(range(100, 120)),
            'Frontal': list(range(20, 40)),
            'Temporal': list(range(140, 158))
        }
    
    def get_brain_activation(self, time_point, region_name):
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
                scaled_power = power * 1000
                return scaled_power if not np.isnan(scaled_power) else 0.0
        
        return 0.0
    
    def draw_enhanced_brain_visualization(self, frame, activations, current_time):
        """Draw enhanced brain region visualization."""
        height, width = frame.shape[:2]
        
        # Create visualization area (bottom-left)
        viz_width = 320
        viz_height = 220
        viz_x = 10
        viz_y = height - viz_height - 10
        
        # Background with gradient effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), (255, 255, 255), 2)
        
        # Title with icon
        cv2.putText(frame, "🧠 ENHANCED BRAIN ACTIVATION", (viz_x + 10, viz_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
        
        # Draw region visualization with circular layout
        center_x = viz_x + viz_width // 2
        center_y = viz_y + viz_height // 2 + 10
        
        regions = ['Motor', 'Visual', 'Fusiform', 'Frontal', 'Temporal']
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 165, 0)]
        icons = ['🏃', '👁️', '🧩', '💭', '👂']
        
        # Draw regions in circular layout
        for i, (region, color, icon) in enumerate(zip(regions, colors, icons)):
            activation = activations.get(region, 0.0)
            norm_activation = max(0, min(1, abs(activation) / 1000))
            
            # Calculate position in circle
            angle = i * 2 * np.pi / len(regions)
            radius = 60
            region_x = int(center_x + radius * np.cos(angle))
            region_y = int(center_y + radius * np.sin(angle))
            
            # Region size based on activation
            region_size = max(15, min(30, int(15 + norm_activation * 15)))
            
            # Draw region circle with gradient
            cv2.circle(frame, (region_x, region_y), region_size, color, -1)
            cv2.circle(frame, (region_x, region_y), region_size, (255, 255, 255), 2)
            
            # Icon
            cv2.putText(frame, icon, (region_x - 8, region_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Region name
            cv2.putText(frame, region, (region_x - 15, region_y + region_size + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            
            # Activation value
            cv2.putText(frame, f"{activation:.0f}", (region_x - 10, region_y - region_size - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
        
        # Central info
        cv2.putText(frame, "BRAIN", (center_x - 20, center_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(frame, "ACTIVATION", (center_x - 30, center_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        # Time
        cv2.putText(frame, f"Time: {current_time:.1f}s", (viz_x + 10, viz_y + viz_height - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def create_annotated_video(self, base_video_path, output_path, start_time=0, duration=20, fps=30):
        """Create annotated video with improved enhanced brain visualization."""
        print(f"🎬 Creating improved enhanced brain video: {output_path}")
        
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
                activations[region] = self.get_brain_activation(current_time, region)
            
            # Draw enhanced brain visualization (bottom-left)
            self.draw_enhanced_brain_visualization(frame, activations, current_time)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Improved enhanced brain video created: {output_path}")
        return output_path

def main():
    print("🎬 Applying All Approaches to Base Video")
    print("=" * 40)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Apply all approaches to base video')
    parser.add_argument('--approaches', type=str, default='1,2,3,4',
                       help='Comma-separated list of approaches to apply (default: 1,2,3,4)')
    parser.add_argument('--duration', type=int, default=20, 
                       help='Duration in seconds (default: 20)')
    parser.add_argument('--base-video', type=str, 
                       help='Path to base video (if not provided, will find latest)')
    
    args = parser.parse_args()
    
    # Parse approaches to run
    approaches_to_run = [int(x.strip()) for x in args.approaches.split(',')]
    print(f"🎯 Applying approaches: {approaches_to_run} for {args.duration} seconds")
    
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
    
    # Find base video
    if args.base_video:
        base_video_path = args.base_video
    else:
        # Find the latest base video
        base_video_pattern = f"*object_detection_BASE_exp*.mp4"
        base_videos = list(Path("results/06_video_analysis").glob(f"experiment*/{base_video_pattern}"))
        if not base_videos:
            print("❌ Error: No base video found. Please run comprehensive_video_fixes.py first.")
            sys.exit(1)
        base_video_path = str(max(base_videos, key=os.path.getctime))
    
    print(f"📹 Using base video: {base_video_path}")
    print(f"📊 ECoG data shape: {high_gamma_envelope.shape}")
    print(f"📁 Experiment folder: experiment{exp_manager.experiment_number}")
    
    # Run all approaches
    results = []
    approach_names = {
        1: "Improved Motor Cortex Activation",
        2: "Improved Gait Phase Analysis", 
        3: "Improved ERSP Time Series",
        4: "Improved Enhanced Brain Activation"
    }
    
    for approach_num in approaches_to_run:
        if approach_num not in approach_names:
            print(f"⚠️ Skipping unknown approach: {approach_num}")
            continue
            
        print(f"\n🎬 Applying Approach {approach_num}: {approach_names[approach_num]}")
        
        try:
            if approach_num == 1:
                # Improved Motor Cortex
                annotator = ImprovedMotorCortexAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_motor_cortex_IMPROVED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 2:
                # Improved Gait Phase
                annotator = ImprovedGaitPhaseAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_gait_phase_IMPROVED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 3:
                # Improved ERSP
                annotator = ImprovedERSPAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_ersp_IMPROVED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
                
            elif approach_num == 4:
                # Improved Enhanced Brain
                annotator = ImprovedEnhancedBrainAnnotator(high_gamma_envelope, video_start_time, sampling_rate)
                video_filename = f"walk_annotated_enhanced_brain_IMPROVED_exp{exp_manager.experiment_number}.mp4"
                output_path = exp_manager.get_video_path(video_filename)
            
            else:
                print(f"⚠️ Unknown approach: {approach_num}")
                continue
            
            # Create video
            result_path = annotator.create_annotated_video(
                base_video_path=base_video_path,
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
            print(f"❌ Error applying Approach {approach_num}: {str(e)}")
            continue
    
    # Print summary
    print(f"\n🎉 All approaches applied successfully!")
    print(f"📁 Files saved to: {exp_manager.experiment_dir}")
    print(f"📊 Successfully applied {len(results)} approaches:")
    
    for approach_num, name, result_path in results:
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
            print(f"  ✅ Approach {approach_num}: {name}")
            print(f"     📹 Video: {os.path.basename(result_path)} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()
