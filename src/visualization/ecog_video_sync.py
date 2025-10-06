#!/usr/bin/env python3
"""
ECoG Video Synchronization Module
Handles synchronization between ECoG data and video frames for real-time annotation.
"""

import numpy as np
import cv2
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ECoGVideoSynchronizer:
    """Synchronizes ECoG data with video frames for real-time annotation."""
    
    def __init__(self, video_path: str, ecog_data: np.ndarray, 
                 sampling_rate: float, video_start_time: float):
        """
        Initialize ECoG video synchronizer.
        
        Args:
            video_path: Path to video file
            ecog_data: ECoG data array (channels x time)
            sampling_rate: ECoG sampling rate in Hz
            video_start_time: Time when video starts in ECoG data (seconds)
        """
        self.video_path = video_path
        self.ecog_data = ecog_data
        self.sampling_rate = sampling_rate
        self.video_start_time = video_start_time
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
        print(f"📹 Video loaded: {self.fps:.2f} FPS, {self.frame_count} frames, {self.duration:.2f}s")
        
    def get_frame_at_time(self, video_time: float) -> Optional[np.ndarray]:
        """Get video frame at specific time."""
        if video_time < 0 or video_time > self.duration:
            return None
            
        frame_number = int(video_time * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            return frame
        return None
    
    def get_ecog_data_at_time(self, video_time: float, window_size: float = 0.5) -> np.ndarray:
        """Get ECoG data at specific video time."""
        # Convert video time to ECoG time
        ecog_time = video_time + self.video_start_time
        
        # Convert to sample indices
        center_sample = int(ecog_time * self.sampling_rate)
        window_samples = int(window_size * self.sampling_rate)
        
        start_idx = max(0, center_sample - window_samples // 2)
        end_idx = min(self.ecog_data.shape[1], center_sample + window_samples // 2)
        
        return self.ecog_data[:, start_idx:end_idx]
    
    def get_synchronized_data(self, video_time: float, window_size: float = 0.5) -> Dict:
        """Get synchronized video frame and ECoG data."""
        frame = self.get_frame_at_time(video_time)
        ecog_data = self.get_ecog_data_at_time(video_time, window_size)
        
        return {
            'frame': frame,
            'ecog_data': ecog_data,
            'video_time': video_time,
            'ecog_time': video_time + self.video_start_time,
            'frame_number': int(video_time * self.fps) if frame is not None else None
        }
    
    def create_time_alignment_map(self, start_time: float = 0, duration: float = 20) -> Dict:
        """Create time alignment map for fast lookup."""
        time_points = np.linspace(start_time, start_time + duration, int(duration * self.fps))
        
        alignment_map = {}
        for t in time_points:
            alignment_map[t] = self.get_synchronized_data(t)
        
        return alignment_map
    
    def close(self):
        """Close video capture."""
        if self.cap:
            self.cap.release()

class BrainRegionMapper:
    """Maps ECoG channels to brain regions."""
    
    def __init__(self):
        """Initialize brain region mapping."""
        self.brain_regions = {
            'Occipital': list(range(131, 161)),  # Primary visual cortex
            'Temporal': list(range(101, 131)),   # Visual association
            'Parietal': list(range(61, 101)),    # Spatial processing
            'Central': list(range(31, 61)),      # Sensorimotor
            'Frontal': list(range(1, 31))        # Executive control
        }
        
        # Top channels from analysis
        self.top_channels = [131, 107, 113, 114, 71, 108, 132, 152, 127, 105]
        
        # Category-specific channels
        self.category_channels = {
            'digit': [70, 77, 104, 105, 126],
            'kanji': [70, 77, 104, 105, 144],
            'face': [70, 104, 126, 144, 147],
            'body': [70, 76, 105, 126, 144],
            'object': [70, 77, 104, 105, 118],
            'hiragana': [70, 104, 105, 126, 144],
            'line': [70, 104, 105, 126, 144]
        }
    
    def get_region_activation(self, ecog_data: np.ndarray, region: str) -> float:
        """Get activation level for specific brain region."""
        if region not in self.brain_regions:
            return 0.0
        
        channels = [ch for ch in self.brain_regions[region] if ch <= ecog_data.shape[0]]
        if not channels:
            return 0.0
        
        region_data = ecog_data[channels, :]
        return np.mean(region_data)
    
    def get_category_response(self, ecog_data: np.ndarray, category: str) -> float:
        """Get response level for specific category."""
        if category not in self.category_channels:
            return 0.0
        
        channels = [ch for ch in self.category_channels[category] if ch <= ecog_data.shape[0]]
        if not channels:
            return 0.0
        
        category_data = ecog_data[channels, :]
        return np.mean(category_data)
    
    def get_top_channel_activity(self, ecog_data: np.ndarray) -> Dict[int, float]:
        """Get activity for top channels."""
        activity = {}
        for ch in self.top_channels:
            if ch <= ecog_data.shape[0]:
                activity[ch] = np.mean(ecog_data[ch-1, :])  # Convert to 0-based indexing
            else:
                activity[ch] = 0.0
        return activity

class AnnotationOverlay:
    """Creates annotation overlays on video frames."""
    
    def __init__(self, brain_mapper: BrainRegionMapper):
        """Initialize annotation overlay."""
        self.brain_mapper = brain_mapper
        
        # Color scheme for different categories
        self.category_colors = {
            'digit': (0, 255, 0),      # Green
            'kanji': (255, 0, 0),      # Blue
            'face': (0, 0, 255),       # Red
            'body': (255, 255, 0),     # Cyan
            'object': (255, 0, 255),   # Magenta
            'hiragana': (0, 255, 255), # Yellow
            'line': (128, 0, 128),     # Purple
            'baseline': (128, 128, 128) # Gray
        }
        
        # Color scheme for brain regions
        self.region_colors = {
            'Occipital': (255, 100, 100),  # Light red
            'Temporal': (100, 255, 100),   # Light green
            'Parietal': (100, 100, 255),   # Light blue
            'Central': (255, 255, 100),    # Light yellow
            'Frontal': (255, 100, 255)     # Light magenta
        }
    
    def add_brain_region_overlay(self, frame: np.ndarray, ecog_data: np.ndarray) -> np.ndarray:
        """Add brain region activation overlay to frame."""
        if frame is None:
            return None
        
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Create region activation display
        region_activations = {}
        for region in self.brain_mapper.brain_regions.keys():
            region_activations[region] = self.brain_mapper.get_region_activation(ecog_data, region)
        
        # Normalize activations
        max_activation = max(region_activations.values()) if region_activations.values() else 1.0
        if max_activation > 0:
            for region in region_activations:
                region_activations[region] /= max_activation
        
        # Draw region indicators
        y_offset = 50
        for i, (region, activation) in enumerate(region_activations.items()):
            color = self.region_colors.get(region, (128, 128, 128))
            intensity = int(activation * 255)
            color = tuple(int(c * activation) for c in color)
            
            # Draw region name and activation bar
            cv2.putText(overlay, f"{region}: {activation:.2f}", 
                       (10, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw activation bar
            bar_width = int(activation * 200)
            cv2.rectangle(overlay, (200, y_offset + i * 30 - 10), 
                         (200 + bar_width, y_offset + i * 30 + 10), color, -1)
        
        return overlay
    
    def add_category_overlay(self, frame: np.ndarray, ecog_data: np.ndarray, 
                           predicted_category: str = None) -> np.ndarray:
        """Add category prediction overlay to frame."""
        if frame is None:
            return None
        
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Get category responses
        category_responses = {}
        for category in self.brain_mapper.category_channels.keys():
            category_responses[category] = self.brain_mapper.get_category_response(ecog_data, category)
        
        # Find best category
        if not category_responses or max(category_responses.values()) < 40.0:
            best_category = 'baseline'
            confidence = 0.1
        else:
            best_category = max(category_responses, key=category_responses.get)
            confidence = min(0.9, max(category_responses.values()) / 50.0)
        
        # Use predicted category if provided
        if predicted_category:
            best_category = predicted_category
        
        # Draw category prediction
        color = self.category_colors.get(best_category, (128, 128, 128))
        cv2.putText(overlay, f"Predicted: {best_category.upper()}", 
                   (width - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(overlay, f"Confidence: {confidence:.2f}", 
                   (width - 300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw category response bars
        y_start = height - 200
        for i, (category, response) in enumerate(category_responses.items()):
            color = self.category_colors.get(category, (128, 128, 128))
            bar_height = int((response / 50.0) * 100)  # Normalize to 50
            bar_height = min(100, max(10, bar_height))
            
            cv2.rectangle(overlay, (i * 80, y_start), 
                         (i * 80 + 60, y_start - bar_height), color, -1)
            cv2.putText(overlay, category[:4], (i * 80, y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return overlay
    
    def add_timeline_overlay(self, frame: np.ndarray, video_time: float, 
                           total_duration: float) -> np.ndarray:
        """Add timeline overlay to frame."""
        if frame is None:
            return None
        
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw timeline
        timeline_y = height - 50
        timeline_width = width - 100
        timeline_x = 50
        
        # Background
        cv2.rectangle(overlay, (timeline_x, timeline_y - 10), 
                     (timeline_x + timeline_width, timeline_y + 10), (64, 64, 64), -1)
        
        # Current time indicator
        current_pos = int((video_time / total_duration) * timeline_width)
        cv2.rectangle(overlay, (timeline_x + current_pos - 2, timeline_y - 15), 
                     (timeline_x + current_pos + 2, timeline_y + 15), (0, 255, 0), -1)
        
        # Time labels
        cv2.putText(overlay, f"{video_time:.1f}s", (timeline_x + current_pos - 20, timeline_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"0s", (timeline_x, timeline_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"{total_duration:.1f}s", (timeline_x + timeline_width - 30, timeline_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay

def create_synchronized_annotation(video_path: str, ecog_data: np.ndarray, 
                                 sampling_rate: float, video_start_time: float,
                                 output_path: str, start_time: float = 0, 
                                 duration: float = 20) -> bool:
    """
    Create synchronized annotation video.
    
    Args:
        video_path: Path to input video
        ecog_data: ECoG data array
        sampling_rate: ECoG sampling rate
        video_start_time: Video start time in ECoG data
        output_path: Path for output video
        start_time: Start time for annotation (seconds)
        duration: Duration of annotation (seconds)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize components
        synchronizer = ECoGVideoSynchronizer(video_path, ecog_data, sampling_rate, video_start_time)
        brain_mapper = BrainRegionMapper()
        overlay = AnnotationOverlay(brain_mapper)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, synchronizer.fps, 
                             (int(synchronizer.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(synchronizer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        # Process frames
        current_time = start_time
        end_time = start_time + duration
        
        print(f"🎬 Creating synchronized annotation video...")
        print(f"📹 Duration: {duration}s, FPS: {synchronizer.fps:.2f}")
        
        while current_time < end_time:
            # Get synchronized data
            sync_data = synchronizer.get_synchronized_data(current_time)
            
            if sync_data['frame'] is not None:
                # Add overlays
                annotated_frame = sync_data['frame'].copy()
                annotated_frame = overlay.add_brain_region_overlay(annotated_frame, sync_data['ecog_data'])
                annotated_frame = overlay.add_category_overlay(annotated_frame, sync_data['ecog_data'])
                annotated_frame = overlay.add_timeline_overlay(annotated_frame, current_time, duration)
                
                # Write frame
                out.write(annotated_frame)
            
            # Move to next frame
            current_time += 1.0 / synchronizer.fps
        
        # Cleanup
        out.release()
        synchronizer.close()
        
        print(f"✅ Synchronized annotation video created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating synchronized annotation: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    print("🎥 ECoG Video Synchronization Module")
    print("This module provides synchronization between ECoG data and video frames.")
    print("Use the functions in your notebook or other scripts.")
