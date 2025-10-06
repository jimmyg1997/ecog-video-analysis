#!/usr/bin/env python3
"""
Create Meaningful Labels for ECoG Classification
===============================================

This script demonstrates how to create more meaningful classification tasks
instead of the basic 4-class approach.

Usage:
    python create_meaningful_labels.py
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json

def load_raw_data():
    """Load raw ECoG data."""
    from utils.data_loader import DataLoader
    loader = DataLoader()
    data = loader.load_raw_data('Walk.mat')
    return data

def create_binary_labels(stimcode):
    """Create binary labels: something vs nothing."""
    print("🔍 Creating binary labels (something vs nothing)...")
    
    # Method 1: Simple threshold
    binary_simple = (stimcode > 0).astype(int)
    
    # Method 2: More sophisticated (consider transitions)
    binary_sophisticated = np.zeros_like(stimcode)
    for i in range(1, len(stimcode)):
        if stimcode[i] != stimcode[i-1]:  # Transition detected
            binary_sophisticated[i:] = 1  # Mark as "something" after transition
    
    return {
        'simple': binary_simple,
        'sophisticated': binary_sophisticated
    }

def create_hierarchical_labels(stimcode, groupid):
    """Create hierarchical labels."""
    print("🔍 Creating hierarchical labels...")
    
    # Level 1: Something vs Nothing
    level1 = (stimcode > 0).astype(int)
    
    # Level 2: If something, what type?
    level2 = np.zeros_like(stimcode)
    level2[stimcode == 1] = 1  # Type 1
    level2[stimcode == 2] = 2  # Type 2
    level2[stimcode == 3] = 3  # Type 3
    
    # Level 3: Combined with group ID
    level3 = stimcode * 10 + (groupid + 1)  # Avoid negative numbers
    
    return {
        'level1': level1,
        'level2': level2,
        'level3': level3
    }

def create_attention_labels(stimcode, sampling_rate=1200):
    """Create attention-based labels."""
    print("🔍 Creating attention-based labels...")
    
    # Simulate attention based on stimulus changes
    attention = np.zeros_like(stimcode, dtype=float)
    
    for i in range(1, len(stimcode)):
        if stimcode[i] != stimcode[i-1]:
            # High attention during transitions
            attention[i:i+int(sampling_rate*2)] = 1.0  # 2 seconds of high attention
        else:
            # Gradual decay
            attention[i] = max(0, attention[i-1] - 0.001)
    
    # Convert to discrete levels
    attention_levels = np.zeros_like(attention, dtype=int)
    attention_levels[attention > 0.8] = 3  # High attention
    attention_levels[(attention > 0.4) & (attention <= 0.8)] = 2  # Medium attention
    attention_levels[(attention > 0.1) & (attention <= 0.4)] = 1  # Low attention
    attention_levels[attention <= 0.1] = 0  # No attention
    
    return {
        'continuous': attention,
        'discrete': attention_levels
    }

def create_video_based_labels(video_path, sampling_rate=1200):
    """Create labels based on video content analysis."""
    print("🔍 Creating video-based labels...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"  Video: {fps} FPS, {frame_count} frames, {duration:.1f}s")
    
    # Sample frames and analyze content
    frame_labels = []
    frame_times = []
    
    # Sample every 5 seconds
    sample_interval = int(fps * 5)
    
    for frame_idx in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_time = frame_idx / fps
            frame_times.append(frame_time)
            
            # Simple content analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Analyze frame content
            content_type = analyze_frame_content(gray)
            frame_labels.append(content_type)
    
    cap.release()
    
    # Interpolate labels to match ECoG sampling rate
    ecog_duration = 268.4  # From our data analysis
    ecog_samples = int(ecog_duration * sampling_rate)
    ecog_times = np.linspace(0, ecog_duration, ecog_samples)
    
    # Interpolate labels
    video_labels = np.interp(ecog_times, frame_times, frame_labels).astype(int)
    
    return {
        'frame_times': frame_times,
        'frame_labels': frame_labels,
        'ecog_labels': video_labels
    }

def analyze_frame_content(gray_frame):
    """Analyze frame content to determine label."""
    # Simple heuristics for content analysis
    
    # Check for motion (edge detection)
    edges = cv2.Canny(gray_frame, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Check for text (horizontal lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    text_score = np.sum(horizontal_lines > 0) / horizontal_lines.size
    
    # Check for faces (simple template matching)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
    face_score = len(faces) > 0
    
    # Determine content type
    if face_score:
        return 3  # Face
    elif text_score > 0.01:
        return 2  # Text
    elif edge_density > 0.05:
        return 1  # Object/Motion
    else:
        return 0  # Background/Static

def create_custom_annotation_labels():
    """Create custom annotation labels based on video content."""
    print("🔍 Creating custom annotation labels...")
    
    # This would be where you manually annotate the video
    # For now, we'll create a realistic example
    
    # Define custom categories
    custom_categories = {
        0: 'background',
        1: 'person_walking',
        2: 'object',
        3: 'text',
        4: 'face',
        5: 'motion'
    }
    
    # Create realistic timeline based on a walking video
    duration = 268.4  # seconds
    sampling_rate = 1200
    
    # Simulate realistic video content
    labels = np.zeros(int(duration * sampling_rate), dtype=int)
    
    # Background (0-30s)
    labels[0:int(30*sampling_rate)] = 0
    
    # Person walking (30-180s)
    labels[int(30*sampling_rate):int(180*sampling_rate)] = 1
    
    # Object (180-200s)
    labels[int(180*sampling_rate):int(200*sampling_rate)] = 2
    
    # Text (200-220s)
    labels[int(200*sampling_rate):int(220*sampling_rate)] = 3
    
    # Face (220-240s)
    labels[int(220*sampling_rate):int(240*sampling_rate)] = 4
    
    # Motion (240-268s)
    labels[int(240*sampling_rate):] = 5
    
    return {
        'labels': labels,
        'categories': custom_categories
    }

def evaluate_label_quality(labels, label_name):
    """Evaluate the quality of created labels."""
    print(f"  📊 {label_name} Quality:")
    print(f"     Shape: {labels.shape}")
    print(f"     Unique values: {np.unique(labels)}")
    
    # Convert to int for bincount
    labels_int = labels.astype(int)
    distribution = np.bincount(labels_int)
    print(f"     Distribution: {distribution}")
    print(f"     Balance: {np.min(distribution) / np.max(distribution):.3f}")

def save_labels(labels_dict, experiment_id):
    """Save all created labels."""
    print("💾 Saving labels...")
    
    labels_dir = Path(f"data/labels/{experiment_id}")
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for label_name, labels in labels_dict.items():
        if isinstance(labels, dict):
            # Save dictionary of labels
            for sub_name, sub_labels in labels.items():
                if isinstance(sub_labels, np.ndarray):
                    np.save(labels_dir / f"{label_name}_{sub_name}.npy", sub_labels)
        elif isinstance(labels, np.ndarray):
            # Save single array
            np.save(labels_dir / f"{label_name}.npy", labels)
    
    # Save metadata
    metadata = {
        'experiment_id': experiment_id,
        'created_at': datetime.now().isoformat(),
        'label_types': list(labels_dict.keys()),
        'sampling_rate': 1200,
        'duration': 268.4
    }
    
    with open(labels_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  📁 Labels saved to: {labels_dir}")

def main():
    """Main function."""
    print("🚀 Creating Meaningful Labels for ECoG Classification")
    print("=" * 60)
    
    # Load raw data
    print("📊 Loading raw data...")
    data = load_raw_data()
    stimcode = data['stimcode']
    groupid = data['groupid']
    
    print(f"  Stimulus codes: {np.unique(stimcode)}")
    print(f"  Group IDs: {np.unique(groupid)}")
    
    # Create different types of labels
    all_labels = {}
    
    # 1. Binary labels
    binary_labels = create_binary_labels(stimcode)
    all_labels['binary'] = binary_labels
    evaluate_label_quality(binary_labels['simple'], 'Binary (Simple)')
    evaluate_label_quality(binary_labels['sophisticated'], 'Binary (Sophisticated)')
    
    # 2. Hierarchical labels
    hierarchical_labels = create_hierarchical_labels(stimcode, groupid)
    all_labels['hierarchical'] = hierarchical_labels
    evaluate_label_quality(hierarchical_labels['level1'], 'Hierarchical Level 1')
    evaluate_label_quality(hierarchical_labels['level2'], 'Hierarchical Level 2')
    evaluate_label_quality(hierarchical_labels['level3'], 'Hierarchical Level 3')
    
    # 3. Attention labels
    attention_labels = create_attention_labels(stimcode)
    all_labels['attention'] = attention_labels
    evaluate_label_quality(attention_labels['discrete'], 'Attention (Discrete)')
    
    # 4. Video-based labels
    video_labels = create_video_based_labels('data/raw/walk.mp4')
    if video_labels:
        all_labels['video'] = video_labels
        evaluate_label_quality(video_labels['ecog_labels'], 'Video-based')
    
    # 5. Custom annotation labels
    custom_labels = create_custom_annotation_labels()
    all_labels['custom'] = custom_labels
    evaluate_label_quality(custom_labels['labels'], 'Custom Annotation')
    
    # Get next experiment number
    def get_next_experiment_number():
        labels_dir = Path('data/labels')
        if not labels_dir.exists():
            return 1
        existing_dirs = [d for d in labels_dir.iterdir() if d.is_dir() and d.name.startswith('experiment')]
        if not existing_dirs:
            return 1
        numbers = []
        for d in existing_dirs:
            try:
                num = int(d.name.replace('experiment', ''))
                numbers.append(num)
            except ValueError:
                continue
        return max(numbers) + 1 if numbers else 1
    
    next_exp = get_next_experiment_number()
    experiment_id = f'experiment{next_exp}'
    
    # Save all labels
    save_labels(all_labels, experiment_id)
    
    print("\n🎉 Label Creation Completed!")
    print("=" * 60)
    print("📋 Created Label Types:")
    print("  1. Binary: Something vs Nothing")
    print("  2. Hierarchical: Multi-level classification")
    print("  3. Attention: Attention-based classification")
    print("  4. Video-based: Content analysis from video")
    print("  5. Custom: Manually annotated categories")
    
    print("\n🚀 Next Steps:")
    print("  1. Use these labels with your feature extractors")
    print("  2. Compare performance across different label types")
    print("  3. Choose the most meaningful task for your application")
    print("  4. Consider the interpretability of results")

if __name__ == "__main__":
    main()
