#!/usr/bin/env python3
"""
Create Real Annotation Labels from Video Annotations
===================================================

This script creates proper 7-class labels from the real video annotations
instead of using synthetic labels.

Usage:
    python create_real_annotation_labels.py
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_video_annotations():
    """Load the real video annotations."""
    annotation_file = Path('results/annotations/video_annotation_data.json')
    
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    return data

def create_category_mapping():
    """Create mapping from category names to numeric labels."""
    return {
        'digit': 0,
        'kanji': 1, 
        'face': 2,
        'body': 3,
        'object': 4,
        'hiragana': 5,
        'line': 6
    }

def create_ecog_labels_from_annotations(annotations, sampling_rate=1200, duration=268.4):
    """Create ECoG labels from video annotations."""
    print("🔍 Creating ECoG labels from video annotations...")
    
    # Create category mapping
    category_map = create_category_mapping()
    
    # Initialize labels array (background = -1, categories = 0-6)
    n_samples = int(duration * sampling_rate)
    labels = np.full(n_samples, -1, dtype=int)  # -1 = background/no stimulus
    
    # Process each annotation
    for annotation in annotations['annotations']:
        start_time = annotation['time_start']
        end_time = annotation['time_end']
        category = annotation['category']
        
        # Convert time to sample indices
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)
        
        # Ensure indices are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)
        
        # Set labels for this time period
        if category in category_map:
            category_id = category_map[category]
            labels[start_sample:end_sample] = category_id
            print(f"  📊 {category} ({category_id}): {start_time:.1f}s-{end_time:.1f}s -> samples {start_sample}-{end_sample}")
    
    return labels, category_map

def create_trial_based_labels(annotations, n_trials=252):
    """Create trial-based labels for the 252 trials."""
    print("🔍 Creating trial-based labels...")
    
    category_map = create_category_mapping()
    
    # Get video info
    video_start = annotations['video_info']['video_start_time']
    video_end = annotations['video_info']['video_end_time']
    video_duration = video_end - video_start
    
    # Calculate trial duration
    trial_duration = video_duration / n_trials
    
    # Initialize trial labels
    trial_labels = np.full(n_trials, -1, dtype=int)  # -1 = background
    
    # Process each annotation
    for annotation in annotations['annotations']:
        start_time = annotation['time_start'] - video_start  # Adjust for video start
        end_time = annotation['time_end'] - video_start
        category = annotation['category']
        
        # Find which trials this annotation covers
        start_trial = int(start_time / trial_duration)
        end_trial = int(end_time / trial_duration)
        
        # Ensure indices are within bounds
        start_trial = max(0, start_trial)
        end_trial = min(n_trials, end_trial)
        
        # Set labels for these trials
        if category in category_map:
            category_id = category_map[category]
            trial_labels[start_trial:end_trial] = category_id
            print(f"  📊 {category} ({category_id}): trials {start_trial}-{end_trial}")
    
    return trial_labels, category_map

def create_color_aware_labels(annotations, n_trials=252):
    """Create color-aware labels (14 classes: 7 categories × 2 colors)."""
    print("🔍 Creating color-aware labels...")
    
    category_map = create_category_mapping()
    
    # Get video info
    video_start = annotations['video_info']['video_start_time']
    video_end = annotations['video_info']['video_end_time']
    video_duration = video_end - video_start
    
    # Calculate trial duration
    trial_duration = video_duration / n_trials
    
    # Initialize trial labels (14 classes: 0-6 gray, 7-13 color)
    trial_labels = np.full(n_trials, -1, dtype=int)  # -1 = background
    
    # Process each annotation
    for annotation in annotations['annotations']:
        start_time = annotation['time_start'] - video_start
        end_time = annotation['time_end'] - video_start
        category = annotation['category']
        color = annotation['color']
        
        # Find which trials this annotation covers
        start_trial = int(start_time / trial_duration)
        end_trial = int(end_time / trial_duration)
        
        # Ensure indices are within bounds
        start_trial = max(0, start_trial)
        end_trial = min(n_trials, end_trial)
        
        # Set labels for these trials
        if category in category_map:
            category_id = category_map[category]
            
            # Add color offset (0-6 for gray, 7-13 for color)
            if 'color' in color.lower() and 'gray' not in color.lower():
                color_offset = 7  # Color images
            else:
                color_offset = 0  # Gray images
            
            final_label = category_id + color_offset
            trial_labels[start_trial:end_trial] = final_label
            
            color_type = "color" if color_offset > 0 else "gray"
            print(f"  📊 {category} {color_type} ({final_label}): trials {start_trial}-{end_trial}")
    
    return trial_labels, category_map

def evaluate_label_quality(labels, label_name, category_map=None):
    """Evaluate the quality of created labels."""
    print(f"  📊 {label_name} Quality:")
    print(f"     Shape: {labels.shape}")
    print(f"     Unique values: {np.unique(labels)}")
    
    # Convert to int for bincount
    labels_int = labels.astype(int)
    unique_labels = np.unique(labels_int)
    
    # Count distribution
    distribution = np.bincount(labels_int + 1)  # +1 to handle -1 values
    print(f"     Distribution: {distribution}")
    
    # Calculate balance (excluding background)
    non_background = labels_int[labels_int >= 0]
    if len(non_background) > 0:
        non_bg_distribution = np.bincount(non_background)
        if len(non_bg_distribution) > 1:
            balance = np.min(non_bg_distribution) / np.max(non_bg_distribution)
            print(f"     Balance (non-background): {balance:.3f}")
    
    # Show category breakdown
    if category_map:
        print(f"     Category breakdown:")
        for category, cat_id in category_map.items():
            count = np.sum(labels_int == cat_id)
            print(f"       {category}: {count} samples")

def save_real_labels(labels_dict, experiment_id):
    """Save the real annotation labels."""
    print("💾 Saving real annotation labels...")
    
    labels_dir = Path(f"data/labels/{experiment_id}")
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for label_name, label_data in labels_dict.items():
        if isinstance(label_data, dict):
            # Save dictionary of labels
            for sub_name, sub_labels in label_data.items():
                if isinstance(sub_labels, np.ndarray):
                    np.save(labels_dir / f"{label_name}_{sub_name}.npy", sub_labels)
        elif isinstance(label_data, np.ndarray):
            # Save single array
            np.save(labels_dir / f"{label_name}.npy", label_data)
    
    # Save metadata
    metadata = {
        'experiment_id': experiment_id,
        'created_at': datetime.now().isoformat(),
        'label_types': list(labels_dict.keys()),
        'sampling_rate': 1200,
        'duration': 268.4,
        'source': 'real_video_annotations',
        'categories': create_category_mapping()
    }
    
    with open(labels_dir / 'real_annotation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  📁 Labels saved to: {labels_dir}")

def main():
    """Main function."""
    print("🚀 Creating Real Annotation Labels from Video Annotations")
    print("=" * 60)
    
    # Load video annotations
    print("📊 Loading video annotations...")
    annotations = load_video_annotations()
    
    print(f"  Video duration: {annotations['video_info']['duration']}s")
    print(f"  Number of annotations: {len(annotations['annotations'])}")
    
    # Show annotation summary
    categories = [ann['category'] for ann in annotations['annotations']]
    unique_categories = list(set(categories))
    print(f"  Categories found: {unique_categories}")
    
    # Create different types of labels
    all_labels = {}
    
    # 1. Continuous ECoG labels (full sampling rate)
    print("\n🔍 Creating continuous ECoG labels...")
    ecog_labels, category_map = create_ecog_labels_from_annotations(annotations)
    all_labels['ecog_continuous'] = ecog_labels
    evaluate_label_quality(ecog_labels, 'ECoG Continuous', category_map)
    
    # 2. Trial-based labels (252 trials)
    print("\n🔍 Creating trial-based labels...")
    trial_labels, _ = create_trial_based_labels(annotations)
    all_labels['trial_based'] = trial_labels
    evaluate_label_quality(trial_labels, 'Trial-based', category_map)
    
    # 3. Color-aware labels (14 classes)
    print("\n🔍 Creating color-aware labels...")
    color_labels, _ = create_color_aware_labels(annotations)
    all_labels['color_aware'] = color_labels
    evaluate_label_quality(color_labels, 'Color-aware', category_map)
    
    # 4. Binary labels (something vs nothing)
    print("\n🔍 Creating binary labels...")
    binary_labels = (trial_labels >= 0).astype(int)
    all_labels['binary'] = binary_labels
    evaluate_label_quality(binary_labels, 'Binary', None)
    
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
    save_real_labels(all_labels, experiment_id)
    
    print("\n🎉 Real Annotation Labels Created!")
    print("=" * 60)
    print("📋 Created Label Types:")
    print("  1. ECoG Continuous: Full sampling rate labels")
    print("  2. Trial-based: 252 trial labels (7 classes)")
    print("  3. Color-aware: 14 class labels (7 categories × 2 colors)")
    print("  4. Binary: Something vs nothing")
    
    print(f"\n📊 Label Statistics:")
    print(f"  Trial-based: {len(np.unique(trial_labels))} classes")
    print(f"  Color-aware: {len(np.unique(color_labels))} classes")
    print(f"  Binary: {len(np.unique(binary_labels))} classes")
    
    print(f"\n🚀 Next Steps:")
    print("  1. Use these labels with existing features")
    print("  2. Compare with paper results (72.9% for 7-class)")
    print("  3. Test real-time classification")
    print("  4. Create confusion matrices for analysis")

if __name__ == "__main__":
    main()
