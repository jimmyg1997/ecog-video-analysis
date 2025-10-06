#!/usr/bin/env python3
"""
Diagnostic script to understand the stimcode labeling issue in preprocessing.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_raw_data():
    """Load raw data using the project's data loader."""
    try:
        from src.utils.data_loader import DataLoader
        loader = DataLoader()
        raw_data = loader.load_raw_data('Walk.mat')
        return raw_data
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return None

def analyze_stimcode_issue():
    """Analyze the stimcode labeling issue."""
    print("🔍 Diagnosing Stimcode Labeling Issue")
    print("=" * 50)
    
    # Load raw data
    print("📊 Loading raw data...")
    raw_data = load_raw_data()
    if raw_data is None:
        print("❌ Could not load raw data")
        return
    
    stimcode = raw_data['stimcode']
    photodiode = raw_data['photodiode']
    groupid = raw_data['groupid']
    sampling_rate = raw_data['sampling_rate']
    
    print(f"✅ Loaded data:")
    print(f"   Sampling rate: {sampling_rate} Hz")
    print(f"   Duration: {len(stimcode) / sampling_rate:.1f} seconds")
    print(f"   Stimcode shape: {stimcode.shape}")
    print(f"   Photodiode shape: {photodiode.shape}")
    
    # Analyze stimcode
    print(f"\n🔍 Stimcode Analysis:")
    unique_stimcodes = np.unique(stimcode)
    print(f"   Unique values: {unique_stimcodes}")
    print(f"   Value counts: {np.bincount(stimcode.astype(int))}")
    
    # Find stimcode changes
    stimcode_diff = np.abs(np.diff(stimcode))
    stimcode_changes = np.where(stimcode_diff > 0)[0]
    print(f"   Number of stimcode changes: {len(stimcode_changes)}")
    print(f"   Change times (first 10): {stimcode_changes[:10] / sampling_rate}")
    
    # Analyze photodiode
    print(f"\n🔍 Photodiode Analysis:")
    photodiode_diff = np.abs(np.diff(photodiode))
    threshold = np.std(photodiode_diff) * 2
    photodiode_changes = np.where(photodiode_diff > threshold)[0]
    print(f"   Number of photodiode changes: {len(photodiode_changes)}")
    print(f"   Change times (first 10): {photodiode_changes[:10] / sampling_rate}")
    
    # Check preprocessed data
    print(f"\n🔍 Preprocessed Data Analysis:")
    exp_path = Path('data/preprocessed/experiment8')
    if exp_path.exists():
        try:
            preprocessed_stimcode = np.load(exp_path / 'stimcode.npy')
            trial_onsets = np.load(exp_path / 'trial_onsets.npy')
            
            print(f"   Preprocessed stimcode shape: {preprocessed_stimcode.shape}")
            print(f"   Preprocessed stimcode unique values: {np.unique(preprocessed_stimcode)}")
            print(f"   Trial onsets shape: {trial_onsets.shape}")
            print(f"   Trial onset times (first 10): {trial_onsets[:10] / sampling_rate}")
            
            # Check what stimcode values are at trial onsets
            if len(trial_onsets) > 0:
                stimcode_at_onsets = stimcode[trial_onsets]
                print(f"   Stimcode values at trial onsets: {np.unique(stimcode_at_onsets)}")
                print(f"   Stimcode at onsets counts: {np.bincount(stimcode_at_onsets.astype(int))}")
                
        except Exception as e:
            print(f"   Error loading preprocessed data: {e}")
    
    # The issue: trial onset detection vs stimcode alignment
    print(f"\n🚨 ISSUE IDENTIFIED:")
    print(f"   The preprocessing pipeline detects trial onsets from photodiode changes")
    print(f"   but then assigns stimcode values at those onset times.")
    print(f"   If the photodiode changes don't align with stimcode changes,")
    print(f"   all trials will get the same stimcode value.")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    print(f"   The trial onset detection should be based on stimcode changes,")
    print(f"   not just photodiode changes, to ensure proper stimulus labeling.")

def create_visualization():
    """Create visualization of the issue."""
    print(f"\n📊 Creating visualization...")
    
    raw_data = load_raw_data()
    if raw_data is None:
        return
    
    stimcode = raw_data['stimcode']
    photodiode = raw_data['photodiode']
    sampling_rate = raw_data['sampling_rate']
    
    # Create time vector
    time = np.arange(len(stimcode)) / sampling_rate
    
    # Find changes
    stimcode_diff = np.abs(np.diff(stimcode))
    stimcode_changes = np.where(stimcode_diff > 0)[0]
    
    photodiode_diff = np.abs(np.diff(photodiode))
    threshold = np.std(photodiode_diff) * 2
    photodiode_changes = np.where(photodiode_diff > threshold)[0]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot stimcode
    ax1.plot(time, stimcode, 'b-', alpha=0.7, label='Stimcode')
    ax1.scatter(stimcode_changes / sampling_rate, stimcode[stimcode_changes], 
               color='red', s=50, label='Stimcode Changes', zorder=5)
    ax1.set_ylabel('Stimcode Value')
    ax1.set_title('Stimcode Signal and Changes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot photodiode
    ax2.plot(time, photodiode, 'g-', alpha=0.7, label='Photodiode')
    ax2.scatter(photodiode_changes / sampling_rate, photodiode[photodiode_changes], 
               color='orange', s=50, label='Photodiode Changes', zorder=5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Photodiode Value')
    ax2.set_title('Photodiode Signal and Changes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stimcode_issue_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"   Visualization saved as 'stimcode_issue_diagnosis.png'")

if __name__ == "__main__":
    analyze_stimcode_issue()
    create_visualization()
