#!/usr/bin/env python3
"""
Feature Importance Analysis for ECoG Video Analysis
Generates comprehensive visualizations for presentation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import cv2
import os
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class FeatureImportanceAnalyzer:
    def __init__(self, data_dir="data", results_dir="results/07_feature_importance"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Define classes (7 objects + background = 8 classes)
        self.classes = {
            0: "background",
            1: "digit", 
            2: "kanji",
            3: "face",
            4: "body",
            5: "object",
            6: "hiragana",
            7: "line"
        }
        
        # Define brain regions (approximate mapping for 156 channels)
        self.brain_regions = self.define_brain_regions()
        
        # Color scheme for classes
        self.class_colors = {
            0: '#808080',  # background - gray
            1: '#FF6B6B',  # digit - red
            2: '#4ECDC4',  # kanji - teal
            3: '#45B7D1',  # face - blue
            4: '#96CEB4',  # body - green
            5: '#FFEAA7',  # object - yellow
            6: '#DDA0DD',  # hiragana - plum
            7: '#98D8C8'   # line - mint
        }
        
    def load_data(self):
        """Load all necessary data"""
        print("Loading data...")
        
        # Load preprocessed data
        self.epochs = np.load(self.data_dir / "preprocessed/experiment8/epochs.npy")
        self.stimcode = np.load(self.data_dir / "preprocessed/experiment8/stimcode.npy")
        self.time_vector = np.load(self.data_dir / "preprocessed/experiment8/time_vector.npy")
        self.good_channels = np.load(self.data_dir / "preprocessed/experiment8/good_channels.npy")
        
        # Load feature data
        self.comprehensive_features = self.load_comprehensive_features()
        self.transformer_features = np.load(self.data_dir / "features/experiment8/transformer/multi_scale_features.npy")
        
        # Load labels (this is metadata, not actual labels)
        # We'll use stimcode to create class labels
        
        print(f"Loaded epochs: {self.epochs.shape}")
        print(f"Loaded stimcode: {self.stimcode.shape}")
        print(f"Loaded transformer features: {self.transformer_features.shape}")
        
    def load_comprehensive_features(self):
        """Load comprehensive features from all frequency bands"""
        bands = ['theta', 'alpha', 'beta', 'gamma', 'gamma_30_80']
        features = []
        
        for band in bands:
            band_data = np.load(self.data_dir / f"features/experiment8/comprehensive/{band}_power.npy")
            features.append(band_data)
            
        return np.concatenate(features, axis=1)  # Shape: (252, 780)
    
    def define_brain_regions(self):
        """Define brain regions for 156 channels"""
        # Approximate mapping based on typical ECoG grid layout
        regions = {}
        
        # Frontal regions (channels 1-40)
        regions['frontal'] = list(range(1, 41))
        
        # Central regions (channels 41-80) 
        regions['central'] = list(range(41, 81))
        
        # Parietal regions (channels 81-120)
        regions['parietal'] = list(range(81, 121))
        
        # Occipital regions (channels 121-156)
        regions['occipital'] = list(range(121, 157))
        
        return regions
    
    def create_class_distribution_analysis(self):
        """Analyze class distribution and create visualizations"""
        print("Creating class distribution analysis...")
        
        # Map stimcode to classes (assuming stimcode 0 = background, 1-7 = objects)
        class_labels = np.zeros_like(self.stimcode)
        for i, stim in enumerate(self.stimcode):
            if stim == 0:
                class_labels[i] = 0  # background
            else:
                class_labels[i] = int(stim)  # object classes
        
        # Create class distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Class counts
        unique_classes, counts = np.unique(class_labels, return_counts=True)
        class_names = [self.classes[c] for c in unique_classes]
        colors = [self.class_colors[c] for c in unique_classes]
        
        # Bar plot
        bars = ax1.bar(class_names, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Class Distribution Across Trials', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Trials')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(counts, labels=class_names, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'class_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return class_labels
    
    def create_brain_region_patterns(self, class_labels):
        """Create brain region pattern analysis"""
        print("Creating brain region pattern analysis...")
        
        # Calculate average power for each class in each brain region
        region_patterns = {}
        
        for region_name, channel_indices in self.brain_regions.items():
            # Filter channels that exist in good_channels
            region_channels = [ch for ch in channel_indices if ch in self.good_channels]
            if not region_channels:
                continue
                
            # Get channel positions in good_channels array
            channel_positions = [np.where(self.good_channels == ch)[0][0] for ch in region_channels]
            
            # Extract region data
            region_data = self.epochs[:, channel_positions, :]  # (trials, channels, time)
            
            # Calculate average power for each class
            region_patterns[region_name] = {}
            
            for class_id, class_name in self.classes.items():
                class_mask = class_labels == class_id
                if np.sum(class_mask) > 0:
                    class_data = region_data[class_mask]
                    # Average across trials and time
                    avg_power = np.mean(class_data, axis=(0, 2))
                    region_patterns[region_name][class_name] = avg_power
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (region_name, patterns) in enumerate(region_patterns.items()):
            ax = axes[i]
            
            # Create heatmap
            pattern_matrix = []
            class_names = []
            
            for class_name, pattern in patterns.items():
                pattern_matrix.append(pattern)
                class_names.append(class_name)
            
            if pattern_matrix:
                pattern_matrix = np.array(pattern_matrix)
                
                # Create heatmap
                im = ax.imshow(pattern_matrix, cmap='viridis', aspect='auto')
                ax.set_title(f'{region_name.title()} Region Patterns', fontsize=12, fontweight='bold')
                ax.set_xlabel('Channel Index')
                ax.set_ylabel('Class')
                ax.set_yticks(range(len(class_names)))
                ax.set_yticklabels(class_names)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='Average Power')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'brain_region_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return region_patterns
    
    def create_channel_correlation_analysis(self, class_labels):
        """Create channel correlation analysis within brain regions"""
        print("Creating channel correlation analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (region_name, channel_indices) in enumerate(self.brain_regions.items()):
            ax = axes[i]
            
            # Filter channels that exist in good_channels
            region_channels = [ch for ch in channel_indices if ch in self.good_channels]
            if not region_channels:
                continue
                
            # Get channel positions
            channel_positions = [np.where(self.good_channels == ch)[0][0] for ch in region_channels]
            
            # Extract region data and calculate average power
            region_data = self.epochs[:, channel_positions, :]
            avg_power = np.mean(region_data, axis=2)  # Average across time
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(avg_power.T)
            
            # Create heatmap
            im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'{region_name.title()} Region Channel Correlations', fontsize=12, fontweight='bold')
            ax.set_xlabel('Channel Index')
            ax.set_ylabel('Channel Index')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Correlation Coefficient')
            
            # Add channel numbers as ticks
            ax.set_xticks(range(len(region_channels)))
            ax.set_yticks(range(len(region_channels)))
            ax.set_xticklabels(region_channels, rotation=45)
            ax.set_yticklabels(region_channels)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'channel_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_transient_behavior_analysis(self, class_labels):
        """Create transient behavior analysis across frames"""
        print("Creating transient behavior analysis...")
        
        # Calculate time-resolved patterns for each class
        time_points = self.time_vector
        n_classes = len(self.classes)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for class_id, class_name in self.classes.items():
            ax = axes[class_id]
            
            # Get trials for this class
            class_mask = class_labels == class_id
            if np.sum(class_mask) == 0:
                ax.text(0.5, 0.5, f'No data for {class_name}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{class_name.title()} - No Data', fontweight='bold')
                continue
            
            class_epochs = self.epochs[class_mask]  # (n_trials, channels, time)
            
            # Calculate average across trials
            avg_epoch = np.mean(class_epochs, axis=0)  # (channels, time)
            
            # Create time-frequency-like plot
            im = ax.imshow(avg_epoch, cmap='viridis', aspect='auto', 
                          extent=[time_points[0], time_points[-1], 0, avg_epoch.shape[0]])
            
            ax.set_title(f'{class_name.title()} - Temporal Pattern', fontweight='bold')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Channel Index')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Average Power')
            
            # Add vertical line at stimulus onset
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'transient_behavior_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_importance_heatmap(self, class_labels):
        """Create feature importance heatmap using comprehensive features"""
        print("Creating feature importance heatmap...")
        
        # Calculate feature importance for each class
        feature_importance = {}
        
        for class_id, class_name in self.classes.items():
            class_mask = class_labels == class_id
            if np.sum(class_mask) < 2:  # Need at least 2 samples
                continue
                
            class_features = self.comprehensive_features[class_mask]
            other_features = self.comprehensive_features[~class_mask]
            
            # Calculate t-statistic for each feature
            t_stats = []
            for i in range(class_features.shape[1]):
                t_stat, _ = stats.ttest_ind(class_features[:, i], other_features[:, i])
                t_stats.append(abs(t_stat))
            
            feature_importance[class_name] = t_stats
        
        # Create heatmap
        if feature_importance:
            importance_matrix = np.array(list(feature_importance.values()))
            class_names = list(feature_importance.keys())
            
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Create heatmap
            im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
            
            ax.set_title('Feature Importance Across Classes (T-statistics)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Class')
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='|T-statistic|')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_class_comparison_visualization(self, class_labels):
        """Create side-by-side comparison of classes with frame examples"""
        print("Creating class comparison visualization...")
        
        # This would ideally show actual video frames, but we'll create a conceptual visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for class_id, class_name in self.classes.items():
            ax = axes[class_id]
            
            # Get trials for this class
            class_mask = class_labels == class_id
            if np.sum(class_mask) == 0:
                ax.text(0.5, 0.5, f'No data for {class_name}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{class_name.title()}', fontweight='bold')
                continue
            
            class_epochs = self.epochs[class_mask]
            
            # Calculate average pattern
            avg_pattern = np.mean(class_epochs, axis=(0, 2))  # Average across trials and time
            
            # Create brain-like visualization
            # Simulate a brain grid layout
            brain_grid = np.zeros((13, 13))  # 13x13 grid for 156 channels
            
            # Fill the grid with channel data
            for i, channel_power in enumerate(avg_pattern):
                if i < 156:  # Ensure we don't exceed grid size
                    row = i // 13
                    col = i % 13
                    brain_grid[row, col] = channel_power
            
            # Create heatmap
            im = ax.imshow(brain_grid, cmap='viridis', aspect='equal')
            ax.set_title(f'{class_name.title()}\n(n={np.sum(class_mask)} trials)', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Avg Power')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'class_comparison_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_statistics(self, class_labels):
        """Create summary statistics and metadata"""
        print("Creating summary statistics...")
        
        # Calculate statistics
        stats_dict = {
            'experiment_id': 'experiment8',
            'analysis_date': datetime.now().isoformat(),
            'total_trials': len(class_labels),
            'total_channels': len(self.good_channels),
            'time_window': f"{self.time_vector[0]:.1f} to {self.time_vector[-1]:.1f} ms",
            'classes': {}
        }
        
        for class_id, class_name in self.classes.items():
            class_mask = class_labels == class_id
            n_trials = np.sum(class_mask)
            
            if n_trials > 0:
                class_epochs = self.epochs[class_mask]
                avg_power = np.mean(class_epochs)
                std_power = np.std(class_epochs)
                
                stats_dict['classes'][class_name] = {
                    'n_trials': int(n_trials),
                    'percentage': float(n_trials / len(class_labels) * 100),
                    'avg_power': float(avg_power),
                    'std_power': float(std_power)
                }
        
        # Save statistics
        with open(self.results_dir / 'analysis_statistics.json', 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Class distribution
        class_counts = [stats_dict['classes'][name]['n_trials'] for name in self.classes.values() 
                       if name in stats_dict['classes']]
        class_names = [name for name in self.classes.values() if name in stats_dict['classes']]
        
        bars = ax1.bar(class_names, class_counts, color=[self.class_colors[i] for i, name in enumerate(self.classes.values()) 
                                                         if name in stats_dict['classes']])
        ax1.set_title('Trial Distribution by Class', fontweight='bold')
        ax1.set_ylabel('Number of Trials')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average power by class
        avg_powers = [stats_dict['classes'][name]['avg_power'] for name in self.classes.values() 
                     if name in stats_dict['classes']]
        
        bars2 = ax2.bar(class_names, avg_powers, color=[self.class_colors[i] for i, name in enumerate(self.classes.values()) 
                                                        if name in stats_dict['classes']])
        ax2.set_title('Average Power by Class', fontweight='bold')
        ax2.set_ylabel('Average Power')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_analysis(self):
        """Run the complete feature importance analysis"""
        print("Starting comprehensive feature importance analysis...")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Create class labels
        class_labels = self.create_class_distribution_analysis()
        
        # Run all analyses
        self.create_brain_region_patterns(class_labels)
        self.create_channel_correlation_analysis(class_labels)
        self.create_transient_behavior_analysis(class_labels)
        self.create_feature_importance_heatmap(class_labels)
        self.create_class_comparison_visualization(class_labels)
        self.create_summary_statistics(class_labels)
        
        print("Analysis complete! All visualizations saved.")
        print(f"Generated files in {self.results_dir}:")
        for file in self.results_dir.glob('*.png'):
            print(f"  - {file.name}")
        print(f"  - analysis_statistics.json")

if __name__ == "__main__":
    analyzer = FeatureImportanceAnalyzer()
    analyzer.run_complete_analysis()
