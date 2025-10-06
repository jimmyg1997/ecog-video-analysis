#!/usr/bin/env python3
"""
Simple Feature Analysis for ECoG Video Analysis
Creates visualizations without sklearn dependencies
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import os
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SimpleFeatureAnalyzer:
    def __init__(self, data_dir="data", results_dir="results/07_feature_importance"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Define classes
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
        
        # Color scheme
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
        print("Loading data for simple analysis...")
        
        # Load preprocessed data
        self.epochs = np.load(self.data_dir / "preprocessed/experiment8/epochs.npy")
        self.stimcode = np.load(self.data_dir / "preprocessed/experiment8/stimcode.npy")
        self.time_vector = np.load(self.data_dir / "preprocessed/experiment8/time_vector.npy")
        self.good_channels = np.load(self.data_dir / "preprocessed/experiment8/good_channels.npy")
        
        # Load feature data
        self.comprehensive_features = self.load_comprehensive_features()
        
        print(f"Loaded epochs: {self.epochs.shape}")
        print(f"Loaded stimcode: {self.stimcode.shape}")
        print(f"Loaded comprehensive features: {self.comprehensive_features.shape}")
        
    def load_comprehensive_features(self):
        """Load comprehensive features from all frequency bands"""
        bands = ['theta', 'alpha', 'beta', 'gamma', 'gamma_30_80']
        features = []
        
        for band in bands:
            band_data = np.load(self.data_dir / f"features/experiment8/comprehensive/{band}_power.npy")
            features.append(band_data)
            
        return np.concatenate(features, axis=1)
    
    def create_class_labels(self):
        """Create class labels from stimcode"""
        class_labels = np.zeros_like(self.stimcode)
        for i, stim in enumerate(self.stimcode):
            if stim == 0:
                class_labels[i] = 0  # background
            else:
                class_labels[i] = int(stim)  # object classes
        return class_labels
    
    def create_pca_analysis(self, class_labels):
        """Create PCA analysis manually"""
        print("Creating PCA analysis...")
        
        # Center the data
        features = self.comprehensive_features
        features_centered = features - np.mean(features, axis=0)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(features_centered.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate explained variance ratio
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        # Project data onto first 4 principal components
        pca_features = np.dot(features_centered, eigenvectors[:, :4])
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # PC1 vs PC2
        ax1 = axes[0, 0]
        for class_id, class_name in self.classes.items():
            mask = class_labels == class_id
            if np.sum(mask) > 0:
                ax1.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                           c=self.class_colors[class_id], label=class_name, alpha=0.7, s=50)
        ax1.set_title(f'PC1 vs PC2 ({explained_variance_ratio[0]:.1%} + {explained_variance_ratio[1]:.1%} variance)', fontweight='bold')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # PC3 vs PC4
        ax2 = axes[0, 1]
        for class_id, class_name in self.classes.items():
            mask = class_labels == class_id
            if np.sum(mask) > 0:
                ax2.scatter(pca_features[mask, 2], pca_features[mask, 3], 
                           c=self.class_colors[class_id], label=class_name, alpha=0.7, s=50)
        ax2.set_title(f'PC3 vs PC4 ({explained_variance_ratio[2]:.1%} + {explained_variance_ratio[3]:.1%} variance)', fontweight='bold')
        ax2.set_xlabel('PC3')
        ax2.set_ylabel('PC4')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Explained variance
        ax3 = axes[1, 0]
        ax3.plot(range(1, len(explained_variance_ratio) + 1), 
                explained_variance_ratio, 'bo-', linewidth=2, markersize=8)
        ax3.set_title('Explained Variance Ratio', fontweight='bold')
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        ax4 = axes[1, 1]
        cumulative_variance = np.cumsum(explained_variance_ratio)
        ax4.plot(range(1, len(cumulative_variance) + 1), 
                cumulative_variance, 'ro-', linewidth=2, markersize=8)
        ax4.set_title('Cumulative Explained Variance', fontweight='bold')
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Cumulative Explained Variance')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% variance')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return pca_features, explained_variance_ratio
    
    def create_class_separability_analysis(self, class_labels):
        """Create class separability analysis"""
        print("Creating class separability analysis...")
        
        # Calculate pairwise distances between class centroids
        n_classes = len(self.classes)
        centroids = np.zeros((n_classes, self.comprehensive_features.shape[1]))
        class_sizes = np.zeros(n_classes)
        
        for class_id in self.classes.keys():
            mask = class_labels == class_id
            if np.sum(mask) > 0:
                centroids[class_id] = np.mean(self.comprehensive_features[mask], axis=0)
                class_sizes[class_id] = np.sum(mask)
        
        # Calculate pairwise distances
        distances = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Distance matrix
        ax1 = axes[0]
        im1 = ax1.imshow(distances, cmap='viridis', aspect='auto')
        ax1.set_title('Pairwise Class Centroid Distances', fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Class')
        ax1.set_xticks(range(n_classes))
        ax1.set_yticks(range(n_classes))
        ax1.set_xticklabels(list(self.classes.values()), rotation=45)
        ax1.set_yticklabels(list(self.classes.values()))
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    text = ax1.text(j, i, f'{distances[i, j]:.2f}',
                                   ha="center", va="center", color="white" if distances[i, j] > np.median(distances) else "black",
                                   fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='Distance')
        
        # Class sizes
        ax2 = axes[1]
        class_names = list(self.classes.values())
        class_colors_list = [self.class_colors[i] for i in range(n_classes)]
        
        bars = ax2.bar(class_names, class_sizes, color=class_colors_list, alpha=0.8, edgecolor='black')
        ax2.set_title('Class Sizes (Number of Trials)', fontweight='bold')
        ax2.set_ylabel('Number of Trials')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, size in zip(bars, class_sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(size)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'class_separability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_importance_by_band(self, class_labels):
        """Create feature importance analysis by frequency band"""
        print("Creating feature importance by band...")
        
        # Load individual frequency band data
        bands = ['theta', 'alpha', 'beta', 'gamma', 'gamma_30_80']
        band_data = {}
        
        for band in bands:
            band_data[band] = np.load(self.data_dir / f"features/experiment8/comprehensive/{band}_power.npy")
        
        # Calculate feature importance for each band
        band_importance = {}
        
        for band, data in band_data.items():
            importance_scores = []
            
            for class_id in self.classes.keys():
                class_mask = class_labels == class_id
                other_mask = class_labels != class_id
                
                if np.sum(class_mask) > 1 and np.sum(other_mask) > 1:
                    # Calculate t-statistic for each channel
                    t_stats = []
                    for ch in range(data.shape[1]):
                        t_stat, _ = stats.ttest_ind(data[class_mask, ch], data[other_mask, ch])
                        t_stats.append(abs(t_stat))
                    
                    importance_scores.append(t_stats)
            
            if importance_scores:
                band_importance[band] = np.array(importance_scores)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (band, importance) in enumerate(band_importance.items()):
            ax = axes[i]
            
            # Calculate average importance across classes
            avg_importance = np.mean(importance, axis=0)
            
            # Create channel importance plot
            channels = range(len(avg_importance))
            bars = ax.bar(channels, avg_importance, alpha=0.8, color='skyblue', edgecolor='black')
            ax.set_title(f'{band.title()} Band - Channel Importance', fontweight='bold')
            ax.set_xlabel('Channel Index')
            ax.set_ylabel('Average |T-statistic|')
            ax.grid(True, alpha=0.3)
            
            # Highlight top 10 most important channels
            top_indices = np.argsort(avg_importance)[-10:]
            for idx in top_indices:
                bars[idx].set_color('red')
                bars[idx].set_alpha(1.0)
        
        # Summary plot
        ax = axes[5]
        band_avg_importance = []
        band_names = []
        
        for band, importance in band_importance.items():
            avg_imp = np.mean(importance)
            band_avg_importance.append(avg_imp)
            band_names.append(band.title())
        
        bars = ax.bar(band_names, band_avg_importance, alpha=0.8, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax.set_title('Average Feature Importance by Band', fontweight='bold')
        ax.set_ylabel('Average |T-statistic|')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, band_avg_importance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance_by_band.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_temporal_evolution_analysis(self, class_labels):
        """Create temporal evolution analysis"""
        print("Creating temporal evolution analysis...")
        
        # Define time windows
        time_windows = [
            (-300, -100, "Pre-stimulus"),
            (-100, 0, "Baseline"),
            (0, 100, "Early Response"),
            (100, 200, "Peak Response"),
            (200, 300, "Late Response")
        ]
        
        # Convert time vector to milliseconds
        time_ms = self.time_vector * 1000  # Convert to ms
        
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
            
            # Calculate power in each time window
            window_powers = []
            window_labels = []
            
            for start_ms, end_ms, label in time_windows:
                # Find time indices
                start_idx = np.argmin(np.abs(time_ms - start_ms))
                end_idx = np.argmin(np.abs(time_ms - end_ms))
                
                # Calculate average power in this window
                window_data = class_epochs[:, :, start_idx:end_idx]
                avg_power = np.mean(window_data, axis=(1, 2))  # Average across channels and time
                window_powers.append(avg_power)
                window_labels.append(label)
            
            # Create box plot
            bp = ax.boxplot(window_powers, labels=window_labels, patch_artist=True)
            
            # Color the boxes
            for patch in bp['boxes']:
                patch.set_facecolor(self.class_colors[class_id])
                patch.set_alpha(0.7)
            
            ax.set_title(f'{class_name.title()}\n(n={np.sum(class_mask)} trials)', fontweight='bold')
            ax.set_ylabel('Average Power')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'temporal_evolution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_simple_analysis(self):
        """Run the complete simple analysis"""
        print("Starting simple feature analysis...")
        
        # Create class labels
        class_labels = self.create_class_labels()
        
        # Run all analyses
        self.create_pca_analysis(class_labels)
        self.create_class_separability_analysis(class_labels)
        self.create_feature_importance_by_band(class_labels)
        self.create_temporal_evolution_analysis(class_labels)
        
        print("Simple analysis complete!")
        print("Generated additional files:")
        for file in self.results_dir.glob('*analysis.png'):
            if 'pca' in file.name or 'separability' in file.name or 'importance_by_band' in file.name or 'temporal_evolution' in file.name:
                print(f"  - {file.name}")

if __name__ == "__main__":
    analyzer = SimpleFeatureAnalyzer()
    analyzer.run_simple_analysis()
