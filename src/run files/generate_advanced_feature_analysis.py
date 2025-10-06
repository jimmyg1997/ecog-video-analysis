#!/usr/bin/env python3
"""
Advanced Feature Analysis for ECoG Video Analysis
Creates sophisticated visualizations for presentation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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

class AdvancedFeatureAnalyzer:
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
        print("Loading data for advanced analysis...")
        
        # Load preprocessed data
        self.epochs = np.load(self.data_dir / "preprocessed/experiment8/epochs.npy")
        self.stimcode = np.load(self.data_dir / "preprocessed/experiment8/stimcode.npy")
        self.time_vector = np.load(self.data_dir / "preprocessed/experiment8/time_vector.npy")
        self.good_channels = np.load(self.data_dir / "preprocessed/experiment8/good_channels.npy")
        
        # Load feature data
        self.comprehensive_features = self.load_comprehensive_features()
        self.transformer_features = np.load(self.data_dir / "features/experiment8/transformer/multi_scale_features.npy")
        
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
    
    def create_dimensionality_reduction_analysis(self, class_labels):
        """Create PCA and t-SNE visualizations"""
        print("Creating dimensionality reduction analysis...")
        
        # Use comprehensive features for dimensionality reduction
        features = self.comprehensive_features
        
        # PCA Analysis
        pca = PCA(n_components=10)
        pca_features = pca.fit_transform(features)
        
        # Skip t-SNE due to compatibility issues, use PCA instead
        pca_tsne = PCA(n_components=2)
        tsne_features = pca_tsne.fit_transform(features)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # PCA - First 2 components
        ax1 = axes[0, 0]
        for class_id, class_name in self.classes.items():
            mask = class_labels == class_id
            if np.sum(mask) > 0:
                ax1.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                           c=self.class_colors[class_id], label=class_name, alpha=0.7, s=50)
        ax1.set_title('PCA - First 2 Components', fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # PCA - Components 3 and 4
        ax2 = axes[0, 1]
        for class_id, class_name in self.classes.items():
            mask = class_labels == class_id
            if np.sum(mask) > 0:
                ax2.scatter(pca_features[mask, 2], pca_features[mask, 3], 
                           c=self.class_colors[class_id], label=class_name, alpha=0.7, s=50)
        ax2.set_title('PCA - Components 3 & 4', fontweight='bold')
        ax2.set_xlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
        ax2.set_ylabel(f'PC4 ({pca.explained_variance_ratio_[3]:.1%} variance)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Alternative PCA visualization
        ax3 = axes[1, 0]
        for class_id, class_name in self.classes.items():
            mask = class_labels == class_id
            if np.sum(mask) > 0:
                ax3.scatter(tsne_features[mask, 0], tsne_features[mask, 1], 
                           c=self.class_colors[class_id], label=class_name, alpha=0.7, s=50)
        ax3.set_title('Alternative PCA Visualization', fontweight='bold')
        ax3.set_xlabel('PC1 (Alternative)')
        ax3.set_ylabel('PC2 (Alternative)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Explained variance
        ax4 = axes[1, 1]
        ax4.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, 'bo-', linewidth=2, markersize=8)
        ax4.set_title('PCA Explained Variance Ratio', fontweight='bold')
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Explained Variance Ratio')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'dimensionality_reduction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return pca_features, tsne_features
    
    def create_temporal_dynamics_analysis(self, class_labels):
        """Create temporal dynamics analysis"""
        print("Creating temporal dynamics analysis...")
        
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
            
            # Calculate temporal dynamics
            # 1. Average across channels
            avg_temporal = np.mean(class_epochs, axis=1)  # (trials, time)
            
            # 2. Calculate mean and std across trials
            mean_temporal = np.mean(avg_temporal, axis=0)
            std_temporal = np.std(avg_temporal, axis=0)
            
            # 3. Plot with error bars
            time_ms = self.time_vector
            ax.plot(time_ms, mean_temporal, color=self.class_colors[class_id], linewidth=2, label='Mean')
            ax.fill_between(time_ms, mean_temporal - std_temporal, mean_temporal + std_temporal, 
                           alpha=0.3, color=self.class_colors[class_id], label='±1 SD')
            
            ax.set_title(f'{class_name.title()}\n(n={np.sum(class_mask)} trials)', fontweight='bold')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Average Power')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'temporal_dynamics_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_frequency_band_analysis(self, class_labels):
        """Create frequency band specific analysis"""
        print("Creating frequency band analysis...")
        
        # Load individual frequency band data
        bands = ['theta', 'alpha', 'beta', 'gamma', 'gamma_30_80']
        band_data = {}
        
        for band in bands:
            band_data[band] = np.load(self.data_dir / f"features/experiment8/comprehensive/{band}_power.npy")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot each frequency band
        for i, (band, data) in enumerate(band_data.items()):
            ax = axes[i]
            
            # Calculate average power for each class
            class_means = []
            class_names = []
            class_colors = []
            
            for class_id, class_name in self.classes.items():
                class_mask = class_labels == class_id
                if np.sum(class_mask) > 0:
                    class_mean = np.mean(data[class_mask])
                    class_means.append(class_mean)
                    class_names.append(class_name)
                    class_colors.append(self.class_colors[class_id])
            
            # Create bar plot
            if class_means:
                bars = ax.bar(class_names, class_means, color=class_colors, alpha=0.8, edgecolor='black', linewidth=1)
                ax.set_title(f'{band.title()} Band Power', fontweight='bold')
                ax.set_ylabel('Average Power')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, class_means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Create frequency band comparison
        ax = axes[5]
        
        # Calculate average power across all classes for each band
        band_means = []
        band_stds = []
        
        for band, data in band_data.items():
            band_means.append(np.mean(data))
            band_stds.append(np.std(data))
        
        bars = ax.bar(bands, band_means, yerr=band_stds, capsize=5, alpha=0.8, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax.set_title('Average Power Across All Classes', fontweight='bold')
        ax.set_ylabel('Average Power')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'frequency_band_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_clustering_analysis(self, class_labels):
        """Create clustering analysis to find natural groupings"""
        print("Creating clustering analysis...")
        
        # Use comprehensive features for clustering
        features = self.comprehensive_features
        
        # Perform K-means clustering
        n_clusters = 8  # Same as number of classes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # PCA visualization with true classes
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(features)
        
        ax1 = axes[0, 0]
        for class_id, class_name in self.classes.items():
            mask = class_labels == class_id
            if np.sum(mask) > 0:
                ax1.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                           c=self.class_colors[class_id], label=class_name, alpha=0.7, s=50)
        ax1.set_title('True Class Labels (PCA)', fontweight='bold')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # PCA visualization with cluster labels
        ax2 = axes[0, 1]
        scatter = ax2.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_labels, 
                             cmap='tab10', alpha=0.7, s=50)
        ax2.set_title('K-means Clustering (PCA)', fontweight='bold')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Cluster')
        
        # Confusion matrix between true classes and clusters
        ax3 = axes[1, 0]
        
        # Create confusion matrix
        confusion_matrix = np.zeros((len(self.classes), n_clusters))
        for i, true_class in enumerate(self.classes.keys()):
            mask = class_labels == true_class
            if np.sum(mask) > 0:
                cluster_counts = np.bincount(cluster_labels[mask], minlength=n_clusters)
                confusion_matrix[i, :] = cluster_counts
        
        # Normalize by row (true class)
        confusion_matrix_norm = confusion_matrix / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        im = ax3.imshow(confusion_matrix_norm, cmap='Blues', aspect='auto')
        ax3.set_title('Class-Cluster Confusion Matrix', fontweight='bold')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('True Class')
        ax3.set_xticks(range(n_clusters))
        ax3.set_yticks(range(len(self.classes)))
        ax3.set_yticklabels(list(self.classes.values()))
        
        # Add text annotations
        for i in range(len(self.classes)):
            for j in range(n_clusters):
                text = ax3.text(j, i, f'{confusion_matrix_norm[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax3, label='Normalized Count')
        
        # Cluster purity analysis
        ax4 = axes[1, 1]
        
        cluster_purity = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_classes = class_labels[cluster_mask]
                most_common_class = stats.mode(cluster_classes)[0][0]
                purity = np.sum(cluster_classes == most_common_class) / len(cluster_classes)
                cluster_purity.append(purity)
            else:
                cluster_purity.append(0)
        
        bars = ax4.bar(range(n_clusters), cluster_purity, alpha=0.8, color='skyblue', edgecolor='black')
        ax4.set_title('Cluster Purity', fontweight='bold')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Purity')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, purity in zip(bars, cluster_purity):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{purity:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_statistical_significance_analysis(self, class_labels):
        """Create statistical significance analysis between classes"""
        print("Creating statistical significance analysis...")
        
        # Calculate pairwise t-tests between all classes
        n_classes = len(self.classes)
        p_values = np.zeros((n_classes, n_classes))
        effect_sizes = np.zeros((n_classes, n_classes))
        
        # Use comprehensive features
        features = self.comprehensive_features
        
        for i, class1 in enumerate(self.classes.keys()):
            for j, class2 in enumerate(self.classes.keys()):
                if i != j:
                    mask1 = class_labels == class1
                    mask2 = class_labels == class2
                    
                    if np.sum(mask1) > 1 and np.sum(mask2) > 1:
                        # Calculate average power for each class
                        power1 = np.mean(features[mask1], axis=1)
                        power2 = np.mean(features[mask2], axis=1)
                        
                        # Perform t-test
                        t_stat, p_val = stats.ttest_ind(power1, power2)
                        p_values[i, j] = p_val
                        
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(power1) - 1) * np.var(power1) + 
                                            (len(power2) - 1) * np.var(power2)) / 
                                           (len(power1) + len(power2) - 2))
                        effect_size = (np.mean(power1) - np.mean(power2)) / pooled_std
                        effect_sizes[i, j] = abs(effect_size)
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # P-values heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(p_values, cmap='Reds', aspect='auto', vmin=0, vmax=0.05)
        ax1.set_title('Pairwise T-test P-values', fontweight='bold')
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
                    text = ax1.text(j, i, f'{p_values[i, j]:.3f}',
                                   ha="center", va="center", color="white" if p_values[i, j] < 0.025 else "black",
                                   fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='P-value')
        
        # Effect sizes heatmap
        ax2 = axes[1]
        im2 = ax2.imshow(effect_sizes, cmap='Blues', aspect='auto')
        ax2.set_title('Effect Sizes (Cohen\'s d)', fontweight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Class')
        ax2.set_xticks(range(n_classes))
        ax2.set_yticks(range(n_classes))
        ax2.set_xticklabels(list(self.classes.values()), rotation=45)
        ax2.set_yticklabels(list(self.classes.values()))
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    text = ax2.text(j, i, f'{effect_sizes[i, j]:.2f}',
                                   ha="center", va="center", color="white" if effect_sizes[i, j] > 0.5 else "black",
                                   fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, label='Effect Size')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'statistical_significance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_advanced_analysis(self):
        """Run the complete advanced analysis"""
        print("Starting advanced feature analysis...")
        
        # Create class labels
        class_labels = self.create_class_labels()
        
        # Run all advanced analyses
        self.create_dimensionality_reduction_analysis(class_labels)
        self.create_temporal_dynamics_analysis(class_labels)
        self.create_frequency_band_analysis(class_labels)
        self.create_clustering_analysis(class_labels)
        self.create_statistical_significance_analysis(class_labels)
        
        print("Advanced analysis complete!")
        print("Generated additional files:")
        for file in self.results_dir.glob('*analysis.png'):
            if 'dimensionality' in file.name or 'temporal' in file.name or 'frequency' in file.name or 'clustering' in file.name or 'statistical' in file.name:
                print(f"  - {file.name}")

if __name__ == "__main__":
    analyzer = AdvancedFeatureAnalyzer()
    analyzer.run_advanced_analysis()
