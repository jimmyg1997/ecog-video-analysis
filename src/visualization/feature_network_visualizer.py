#!/usr/bin/env python3
"""
Multi-Scale Feature Correlation Network Visualizer
IEEE-SMC-2025 ECoG Video Analysis Competition

This module creates sophisticated network visualizations showing relationships
between all feature types and brain regions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureNetworkVisualizer:
    """Multi-scale feature correlation network visualizer."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the feature network visualizer."""
        self.config = config or {}
        
        # Feature categories
        self.feature_categories = {
            'template_correlation': {
                'color': '#FF6B6B',
                'shape': 'circle',
                'size': 20
            },
            'csp_lda': {
                'color': '#4ECDC4',
                'shape': 'square',
                'size': 18
            },
            'eegnet': {
                'color': '#45B7D1',
                'shape': 'diamond',
                'size': 22
            },
            'transformer': {
                'color': '#96CEB4',
                'shape': 'triangle-up',
                'size': 20
            },
            'comprehensive': {
                'color': '#FFEAA7',
                'shape': 'star',
                'size': 16
            }
        }
        
        # Brain regions
        self.brain_regions = {
            'frontal': {'color': '#FF6B6B', 'coords': (0, 0)},
            'parietal': {'color': '#4ECDC4', 'coords': (1, 0)},
            'temporal': {'color': '#45B7D1', 'coords': (0, 1)},
            'occipital': {'color': '#96CEB4', 'coords': (1, 1)},
            'central': {'color': '#FFEAA7', 'coords': (0.5, 0.5)}
        }
    
    def extract_feature_vectors(self, all_features: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Extract feature vectors from all feature types."""
        print("🔧 Extracting feature vectors from all feature types")
        
        feature_vectors = {}
        
        for extractor_name, features in all_features.items():
            if extractor_name == 'template_correlation':
                if 'template_correlations' in features:
                    feature_vectors[extractor_name] = features['template_correlations'].flatten()
                elif 'loocv_features' in features:
                    feature_vectors[extractor_name] = features['loocv_features'].flatten()
                    
            elif extractor_name == 'csp_lda':
                if 'csp_features' in features:
                    feature_vectors[extractor_name] = features['csp_features'].flatten()
                elif 'spatial_features' in features:
                    feature_vectors[extractor_name] = features['spatial_features'].flatten()
                    
            elif extractor_name == 'eegnet':
                if 'cnn_input' in features:
                    cnn_data = features['cnn_input']
                    if len(cnn_data.shape) > 2:
                        feature_vectors[extractor_name] = cnn_data.reshape(cnn_data.shape[0], -1).flatten()
                    else:
                        feature_vectors[extractor_name] = cnn_data.flatten()
                        
            elif extractor_name == 'transformer':
                if 'transformer_input' in features:
                    feature_vectors[extractor_name] = features['transformer_input'].flatten()
                elif 'attention_features' in features:
                    feature_vectors[extractor_name] = features['attention_features'].flatten()
                    
            elif extractor_name == 'comprehensive':
                if 'gamma_power' in features:
                    feature_vectors[extractor_name] = features['gamma_power'].flatten()
                elif 'band_powers' in features:
                    # Combine all band powers
                    band_data = []
                    for band_name, band_data_array in features['band_powers'].items():
                        band_data.append(band_data_array.flatten())
                    feature_vectors[extractor_name] = np.concatenate(band_data)
        
        print(f"   📊 Extracted {len(feature_vectors)} feature vectors")
        for name, vector in feature_vectors.items():
            print(f"     • {name}: {vector.shape}")
        
        return feature_vectors
    
    def compute_feature_correlations(self, feature_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute correlation matrix between all feature types."""
        print("🔧 Computing feature correlations")
        
        # Align feature vectors to same length
        min_length = min(len(vector) for vector in feature_vectors.values())
        aligned_vectors = {}
        
        for name, vector in feature_vectors.items():
            if len(vector) > min_length:
                # Truncate to minimum length
                aligned_vectors[name] = vector[:min_length]
            else:
                aligned_vectors[name] = vector
        
        # Create feature matrix
        feature_names = list(aligned_vectors.keys())
        feature_matrix = np.array([aligned_vectors[name] for name in feature_names])
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(feature_matrix)
        
        print(f"   📊 Correlation matrix shape: {correlation_matrix.shape}")
        print(f"   📊 Feature names: {feature_names}")
        
        return correlation_matrix, feature_names
    
    def create_feature_network_graph(self, correlation_matrix: np.ndarray,
                                   feature_names: List[str],
                                   threshold: float = 0.3,
                                   save_path: Path = None) -> go.Figure:
        """Create network graph of feature correlations."""
        print("🎨 Creating feature correlation network graph")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(feature_names):
            category_info = self.feature_categories.get(name, {
                'color': '#CCCCCC',
                'shape': 'circle',
                'size': 15
            })
            
            G.add_node(name, 
                      color=category_info['color'],
                      shape=category_info['shape'],
                      size=category_info['size'])
        
        # Add edges based on correlations
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                correlation = abs(correlation_matrix[i, j])
                if correlation > threshold:
                    G.add_edge(feature_names[i], feature_names[j], 
                             weight=correlation,
                             correlation=correlation_matrix[i, j])
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = G[edge[0]][edge[1]]['weight']
            correlation = G[edge[0]][edge[1]]['correlation']
            edge_info.append(f"{edge[0]} ↔ {edge[1]}<br>Correlation: {correlation:.3f}<br>Weight: {weight:.3f}")
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(0,0,0,0.3)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            node_info = G.nodes[node]
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(
                    size=node_info['size'],
                    color=node_info['color'],
                    symbol=node_info['shape'],
                    line=dict(width=2, color='black')
                ),
                text=node,
                hovertemplate=f"<b>{node}</b><br>Feature Type<br><extra></extra>",
                name=node,
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Feature Correlation Network (threshold: {threshold})",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600,
            showlegend=True
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / f"feature_network_threshold_{threshold}.html")
            fig.write_image(save_path / f"feature_network_threshold_{threshold}.png")
        
        return fig
    
    def create_hierarchical_clustering(self, correlation_matrix: np.ndarray,
                                     feature_names: List[str],
                                     save_path: Path = None) -> go.Figure:
        """Create hierarchical clustering dendrogram."""
        print("🎨 Creating hierarchical clustering dendrogram")
        
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        
        # Create dendrogram
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dendrogram(linkage_matrix, 
                  labels=feature_names,
                  ax=ax,
                  leaf_rotation=90,
                  leaf_font_size=12)
        
        ax.set_title('Hierarchical Clustering of Feature Types', fontsize=16, fontweight='bold')
        ax.set_xlabel('Feature Types', fontsize=14)
        ax.set_ylabel('Distance', fontsize=14)
        
        # Color code by feature category
        for i, name in enumerate(feature_names):
            category_info = self.feature_categories.get(name, {'color': '#CCCCCC'})
            ax.get_xticklabels()[i].set_color(category_info['color'])
            ax.get_xticklabels()[i].set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path / "hierarchical_clustering.png", dpi=300, bbox_inches='tight')
            plt.savefig(save_path / "hierarchical_clustering.svg", bbox_inches='tight')
        
        return fig
    
    def create_feature_heatmap(self, correlation_matrix: np.ndarray,
                             feature_names: List[str],
                             save_path: Path = None) -> go.Figure:
        """Create correlation heatmap with annotations."""
        print("🎨 Creating feature correlation heatmap")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=feature_names,
            y=feature_names,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis_title="Feature Types",
            yaxis_title="Feature Types",
            width=800,
            height=600
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / "feature_correlation_heatmap.html")
            fig.write_image(save_path / "feature_correlation_heatmap.png")
        
        return fig
    
    def create_feature_importance_network(self, all_features: Dict[str, Dict],
                                        save_path: Path = None) -> go.Figure:
        """Create network showing feature importance across brain regions."""
        print("🎨 Creating feature importance network")
        
        # Extract feature importance data
        importance_data = {}
        
        for extractor_name, features in all_features.items():
            if 'feature_importance' in features:
                importance_data[extractor_name] = features['feature_importance']
            elif 'gamma_power' in features:
                # Use gamma power as importance proxy
                importance_data[extractor_name] = np.mean(features['gamma_power'], axis=0)
        
        if not importance_data:
            print("   ⚠️ No importance data available")
            return None
        
        # Create network graph
        fig = go.Figure()
        
        # Add nodes for each feature type
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for i, (feature_name, importance) in enumerate(importance_data.items()):
            category_info = self.feature_categories.get(feature_name, {
                'color': '#CCCCCC',
                'size': 15
            })
            
            node_x.append(i)
            node_y.append(0)
            node_text.append(feature_name)
            node_colors.append(category_info['color'])
            node_sizes.append(category_info['size'])
        
        # Add feature type nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black')
            ),
            text=node_text,
            textposition="top center",
            name="Feature Types",
            showlegend=True
        ))
        
        # Add brain region nodes
        region_x = []
        region_y = []
        region_text = []
        region_colors = []
        
        for i, (region_name, region_info) in enumerate(self.brain_regions.items()):
            region_x.append(i)
            region_y.append(1)
            region_text.append(region_name.title())
            region_colors.append(region_info['color'])
        
        fig.add_trace(go.Scatter(
            x=region_x, y=region_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=region_colors,
                line=dict(width=2, color='black')
            ),
            text=region_text,
            textposition="bottom center",
            name="Brain Regions",
            showlegend=True
        ))
        
        # Add connections (simplified)
        for i, feature_name in enumerate(importance_data.keys()):
            for j, region_name in enumerate(self.brain_regions.keys()):
                # Add connection line
                fig.add_trace(go.Scatter(
                    x=[i, j], y=[0, 1],
                    mode='lines',
                    line=dict(width=1, color='rgba(0,0,0,0.3)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Update layout
        fig.update_layout(
            title="Feature Importance Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600,
            showlegend=True
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / "feature_importance_network.html")
            fig.write_image(save_path / "feature_importance_network.png")
        
        return fig
    
    def create_comprehensive_dashboard(self, all_features: Dict[str, Dict],
                                     save_path: Path = None) -> go.Figure:
        """Create comprehensive dashboard with all network visualizations."""
        print("🎨 Creating comprehensive feature network dashboard")
        
        # Extract feature vectors
        feature_vectors = self.extract_feature_vectors(all_features)
        
        if not feature_vectors:
            print("   ⚠️ No feature vectors available")
            return None
        
        # Compute correlations
        correlation_matrix, feature_names = self.compute_feature_correlations(feature_vectors)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Feature Correlation Heatmap",
                "Feature Network Graph",
                "Feature Importance Network",
                "Feature Statistics"
            ],
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # 1. Correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix,
                x=feature_names,
                y=feature_names,
                colorscale='RdBu',
                zmid=0,
                showscale=False
            ),
            row=1, col=1
        )
        
        # 2. Network graph (simplified)
        # Create positions for network nodes
        n_features = len(feature_names)
        angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Add nodes
        for i, name in enumerate(feature_names):
            category_info = self.feature_categories.get(name, {
                'color': '#CCCCCC',
                'size': 15
            })
            
            fig.add_trace(
                go.Scatter(
                    x=[x_pos[i]], y=[y_pos[i]],
                    mode='markers+text',
                    marker=dict(
                        size=category_info['size'],
                        color=category_info['color']
                    ),
                    text=name,
                    textposition="top center",
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Feature importance (simplified)
        importance_values = []
        for name in feature_names:
            if name in feature_vectors:
                importance_values.append(np.mean(np.abs(feature_vectors[name])))
            else:
                importance_values.append(0)
        
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=importance_values,
                marker_color=[self.feature_categories.get(name, {'color': '#CCCCCC'})['color'] 
                             for name in feature_names]
            ),
            row=2, col=1
        )
        
        # 4. Feature statistics
        feature_sizes = [len(feature_vectors[name]) for name in feature_names]
        
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=feature_sizes,
                marker_color=[self.feature_categories.get(name, {'color': '#CCCCCC'})['color'] 
                             for name in feature_names]
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Comprehensive Feature Network Dashboard",
            height=800,
            width=1200,
            showlegend=False
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / "comprehensive_feature_dashboard.html")
            fig.write_image(save_path / "comprehensive_feature_dashboard.png")
        
        return fig
    
    def get_summary_report(self) -> str:
        """Generate a summary report of the feature network visualizations."""
        report = []
        report.append("🎨 Feature Network Visualizer Summary")
        report.append("=" * 50)
        
        report.append(f"📊 Feature Categories: {len(self.feature_categories)}")
        for category, info in self.feature_categories.items():
            report.append(f"   • {category}: {info['color']} ({info['shape']})")
        
        report.append(f"\n📊 Brain Regions: {len(self.brain_regions)}")
        for region, info in self.brain_regions.items():
            report.append(f"   • {region.title()}: {info['coords']}")
        
        report.append(f"\n📊 Visualization Types:")
        report.append("   • Feature Correlation Networks")
        report.append("   • Hierarchical Clustering Dendrograms")
        report.append("   • Correlation Heatmaps")
        report.append("   • Feature Importance Networks")
        report.append("   • Comprehensive Dashboards")
        
        return "\n".join(report)
