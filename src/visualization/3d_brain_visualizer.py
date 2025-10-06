#!/usr/bin/env python3
"""
Interactive 3D Brain Activity Heatmap with Temporal Evolution
IEEE-SMC-2025 ECoG Video Analysis Competition

This module creates sophisticated 3D visualizations of brain activity
with real-time temporal evolution and interactive controls.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class Brain3DVisualizer:
    """3D brain visualization with temporal evolution."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the 3D brain visualizer."""
        self.config = config or {}
        
        # Brain atlas coordinates (simplified)
        self.brain_regions = {
            'frontal': {'coords': (0, 0, 1), 'color': '#FF6B6B'},
            'parietal': {'coords': (0, 1, 0), 'color': '#4ECDC4'},
            'temporal': {'coords': (1, 0, 0), 'color': '#45B7D1'},
            'occipital': {'coords': (0, -1, 0), 'color': '#96CEB4'},
            'central': {'coords': (0, 0, 0), 'color': '#FFEAA7'}
        }
        
        # Channel to region mapping (simplified)
        self.channel_regions = self._create_channel_mapping()
        
    def _create_channel_mapping(self) -> Dict[int, str]:
        """Create a mapping from channels to brain regions."""
        # This is a simplified mapping - in practice, you'd use real brain atlas data
        mapping = {}
        regions = list(self.brain_regions.keys())
        
        for ch in range(160):  # Assuming 160 channels
            region_idx = ch % len(regions)
            mapping[ch] = regions[region_idx]
        
        return mapping
    
    def create_3d_brain_activity(self, data: np.ndarray, 
                                time_points: np.ndarray = None,
                                feature_name: str = "Activity",
                                save_path: Path = None) -> go.Figure:
        """Create interactive 3D brain activity visualization."""
        print("🎨 Creating 3D brain activity visualization")
        
        # Prepare data
        if len(data.shape) == 3:  # (trials, channels, time)
            # Average across trials
            activity_data = np.mean(data, axis=0)
        elif len(data.shape) == 2:  # (channels, time) or (trials, channels)
            if data.shape[1] > data.shape[0]:  # (channels, time)
                activity_data = data
            else:  # (trials, channels)
                activity_data = np.mean(data, axis=0).reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        # Create time points if not provided
        if time_points is None:
            time_points = np.arange(activity_data.shape[1])
        
        # Prepare data for visualization
        x_coords = []
        y_coords = []
        z_coords = []
        activity_values = []
        channel_ids = []
        region_names = []
        colors = []
        
        for ch in range(activity_data.shape[0]):
            region = self.channel_regions.get(ch, 'central')
            region_info = self.brain_regions[region]
            
            # Add some noise to coordinates for better visualization
            base_coords = np.array(region_info['coords'])
            noise = np.random.normal(0, 0.1, 3)
            coords = base_coords + noise
            
            x_coords.append(coords[0])
            y_coords.append(coords[1])
            z_coords.append(coords[2])
            
            # Average activity across time
            activity_values.append(np.mean(activity_data[ch, :]))
            channel_ids.append(f"Ch{ch+1}")
            region_names.append(region)
            colors.append(region_info['color'])
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add brain regions as separate traces
        for region, region_info in self.brain_regions.items():
            region_mask = [r == region for r in region_names]
            
            if any(region_mask):
                fig.add_trace(go.Scatter3d(
                    x=[x_coords[i] for i in range(len(x_coords)) if region_mask[i]],
                    y=[y_coords[i] for i in range(len(y_coords)) if region_mask[i]],
                    z=[z_coords[i] for i in range(len(z_coords)) if region_mask[i]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=[activity_values[i] for i in range(len(activity_values)) if region_mask[i]],
                        colorscale='Viridis',
                        colorbar=dict(title=feature_name),
                        line=dict(width=2, color=region_info['color'])
                    ),
                    text=[f"Ch{ch+1}<br>Region: {region}<br>Activity: {activity_values[i]:.3f}" 
                          for i, ch in enumerate(range(len(activity_values))) if region_mask[i]],
                    hovertemplate='%{text}<extra></extra>',
                    name=region.title(),
                    showlegend=True
                ))
        
        # Update layout
        fig.update_layout(
            title=f"3D Brain Activity Map - {feature_name}",
            scene=dict(
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                zaxis_title="Z Coordinate",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / f"3d_brain_activity_{feature_name.lower().replace(' ', '_')}.html")
            fig.write_image(save_path / f"3d_brain_activity_{feature_name.lower().replace(' ', '_')}.png")
        
        return fig
    
    def create_temporal_evolution(self, data: np.ndarray,
                                 time_points: np.ndarray = None,
                                 feature_name: str = "Activity",
                                 save_path: Path = None) -> go.Figure:
        """Create temporal evolution animation of brain activity."""
        print("🎨 Creating temporal evolution animation")
        
        # Prepare data
        if len(data.shape) == 3:  # (trials, channels, time)
            activity_data = np.mean(data, axis=0)
        elif len(data.shape) == 2:  # (channels, time)
            activity_data = data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        # Create time points if not provided
        if time_points is None:
            time_points = np.arange(activity_data.shape[1])
        
        # Create frames for animation
        frames = []
        
        for t in range(0, activity_data.shape[1], max(1, activity_data.shape[1] // 50)):
            frame_data = []
            
            for ch in range(activity_data.shape[0]):
                region = self.channel_regions.get(ch, 'central')
                region_info = self.brain_regions[region]
                
                # Add noise to coordinates
                base_coords = np.array(region_info['coords'])
                noise = np.random.normal(0, 0.1, 3)
                coords = base_coords + noise
                
                frame_data.append({
                    'x': coords[0],
                    'y': coords[1],
                    'z': coords[2],
                    'activity': activity_data[ch, t],
                    'channel': f"Ch{ch+1}",
                    'region': region,
                    'color': region_info['color']
                })
            
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=[d['x'] for d in frame_data],
                    y=[d['y'] for d in frame_data],
                    z=[d['z'] for d in frame_data],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=[d['activity'] for d in frame_data],
                        colorscale='Viridis',
                        colorbar=dict(title=feature_name)
                    ),
                    text=[f"{d['channel']}<br>Region: {d['region']}<br>Activity: {d['activity']:.3f}" 
                          for d in frame_data],
                    hovertemplate='%{text}<extra></extra>'
                )],
                name=f"t={t}"
            ))
        
        # Create initial figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add play button
        fig.update_layout(
            title=f"Temporal Evolution of Brain Activity - {feature_name}",
            scene=dict(
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                zaxis_title="Z Coordinate"
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 100, 'redraw': True}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}}]
                    }
                ]
            }],
            width=800,
            height=600
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / f"temporal_evolution_{feature_name.lower().replace(' ', '_')}.html")
        
        return fig
    
    def create_region_comparison(self, data_dict: Dict[str, np.ndarray],
                                save_path: Path = None) -> go.Figure:
        """Create comparison of activity across different features/conditions."""
        print("🎨 Creating region comparison visualization")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(data_dict.keys()),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, (feature_name, data) in enumerate(data_dict.items()):
            if idx >= 4:  # Limit to 4 subplots
                break
            
            row, col = subplot_positions[idx]
            
            # Prepare data
            if len(data.shape) == 3:
                activity_data = np.mean(data, axis=0)
            elif len(data.shape) == 2:
                activity_data = data
            else:
                continue
            
            # Average across time
            avg_activity = np.mean(activity_data, axis=1)
            
            # Prepare coordinates
            x_coords = []
            y_coords = []
            z_coords = []
            activity_values = []
            colors = []
            
            for ch in range(len(avg_activity)):
                region = self.channel_regions.get(ch, 'central')
                region_info = self.brain_regions[region]
                
                base_coords = np.array(region_info['coords'])
                noise = np.random.normal(0, 0.1, 3)
                coords = base_coords + noise
                
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                z_coords.append(coords[2])
                activity_values.append(avg_activity[ch])
                colors.append(region_info['color'])
            
            # Add trace
            fig.add_trace(
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=activity_values,
                        colorscale='Viridis',
                        colorbar=dict(title=feature_name)
                    ),
                    name=feature_name,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Brain Activity Comparison Across Features",
            height=800,
            width=1000
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / "region_comparison.html")
            fig.write_image(save_path / "region_comparison.png")
        
        return fig
    
    def create_interactive_dashboard(self, all_features: Dict[str, Dict],
                                   save_path: Path = None) -> go.Figure:
        """Create interactive dashboard with multiple visualizations."""
        print("🎨 Creating interactive 3D brain dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Gamma Power Activity",
                "Template Correlation",
                "EEGNet Features",
                "Transformer Features"
            ],
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        # Process each feature type
        feature_data = {}
        
        # Gamma power
        if 'comprehensive' in all_features and 'gamma_power' in all_features['comprehensive']:
            feature_data['Gamma Power'] = all_features['comprehensive']['gamma_power']
        
        # Template correlation
        if 'template_correlation' in all_features and 'template_correlations' in all_features['template_correlation']:
            feature_data['Template Correlation'] = all_features['template_correlation']['template_correlations']
        
        # EEGNet features
        if 'eegnet' in all_features and 'cnn_input' in all_features['eegnet']:
            cnn_data = all_features['eegnet']['cnn_input']
            if len(cnn_data.shape) > 2:
                feature_data['EEGNet'] = cnn_data.reshape(cnn_data.shape[0], -1)
            else:
                feature_data['EEGNet'] = cnn_data
        
        # Transformer features
        if 'transformer' in all_features and 'transformer_input' in all_features['transformer']:
            feature_data['Transformer'] = all_features['transformer']['transformer_input']
        
        # Create visualizations
        subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, (feature_name, data) in enumerate(feature_data.items()):
            if idx >= 4:
                break
            
            row, col = subplot_positions[idx]
            
            # Prepare data
            if len(data.shape) == 2:
                # Average across trials if needed
                if data.shape[0] > data.shape[1]:
                    activity_data = np.mean(data, axis=0)
                else:
                    activity_data = data
            else:
                continue
            
            # Prepare coordinates and activity
            x_coords = []
            y_coords = []
            z_coords = []
            activity_values = []
            
            for ch in range(min(len(activity_data), 160)):  # Limit to 160 channels
                region = self.channel_regions.get(ch, 'central')
                region_info = self.brain_regions[region]
                
                base_coords = np.array(region_info['coords'])
                noise = np.random.normal(0, 0.1, 3)
                coords = base_coords + noise
                
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                z_coords.append(coords[2])
                activity_values.append(activity_data[ch])
            
            # Add trace
            fig.add_trace(
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=activity_values,
                        colorscale='Viridis',
                        colorbar=dict(title=feature_name)
                    ),
                    name=feature_name,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Interactive 3D Brain Activity Dashboard",
            height=800,
            width=1000
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / "interactive_3d_dashboard.html")
            fig.write_image(save_path / "interactive_3d_dashboard.png")
        
        return fig
    
    def create_brain_network_graph(self, correlation_matrix: np.ndarray,
                                  threshold: float = 0.3,
                                  save_path: Path = None) -> go.Figure:
        """Create brain network graph based on correlations."""
        print("🎨 Creating brain network graph")
        
        # Create network data
        nodes = []
        edges = []
        
        # Add nodes (channels)
        for ch in range(min(correlation_matrix.shape[0], 160)):
            region = self.channel_regions.get(ch, 'central')
            region_info = self.brain_regions[region]
            
            base_coords = np.array(region_info['coords'])
            noise = np.random.normal(0, 0.1, 3)
            coords = base_coords + noise
            
            nodes.append({
                'id': ch,
                'label': f"Ch{ch+1}",
                'x': coords[0],
                'y': coords[1],
                'z': coords[2],
                'color': region_info['color'],
                'region': region
            })
        
        # Add edges (correlations above threshold)
        for i in range(correlation_matrix.shape[0]):
            for j in range(i+1, correlation_matrix.shape[1]):
                if abs(correlation_matrix[i, j]) > threshold:
                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': abs(correlation_matrix[i, j]),
                        'color': 'rgba(0,0,0,0.3)'
                    })
        
        # Create 3D network visualization
        fig = go.Figure()
        
        # Add edges
        for edge in edges:
            source_node = nodes[edge['source']]
            target_node = nodes[edge['target']]
            
            fig.add_trace(go.Scatter3d(
                x=[source_node['x'], target_node['x']],
                y=[source_node['y'], target_node['y']],
                z=[source_node['z'], target_node['z']],
                mode='lines',
                line=dict(
                    color=edge['color'],
                    width=edge['weight'] * 5
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add nodes
        for region, region_info in self.brain_regions.items():
            region_nodes = [n for n in nodes if n['region'] == region]
            
            if region_nodes:
                fig.add_trace(go.Scatter3d(
                    x=[n['x'] for n in region_nodes],
                    y=[n['y'] for n in region_nodes],
                    z=[n['z'] for n in region_nodes],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=region_info['color']
                    ),
                    text=[n['label'] for n in region_nodes],
                    hovertemplate='%{text}<extra></extra>',
                    name=region.title(),
                    showlegend=True
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Brain Network Graph (threshold: {threshold})",
            scene=dict(
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                zaxis_title="Z Coordinate"
            ),
            width=800,
            height=600
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / f"brain_network_threshold_{threshold}.html")
            fig.write_image(save_path / f"brain_network_threshold_{threshold}.png")
        
        return fig
    
    def get_summary_report(self) -> str:
        """Generate a summary report of the 3D visualizations."""
        report = []
        report.append("🎨 3D Brain Visualizer Summary")
        report.append("=" * 50)
        
        report.append(f"📊 Brain Regions: {len(self.brain_regions)}")
        for region, info in self.brain_regions.items():
            report.append(f"   • {region.title()}: {info['coords']}")
        
        report.append(f"\n📊 Channel Mapping: {len(self.channel_regions)} channels")
        report.append(f"📊 Visualization Types:")
        report.append("   • 3D Brain Activity Maps")
        report.append("   • Temporal Evolution Animations")
        report.append("   • Region Comparison Plots")
        report.append("   • Interactive Dashboards")
        report.append("   • Brain Network Graphs")
        
        return "\n".join(report)
