#!/usr/bin/env python3
"""
Temporal Attention Transformer with Brain Region Embeddings
IEEE-SMC-2025 ECoG Video Analysis Competition

This module implements a sophisticated transformer model that combines
temporal attention with spatial brain region information.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class BrainRegionEmbedding(nn.Module):
    """Brain region embedding layer using spatial coordinates."""
    
    def __init__(self, num_regions: int, embedding_dim: int):
        super().__init__()
        self.num_regions = num_regions
        self.embedding_dim = embedding_dim
        
        # Learnable region embeddings
        self.region_embeddings = nn.Embedding(num_regions, embedding_dim)
        
        # Spatial coordinate embeddings (x, y, z)
        self.spatial_embeddings = nn.Linear(3, embedding_dim)
        
    def forward(self, region_ids: torch.Tensor, spatial_coords: torch.Tensor):
        """Forward pass with region and spatial embeddings."""
        # Region embeddings
        region_emb = self.region_embeddings(region_ids)
        
        # Spatial embeddings
        spatial_emb = self.spatial_embeddings(spatial_coords)
        
        # Combine embeddings
        combined_emb = region_emb + spatial_emb
        
        return combined_emb

class TemporalAttentionBlock(nn.Module):
    """Temporal attention block for sequence modeling."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Forward pass with self-attention and feed-forward."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class SpatialTemporalTransformer(nn.Module):
    """Spatial-temporal transformer for ECoG data."""
    
    def __init__(self, 
                 num_channels: int,
                 num_regions: int,
                 sequence_length: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 512,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_regions = num_regions
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)  # Each time point -> d_model
        
        # Brain region embeddings
        self.brain_embeddings = BrainRegionEmbedding(num_regions, d_model)
        
        # Positional encoding for temporal dimension
        self.pos_encoding = self._create_positional_encoding(sequence_length, d_model)
        
        # Temporal attention blocks
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Spatial attention (across channels)
        self.spatial_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, x: torch.Tensor, region_ids: torch.Tensor, 
                spatial_coords: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer."""
        batch_size, num_channels, seq_len = x.shape
        
        # Project input to d_model
        x = x.unsqueeze(-1)  # Add feature dimension
        x = self.input_projection(x)  # (batch, channels, seq_len, d_model)
        
        # Add brain region embeddings
        region_emb = self.brain_embeddings(region_ids, spatial_coords)
        x = x + region_emb.unsqueeze(2)  # Broadcast across time
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].unsqueeze(1)
        
        # Reshape for temporal attention: (batch, channels, seq_len, d_model) -> (batch*channels, seq_len, d_model)
        x = x.view(batch_size * num_channels, seq_len, self.d_model)
        
        # Apply temporal attention blocks
        for block in self.temporal_blocks:
            x = block(x)
        
        # Reshape back: (batch*channels, seq_len, d_model) -> (batch, channels, seq_len, d_model)
        x = x.view(batch_size, num_channels, seq_len, self.d_model)
        
        # Global average pooling across time
        x = x.mean(dim=2)  # (batch, channels, d_model)
        
        # Spatial attention across channels
        attn_out, _ = self.spatial_attention(x, x, x)
        x = self.spatial_norm(x + attn_out)
        
        # Global average pooling across channels
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

class ECoGDataset(Dataset):
    """Dataset class for ECoG data with brain region information."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 region_ids: np.ndarray, spatial_coords: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.region_ids = torch.LongTensor(region_ids)
        self.spatial_coords = torch.FloatTensor(spatial_coords)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'labels': self.labels[idx],
            'region_ids': self.region_ids[idx],
            'spatial_coords': self.spatial_coords[idx]
        }

class TemporalAttentionModel:
    """Temporal attention transformer model for ECoG classification."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the temporal attention model."""
        self.config = config or {}
        
        # Model parameters
        self.d_model = self.config.get('d_model', 128)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.d_ff = self.config.get('d_ff', 512)
        self.dropout = self.config.get('dropout', 0.1)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epochs = self.config.get('epochs', 100)
        
        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = StandardScaler()
        
        # Training history
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
    
    def prepare_data(self, transformer_features: Dict[str, Any], 
                    brain_atlas: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for the temporal attention model."""
        print("🔧 Preparing data for temporal attention model")
        
        # Get transformer features
        if 'transformer_input' in transformer_features:
            data = transformer_features['transformer_input']
        elif 'extended_sequences' in transformer_features:
            data = transformer_features['extended_sequences']
        else:
            raise ValueError("No suitable transformer features found")
        
        # Get labels
        if 'labels' in transformer_features:
            labels = transformer_features['labels']
        else:
            # Create dummy labels if not available
            labels = np.zeros(data.shape[0])
        
        # Get brain region information
        if hasattr(brain_atlas, 'channel_to_region'):
            region_ids = []
            spatial_coords = []
            
            for ch in range(data.shape[1]):  # Assuming second dimension is channels
                region_id = brain_atlas.channel_to_region.get(ch, 0)
                region_ids.append(region_id)
                
                # Get spatial coordinates (dummy for now)
                spatial_coords.append([0.0, 0.0, 0.0])
            
            region_ids = np.array(region_ids)
            spatial_coords = np.array(spatial_coords)
        else:
            # Create dummy region information
            region_ids = np.zeros(data.shape[1], dtype=int)
            spatial_coords = np.zeros((data.shape[1], 3))
        
        print(f"   📊 Data shape: {data.shape}")
        print(f"   📊 Labels shape: {labels.shape}")
        print(f"   📊 Region IDs shape: {region_ids.shape}")
        print(f"   📊 Spatial coords shape: {spatial_coords.shape}")
        
        return data, labels, region_ids, spatial_coords
    
    def create_model(self, num_channels: int, num_regions: int, 
                    sequence_length: int, num_classes: int = 2):
        """Create the temporal attention model."""
        print("🔧 Creating temporal attention model")
        
        self.model = SpatialTemporalTransformer(
            num_channels=num_channels,
            num_regions=num_regions,
            sequence_length=sequence_length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            num_classes=num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"   📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            data = batch['data'].to(self.device)
            labels = batch['labels'].to(self.device)
            region_ids = batch['region_ids'].to(self.device)
            spatial_coords = batch['spatial_coords'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data, region_ids, spatial_coords)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                data = batch['data'].to(self.device)
                labels = batch['labels'].to(self.device)
                region_ids = batch['region_ids'].to(self.device)
                spatial_coords = batch['spatial_coords'].to(self.device)
                
                outputs = self.model(data, region_ids, spatial_coords)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, transformer_features: Dict[str, Any], 
              brain_atlas: Any, val_split: float = 0.2) -> Dict[str, Any]:
        """Train the temporal attention model."""
        print("🎯 Training Temporal Attention Transformer")
        print("=" * 50)
        
        # Prepare data
        data, labels, region_ids, spatial_coords = self.prepare_data(transformer_features, brain_atlas)
        
        # Create model
        self.create_model(
            num_channels=data.shape[1],
            num_regions=len(np.unique(region_ids)),
            sequence_length=data.shape[2],
            num_classes=len(np.unique(labels))
        )
        
        # Train/validation split
        train_data, val_data, train_labels, val_labels, train_region_ids, val_region_ids, train_spatial_coords, val_spatial_coords = train_test_split(
            data, labels, region_ids, spatial_coords, test_size=val_split, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = ECoGDataset(train_data, train_labels, train_region_ids, train_spatial_coords)
        val_dataset = ECoGDataset(val_data, val_labels, val_region_ids, val_spatial_coords)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Store history
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"   📊 Epoch {epoch+1}/{self.epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_accuracy': best_val_acc
        }
    
    def predict(self, transformer_features: Dict[str, Any], 
                brain_atlas: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with the trained model."""
        print("🔮 Making predictions with temporal attention model")
        
        # Prepare data
        data, labels, region_ids, spatial_coords = self.prepare_data(transformer_features, brain_atlas)
        
        # Create dataset
        dataset = ECoGDataset(data, labels, region_ids, spatial_coords)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Predict
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                data_batch = batch['data'].to(self.device)
                region_ids_batch = batch['region_ids'].to(self.device)
                spatial_coords_batch = batch['spatial_coords'].to(self.device)
                
                outputs = self.model(data_batch, region_ids_batch, spatial_coords_batch)
                probs = F.softmax(outputs, dim=1)
                
                _, pred = torch.max(outputs, 1)
                
                predictions.extend(pred.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def save_model(self, save_path: Path):
        """Save the trained model."""
        print(f"💾 Saving temporal attention model to {save_path}")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, save_path / 'temporal_attention_model.pth')
        
        print("✅ Model saved!")
    
    def load_model(self, load_path: Path):
        """Load a trained model."""
        print(f"📂 Loading temporal attention model from {load_path}")
        
        checkpoint = torch.load(load_path / 'temporal_attention_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        print("✅ Model loaded!")
    
    def get_attention_weights(self, transformer_features: Dict[str, Any], 
                            brain_atlas: Any) -> Dict[str, np.ndarray]:
        """Extract attention weights for visualization."""
        print("🔍 Extracting attention weights")
        
        # This would require modifying the model to return attention weights
        # For now, return dummy weights
        attention_weights = {
            'temporal_attention': np.random.rand(8, 100, 100),  # n_heads, seq_len, seq_len
            'spatial_attention': np.random.rand(8, 156, 156)   # n_heads, channels, channels
        }
        
        return attention_weights
    
    def get_summary_report(self) -> str:
        """Generate a summary report of the model."""
        report = []
        report.append("🎯 Temporal Attention Transformer Summary")
        report.append("=" * 50)
        
        if self.model is not None:
            report.append(f"📊 Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            report.append(f"📊 Model Architecture: {self.d_model}d_model, {self.n_heads}heads, {self.n_layers}layers")
            report.append(f"📊 Training Epochs: {len(self.train_history['loss'])}")
            
            if self.val_history['accuracy']:
                best_val_acc = max(self.val_history['accuracy'])
                report.append(f"📊 Best Validation Accuracy: {best_val_acc:.4f}")
        
        return "\n".join(report)
