import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlowFusion(nn.Module):
    def __init__(self, vlm_embed_dim=1152, spatial_channels=128, spatial_size=64):
        super().__init__()
        self.vlm_embed_dim = vlm_embed_dim
        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size
        
        # Project VLM temporal embedding to spatial features
        # VLM already has temporal context from seeing multiple frames
        self.temporal_proj = nn.Sequential(
            nn.Linear(vlm_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, spatial_channels)
        )
        
        # Reshape VLM temporal features to spatial grid
        self.temporal_to_spatial = nn.ConvTranspose2d(
            spatial_channels, spatial_channels, 
            kernel_size=4, stride=2, padding=1  # 27x27 -> 54x54
        )
        
        # Cross-attention between temporal and spatial features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_channels, 
            num_heads=8, 
            batch_first=True
        )
        
        # Temporal-spatial fusion network
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(spatial_channels * 2, spatial_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(spatial_channels, spatial_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(spatial_channels // 2, 64, 3, padding=1),
            nn.ReLU()
        )
        
        # Final optical flow prediction (2 channels for x,y flow)
        self.flow_head = nn.Conv2d(64, 2, 1)
        
    def forward(self, vlm_temporal_embedding, spatial_encoding):
        """
        Args:
            vlm_temporal_embedding: (B, 729, 1152) - VLM embedding for current frame 
                                   (but with temporal context from seeing multiple frames)
            spatial_encoding: (B, 128, 64, 64) - Spatial features for the same current frame
            
        Returns:
            optical_flow: (B, 2, 64, 64) - Flow map (x,y components)
        """
        B, patches, embed_dim = vlm_temporal_embedding.shape
        
        # Project VLM temporal features 
        temporal_features = self.temporal_proj(vlm_temporal_embedding)  # (B, 729, 128)
        
        # Reshape to spatial grid (27x27 patches -> spatial map)
        temporal_spatial = temporal_features.view(B, 27, 27, self.spatial_channels)
        temporal_spatial = temporal_spatial.permute(0, 3, 1, 2)  # (B, 128, 27, 27)
        
        # Upsample temporal features to match spatial encoding size (64x64)
        temporal_upsampled = self.temporal_to_spatial(temporal_spatial)  # (B, 128, 54, 54)
        temporal_upsampled = F.interpolate(
            temporal_upsampled, size=self.spatial_size, 
            mode='bilinear', align_corners=False
        )  # (B, 128, 64, 64)
        
        # Cross-attention between temporal and spatial features
        # Flatten spatial dimensions for attention
        temporal_flat = temporal_upsampled.flatten(2).transpose(1, 2)  # (B, 4096, 128)
        spatial_flat = spatial_encoding.flatten(2).transpose(1, 2)  # (B, 4096, 128)
        
        # Spatial features attend to temporal features
        # This allows spatial details to focus on relevant temporal/motion cues
        attended_spatial, _ = self.cross_attention(
            spatial_flat, temporal_flat, temporal_flat
        )  # (B, 4096, 128)
        
        # Reshape back to spatial
        attended_spatial = attended_spatial.transpose(1, 2).view(
            B, self.spatial_channels, self.spatial_size, self.spatial_size
        )
        
        # Fuse temporal and attended spatial features
        fused = torch.cat([temporal_upsampled, attended_spatial], dim=1)  # (B, 256, 64, 64)
        
        # Apply fusion convolutions
        fused_features = self.fusion_conv(fused)  # (B, 64, 64, 64)
        
        # Predict optical flow for this frame
        optical_flow = self.flow_head(fused_features)  # (B, 2, 64, 64)
        
        return optical_flow

# Usage example for processing frames one at a time:
def process_video_sequence():
    """Example of processing a video sequence frame by frame"""
    
    flow_fusion = OpticalFlowFusion()
    batch_size = 4
    
    # Simulate VLM processing multiple frames and outputting temporal embeddings
    # Each frame gets its own embedding but with temporal context
    vlm_sequence_embeddings = torch.randn(batch_size, 5, 729, 1152)  # 5 frames
    spatial_sequence_encodings = torch.randn(batch_size, 5, 128, 64, 64)  # 5 frames
    
    optical_flows = []
    
    # Process each frame individually
    for frame_idx in range(5):
        # Current frame's VLM embedding (already contains temporal context)
        current_vlm_embedding = vlm_sequence_embeddings[:, frame_idx]  # (B, 729, 1152)
        
        # Current frame's spatial encoding
        current_spatial_encoding = spatial_sequence_encodings[:, frame_idx]  # (B, 128, 64, 64)
        
        # Predict optical flow for this frame
        flow = flow_fusion(current_vlm_embedding, current_spatial_encoding)
        optical_flows.append(flow)
        
        print(f"Frame {frame_idx} flow shape: {flow.shape}")
    
    return torch.stack(optical_flows, dim=1)  # (B, 5, 2, 64, 64)

# Training loss function
class OpticalFlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_flow, gt_flow):
        # L1 loss for optical flow
        l1_loss = F.l1_loss(pred_flow, gt_flow)
        
        # Smoothness regularization
        dx = pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1]
        dy = pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :]
        smoothness_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        
        return l1_loss + 0.1 * smoothness_loss