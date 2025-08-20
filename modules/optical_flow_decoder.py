import torch
from torch import nn
import torch.nn.functional as F


class OpticalFlowDecoder(nn.Module):
    """
    Decoder layer that generates optical flow from video transformer encodings.
    
    Input: [B, N, 81, 2048] - Batch, Frames, Spatial tokens, Feature dim
    Output: [B, N-1, H, W, 2] - Optical flow between consecutive frames
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        output_height: int = 224,
        output_width: int = 224,
        spatial_tokens: int = 81,  # Assuming 9x9 spatial grid
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_height = output_height
        self.output_width = output_width
        self.spatial_tokens = spatial_tokens
        self.spatial_size = int(spatial_tokens ** 0.5)  # 9 for 81 tokens
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal-spatial transformer layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Spatial upsampling layers
        self.spatial_decoder = SpatialDecoder(
            hidden_dim, 
            output_height, 
            output_width, 
            self.spatial_size
        )
        
        # Optical flow head
        self.flow_head = FlowHead(hidden_dim, dropout)
        
        # Positional encodings
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, 100, 1, hidden_dim) * 0.02  # Max 100 frames
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, spatial_tokens, hidden_dim) * 0.02
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 81, 2048] - Video transformer encodings
            
        Returns:
            flow: [B, N-1, H, W, 2] - Optical flow between consecutive frames
        """
        B, N, S, D = x.shape
        
        # Project to hidden dimension
        x = self.input_proj(x)  # [B, N, 81, hidden_dim]
        
        # Add positional embeddings
        x = x + self.temporal_pos_embed[:, :N] + self.spatial_pos_embed
        
        # Apply temporal-spatial attention layers
        for layer in self.temporal_layers:
            x = layer(x)
        
        # Generate frame pairs for optical flow
        # We need consecutive frame pairs: (f0,f1), (f1,f2), ..., (fN-2,fN-1)
        flow_pairs = []
        
        for i in range(N - 1):
            frame_curr = x[:, i]      # [B, 81, hidden_dim]
            frame_next = x[:, i + 1]  # [B, 81, hidden_dim]
            
            # Concatenate frame features
            frame_pair = torch.cat([frame_curr, frame_next], dim=-1)  # [B, 81, 2*hidden_dim]
            flow_pairs.append(frame_pair)
        
        # Stack all frame pairs
        flow_pairs = torch.stack(flow_pairs, dim=1)  # [B, N-1, 81, 2*hidden_dim]
        
        # Decode spatial features to full resolution
        flows = []
        for i in range(N - 1):
            pair_features = flow_pairs[:, i]  # [B, 81, 2*hidden_dim]
            
            # Upsample spatial features
            spatial_features = self.spatial_decoder(pair_features)  # [B, hidden_dim, H, W]
            
            # Generate optical flow
            flow = self.flow_head(spatial_features)  # [B, 2, H, W]
            flows.append(flow)
        
        # Stack all flows
        flows = torch.stack(flows, dim=1)  # [B, N-1, 2, H, W]
        flows = flows.permute(0, 1, 3, 4, 2)  # [B, N-1, H, W, 2]
        
        return flows


class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer that processes frame sequences"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, S, D = x.shape
        
        # Temporal attention across frames
        x_temporal = x.view(B * S, N, D)  # Reshape for temporal attention
        x_temporal = x_temporal + self.temporal_attn(
            x_temporal, x_temporal, x_temporal
        )[0]
        x_temporal = self.norm1(x_temporal)
        x_temporal = x_temporal.view(B, N, S, D)
        
        # Spatial attention within frames
        x_spatial = x_temporal.view(B * N, S, D)  # Reshape for spatial attention
        x_spatial = x_spatial + self.spatial_attn(
            x_spatial, x_spatial, x_spatial
        )[0]
        x_spatial = self.norm2(x_spatial)
        x_spatial = x_spatial.view(B, N, S, D)
        
        # Feed-forward network
        x_out = x_spatial + self.ffn(x_spatial)
        x_out = self.norm3(x_out)
        
        return x_out


class SpatialDecoder(nn.Module):
    """Upsamples spatial features from tokens to full resolution"""
    
    def __init__(self, hidden_dim: int, output_height: int, output_width: int, spatial_size: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_height = output_height
        self.output_width = output_width
        self.spatial_size = spatial_size
        
        # Project concatenated features back to hidden_dim
        self.feature_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Convolutional upsampling layers
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),  # 2x upsample
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1),  # 4x upsample
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, 2, 1),  # 8x upsample
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
        )
        
        # Final adjustment layer to match exact output size
        self.final_conv = nn.Conv2d(hidden_dim // 8, hidden_dim, 3, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Project features
        x = self.feature_proj(x)  # [B, 81, hidden_dim]
        
        # Reshape to spatial grid
        x = x.view(B, self.spatial_size, self.spatial_size, self.hidden_dim)
        x = x.permute(0, 3, 1, 2)  # [B, hidden_dim, 9, 9]
        
        # Upsample through conv layers
        x = self.conv_layers(x)  # [B, hidden_dim//8, 72, 72] (9*8=72)
        
        # Final convolution
        x = self.final_conv(x)  # [B, hidden_dim, 72, 72]
        
        # Interpolate to exact output size if needed
        if x.shape[-2:] != (self.output_height, self.output_width):
            x = F.interpolate(
                x, 
                size=(self.output_height, self.output_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        return x


class FlowHead(nn.Module):
    """Generates optical flow from spatial features"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.flow_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(hidden_dim // 4, 2, 3, 1, 1),  # 2 channels for (dx, dy)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow_conv(x)