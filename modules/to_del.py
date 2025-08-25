import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def rope_2d(embeddings: torch.Tensor, positions: torch.Tensor, d_model: int):
    """
    Apply 2D RoPE to embeddings based on 2D positions.
    
    Args:
        embeddings: [B, N, D] - embeddings to apply RoPE to (can be queries or keys)
        positions: [B, N, 2] - 2D coordinates (x, y) 
        d_model: dimension of embeddings
    
    Returns: [B, N, D] RoPE applied embeddings
    """
    B, N, D = embeddings.shape
    assert D % 4 == 0, "Embedding dim must be divisible by 4"

    half = D // 2
    quarter = D // 4

    # Split into x- and y- halves
    x_part, y_part = embeddings[:, :, :half], embeddings[:, :, half:]

    # Frequencies (like classic RoPE)
    idx = torch.arange(quarter, dtype=embeddings.dtype, device=embeddings.device)
    theta = 1.0 / (10000 ** (2 * idx / half))   # [quarter]

    # Positions
    pos_x = positions[..., 0].float()  # [B, N]
    pos_y = positions[..., 1].float()  # [B, N]

    # Compute angles
    freqs_x = torch.einsum('bn,q->bnq', pos_x, theta)  # [B, N, quarter]
    freqs_y = torch.einsum('bn,q->bnq', pos_y, theta)  # [B, N, quarter]

    # cos/sin
    cos_x, sin_x = freqs_x.cos(), freqs_x.sin()
    cos_y, sin_y = freqs_y.cos(), freqs_y.sin()

    # Reshape to interleave pairs and apply rotation
    def apply_rope(part, cos, sin):
        part = part.view(B, N, quarter, 2)
        x1, x2 = part[..., 0], part[..., 1]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).view(B, N, -1)

    x_out = apply_rope(x_part, cos_x, sin_x)
    y_out = apply_rope(y_part, cos_y, sin_y)

    return torch.cat([x_out, y_out], dim=-1)


def create_2d_positional_embedding(positions: torch.Tensor, d_model: int):
    """
    Create 2D positional embeddings using RoPE mechanism.
    This replaces the linear layer approach.
    
    Args:
        positions: [N, 2] or [B, N, 2] - 2D coordinates
        d_model: embedding dimension
        
    Returns:
        embeddings: [N, d_model] or [B, N, d_model] - positional embeddings
    """
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False
        
    B, N, _ = positions.shape
    
    # Initialize zero embeddings
    embeddings = torch.zeros(B, N, d_model, device=positions.device, dtype=positions.dtype)
    
    # Apply RoPE to get positional embeddings
    embeddings = rope_2d(embeddings, positions, d_model)
    
    if squeeze_output:
        embeddings = embeddings.squeeze(0)
        
    return embeddings


class QKNormalization(nn.Module):
    """Query-Key normalization to stabilize attention computation."""
    
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-6
        self.q_norm = nn.LayerNorm(d_model, eps=1e-6, bias=False)
        self.k_norm = nn.LayerNorm(d_model, eps=1e-6, bias=False)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize queries and keys."""
        return self.q_norm(q), self.k_norm(k)


class AttentionalSampling(nn.Module):
    """Attentional Sampling module that pools information from video features to track tokens."""
    
    def __init__(self, d_model: int, d_k: int, num_heads: int = 8, sigma: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads
        self.sigma = sigma
        self.head_dim = d_k // num_heads
        
        # Projection layers
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # QK normalization
        self.qk_norm = QKNormalization(d_k)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, features: torch.Tensor, tracks: torch.Tensor, 
                track_pos_embeddings: torch.Tensor, feature_pos_embeddings: torch.Tensor,
                feature_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [T, HW, d_model] - flattened video features
            tracks: [T, M, 2] - track positions (x, y coordinates)
            track_pos_embeddings: [T, M, d_model] - positional embeddings for tracks
            feature_pos_embeddings: [T, HW, d_model] - positional embeddings for features
            feature_positions: [HW, 2] - spatial positions of feature tokens
            
        Returns:
            sampled_features: [T, M, d_model] - features sampled at track locations
        """
        T, HW, _ = features.shape
        _, M, _ = tracks.shape
        
        # Create queries from track positional embeddings
        Q = self.W_q(track_pos_embeddings)  # [T, M, d_k]
        
        # Create keys from feature embeddings + positional embeddings  
        feature_with_pos = features + feature_pos_embeddings
        K = self.W_k(feature_with_pos)  # [T, HW, d_k]
        
        # Values are just the features
        V = self.W_v(features)  # [T, HW, d_model]
        
        # Apply QK normalization
        Q, K = self.qk_norm(Q, K)
        
        # Reshape for multi-head attention
        Q = Q.view(T, M, self.num_heads, self.head_dim).transpose(1, 2)      # [T, num_heads, M, head_dim]
        K = K.view(T, HW, self.num_heads, self.head_dim).transpose(1, 2)     # [T, num_heads, HW, head_dim]
        V = V.view(T, HW, self.num_heads, -1).transpose(1, 2)                # [T, num_heads, HW, d_model//num_heads]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [T, num_heads, M, HW]
        
        # Add spatial bias
        bias = torch.zeros_like(scores)
        for t in range(T):
            track_pos = tracks[t]  # [M, 2]
            feat_pos = feature_positions  # [HW, 2]
            distances = torch.cdist(track_pos, feat_pos, p=2)  # [M, HW]
            spatial_bias = -distances.pow(2) / (2 * self.sigma**2)  # [M, HW]
            bias[t] = spatial_bias.unsqueeze(0).expand(self.num_heads, -1, -1)  # [num_heads, M, HW]

        scores = scores + bias
        attn_weights = F.softmax(scores, dim=-1)  # [T, num_heads, M, HW]

        # Apply attention
        sampled = torch.matmul(attn_weights, V)  # [T, num_heads, M, d_model//num_heads]
        sampled = sampled.transpose(1, 2).contiguous().view(T, M, -1)  # [T, M, d_model]

        return self.out_proj(sampled)


class TrackTransformer(nn.Module):
    """Track Transformer that processes track tokens temporally."""
    
    def __init__(self, d_model: int, num_heads: int = 8, num_layers: int = 2, 
                 dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Sinusoidal positional encoding for temporal dimension
        self.pos_encoding = self._create_positional_encoding(d_model, max_len=1000)
        
    def _create_positional_encoding(self, d_model: int, max_len: int = 1000) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, track_tokens: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Args:
            track_tokens: [B*T, M, d_model] - track tokens from attentional sampling
            B: batch size
            T: temporal dimension
            
        Returns:
            updated_track_tokens: [B*T, M, d_model] - temporally updated track tokens
        """
        BT, M, d_model = track_tokens.shape
        
        # Reshape to separate batch and temporal dimensions
        track_tokens = track_tokens.view(B, T, M, d_model)
        track_tokens = track_tokens.permute(0, 2, 1, 3)  # [B, M, T, d_model]
        track_tokens_reshaped = track_tokens.reshape(B * M, T, d_model)
        
        # Add temporal positional encoding
        pos_enc = self.pos_encoding[:T, :].unsqueeze(0).expand(B*M, -1, -1).to(track_tokens.device)
        track_tokens_with_pos = track_tokens_reshaped + pos_enc
        
        # Process each track independently using transformer
        updated_tokens = self.transformer(track_tokens_with_pos)  # [B*M, T, d_model]
        
        # Reshape back to original format
        updated_tokens = updated_tokens.view(B, M, T, d_model)  # [B, M, T, d_model]
        updated_tokens = updated_tokens.permute(0, 2, 1, 3)     # [B, T, M, d_model]
        updated_tokens = updated_tokens.reshape(BT, M, d_model) # [B*T, M, d_model]
        
        return updated_tokens


class AttentionalSplatting(nn.Module):
    """Attentional Splatting module that maps updated track tokens back to feature maps."""
    
    def __init__(self, d_model: int, d_k: int, num_heads: int = 8, sigma: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads
        self.sigma = sigma
        self.head_dim = d_k // num_heads
        
        # Projection layers (reversed roles compared to sampling)
        self.W_q = nn.Linear(d_model, d_k, bias=False)  # Now for grid coordinates
        self.W_k = nn.Linear(d_model, d_k, bias=False)  # Now for track tokens
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Values from tracks
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        
        # QK normalization
        self.qk_norm = QKNormalization(d_k)
        
        # Initialize output projection to zero for residual connection
        nn.init.zeros_(self.W_out.weight)
        
    def forward(self, updated_track_tokens: torch.Tensor, tracks: torch.Tensor,
                feature_positions: torch.Tensor, feature_pos_embeddings: torch.Tensor,
                track_pos_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            updated_track_tokens: [T, M, d_model] - updated track tokens from Track Transformer
            tracks: [T, M, 2] - track positions
            feature_positions: [HW, 2] - spatial positions of feature tokens
            feature_pos_embeddings: [T, HW, d_model] - positional embeddings for features
            track_pos_embeddings: [T, M, d_model] - positional embeddings for tracks
            
        Returns:
            feature_updates: [T, HW, d_model] - updates to be added to original features
        """
        T, M, _ = updated_track_tokens.shape
        HW = feature_positions.shape[0]
        
        # Create queries from feature positional embeddings
        Q = self.W_q(feature_pos_embeddings)  # [T, HW, d_k] - queries from grid
        
        # Create keys from track positional embeddings
        K = self.W_k(track_pos_embeddings)    # [T, M, d_k] - keys from tracks
        
        # Values are the updated track tokens
        V = self.W_v(updated_track_tokens)    # [T, M, d_model] - values from tracks
        
        # Apply QK normalization
        Q, K = self.qk_norm(Q, K)
        
        # Reshape for multi-head attention
        Q = Q.view(T, HW, self.num_heads, self.head_dim).transpose(1, 2)     # [T, num_heads, HW, head_dim]
        K = K.view(T, M, self.num_heads, self.head_dim).transpose(1, 2)      # [T, num_heads, M, head_dim]
        V = V.view(T, M, self.num_heads, -1).transpose(1, 2)                 # [T, num_heads, M, d_model//num_heads]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [T, num_heads, HW, M]
        
        # Add spatial bias (transposed from sampling)
        bias = torch.zeros_like(scores)
        for t in range(T):
            feat_pos = feature_positions  # [HW, 2]
            track_pos = tracks[t]         # [M, 2]
            distances = torch.cdist(feat_pos, track_pos, p=2)  # [HW, M]
            spatial_bias = -distances.pow(2) / (2 * self.sigma**2)  # [HW, M]
            bias[t] = spatial_bias.unsqueeze(0).expand(self.num_heads, -1, -1)  # [num_heads, HW, M]
        
        scores = scores + bias
        attn_weights = F.softmax(scores, dim=-1)  # [T, num_heads, HW, M]
        
        # Apply attention
        updates = torch.matmul(attn_weights, V)  # [T, num_heads, HW, d_model//num_heads]
        updates = updates.transpose(1, 2).contiguous().view(T, HW, -1)  # [T, HW, d_model]
        
        # Apply output projection
        feature_updates = self.W_out(updates)
        
        return feature_updates


class TracktentionLayer(nn.Module):
    """Complete Tracktention Layer combining all components."""
    
    def __init__(self, d_model: int, d_k: Optional[int] = None, num_heads: int = 8, 
                 num_transformer_layers: int = 2, sigma: float = 0.5, dropout: float = 0.0,
                 patch_size: int = 14):
        super().__init__()
        
        if d_k is None:
            d_k = d_model
            
        self.d_model = d_model
        self.d_k = d_k
        self.patch_size = patch_size
        
        # Main components
        self.attentional_sampling    = AttentionalSampling(d_model, d_k, num_heads, sigma)
        self.track_transformer       = TrackTransformer(d_model, num_heads, num_transformer_layers, dropout)
        self.attentional_splatting   = AttentionalSplatting(d_model, d_k, num_heads, sigma)
        
    def create_feature_positions(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create grid positions for feature map tokens accounting for patch size."""
        # Create grid coordinates in patch space, then scale to actual image coordinates
        y, x = torch.meshgrid(torch.arange(H, device=device), 
                             torch.arange(W, device=device), indexing='ij')
        
        # Scale by patch size to get actual image coordinates
        # Assuming patches are centered, add half patch size for center coordinates
        x = x.flatten().float() * self.patch_size + self.patch_size // 2
        y = y.flatten().float() * self.patch_size + self.patch_size // 2
        
        positions = torch.stack([x, y], dim=-1)  # [HW, 2]
        return positions
    
    def forward(self, features: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B*T, H, W, d_model] - input video features
            tracks: [B, T, M, 2] - point tracks (x, y coordinates in image space)
            
        Returns:
            output: [B*T, H, W, d_model] - updated features with temporal consistency
        """
        BT, H, W, d_model = features.shape
        B, T, M, _ = tracks.shape
        
        # Flatten spatial dimensions
        features_flat = features.view(BT, H * W, d_model)  # [B*T, HW, d_model]
        tracks_flat = tracks.view(BT, M, 2)  # [B*T, M, 2]
        
        # Create feature positions (accounting for patch size)
        feature_positions = self.create_feature_positions(H, W, features.device)  # [HW, 2]
        
        # Create positional embeddings using RoPE mechanism instead of linear layer
        feature_pos_expanded = feature_positions.unsqueeze(0).expand(BT, -1, -1)  # [BT, HW, 2]
        feature_pos_embeddings = create_2d_positional_embedding(feature_pos_expanded, d_model)  # [BT, HW, d_model]
        
        track_pos_embeddings = create_2d_positional_embedding(tracks_flat.unsqueeze(0), d_model).squeeze(0)  # [BT, M, d_model]
        
        # 1. Attentional Sampling: pool information from features to tracks
        sampled_features = self.attentional_sampling(
            features_flat, tracks_flat, track_pos_embeddings, feature_pos_embeddings, feature_positions
        )  # [BT, M, d_model]
        
        # 2. Track Transformer: process tracks temporally
        updated_track_tokens = self.track_transformer(sampled_features, B, T)  # [BT, M, d_model]
        
        # 3. Attentional Splatting: map track tokens back to features
        feature_updates = self.attentional_splatting(
            updated_track_tokens, tracks_flat, feature_positions, feature_pos_embeddings, track_pos_embeddings
        )  # [BT, HW, d_model]
        
        # Add residual connection
        updated_features = features_flat + feature_updates
        
        # Reshape back to original dimensions
        output = updated_features.view(BT, H, W, d_model)
        
        return output