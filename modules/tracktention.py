
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


def rope_2d(keys: torch.Tensor, positions: torch.Tensor):
    """
    keys: [B, N, D]   (e.g., [20, 729, 1152])
    positions: [B, N, 2]  integer (x, y) coords
    
    Returns: [B, N, D] RoPE applied
    """
    B, N, D = keys.shape
    assert D % 4 == 0, "Embedding dim must be divisible by 4"

    half = D // 2
    quarter = D // 4

    # Split into x- and y- halves
    x_part, y_part = keys[:, :, :half], keys[:, :, half:]

    # Frequencies (like classic RoPE)
    idx = torch.arange(quarter, dtype=keys.dtype, device=keys.device)
    theta = 1.0 / (10000 ** (2 * idx / half))   # [quarter]

    # Positions
    pos_x = positions[..., 0]  # [B, N]
    pos_y = positions[..., 1]  # [B, N]

    # Compute angles
    freqs_x = torch.einsum('bn,q->bnq', pos_x, theta)  # [B, N, quarter]
    freqs_y = torch.einsum('bn,q->bnq', pos_y, theta)  # [B, N, quarter]

    # cos/sin
    cos_x, sin_x = freqs_x.cos(), freqs_x.sin()
    cos_y, sin_y = freqs_y.cos(), freqs_y.sin()

    # Reshape to interleave pairs
    def apply_rope(part, cos, sin):
        part = part.view(B, N, quarter, 2)
        x1, x2 = part[..., 0], part[..., 1]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).view(B, N, -1)

    x_out = apply_rope(x_part, cos_x, sin_x)
    y_out = apply_rope(y_part, cos_y, sin_y)

    return torch.cat([x_out, y_out], dim=-1)


class RoPEEmbedding(nn.Module):
    """Rotational Position Embedding for 2D coordinates."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Create inverse frequency tensor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model // 4, 2).float() / (d_model // 2)))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to features based on 2D positions.
        
        Args:
            features: [batch_size, seq_len, d_model]
            positions: [batch_size, seq_len, 2] - (x, y) coordinates
            
        Returns:
            features_with_rope: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = features.shape
        x_pos = positions[:, :, 0]  # [batch_size, seq_len]
        y_pos = positions[:, :, 1]  # [batch_size, seq_len]
        
        # Split features into x and y components
        d_half = self.d_model // 2
        d_quarter = d_half // 2
        
        features_x = features[:, :, :d_half]  # First half for x encoding
        features_y = features[:, :, d_half:]  # Second half for y encoding
        
        # Apply RoPE to x components
        features_x_reshaped = features_x.reshape(batch_size, seq_len, d_quarter, 2)
        cos_x = torch.cos(x_pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0))
        sin_x = torch.sin(x_pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0))
        
        features_x_rope = torch.zeros_like(features_x_reshaped)
        features_x_rope[:, :, :, 0] = features_x_reshaped[:, :, :, 0] * cos_x - features_x_reshaped[:, :, :, 1] * sin_x
        features_x_rope[:, :, :, 1] = features_x_reshaped[:, :, :, 0] * sin_x + features_x_reshaped[:, :, :, 1] * cos_x
        features_x_rope = features_x_rope.reshape(batch_size, seq_len, d_half)
        
        # Apply RoPE to y components
        features_y_reshaped = features_y.reshape(batch_size, seq_len, d_quarter, 2)
        cos_y = torch.cos(y_pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0))
        sin_y = torch.sin(y_pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0))
        
        features_y_rope = torch.zeros_like(features_y_reshaped)
        features_y_rope[:, :, :, 0] = features_y_reshaped[:, :, :, 0] * cos_y - features_y_reshaped[:, :, :, 1] * sin_y
        features_y_rope[:, :, :, 1] = features_y_reshaped[:, :, :, 0] * sin_y + features_y_reshaped[:, :, :, 1] * cos_y
        features_y_rope = features_y_rope.reshape(batch_size, seq_len, d_half)
        
        return torch.cat([features_x_rope, features_y_rope], dim=-1)




class QKNormalization(nn.Module):
    """Query-Key normalization to stabilize attention computation."""
    
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-6
        self.q_norm = nn.LayerNorm(d_model, eps=1e-6,bias=False, dtype= torch.bfloat16)
        self.k_norm = nn.LayerNorm(d_model, eps=1e-6,bias=False, dtype= torch.bfloat16)
        
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
        
        # RoPE and normalization
        self.rope = RoPEEmbedding(d_k)
        self.qk_norm = QKNormalization(d_k)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        
    def forward(self, features: torch.Tensor, tracks: torch.Tensor, 
                track_tokens: torch.Tensor, feature_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [T, HW, d_model] - flattened video features
            tracks: [T, M, 2] - track positions (x, y coordinates)
            track_tokens: [T, M, d_model] - track token embeddings
            feature_positions: [HW, 2] - spatial positions of feature tokens
            
        Returns:
            sampled_features: [T, M, d_model] - features sampled at track locations
        """

        #print(features.shape)
        #print(tracks.shape)
        #print(track_tokens.shape)
        #print(feature_positions)
        

        


        T, HW, _ = features.shape
        _, M, _ = tracks.shape
        
        # Project to queries and keys
        Q = self.W_q(track_tokens)  # [T, M, d_k]
        K = self.W_k(features)      # [T, HW, d_k]
        
        # Apply RoPE to keys using feature positions
        feature_pos_expanded = feature_positions.unsqueeze(0).expand(T, -1, -1)  # [T, HW, 2]

       

        K = rope_2d(K, feature_pos_expanded)
        
        
        # Apply QK normalization
        Q, K = self.qk_norm(Q, K)
        
        # Reshape for multi-head attention
        Q = Q.view(T, M, self.num_heads, self.head_dim).transpose(1, 2)      # [T, num_heads, M, head_dim]
        K = K.view(T, HW, self.num_heads, self.head_dim).transpose(1, 2)     # [T, num_heads, HW, head_dim]
        V = features.view(T, HW, self.num_heads, -1).transpose(1, 2)         # [T, num_heads, HW, d_model//num_heads]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [num_heads, M, HW]

        bias = torch.zeros_like(scores)
        for t in range(T):
            track_pos = tracks[t]  # [M, 2]
            feat_pos = feature_positions  # [HW, 2]
            distances = torch.cdist(track_pos, feat_pos, p=2)  # [M, HW]
            bias[t] = -distances.pow(2) / (2 * self.sigma**2)    # [M, HW]


    
        scores = scores + bias
        attn_weights = F.softmax(scores, dim=-1)  # [num_heads, M, HW]

        sampled = torch.matmul(attn_weights, V)  # [num_heads, M, d_model//num_heads]
        sampled = sampled.transpose(1, 2).contiguous().view(T, M, -1)  # [M, d_model]

        res = self.out_proj(sampled)
        return res
    




        
        print(scores.shape)
        exit(1)

        sampled_features_list = []
        
        for t in range(T):
            # Compute attention scores
            scores = torch.matmul(Q[t], K[t].transpose(-2, -1)) / math.sqrt(self.head_dim)  # [num_heads, M, HW]
            
            # Compute spatial bias term
            track_pos = tracks[t]  # [M, 2]
            feat_pos = feature_positions  # [HW, 2]
            
            # Calculate distance-based bias
            distances = torch.cdist(track_pos, feat_pos, p=2)  # [M, HW]
            bias = -distances.pow(2) / (2 * self.sigma**2)    # [M, HW]
            bias = bias.unsqueeze(0).expand(self.num_heads, -1, -1)  # [num_heads, M, HW]
            
            # Add bias and apply softmax
            scores = scores + bias
            attn_weights = F.softmax(scores, dim=-1)  # [num_heads, M, HW]
            
            # Apply attention to values
            sampled = torch.matmul(attn_weights, V[t])  # [num_heads, M, d_model//num_heads]
            sampled = sampled.transpose(0, 1).contiguous().view(M, -1)  # [M, d_model]
            sampled_features_list.append(sampled)
        
        return torch.stack(sampled_features_list, dim=0)  # [T, M, d_model]s






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
    
    def forward(self, track_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            track_tokens: [T, M, d_model] - track tokens from attentional sampling
            
        Returns:
            updated_track_tokens: [T, M, d_model] - temporally updated track tokens
        """
        T, M, d_model = track_tokens.shape
        
        # Swap dimensions: [T, M, d_model] -> [M, T, d_model]
        # This allows us to process each track's temporal sequence independently
        track_tokens_swapped = track_tokens.transpose(0, 1)  # [M, T, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:T, :].unsqueeze(0).expand(M, -1, -1).to(track_tokens.device)
        track_tokens_with_pos = track_tokens_swapped + pos_enc
        
        # Process each track independently using transformer
        # Reshape to batch dimension for parallel processing
        track_tokens_flat = track_tokens_with_pos.reshape(M, T, d_model)
        updated_tokens = self.transformer(track_tokens_flat)  # [M, T, d_model]
        
        # Swap dimensions back: [M, T, d_model] -> [T, M, d_model]
        updated_track_tokens = updated_tokens.transpose(0, 1)
        
        return updated_track_tokens


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
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE and normalization
        self.rope = RoPEEmbedding(d_k)
        self.qk_norm = QKNormalization(d_k)
        
        # Initialize output projection to zero
        nn.init.zeros_(self.W_out.weight)
        

        
    def forward(self, updated_track_tokens: torch.Tensor, tracks: torch.Tensor,
                feature_positions: torch.Tensor, original_features: torch.Tensor, grid_coords_tokens: torch.Tensor,) -> torch.Tensor:
        """
        Args:
            updated_track_tokens: [T, M, d_model] - updated track tokens from Track Transformer
            tracks: [T, M, 2] - track positions
            feature_positions: [HW, 2] - spatial positions of feature tokens
            original_features: [T, HW, d_model] - original feature maps
            
        Returns:
            updated_features: [T, HW, d_model] - features updated with track information
        """
        T, HW, _ = original_features.shape
        _, M, _ = updated_track_tokens.shape
        
        # Create dummy grid tokens for querying (using original features)
        grid_tokens = grid_coords_tokens  # [T, HW, d_model]
        
        # Project to queries and keys (roles reversed from sampling)
        Q = self.W_q(grid_tokens)            # [T, HW, d_k] - queries from grid
        K = self.W_k(updated_track_tokens)   # [T, M, d_k] - keys from tracks
        V = updated_track_tokens             # [T, M, d_model] - values from tracks
        

        
      
        K = rope_2d(K, tracks)
        V = rope_2d(V, tracks)

        
        # Apply QK normalization
        Q, K = self.qk_norm(Q, K)
        
        # Reshape for multi-head attention
        Q = Q.view(T, HW, self.num_heads, self.head_dim).transpose(1, 2)     # [T, num_heads, HW, head_dim]
        K = K.view(T, M, self.num_heads, self.head_dim).transpose(1, 2)      # [T, num_heads, M, head_dim]
        V = V.view(T, M, self.num_heads, -1).transpose(1, 2)                 # [T, num_heads, M, d_model//num_heads]
        

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [num_heads, M, HW]
        bias = torch.zeros_like(scores)
        for t in range(T):
            feat_pos = feature_positions
            track_pos = tracks[t] 
            distances = torch.cdist(feat_pos, track_pos, p=2)  # [HW, M]
            bias[t] = -distances.pow(2) / (2 * self.sigma**2)    # [M, HW]

        scores = scores + bias
        attn_weights = F.softmax(scores, dim=-1)  # [num_heads, M, HW]

        sampled = torch.matmul(attn_weights, V)  # [num_heads, M, d_model//num_heads]
        sampled = sampled.transpose(1, 2).contiguous().view(T, HW, -1)  # [M, d_model]
        res = self.W_out(sampled)
        return res

        updated_features_list = []
        
        for t in range(T):
            # Compute attention scores (HW queries attending to M keys)
            scores = torch.matmul(Q[t], K[t].transpose(-2, -1)) / math.sqrt(self.head_dim)  # [num_heads, HW, M]
            
            # Compute spatial bias term (transposed from sampling)
            feat_pos = feature_positions  # [HW, 2]
            track_pos = tracks[t]         # [M, 2]
            
            # Calculate distance-based bias
            distances = torch.cdist(feat_pos, track_pos, p=2)  # [HW, M]
            bias = -distances.pow(2) / (2 * self.sigma**2)    # [HW, M]
            bias = bias.unsqueeze(0).expand(self.num_heads, -1, -1)  # [num_heads, HW, M]
            
            # Add bias and apply softmax
            scores = scores + bias
            attn_weights = F.softmax(scores, dim=-1)  # [num_heads, HW, M]
            
            # Apply attention to values
            updates = torch.matmul(attn_weights, V[t])  # [num_heads, HW, d_model//num_heads]
            updates = updates.transpose(0, 1).contiguous().view(HW, -1)  # [HW, d_model]
            updated_features_list.append(updates)
        
        updated_features = torch.stack(updated_features_list, dim=0)  # [T, HW, d_model]
        
        # Apply output projection
        updated_features = self.W_out(updated_features)
        
        return updated_features







class TracktentionLayer(nn.Module):
    """Complete Tracktention Layer combining all components."""
    
    def __init__(self, d_model: int, d_k: Optional[int] = None, num_heads: int = 8, 
                 num_transformer_layers: int = 2, sigma: float = 0.5, dropout: float = 0.0):
        super().__init__()
        
        if d_k is None:
            d_k = d_model
            
        self.d_model = d_model
        self.d_k = d_k
        
        # Track token embedding (simple positional embedding for track coordinates)
        self.track_embedding = nn.Linear(2, d_model, bias=False)  # Embed 2D coordinates
        
        # Main components
        self.attentional_sampling    = AttentionalSampling(d_model, d_k, num_heads, sigma)
        self.track_transformer       = TrackTransformer(d_model, num_heads, num_transformer_layers, dropout)
        self.attentional_splatting   = AttentionalSplatting(d_model, d_k, num_heads, sigma)
        
    def create_feature_positions(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create grid positions for feature map tokens."""
        y, x = torch.meshgrid(torch.arange(H, device=device), 
                             torch.arange(W, device=device), indexing='ij')
        positions = torch.stack([x.flatten(), y.flatten()], dim=-1).to(torch.bfloat16)  # [HW, 2]
        return positions
    
    def forward(self, features: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [T, H, W, d_model] - input video features
            tracks: [T, M, 2] - point tracks (x, y coordinates)
            
        Returns:
            output: [T, H, W, d_model] - updated features with temporal consistency
        """
        T, H, W, d_model = features.shape
      
        _, M, _ = tracks.shape
        
        # Flatten spatial dimensions
        features_flat = features.view(T, H * W, d_model)  # [T, HW, d_model]
        
        # Create feature positions
        feature_positions = self.create_feature_positions(H, W, features.device)  # [HW, 2]
        
        # Create track token embeddings
        track_tokens = self.track_embedding(tracks)  # [T, M, d_model]

        grid_coords_tokens = self.track_embedding(feature_positions)
        
        # 1. Attentional Sampling: pool information from features to tracks
        sampled_features = self.attentional_sampling(
            features_flat, tracks, track_tokens, feature_positions
        )  # [T, M, d_model]
        
        # 2. Track Transformer: process tracks temporally
        updated_track_tokens = self.track_transformer(sampled_features)  # [T, M, d_model]
        
        
        # 3. Attentional Splatting: map track tokens back to features
        feature_updates = self.attentional_splatting(
            updated_track_tokens, tracks, feature_positions, features_flat, grid_coords_tokens.expand(T, -1,-1)
        )  # [T, HW, d_model]
        
        # Reshape back to original dimensions
        feature_updates = feature_updates.view(T, H*W, d_model)
        
        return feature_updates