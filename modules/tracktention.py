
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
    theta = 100 ** (-2 * idx / half)   # [quarter]


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


def create_2d_positional_embedding(positions: torch.Tensor, d_model: int):
    """
    Create 2D positional embeddings using sinusoidal-style encoding
    with RoPE-like frequency scaling.

    Args:
        positions: [N, 2] or [B, N, 2] - 2D coordinates
        d_model: embedding dimension (must be divisible by 4)

    Returns:
        embeddings: [N, d_model] or [B, N, d_model]
    """
    if d_model % 4 != 0:
        raise ValueError("d_model must be divisible by 4 (2 for x, 2 for y)")

    squeeze_output = False
    if positions.dim() == 2:   # [N, 2]
        positions = positions.unsqueeze(0)  # -> [1, N, 2]
        squeeze_output = True

    B, N, _ = positions.shape
    half = d_model // 2
    quarter = d_model // 4

    # Frequencies
    idx = torch.arange(quarter, device=positions.device, dtype=positions.dtype)
    theta = 10000 ** (-2 * idx / half)  # [quarter]

    # Extract coords
    pos_x, pos_y = positions[..., 0], positions[..., 1]  # [B, N]

    # Apply frequencies
    freqs_x = torch.einsum('bn,q->bnq', pos_x, theta)  # [B, N, quarter]
    freqs_y = torch.einsum('bn,q->bnq', pos_y, theta)  # [B, N, quarter]

    # Encode as cos/sin pairs
    emb_x = torch.cat([freqs_x.sin(), freqs_x.cos()], dim=-1)  # [B, N, 2*quarter]
    emb_y = torch.cat([freqs_y.sin(), freqs_y.cos()], dim=-1)  # [B, N, 2*quarter]

    embeddings = torch.cat([emb_x, emb_y], dim=-1)  # [B, N, d_model]

    if squeeze_output:
        embeddings = embeddings.squeeze(0)

    return embeddings


class QKNormalization(nn.Module):
    """Query-Key normalization to stabilize attention computation."""
    
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-6
        self.q_norm = nn.LayerNorm(d_model, eps=1e-6,bias=False, )
        self.k_norm = nn.LayerNorm(d_model, eps=1e-6,bias=False,)
        
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
        self.qk_norm = QKNormalization(d_k)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

   
        
    def forward(self, features: torch.Tensor, tracks: torch.Tensor, 
                track_tokens: torch.Tensor, 
                feature_positions: torch.Tensor) -> torch.Tensor:
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

       
        #Q = rope_2d(Q, tracks)
        K = rope_2d(K, feature_pos_expanded)
        V = rope_2d(features, feature_pos_expanded)

        
        # Apply QK normalization
        Q, K = self.qk_norm(Q, K)

    
        
        # Reshape for multi-head attention
        Q = Q.view(T, M, self.num_heads, self.head_dim).transpose(1, 2)      # [T, num_heads, M, head_dim]
        K = K.view(T, HW, self.num_heads, self.head_dim).transpose(1, 2)     # [T, num_heads, HW, head_dim]
        V = V.view(T, HW, self.num_heads, -1).transpose(1, 2)         # [T, num_heads, HW, d_model//num_heads]
        
        
        
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
                           (-math.log(30) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
        
    
    def forward(self, track_tokens: torch.Tensor, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            track_tokens: [T, M, d_model] - track tokens from attentional sampling
            
        Returns:
            updated_track_tokens: [T, M, d_model] - temporally updated track tokens
        """
        BT, M, d_model = track_tokens.shape
        track_tokens = track_tokens.view(B, T, M, d_model)
        track_tokens = track_tokens.permute(0, 2, 1, 3)
        track_tokens_swapped = track_tokens.reshape(B * M, T, d_model)
        
        # Swap dimensions: [T, M, d_model] -> [M, T, d_model]
        # This allows us to process each track's temporal sequence independently
        #track_tokens_swapped = track_tokens.transpose(0, 1)  # [M, T, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:T, :].unsqueeze(0).expand(B*M, -1, -1).to(track_tokens.device)
        track_tokens_with_pos = track_tokens_swapped + pos_enc
        
        # Process each track independently using transformer
        # Reshape to batch dimension for parallel processing
        #track_tokens_flat = track_tokens_with_pos.reshape(M, T, d_model)
        updated_tokens = self.transformer(track_tokens_with_pos)  # [M, T, d_model]
        
        # Swap dimensions back: [M, T, d_model] -> [T, M, d_model]
        updated_tokens = updated_tokens.view(B, M, T, d_model)  # [B, M, T, d_model]
        updated_tokens = updated_tokens.permute(0, 2, 1, 3)     # [B, T, M, d_model]
        updated_tokens = updated_tokens.reshape(BT, M, d_model)    # [B*T, M, d_model]
        
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
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE and normalization
        self.qk_norm = QKNormalization(d_k)
        
        # Initialize output projection to zero
        #nn.init.zeros_(self.W_out.weight)
        

        
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
        

        feature_pos_expanded = feature_positions.unsqueeze(0).expand(T, -1, -1)  # [T, HW, 2]
        
        #Q = rope_2d(Q, feature_pos_expanded)
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
            features: [T, H, W, d_model] - input video features
            tracks: [T, M, 2] - point tracks (x, y coordinates)
            
        Returns:
            output: [T, H, W, d_model] - updated features with temporal consistency
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
        
        track_pos_embeddings = create_2d_positional_embedding(tracks_flat, d_model).squeeze(0)  # [BT, M, d_model]
        # 1. Attentional Sampling: pool information from features to tracks
        
        sampled_features = self.attentional_sampling(
            features_flat, tracks_flat, track_pos_embeddings, feature_positions
        )  # [T, M, d_model]
        
        # 2. Track Transformer: process tracks temporally
        updated_track_tokens = self.track_transformer(sampled_features, B, T)  # [T, M, d_model]
        
        
        # 3. Attentional Splatting: map track tokens back to features
        feature_updates = self.attentional_splatting(
            updated_track_tokens, tracks_flat, feature_positions, features_flat, feature_pos_embeddings
        )  # [T, HW, d_model]
        
        # Reshape back to original dimensions
        feature_updates = feature_updates.view(BT, H*W, d_model)
        
        return feature_updates







def get_sinusoidal_embeddings(n_positions, dim):
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(100.0) / dim))
    embeddings = torch.zeros(n_positions, dim)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings





class MovementVectorPredictor(nn.Module):
    """Complete Tracktention Layer combining all components."""
    
    def __init__(self, config, num_heads: int = 8, 
                 num_transformer_layers: int = 1, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = config.hidden_size

        #self.image_size = config.image_size
        #self.patch_size = config.patch_size

        #self.num_patches_per_side = self.image_size // self.patch_size
        #self.num_patches = self.num_patches_per_side**2
        #self.num_positions = self.num_patches
        self.max_points = 900
        self.point_embeddings = nn.Parameter(
            torch.randn(self.max_points, self.embed_dim) * 0.02
        )

        #self.point_embeddings+= get_sinusoidal_embeddings(self.max_points, self.embed_dim)

        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(self.embed_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.movement_predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, 2)  # Predict dx, dy
        )

        self.visibility_predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim // 2, 1)  # Predict visibility probability
        )

        self.layer_norm = nn.LayerNorm(self.embed_dim)

        

        
 
    
    def forward(self, features: torch.Tensor, tracks: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [BT, H, W, d_model] - input video features
            tracks: [B, T, M, 2] - point tracks (x, y coordinates)
            visibility: [B, T, M] - boolean

      """

        BT, HW, d_model = features.shape
        B,T,M, _ = tracks.shape
        diff = tracks[:, 1:, :, :] - tracks[:, :-1, :, :]  # [B, T-1, M, 2]
        visibility_flat = visibility.view(BT, M)
        # Pad zeros for the first frame
        zeros = torch.zeros(B, 1, M, 2, device=tracks.device, dtype=tracks.dtype)
        tracks_diff = torch.cat([zeros, diff], dim=1) 
        tracks_diff = tracks_diff.view(BT,M,2)

        print(tracks_diff.shape)

        point_queries = self.point_embeddings.unsqueeze(0).to(tracks.device)  # [1, 1, M, d_model]
        point_queries = point_queries.expand(BT, -1, -1)  # [B, T, M, d_model]
        

        attended_points = point_queries
        for layer in self.cross_attention_layers:
            attended_points = layer(
                query=attended_points,  # [BT, M, d_model]
                key_value=features      # [BT, H*W, d_model]
            )
        
        # Apply layer normalization
        attended_points = self.layer_norm(attended_points)  # [BT, M, d_model]

        predicted_movements = self.movement_predictor(attended_points)  # [BT, M, 2]
        predicted_visibility = self.visibility_predictor(attended_points).squeeze(-1)  # [BT, M]

        
        # Compute losses
        # Movement loss - only for visible points 
        visibility_mask = visibility_flat.float()  # [BT, M]
        movement_mask = visibility_mask.unsqueeze(-1)  # [BT, M, 1]
        
        movement_error = (predicted_movements - tracks_diff) ** 2  # [BT, M, 2]
        movement_error_masked = movement_error * movement_mask

        
        movement_loss = (movement_error_masked.sum()) / (movement_mask.sum() + 1e-8)
        
        # Visibility loss - binary cross entropy
        visibility_targets = visibility_flat.float()  # [BT, M]
        visibility_loss = F.binary_cross_entropy_with_logits(
            predicted_visibility, visibility_targets, reduction='mean'
        )
        
        return (movement_loss + 4 *visibility_loss)
        


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer for point-to-patch attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, T, M, d_model] - point queries
            key_value: [BT, H*W, d_model] - patch features (keys and values)
        """
        BT, M, d_model = query.shape
        _, HW, _ = key_value.shape
        
        # Reshape for batch processing: [B*T, seq_len, d_model]
        query_flat = query
        kv_flat = key_value
        
        # Linear projections
        Q = self.q_proj(query_flat)  # [B*T, M, d_model]
        K = self.k_proj(kv_flat)     # [B*T, H*W, d_model]
        V = self.v_proj(kv_flat)     # [B*T, H*W, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(BT, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B*T, num_heads, M, head_dim]
        K = K.view(BT, HW, self.num_heads, self.head_dim).transpose(1, 2)  # [B*T, num_heads, H*W, head_dim]
        V = V.view(BT, HW, self.num_heads, self.head_dim).transpose(1, 2)  # [B*T, num_heads, H*W, head_dim]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B*T, num_heads, M, H*W]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, V)  # [B*T, num_heads, M, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(BT, M, d_model)
        output = self.out_proj(attn_output)  # [B*T, M, d_model]
        
        
        # Residual connection
        return output







class LatentRegulaizer(nn.Module):
    """Complete Tracktention Layer combining all components."""
    
    def __init__(self, config, num_heads: int = 8, 
                 num_transformer_layers: int = 1, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.patch_size = 14

        self.max_points = 900


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

    def map_point_to_patch(self, feature_positions: torch.Tensor, tracks: torch.Tensor):
        """
        Args:
            feature_positions: [HW, 2]
            tracks: [B, T, M, 2]
        Returns:
            nn_idx: [B, T, M] -> indices of nearest patches
        """
        HW = feature_positions.shape[0]
        # Compute squared distances
        diff = tracks.unsqueeze(-2) - feature_positions.view(1, 1, 1, HW, 2)
        dist_sq = (diff ** 2).sum(-1)   # [B, T, M, HW]
        nn_idx = dist_sq.argmin(-1)     # [B, T, M]
        return nn_idx


    
    def forward(self, features: torch.Tensor, tracks: torch.Tensor, visibility: torch.Tensor):
        """
        features: [B*T, H, W, d_model]
        tracks: [B, T, M, 2]
        visibility: [B, T, M]
        """

        *prefix, P, D = features.shape
        sqrt_P = int(P**0.5)
        new_shape = (*prefix, sqrt_P, sqrt_P, D)
        features = features.view(*new_shape)

        B, T, M, _ = tracks.shape
        _, H, W, d_model = features.shape
        HW = H * W

        # [HW, 2] patch centers
        feature_positions = self.create_feature_positions(H, W, features.device)

        # Map each point → nearest patch index
        nn_idx = self.map_point_to_patch(feature_positions, tracks)   # [B,T,M]

        # Flatten features for indexing
        features = features.view(B, T, H*W, d_model)  # [B,T,HW,d]

        # Gather embeddings at point locations
        # nn_idx: [B,T,M] → expand for gather
        point_features = features.gather(
            2, nn_idx.unsqueeze(-1).expand(-1, -1, -1, d_model)
        )   # [B, T, M, d_model]

        # =========================
        # 1) Consecutive differences
        # =========================
        f_t     = point_features[:, :-1]   # [B,T-1,M,d]
        f_next  = point_features[:, 1:]    # [B,T-1,M,d]

        vis_t    = visibility[:, :-1]      # [B,T-1,M]
        vis_next = visibility[:, 1:]       # [B,T-1,M]

        mask_consec = (vis_t & vis_next).unsqueeze(-1)  # [B,T-1,M,1]

        diff_consec = ((f_t - f_next).abs()).sum(-1)     # [B,T-1,M]
        diff_consec = diff_consec * mask_consec.squeeze(-1)

        # =========================
        # 2) Difference from first frame
        # =========================
        f_first = point_features[:, 0:1]   # [B,1,M,d]
        vis_first = visibility[:, 0:1]     # [B,1,M]

        f_rest = point_features[:, 1:]     # [B,T-1,M,d]
        vis_rest = visibility[:, 1:]       # [B,T-1,M]

        mask_ref = (vis_first & vis_rest).unsqueeze(-1) # [B,T-1,M,1]

        diff_ref = ((f_rest - f_first).abs()).sum(-1)    # [B,T-1,M]
        diff_ref = diff_ref * mask_ref.squeeze(-1)


        return 0.01* diff_consec.mean() + 0.01*diff_ref.mean()