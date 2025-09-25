import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def global_correlation_softmax(feature0, feature1,
                           
                               ):
    # global correlation
    b, hw, d = feature0.shape
    h = w = int(hw ** 0.5)
   
    correlation = torch.matmul(feature0, feature1.transpose(1, 2)) / (d ** 0.5)

    # flow from softmax
    init_grid = coords_grid(b, h, w).to(correlation.device)  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    
    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    return flow, prob





class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, hw,c = x.size()
        h = int(math.sqrt(hw))
        w = int(math.sqrt(hw))

        

        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)

        pos = pos.view(pos.shape[0],pos.shape[1]*pos.shape[2],pos.shape[3] )

        return pos








def feature_add_position(feature0, feature1):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature0.shape[-1] // 2)
    if feature0.shape[-2] == 1025:
        position = pos_enc(feature0[:,1:,:])
        cls_token = torch.zeros(position.shape[0], 1, position.shape[2], device = position.device)
        position = torch.cat([cls_token, position], dim=1)

        feature0 = feature0 + position
        feature1 = feature1 + position
    else:
        position = pos_enc(feature0)
        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1





class MovementCrossAttn(nn.Module):
    def __init__(self,
                 feature_channels=128,
                 attention_type='swin',
                 num_heads=4,
                 **kwargs,
                 ):
        super().__init__()

        self.feature_channels = feature_channels

   

        # Transformer
        self.transformer = nn.MultiheadAttention(feature_channels, num_heads,batch_first=True, bias=False)

    

    def forward(self, hidden_t, hidden_n,B,T,H,D):
        feature0, feature1 = feature_add_position(hidden_t, hidden_n)
        
        res = self.transformer(feature0, feature1,feature1)[0]
        res = res.view(B, T-1, H, D)

        zero_frame = torch.zeros(B, 1, H, D, device=feature0.device, dtype=feature0.dtype)

        # concatenate zeros at the end along the temporal dimension
        res = torch.cat([res, zero_frame], dim=1)

        # reshape back to [B*T, H, D]
        res = res.view(B*T, H, D)
        
        return res

        
        '''
        feature0, feature1 = feature_add_position(hidden_t, hidden_n)

        concat0 = torch.cat((feature0, feature1), dim=0)  # [2B, H*W, C]
        concat1 = torch.cat((feature1, feature0), dim=0)  # [2B, H*W, C]
      
        res = self.transformer(concat0, concat1,concat1)[0]
        

        feature0, feature1 = res.chunk(chunks=2, dim=0)
        

        flow_pred = global_correlation_softmax(feature0, feature1)[0]

        return flow_pred'''


        


