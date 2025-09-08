
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import math

import torch
from torch import nn
import cv2
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs
from transformers.models.auto import AutoModel
from configuration_smolvlm import SmolVLMConfig, SmolVLMVisionConfig
from convert_utils import flow_to_image
from modules.gmflow.gmflow.gmflow import GMFlow
from modules.tracktention import TracktentionLayer, MovementVectorPredictor, LatentRegulaizer
import json
from modules.movement_cross_attn import MovementCrossAttn , global_correlation_softmax



def load_json(filename):
    """Load JSON file as a dictionary."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_json(filename, data):
    """Save dictionary to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def update_json(filename, new_dict):
    """Update or add a key-value pair to the JSON file."""
    data = load_json(filename)
    data.update(new_dict)
    #print(data)
    save_json(filename, data)




class SmolVLMPreTrainedModel(PreTrainedModel):
    config_class = SmolVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SmolVLMVisionAttention", "SmolVLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


class SmolVLMVisionEmbeddings(nn.Module):
    """
    This is a modified version of `siglip.modelign_siglip.SiglipVisionEmbeddings` to enable images of variable
    resolution.

    The modifications are adapted from [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
    which allows treating images in their native aspect ratio and without the need to resize them to the same
    fixed size. In particular, we start from the original pre-trained SigLIP model
    (which uses images of fixed-size square images) and adapt it by training on images of variable resolutions.
    """

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
  

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class SmolVLMVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Ignore copy
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

       

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class SmolVLMVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states





class SmolVLMEncoderLayer(nn.Module):
    def __init__(self, config: SmolVLMVisionConfig, i: 0):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SmolVLMVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SmolVLMVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.depth = i

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        movement_vectors: Optional[torch.FloatTensor] = None,
        pred_visibility : Optional[torch.BoolTensor] = None,

        
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        #print(self.depth)
       
        outputs = (hidden_states,)

        if output_attentions:
            #print("HERE")
            outputs += (attn_weights,)

        return outputs


class SmolVLMEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SmolVLMEncoderLayer`].

    Args:
        config: SmolVLMConfig
    """

    def __init__(self, config: SmolVLMConfig):
        super().__init__()
        self.config = config
        self.num_MCA_layers = 4
        self.use_cfg = True
        self.use_mask = config.use_mask


        layers_lst = []

        for i in range(config.num_hidden_layers):
            layers_lst.append(SmolVLMEncoderLayer(config, i))

        self.layers = nn.ModuleList(layers_lst)
        if config.use_cfg:
            self.MCA_layers = nn.ModuleList([
                MovementCrossAttn(feature_channels=config.hidden_size) for i in range(self.num_MCA_layers)
            ])

            if self.use_mask:
            
                last_layers = [8,13,18,24]
            else:
                last_layers = list(range(config.num_hidden_layers - self.num_MCA_layers -1, config.num_hidden_layers-1))
            
            #print(last_layers)
            #exit(1)
            self.MCA_positions = last_layers

            #self.MCA_positions = [
            #    ((i + 1) * config.num_hidden_layers // (self.num_MCA_layers + 1))
            #    for i in range(self.num_MCA_layers)
            #]

            
        
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        movement_vectors: Optional[torch.FloatTensor] = None,
        pred_visibility : Optional[torch.BoolTensor] = None,

        
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        MCA_layer_idx = 0

       
        for layer_idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                
                )
              

            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                  
                )
                
            
            
            hidden_states = layer_outputs[0]


            if (self.use_cfg and 
                MCA_layer_idx < len(self.MCA_positions) and 
                layer_idx == self.MCA_positions[MCA_layer_idx]):
                residual = hidden_states

                T = 30
                B = hidden_states.shape[0] // T
                H, D = hidden_states.shape[1:]  # 729, 1152 
                hidden_states_reshape = hidden_states.view(B,T,H,D)
                f_t     = hidden_states_reshape[:, :-1]   # [B,T-1,M,d]
                f_next  = hidden_states_reshape[:, 1:]    # [B,T-1,M,d] 

                f_t        = f_t.reshape(B*(T-1),H,D)
                f_next     = f_next.reshape(B*(T-1),H,D)

                
                if False:
                    hidden_states = self._gradient_checkpointing_func(
                        self.MCA_layers[MCA_layer_idx].__call__,
                         f_t,
                         f_next,
                         B,
                         T,
                         H,
                         D
                    )
                else:
                    hidden_states = self.MCA_layers[MCA_layer_idx](
                        f_t,
                        f_next,
                        B,
                        T,
                        H,
                        D
                    )
                
                
                hidden_states = hidden_states +residual
                MCA_layer_idx += 1



            if output_attentions:

                all_attentions = all_attentions + (layer_outputs[1],)

           
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )



class SmolVLMVisionTransformer(SmolVLMPreTrainedModel):
    config_class = SmolVLMVisionConfig
    _supports_sdpa = True
    _supports_flash_attention_2 = True
    _supports_flex_attn = True

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__(config)
        embed_dim = config.hidden_size
        
        self.embeddings = SmolVLMVisionEmbeddings(config)
        self.encoder = SmolVLMEncoder(config)
        self.patch_size = config.patch_size
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.use_cfg = True
        self.use_optflow = config.use_optflow
        self.use_mask = config.use_mask


        #if config.use_cfg:
        #    self.movement_vector_predictor =  LatentRegulaizer(config)

       


    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def masked_optical_flow_loss(self,pred, target, pred_visibility,use_mask=False, magnitude_threshold=384):
        """
        L1 loss between optical flow vectors, masked when magnitude > threshold

        Args:
            pred: torch.Tensor of shape [116, 729, 2] - predicted flow
            target: torch.Tensor of shape [116, 729, 2] - target flow  
            magnitude_threshold: float - mask vectors with magnitude > this value

        Returns:
            torch.Tensor - scalar loss value
        """
        # Calculate magnitudes for both pred and target
        pred_mag = torch.norm(pred, dim=-1)  # [116, 729]
        target_mag = torch.norm(target, dim=-1)  # [116, 729]

        # Create mask - keep vectors where both magnitudes <= threshold
        if use_mask:
            mask =  pred_visibility
        else:
            mask = (pred_mag <= magnitude_threshold) & (target_mag <= magnitude_threshold)
        


        # Apply mask and calculate L1 loss only on valid vectors
        if mask.sum() > 0:
            masked_pred = pred[mask]  # [N, 2] where N is number of valid vectors
            masked_target = target[mask]  # [N, 2]
            loss = F.l1_loss(masked_pred, masked_target, reduction='mean')
        else:
            # If no valid vectors, return zero loss
            loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

        return loss

    def forward(
        self,
        pixel_values,
        pixel_values_copy,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        movement_vectors: Optional[torch.FloatTensor] = None,
        pred_visibility : Optional[torch.BoolTensor] = None,

        
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        elif not self._use_flash_attention_2:
            patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            movement_vectors = movement_vectors,
            

        )

        last_hidden_state = encoder_outputs[0]



        loss_rec = None
        movement_vectors_loss = []


            #import numpy as np
            #l2 = torch.sum(last_hidden_state[18], dim=1)  # [729]
            ## Reshape to 27x27
            #heatmap = l2.view(1, 1, 27, 27)  # [1, 1, H, W] for interpolate
            ## Upsample to 128x128
            #heatmap_up = F.interpolate(heatmap, size=(128, 128), mode='bilinear', align_corners=False)
            #heatmap_np = heatmap_up.squeeze().detach().cpu().numpy()  # shape: [128, 128]
            ## Normalize to 0-255 and convert to uint8
            #heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX)
            #heatmap_uint8 = heatmap_norm.astype(np.uint8)
            ## Apply colormap
            #heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            ## Save the image
            #cv2.imwrite(f'to_del/b.png', heatmap_color)
            ##print("REACHED")
            #exit(1)
        


        last_hidden_state = self.post_layernorm(last_hidden_state)

         #TEMPORAL
        if self.use_cfg and self.training:

            T = 30
            B = hidden_states.shape[0] // T
            H, D = hidden_states.shape[1:]  # 729, 1152 
            hidden_states_reshape = last_hidden_state.view(B,T,H,D)
            f_t     = hidden_states_reshape[:, :-1]   # [B,T-1,M,d]
            f_next  = hidden_states_reshape[:, 1:]    # [B,T-1,M,d] 
            f_t        = f_t.reshape(B*(T-1),H,D)
            f_next     = f_next.reshape(B*(T-1),H,D)

            #print(f_t.shape)
          

            flow, _ = global_correlation_softmax(f_t, f_next)
            flow = flow.permute(0,2,3,1).view(B*(T-1),H,2)
            '''
            import numpy as np
            import matplotlib.pyplot as plt
            print(flow.shape)
            print(pixel_values_copy.shape)
            grid_size = 27
            patch_h = 384 // grid_size
            patch_w = 384 // grid_size
            y_centers = np.arange(patch_h//2, 384, patch_h)
            x_centers = np.arange(patch_w//2, 384, patch_w)
            grid_y, grid_x = np.meshgrid(y_centers, x_centers, indexing="ij")
            centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # shape [729, 2]

            for i in range(29):
                x = pixel_values_copy[0]
                x = x[i,:,:,:]

                x = x.permute(1, 2, 0)
                x = x.cpu().float().numpy()
                x = (x - x.min()) / (x.max() - x.min())
                x = np.float32(x)
                x =  np.uint8(255 * x)
                x = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                
                displacements = flow[i].detach().cpu().numpy()
                magnitudes = np.linalg.norm(displacements, axis=1)

                plt.figure(figsize=(6, 6))
                plt.imshow(x)
                plt.quiver(
                    centers[:, 0], centers[:, 1],   # X, Y start points
                    displacements[:, 0], displacements[:, 1],  # U, V (dx, dy)
                    magnitudes,  # color by magnitude
                    angles="xy", scale_units="xy", scale=1.5, cmap="turbo", width=0.003
                )
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(f"to_del3/frame_{i:03d}X.png", dpi=150)
                plt.close()

            print(flow[0,:,:,:])
            exit(1)'''
            #print(flow.shape)
            
            loss_rec = 0.1* self.masked_optical_flow_loss(flow, movement_vectors,pred_visibility,use_mask=self.use_mask)
                       
            movement_vectors_loss = [loss_rec]


        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), movement_vectors_loss


@dataclass
class SmolVLMBaseModelOutputWithPast(ModelOutput):
    """
    Base class for SmolVLM model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    movement_vectors_loss: Optional[List[torch.FloatTensor]] = None


class SmolVLMSimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)




class VisionToFlowTransform(nn.Module):
    """
    Transforms vision embeddings from (B, N, 729, 1152) to (B, N, 2304, 8, 128)
    for optical flow decoder input.
    
    Args:
        input_dim (int): Input embedding dimension (1152)
        output_dim (int): Output embedding dimension (128)
        num_tokens (int): Number of tokens per patch in output (8)
        input_spatial_size (tuple): Input spatial dimensions (27, 27)
        output_spatial_size (tuple): Output spatial dimensions (48, 48)
    """
    
    def __init__(self, 
                 input_dim=1152, 
                 output_dim=128, 
                 num_tokens=8,
                 input_spatial_size=(27, 27),
                 output_spatial_size=48):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.input_spatial_size = input_spatial_size
        self.output_spatial_size = output_spatial_size
        self.hierarchical_transform =  nn.Linear(input_dim, output_dim*num_tokens, bias=False)


    def forward(self, x):
        """
        Method 3: Process each patch individually, then handle spatial upsampling
        """
        B, N, patches, dim = x.shape
        
        # First transform each patch: (B, N, 729, 1152) -> (B, N, 729, 8*128)
        x_patch_transformed = self.hierarchical_transform(x)
        #exit(1)

        x_patch_tokens = x_patch_transformed.view(B, N, patches, self.num_tokens, self.output_dim)

        
        # Now we need to upsample from 729 patches to 2304 patches
        # Reshape to spatial format for upsampling
        x_spatial = x_patch_tokens.view(B, N, self.input_spatial_size[0], self.input_spatial_size[1], self.num_tokens * self.output_dim)


        x_spatial = x_spatial.view(B*N, self.input_spatial_size[0], self.input_spatial_size[1], self.num_tokens * self.output_dim)
        x_spatial = x_spatial.permute(0, 3, 1, 2)  # (B*N, 8*128, 27, 27)
        
        # Upsample spatially
        x_upsampled = F.interpolate(x_spatial, size=(self.output_spatial_size, self.output_spatial_size), mode='bilinear', align_corners=False)
        
        # Reshape back to final format
        x_upsampled = x_upsampled.permute(0, 2, 3, 1)  # (B*N, 48, 48, 8*128)

        x_upsampled = x_upsampled.view(B*N, self.output_spatial_size*self.output_spatial_size, self.num_tokens, self.output_dim)
        
        # Reshape to final output format: (N, B*2304, 8, 128)
        x_final = x_upsampled.view(B, N, self.output_spatial_size*self.output_spatial_size, self.num_tokens, self.output_dim)
        x_final = x_final.permute(1, 0, 2, 3, 4)  # (N, B, 2304, 8, 128)
        x_final = x_final.contiguous().view(N, B*self.output_spatial_size*self.output_spatial_size, self.num_tokens, self.output_dim)  # (N, B*2304, 8, 128)
       
        return x_final








class VisionToFlowTransformGM(nn.Module):
     def __init__(self, input_dim=1152,input_spatial_size=(27, 27), output_dim=128, output_spatial_size=64):
        """
        Module to transform tensor from (B, N, input_spatial_size_p2, D1) 
        to (N, B, output_dim, output_spatial_size, output_spatial_size)
        
        Args:
            input_dim (int): Input embedding dimension D1
            output_dim (int): Output embedding dimension (default: 128)
            output_spatial_size (int): Output spatial size (default: 64 for 64x64)
        """
        super(VisionToFlowTransformGM, self).__init__()
        
        self.output_dim = output_dim
        self.output_spatial_size = output_spatial_size
        self.input_spatial_size = input_spatial_size

        
        # Linear layer to transform embedding dimension and spatial size
        self.linear = nn.Linear(input_dim, output_dim)
        
     def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, N, input_spatial_size_p2, D1)
            
        Returns:
            Output tensor of shape (N, B, output_dim, output_spatial_size, output_spatial_size)
        """
        B, N, input_spatial_size_p2, D1 = x.shape
        input_spatial_size_p2 = self.input_spatial_size[0] ** 2
        # Apply linear transformation to embedding dimension
        # Shape: (B, N, input_spatial_size_p2, D1) -> (B, N, input_spatial_size_p2, output_dim)
        x = self.linear(x)
        
        # Check if we need to interpolate the spatial dimension
        # Reshape to treat spatial dimension as 2D for interpolation
        
        # Reshape: (B, N, input_spatial_size_p2, output_dim) -> (B*N, output_dim, input_spatial_size, input_spatial_size)
        x = x.view(B * N, input_spatial_size_p2, self.output_dim)
        x = x.permute(0, 2, 1)  # (B*N, output_dim, input_spatial_size_p2)
        x = x.view(B * N, self.output_dim, self.input_spatial_size[0], self.input_spatial_size[1])
        
        # Interpolate to target spatial size
        x = torch.nn.functional.interpolate(
            x, 
            size=(self.output_spatial_size, self.output_spatial_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Reshape back: (B*N, output_dim, output_spatial_size, output_spatial_size) -> (B, N, output_dim, output_spatial_size, output_spatial_size)
        x = x.view(B, N, self.output_dim, self.output_spatial_size, self.output_spatial_size)
       
        
        # Permute to get final shape: (B, N, output_dim, output_spatial_size, output_spatial_size) -> (N, B, output_dim, output_spatial_size, output_spatial_size)
        x = x.permute(1, 0, 2, 3, 4)
        
        return x









class SmolVLMConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.scale_factor
     

        self.modality_projection = SmolVLMSimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states):



        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)

        image_hidden_states = self.modality_projection(image_hidden_states)

        return image_hidden_states


class SmolVLMModel(SmolVLMPreTrainedModel):
    """
    A subclass of Idefics3Model. We do *not* remove or block the call to inputs_merger
    in forward. Instead, we override inputs_merger here with custom logic.
    """

    def __init__(self, config: SmolVLMConfig, use_cfg: False, use_optflow : False, use_mask : False):
        super().__init__(config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size
        config.vision_config.use_cfg     = True
        config.vision_config.use_optflow = True if use_optflow else False
        config.vision_config.use_mask    = True if use_mask else False


       
        self.vision_model = SmolVLMVisionTransformer._from_config(config.vision_config)
        self.connector = SmolVLMConnector(config)
        self.text_model = AutoModel.from_config(config.text_config)

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )
        self.image_token_id = self.config.image_token_id

        self._use_flash_attention_2 = config.text_config._attn_implementation == "flash_attention_2"

        self.post_init()

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.

        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032

        Override to set output.requires_grad = True for both the decoder's and vision model's embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = get_lowest_module(self.vision_model).register_forward_hook(
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def inputs_merger(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
    ):
        """
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder and that hidden state, after a pixel shuffle operation, is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """
        _, patch_size, _ = image_hidden_states.shape

        image_mask = input_ids == self.image_token_id
        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("At least one sample has <image> tokens not divisible by patch_size.")

        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]


        merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)

        return merged_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, pixel_attention_mask: torch.LongTensor = None,  movement_vectors:torch.LongTensor = None, pred_visibility : Optional[torch.BoolTensor] = None ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            pixel_attention_mask (`torch.LongTensor`, *optional*):
                The attention mask indicating padded regions in the image.
        """
        movement_vectors_loss = None
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        #pixel_values_copy = pixel_values.clone()
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
        pixel_values_copy = pixel_values.view(batch_size,num_images,*pixel_values.shape[1:])



        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        if not any(real_images_inds):
            # no images, leave one empty image.
            real_images_inds[0] = True

        pixel_values = pixel_values[real_images_inds].contiguous()

        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()
        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states, movement_vectors_loss = self.vision_model(pixel_values=pixel_values,pixel_values_copy=pixel_values_copy, patch_attention_mask=patch_attention_mask, movement_vectors= movement_vectors, pred_visibility = pred_visibility )


      

        image_hidden_states = image_hidden_states.last_hidden_state
        ##print(f"image_hidden_states2 {image_hidden_states.shape}")


        # Modality projection & resampling
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states, movement_vectors_loss

    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        movement_vectors : Optional[torch.LongTensor] = None,
        
        pred_visibility : Optional[torch.BoolTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, SmolVLMBaseModelOutputWithPast]:
        
        movement_vectors_loss = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            image_hidden_states, movement_vectors_loss = self.get_image_features(pixel_values, pixel_attention_mask, movement_vectors,pred_visibility )
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if inputs_embeds is not None and image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )


        return SmolVLMBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
            movement_vectors_loss = movement_vectors_loss
        )


@dataclass
class SmolVLMCausalLMOutputWithPast(ModelOutput):
 
    loss: Optional[torch.FloatTensor] = None
    cap_loss: Optional[torch.FloatTensor] = None
    movement_vectors_loss: Optional[torch.FloatTensor] = None

    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class SmolVLMForConditionalGeneration(SmolVLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, use_cfg=False, use_optflow = False,use_mask=False ):
        super().__init__(config)
        self.model = SmolVLMModel(config, use_cfg=use_cfg, use_optflow = use_optflow, use_mask= use_mask)
        self.image_token_id = self.config.image_token_id
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size


        # Initialize weights and apply final processing
        self.post_init()

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = self.model.vision_model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.model.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.text_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings



    
    def smooth_l1_loss(self, diff):
        cond = diff.abs() < 1
        loss = torch.where(cond, 0.5*diff**2, diff.abs()-0.5)
        return loss

    
    

   
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        
        movement_vectors: Optional[torch.FloatTensor] = None,
        pred_visibility : Optional[torch.BoolTensor] = None,

        



        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, SmolVLMCausalLMOutputWithPast]:
      


       
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
 
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            movement_vectors = movement_vectors,
            pred_visibility = pred_visibility,
            

            **kwargs,
        )

        

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])


        loss = None
        cap_loss = None
        movement_vectors_loss = None

        if labels is not None:
            cap_loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
          

        #TEMPORAL
        if (movement_vectors is not None ) and (outputs.movement_vectors_loss):
            movement_vectors_loss = outputs.movement_vectors_loss[0]
            print(movement_vectors_loss)
            

       

        if cap_loss is not None and movement_vectors_loss is None:
            loss = cap_loss
        
        elif (cap_loss is not None) and (movement_vectors_loss is not None):
            loss = movement_vectors_loss + cap_loss
        else:
            loss = movement_vectors_loss
        

       
        
        if self.training:
            print(loss)

        return SmolVLMCausalLMOutputWithPast(
            loss=loss,
            cap_loss = cap_loss,
            movement_vectors_loss = movement_vectors_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- there are mutually exclusive inputs (if the logic to make `image_hidden_states` take
        # precedence is moved to the model, we can remove this fn)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # but IDEFICS requires both ids and embeds to be present
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs["input_ids"] = input_ids

        if image_hidden_states is not None:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        # Get the precomputed image_hidden_states
        model_kwargs["image_hidden_states"] = outputs.image_hidden_states
        return model_kwargs


__all__ = ["SmolVLMForConditionalGeneration", "SmolVLMPreTrainedModel", "SmolVLMModel", "SmolVLMVisionTransformer"]