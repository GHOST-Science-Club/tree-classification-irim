import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import ViTModel, ViTConfig
from transformers.models.vit.modeling_vit import ViTLayer


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, act=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.bmm(adj, support)
        return self.act(output)


class SILModule(nn.Module):
    def __init__(self, dim, num_patches, h, w):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.num_patches = num_patches

        self.gcn1 = GraphConvolution(dim, dim)
        self.gcn2 = GraphConvolution(dim, dim, act=False)

        self.pos_embed_mlp = nn.Sequential(
            nn.Linear(2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)
        )

    def get_polar_coordinates(self, ref_idx, batch_size, device):
        y_coords = torch.arange(self.h, device=device).repeat_interleave(self.w).float()
        x_coords = torch.arange(self.w, device=device).repeat(self.h).float()

        ref_y = (ref_idx // self.w).float().unsqueeze(1)
        ref_x = (ref_idx % self.w).float().unsqueeze(1)

        y_grid = y_coords.unsqueeze(0).expand(batch_size, -1)
        x_grid = x_coords.unsqueeze(0).expand(batch_size, -1)

        rho = torch.sqrt(((x_grid - ref_x) / self.w)**2 + ((y_grid - ref_y) / self.h)**2)
        theta = (torch.atan2(y_grid - ref_y, x_grid - ref_x) + math.pi) / (2 * math.pi)

        return torch.stack([rho, theta], dim=-1)

    def forward(self, x, attn_weights):
        B, _N, C = x.shape

        cls_attn = attn_weights[:, :, 0, 1:].mean(dim=1) # [B, N]
        _max_vals, max_indices = torch.max(cls_attn, dim=1)

        polar_coords = self.get_polar_coordinates(max_indices, B, x.device)
        struct_pos_embed = self.pos_embed_mlp(polar_coords)
        x_struct = x + struct_pos_embed

        threshold = cls_attn.mean(dim=1, keepdim=True)
        mask = (cls_attn > threshold).float().unsqueeze(2)
        adj = torch.bmm(mask, mask.transpose(1, 2))
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)

        s_feat = self.gcn1(x_struct, adj)
        s_feat = self.gcn2(s_feat, adj)

        ref_idx_expanded = max_indices.view(B, 1, 1).expand(-1, -1, C)
        object_structure_feature = torch.gather(s_feat, 1, ref_idx_expanded).squeeze(1)

        return object_structure_feature


class ViTLayerWithSIL(nn.Module):
    def __init__(self, config: ViTConfig, use_sil=False):
        super().__init__()
        self.hf_layer = ViTLayer(config)
        self.use_sil = use_sil

        if self.use_sil:
            num_patches = (config.image_size // config.patch_size) ** 2
            h = config.image_size // config.patch_size
            w = config.image_size // config.patch_size

            self.sil_module = SILModule(config.hidden_size, num_patches, h, w)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        need_attn = output_attentions or self.use_sil

        attention_outputs = self.hf_layer.attention(
            hidden_states, head_mask, output_attentions=need_attn
        )
        attention_output = attention_outputs[0]
        attn_weights = attention_outputs[1] if need_attn else None

        if self.use_sil:
            patches = attention_output[:, 1:, :]
            struct_feat = self.sil_module(patches, attn_weights)
            attention_output[:, 0, :] = attention_output[:, 0, :] + struct_feat

        layer_output = self.hf_layer.intermediate(attention_output)
        layer_output = self.hf_layer.output(layer_output, attention_output)

        outputs = (layer_output,)
        if output_attentions:
            outputs = (*outputs, attn_weights)

        return outputs

class SIMTransHF(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224', num_classes=200):
        super().__init__()

        print(f"Loading pretrained ViT: {model_name}...")
        self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        config = self.vit.config

        total_layers = len(self.vit.encoder.layer)
        sil_start_layer = total_layers - 3

        for i in range(total_layers):
            original_layer = self.vit.encoder.layer[i]
            use_sil = i >= sil_start_layer

            new_layer = ViTLayerWithSIL(config, use_sil=use_sil)

            new_layer.hf_layer.load_state_dict(original_layer.state_dict())

            self.vit.encoder.layer[i] = new_layer

        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size * 3, num_classes)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values, output_hidden_states=True)

        all_hidden_states = outputs.hidden_states

        cls_features = []

        for i in range(3, 0, -1):
            layer_out = all_hidden_states[-i]
            cls_token = layer_out[:, 0, :]
            cls_features.append(self.norm(cls_token))

        final_feature = torch.cat(cls_features, dim=-1)

        logits = self.head(final_feature)

        return logits
