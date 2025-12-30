import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import math

class GraphConvolution(nn.Module):
    """
    Standardowa warstwa GCN: X' = ReLU(Adj * X * W + b)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (Batch, Nodes, Features)
        # adj: (Batch, Nodes, Nodes)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support) + self.bias
        return output

class StructureInformationLearning(nn.Module):
    def __init__(self, hidden_dim, num_patches=196, grid_size=14):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = num_patches
        
        # Projekcja cech wizualnych przed GCN
        self.vis_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Kodowanie informacji przestrzennej (Polar Coordinates)
        # Wejście: 2 (rho, theta), Wyjście: hidden_dim
        self.spatial_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Fuzja cech wizualnych i przestrzennych
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Warstwy GCN
        self.gcn1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def get_polar_coordinates(self, max_attn_indices, batch_size, device):
        """
        Implementacja Równań 9 i 10 z artykułu.
        Oblicza współrzędne biegunowe względem patcha o maksymalnej uwadze.
        """
        # Generowanie siatki współrzędnych (y, x)
        y_coords = torch.arange(self.grid_size, device=device).repeat_interleave(self.grid_size)
        x_coords = torch.arange(self.grid_size, device=device).repeat(self.grid_size)
        
        # Współrzędne wszystkich patchy: (1, 196, 2)
        all_coords = torch.stack([y_coords, x_coords], dim=-1).unsqueeze(0).float()
        
        # Współrzędne patcha referencyjnego (max attention) dla każdego obrazu w batchu
        # max_attn_indices: (Batch,)
        ref_y = max_attn_indices // self.grid_size
        ref_x = max_attn_indices % self.grid_size
        ref_coords = torch.stack([ref_y, ref_x], dim=-1).unsqueeze(1).float() # (Batch, 1, 2)
        
        # Obliczanie różnic
        delta = all_coords - ref_coords # (Batch, 196, 2)
        dy = delta[:, :, 0]
        dx = delta[:, :, 1]
        
        # Równanie 9: Rho (odległość)
        rho = torch.sqrt(dy**2 + dx**2)
        # Normalizacja rho do zakresu [0, 1] dla stabilności
        rho = rho / (math.sqrt(self.grid_size**2 + self.grid_size**2))
        
        # Równanie 10: Theta (kąt)
        theta = torch.atan2(dy, dx)
        # Normalizacja theta do zakresu [0, 1]
        theta = (theta + math.pi) / (2 * math.pi)
        
        # Złączenie: (Batch, 196, 2)
        polar_coords = torch.stack([rho, theta], dim=-1)
        return polar_coords

    def forward(self, hidden_states, attentions):
        """
        hidden_states: (Batch, 197, Hidden_Dim) - ostatnia warstwa ViT
        attentions: List of tensors, bierzemy ostatnią warstwę
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # 1. Przygotowanie danych
        # Usuwamy CLS token z cech wizualnych -> (Batch, 196, Dim)
        patch_features = hidden_states[:, 1:, :]
        
        # Pobieramy mapę uwagi z ostatniej warstwy dla tokenu CLS
        # attentions[-1] shape: (Batch, Num_Heads, 197, 197)
        # Uśredniamy po głowicach -> (Batch, 197, 197)
        last_attn = attentions[-1].mean(dim=1)
        # Bierzemy uwagę CLS do patchy (wiersz 0, kolumny 1:) -> (Batch, 196)
        cls_attn = last_attn[:, 0, 1:]

        # 2. Maskowanie (Równanie 8)
        # Wybieramy patche, których uwaga jest większa niż średnia
        mean_attn = cls_attn.mean(dim=1, keepdim=True)
        mask = (cls_attn > mean_attn).float() # (Batch, 196)
        
        # Zerujemy uwagę dla nieistotnych patchy
        masked_attn = cls_attn * mask
        
        # Normalizacja uwagi (żeby sumowała się do 1 lub była w rozsądnym zakresie)
        masked_attn = masked_attn / (masked_attn.sum(dim=1, keepdim=True) + 1e-8)

        # 3. Budowa Macierzy Sąsiedztwa (Równanie 11)
        # Adj = A_new * (A_new)^T
        # (Batch, 196, 1) @ (Batch, 1, 196) -> (Batch, 196, 196)
        adj = torch.bmm(masked_attn.unsqueeze(2), masked_attn.unsqueeze(1))
        
        # Dodajemy pętle własne (self-loops) i normalizujemy wierszami (standard GCN)
        eye = torch.eye(self.num_patches, device=device).unsqueeze(0)
        adj = adj + eye
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. Informacja Przestrzenna (Polar Coordinates)
        # Znajdujemy indeks patcha z maksymalną uwagą
        max_indices = torch.argmax(cls_attn, dim=1) # (Batch,)
        polar_coords = self.get_polar_coordinates(max_indices, batch_size, device)
        
        # Embedding współrzędnych
        spatial_features = self.spatial_embed(polar_coords) # (Batch, 196, Hidden_Dim)

        # 5. Fuzja cech (Concatenation -> Projection)
        # Artykuł mówi o konkatenacji cech węzłów z informacją pozycyjną
        combined_features = torch.cat([patch_features, spatial_features], dim=-1)
        node_features = self.fusion_proj(combined_features)
        
        # Aplikujemy maskę również na cechy (zerujemy cechy odrzuconych patchy)
        node_features = node_features * mask.unsqueeze(-1)

        # 6. Przetwarzanie GCN
        x = self.gcn1(node_features, adj)
        x = self.act(x)
        x = self.gcn2(x, adj) # (Batch, 196, Hidden_Dim)

        # 7. Agregacja (Global Average Pooling ważony maską)
        # Sumujemy cechy aktywnych węzłów i dzielimy przez liczbę aktywnych węzłów
        num_active = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        structure_feature = x.sum(dim=1) / num_active # (Batch, Hidden_Dim)

        return structure_feature

class MultiLevelFeatureBoosting(nn.Module):
    def __init__(self, hidden_dim, num_levels=3):
        super().__init__()
        # Fuzja cech z różnych poziomów ViT
        self.fusion = nn.Linear(hidden_dim * num_levels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, multi_level_features):
        # multi_level_features: lista tensorów [CLS_L-2, CLS_L-1, CLS_L]
        concatenated = torch.cat(multi_level_features, dim=-1)
        fused = self.fusion(concatenated)
        fused = self.norm(fused)
        return fused

class SIMTransHF(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained_model="google/vit-base-patch16-224",
        hidden_dim=768,
        num_patches=196,
        drop_rate=0.1
    ):
        super().__init__()

        # Backbone
        self.vit = ViTModel.from_pretrained(
            pretrained_model,
            output_attentions=True,
            output_hidden_states=True
        )
        self.hidden_dim = hidden_dim

        # Moduł SIL (Structure Information Learning)
        self.sil = StructureInformationLearning(
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            grid_size=int(math.sqrt(num_patches)) # Zakładamy kwadratowy grid (14x14)
        )

        # Moduł MFB (Multi-Level Feature Boosting)
        self.mfb = MultiLevelFeatureBoosting(hidden_dim=hidden_dim, num_levels=3)

        # Klasyfikator
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x)

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        # --- Multi-Level Feature Boosting (MFB) ---
        # Pobieramy tokeny CLS z ostatnich 3 warstw
        multi_level_cls = [
            hidden_states[-3][:, 0],
            hidden_states[-2][:, 0],
            hidden_states[-1][:, 0]
        ]
        mfb_feature = self.mfb(multi_level_cls)

        # --- Structure Information Learning (SIL) ---
        # Używamy ostatniej warstwy ukrytej i map uwagi
        last_hidden = hidden_states[-1]
        structure_feature = self.sil(last_hidden, attentions)

        # --- Fuzja i Klasyfikacja ---
        combined = torch.cat([mfb_feature, structure_feature], dim=-1)
        logits = self.classifier(combined)

        # Zwracamy logits ORAZ mfb_feature.
        # mfb_feature jest potrzebne do obliczenia Contrastive Loss podczas treningu.
        return logits, mfb_feature