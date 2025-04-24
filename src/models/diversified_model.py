import torch
import torch.nn as nn
from torchvision.models import resnet50


class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(weights="DEFAULT")
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.conv1x1 = nn.Conv2d(2048, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1x1(x)
        return x


class DiversificationBlock(nn.Module):
    def __init__(self, p_peak=0.5, p_patch=0.5, grid_size=7, alpha=0.1):
        super().__init__()
        self.p_peak = p_peak
        self.p_patch = p_patch
        self.grid_size = grid_size
        self.alpha = alpha

    def forward(self, activation_maps):
        batch_size, num_classes, H, W = activation_maps.shape
        device = activation_maps.device

        masks = torch.zeros_like(activation_maps).to(device)

        for b in range(batch_size):
            for c in range(num_classes):
                map_ = activation_maps[b, c]

                max_val = torch.max(map_)
                peak_mask = (map_ == max_val).float()
                r_peak = torch.bernoulli(torch.tensor(self.p_peak, device=device)).item()
                B_prime = r_peak * peak_mask

                patches = map_.unfold(0, self.grid_size, self.grid_size).unfold(1, self.grid_size, self.grid_size)
                patch_mask = torch.zeros_like(map_)

                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        r_patch = torch.bernoulli(torch.tensor(self.p_patch, device=device)).item()
                        if r_patch:
                            x_start = i * self.grid_size
                            y_start = j * self.grid_size
                            patch = map_[x_start:x_start+self.grid_size, y_start:y_start+self.grid_size]
                            if not (patch == max_val).any():
                                patch_mask[x_start:x_start+self.grid_size, y_start:y_start+self.grid_size] = 1

                B = B_prime + patch_mask
                masks[b, c] = B

        suppressed_maps = activation_maps * (1 - masks) + activation_maps * masks * self.alpha
        return suppressed_maps


class GradientBoostingLoss(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.k = k

    def forward(self, logits, labels):
        batch_size, num_classes = logits.shape
        loss = 0.0

        for b in range(batch_size):
            logit = logits[b]
            label = labels[b]

            device = logit.device
            neg_logit = logit[torch.arange(num_classes, device=device) != label]
            neg_labels = torch.arange(num_classes, device=device)[torch.arange(num_classes, device=device) != label]

            topk_values, topk_indices = torch.topk(neg_logit, self.k)
            J_prime = neg_labels[topk_indices]

            numerator = torch.exp(logit[label])
            denominator = numerator + torch.sum(torch.exp(logit[J_prime]))
            loss_b = -torch.log(numerator / denominator)
            loss += loss_b

        return loss / batch_size


class FineGrainedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = ModifiedResNet50(num_classes)
        self.diversification = DiversificationBlock()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, is_train=True):
        activation_maps = self.feature_extractor(x)  # [batch, C, H, W]

        if is_train:
            activation_maps = self.diversification(activation_maps)

        pooled = self.pool(activation_maps).squeeze()  # [batch, C]
        return pooled
