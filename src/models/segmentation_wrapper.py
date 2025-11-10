import torch
import torch.nn as nn

class SegmentationWrapper(nn.Module):
    def __init__(self, classifier: nn.Module, mask_size: int = 224):
        super().__init__()
        self.classifier = classifier.eval()
        self.mask_size = mask_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        preds = torch.argmax(logits, dim=1)

        mask = preds.expand(-1, 1, self.mask_size, self.mask_size)  # (B, 1, H, W)
        return mask
