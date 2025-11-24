import torch
import torch.nn as nn

class SegmentationWrapper(nn.Module):
    def __init__(self, classifier: nn.Module, mask_size: int = 224, mean=None, std=None, input_rescale=False):
        super().__init__()
        self.classifier = classifier.eval()
        self.mask_size = mask_size
        self.input_rescale = input_rescale
        
        if mean is None:
            mean = [0.0, 0.0, 0.0]
        if std is None:
            std = [1.0, 1.0, 1.0]
            
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_rescale:
            x = x / 255.0
            
        x = (x - self.mean) / self.std
        
        logits = self.classifier(x)
        
        # Ensure logits is 2D [batch, num_classes]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        probs = torch.softmax(logits, dim=1)
        mask = probs[:, :, None, None].expand(-1, -1, self.mask_size, self.mask_size) # (B, C, H, W)
        return mask
