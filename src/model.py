import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# EfficientNetV2-S produces a feature map with 1280 channels at the end
_FEAT_CH = 1280


# --- Channel Attention ---
# This part looks at all the channels in the feature map and decides
# which ones are more important. It squeezes each channel into a single
# number, passes it through a small network, then multiplies back.
class ChannelAttention(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        # small MLP that scores each channel
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        # average pooling and max pooling across spatial dimensions
        avg = x.mean(dim=[2, 3])
        mx  = x.flatten(2).max(dim=2).values
        # combine both and get a weight per channel (0 to 1)
        attn = self.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn.view(b, c, 1, 1)


# --- Spatial Attention ---
# After channel attention, this part looks at WHERE in the image
# to focus. It finds which spatial locations are most important.
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        # one conv layer that takes avg + max across channels and outputs a spatial mask
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        # generate spatial mask and apply it
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


# --- CBAM (Convolutional Block Attention Module) ---
# Runs channel attention first, then spatial attention.
# This helps the model focus on the right features in the right places
# which is very useful for small eye images.
class CBAM(nn.Module):

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


# --- Main Model ---
# EfficientNetV2-S is our backbone (pretrained on ImageNet).
# We insert CBAM right before the final pooling layer.
# The output is 2 classes: open or closed.
class EfficientNetV2S_CBAM(nn.Module):

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        # load pretrained EfficientNetV2-S
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        self.features   = backbone.features   # the main convolutional blocks
        self.cbam       = CBAM(_FEAT_CH)      # our attention module
        self.avgpool    = backbone.avgpool    # shrinks spatial dims to 1x1
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),  # dropout to reduce overfitting
            nn.Linear(_FEAT_CH, num_classes),
        )

        # in phase 1 we freeze the backbone so only CBAM + head get trained
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)    # extract features using EfficientNetV2-S
        x = self.cbam(x)        # apply attention
        x = self.avgpool(x)     # pool to vector
        x = torch.flatten(x, 1)
        return self.classifier(x)


# build_model is the function used by train.py, evaluate.py, and demo.py
def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    return EfficientNetV2S_CBAM(num_classes=num_classes, freeze_backbone=freeze_backbone)


# called at the start of phase 2 to allow training the full network
def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    m = build_model(freeze_backbone=True)
    dummy = torch.zeros(2, 3, 224, 224)
    out = m(dummy)
    print(f"Output shape : {out.shape}")

    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    print(f"Params       : {trainable:,} trainable / {total:,} total")
    print()
    print("Per-module trainable params (Phase 1):")
    for name, mod in m.named_children():
        tr  = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in mod.parameters())
        print(f"  {name:12s}  {tr:>10,} / {tot:>10,}")
