"""
Frozen, ImageNet-pretrained backbones that turn an image into a grid of local patch features.

PatchCore uses MID-level features: deep enough to be semantic, shallow enough to keep spatial detail.
For a 224x224 input:

  wrn50      WideResNet-50-2 (the paper's choice)   layer2 (512 ch, 28x28) + layer3 (1024 ch, 14x14)
  convnext_t ConvNeXt-T (ImageNet-1k)               stage 1 (192 ch, 28x28) + stage 2 (384 ch, 14x14)
  dinov2_s   DINOv2 ViT-S/14 (self-supervised)      blocks 8 and 11 (384 ch each, 16x16)

Each feature map is smoothed with a 3x3 average pool (PatchCore's "local neighbourhood aggregation": every
patch also sees its neighbours), the coarser map is upsampled to the finer one, and the two are
concatenated along channels. One feature vector per grid cell = one "patch".
"""
import timm
import torch
import torch.nn.functional as F

BACKBONES = {
    'wrn50': dict(name='wide_resnet50_2.tv_in1k', out_indices=(2, 3)),
    'convnext_t': dict(name='convnext_tiny.fb_in1k', out_indices=(1, 2)),
    'dinov2_s': dict(name='vit_small_patch14_dinov2.lvd142m', out_indices=(8, 11)),
}


class PatchFeatures(torch.nn.Module):
    def __init__(self, backbone='wrn50'):
        super().__init__()
        spec = BACKBONES[backbone]
        kw = dict(img_size=224) if 'dinov2' in spec['name'] else {}
        self.net = timm.create_model(spec['name'], pretrained=True, features_only=True,
                                     out_indices=spec['out_indices'], **kw).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        """x: (B, 3, 224, 224) -> features (B, H*W, D) and grid size (H, W)."""
        maps = [F.avg_pool2d(f.float(), 3, stride=1, padding=1) for f in self.net(x)]
        h, w = maps[0].shape[-2:]
        maps = [m if m.shape[-2:] == (h, w) else F.interpolate(m, size=(h, w), mode='bilinear', align_corners=False)
                for m in maps]
        f = torch.cat(maps, dim=1)                       # (B, D, H, W)
        return f.flatten(2).transpose(1, 2), (h, w)
