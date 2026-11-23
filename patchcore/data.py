"""
MVTec AD (Bergmann et al., CVPR 2019): 15 categories of industrial objects and textures. Training images are
all defect-free ("good"); the test set mixes good and defective images, with a pixel mask for every defect.

Read from the FiftyOne export on the Hugging Face Hub (Voxel51/mvtec-ad, CC BY-NC-SA 4.0), whose
samples.json lists every image with its category, defect type, split and mask path.

Pre-processing as in PatchCore: resize to 256, centre-crop 224, ImageNet normalisation.
"""
import json
import os

import numpy as np
import torch
from PIL import Image

CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
              'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def index(root):
    samples = json.load(open(os.path.join(root, 'samples.json')))['samples']
    out = {}
    for s in samples:
        mask = s.get('defect_mask')
        out.setdefault((s['category']['label'], s['split']), []).append(dict(
            path=os.path.join(root, s['filepath']), defect=s['defect']['label'],
            mask=os.path.join(root, mask['mask_path']) if mask else None))
    for v in out.values():
        v.sort(key=lambda r: r['path'])
    return out


def _resize_crop(img, resize=256, crop=224, resample=Image.BILINEAR):
    w, h = img.size
    s = resize / min(w, h)
    img = img.resize((round(w * s), round(h * s)), resample)
    w, h = img.size
    l, t = (w - crop) // 2, (h - crop) // 2
    return img.crop((l, t, l + crop, t + crop))


def load_image(path, crop=224):
    img = _resize_crop(Image.open(path).convert('RGB'), crop=crop)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


def load_mask(path, crop=224):
    if path is None:
        return torch.zeros(crop, crop, dtype=torch.bool)
    m = _resize_crop(Image.open(path).convert('L'), crop=crop, resample=Image.NEAREST)
    return torch.from_numpy(np.asarray(m) > 127)


def raw_image(path, crop=224):
    """The cropped image as uint8 HxWx3, for figures."""
    return np.asarray(_resize_crop(Image.open(path).convert('RGB'), crop=crop))
