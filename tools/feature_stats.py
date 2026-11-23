"""
RQ5 evidence: how unevenly is feature variance spread over channels, per backbone?

For every backbone and category, over all patches of the GOOD training images:
  share   fraction of the total variance held by the single largest-variance channel
  channel which channel that is, and the largest |value| it takes
  median  the median over channels of each channel's largest |value| (a "typical" channel)

  python tools/feature_stats.py --root data/mvtec-ad   -> results/feature_stats.json and .md

ConvNeXt-T: one channel (index 223 of the 576 concatenated = stage-2 channel 31) holds ~95% of the variance
with values above 1000, while a typical channel stays below ~5. Euclidean nearest-neighbour distance is then
almost a comparison along that single channel. WideResNet-50 and DINOv2 spread their variance over many channels.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patchcore.data import CATEGORIES, index
from patchcore.features import BACKBONES, PatchFeatures
from run_study import channel_stats, extract

NAMES = {'wrn50': 'WideResNet-50', 'convnext_t': 'ConvNeXt-T', 'dinov2_s': 'DINOv2 ViT-S/14'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/mvtec-ad')
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    idx = index(a.root)
    out = {}
    for bb in BACKBONES:
        model = PatchFeatures(bb).to(dev)
        out[bb] = {}
        for cat in CATEGORIES:
            feats, _ = extract(model, idx[(cat, 'train')], dev)
            patches = feats.reshape(-1, feats.shape[-1])
            var = channel_stats(patches)[1]
            ch = int(var.argmax())
            absmax = torch.stack([patches[i:i + 65536].abs().amax(0) for i in range(0, len(patches), 65536)]).amax(0).float()
            out[bb][cat] = dict(share=float(var.max() / var.sum()), channel=ch, channel_absmax=float(absmax[ch]),
                                median_absmax=float(absmax.median()))
            print(f"{bb}/{cat}: top channel {ch} holds {100 * out[bb][cat]['share']:.1f}% of the variance, "
                  f"max |value| {absmax[ch]:.0f} (median channel {absmax.median():.1f})", flush=True)
            del feats, patches
        del model
        if dev == 'cuda':
            torch.cuda.empty_cache()
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/feature_stats.json', 'w'), indent=1)
    lines = ["| backbone | variance held by the top channel (mean, min to max over 15 categories) | top channel | its max abs value | typical channel's max abs value |",
             "|---|---|---|---|---|"]
    for bb, d in out.items():
        sh = np.array([v['share'] for v in d.values()]) * 100
        chans = sorted({v['channel'] for v in d.values()})
        lines.append(f"| {NAMES[bb]} | {sh.mean():.1f}% ({sh.min():.1f} to {sh.max():.1f}%) | "
                     f"{', '.join(map(str, chans))} | {np.median([v['channel_absmax'] for v in d.values()]):.0f} | "
                     f"{np.median([v['median_absmax'] for v in d.values()]):.1f} |")
    open('results/feature_stats.md', 'w').write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == '__main__':
    main()
