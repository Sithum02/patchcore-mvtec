"""
Tables and figures from results/<backbone>.json:
  results/summary.md                 per-category AUROC (10% coreset), coreset trade-off, few-shot table
  results/figures/fewshot.png        mean image / pixel AUROC vs number of good training images
  results/figures/heatmaps.png       anomaly maps vs ground truth on defective test images
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patchcore.data import CATEGORIES, raw_image

NAMES = {'wrn50': 'WideResNet-50 (paper)', 'convnext_t': 'ConvNeXt-T', 'convnext_t_std': 'ConvNeXt-T, standardised',
         'dinov2_s': 'DINOv2 ViT-S/14'}
os.makedirs('results/figures', exist_ok=True)
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob('results/*.json'))}
R = {k: v for k, v in R.items() if k in NAMES}
BBS = [b for b in NAMES if b in R]


def mean(bb, part, key, metric):
    vals = [R[bb][c][part][key][metric] for c in CATEGORIES if c in R[bb]]
    return float(np.mean(vals)) if vals else float('nan')


def per_category_table():
    head = "| category | " + " | ".join(f"{NAMES[b]} image / pixel" for b in BBS) + " |"
    lines = [head, "|---" * (len(BBS) + 1) + "|"]
    for c in CATEGORIES:
        cells = [f"{R[b][c]['coreset']['0.1']['img_auroc']:.1f} / {R[b][c]['coreset']['0.1']['pix_auroc']:.1f}"
                 if c in R[b] else '' for b in BBS]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("| **mean** | " + " | ".join(
        f"**{mean(b, 'coreset', '0.1', 'img_auroc'):.1f} / {mean(b, 'coreset', '0.1', 'pix_auroc'):.1f}**" for b in BBS) + " |")
    return "\n".join(lines)


def coreset_table():
    lines = ["| backbone | coreset | mean image AUROC | mean pixel AUROC | mean bank size (patches) | ms / image |",
             "|---|---|---|---|---|---|"]
    for b in BBS:
        for f in ('0.01', '0.1'):
            cats = [c for c in CATEGORIES if c in R[b]]
            bank = np.mean([R[b][c]['coreset'][f]['bank'] for c in cats])
            ms = np.mean([R[b][c]['coreset'][f]['ms_per_image'] for c in cats])
            lines.append(f"| {NAMES[b]} | {float(f) * 100:g}% | {mean(b, 'coreset', f, 'img_auroc'):.1f} | "
                         f"{mean(b, 'coreset', f, 'pix_auroc'):.1f} | {bank:,.0f} | {ms:.1f} |")
    return "\n".join(lines)


def fewshot_table():
    shots = ['1', '2', '4', '8', '16']
    lines = ["| backbone | " + " | ".join(f"{k}-shot" for k in shots) + " | all images (10% coreset) |",
             "|---" * (len(shots) + 2) + "|"]
    for b in BBS:
        lines.append(f"| {NAMES[b]} | " + " | ".join(f"{mean(b, 'fewshot', k, 'img_auroc'):.1f}" for k in shots)
                     + f" | {mean(b, 'coreset', '0.1', 'img_auroc'):.1f} |")
    return "\n".join(lines)


def fewshot_figure():
    shots = [1, 2, 4, 8, 16]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in zip(axes, ('img_auroc', 'pix_auroc'), ('image AUROC (detection)', 'pixel AUROC (localisation)')):
        for b in BBS:
            line, = ax.plot(shots, [mean(b, 'fewshot', str(k), metric) for k in shots], marker='o', label=NAMES[b])
            ax.axhline(mean(b, 'coreset', '0.1', metric), color=line.get_color(), ls='--', lw=1)
        ax.set_xscale('log', base=2)
        ax.set_xticks(shots, [str(k) for k in shots])
        ax.set_xlabel('good training images used')
        ax.set_ylabel(f'mean {title} (%)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Few-shot PatchCore on MVTec AD (dashed: all training images)', fontsize=10)
    fig.tight_layout()
    fig.savefig('results/figures/fewshot.png', dpi=150)
    plt.close(fig)


def heatmap_figure(cats=('bottle', 'cable', 'hazelnut', 'screw', 'carpet', 'transistor'), bbs=None,
                   path='results/figures/heatmaps.png'):
    bbs = [b for b in (bbs or BBS) if os.path.exists(f'results/maps/{b}_{cats[0]}.npz')]
    if not bbs:
        return
    cols = 2 + len(bbs)
    fig, axes = plt.subplots(len(cats), cols, figsize=(2.0 * cols, 2.0 * len(cats)))
    for r, c in enumerate(cats):
        d0 = np.load(f'results/maps/{bbs[0]}_{c}.npz')
        axes[r, 0].imshow(raw_image(str(d0['paths'][0])))
        axes[r, 1].imshow(d0['masks'][0], cmap='gray')
        for j, b in enumerate(bbs):
            d = np.load(f'results/maps/{b}_{c}.npz')
            axes[r, 2 + j].imshow(raw_image(str(d['paths'][0])))
            axes[r, 2 + j].imshow(d['maps'][0].astype(np.float32), cmap='jet', alpha=0.5)
            if r == 0:
                axes[r, 2 + j].set_title(NAMES[b], fontsize=8)
        axes[r, 0].set_ylabel(c, fontsize=8)
    axes[0, 0].set_title('test image', fontsize=8)
    axes[0, 1].set_title('true defect', fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    text = ("## Per category, 10% coreset (image AUROC / pixel AUROC, %)\n\n" + per_category_table() +
            "\n\n## Coreset size: accuracy vs memory vs speed (means over categories; ms = nearest-neighbour "
            "scoring per test image, backbone forward pass not included)\n\n" + coreset_table() +
            "\n\n## Few-shot: mean image AUROC with k good training images (3 random draws)\n\n" + fewshot_table() + "\n")
    if os.path.exists('results/feature_stats.md'):
        text += ("\n## Feature variance per channel (good training patches; tools/feature_stats.py)\n\n"
                 + open('results/feature_stats.md').read())
    open('results/summary.md', 'w').write(text)
    print(text)
    fewshot_figure()
    heatmap_figure()
    heatmap_figure(cats=('cable', 'screw', 'zipper'), path='results/figures/heatmaps_short.png')
    print("figures in results/figures/")
