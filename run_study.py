"""
The whole PatchCore study on the 15 MVTec AD categories.

For each backbone and category: extract patch features of all training (good) and test images once, then
  coreset  : memory bank = 1% / 10% of all training patches          -> accuracy vs memory and speed (RQ1, RQ2, RQ4)
  few-shot : memory bank from only k = 1, 2, 4, 8, 16 good images     -> how much good data is needed (RQ3)
             (3 random draws of the k images, averaged)

Metrics: image-level AUROC (is the part defective?) and pixel-level AUROC (where is the defect?).
Results are saved after every category to results/<backbone>.json, so the run can be resumed.

  python run_study.py --root data/mvtec-ad
  python run_study.py --root data/mvtec-ad --backbones wrn50 --categories bottle screw   # quick
  python run_study.py --root data/mvtec-ad --backbones convnext_t --standardize          # RQ5

--standardize (RQ5): z-score every feature channel with the mean / std of the GOOD training patches in the
memory bank (in few-shot: of the k images only), then apply the same transform to the test patches. ConvNeXt
has one "massive activation" channel that holds ~95% of all feature variance, so plain Euclidean distance
is almost a 1-D comparison along that channel; standardising gives every channel an equal say.
Results go to results/<backbone>_std.json.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from patchcore.data import CATEGORIES, index, load_image, load_mask
from patchcore.features import BACKBONES, PatchFeatures
from patchcore.patchcore import greedy_coreset, score

CORESETS = [0.01, 0.1]          # the paper's two main settings; a 100% bank is ~10x slower on a 4 GB laptop GPU
SHOTS = [1, 2, 4, 8, 16]
DRAWS = 3


@torch.no_grad()
def extract(model, items, dev, batch=16):
    feats = []
    for i in range(0, len(items), batch):
        x = torch.stack([load_image(r['path']) for r in items[i:i + batch]]).to(dev)
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            f, grid = model(x)
        feats.append(f.half())
    return torch.cat(feats), grid


CHUNK = 65536


def channel_stats(patches):
    """Per-channel mean and variance of (N, D) patches, in chunks with float64 sums: a full fp32 copy of
    300k x 1536 features would not fit next to everything else in 4 GB, and fp32 sums of x^2 lose precision
    for channels whose values reach ~1000."""
    n = patches.shape[0]
    s1 = s2 = 0
    for i in range(0, n, CHUNK):
        c = patches[i:i + CHUNK].double()
        s1, s2 = s1 + c.sum(0), s2 + c.pow(2).sum(0)
    mu = s1 / n
    var = ((s2 / n - mu ** 2) * n / max(n - 1, 1)).clamp(min=0)
    return mu.float(), var.float()


def standardizer(patches, enabled):
    """Per-channel z-score fitted on good training patches only (identity when disabled)."""
    if not enabled:
        return lambda x: x
    mu, var = channel_stats(patches)
    sd = var.sqrt() + 1e-6

    def apply(x):
        out = torch.empty(x.shape, dtype=torch.float16, device=x.device)
        flat, o = x.reshape(-1, x.shape[-1]), out.view(-1, x.shape[-1])
        for i in range(0, flat.shape[0], CHUNK):
            o[i:i + CHUNK] = ((flat[i:i + CHUNK].float() - mu) / sd).half()
        return out
    return apply


def top_channel_share(patches):
    """Fraction of the total feature variance held by the single largest-variance channel."""
    var = channel_stats(patches)[1]
    return float(var.max() / var.sum()), int(var.argmax())


def evaluate(test_feats, grid, bank, labels, masks, dev):
    scores, maps = [], []
    if dev == 'cuda':
        torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(0, len(test_feats), 8):
        s, m = score(test_feats[i:i + 8], grid, bank)
        scores.append(s.float().cpu())
        maps.append(m.half().cpu())
    if dev == 'cuda':
        torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t) / len(test_feats)
    scores, maps = torch.cat(scores).numpy(), torch.cat(maps)
    img_auc = roc_auc_score(labels, scores)
    pix_auc = roc_auc_score(masks.flatten().numpy(), maps.float().flatten().numpy())
    return dict(img_auroc=100 * img_auc, pix_auroc=100 * pix_auc, bank=int(bank.shape[0]), ms_per_image=ms), maps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/mvtec-ad')
    p.add_argument('--backbones', nargs='+', default=list(BACKBONES))
    p.add_argument('--categories', nargs='+', default=CATEGORIES)
    p.add_argument('--out', default='results')
    p.add_argument('--standardize', action='store_true', help='z-score feature channels (RQ5)')
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    idx = index(a.root)
    os.makedirs(os.path.join(a.out, 'maps'), exist_ok=True)

    for bb in a.backbones:
        tag = bb + ('_std' if a.standardize else '')
        path = os.path.join(a.out, f'{tag}.json')
        res = json.load(open(path)) if os.path.exists(path) else {}
        model = PatchFeatures(bb).to(dev)
        for cat in a.categories:
            if cat in res:
                print(f"{bb}/{cat}: done already")
                continue
            t0 = time.time()
            train, test = idx[(cat, 'train')], idx[(cat, 'test')]
            train_feats, grid = extract(model, train, dev)               # (N, P, D)
            test_feats, _ = extract(model, test, dev)
            labels = np.array([r['defect'] != 'good' for r in test], dtype=int)
            masks = torch.stack([load_mask(r['mask']) for r in test])
            all_patches = train_feats.reshape(-1, train_feats.shape[-1])
            share, ch = top_channel_share(all_patches)
            r = dict(n_train=len(train), n_test=len(test), n_defective=int(labels.sum()), grid=list(grid),
                     dim=int(all_patches.shape[1]), top_channel_var_share=share, top_channel=ch, coreset={}, fewshot={})

            norm = standardizer(all_patches, a.standardize)
            normed, test_normed = norm(all_patches), norm(test_feats)
            for frac in CORESETS:
                t = time.time()
                bank = normed[greedy_coreset(normed, frac)]
                metrics, maps = evaluate(test_normed, grid, bank, labels, masks, dev)
                metrics['coreset_seconds'] = time.time() - t - metrics['ms_per_image'] * len(test) / 1000
                r['coreset'][str(frac)] = metrics
                if frac == 0.1:   # keep a few maps for the qualitative figure
                    bad = np.flatnonzero(labels)[:6]
                    np.savez_compressed(os.path.join(a.out, 'maps', f'{tag}_{cat}.npz'), maps=maps[bad].numpy(),
                                        masks=masks[bad].numpy(), paths=np.array([test[i]['path'] for i in bad]),
                                        scores_good_max=float(maps[labels == 0].float().amax().item()))
                del bank
            for k in SHOTS:
                runs = []
                for draw in range(DRAWS):
                    pick = random.Random(draw).sample(range(len(train)), k)
                    bank = train_feats[pick].reshape(-1, train_feats.shape[-1])
                    norm_k = standardizer(bank, a.standardize)       # statistics from the k images only
                    runs.append(evaluate(norm_k(test_feats), grid, norm_k(bank), labels, masks, dev)[0])
                r['fewshot'][str(k)] = {m: float(np.mean([x[m] for x in runs])) for m in ('img_auroc', 'pix_auroc')}
                r['fewshot'][str(k)].update({f'{m}_std': float(np.std([x[m] for x in runs])) for m in ('img_auroc', 'pix_auroc')})
            res[cat] = r
            json.dump(res, open(path, 'w'), indent=1)
            c10 = r['coreset']['0.1']
            print(f"{tag}/{cat}: image AUROC {c10['img_auroc']:.1f}  pixel AUROC {c10['pix_auroc']:.1f}  "
                  f"(10% coreset)  16-shot {r['fewshot']['16']['img_auroc']:.1f}  [{time.time() - t0:.0f} s]")
            del train_feats, test_feats, all_patches, normed, test_normed
            if dev == 'cuda':
                torch.cuda.empty_cache()
        del model


if __name__ == '__main__':
    main()
