"""
PatchCore (Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022).

Training = no training. Collect the patch features of all defect-free training images into a MEMORY BANK.
Testing  = for every patch of a test image, the distance to its nearest neighbour in the memory bank.
           A defect produces patches unlike anything seen on good parts, so large distance = anomalous.
             anomaly map  : nearest-neighbour distance per patch -> upsample to 224x224 -> Gaussian blur (sigma 4)
             image score  : the largest patch distance in the image

Coreset subsampling: neighbouring patches of good images are very similar, so most of the bank is redundant.
Greedy k-center selection keeps a small subset (e.g. 1% or 10%) that still COVERS the feature space: each new
pick is the patch farthest from everything already picked. Distances are computed on a random 128-d
projection to save time, as in the paper.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_coreset(feats, fraction, proj_dim=128, seed=0):
    """feats (N, D) -> indices of the selected coreset (about fraction * N)."""
    n = feats.shape[0]
    m = max(1, int(n * fraction))
    if m >= n:
        return torch.arange(n, device=feats.device)
    g = torch.Generator(device=feats.device).manual_seed(seed)
    proj = torch.randn(feats.shape[1], proj_dim, device=feats.device, generator=g) / proj_dim ** 0.5
    z = feats.float() @ proj                                         # (N, 128)
    z_sq = z.pow(2).sum(1)
    sel = torch.empty(m, dtype=torch.long, device=feats.device)
    sel[0] = torch.randint(n, (1,), device=feats.device, generator=g)
    # squared distance of every point to its nearest selected point; one matrix-vector product per step
    min_d = z_sq - 2 * (z @ z[sel[0]]) + z_sq[sel[0]]
    for i in range(1, m):
        j = torch.argmax(min_d)                                       # stays on the GPU: no CPU sync per step
        sel[i] = j
        min_d = torch.minimum(min_d, z_sq - 2 * (z @ z[j]) + z_sq[j])
    return sel


@torch.no_grad()
def nn_distance(queries, bank, q_chunk=1024, b_chunk=32768, candidates=10):
    """For every query (Q, D), the Euclidean distance to its nearest neighbour in bank (M, D).

    Two stages, both chunked so even a large bank fits in 4 GB of GPU memory:
      1. shortlist the `candidates` nearest bank entries with the fast matrix form
         ||q - b||^2 = ||q||^2 - 2 q.b + ||b||^2   (fp32)
      2. recompute the shortlisted distances exactly as ||q - b|| by direct subtraction.
    Why: ConvNeXt features contain a few "massive activation" channels (values > 1000 while the rest are ~1).
    Then ||q||^2 is huge compared with the distance itself and the fast form cancels catastrophically
    (5% error even in fp32), which would unfairly handicap that backbone. The exact second stage removes this.
    """
    bank_sq = bank.float().pow(2).sum(1)
    out = []
    for i in range(0, queries.shape[0], q_chunk):
        q = queries[i:i + q_chunk].float()
        q_sq = q.pow(2).sum(1, keepdim=True)
        best_d = best_i = None
        for j in range(0, bank.shape[0], b_chunk):
            d2 = q_sq - 2 * (q @ bank[j:j + b_chunk].float().T) + bank_sq[j:j + b_chunk][None]
            d, idx = d2.topk(min(candidates, d2.shape[1]), dim=1, largest=False)
            idx = idx + j
            if best_d is not None:
                d, pos = torch.cat([best_d, d], 1).topk(min(candidates, best_d.shape[1] + d.shape[1]), dim=1, largest=False)
                idx = torch.cat([best_i, idx], 1).gather(1, pos)
            best_d, best_i = d, idx
        exact = (q[:, None, :] - bank[best_i].float()).pow(2).sum(-1)     # (q_chunk, candidates)
        out.append(exact.min(dim=1).values.sqrt())
    return torch.cat(out)


def gaussian_blur(maps, sigma=4):
    """maps (B, 1, H, W): blur as in PatchCore, kernel size 2 * round(4 sigma) + 1."""
    r = int(4 * sigma + 0.5)
    x = torch.arange(-r, r + 1, device=maps.device, dtype=maps.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    maps = F.conv2d(F.pad(maps, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(maps, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))


@torch.no_grad()
def score(test_feats, grid, bank, out_size=224):
    """test_feats (B, H*W, D) -> image scores (B,) and anomaly maps (B, out_size, out_size)."""
    b, p, d = test_feats.shape
    dist = nn_distance(test_feats.reshape(-1, d), bank).view(b, 1, *grid)
    maps = F.interpolate(dist, size=(out_size, out_size), mode='bilinear', align_corners=False)
    maps = gaussian_blur(maps)[:, 0]
    return dist.flatten(1).max(dim=1).values, maps
