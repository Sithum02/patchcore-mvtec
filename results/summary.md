## Per category, 10% coreset (image AUROC / pixel AUROC, %)

| category | WideResNet-50 (paper) image / pixel | ConvNeXt-T image / pixel | ConvNeXt-T, standardised image / pixel | DINOv2 ViT-S/14 image / pixel |
|---|---|---|---|---|
| bottle | 100.0 / 98.3 | 100.0 / 97.3 | 100.0 / 98.1 | 100.0 / 97.6 |
| cable | 99.7 / 98.5 | 99.0 / 97.7 | 99.8 / 98.2 | 95.7 / 95.0 |
| capsule | 97.5 / 98.7 | 86.7 / 97.9 | 95.1 / 98.7 | 93.7 / 97.7 |
| carpet | 98.8 / 98.8 | 100.0 / 98.9 | 100.0 / 99.0 | 99.9 / 98.8 |
| grid | 97.7 / 97.6 | 83.0 / 94.1 | 95.7 / 96.7 | 99.0 / 96.4 |
| hazelnut | 100.0 / 98.4 | 100.0 / 98.5 | 100.0 / 98.7 | 99.8 / 99.0 |
| leather | 100.0 / 99.1 | 100.0 / 98.1 | 100.0 / 98.9 | 100.0 / 97.5 |
| metal_nut | 99.9 / 98.5 | 98.2 / 96.7 | 99.9 / 98.3 | 99.5 / 98.0 |
| pill | 94.7 / 97.6 | 91.3 / 97.2 | 95.3 / 98.1 | 96.0 / 98.3 |
| screw | 98.1 / 98.9 | 94.3 / 95.9 | 95.3 / 98.8 | 90.0 / 92.0 |
| tile | 98.6 / 95.3 | 98.8 / 93.8 | 98.9 / 95.4 | 99.6 / 94.1 |
| toothbrush | 99.4 / 98.6 | 90.0 / 97.1 | 96.1 / 98.3 | 98.3 / 97.7 |
| transistor | 99.8 / 97.7 | 99.4 / 98.0 | 100.0 / 97.8 | 95.2 / 96.8 |
| wood | 99.0 / 93.8 | 99.8 / 91.3 | 99.4 / 94.5 | 98.9 / 91.4 |
| zipper | 99.4 / 98.3 | 88.8 / 92.0 | 97.5 / 95.9 | 99.2 / 93.2 |
| **mean** | **98.8 / 97.9** | **95.3 / 96.3** | **98.2 / 97.7** | **97.6 / 96.2** |

## Coreset size: accuracy vs memory vs speed (means over categories; ms = nearest-neighbour scoring per test image, backbone forward pass not included)

| backbone | coreset | mean image AUROC | mean pixel AUROC | mean bank size (patches) | ms / image |
|---|---|---|---|---|---|
| WideResNet-50 (paper) | 1% | 98.7 | 97.7 | 1,896 | 6.0 |
| WideResNet-50 (paper) | 10% | 98.8 | 97.9 | 18,967 | 39.6 |
| ConvNeXt-T | 1% | 94.3 | 95.0 | 1,896 | 2.9 |
| ConvNeXt-T | 10% | 95.3 | 96.3 | 18,967 | 14.2 |
| ConvNeXt-T, standardised | 1% | 98.2 | 97.6 | 1,896 | 2.9 |
| ConvNeXt-T, standardised | 10% | 98.2 | 97.7 | 18,967 | 14.1 |
| DINOv2 ViT-S/14 | 1% | 97.1 | 95.6 | 619 | 1.2 |
| DINOv2 ViT-S/14 | 10% | 97.6 | 96.2 | 6,193 | 2.2 |

## Few-shot: mean image AUROC with k good training images (3 random draws)

| backbone | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot | all images (10% coreset) |
|---|---|---|---|---|---|---|
| WideResNet-50 (paper) | 84.3 | 86.6 | 89.1 | 92.1 | 94.0 | 98.8 |
| ConvNeXt-T | 68.7 | 75.9 | 81.0 | 85.9 | 88.7 | 95.3 |
| ConvNeXt-T, standardised | 85.5 | 86.8 | 89.6 | 91.8 | 93.6 | 98.2 |
| DINOv2 ViT-S/14 | 88.8 | 90.3 | 91.8 | 93.7 | 94.8 | 97.6 |

## Feature variance per channel (good training patches; tools/feature_stats.py)

| backbone | variance held by the top channel (mean, min to max over 15 categories) | top channel | its max abs value | typical channel's max abs value |
|---|---|---|---|---|
| WideResNet-50 | 1.4% (0.6 to 5.2%) | 66, 223, 407, 608, 687, 728, 868, 957, 978, 1132, 1169, 1176, 1181, 1334, 1388 | 1 | 0.3 |
| ConvNeXt-T | 91.3% (69.2 to 97.5%) | 223 | 1143 | 4.5 |
| DINOv2 ViT-S/14 | 2.5% (1.3 to 4.8%) | 463, 501, 522, 538, 662, 753 | 8 | 0.7 |
