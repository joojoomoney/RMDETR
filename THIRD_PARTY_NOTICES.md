# Third-Party Notices

RM-DETR contains code derived from, adapted from, or based on several
open-source research projects. Third-party components remain subject to
their respective licenses and copyright notices.

## RT-DETR
Source: https://github.com/lyuwenyu/RT-DETR
License: Apache License 2.0

Portions of the training framework, transformer components, utilities,
EMA implementation, and related infrastructure are derived from RT-DETR.

The RT-DETR-derived EMA implementation retains upstream references to
YOLOv5 and pytorch-image-models (timm) in the original source comments.

## DETR
Source: https://github.com/facebookresearch/detr
License: Apache License 2.0

Portions of the criterion, Hungarian matching, transformer utilities,
box operations, distributed utilities, and evaluation code are derived
from or based on DETR.

## Deformable DETR
Source: https://github.com/fundamentalvision/Deformable-DETR
License: Apache License 2.0

Portions of the bounding-box utilities are derived from or based on
Deformable DETR. Original copyright and license headers are retained
in the relevant source files.

## PaddleDetection / HGNetV2
Source: https://github.com/PaddlePaddle/PaddleDetection
License: Apache License 2.0

The HGNetV2 backbone implementation is based on PaddleDetection and
related open-source implementations.

## D-FINE
Source: https://github.com/Peterande/D-FINE
License: Apache License 2.0

Portions of the HGNetV2 implementation and pretrained model references
are derived from or based on D-FINE.

## Detectron2
Source: https://github.com/facebookresearch/detectron2
License: Apache License 2.0

Point-sampling functionality used for mask-related matching is based on
the Detectron2 implementation.

## Mask2Former
Source: https://github.com/facebookresearch/Mask2Former
License: MIT License

Portions of the mask matching logic, including point-based mask matching,
Dice cost, and sigmoid cross-entropy cost, are based on or adapted from
Mask2Former-style implementations.

## torchvision
Source: https://github.com/pytorch/vision
License: BSD 3-Clause License

Portions of the COCO dataset and evaluation utilities are based on the
torchvision detection references.

Individual source files may contain additional copyright and attribution
notices. Those notices should be retained.

This file does not replace or modify the license terms of any
third-party component.
