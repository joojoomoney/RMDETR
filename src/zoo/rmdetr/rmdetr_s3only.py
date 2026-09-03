"""
RMDETR_s3only: RM-DETR variant that only uses S3 feature for mask embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.nn.transformer import MLP
from src.core import register


__all__ = [
    "RMDETR_s3only",
]


@register
class RMDETR_s3only(nn.Module):
    """
    RM-DETR model that only uses S3 (stride 16, 40x40) feature for mask embedding.
    This is a variant to test the effect of using mid-level features.
    """

    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder,
        decoder,
        multi_scale=None,
        backbone_init_dim=64,
        mask_dim=256,
        query_dim=256,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale

        # Mask embedding projection using only S3 feature
        # For ResNet50: x[1] is stage3 output with 512 channels (128 * BottleNeck.expansion=4)
        # Get the actual channel count from backbone
        s3_channels = (
            backbone.out_channels[1] if hasattr(backbone, "out_channels") else 512
        )

        self.mask_embed = MLP(query_dim, query_dim * 4, mask_dim, num_layers=3)

        # Mask feature projection from S3 only
        self.m1 = nn.Sequential(
            nn.Conv2d(s3_channels, mask_dim, kernel_size=1),
            nn.BatchNorm2d(mask_dim),
        )

        # No m2 since we're only using S3 (not combining with encoder output)
        self.fconv = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, kernel_size=1),
            nn.BatchNorm2d(mask_dim),
            nn.ReLU(),
        )

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        # Get backbone features
        x = self.backbone(x)

        # Encoder forward (using S3, S4, S5)
        encoder_out = self.encoder(x[1:])

        # Decoder forward
        output, out_query = self.decoder(encoder_out, targets)

        # Get mask from S3 features only
        mask = self.get_mask(out_query[-1], output["pred_boxes"], x[1])
        output["pred_masks"] = mask

        return output

    def get_mask(self, query, box, from_backbone):
        """
        Generate masks using only S3 backbone features (no encoder features).

        Args:
            query: decoder output queries [B, N, C]
            box: predicted boxes [B, N, 4]
            from_backbone: S3 features from backbone [B, 512, H, W]
        """
        # Process S3 features
        m1 = self.m1(from_backbone)
        mask = self.fconv(m1)

        # Get query embedding for masks
        n_query = box.shape[1]
        mask_embed = self.mask_embed(query[:, -n_query:, :])

        # Generate masks
        mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask)

        return mask

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
