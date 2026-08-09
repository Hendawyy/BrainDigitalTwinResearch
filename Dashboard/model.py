import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121


class MultimodalTransformer(nn.Module):
    def __init__(self, tabular_dim=4, num_classes=3,
                 transformer_heads=8, transformer_dim=512,
                 transformer_layers=2, dropout=0.1):
        super().__init__()

        self.cnn_backbone = DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=1024)

        total_dim     = 1024 + tabular_dim
        projected_dim = ((total_dim + transformer_heads - 1)
                         // transformer_heads) * transformer_heads

        self.projection = nn.Sequential(
            nn.Linear(total_dim, projected_dim),
            nn.LayerNorm(projected_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projected_dim, nhead=transformer_heads,
            dim_feedforward=transformer_dim, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers)

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(projected_dim, num_classes))

    def forward(self, image, tabular):
        img_emb = self.cnn_backbone(image).flatten(1)
        fused   = torch.cat([img_emb, tabular], dim=1)
        proj    = self.projection(fused).unsqueeze(1)
        enc     = self.transformer_encoder(proj).squeeze(1)
        return self.classifier_head(enc)
