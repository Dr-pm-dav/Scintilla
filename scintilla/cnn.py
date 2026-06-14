from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as error:  # pragma: no cover - exercised only without optional extra
    raise ImportError("Install requirements-cnn.txt to use scintilla.cnn") from error


class MultiTaskSpectrumCNN(nn.Module):
    """1D CNN with presence and non-negative activity heads."""

    def __init__(self, channels: int, isotope_count: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=11, padding=5),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(24, 48, kernel_size=9, padding=4),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(48, 96, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(96 * 16, 192), nn.GELU(), nn.Dropout(0.12))
        self.presence_head = nn.Linear(192, isotope_count)
        self.activity_head = nn.Linear(192, isotope_count)

    def forward(self, spectra: Tensor) -> tuple[Tensor, Tensor]:
        if spectra.ndim == 2:
            spectra = spectra.unsqueeze(1)
        features = self.shared(self.encoder(spectra))
        return self.presence_head(features), F.softplus(self.activity_head(features))


def multitask_loss(
    presence_logits: Tensor,
    predicted_activities: Tensor,
    target_activities: Tensor,
    presence_threshold_bq: float = 1.0,
) -> Tensor:
    target_presence = (target_activities >= presence_threshold_bq).float()
    presence_loss = F.binary_cross_entropy_with_logits(presence_logits, target_presence)
    activity_loss = F.smooth_l1_loss(torch.log1p(predicted_activities), torch.log1p(target_activities))
    return presence_loss + activity_loss

