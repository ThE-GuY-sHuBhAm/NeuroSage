# src/models/wavlm_encoder.py
"""
WavLM Acoustic Encoder

Takes raw 4-second audio → outputs 256-dim acoustic embedding.

WavLM-Large is chosen over Wav2Vec2 because:
- Trained with masked speech prediction + denoising objective
- More robust to background noise in clinical recordings
- State-of-the-art on SUPERB benchmark across all speech tasks
"""

import torch
import torch.nn as nn
from transformers import WavLMModel


class WavLMEncoder(nn.Module):

    def __init__(
        self,
        model_name : str = "microsoft/wavlm-large",
        output_dim : int = 256,
        freeze_cnn : bool = True
    ):
        super().__init__()

        # Load pretrained WavLM-Large
        self.wavlm = WavLMModel.from_pretrained(model_name)

        # Freeze the CNN feature extractor (low-level, no benefit fine-tuning)
        # Only the transformer layers get fine-tuned
        if freeze_cnn:
            for param in self.wavlm.feature_extractor.parameters():
                param.requires_grad = False

        # WavLM-Large hidden size = 1024
        wavlm_dim = self.wavlm.config.hidden_size

        # Project 1024 → 256
        self.projection = nn.Sequential(
            nn.Linear(wavlm_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio : [batch, time_samples]  raw waveform at 16kHz
                    4 seconds = 64,000 samples

        Returns:
            embedding : [batch, 256]
        """
        # WavLM transformer layers → frame-level features
        out    = self.wavlm(input_values=audio)
        hidden = out.last_hidden_state        # [batch, frames, 1024]

        # Mean pool all frames into one vector per clip
        pooled = hidden.mean(dim=1)           # [batch, 1024]

        # Project to output dim
        return self.projection(pooled)        # [batch, 256]

    def trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"WavLMEncoder — trainable: {trainable:,} / total: {total:,} "
              f"({100*trainable/total:.1f}%)")