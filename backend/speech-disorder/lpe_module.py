# src/models/lpe_module.py
"""
Linguistic Position Encoding (LPE) Module

Core novel contribution of ChildFluency-Net.

Encodes 6 loci-of-stuttering features into a 32-dim vector.
These features are pre-computed by build_features.py using
WhisperX transcription + spaCy linguistic parsing.

Clinical basis (Bloodstein 1960, Bernstein-Ratner 1997):
    Stuttering is 3-5x more likely on:
    - Sentence-initial positions
    - Content words (nouns, verbs) vs function words
    - Longer, rarer words
    - Near clause boundaries (cognitive planning spikes)
    - In longer, more complex utterances (higher MLU)

Input features [6 scalars, all in 0-1 range]:
    0: sentence position     (0=initial=high risk, 1=final)
    1: word class            (1=content, 0=function)
    2: syllable count        (normalized, max=6 syllables)
    3: clause boundary prox  (1=near boundary)
    4: word rarity           (1=rare=high risk, 0=common)
    5: MLU                   (sentence length / 20)
"""

import torch
import torch.nn as nn


class LPEModule(nn.Module):

    def __init__(
        self,
        input_dim  : int = 6,
        output_dim : int = 32
    ):
        super().__init__()

        # Small MLP: 6 → 16 → 32
        # Kept intentionally small — 6 features don't need
        # a large network, just a learned non-linear transformation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, 6]   pre-computed LPE feature vectors

        Returns:
              : [batch, 32]  LPE embeddings
        """
        return self.encoder(x)