# src/models/childfluency.py
"""
ChildFluency-Net — Full Model

Combines WavLM acoustic encoder + LPE linguistic encoder
into a unified architecture with dual SLD/OD classification heads.

Architecture:
    [WavLM → 256-dim] + [LPE → 32-dim]
          ↓ concat [288-dim]
        Shared MLP [64-dim]
          ↓
    [SLD Head]   [OD Head]
          ↓
    Mean pool across windows → SLD rate % (clinical output)

Clinical output:
    SLD rate > 3% → refer for specialist assessment
"""

import torch
import torch.nn as nn
from typing import Dict
from wavlm_encoder import WavLMEncoder
from lpe_module import LPEModule


class ChildFluencyNet(nn.Module):

    def __init__(
        self,
        wavlm_name   : str   = "microsoft/wavlm-large",
        acoustic_dim : int   = 256,
        lpe_dim      : int   = 32,
        hidden_dim   : int   = 128,
        dropout      : float = 0.3
    ):
        super().__init__()

        # Branch A — acoustic
        self.acoustic_encoder = WavLMEncoder(
            model_name = wavlm_name,
            output_dim = acoustic_dim
        )

        # Branch B — linguistic position
        self.lpe_module = LPEModule(
            input_dim  = 6,
            output_dim = lpe_dim
        )

        fusion_dim = acoustic_dim + lpe_dim   # 256 + 32 = 288

        # Shared MLP after fusion
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)  # → 64-dim
        )

        # Head 1 — SLD (Stuttering-Like Disfluencies)
        # Detects: blocks, prolongations, sound/syllable repetitions
        # These are PATHOLOGICAL — what clinicians diagnose
        self.sld_head = nn.Linear(hidden_dim // 2, 1)

        # Head 2 — OD (Other Disfluencies)
        # Detects: word repetitions, interjections (um, uh)
        # These are NORMAL — present in all speakers
        self.od_head  = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        audio        : torch.Tensor,   # [batch, 64000]
        lpe_features : torch.Tensor    # [batch, 6]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio        : raw waveform, 4 seconds at 16kHz
            lpe_features : 6 linguistic position features per window

        Returns dict with:
            sld_logit : [batch]   raw score for SLD detection
            od_logit  : [batch]   raw score for OD detection
            embedding : [batch, 64]  shared representation
        """
        # Branch A
        acoustic_emb = self.acoustic_encoder(audio)          # [batch, 256]

        # Branch B
        lpe_emb = self.lpe_module(lpe_features)              # [batch, 32]

        # Fuse
        fused  = torch.cat([acoustic_emb, lpe_emb], dim=-1)  # [batch, 288]
        shared = self.shared_mlp(fused)                       # [batch, 64]

        return {
            "sld_logit" : self.sld_head(shared).squeeze(-1),  # [batch]
            "od_logit"  : self.od_head(shared).squeeze(-1),   # [batch]
            "embedding" : shared                               # [batch, 64]
        }

    @torch.no_grad()
    def predict_recording(self, window_outputs: list) -> Dict:
        """
        Aggregates window-level predictions into recording-level diagnosis.
        Called during inference on a full recording.

        Args:
            window_outputs : list of dicts from forward(), one per window

        Returns:
            sld_rate    : float  SLD rate % (clinical diagnostic metric)
            has_stutter : bool   True if SLD rate > 3%
            sld_timeline: list   per-window SLD probability (for visualization)
        """
        sld_probs = torch.sigmoid(
            torch.stack([o["sld_logit"] for o in window_outputs])
        )
        od_probs = torch.sigmoid(
            torch.stack([o["od_logit"] for o in window_outputs])
        )

        sld_rate = (sld_probs > 0.5).float().mean().item() * 100.0

        return {
            "sld_rate"     : round(sld_rate, 2),
            "has_stutter"  : sld_rate > 3.0,
            "mean_sld_prob": round(sld_probs.mean().item(), 4),
            "mean_od_prob" : round(od_probs.mean().item(), 4),
            "sld_timeline" : sld_probs.cpu().tolist()
        }

    def count_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"ChildFluencyNet")
        print(f"  Trainable : {trainable:,}")
        print(f"  Total     : {total:,}")
        print(f"  Frozen    : {total - trainable:,}")