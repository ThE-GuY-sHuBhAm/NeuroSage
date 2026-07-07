# -*- coding: utf-8 -*-
"""
evaluate_model_weight.py -- Weighted Combo Sweep (MobileNet V3)
===============================================================
Weighted combo sweep using MobileNet V3 as the backbone.
Automatically finds the optimal image/questionnaire weight split and
per-weight-pair threshold that minimises false negatives (missed
dysgraphia cases) -- the most critical metric for a clinical screener.

Sweep design
------------
We test every (w_I, w_Q) pair where w_I + w_Q = 1.0, stepping in 0.05
increments:
    (1.00, 0.00) -> image-only baseline
    (0.95, 0.05)
    (0.90, 0.10)
    ...
    (0.70, 0.30)  <- the clinically motivated target
    ...
    (0.00, 1.00) -> questionnaire-only baseline

Combined score formula (normalised so both signals are on 0-100):
    I_norm = anomaly_score                        (already 0-100+)
    Q_norm = (q_score - 10) / 40 * 100           (maps 10-50 -> 0-100)
    C      = w_I * I_norm  +  w_Q * Q_norm

For each weight pair the optimal threshold is found automatically by
scanning 200 candidate thresholds and picking the one that:
  1. Minimises false negatives  (primary -- clinical safety)
  2. Among ties, maximises F1   (secondary -- overall balance)

Bucketing / borderline zone (MobileNet V3, IMAGE_THRESHOLD = 52.1448493)
------------------------------------------------------------------------
BORDER_LOW  = 43.45  (proportionally scaled from 60.0 at EfficientNet threshold 72.0)
BORDER_HIGH = 57.94  (proportionally scaled from 80.0 at EfficientNet threshold 72.0)

  Score < BORDER_LOW                    -> clear_typical   (well below threshold)
  BORDER_LOW  <= Score < IMAGE_THRESHOLD -> borderline_typical
  IMAGE_THRESHOLD <= Score < BORDER_HIGH -> borderline_atypical
  Score >= BORDER_HIGH                  -> clear_atypical   (well above threshold)

Key outputs
-----------
  1. weight_sweep_fn_heatmap.png      -- FN count across all weight combos
  2. weight_sweep_recall_heatmap.png  -- Recall (= sensitivity) heatmap
  3. weight_sweep_f1_heatmap.png      -- F1 heatmap
  4. weight_sweep_ranking.png         -- Top-15 combos ranked by FN/F1
  5. weight_sweep_all_metrics.png     -- Multi-metric line chart vs w_I
  6. best_combo_confusion_matrix.png  -- Confusion matrix for best combo
  7. best_combo_roc_curve.png         -- ROC for best combo
  8. best_combo_pr_curve.png          -- PR for best combo
  9. best_combo_score_distribution.png
 10. comparison_image_vs_best.png     -- Side-by-side: image-only vs best combo
 11. All per-bucket and edge-case plots for best combo
 12. evaluation_log_mobilenet.csv     -- per-image scores for every weight
 13. weight_sweep_results.json        -- full metrics table for all weights
"""

import os
import sys
import json
import time
import random
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, ConvertImageDtype, Pad, Resize, PILToTensor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score
)

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
DYSGRAPHIA_DIR  = r"dysgraphia_samples"
IAM_TESTSET_DIR = r"IAM\testset"
OUTPUT_DIR      = r"evaluation_results_mobilenet"

MODEL_NAME      = "mobilenet_v3"
CHECKPOINTS_DIR = r"checkpoints"
BASELINE_FILE   = r"baseline_mobilenet_v3.pt"

# IMAGE_THRESHOLD: decision boundary for image-only prediction.
# Derived from MobileNet V3 calibration (replaces EfficientNet's 72.0).
IMAGE_THRESHOLD = 52.1448493

MAX_WIDTH  = 2479
MAX_HEIGHT = 3542
IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Borderline zone boundaries -- proportionally scaled from the old
# EfficientNet values (BORDER_LOW=60.0, BORDER_HIGH=80.0 at threshold 72.0)
# to the new MobileNet threshold of 52.1448493:
#   BORDER_LOW  = 52.1448493 * (60.0 / 72.0) = 43.45
#   BORDER_HIGH = 52.1448493 * (80.0 / 72.0) = 57.94
#
# Bucketing logic:
#   anomaly_score <  BORDER_LOW                    -> clear_typical
#   BORDER_LOW   <= anomaly_score < IMAGE_THRESHOLD -> borderline_typical
#   IMAGE_THRESHOLD <= anomaly_score < BORDER_HIGH  -> borderline_atypical
#   anomaly_score >= BORDER_HIGH                   -> clear_atypical
BORDER_LOW  = 43.45
BORDER_HIGH = 57.94

EDGE_CASE_FRACTION = 0.15
SEED               = 42

# Sweep config
WEIGHT_STEP       = 0.05    # granularity of w_I sweep
N_THRESHOLD_STEPS = 200     # candidate thresholds per weight pair
Q_MIN, Q_MAX      = 10, 50  # Q score range for normalisation

random.seed(SEED)
np.random.seed(SEED)


# -----------------------------------------------------------------
# Q-SCORE SIMULATION
# -----------------------------------------------------------------
HIGH_SYMPTOM_PROBS = [0.00, 0.05, 0.15, 0.40, 0.40]
LOW_SYMPTOM_PROBS  = [0.45, 0.40, 0.10, 0.05, 0.00]
VERY_HIGH_PROBS    = [0.00, 0.00, 0.05, 0.35, 0.60]
VERY_LOW_PROBS     = [0.60, 0.35, 0.05, 0.00, 0.00]
ANSWER_VALUES      = [0, 1, 2, 3, 4]
N_QUESTIONS        = 10


def simulate_q_score(profile: str) -> int:
    probs = {'high': HIGH_SYMPTOM_PROBS, 'low': LOW_SYMPTOM_PROBS,
             'very_high': VERY_HIGH_PROBS, 'very_low': VERY_LOW_PROBS}[profile]
    answers = np.random.choice(ANSWER_VALUES, size=N_QUESTIONS, p=probs)
    return int(sum(answers + 1))


def get_bucket(anomaly_score: float, true_label: int) -> str:
    """
    Assign a confidence bucket based on anomaly score and ground truth.

    For atypical (dysgraphia) samples:
      - clear_atypical     : score >= BORDER_HIGH  (model clearly correct)
      - borderline_atypical: score in [IMAGE_THRESHOLD, BORDER_HIGH)

    For typical (normal) samples:
      - clear_typical      : score < BORDER_LOW    (model clearly correct)
      - borderline_typical : score in [BORDER_LOW, IMAGE_THRESHOLD)

    A typical sample scoring >= IMAGE_THRESHOLD is a false positive that
    sits near the decision boundary -- the Q-score has the best chance
    of correcting the combined prediction in that zone.
    """
    if true_label == 1:
        return 'clear_atypical' if anomaly_score >= BORDER_HIGH else 'borderline_atypical'
    else:
        return 'clear_typical' if anomaly_score < BORDER_LOW else 'borderline_typical'


def assign_q_profile(bucket: str, edge_case) -> str:
    if edge_case == 'E1_messy_writer':   return 'very_low'
    if edge_case == 'E2_under_reporter': return 'very_low'
    if edge_case == 'E3_anxious_parent': return 'very_high'
    return 'high' if bucket in ('clear_atypical', 'borderline_atypical') else 'low'


def assign_edge_case(bucket: str, rng: random.Random):
    if rng.random() > EDGE_CASE_FRACTION:
        return None
    return {'borderline_typical': 'E1_messy_writer',
            'clear_atypical':     'E2_under_reporter',
            'clear_typical':      'E3_anxious_parent'}.get(bucket)


# -----------------------------------------------------------------
# SCORE NORMALISATION
# -----------------------------------------------------------------
def normalise_q(q: float) -> float:
    """Map Q score 10-50 to 0-100."""
    return (q - Q_MIN) / (Q_MAX - Q_MIN) * 100.0


def weighted_combo(anomaly_score: float, q_score: float,
                   w_i: float, w_q: float) -> float:
    """
    C = w_I * I_norm + w_Q * Q_norm
    Both signals on 0-100 scale so weights are directly comparable.
    """
    return w_i * anomaly_score + w_q * normalise_q(q_score)


# -----------------------------------------------------------------
# THRESHOLD OPTIMISATION  (minimise FN, break ties by max F1)
# -----------------------------------------------------------------
def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray):
    """
    Scan N_THRESHOLD_STEPS thresholds and return the one minimising FN.
    Among equal-FN thresholds, pick the one with highest F1.
    Returns (best_threshold, metrics_dict, confusion_matrix).
    """
    lo, hi = scores.min(), scores.max()
    candidates = np.linspace(lo, hi, N_THRESHOLD_STEPS)

    best_thresh = candidates[0]
    best_fn     = len(y_true)
    best_f1     = 0.0

    for t in candidates:
        preds = (scores >= t).astype(int)
        fn    = int(((y_true == 1) & (preds == 0)).sum())
        f1    = f1_score(y_true, preds, zero_division=0)
        if fn < best_fn or (fn == best_fn and f1 > best_f1):
            best_fn     = fn
            best_f1     = f1
            best_thresh = t

    best_preds = (scores >= best_thresh).astype(int)
    m, cm      = compute_metrics(y_true, best_preds, scores)
    return best_thresh, m, cm


# -----------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy':          accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision':         precision_score(y_true, y_pred, zero_division=0),
        'recall':            recall_score(y_true, y_pred, zero_division=0),
        'specificity':       spec,
        'f1':                f1_score(y_true, y_pred, zero_division=0),
        'mcc':               matthews_corrcoef(y_true, y_pred),
        'kappa':             cohen_kappa_score(y_true, y_pred),
        'auc_roc':           roc_auc_score(y_true, y_score),
        'auc_pr':            average_precision_score(y_true, y_score),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'n_total':    len(y_true),
        'n_positive': int(sum(y_true)),
        'n_negative': int(len(y_true) - sum(y_true)),
    }, cm


def print_metrics_table(metrics, title):
    sep = "-" * 52
    logger.info("\n%s\n  %s\n%s", sep, title, sep)
    for k, v in metrics.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        logger.info("  %-25s %s", k, val)
    logger.info(sep)


# -----------------------------------------------------------------
# MODEL / IMAGE UTILITIES
# -----------------------------------------------------------------
def load_encoder(model_name, checkpoints_dir, device):
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from model import EncoderFactory
    wrapper    = EncoderFactory(backbone_name=model_name, device=device)
    model_path = os.path.join(checkpoints_dir, f"{model_name}_model_best.pth")
    if not os.path.exists(model_path):
        model_path = f"{model_name}_model_best.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    logger.info("Loading weights from %s ...", model_path)
    ckpt  = torch.load(model_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    wrapper.load_state_dict(state)
    model = wrapper.get_model()
    model.eval()
    logger.info("Model loaded.")
    return model


def load_baseline(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline not found: {path}")
    vec = torch.load(path, map_location=device)
    logger.info("Baseline loaded, shape=%s", vec.shape)
    return vec


def get_anomaly_score(pil_img, model, baseline, device, max_w, max_h):
    img = pil_img.convert('L')
    iw, ih = img.size
    if iw > max_w or ih > max_h:
        scale = min(max_w / iw, max_h / ih)
        iw, ih = int(iw * scale), int(ih * scale)
        img = img.resize((iw, ih), Image.LANCZOS)
    transform = Compose([PILToTensor(), ConvertImageDtype(torch.float),
                         Pad((0, 0, max_w - iw, max_h - ih), fill=1.0),
                         Resize((128, 1024))])
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(tensor)
    sim = F.cosine_similarity(vec, baseline.unsqueeze(0))
    return (1 - sim.item()) * 100


def collect_images(directory, label, label_name):
    entries = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in IMG_EXTS:
                entries.append((os.path.join(root, fname), label, label_name))
    logger.info("  %d images in '%s' -> %s", len(entries), directory, label_name)
    return entries


# -----------------------------------------------------------------
# PLOTS -- SWEEP VISUALISATIONS
# -----------------------------------------------------------------
PALETTE = {'Atypical': '#E74C3C', 'Typical': '#2ECC71'}
CMAP_FN  = 'RdYlGn_r'
CMAP_REC = 'RdYlGn'
CMAP_F1  = 'RdYlGn'


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", path)


def plot_fn_heatmap(sweep_df, path):
    fig, ax = plt.subplots(figsize=(14, 3))
    data    = sweep_df[['w_i', 'fn']].set_index('w_i').T
    sns.heatmap(data, annot=True, fmt='.0f', cmap=CMAP_FN,
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'False Negatives (lower is better)'})
    ax.set_title('False Negatives by Image Weight  (w_I + w_Q = 1.0)\n'
                 'GREEN = fewer missed dysgraphia cases  [CLINICAL PRIORITY]',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Image Weight (w_I)', fontsize=11)
    ax.set_yticks([])
    ax.set_xticklabels([f'{v:.2f}' for v in sweep_df['w_i']], rotation=45, ha='right')
    _save(path)


def plot_recall_heatmap(sweep_df, path):
    fig, ax = plt.subplots(figsize=(14, 3))
    data    = sweep_df[['w_i', 'recall']].set_index('w_i').T
    sns.heatmap(data, annot=True, fmt='.3f', cmap=CMAP_REC,
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'Recall / Sensitivity (higher is better)'})
    ax.set_title('Recall (Sensitivity) by Image Weight  (w_I + w_Q = 1.0)\n'
                 'GREEN = higher recall  [Captures more true dysgraphia cases]',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Image Weight (w_I)', fontsize=11)
    ax.set_yticks([])
    ax.set_xticklabels([f'{v:.2f}' for v in sweep_df['w_i']], rotation=45, ha='right')
    _save(path)


def plot_f1_heatmap(sweep_df, path):
    fig, ax = plt.subplots(figsize=(14, 3))
    data    = sweep_df[['w_i', 'f1']].set_index('w_i').T
    sns.heatmap(data, annot=True, fmt='.3f', cmap=CMAP_F1,
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'F1 Score (higher is better)'})
    ax.set_title('F1 Score by Image Weight  (w_I + w_Q = 1.0)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Image Weight (w_I)', fontsize=11)
    ax.set_yticks([])
    ax.set_xticklabels([f'{v:.2f}' for v in sweep_df['w_i']], rotation=45, ha='right')
    _save(path)


def plot_all_metrics_line(sweep_df, best_w_i, path):
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    metric_lines = [
        ('recall',            'Recall (Sensitivity)', '#E74C3C', '-',  2.5),
        ('f1',                'F1',                   '#3498DB', '-',  2.0),
        ('precision',         'Precision',            '#27AE60', '--', 1.5),
        ('specificity',       'Specificity',          '#8E44AD', '--', 1.5),
        ('balanced_accuracy', 'Balanced Accuracy',    '#E67E22', ':',  1.5),
        ('auc_roc',           'AUC-ROC',              '#1ABC9C', ':',  1.5),
    ]

    for col, label, color, ls, lw in metric_lines:
        ax1.plot(sweep_df['w_i'], sweep_df[col],
                 color=color, ls=ls, lw=lw, label=label, marker='o', ms=4)

    ax2.bar(sweep_df['w_i'], sweep_df['fn'],
            width=WEIGHT_STEP * 0.6, alpha=0.25, color='#E74C3C',
            label='False Negatives (right axis)')
    ax2.set_ylabel('False Negative Count', fontsize=11, color='#E74C3C')
    ax2.tick_params(axis='y', labelcolor='#E74C3C')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax1.axvline(best_w_i, color='black', ls='--', lw=2, alpha=0.7,
                label=f'Best: w_I={best_w_i:.2f}')
    ax1.axvspan(best_w_i - WEIGHT_STEP / 2, best_w_i + WEIGHT_STEP / 2,
                alpha=0.08, color='gold')
    ax1.axvline(0.70, color='steelblue', ls=':', lw=1.5, alpha=0.6,
                label='Reference: w_I=0.70')

    ax1.set_xlabel('Image Weight  (w_I;  w_Q = 1 - w_I)', fontsize=12)
    ax1.set_ylabel('Score (0-1)', fontsize=12)
    ax1.set_ylim([0, 1.08])
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_title('All Metrics vs. Image/Questionnaire Weight Split\n'
                  'Red line = Recall (most critical)  |  Bars = False Negative count',
                  fontsize=13, fontweight='bold')

    lines1, labs1 = ax1.get_legend_handles_labels()
    bars2, labs2  = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + bars2, labs1 + labs2, loc='lower left', fontsize=9, ncol=2)
    ax1.yaxis.grid(True, ls='--', alpha=0.4)
    _save(path)


def plot_ranking(sweep_df, best_w_i, path):
    ranked = sweep_df.sort_values(['fn', 'f1'], ascending=[True, False]).head(15).copy()
    ranked['label'] = ranked.apply(
        lambda r: f"w_I={r['w_i']:.2f} / w_Q={r['w_q']:.2f}", axis=1)

    norm   = plt.Normalize(ranked['fn'].min(), ranked['fn'].max())
    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(norm(fn)) for fn in ranked['fn']]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bars = axes[0].barh(ranked['label'], ranked['f1'],
                        color=colors, edgecolor='gray', height=0.7)
    for bar, row in zip(bars, ranked.itertuples()):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'F1={row.f1:.3f}  FN={row.fn}',
                     va='center', fontsize=8.5)
    axes[0].set_xlim([0, 1.12])
    axes[0].set_xlabel('F1 Score', fontsize=11)
    axes[0].set_title('Top-15 Weight Combos\nRanked by: FN (fewer=better), then F1 (higher=better)',
                      fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[0], label='False Negatives (lower=greener)')

    norm2   = plt.Normalize(0, 1)
    colors2 = [plt.cm.RdYlGn(norm2(r)) for r in ranked['recall']]
    bars2   = axes[1].barh(ranked['label'], ranked['fn'],
                           color=colors2, edgecolor='gray', height=0.7)
    for bar, row in zip(bars2, ranked.itertuples()):
        axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                     f'Recall={row.recall:.3f}', va='center', fontsize=8.5)
    max_fn = ranked['fn'].max()
    axes[1].set_xlim([0, max_fn + 2])
    axes[1].set_xlabel('False Negative Count  (lower is better)', fontsize=11)
    axes[1].set_title('False Negatives per Weight Combo\nBar colour = Recall (green=high)',
                      fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    sm2 = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm2)
    sm2.set_array([])
    plt.colorbar(sm2, ax=axes[1], label='Recall (higher=greener)')

    fig.suptitle('Weight Sweep Ranking  --  Clinical Priority: Minimise False Negatives',
                 fontsize=14, fontweight='bold')
    _save(path)


def plot_comparison_image_vs_best(img_m, best_m, best_w_i, best_w_q, path):
    keys   = ['recall', 'precision', 'specificity', 'f1',
              'balanced_accuracy', 'auc_roc', 'auc_pr', 'mcc']
    labels = ['Recall\n(Sensitivity)', 'Precision', 'Specificity', 'F1',
              'Balanced\nAccuracy', 'AUC-ROC', 'AUC-PR', 'MCC']

    iv = [img_m[k]  for k in keys]
    bv = [best_m[k] for k in keys]
    x  = np.arange(len(keys)); w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    b1 = axes[0].bar(x - w / 2, iv, w, label='Image-Only  (w_I=1.0)',
                     color='steelblue', alpha=0.85)
    b2 = axes[0].bar(x + w / 2, bv, w,
                     label=f'Best Combo  (w_I={best_w_i:.2f}, w_Q={best_w_q:.2f})',
                     color='darkorange', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim([0, 1.15])
    axes[0].set_ylabel('Score')
    axes[0].set_title('Image-Only vs. Best Weighted Combo\n(All metrics)',
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].yaxis.grid(True, ls='--', alpha=0.4)
    for b in list(b1) + list(b2):
        h = b.get_height()
        axes[0].annotate(f'{h:.3f}',
                         xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=7)
    axes[0].get_children()[0].set_edgecolor('red')
    axes[0].get_children()[0].set_linewidth(2)

    counts_img  = [img_m['tp'],  img_m['tn'],  img_m['fp'],  img_m['fn']]
    counts_best = [best_m['tp'], best_m['tn'], best_m['fp'], best_m['fn']]
    count_labels = ['TP', 'TN', 'FP', 'FN']
    count_colors = ['#2ECC71', '#27AE60', '#E67E22', '#E74C3C']
    x2 = np.arange(4)
    b3 = axes[1].bar(x2 - w / 2, counts_img,  w, label='Image-Only',
                     color=count_colors, alpha=0.55, edgecolor='gray')
    b4 = axes[1].bar(x2 + w / 2, counts_best, w, label='Best Combo',
                     color=count_colors, alpha=0.9, edgecolor='black', lw=1.2)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(count_labels, fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confusion Matrix Counts\nImage-Only vs. Best Combo',
                      fontsize=12, fontweight='bold')
    axes[1].yaxis.grid(True, ls='--', alpha=0.4)
    fn_delta = best_m['fn'] - img_m['fn']
    fn_sign  = '+' if fn_delta >= 0 else ''
    axes[1].annotate(f'FN delta = {fn_sign}{fn_delta}',
                     xy=(3 + w / 2, best_m['fn']),
                     xytext=(3 + w / 2 + 0.3, best_m['fn'] + 1),
                     fontsize=11, fontweight='bold', color='#E74C3C',
                     arrowprops=dict(arrowstyle='->', color='#E74C3C'))
    for b in list(b3) + list(b4):
        h = b.get_height()
        if h > 0:
            axes[1].text(b.get_x() + b.get_width() / 2, h + 0.3,
                         str(int(h)), ha='center', fontsize=8)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(facecolor='gray',  alpha=0.55, label='Image-Only'),
        Patch(facecolor='black', alpha=0.9,  label='Best Combo'),
    ], fontsize=9)

    fig.suptitle(
        f'Final Comparison  --  Image-Only  vs  Best Weighted Combo\n'
        f'FN: {img_m["fn"]} -> {best_m["fn"]}   |   '
        f'Recall: {img_m["recall"]:.3f} -> {best_m["recall"]:.3f}   |   '
        f'F1: {img_m["f1"]:.3f} -> {best_m["f1"]:.3f}',
        fontsize=13, fontweight='bold')
    _save(path)


# -----------------------------------------------------------------
# PLOTS -- STANDARD SET
# -----------------------------------------------------------------
def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Typical', 'Pred Atypical'],
                yticklabels=['True Typical', 'True Atypical'],
                ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Ground Truth', fontsize=11)
    ax.set_xlabel('Prediction', fontsize=11)
    _save(path)


def plot_roc(y_true, y_score, auc, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel='False Positive Rate', ylabel='True Positive Rate', title=title)
    ax.legend(loc='lower right')
    _save(path)


def plot_pr(y_true, y_score, auc_pr, title, path):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color='darkorange', lw=2, label=f'AP = {auc_pr:.3f}')
    ax.axhline(sum(y_true) / len(y_true), color='gray', ls='--', lw=1, label='Baseline')
    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel='Recall', ylabel='Precision', title=title)
    ax.legend(loc='upper right')
    _save(path)


def plot_score_dist(scores_typ, scores_atyp, threshold, xlabel, title, path):
    all_s = scores_typ + scores_atyp
    bins  = np.linspace(min(all_s), max(all_s), 40)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores_typ,  bins=bins, alpha=0.6, color=PALETTE['Typical'],
            label='Typical',  edgecolor='white')
    ax.hist(scores_atyp, bins=bins, alpha=0.6, color=PALETTE['Atypical'],
            label='Atypical', edgecolor='white')
    ax.axvline(threshold, color='black', ls='--', lw=2,
               label=f'Optimal Threshold = {threshold:.2f}')
    ax.set(xlabel=xlabel, ylabel='Count', title=title)
    ax.legend()
    _save(path)


def plot_bucket_breakdown(df, combo_col_correct, path):
    buckets = sorted(df['bucket'].unique())
    summary = []
    for b in buckets:
        sub = df[df['bucket'] == b]
        summary.append({
            'bucket':     b,
            'n':          len(sub),
            'img_acc':    sub['img_correct'].mean(),
            'combo_acc':  sub[combo_col_correct].mean(),
            'edge_cases': sub['edge_case'].notna().sum(),
        })
    sdf = pd.DataFrame(summary)
    x = np.arange(len(sdf)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, sdf['img_acc'],   w, label='Image Only',
                color='steelblue',  alpha=0.85)
    b2 = ax.bar(x + w / 2, sdf['combo_acc'], w, label='Best Combo',
                color='darkorange', alpha=0.85)
    labels = [f"{row['bucket']}\n(n={row['n']}, ec={row['edge_cases']})"
              for _, row in sdf.iterrows()]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim([0, 1.15]); ax.set_ylabel('Accuracy')
    ax.set_title('Per-Bucket Accuracy: Image-Only vs. Best Combo',
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.yaxis.grid(True, ls='--', alpha=0.5)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    _save(path)


def plot_edge_case_analysis(df, combo_col_correct, path):
    df = df.copy()
    df['sample_type'] = df['edge_case'].apply(
        lambda x: str(x) if pd.notna(x) and x else 'Normal')
    groups = df.groupby('sample_type').agg(
        n=('img_correct', 'count'),
        img_acc=('img_correct', 'mean'),
        combo_acc=(combo_col_correct, 'mean')
    ).reset_index()
    x = np.arange(len(groups)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, groups['img_acc'],   w, label='Image Only',
                color='steelblue',  alpha=0.85)
    b2 = ax.bar(x + w / 2, groups['combo_acc'], w, label='Best Combo',
                color='darkorange', alpha=0.85)
    labels = [f"{row['sample_type']}\n(n={row['n']})" for _, row in groups.iterrows()]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim([0, 1.15]); ax.set_ylabel('Accuracy')
    ax.set_title('Edge Case vs Normal: Image-Only vs. Best Combo',
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.yaxis.grid(True, ls='--', alpha=0.5)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    _save(path)


# -----------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR,
                        f"run_mobilenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
def main():
    t_start = time.time()
    rng = random.Random(SEED)

    logger.info("=" * 60)
    logger.info("  DYSGRAPHIA SCREENER -- EVALUATION (MobileNet V3)")
    logger.info("  Weighted Combo Sweep with auto threshold optimisation")
    logger.info("  IMAGE_THRESHOLD=%.7f  BORDER_LOW=%.2f  BORDER_HIGH=%.2f",
                IMAGE_THRESHOLD, BORDER_LOW, BORDER_HIGH)
    logger.info("  Run started: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=" * 60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info("Device: %s", device)

    model    = load_encoder(MODEL_NAME, CHECKPOINTS_DIR, device)
    baseline = load_baseline(BASELINE_FILE, device)

    logger.info("\nCollecting images ...")
    dysgraphia_entries = collect_images(DYSGRAPHIA_DIR, 1, 'Atypical')
    iam_entries        = collect_images(IAM_TESTSET_DIR, 0, 'Typical')
    all_entries        = dysgraphia_entries + iam_entries
    logger.info("Total: %d images", len(all_entries))

    if not all_entries:
        logger.error("No images found.")
        sys.exit(1)

    # -- Phase 1: Inference + Q simulation --
    logger.info("\nPhase 1: Inference + grounded Q simulation ...")
    records = []

    for idx, (img_path, true_label, label_name) in enumerate(all_entries):
        try:
            pil_img = Image.open(img_path)
        except Exception as e:
            logger.warning("Cannot open '%s': %s", img_path, e)
            continue

        t0            = time.perf_counter()
        anomaly_score = get_anomaly_score(pil_img, model, baseline,
                                          device, MAX_WIDTH, MAX_HEIGHT)
        img_lat_ms    = (time.perf_counter() - t0) * 1000.0
        img_pred      = 1 if anomaly_score > IMAGE_THRESHOLD else 0

        bucket    = get_bucket(anomaly_score, true_label)
        edge_case = assign_edge_case(bucket, rng)
        q_profile = assign_q_profile(bucket, edge_case)
        q_score   = simulate_q_score(q_profile)
        q_norm    = normalise_q(q_score)

        records.append({
            'image_path':       img_path,
            'true_label':       true_label,
            'true_label_name':  label_name,
            'bucket':           bucket,
            'edge_case':        edge_case,
            'q_profile':        q_profile,
            'anomaly_score':    round(anomaly_score, 4),
            'q_score':          q_score,
            'q_score_norm':     round(q_norm, 4),
            'img_pred':         img_pred,
            'img_correct':      int(img_pred == true_label),
            'image_latency_ms': round(img_lat_ms, 3),
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(all_entries):
            logger.info("  %d/%d processed ...", idx + 1, len(all_entries))

    df = pd.DataFrame(records)

    y_true      = df['true_label'].values
    anomaly_arr = df['anomaly_score'].values
    q_norm_arr  = df['q_score_norm'].values

    # -- Phase 2: Image-only baseline metrics --
    logger.info("\nPhase 2: Computing image-only baseline ...")
    img_pred_arr  = df['img_pred'].values
    img_m, img_cm = compute_metrics(y_true, img_pred_arr, anomaly_arr)
    print_metrics_table(img_m, f"IMAGE-ONLY BASELINE (threshold={IMAGE_THRESHOLD})")

    # -- Phase 3: Weight sweep --
    logger.info("\nPhase 3: Weight sweep ...")
    w_i_values = np.round(np.arange(0.0, 1.0 + WEIGHT_STEP, WEIGHT_STEP), 2)
    sweep_rows = []

    for w_i in w_i_values:
        w_q    = round(1.0 - w_i, 2)
        scores = w_i * anomaly_arr + w_q * q_norm_arr

        best_thresh, m, cm = find_optimal_threshold(y_true, scores)

        row = {
            'w_i':               w_i,
            'w_q':               w_q,
            'label':             f'w_I={w_i:.2f}',
            'opt_threshold':     round(best_thresh, 4),
            'accuracy':          round(m['accuracy'], 5),
            'balanced_accuracy': round(m['balanced_accuracy'], 5),
            'precision':         round(m['precision'], 5),
            'recall':            round(m['recall'], 5),
            'specificity':       round(m['specificity'], 5),
            'f1':                round(m['f1'], 5),
            'mcc':               round(m['mcc'], 5),
            'kappa':             round(m['kappa'], 5),
            'auc_roc':           round(m['auc_roc'], 5),
            'auc_pr':            round(m['auc_pr'], 5),
            'tp':                m['tp'],
            'tn':                m['tn'],
            'fp':                m['fp'],
            'fn':                m['fn'],
        }
        sweep_rows.append(row)

        logger.info("  w_I=%.2f w_Q=%.2f | thresh=%.2f | FN=%2d  Recall=%.3f  F1=%.3f  FP=%2d",
                    w_i, w_q, best_thresh, m['fn'], m['recall'], m['f1'], m['fp'])

    sweep_df = pd.DataFrame(sweep_rows)

    # -- Phase 4: Identify best combo --
    # Primary: min FN  |  Secondary: max F1
    best_row    = sweep_df.sort_values(['fn', 'f1'], ascending=[True, False]).iloc[0]
    best_w_i    = best_row['w_i']
    best_w_q    = best_row['w_q']
    best_thresh = best_row['opt_threshold']

    logger.info("\n%s", "=" * 60)
    logger.info("  BEST COMBO: w_I=%.2f  w_Q=%.2f  threshold=%.2f",
                best_w_i, best_w_q, best_thresh)
    logger.info("  FN=%s  Recall=%.4f  F1=%.4f",
                best_row['fn'], best_row['recall'], best_row['f1'])
    logger.info("%s", "=" * 60)

    # Recompute best combo scores + per-image predictions for plots
    best_scores     = best_w_i * anomaly_arr + best_w_q * q_norm_arr
    best_preds      = (best_scores >= best_thresh).astype(int)
    best_m, best_cm = compute_metrics(y_true, best_preds, best_scores)
    print_metrics_table(best_m, f"BEST COMBO (w_I={best_w_i}, w_Q={best_w_q})")

    df['best_combo_score']   = best_scores
    df['best_combo_pred']    = best_preds
    df['best_combo_correct'] = (best_preds == y_true).astype(int)

    # -- Phase 5: Save outputs --
    for row in sweep_rows:
        wi  = row['w_i']
        wq  = row['w_q']
        col = f'combo_score_wi{wi:.2f}'
        df[col] = wi * anomaly_arr + wq * q_norm_arr

    csv_path = os.path.join(OUTPUT_DIR, 'evaluation_log_mobilenet.csv')
    df.to_csv(csv_path, index=False)
    logger.info("\nEvaluation log saved: %s", csv_path)

    sweep_csv = os.path.join(OUTPUT_DIR, 'weight_sweep_results.csv')
    sweep_df.to_csv(sweep_csv, index=False)
    logger.info("Sweep results saved: %s", sweep_csv)

    summary = {
        'run_datetime':      datetime.now().isoformat(),
        'device':            device,
        'model':             MODEL_NAME,
        'baseline_file':     BASELINE_FILE,
        'version':           'mobilenet_weighted_sweep',
        'image_threshold':   IMAGE_THRESHOLD,
        'border_low':        BORDER_LOW,
        'border_high':       BORDER_HIGH,
        'n_total':           len(df),
        'n_atypical':        int(sum(y_true)),
        'n_typical':         int(len(y_true) - sum(y_true)),
        'weight_step':       WEIGHT_STEP,
        'n_threshold_steps': N_THRESHOLD_STEPS,
        'image_only':        {k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in img_m.items()},
        'best_combo': {
            'w_i':       best_w_i,
            'w_q':       best_w_q,
            'threshold': best_thresh,
            **{k: (round(v, 5) if isinstance(v, float) else v)
               for k, v in best_m.items()}
        },
        'all_weights': sweep_rows,
    }
    json_path = os.path.join(OUTPUT_DIR, 'weight_sweep_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON saved: %s", json_path)

    # -- Phase 6: Plots --
    logger.info("\nGenerating plots ...")
    od = OUTPUT_DIR

    plot_fn_heatmap(sweep_df,     f'{od}/weight_sweep_fn_heatmap.png')
    plot_recall_heatmap(sweep_df, f'{od}/weight_sweep_recall_heatmap.png')
    plot_f1_heatmap(sweep_df,     f'{od}/weight_sweep_f1_heatmap.png')
    plot_all_metrics_line(sweep_df, best_w_i, f'{od}/weight_sweep_all_metrics.png')
    plot_ranking(sweep_df, best_w_i, f'{od}/weight_sweep_ranking.png')
    plot_comparison_image_vs_best(img_m, best_m, best_w_i, best_w_q,
        f'{od}/comparison_image_vs_best.png')

    typ_best  = df[df['true_label'] == 0]['best_combo_score'].tolist()
    atyp_best = df[df['true_label'] == 1]['best_combo_score'].tolist()
    typ_img   = df[df['true_label'] == 0]['anomaly_score'].tolist()
    atyp_img  = df[df['true_label'] == 1]['anomaly_score'].tolist()

    plot_confusion_matrix(img_cm,
        'Image-Only Confusion Matrix',
        f'{od}/image_only_confusion_matrix.png')
    plot_confusion_matrix(best_cm,
        f'Best Combo (w_I={best_w_i:.2f}, w_Q={best_w_q:.2f}) Confusion Matrix',
        f'{od}/best_combo_confusion_matrix.png')
    plot_roc(y_true, anomaly_arr, img_m['auc_roc'],
        'Image-Only ROC Curve',
        f'{od}/image_only_roc_curve.png')
    plot_roc(y_true, best_scores, best_m['auc_roc'],
        f'Best Combo (w_I={best_w_i:.2f}) ROC Curve',
        f'{od}/best_combo_roc_curve.png')
    plot_pr(y_true, anomaly_arr, img_m['auc_pr'],
        'Image-Only PR Curve',
        f'{od}/image_only_pr_curve.png')
    plot_pr(y_true, best_scores, best_m['auc_pr'],
        f'Best Combo (w_I={best_w_i:.2f}) PR Curve',
        f'{od}/best_combo_pr_curve.png')
    plot_score_dist(typ_img, atyp_img, IMAGE_THRESHOLD,
        'Anomaly Score', 'Image-Only Score Distribution',
        f'{od}/image_only_score_distribution.png')
    plot_score_dist(typ_best, atyp_best, best_thresh,
        f'Weighted Score (w_I={best_w_i:.2f} * I + w_Q={best_w_q:.2f} * Q_norm)',
        'Best Combo Score Distribution',
        f'{od}/best_combo_score_distribution.png')
    plot_bucket_breakdown(df, 'best_combo_correct', f'{od}/bucket_breakdown.png')
    plot_edge_case_analysis(df, 'best_combo_correct', f'{od}/edge_case_analysis.png')

    logger.info("\n%s", "=" * 60)
    logger.info("  Done in %.1fs  |  Results: %s", time.time() - t_start, OUTPUT_DIR)
    logger.info("  Best combo : w_I=%.2f  w_Q=%.2f  FN=%s  Recall=%.4f  F1=%.4f",
                best_w_i, best_w_q, best_m['fn'], best_m['recall'], best_m['f1'])
    logger.info("  Image-only : FN=%s  Recall=%.4f  F1=%.4f",
                img_m['fn'], img_m['recall'], img_m['f1'])
    logger.info("%s", "=" * 60)


if __name__ == '__main__':
    main()