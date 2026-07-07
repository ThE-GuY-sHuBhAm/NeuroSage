import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import os
import sys
import glob
import time
import logging
from datetime import datetime

# --- CONFIG ---
N_DYSGRAPHIA_SAMPLES = 35
BASE_OUTPUT_DIR = 'paper_plots'

# Create a timestamped subfolder for this run
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f'run_{RUN_TIMESTAMP}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOGGING SETUP ---
LOG_FILE = os.path.join(OUTPUT_DIR, 'analysis_log.txt')
_file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
_stream_handler = logging.StreamHandler(sys.stdout)
# Force UTF-8 on Windows consoles that default to cp1252
if hasattr(_stream_handler.stream, 'reconfigure'):
    try:
        _stream_handler.stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[_file_handler, _stream_handler]
)
log = logging.getLogger(__name__)


def calculate_metrics(y_true, y_scores, threshold):
    """Calculates all scientific metrics for a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'Threshold': threshold,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity (Recall)': recall_score(y_true, y_pred),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred),
        'False Positive Rate': fp / (tn + fp) if (tn + fp) > 0 else 0,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }


def find_optimal_threshold(y_true, y_scores):
    """Finds the best threshold using Youden's J statistic (maximises Sensitivity + Specificity)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], fpr, tpr, auc(fpr, tpr)


def time_model_scoring(df):
    """
    Simulates per-sample latency by timing a vectorised anomaly score pass.
    If real per-sample timestamps exist in the CSV ('inference_time_ms'), those are used instead.
    Returns: mean latency (ms), std latency (ms), total latency (ms)
    """
    if 'inference_time_ms' in df.columns:
        lat = df['inference_time_ms'].values
        log.info("  Using per-sample 'inference_time_ms' column from CSV.")
    else:
        # Benchmark: time 100 passes over the score array and normalise
        scores = df['anomaly_score'].values
        n = len(scores)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = (scores >= 50).astype(int)   # representative threshold check
        elapsed_ms = (time.perf_counter() - t0) * 1000 / 100  # mean over 100 runs (ms)
        lat = np.full(n, elapsed_ms / n)     # distribute evenly across samples
        log.info("  No 'inference_time_ms' column found — latency estimated from score-pass benchmark.")

    return lat.mean(), lat.std(), lat.sum()


def analyze_single_model(df, backbone_name, is_best=False):
    """Image-only analysis for one backbone. Detailed plots only for the best model."""
    t_start = time.perf_counter()

    y_true = df['label'].values
    y_scores = df['anomaly_score'].values

    # Sanity check against expected dysgraphia count
    n_dysgraphia = int(y_true.sum())
    if n_dysgraphia != N_DYSGRAPHIA_SAMPLES:
        log.warning(
            f"  [{backbone_name}] Expected {N_DYSGRAPHIA_SAMPLES} dysgraphia samples "
            f"but found {n_dysgraphia} positive labels."
        )
    else:
        log.info(f"  [{backbone_name}] Sample count OK: {n_dysgraphia} dysgraphia / {len(y_true) - n_dysgraphia} normal.")

    best_thresh, fpr, tpr, roc_auc = find_optimal_threshold(y_true, y_scores)
    metrics = calculate_metrics(y_true, y_scores, best_thresh)
    metrics['AUC'] = roc_auc
    metrics['Model'] = backbone_name
    metrics['Type'] = 'Image Only'

    # Latency
    lat_mean, lat_std, lat_total = time_model_scoring(df)
    metrics['Latency_mean_ms'] = round(lat_mean, 4)
    metrics['Latency_std_ms'] = round(lat_std, 4)
    metrics['Latency_total_ms'] = round(lat_total, 4)

    t_elapsed = (time.perf_counter() - t_start) * 1000
    log.info(
        f"  [{backbone_name}] AUC={roc_auc:.4f} | Threshold={best_thresh:.4f} | "
        f"F1={metrics['F1 Score']:.4f} | Sens={metrics['Sensitivity (Recall)']:.4f} | "
        f"Spec={metrics['Specificity']:.4f} | "
        f"Lat={lat_mean:.3f}ms±{lat_std:.3f} | Analysis time={t_elapsed:.1f}ms"
    )

    # --- BEST MODEL: detailed plots ---
    if is_best:
        log.info(f"\n>> Generating detailed plots for WINNER: {backbone_name}...")

        # A. Histogram of anomaly scores
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df, x='anomaly_score', hue='group',
            kde=True, bins=20, palette=['#2ecc71', '#e74c3c'], alpha=0.65
        )
        plt.title(
            f'Anomaly Score Distribution — {backbone_name}\n'
            f'Normal vs. Dysgraphia  (n={len(y_true)}, dysgraphia={n_dysgraphia})',
            fontsize=13
        )
        plt.xlabel('Anomaly Score (0–100)')
        plt.ylabel('Count')
        plt.axvline(
            x=best_thresh, color='black', linestyle='--', linewidth=1.8,
            label=f'Optimal Threshold = {best_thresh:.2f}'
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'best_model_histogram.png'), dpi=150)
        plt.close()

        # B. ROC curve
        plt.figure(figsize=(8, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2.5,
                 label=f'Image Only — {backbone_name}  (AUC = {roc_auc:.4f})')
        plt.scatter(
            metrics['False Positive Rate'], metrics['Sensitivity (Recall)'],
            color='black', zorder=5, s=80,
            label=f'Operating Point  (Thresh={best_thresh:.2f})'
        )
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve — Best Model: {backbone_name}', fontsize=13)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'best_model_roc.png'), dpi=150)
        plt.close()

        # C. Confusion matrix heatmap
        cm = np.array([[metrics['TN'], metrics['FP']],
                       [metrics['FN'], metrics['TP']]])
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Normal', 'Pred Dysgraphia'],
            yticklabels=['True Normal', 'True Dysgraphia']
        )
        plt.title(f'Confusion Matrix — {backbone_name}\n(Threshold = {best_thresh:.2f})', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'best_model_confusion_matrix.png'), dpi=150)
        plt.close()

        log.info(f"  Saved: best_model_histogram.png, best_model_roc.png, best_model_confusion_matrix.png")

    return metrics, (fpr, tpr, roc_auc)


def log_winner_summary(winner_row):
    """Prints and logs a formatted summary block for the winning model."""
    separator = "=" * 60
    lines = [
        separator,
        f"  WINNING MODEL: {winner_row['Model']}",
        separator,
        f"  AUC              : {winner_row['AUC']:.4f}",
        f"  Optimal Threshold: {winner_row['Threshold']:.4f}",
        f"  Accuracy         : {winner_row['Accuracy']:.4f}",
        f"  Sensitivity      : {winner_row['Sensitivity (Recall)']:.4f}",
        f"  Specificity      : {winner_row['Specificity']:.4f}",
        f"  Precision        : {winner_row['Precision']:.4f}",
        f"  F1 Score         : {winner_row['F1 Score']:.4f}",
        f"  False Positive Rate: {winner_row['False Positive Rate']:.4f}",
        f"  TP={int(winner_row['TP'])}  TN={int(winner_row['TN'])}  "
        f"FP={int(winner_row['FP'])}  FN={int(winner_row['FN'])}",
        "",
        f"  Latency (mean)   : {winner_row['Latency_mean_ms']:.4f} ms/sample",
        f"  Latency (std)    : {winner_row['Latency_std_ms']:.4f} ms",
        f"  Latency (total)  : {winner_row['Latency_total_ms']:.4f} ms",
        separator,
    ]
    for line in lines:
        log.info(line)


def compare_all():
    csv_files = glob.glob("results_*.csv")
    if not csv_files:
        log.error("No results_*.csv files found! Did you run evaluate_experiment.py?")
        return

    log.info(f"Output directory : {OUTPUT_DIR}")
    log.info(f"Log file         : {LOG_FILE}")
    log.info(f"Found {len(csv_files)} experiment(s): {csv_files}")
    log.info(f"Expected dysgraphia samples per file: {N_DYSGRAPHIA_SAMPLES}")

    all_metrics = []
    roc_data = {}

    # 1. First pass — score every backbone
    log.info("\n--- Pass 1: Scoring all backbones ---")
    for filepath in sorted(csv_files):
        backbone = filepath.replace("results_", "").replace(".csv", "")
        log.info(f"\nAnalysing: {backbone}")
        df = pd.read_csv(filepath)
        m, roc_tuple = analyze_single_model(df, backbone, is_best=False)
        all_metrics.append(m)
        roc_data[backbone] = roc_tuple

    # 2. Build leaderboard
    results_df = pd.DataFrame(all_metrics).sort_values('AUC', ascending=False).reset_index(drop=True)
    results_df.insert(0, 'Rank', results_df.index + 1)

    table_path = os.path.join(OUTPUT_DIR, 'model_comparison_table.csv')
    results_df.to_csv(table_path, index=False)
    log.info(f"\n[+] Saved comparison table: {table_path}")

    log.info("\n--- Leaderboard (Image Only, sorted by AUC) ---")
    display_cols = ['Rank', 'Model', 'AUC', 'Threshold', 'Sensitivity (Recall)',
                    'Specificity', 'Precision', 'F1 Score',
                    'Latency_mean_ms', 'Latency_total_ms']
    log.info("\n" + results_df[display_cols].to_string(index=False))

    # 3. Identify winner
    winner_row = results_df.iloc[0]
    winner_name = winner_row['Model']
    log.info(f"\n*** WINNER: {winner_name} (AUC = {winner_row['AUC']:.4f}) ***\n")
    log_winner_summary(winner_row)

    # 4. Backbone comparison ROC plot
    plt.figure(figsize=(10, 8))
    cmap = plt.colormaps['tab10'].resampled(len(roc_data))
    for i, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        lw = 3 if name == winner_name else 1.5
        ls = '-' if name == winner_name else '--'
        label = f'★ {name}  (AUC={roc_auc:.4f})' if name == winner_name else f'{name}  (AUC={roc_auc:.4f})'
        plt.plot(fpr, tpr, lw=lw, linestyle=ls, color=cmap(i), label=label)

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Backbone Comparison — Image Anomaly Score Only', fontsize=13)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    comparison_roc_path = os.path.join(OUTPUT_DIR, 'backbone_comparison_roc.png')
    plt.savefig(comparison_roc_path, dpi=150)
    plt.close()
    log.info(f"[+] Saved backbone comparison ROC: {comparison_roc_path}")

    # 5. Second pass — detailed plots for winner only
    log.info(f"\n--- Pass 2: Detailed plots for winner ({winner_name}) ---")
    winner_file = f"results_{winner_name}.csv"
    df_winner = pd.read_csv(winner_file)
    analyze_single_model(df_winner, winner_name, is_best=True)

    log.info(f"\n[✓] All outputs saved to: {OUTPUT_DIR}")
    log.info(f"[✓] Full log saved to   : {LOG_FILE}")


if __name__ == "__main__":
    compare_all()