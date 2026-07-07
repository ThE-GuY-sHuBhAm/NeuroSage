import torch
import os
import time
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, ConvertImageDtype, Pad, Resize, PILToTensor
from pathlib import Path
from datetime import datetime

# Import your modules
from model import EncoderFactory
from data import IAMDL

# --- CONFIGURATION ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IAM_TEST_PATH = Path('IAM') / 'testset'
DYSGRAPHIA_PATH = Path('dysgraphia_samples')   # Must contain exactly 35 images
N_DYSGRAPHIA_EXPECTED = 35
N_NORMAL_SAMPLES = 200                          # Control group cap

# Output: all CSVs go into a timestamped subfolder
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = Path(f'results_{RUN_TIMESTAMP}')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All four backbones whose baselines already exist on disk
BACKBONES = ['resnet18', 'resnet34', 'mobilenet_v3', 'efficientnet_b0']

# Hardcoded image stats (from get_stats.py)
MAX_W, MAX_H = 2479, 3542


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path):
    """Loads and preprocesses a single handwriting image to a model-ready tensor."""
    try:
        img = Image.open(path).convert('L')
        w, h = img.size
        transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Pad((0, 0, MAX_W - w, MAX_H - h), fill=1.),
            Resize((128, 1024))
        ])
        return transform(img).unsqueeze(0)
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def score_image(model, img_tensor, baseline, device):
    """
    Runs one forward pass and returns:
      - anomaly_score : float  (0–100, higher = more anomalous)
      - inference_time_ms : float  (wall-clock time for the forward pass, ms)
    """
    img_tensor = img_tensor.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        embed = model(img_tensor)
        sim = F.cosine_similarity(embed, baseline.unsqueeze(0))
    inference_time_ms = (time.perf_counter() - t0) * 1000.0

    anomaly_score = (1.0 - sim.item()) * 100.0
    return anomaly_score, inference_time_ms


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(backbone_name):
    print(f"\n{'='*60}")
    print(f"  Backbone: {backbone_name}")
    print(f"{'='*60}")

    # 1. Load model weights
    wrapper = EncoderFactory(backbone_name, device=DEVICE)
    model_path = f'{backbone_name}_model_best.pth'
    if not os.path.exists(model_path):
        model_path = os.path.join('checkpoints', model_path)

    try:
        wrapper.load_state(os.path.basename(model_path))
        print(f"  Loaded weights: {model_path}")
    except Exception as e:
        print(f"  [WARN] Could not load weights ({e}). Using random weights — results meaningless!")

    model = wrapper.get_model()
    model.eval()

    # 2. Load the pre-computed baseline vector (must already exist)
    baseline_path = f'baseline_{backbone_name}.pt'
    if not os.path.exists(baseline_path):
        print(f"  [ERROR] Baseline not found: {baseline_path}. Skipping backbone.")
        return

    baseline = torch.load(baseline_path, map_location=DEVICE)
    print(f"  Loaded baseline: {baseline_path}")

    results = []

    # 3. Normal / Control group ------------------------------------------------
    print(f"\n  Processing Normal (Control) group — up to {N_NORMAL_SAMPLES} samples...")
    normal_files = [
        os.path.join(r, f)
        for r, _, files in os.walk(IAM_TEST_PATH)
        for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    normal_files = normal_files[:N_NORMAL_SAMPLES]

    if not normal_files:
        print(f"  [WARN] No normal images found under {IAM_TEST_PATH}!")

    for fpath in tqdm(normal_files, desc=f"  [{backbone_name}] Normal"):
        img = load_image(fpath)
        if img is None:
            continue
        anomaly_score, lat_ms = score_image(model, img, baseline, DEVICE)
        results.append({
            'filename': os.path.basename(fpath),
            'group': 'normal',
            'label': 0,
            'anomaly_score': round(anomaly_score, 6),
            'inference_time_ms': round(lat_ms, 4),
        })

    # 4. Dysgraphia / Test group -----------------------------------------------
    print(f"\n  Processing Dysgraphia (Test) group — expecting {N_DYSGRAPHIA_EXPECTED} samples...")
    dys_files = [
        str(p) for p in DYSGRAPHIA_PATH.glob('*')
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ]

    if not dys_files:
        print(f"  [ERROR] No images found in {DYSGRAPHIA_PATH}! Aborting backbone.")
        return

    if len(dys_files) != N_DYSGRAPHIA_EXPECTED:
        print(
            f"  [WARN] Expected {N_DYSGRAPHIA_EXPECTED} dysgraphia images, "
            f"found {len(dys_files)}."
        )
    else:
        print(f"  Sample count OK: {len(dys_files)} dysgraphia images found.")

    for fpath in tqdm(dys_files, desc=f"  [{backbone_name}] Dysgraphia"):
        img = load_image(fpath)
        if img is None:
            continue
        anomaly_score, lat_ms = score_image(model, img, baseline, DEVICE)
        results.append({
            'filename': os.path.basename(fpath),
            'group': 'dysgraphia',
            'label': 1,
            'anomaly_score': round(anomaly_score, 6),
            'inference_time_ms': round(lat_ms, 4),
        })

    # 5. Save CSV --------------------------------------------------------------
    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / f'results_{backbone_name}.csv'
    df.to_csv(out_path, index=False)
    print(f"\n  [+] Saved: {out_path}")

    # Quick stats
    print("  --- Quick Stats ---")
    stats = df.groupby('group')[['anomaly_score', 'inference_time_ms']].agg(['mean', 'std'])
    print(stats.to_string())
    print(f"  Total samples: {len(df)}  "
          f"(normal={len(df[df.label==0])}, dysgraphia={len(df[df.label==1])})")
    print(f"  Mean latency : {df['inference_time_ms'].mean():.3f} ms/sample")
    print("  -------------------")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate handwriting anomaly detection backbones.'
    )
    parser.add_argument(
        '--backbone', type=str, default='all',
        help=(
            'Backbone to evaluate. One of: resnet18, resnet34, mobilenet_v3, '
            'efficientnet_b0  — or "all" to run every backbone sequentially (default).'
        )
    )
    args = parser.parse_args()

    targets = BACKBONES if args.backbone == 'all' else [args.backbone]

    print(f"\nDevice          : {DEVICE}")
    print(f"Output folder   : {RESULTS_DIR}")
    print(f"Backbones to run: {targets}")
    print(f"Dysgraphia path : {DYSGRAPHIA_PATH}  (expecting {N_DYSGRAPHIA_EXPECTED} images)")
    print(f"Normal cap      : {N_NORMAL_SAMPLES} samples from {IAM_TEST_PATH}")

    for backbone in targets:
        run_experiment(backbone)

    print(f"\n{'='*60}")
    print(f"  All done. Results saved to: {RESULTS_DIR}/")
    print(f"{'='*60}\n")