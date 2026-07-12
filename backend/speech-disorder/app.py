# src/api/app_fyp.py
"""
FYP — Speech Disfluency Detection API
--------------------------------------
FastAPI application wrapping ChildFluencyNet (stage2_fyp_best.pt).
One endpoint of a larger multi-modal neurodevelopmental assessment system.

Design decisions
----------------
- LPE features set to neutral (0.5) at inference time.
  Ablation study showed LPE Δ F1 = +0.002 (negligible); avoids
  the 30-60s WhisperX transcription cost for real-time API use.
- Model loaded once at startup, shared across requests.
- Mixed precision (autocast) for fast GPU inference.

Endpoints
---------
  GET  /health           → service health check
  GET  /model/info       → loaded checkpoint metadata
  POST /predict          → SLD detection from audio file

Usage
-----
  pip install fastapi uvicorn python-multipart
  uvicorn src.api.app_fyp:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import json
import os
import re
import sys
import time
import yaml
import tempfile
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import librosa
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────
# Set ROOT to the Backend folder so params.yaml and model files resolve
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Workspace places model files directly under Backend/, import local modules
from childfluency import ChildFluencyNet

# ── Load params ───────────────────────────────────────────────────────────
with open(ROOT / 'params.yaml') as f:
    params = yaml.safe_load(f)

TARGET_SR  = params['data']['target_sr']   # 16000
WINDOW_SEC = params['data']['window_sec']  # 4.0
STRIDE_SEC = params['data']['stride_sec']  # 2.0
MIN_RMS    = params['data']['min_rms']     # 0.001

# ── Two separate LPE constants — DO NOT confuse these ────────────────────
LPE_OUTPUT_DIM     = params['model']['lpe_dim']  # 32 — output dim, for model init
LPE_INPUT_FEATURES = 6                           # 6  — input features, for lpe_t

DEFAULT_CKPT = os.environ.get('DISFLUENCY_CHECKPOINT', str(ROOT / 'stage2_fyp_best.pt'))

# ── Gemini (optional) ─────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GROK_API_KEY   = os.environ.get('GROK_API_KEY')
gemini_model = None

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("[startup] Gemini API initialised.")
    except Exception as e:
        print(f"[startup] Warning: Gemini init failed: {e}")
else:
    print("[startup] GEMINI_API_KEY not set — will rely on Groq fallback.")

if GROK_API_KEY:
    print("[startup] Groq API key found — available as fallback.")
else:
    print("[startup] GROK_API_KEY not set — Groq fallback disabled.")

_state: dict = {}


# ── Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = Path(DEFAULT_CKPT)

    print(f"\n[startup] Device     : {device}")
    print(f"[startup] Checkpoint : {ckpt_path}")

    model = ChildFluencyNet(
        wavlm_name   = params['model'].get('wavlm_name', 'microsoft/wavlm-large'),
        acoustic_dim = params['model']['acoustic_dim'],
        lpe_dim      = LPE_OUTPUT_DIM,   # 32 — output dim of LPEModule
        hidden_dim   = params['model']['hidden_dim'],
        dropout      = params['model']['dropout']
    ).to(device)

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt['model'])
        _state['epoch']     = ckpt.get('epoch', '?')
        _state['val_f1']    = round(float(ckpt.get('val_f1', 0)), 4)
        _state['ckpt_path'] = str(ckpt_path)
        print(f"[startup] Loaded → epoch {_state['epoch']}, "
              f"val_f1={_state['val_f1']}")
    else:
        print(f"[startup] WARNING: checkpoint not found. Random weights.")
        _state['epoch'] = _state['val_f1'] = _state['ckpt_path'] = None

    model.eval()
    _state['model']  = model
    _state['device'] = device
    yield
    _state.clear()
    print("[shutdown] Model unloaded.")


# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Speech Disfluency Detection",
    description = (
        "Detects Stutter-Like Disfluencies (SLD) in speech audio. "
        "Part of a multi-modal neurodevelopmental assessment system."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)
app.openapi_version = "3.0.2"

# CORS — allow any origin so the React team can call this immediately
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────
class WindowScore(BaseModel):
    window_idx : int
    start_sec  : float
    end_sec    : float
    sld_prob   : float
    prediction : str       # "SLD" | "fluent"


class PredictionResponse(BaseModel):
    filename             : str
    duration_sec         : float
    sld_probability      : float
    prediction           : str     # "stuttering_detected" | "fluent"
    confidence           : str     # "high" | "medium" | "low"
    n_windows_total      : int
    n_windows_positive   : int
    positive_rate        : float
    window_scores        : List[WindowScore]
    processing_time_sec  : float
    note                 : Optional[str] = None


class ModelInfo(BaseModel):
    checkpoint : Optional[str]
    epoch      : Optional[int]
    val_f1     : Optional[float]
    device     : str
    window_sec : float
    stride_sec : float
    target_sr  : int


class HealthResponse(BaseModel):
    status       : str
    model_loaded : bool
    device       : str


# ── Inference ─────────────────────────────────────────────────────────────
def segment_audio(audio: np.ndarray):
    window_samples = int(WINDOW_SEC * TARGET_SR)
    stride_samples = int(STRIDE_SEC * TARGET_SR)
    windows, start = [], 0
    while start + window_samples <= len(audio):
        seg = audio[start: start + window_samples]
        if float(np.sqrt(np.mean(seg ** 2))) >= MIN_RMS:
            windows.append((
                start,
                start + window_samples,
                round(start / TARGET_SR, 3),
                round((start + window_samples) / TARGET_SR, 3),
            ))
        start += stride_samples
    return windows


def predict_windows(audio: np.ndarray, model, device: str,
                    threshold: float = 0.5,
                    batch_size: int = 16) -> List[dict]:
    """
    Segment audio and run SLD inference.
    LPE tensor shape is always [batch, 6] — the 6 linguistic input features.
    LPE_OUTPUT_DIM (32) is the model architecture dimension, separate from this.
    """
    windows = segment_audio(audio)
    if not windows:
        return []

    results = []
    for b in range(0, len(windows), batch_size):
        batch   = windows[b: b + batch_size]
        audio_t = torch.from_numpy(
            np.stack([audio[s:e].astype(np.float32) for s, e, _, _ in batch])
        ).to(device)

        # Always [batch, 6] — 6 linguistic input scalars, set to neutral 0.5
        lpe_t = torch.full(
            (len(batch), LPE_INPUT_FEATURES), 0.5, dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    out = model(audio_t, lpe_t)
            else:
                out = model(audio_t, lpe_t)

        probs = torch.sigmoid(out['sld_logit']).cpu().numpy().flatten()

        for i, (_, _, s_sec, e_sec) in enumerate(batch):
            p = float(probs[i])
            results.append({
                'window_idx': b + i,
                'start_sec' : s_sec,
                'end_sec'   : e_sec,
                'sld_prob'  : round(p, 4),
                'prediction': 'SLD' if p >= threshold else 'fluent',
            })

    return results


def compute_confidence(positive_rate: float, n_windows: int) -> str:
    if n_windows < 5:
        return 'low'
    consensus = max(positive_rate, 1 - positive_rate)
    if consensus >= 0.80:
        return 'high'
    elif consensus >= 0.60:
        return 'medium'
    return 'low'


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get('/health', response_model=HealthResponse, tags=['System'])
def health():
    """Service health check."""
    return {
        'status'       : 'ok',
        'model_loaded' : 'model' in _state,
        'device'       : _state.get('device', 'unknown'),
    }


@app.get('/model/info', response_model=ModelInfo, tags=['System'])
def model_info():
    """Loaded checkpoint metadata."""
    return {
        'checkpoint' : _state.get('ckpt_path'),
        'epoch'      : _state.get('epoch'),
        'val_f1'     : _state.get('val_f1'),
        'device'     : _state.get('device', 'unknown'),
        'window_sec' : WINDOW_SEC,
        'stride_sec' : STRIDE_SEC,
        'target_sr'  : TARGET_SR,
    }


@app.post('/predict', response_model=PredictionResponse, tags=['Prediction'])
async def predict(
    file           : UploadFile = File(description="Audio file. WAV recommended. Min 4 seconds."),
    threshold      : float      = Query(0.5, ge=0.0, le=1.0,
                                   description="SLD threshold. Default 0.5."),
    return_windows : bool       = Query(True,
                                   description="Include per-window scores in response."),
):
    """
    Detect Stutter-Like Disfluencies (SLD) in a speech recording.

    **Threshold guidance:**
    - 0.50 → balanced  (F1=0.941, default)
    - 0.10 → high recall  (catches more, more false positives)
    - 0.70 → high precision  (fewer false positives, may miss mild stuttering)

    **For the React team:**
    ```js
    const form = new FormData();
    form.append('file', audioBlob, 'recording.wav');
    const res = await fetch('http://localhost:8000/predict', {
        method: 'POST', body: form
    });
    const data = await res.json();
    ```
    """
    if 'model' not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t_start = time.time()
    suffix  = Path(file.filename or 'audio').suffix or '.wav'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        content  = await file.read()
        tmp.write(content)

    try:
        try:
            audio, _ = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            raise HTTPException(status_code=422,
                                detail=f"Cannot load audio: {e}")

        duration_sec = round(len(audio) / TARGET_SR, 2)
        if duration_sec < WINDOW_SEC:
            raise HTTPException(
                status_code=422,
                detail=f"Audio too short ({duration_sec}s). Need ≥{WINDOW_SEC}s."
            )

        window_scores = predict_windows(
            audio, _state['model'], _state['device'], threshold
        )
        if not window_scores:
            raise HTTPException(status_code=422,
                                detail="No valid speech windows found.")

        n_total    = len(window_scores)
        n_positive = sum(1 for w in window_scores if w['prediction'] == 'SLD')
        pos_rate   = round(n_positive / n_total, 4)

        pos_probs = [w['sld_prob'] for w in window_scores
                     if w['prediction'] == 'SLD']
        sld_prob  = round(
            float(np.mean(pos_probs)) if pos_probs
            else float(np.mean([w['sld_prob'] for w in window_scores])),
            4
        )

        elapsed = round(time.time() - t_start, 2)

        return {
            'filename'            : file.filename or 'uploaded_audio',
            'duration_sec'        : duration_sec,
            'sld_probability'     : sld_prob,
            'prediction'          : 'stuttering_detected' if n_positive > 0
                                    else 'fluent',
            'confidence'          : compute_confidence(pos_rate, n_total),
            'n_windows_total'     : n_total,
            'n_windows_positive'  : n_positive,
            'positive_rate'       : pos_rate,
            'window_scores'       : window_scores if return_windows else [],
            'processing_time_sec' : elapsed,
            'note'                : (
                "LPE features set to neutral (0.5). "
                "Ablation study: ΔF1=+0.002 (negligible)."
            ),
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500,
                            detail=f"Inference failed:\n{traceback.format_exc()}")
    finally:
        os.unlink(tmp_path)


# ── Generate Analysis ─────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    userInfo:     Dict[str, Any]
    positive_rate: float
    prediction:   str
    confidence:   str
    n_windows_total:    int
    n_windows_positive: int
    duration_sec: float

@app.post('/generate-analysis', tags=['Analysis'])
async def generate_analysis(req: AnalysisRequest):
    """Generate a Gemini-powered narrative for the SLD screening result."""
    sld_pct  = round(req.positive_rate * 100, 1)
    name     = req.userInfo.get('name', 'the child')
    age      = req.userInfo.get('age', '?')

    severity = (
        'Severe'   if req.positive_rate >= 0.50 else
        'Moderate' if req.positive_rate >= 0.25 else
        'Mild'     if req.positive_rate >= 0.10 else
        'Minimal'
    )
    detected = req.prediction == 'stuttering_detected'

    fallback = (
        f"OVERALL ASSESSMENT\n"
        f"{name} ({age} years) showed {severity.lower()} stutter-like disfluencies "
        f"in {sld_pct}% of the recorded speech "
        f"({'stuttering detected' if detected else 'no significant stuttering detected'}).\n\n"
        f"ACOUSTIC DETAILS\n"
        f"Total windows analysed: {req.n_windows_total}\n"
        f"Windows with SLD: {req.n_windows_positive}\n"
        f"SLD rate: {sld_pct}%\n"
        f"Confidence: {req.confidence}\n"
        f"Recording duration: {req.duration_sec:.1f}s\n\n"
        f"RECOMMENDATIONS\n"
        f"1. {'Consult a Speech-Language Pathologist for a formal evaluation.' if detected else 'Continue monitoring speech fluency over time.'}\n"
        f"2. Use slow, relaxed speech when speaking with the child.\n"
        f"3. Create a low-pressure environment to reduce communication anxiety.\n\n"
        f"IMPORTANT DISCLAIMER\n"
        f"This is an automated screening aid and does not constitute a clinical diagnosis. "
        f"A qualified Speech-Language Pathologist must evaluate the child formally."
    )

    prompt = f"""You are a Speech-Language Pathologist writing a screening summary for a parent.

YOUR INSTRUCTIONS:
1. DO NOT recalculate any scores.
2. DO NOT change the classification.
3. Your ONLY job is to format the provided data into a warm, empathetic, easy-to-read report.

CHILD INFORMATION:
- Name: {name}
- Age: {age} years

ACOUSTIC SCREENING RESULTS (Do not alter):
- SLD (Stutter-Like Disfluency) Rate: {sld_pct}%
- Severity Category: {severity}
- System Classification: {'STUTTERING DETECTED' if detected else 'FLUENT (No significant stuttering)'}
- Confidence: {req.confidence}
- Recording Duration: {req.duration_sec:.1f} seconds
- Windows Analysed: {req.n_windows_total}, Windows with SLD: {req.n_windows_positive}

Please write a comprehensive, plain-text summary (no markdown, no **, no ##, no special characters).

Structure your response with these exact section headings on their own lines:

OVERALL ASSESSMENT
2-3 sentences summarising the result in plain, reassuring language.

WHAT THIS MEANS
Explain what stutter-like disfluencies are and how this result relates to typical speech development for this age.

RECOMMENDATIONS
Simple numbered points (1., 2., 3.) — at least three actionable suggestions for parents.

IMPORTANT DISCLAIMER
State clearly this is a screening tool, not a clinical diagnosis, and that a Speech-Language Pathologist evaluation is required.

Keep the tone warm, empathetic, and parent-friendly. Plain text only."""

    # --- TIER 1: Gemini (Primary) ---
    if gemini_model is not None:
        try:
            response = gemini_model.generate_content(prompt)
            return {'analysis': response.text, 'source': 'gemini'}
        except Exception as e:
            print(f"[generate-analysis] Gemini error: {e}. Falling back to Groq.")

    # --- TIER 2: Groq (Fallback) ---
    if GROK_API_KEY:
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROK_API_KEY)
            groq_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65,
                max_tokens=1024,
            )
            return {'analysis': groq_response.choices[0].message.content, 'source': 'groq'}
        except Exception as e:
            print(f"[generate-analysis] Groq error: {e}. Using rule-based fallback.")

    # --- TIER 3: Hardcoded fallback ---
    return {'analysis': fallback, 'source': 'rule_based'}


# ── Find Doctors ──────────────────────────────────────────────────────────

class DoctorRequest(BaseModel):
    pincode: str

@app.post('/find-doctors', tags=['Analysis'])
async def find_doctors(req: DoctorRequest):
    """Find nearby Speech-Language Pathologists using Gemini then Groq as fallback."""
    if gemini_model is None and not GROK_API_KEY:
        return {'doctors': [], 'message': 'Set GEMINI_API_KEY or GROK_API_KEY to enable doctor search.'}

    doctor_prompt = f"""Find exactly 3 Speech-Language Pathologists or pediatric speech therapists near pincode {req.pincode} in India.

Format as a JSON array with exactly 3 entries:
[
  {{
    "name": "Dr. Name / Clinic Name",
    "specialty": "Speech-Language Pathologist",
    "address": "Full address",
    "city": "City",
    "distance": "X km",
    "phone": "Phone number or 'Contact via clinic'"
  }}
]

Return ONLY the JSON array, no other text."""

    def parse_doctors(text: str):
        text = text.strip()
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            return json.loads(json_match.group(0))
        return []

    # --- TIER 1: Gemini (Primary) ---
    if gemini_model is not None:
        try:
            response = gemini_model.generate_content(doctor_prompt)
            return {'doctors': parse_doctors(response.text)}
        except Exception as e:
            print(f"[find-doctors] Gemini error: {e}. Falling back to Groq.")

    # --- TIER 2: Groq (Fallback) ---
    if GROK_API_KEY:
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROK_API_KEY)
            groq_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": doctor_prompt}],
                temperature=0.4,
                max_tokens=512,
            )
            return {'doctors': parse_doctors(groq_response.choices[0].message.content)}
        except Exception as e:
            print(f"[find-doctors] Groq error: {e}.")
            return {'doctors': [], 'error': str(e)}

    return {'doctors': []}


# ── Download Report ───────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    report: str
    name:   str = 'Child'

@app.post('/download-report', tags=['Analysis'])
async def download_report(req: ReportRequest):
    """Return the analysis as a downloadable .txt file."""
    child_name = req.name.strip().replace(' ', '_') or 'Child'
    content = (
        f"=========================================\n"
        f" NEUROSAGE SPEECH DISORDER SCREENING REPORT\n"
        f"=========================================\n\n"
        f"{req.report}\n\n"
        f"=========================================\n"
        f" Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f" DISCLAIMER: Automated screening only. Not a clinical diagnosis.\n"
        f"=========================================\n"
    )
    buf = io.BytesIO(content.encode('utf-8'))
    return StreamingResponse(
        buf,
        media_type='text/plain',
        headers={'Content-Disposition': f'attachment; filename="{child_name}_NeuroSage_Speech_Report.txt"'}
    )