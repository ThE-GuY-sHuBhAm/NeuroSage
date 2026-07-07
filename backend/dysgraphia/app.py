"""
app.py  (updated)
=================
Flask backend for the Dysgraphia Screening System.

Changes from original:
  - /api/generate-questions now uses QuestionnaireEngine (Excel-based, no Gemini API).
  - /api/analyze-image scoring unchanged (EfficientNet-B0 anomaly score).
  - /api/generate-analysis uses updated combined scoring (Q + I/2, threshold 66.2).
  - Gemini API is now ONLY used for /api/generate-analysis and /api/find-doctors.
    If GEMINI_API_KEY is absent those two endpoints return graceful errors;
    the core screening flow still works without it.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import torch
from PIL import Image
import io
from dotenv import load_dotenv
from image_analyzer import ImageAnalyzer
from questionnaire_engine import QuestionnaireEngine

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Gemini (optional — only for analysis narrative and doctor search) ──────────
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_model = None

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        print("Gemini API initialised successfully.")
    except Exception as e:
        print(f"Warning: Gemini initialisation failed: {e}. Analysis narrative disabled.")
else:
    print("GEMINI_API_KEY not set. AI narrative and doctor-search endpoints disabled.")

# ── Image Analyser ─────────────────────────────────────────────────────────────
default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = os.getenv('DEVICE', default_device)

print(f"Initialising Image Analyser on {DEVICE} …")
image_analyzer = ImageAnalyzer(device=DEVICE)

# ── Questionnaire Engine (static, Excel-backed) ────────────────────────────────
QUESTIONNAIRE_PATH = os.getenv('QUESTIONNAIRE_PATH', 'Dysgraphia_Questionnaire.xlsx')
print(f"Loading questionnaire from: {QUESTIONNAIRE_PATH}")
questionnaire_engine = QuestionnaireEngine(QUESTIONNAIRE_PATH)

# ── Scoring constants ──────────────────────────────────────────────────────────
IMAGE_THRESHOLD  = 52.14  # MobileNetV3-Small Youden-optimal image-only threshold
COMBO_THRESHOLD  = 45.68  # 80/20 weighted fusion threshold (zero-FN optimised)
Q_THRESHOLD      = 30     # questionnaire-only threshold (raw score, 10–50)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status':           'healthy',
        'model_loaded':     image_analyzer.is_loaded(),
        'device':           DEVICE,
        'gemini_available': gemini_model is not None,
        'questionnaire':    {
            'source':     QUESTIONNAIRE_PATH,
            'ages':       questionnaire_engine.available_ages,
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE QUESTIONS  (now static, no Gemini)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    """
    Return 10 age-appropriate questions from the Excel questionnaire bank.
    No external API calls required.
    """
    try:
        data = request.json or {}
        age  = int(data.get('age', 8))

        questions = questionnaire_engine.get_questions(age=age)

        return jsonify({
            'questions': questions,
            'source':    'static_excel',
            'age_used':  age,
        })

    except Exception as e:
        print(f"Error generating questions: {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE IMAGE
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyse a handwriting image and return anomaly score."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file  = request.files['image']
        image_bytes = image_file.read()
        image       = Image.open(io.BytesIO(image_bytes))

        t0            = time.perf_counter()
        anomaly_score = image_analyzer.get_anomaly_score(image)
        latency_ms    = (time.perf_counter() - t0) * 1000.0

        if anomaly_score is None:
            return jsonify({'error': 'Failed to analyse image'}), 500

        is_atypical = anomaly_score > IMAGE_THRESHOLD

        return jsonify({
            'anomaly_score': float(anomaly_score),
            'is_atypical':   is_atypical,
            'threshold':     IMAGE_THRESHOLD,
            'latency_ms':    round(latency_ms, 2),
        })

    except Exception as e:
        print(f"Error analysing image: {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE COMBINED SCORE  (new endpoint — pure logic, no AI)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/compute-score', methods=['POST'])
def compute_score():
    """
    Compute questionnaire score, combined score, and final classification.
    Does NOT require Gemini.

    Body JSON:
        {
            "answers":      { "0": 2, "1": 3, … "9": 1 },   // 0-indexed, values 0-4
            "image_score":  45.7                              // anomaly score from /api/analyze-image
        }
    """
    try:
        data         = request.json or {}
        answers      = data.get('answers', {})
        image_score  = float(data.get('image_score', 0.0))

        q_result     = QuestionnaireEngine.compute_score(answers)
        combo_result = QuestionnaireEngine.compute_combined_score(
                           q_raw_score=q_result['raw_score'],
                           image_anomaly_score=image_score
                       )

        return jsonify({
            'questionnaire': q_result,
            'combined':      combo_result,
        })

    except Exception as e:
        print(f"Error computing score: {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ANALYSIS NARRATIVE  (still uses Gemini if available)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/generate-analysis', methods=['POST'])
def generate_analysis():
    """
    Generate a detailed narrative analysis using Gemini.
    If Gemini is unavailable, returns a structured rule-based summary instead.
    """
    try:
        data                = request.json or {}
        user_info           = data.get('userInfo', {})
        questions           = data.get('questions', [])
        answers             = data.get('answers', {})
        questionnaire_score = float(data.get('questionnaireScore', 0))
        image_score         = float(data.get('imageScore', 0))
        final_score         = float(data.get('finalScore', 0))

        is_atypical = final_score >= COMBO_THRESHOLD

        # ── Rule-based fallback (always computable) ────────────────────────
        severity = "high concern" if final_score >= 80 else \
                   "moderate concern" if final_score >= COMBO_THRESHOLD else \
                   "low concern"

        fallback_analysis = (
            f"OVERALL ASSESSMENT\n"
            f"Based on the screening results, {user_info.get('name', 'the child')} "
            f"({user_info.get('age', '?')} years old) shows {severity}. "
            f"The combined score is {final_score:.1f}/100 "
            f"(threshold: {COMBO_THRESHOLD}), which indicates "
            f"{'atypical' if is_atypical else 'typical'} handwriting patterns.\n\n"
            f"COMPONENT SCORES\n"
            f"Questionnaire Score: {questionnaire_score:.1f}/50 "
            f"(threshold for concern: >= {Q_THRESHOLD})\n"
            f"Image Anomaly Score: {image_score:.1f}/100 "
            f"(threshold for concern: > {IMAGE_THRESHOLD})\n"
            f"Combined Score (80% Image + 20% Quiz): {final_score:.1f}/100 "
            f"(threshold: {COMBO_THRESHOLD})\n\n"
            f"RECOMMENDATIONS\n"
            f"1. {'Seek evaluation from a pediatric occupational therapist.' if is_atypical else 'Continue monitoring handwriting development.'}\n"
            f"2. Practise fine motor activities such as drawing, cutting, and threading beads.\n"
            f"3. Ensure proper pencil grip and posture during writing tasks.\n\n"
            f"IMPORTANT DISCLAIMER\n"
            f"This is a screening tool and does not constitute a clinical diagnosis. "
            f"A qualified professional evaluation is required for a formal diagnosis."
        )

        if gemini_model is None:
            return jsonify({'analysis': fallback_analysis, 'source': 'rule_based'})

        # ── Gemini narrative ───────────────────────────────────────────────
        score_labels = ['Never', 'Rarely', 'Sometimes', 'Frequently', 'Always']
        answer_details = []
        for i, q in enumerate(questions):
            val   = answers.get(str(i), answers.get(i, 0))
            val   = int(val)
            label = score_labels[val] if 0 <= val < len(score_labels) else "Unknown"
            answer_details.append(
                f"Q: {q['question']}\nCategory: {q['category']}\nAnswer: {label} ({val}/4)"
            )

        prompt = f"""You are a pediatric occupational therapist writing a screening summary for a parent.

YOUR INSTRUCTIONS:
1. DO NOT recalculate any scores.
2. DO NOT change the classification.
3. Your ONLY job is to format the provided scores into a warm, empathetic, and easy-to-read report.

CHILD INFORMATION:
- Name: {user_info.get('name')}
- Age: {user_info.get('age')} years

QUESTIONNAIRE RESPONSES:
{chr(10).join(answer_details)}

HARDCODED BACKEND SCORES (Do not alter):
- Questionnaire Score: {questionnaire_score:.1f}/50 (threshold for concern: >= {Q_THRESHOLD})
- Image Anomaly Score: {image_score:.1f}/100 (threshold for concern: > {IMAGE_THRESHOLD})
- Final Combined Score (80% Image + 20% Quiz): {final_score:.1f}/100 (threshold: {COMBO_THRESHOLD})
- System Classification: {'ATYPICAL (At Risk)' if is_atypical else 'TYPICAL (Low Risk)'}

Please write a comprehensive, plain-text summary (no markdown formatting, no ##, **, _, or special characters).

Structure your response with these exact section headings on their own lines:

OVERALL ASSESSMENT
Write 2-3 sentences summarising the results in plain, reassuring language.

KEY FINDINGS BY CATEGORY
Analyse patterns in the questionnaire responses across different categories. Write in clear paragraphs.

DEVELOPMENTAL CONTEXT
Explain how these findings relate to typical development for a child of this age.

RECOMMENDATIONS
Use simple numbered points (1., 2., 3.) — at least three actionable suggestions.

IMPORTANT DISCLAIMER
Clearly state this is a screening tool, not a clinical diagnosis, and that a qualified professional evaluation is required.

Keep the tone empathetic, informative, and parent-friendly. Write in plain text only."""

        response = gemini_model.generate_content(prompt)
        return jsonify({'analysis': response.text, 'source': 'gemini'})

    except Exception as e:
        try:
            import os
            from groq import Groq
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key:
                client = Groq(api_key=groq_api_key)
                groq_response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192"
                )
                return jsonify({'analysis': groq_response.choices[0].message.content, 'source': 'groq'})
        except Exception as groq_e:
            print(f"Groq fallback failed: {groq_e}")
            
        print(f"Error generating analysis: {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# FIND DOCTORS  (Gemini-dependent, graceful failure)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/find-doctors', methods=['POST'])
def find_doctors():
    """Find specialists near a pincode (requires Gemini)."""
    if gemini_model is None:
        return jsonify({
            'doctors': [],
            'message': 'Doctor search requires Gemini API. Set GEMINI_API_KEY in your .env file.'
        }), 200

    try:
        data    = request.json or {}
        pincode = data.get('pincode', '')

        prompt = f"""Find pediatric occupational therapists, developmental pediatricians, and specialists who treat dysgraphia near pincode {pincode} in India.

Provide 5-7 recommendations with:
- Doctor/Clinic name, Specialty, Address with city, Approximate distance from pincode {pincode}, Contact info if available

Format as JSON array:
[
  {{
    "name": "Dr. Name / Clinic Name",
    "specialty": "Pediatric Occupational Therapist",
    "address": "Full address",
    "city": "City",
    "distance": "X km",
    "phone": "Phone number or 'Contact via clinic'"
  }}
]

Return ONLY the JSON array, no other text."""

        import json, re
        response    = gemini_model.generate_content(prompt)
        text        = response.text.strip()
        json_match  = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            doctors = json.loads(json_match.group(0))
            return jsonify({'doctors': doctors})
        return jsonify({'doctors': []})

    except Exception as e:
        try:
            import os
            from groq import Groq
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key:
                client = Groq(api_key=groq_api_key)
                groq_response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192"
                )
                text = groq_response.choices[0].message.content.strip()
                import json, re
                json_match = re.search(r'\[[\s\S]*\]', text)
                if json_match:
                    doctors = json.loads(json_match.group(0))
                    return jsonify({'doctors': doctors})
        except Exception as groq_e:
            print(f"Groq fallback failed: {groq_e}")
            
        print(f"Error finding doctors: {e}")
        return jsonify({'error': str(e), 'doctors': []}), 200



# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD REPORT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/download-report', methods=['POST'])
def download_report():
    """
    Takes the generated analysis text and returns it as a downloadable .txt file.

    Body JSON:
        {
            "report": "Full report text...",
            "name":   "Child Name"          // used for the filename
        }
    """
    try:
        data        = request.json or {}
        report_text = data.get('report', 'No report content available.')
        child_name  = data.get('name', 'Child').strip().replace(' ', '_')

        formatted_report = (
            f"=========================================\n"
            f" NEUROSAGE DYSGRAPHIA SCREENING REPORT\n"
            f"=========================================\n\n"
            f"{report_text}\n\n"
            f"=========================================\n"
            f" Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f" DISCLAIMER: This report is produced by an automated screening\n"
            f" prototype and does not constitute a clinical diagnosis.\n"
            f"=========================================\n"
        )

        mem_file = io.BytesIO()
        mem_file.write(formatted_report.encode('utf-8'))
        mem_file.seek(0)

        return send_file(
            mem_file,
            as_attachment=True,
            download_name=f"{child_name}_NeuroSage_Report.txt",
            mimetype='text/plain'
        )

    except Exception as e:
        print(f"Error generating download: {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.getenv('FLASK_ENV') == 'development'
    )