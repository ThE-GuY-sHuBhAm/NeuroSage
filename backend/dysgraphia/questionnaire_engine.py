"""
questionnaire_engine.py
=======================
Static questionnaire engine — reads questions from Dysgraphia_Questionnaire.xlsx.
No Gemini API key required.

Usage:
    from questionnaire_engine import QuestionnaireEngine

    engine = QuestionnaireEngine('Dysgraphia_Questionnaire.xlsx')
    questions = engine.get_questions(age=7)   # list of 10 dicts
    score     = engine.compute_score(answers) # answers = {0: 2, 1: 4, ...}

Scoring (per scoring_logic.docx):
    0 (Never)      = 1 point
    1 (Rarely)     = 2 points
    2 (Sometimes)  = 3 points
    3 (Frequently) = 4 points
    4 (Always)     = 5 points

    Total Q score  = sum of 10 answers' points  (range: 10 – 50)
    Questionnaire threshold = 30
"""

import os
import random
import logging
from pathlib import Path

import openpyxl

logger = logging.getLogger(__name__)

# Map 0-indexed frontend answer value → score points
ANSWER_TO_POINTS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
ANSWER_LABELS    = {0: 'Never', 1: 'Rarely', 2: 'Sometimes', 3: 'Frequently', 4: 'Always'}
Q_THRESHOLD      = 30   # questionnaire score threshold for concern
N_QUESTIONS      = 10   # number of questions to sample per session


class QuestionnaireEngine:
    """
    Loads questions from Dysgraphia_Questionnaire.xlsx and provides
    age-appropriate question selection and scoring.
    """

    def __init__(self, xlsx_path: str):
        self.xlsx_path = xlsx_path
        self._db: dict[int, list[dict]] = {}  # age → list of question dicts
        self._load()

    # ─────────────────────────────────────────────────────────
    # LOADING
    # ─────────────────────────────────────────────────────────
    def _load(self):
        if not os.path.exists(self.xlsx_path):
            raise FileNotFoundError(f"Questionnaire file not found: {self.xlsx_path}")

        wb = openpyxl.load_workbook(self.xlsx_path, read_only=True, data_only=True)
        ws = wb.active

        rows_loaded = 0
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] is None:
                continue
            age_raw, category, question = row[0], row[1], row[2]
            age = int(float(age_raw))
            if age not in self._db:
                self._db[age] = []
            self._db[age].append({'question': str(question), 'category': str(category)})
            rows_loaded += 1

        wb.close()
        logger.info(f"QuestionnaireEngine: loaded {rows_loaded} questions "
                    f"for ages {sorted(self._db.keys())}")

    # ─────────────────────────────────────────────────────────
    # QUESTION SELECTION
    # ─────────────────────────────────────────────────────────
    def _nearest_age(self, age: int) -> int:
        """Find the closest available age bracket."""
        available = sorted(self._db.keys())
        return min(available, key=lambda a: abs(a - age))

    def get_questions(self, age: int, seed: int | None = None) -> list[dict]:
        """
        Return N_QUESTIONS (10) questions appropriate for the given age.

        Strategy:
        - Use the exact age bracket if available.
        - Fall back to the nearest bracket (e.g., age 13 → bracket 12).
        - Sample across categories proportionally; if not enough unique
          categories, fill remaining slots from the same bracket randomly.
        - Each returned dict has keys: 'question', 'category'
        """
        rng = random.Random(seed)

        bracket = self._nearest_age(age)
        pool = self._db[bracket].copy()

        # Stratified sample: try to cover all categories evenly
        from collections import defaultdict
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for q in pool:
            by_cat[q['category']].append(q)

        selected = []
        categories = list(by_cat.keys())
        rng.shuffle(categories)

        # Round-robin pick one from each category until we have N_QUESTIONS
        cat_iters = {c: rng.sample(qs, len(qs)) for c, qs in by_cat.items()}
        cat_cycle = categories * (N_QUESTIONS // len(categories) + 1)
        used_indices = {c: 0 for c in categories}

        for cat in cat_cycle:
            if len(selected) >= N_QUESTIONS:
                break
            idx = used_indices[cat]
            if idx < len(cat_iters[cat]):
                selected.append(cat_iters[cat][idx])
                used_indices[cat] += 1

        # If still short, fill with remaining questions from pool
        if len(selected) < N_QUESTIONS:
            used_set = set(id(q) for q in selected)
            remaining = [q for q in pool if id(q) not in used_set]
            rng.shuffle(remaining)
            selected.extend(remaining[:N_QUESTIONS - len(selected)])

        return selected[:N_QUESTIONS]

    # ─────────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def compute_score(answers: dict) -> dict:
        """
        Compute questionnaire score from answers dict.

        Args:
            answers: dict mapping question index (int or str) to answer value (0-4)

        Returns:
            dict with:
                'raw_score'        : int   (10-50)
                'normalized_score' : float (0-100, scaled from 10-50)
                'threshold_exceeded': bool (raw_score >= Q_THRESHOLD)
                'answer_details'   : list of dicts
        """
        total_points = 0
        details = []
        for i in range(N_QUESTIONS):
            key = i if i in answers else str(i)
            val = int(answers.get(key, 0))
            val = max(0, min(4, val))  # clamp to 0-4
            pts = ANSWER_TO_POINTS[val]
            total_points += pts
            details.append({
                'question_index': i,
                'answer_value':   val,
                'answer_label':   ANSWER_LABELS[val],
                'points':         pts,
            })

        # Normalize to 0-100: (raw - 10) / (50 - 10) * 100
        normalized = (total_points - 10) / 40.0 * 100.0

        return {
            'raw_score':          total_points,
            'normalized_score':   round(normalized, 2),
            'threshold_exceeded': total_points >= Q_THRESHOLD,
            'answer_details':     details,
        }

    @staticmethod
    def compute_combined_score(q_raw_score: float, image_anomaly_score: float) -> dict:
        """
        Compute combined score per paper Eq. 8 & 9:
            Qnorm = (Qraw - 10) / 40 * 100          (normalise raw score to 0-100)
            C     = 0.80 * Simg + 0.20 * Qnorm      (80/20 weighted fusion)
            Classification = At Risk if C >= 45.68, else Typical

        Grid-search optimal weights: wI = 0.80, wQ = 0.20, τFN = 45.68
        (MobileNetV3-Small backbone, zero false-negative objective)

        Args:
            q_raw_score         : questionnaire raw score (10–50)
            image_anomaly_score : image anomaly score (0–100)

        Returns:
            dict with combined score, prediction, and all component details
        """
        # Normalise questionnaire raw score (10–50) to 0–100
        q_norm = ((q_raw_score - 10) / 40.0) * 100.0

        # Weighted fusion
        combined = (0.80 * image_anomaly_score) + (0.20 * q_norm)
        dysgraphia_detected = combined >= 45.68

        return {
            'q_raw_score':          q_raw_score,
            'q_normalized_score':   round(q_norm, 2),
            'image_anomaly_score':  image_anomaly_score,
            'combined_score':       round(combined, 3),
            'combined_threshold':   45.68,
            'dysgraphia_detected':  dysgraphia_detected,
            'classification':       'At Risk' if dysgraphia_detected else 'Typical',
            'component_q_weighted': round(0.20 * q_norm, 3),
            'component_i_weighted': round(0.80 * image_anomaly_score, 3),
        }

    # ─────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────
    @property
    def available_ages(self) -> list[int]:
        return sorted(self._db.keys())

    def question_count_for_age(self, age: int) -> int:
        bracket = self._nearest_age(age)
        return len(self._db.get(bracket, []))