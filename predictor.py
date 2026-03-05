"""
predictor.py  —  inference module for both classifier variants.

Standalone usage:
    from predictor import RegionPredictor

    p = RegionPredictor(language='english')
    result  = p.predict_one("I want a full refund.")
    results = p.predict_batch(["text 1", "text 2"])

Flask usage:
    See app.py — call RegionPredictor.predict_batch(texts) inside the route.

Expected directory layout:
    models/
        label_encoder.pkl
        svm.pkl / logistic_regression.pkl / random_forest.pkl
    models_country/
        label_encoder.pkl
        svm.pkl / logistic_regression.pkl / random_forest.pkl
    predictor.py
    app.py
    index.html
"""

import os, joblib
import numpy as np
from typing import Optional, List
from sentence_transformers import SentenceTransformer

_ENCODER_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
_encoder: Optional[SentenceTransformer] = None

def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(_ENCODER_NAME)
    return _encoder


CONFIGS = {
    'chinese': {
        'model_dir': 'models_CN',
        'classes':   ['Central', 'North', 'Wu_Min', 'Xian_Yue'],
    },
    'english': {
        'model_dir': 'models_EN',
        'classes':   ['Mexico', 'South Africa', 'U.K.', 'U.S.'],
    },
}

AVAILABLE_MODELS = ['svm', 'logistic_regression', 'random_forest']


class RegionPredictor:
    """
    Wraps a trained sklearn classifier + sentence-transformers encoder.

    Parameters
    ----------
    language   : 'english' | 'chinese'
    model_name : 'svm' | 'logistic_regression' | 'random_forest'
    """

    def __init__(self, language: str = 'english', model_name: str = 'svm'):
        if language not in CONFIGS:
            raise ValueError(f"language must be one of {list(CONFIGS)}")
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"model_name must be one of {AVAILABLE_MODELS}")

        cfg = CONFIGS[language]
        self.language   = language
        self.model_name = model_name

        model_path = os.path.join(cfg['model_dir'], f'{model_name}.pkl')
        le_path    = os.path.join(cfg['model_dir'], 'label_encoder.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"Label encoder not found: {le_path}")

        self.clf     = joblib.load(model_path)
        self.le      = joblib.load(le_path)
        self.classes = list(self.le.classes_)

    def _embed(self, texts: List[str]) -> np.ndarray:
        return _get_encoder().encode(texts, batch_size=32, show_progress_bar=False)

    def _fmt(self, text: str, proba: np.ndarray) -> dict:
        idx = int(np.argmax(proba))
        return {
            'text':            text,
            'predicted_class': self.classes[idx],
            'confidence':      float(proba[idx]),
            'probabilities':   {c: float(p) for c, p in zip(self.classes, proba)},
        }

    def predict_one(self, text: str) -> dict:
        """Predict a single utterance."""
        emb = self._embed([text])
        return self._fmt(text, self.clf.predict_proba(emb)[0])

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Predict a list of utterances (batched encoding)."""
        if not texts:
            return []
        embs   = self._embed(texts)
        probas = self.clf.predict_proba(embs)
        return [self._fmt(t, p) for t, p in zip(texts, probas)]


# Quick CLI test
if __name__ == '__main__':
    samples = {
        'english': [
            "I want a full refund, this product is defective.",
            "I'll leave a bad review unless you sort this out.",
            "Could we maybe split the refund? I understand your position.",
        ],
        'chinese': [
            "我要退款，货不对板。",
            "你们不退钱我就投诉！",
            "亲亲，麻烦您理解一下我们的规定哟。",
        ],
    }

    for lang, texts in samples.items():
        print(f"\n{'='*50}\n  {lang}\n{'='*50}")
        pred = RegionPredictor(language=lang, model_name='svm')
        for r in pred.predict_batch(texts):
            print(f"  [{r['predicted_class']:14s}] {r['confidence']*100:5.1f}%  →  {r['text'][:55]}")
