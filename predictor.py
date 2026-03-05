"""Dummy regional predictor interface for NegotiationDashboard.
Replace internals with your real model implementation.
"""

from __future__ import annotations


class RegionPredictor:
    def __init__(self, language: str = "english", model_name: str = "svm") -> None:
        self.language = language
        self.model_name = model_name

    def predict_one(self, text: str):
        text_l = (text or "").lower()

        # Lightweight placeholder behavior. Replace with your trained inference.
        if any(k in text_l for k in ["mate", "cheers", "bloody"]):
            probs = {"U.K.": 0.63, "U.S.": 0.18, "South Africa": 0.11, "Mexico": 0.08}
        elif any(k in text_l for k in ["refund", "attorney", "lawsuit"]):
            probs = {"U.S.": 0.71, "U.K.": 0.14, "Mexico": 0.09, "South Africa": 0.06}
        else:
            probs = {"U.S.": 0.4, "U.K.": 0.25, "Mexico": 0.2, "South Africa": 0.15}

        pred = max(probs, key=probs.get)
        return {
            "text": text,
            "predicted_class": pred,
            "confidence": float(probs[pred]),
            "probabilities": probs,
        }
