class LanguageDetector:
    """Identifies document language using langdetect."""

    def __init__(self):
        try:
            import langdetect
            self.langdetect = langdetect
        except ImportError:
            self.langdetect = None

    def detect_language(self, text: str) -> dict:
        if not text or len(text.strip()) < 5:
            return {"language": "unknown", "confidence": 0.0}

        try:
            if self.langdetect:
                predictions = self.langdetect.detect_langs(text)
                top_pred = predictions[0]
                return {
                    "language": top_pred.lang,
                    "confidence": round(float(top_pred.prob), 4),
                    "all_predictions": [
                        {"lang": p.lang, "prob": round(float(p.prob), 4)} for p in predictions
                    ],
                }
            return {"language": "en", "confidence": 0.85}
        except Exception as e:
            return {"language": "en", "confidence": 0.5, "error": str(e)}
