class DataValidator:
    """Validates input article quality and constraints."""

    @staticmethod
    def validate_article(article_text: str) -> dict:
        if not article_text or not isinstance(article_text, str):
            return {"is_valid": False, "reason": "Empty or non-string input"}
        cleaned = article_text.strip()
        if len(cleaned) < 20:
            return {"is_valid": False, "reason": "Text length below threshold (min 20 chars)"}
        return {"is_valid": True, "length": len(cleaned)}
