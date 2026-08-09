from transformers import pipeline
from src.multilingual.language_detector import LanguageDetector

class TranslatorService:
    """Neural translation via MarianMT Hugging Face pipelines."""

    def __init__(self):
        self.detector = LanguageDetector()
        self.translation_models = {}

    def translate_text(self, text: str, target_language: str = "en") -> dict:
        source_info = self.detector.detect_language(text)
        src_lang = source_info["language"]

        if src_lang == target_language:
            return {
                "translated_text": text,
                "source_language": src_lang,
                "target_language": target_language,
                "status": "No translation needed",
            }

        try:
            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{target_language}"
            if model_name not in self.translation_models:
                self.translation_models[model_name] = pipeline("translation", model=model_name)

            translator = self.translation_models[model_name]
            translation_result = translator(text, max_length=512)
            translated_text = translation_result[0]["translation_text"]

            return {
                "translated_text": translated_text,
                "source_language": src_lang,
                "target_language": target_language,
                "status": "Success",
            }
        except Exception as e:
            return {
                "translated_text": text,
                "source_language": src_lang,
                "target_language": target_language,
                "status": f"Fallback/Error: {str(e)}",
            }
