class CrossLingualAnalyzer:
    """Cross-lingual perspective and cultural term extraction."""

    def analyze_cross_lingual(self, articles_by_language: dict[str, list[str]]) -> dict:
        cross_lingual_insights = {}
        for lang, articles in articles_by_language.items():
            total_words = sum(len(art.split()) for art in articles)
            avg_length = total_words / max(len(articles), 1)
            cross_lingual_insights[lang] = {
                "article_count": len(articles),
                "avg_article_length": round(avg_length, 2),
                "sample_preview": articles[0][:150] + "..." if articles else "",
            }

        return {
            "languages_analyzed": list(articles_by_language.keys()),
            "language_breakdown": cross_lingual_insights,
            "comparative_summary": f"Analyzed coverage across {len(articles_by_language)} languages.",
        }

    def extract_cultural_context(self, text: str, source_language: str) -> list[dict]:
        cultural_keywords = {
            "es": ["siesta", "autonomía", "congreso", "comunidad"],
            "fr": ["laïcité", "mouvement social", "département", "préfecture"],
            "de": ["bundestag", "energiewende", "länder", "volksentscheid"],
        }
        detected_context = []
        words = text.lower().split()
        relevant_terms = cultural_keywords.get(source_language, [])

        for term in relevant_terms:
            if term in words:
                detected_context.append({
                    "cultural_term": term,
                    "source_language": source_language,
                    "context_note": f"Key regional/cultural term identified: '{term}'",
                })
        return detected_context
