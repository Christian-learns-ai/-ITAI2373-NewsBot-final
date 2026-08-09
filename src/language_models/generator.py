from collections import Counter
import spacy

class ContentEnhancer:
    """Content augmentation, gap analysis, and headline generation."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self.knowledge_base = {
            "openai": "OpenAI is an AI research organization created to promote and develop friendly AI.",
            "tesla": "Tesla, Inc. is an American electric vehicle and clean energy company.",
            "apple": "Apple Inc. is a multinational technology company specializing in consumer electronics.",
            "climate change": "Global warming and climate change refer to the long-term shifts in temperatures.",
        }

    def enhance_article(self, article_text: str) -> dict:
        doc = self.nlp(article_text)
        entities = list(set([ent.text for ent in doc.ents]))

        entity_context = {}
        for ent in entities:
            ent_key = ent.lower()
            if ent_key in self.knowledge_base:
                entity_context[ent] = self.knowledge_base[ent_key]

        highlights = [
            sent.text.strip()
            for sent in doc.sents
            if any(ent in sent.text for ent in entities[:3])
        ][:3]

        return {
            "original_length": len(article_text),
            "key_entities_found": entities,
            "entity_background": entity_context,
            "article_highlights": highlights,
        }

    def generate_insights(self, articles: list[str]) -> dict:
        all_entities = []
        for art in articles:
            doc = self.nlp(art)
            all_entities.extend([ent.text for ent in doc.ents])

        entity_counts = Counter(all_entities)
        top_stakeholders = entity_counts.most_common(5)

        return {
            "total_articles_analyzed": len(articles),
            "top_key_stakeholders": dict(top_stakeholders),
            "dominant_themes": [ent for ent, _ in top_stakeholders],
            "insight_summary": f"Coverage is heavily focused around {', '.join([e[0] for e in top_stakeholders[:3]])}.",
        }

    def detect_information_gaps(self, articles: list[str], topic: str) -> dict:
        combined_text = " ".join(articles).lower()
        expected_angles = [
            "economic impact", "regulatory policy", "environmental aspect",
            "public reaction", "future outlook", "historical context",
        ]
        missing_angles = [angle for angle in expected_angles if angle not in combined_text]
        covered_angles = [angle for angle in expected_angles if angle in combined_text]

        return {
            "target_topic": topic,
            "covered_perspective_angles": covered_angles,
            "detected_information_gaps": missing_angles,
            "recommendation": f"Consider adding coverage on: {', '.join(missing_angles)}." if missing_angles else "Coverage is well-rounded.",
        }
