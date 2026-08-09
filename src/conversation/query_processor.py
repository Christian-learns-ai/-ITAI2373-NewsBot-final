import spacy

class QueryProcessor:
    """Entities and slot parsing for conversational queries."""

    def __init__(self, spacy_model="en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    def extract_query_entities(self, user_query: str) -> dict:
        doc = self.nlp(user_query)
        entities = {
            "named_entities": [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE"]],
            "timeframe": None,
            "sentiment_filter": None,
            "category": None,
        }
        time_ents = [ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]
        if time_ents:
            entities["timeframe"] = time_ents[0]

        query_lower = user_query.lower()
        if "positive" in query_lower:
            entities["sentiment_filter"] = "positive"
        elif "negative" in query_lower:
            entities["sentiment_filter"] = "negative"

        categories = ["tech", "technology", "politics", "business", "sports", "entertainment"]
        for cat in categories:
            if cat in query_lower:
                entities["category"] = cat
                break

        return entities
