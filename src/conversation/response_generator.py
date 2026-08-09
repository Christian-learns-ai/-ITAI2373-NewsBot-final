from src.conversation.intent_classifier import IntentClassifier
from src.conversation.query_processor import QueryProcessor

class ResponseGenerator:
    """Generates responses and manages dialog context continuity."""

    def __init__(self, newsbot_system=None):
        self.newsbot = newsbot_system
        self.intent_classifier = IntentClassifier()
        self.query_processor = QueryProcessor()

    def process_query(self, user_query: str, conversation_context: dict = None) -> dict:
        if conversation_context and "last_entities" in conversation_context:
            entities = self.query_processor.extract_query_entities(user_query)
            for key, val in conversation_context["last_entities"].items():
                if not entities.get(key):
                    entities[key] = val
        else:
            entities = self.query_processor.extract_query_entities(user_query)

        intent = self.intent_classifier.classify_intent(user_query)
        results = f"Executed {intent.upper()} operation for entities: {entities}"
        response_text = self.format_response(results, intent, entities)

        return {
            "response": response_text,
            "intent": intent,
            "entities": entities,
            "context": {"last_intent": intent, "last_entities": entities},
        }

    def format_response(self, query_results: str, intent: str, entities: dict) -> str:
        topic_str = entities.get("category") or (
            ", ".join(entities["named_entities"]) if entities["named_entities"] else "your topic"
        )
        if intent == "summarize":
            return f" Here is the executive summary regarding {topic_str}:\n{query_results}"
        elif intent == "compare":
            return f" Here is the comparative coverage analysis for {topic_str}:\n{query_results}"
        elif intent == "analyze":
            return f"📊 Sentiment and trend analysis for {topic_str}:\n{query_results}"
        else:
            return f"🔍 Found the following top articles regarding {topic_str}:\n{query_results}"

    def handle_follow_up(self, follow_up_query: str, conversation_history: list) -> dict:
        last_context = conversation_history[-1].get("context", {}) if conversation_history else None
        return self.process_query(follow_up_query, conversation_context=last_context)
