from collections import defaultdict
import networkx as nx
import spacy

class EntityRelationshipMapper:
    """Named Entity Recognition and NetworkX Knowledge Graph creation."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)
        self.graph = nx.Graph()

    def extract_entities(self, article_text: str) -> dict:
        doc = self.nlp(article_text)
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        return {label: sorted(list(set(items))) for label, items in entities.items()}

    def extract_relationships(self, article_text: str) -> list[tuple]:
        doc = self.nlp(article_text)
        relationships = []
        for sent in doc.sents:
            sent_doc = self.nlp(sent.text)
            entities_in_sent = [ent.text for ent in sent_doc.ents]
            if len(entities_in_sent) >= 2:
                for i in range(len(entities_in_sent) - 1):
                    e1 = entities_in_sent[i]
                    e2 = entities_in_sent[i + 1]
                    verbs = [token.lemma_ for token in sent_doc if token.pos_ == "VERB"]
                    relation = verbs[0] if verbs else "associated_with"
                    relationships.append((e1, relation, e2))
        return relationships

    def build_knowledge_graph(self, articles: list[str]):
        for article in articles:
            rels = self.extract_relationships(article)
            for source, relation, target in rels:
                self.graph.add_node(source)
                self.graph.add_node(target)
                self.graph.add_edge(source, target, relationship=relation)

    def find_entity_connections(self, entity1: str, entity2: str) -> list[list[str]]:
        if not self.graph.has_node(entity1) or not self.graph.has_node(entity2):
            return []
        try:
            return list(nx.all_shortest_paths(self.graph, source=entity1, target=entity2))
        except nx.NetworkXNoPath:
            return []
