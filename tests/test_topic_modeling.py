from src.analysis.topic_modeler import TopicDiscoveryEngine

def test_topic_modeler():
    engine = TopicDiscoveryEngine(n_topics=2, method="nmf")
    docs = ["Technology market stocks soar.", "Sports final championship game ends."]
    engine.fit_topics(docs)
    topics = engine.get_article_topics("Technology stocks")
    assert isinstance(topics, list)
