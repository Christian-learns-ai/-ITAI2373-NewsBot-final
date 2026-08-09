from src.data_processing.text_preprocessor import TextPreprocessor

def test_text_preprocessor():
    tp = TextPreprocessor()
    text = "  Apple is launching new products in California!  "
    cleaned = tp.clean(text)
    assert cleaned == "Apple is launching new products in California!"
    tokens = tp.tokenize_and_clean(cleaned)
    assert "apple" in tokens
