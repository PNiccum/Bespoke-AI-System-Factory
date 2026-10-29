from text.slug import slugify


def test_lower_case():
    assert slugify("Hello World") == "hello-world"


def test_punctuation():
    assert slugify("a, b!") == "a-b"
