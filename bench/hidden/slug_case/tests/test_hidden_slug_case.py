from text.slug import slugify


def test_other_titles():
    assert slugify("Python 3 Rocks!") == "python-3-rocks"
    assert slugify("NASA Rocks") == "nasa-rocks"
    assert slugify("ALL CAPS") == "all-caps"
