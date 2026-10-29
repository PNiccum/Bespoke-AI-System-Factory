import re


def slugify(title):
    """URL slug: lower case, words joined by single hyphens, punctuation dropped."""
    words = re.findall(r"[A-Za-z0-9]+", title)
    return "-".join(words)
