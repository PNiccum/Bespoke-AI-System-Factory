"""A ranked, cited reading list on a topic, from the OpenAlex works API (or a saved response)."""

import argparse
import json
import sys
import urllib.parse
import urllib.request

API = "https://api.openalex.org/works"


def fetch(topic, n=25, mailto=None):
    params = {"search": topic, "per-page": n} | ({"mailto": mailto} if mailto else {})
    with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}", timeout=30) as r:  # noqa: S310
        return json.load(r)["results"]


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    at = {pos: word for word, positions in inverted_index.items() for pos in positions}
    return " ".join(at[i] for i in sorted(at))


def rank(works, top=10, since_year=None):
    kept = [w for w in works if w.get("title") and (since_year is None or (w.get("publication_year") or 0) >= since_year)]
    kept.sort(key=lambda w: (-(w.get("cited_by_count") or 0), -(w.get("publication_year") or 0), w["title"]))
    return kept[:top]


def _authors(w):
    names = [a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author", {}).get("display_name")]
    return ", ".join(names[:3]) + (" et al." if len(names) > 3 else "") if names else "Unknown authors"


def render(works, topic):
    lines = [f"# Reading list: {topic}", ""]
    for i, w in enumerate(works, 1):
        doi = w.get("doi")
        link = f" [{doi}]({doi})" if doi else ""
        lines.append(f"{i}. **{w['title']}** ({w.get('publication_year', 'n.d.')}). {_authors(w)}. "
                     f"Cited by {w.get('cited_by_count', 0)}.{link}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("topic")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--since", type=int)
    ap.add_argument("--from-file")
    ap.add_argument("-o", "--out")
    args = ap.parse_args(argv)
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as f:
            works = json.load(f)["results"]
    else:
        works = fetch(args.topic)
    text = render(rank(works, args.top, args.since), args.topic)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
