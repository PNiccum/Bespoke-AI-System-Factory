"""Most frequent words in a text file, without common English stopwords."""

import argparse
import json
import re
import sys
from collections import Counter

STOPWORDS = frozenset("""a an and are as at be but by for from had has have he her his i in is it its of on or
she that the their them they this to was were will with you your not we our""".split())
WORD = re.compile(r"[a-z']+")


def top_words(text, n=10, stopwords=None):
    stop = STOPWORDS if stopwords is None else stopwords
    words = (w.strip("'") for w in WORD.findall(text.lower()))
    counts = Counter(w for w in words if w and w not in stop)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--keep-stopwords", action="store_true")
    args = ap.parse_args(argv)
    try:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    rows = top_words(text, args.top, set() if args.keep_stopwords else None)
    if args.json:
        print(json.dumps([{"word": w, "count": c} for w, c in rows]))
    else:
        for w, c in rows:
            print(f"{w}\t{c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
