"""Acceptance tests run from the top of the delivered output/ folder; make its modules importable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
