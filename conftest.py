"""Pytest configuration ensuring src is in python path."""
import sys
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
