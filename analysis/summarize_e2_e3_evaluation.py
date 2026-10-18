"""Summarize saved E2/E3 panels with the same arithmetic as development."""
from pathlib import Path
import runpy
import sys
sys.argv += ["--stage", "evaluation"]
runpy.run_path(str(Path(__file__).with_name("summarize_e2_e3_development.py")), run_name="__main__")
