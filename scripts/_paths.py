"""
Helper module, inserts the src directory to the PYTHONPATH variable.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_dir / "src"))
