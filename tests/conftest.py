import sys
from pathlib import Path

# make the repo root importable (agentic_rag, single_shot_rag, rag_eval)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
