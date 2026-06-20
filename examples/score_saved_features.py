"""Score saved DiffGate features without running generation."""

import json
from pathlib import Path

from diffgate import TrainingFreeHealthScorer


features = json.loads(Path("outputs/example/features.json").read_text())
score = TrainingFreeHealthScorer().score(features)
print({"health_score": score})
