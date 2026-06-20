"""DiffGate: early trajectory assessment for diffusion image generation."""

from .abort import ThresholdAbortPolicy
from .config import DiffGateConfig, EarlyAbortConfig
from .features import PrefixFeatureExtractor
from .health import TrainingFreeHealthScorer
from .pipeline import DiffGateSD35, EarlyAbortSD35
from .recorders import LatentRecorder, RichSD35Recorder
from .results import GenerationResult
from .supervised import SupervisedHealthScorer

__version__ = "0.1.0"

__all__ = [
    "DiffGateConfig",
    "DiffGateSD35",
    "EarlyAbortConfig",
    "EarlyAbortSD35",
    "GenerationResult",
    "LatentRecorder",
    "PrefixFeatureExtractor",
    "RichSD35Recorder",
    "SupervisedHealthScorer",
    "ThresholdAbortPolicy",
    "TrainingFreeHealthScorer",
]
