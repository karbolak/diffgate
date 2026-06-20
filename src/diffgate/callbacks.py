"""Backward-compatible callback exports.

New code should import recorders from :mod:`diffgate.recorders`.
"""

from .recorders import AbortGeneration, LatentRecorder, RichSD35Recorder

TrajectoryRecorder = LatentRecorder

__all__ = ["AbortGeneration", "LatentRecorder", "RichSD35Recorder", "TrajectoryRecorder"]
