from diffgate import DiffGateSD35, PrefixFeatureExtractor, TrainingFreeHealthScorer
from diffgate.cli import build_parser


def test_imports():
    assert DiffGateSD35 is not None
    assert PrefixFeatureExtractor is not None
    assert TrainingFreeHealthScorer is not None


def test_cli_parser_generate():
    parser = build_parser()
    args = parser.parse_args(["generate", "--prompt", "hello"])
    assert args.command == "generate"
    assert args.record_mode == "latent"
