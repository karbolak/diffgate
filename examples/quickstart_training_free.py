"""Quickstart for training-free DiffGate generation.

This example requires a CUDA machine with SD3.5 dependencies installed:

    pip install -e ".[sd35]"
"""

from diffgate import DiffGateSD35


gate = DiffGateSD35.from_pretrained(
    record_mode="rich",
    scorer="training_free",
    threshold=25,
    device="cuda",
    torch_dtype="float16",
)

result = gate.generate(
    prompt="A red car on a snowy road",
    seed=12345,
    abort=True,
    return_trajectory=True,
)

print(result.to_dict(include_features=False))
if result.image is not None:
    result.image.save("diffgate_example.png")
