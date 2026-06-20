"""Quickstart for supervised DiffGate generation.

Requires exported predictor artifacts in models/.
"""

from diffgate import DiffGateSD35


gate = DiffGateSD35.from_pretrained(
    record_mode="rich",
    scorer="supervised",
    predictor_path="models/hpsv2_prefix5_logreg.joblib",
    scaler_path="models/hpsv2_prefix5_scaler.joblib",
    feature_schema_path="models/hpsv2_prefix5_schema.json",
    threshold=25,
    device="cuda",
    torch_dtype="float16",
)

result = gate.generate(
    prompt="A red car on a snowy road",
    seed=12345,
    abort=True,
)

print(result.to_dict(include_features=False))
