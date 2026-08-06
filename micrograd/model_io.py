import json
from pathlib import Path

from micrograd.nn import MLP


def save_model(
    nin: int,
    nouts_arr: list[int],
    model: MLP,
    filepath: str,
    metadata: dict | None = None,
) -> None:
    payload = {
        "architecture": {
            "nin": nin,
            "nouts_arr": nouts_arr,
        },
        "metadata": metadata,
        "parameters": [float(parameter.data) for parameter in model.parameters()],
    }

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_model(filepath: str) -> tuple[MLP, dict]:
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    architecture = payload["architecture"]

    nin = int(architecture["nin"])
    nouts_arr = [int(nout) for nout in architecture["nouts_arr"]]

    # Recreate the same model architecture.
    model = MLP(nin, nouts_arr)

    model_parameters = model.parameters()
    saved_parameters = payload["parameters"]

    if len(model_parameters) != len(saved_parameters):
        raise ValueError(
            "Parameter count mismatch: "
            f"model expects {len(model_parameters)}, "
            f"but file contains {len(saved_parameters)}."
        )

    # Copy saved weights and biases into the new model.
    for parameter, saved_value in zip(
        model_parameters,
        saved_parameters,
    ):
        parameter.data = float(saved_value)
        parameter.grad = 0.0

    metadata = payload.get("metadata") or {}

    return model, metadata
