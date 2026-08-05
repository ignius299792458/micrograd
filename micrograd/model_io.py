import json
from pathlib import Path

from micrograd.nn import MLP


def save_model(
    model: MLP,
    filepath: str,
    metadata: dict | None = None,
) -> None:
    payload = {
        "architecture": {
            "nin": model.nin,
            "nouts": model.nouts,
        },
        "parameters": [float(parameter.data) for parameter in model.parameters()],
        "metadata": metadata or {},
    }

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
