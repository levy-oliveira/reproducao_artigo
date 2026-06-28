import json
from pathlib import Path


def load_dataset(file_path: Path):
    """
    Carrega um dataset JSON.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    list
        Lista contendo todos os registros do dataset.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


REQUIRED_FIELDS = {
    "prompt",
    "condition",
    "signature",
    "tag",
}


def validate_dataset(dataset):

    for i, item in enumerate(dataset):

        missing = REQUIRED_FIELDS - item.keys()

        if missing:
            raise ValueError(
                f"Registro {i} está sem os campos {missing}"
            )

    print("Dataset validado com sucesso.")
