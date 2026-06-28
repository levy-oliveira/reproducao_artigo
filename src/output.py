import json
from pathlib import Path


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_predictions(predictions, filename="predictions.json"):
    output_file = OUTPUT_DIR / filename

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            predictions,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nResultados salvos em: {output_file}")