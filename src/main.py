from config import (
    JDOCTOR_PARAM_10_SR,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)
from dataset import (
    load_dataset,
    validate_dataset,
)
from tqdm import tqdm
from llm.factory import get_provider
from output import save_predictions


def main():

    dataset = load_dataset(JDOCTOR_PARAM_10_SR)

    records = dataset["data"]

    validate_dataset(records)

    print("=" * 60)
    print(f"Quantidade de registros: {len(records)}")
    print("=" * 60)

    first = records[4]

    provider = get_provider()
    predictions = []

    for index, sample in enumerate(tqdm(records), start=1):

        result = provider.generate(
            prompt=sample["prompt"],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        predictions.append(
            {
                "id": index,
                "signature": sample["signature"],
                "ground_truth": sample["condition"],
                "prediction": result["prediction"],
                "match": result["prediction"] == sample["condition"],
                "model": result["model"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "latency_ms": result["latency_ms"],
            }
        )

    save_predictions(predictions)

    provider = get_provider()
    prediction = provider.generate(
        prompt=first["prompt"],
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    print("=" * 60)
    print("GROUND TRUTH")
    print("=" * 60)
    print(first["condition"])

    print()

    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print(prediction["prediction"])

    print()

    print("=" * 60)
    print("MATCH")
    print("=" * 60)
    print(prediction["prediction"] == first["condition"])


if __name__ == "__main__":
    main()
