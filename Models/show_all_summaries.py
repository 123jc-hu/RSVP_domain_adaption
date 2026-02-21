from Models.model_registry import model_dict
from Models.summary_utils import summarize_model


def main() -> None:
    registry = model_dict()
    for model_name, entry in registry.items():
        summarize_model(model_name, entry.Model)


if __name__ == "__main__":
    main()

