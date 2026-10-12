import json
import sys
from pydantic import ValidationError, BaseModel
from typing import TypeVar

ModelT = TypeVar("ModelT", bound=BaseModel)


def loader(path: str, func: type[ModelT]) -> list[ModelT]:
    """Load a JSON array from a file and validate every item.

    Used for both input files: the function definitions and the test
    prompts, with the matching model passed in ``func``.

    On a missing or unreadable file, broken JSON, a root that is not an
    array of objects, or an item that does not match the model, it prints
    the reason and exits with code 1.

    Args:
        path: Path to a JSON file holding an array of objects.
        func: The pydantic model class each item must match. Building it
            validates the item, so a bad field raises ``ValidationError``.

    Returns:
        One model instance per item, in file order. Callers reach the
        fields as attributes, not as dictionary keys.
    """
    try:
        with open(path, "r", encoding="utf-8") as j_file:
            raw = json.load(j_file)
    except FileNotFoundError:
        print(f"File doesn't exist: {path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"JSON File format is broken: {path}")
        sys.exit(1)
    except OSError as error:
        print(f"Could not read {path}: {error}")
        sys.exit(1)

    if not isinstance(raw, list):
        print(f"{path} must contain a JSON array, not a {type(raw).__name__}")
        sys.exit(1)
    for item in raw:
        if not isinstance(item, dict):
            print(f"{path} must contain a JSON array of objects")
            sys.exit(1)

    try:
        return [func(**item) for item in raw]
    except ValidationError:
        print(f"Valid JSON File, invalid format: {path}")
        sys.exit(1)
