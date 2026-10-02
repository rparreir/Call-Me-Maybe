import json
import sys
from pydantic import ValidationError, BaseModel


def loader(path: str, func: type[BaseModel]):
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
