import json
import sys
from pydantic import ValidationError, BaseModel


def loader(path: str, func: type[BaseModel]):
    try:
        with open(path, "r", encoding="utf-8") as j_file:
            raw = json.load(j_file)
            return [func(**item) for item in raw]
    except FileNotFoundError:
        print("File doesn't exist!")
        sys.exit(1)
    except json.JSONDecodeError:
        print("JSON File format is broken!")
        sys.exit(1)
    except ValidationError:
        print("Valid JSON File, invalid format!")
        sys.exit(1)
