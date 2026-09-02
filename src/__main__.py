import argparse
from .loader import loader
from .models import FunctionDef, TestCase


def main():
    parse = argparse.ArgumentParser(description="...")

    parse.add_argument("--functions_definition",
                       default="data/input/functions_definition.json")
    parse.add_argument("--input",
                       default="data/input/function_calling_tests.json")
    parse.add_argument("--output",
                       default="data/output/function_calling_results.json")

    args = parse.parse_args()

    loaded_func_def = loader(args.functions_definition, FunctionDef)
    loaded_input = loader(args.input, TestCase)
    print(loaded_func_def)
    print()
    print(loaded_input)


if __name__ == "__main__":
    main()
