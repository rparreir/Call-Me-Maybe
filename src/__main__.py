import sys

try:
    from llm_sdk import Small_LLM_Model
    import argparse
    import os
    import json
    from .loader import loader
    from .models import FunctionDef, TestCase
    from .tokenizer import decode_vocab, encode_text
    from . generator import generator
    from .build_prompt import build_prompt
except KeyboardInterrupt:
    print("\nExited")
    sys.exit(1)


def main() -> None:
    """Run the whole pipeline once.

    Loads the model and the two input files, builds a prompt for each
    test case, generates its function call, and writes every result to
    the output file as a single JSON array.
    """
    model = Small_LLM_Model()
    parse = argparse.ArgumentParser(description="Turn natural language prompts"
                                    " into JSON function calls.")

    parse.add_argument("--functions_definition",
                       default="data/input/functions_definition.json",
                       help="JSON file with the available functions")
    parse.add_argument("--input",
                       default="data/input/function_calling_tests.json",
                       help="JSON file with the prompts to process")
    parse.add_argument("--output",
                       default="data/output/function_calling_results.json",
                       help="where to write the results")

    args = parse.parse_args()

    loaded_func_def = loader(args.functions_definition, FunctionDef)
    loaded_input = loader(args.input, TestCase)
    if not loaded_func_def:
        print(f"No functions defined in {args.functions_definition}")
        sys.exit(1)
    token_list_by_id = decode_vocab(model)

    final_json = []
    for i in range(len(loaded_input)):
        request = build_prompt(loaded_func_def, loaded_input[i].prompt)
        encode_request = encode_text(model, request)

        result = generator(model, encode_request, loaded_func_def,
                           token_list_by_id, loaded_input[i].prompt)
        final_json.append(result)

    try:
        output_folder = os.path.dirname(args.output)
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fuc_call:
            json.dump(final_json, fuc_call, ensure_ascii=False, indent=4)
    except OSError as error:
        print(f"Could not write {args.output}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited")
        sys.exit(1)
