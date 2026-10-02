from llm_sdk import Small_LLM_Model
import argparse
import os
import json
import sys
from .loader import loader
from .models import FunctionDef, TestCase
from .tokenizer import decode_vocab, encode_text
from . generator import generator
from .build_prompt import build_prompt


def main():
    model = Small_LLM_Model()
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
    main()
