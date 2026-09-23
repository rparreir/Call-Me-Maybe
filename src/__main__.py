from llm_sdk import Small_LLM_Model
import argparse
import os
import json
from .loader import loader
from .models import FunctionDef, TestCase
from .tokenizer import decode_vocab, encode_prompt, decode_ids
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
    
    token_list_by_id = decode_vocab(model)
    
    final_json = []
    for i in range(len(loaded_input)):
        request = build_prompt(loaded_func_def, loaded_input[i].prompt)
        encode_request = encode_prompt(model, request)
        
        result = generator(model, encode_request, loaded_func_def,
                        token_list_by_id, loaded_input[i].prompt)
        print(result)
        final_json.append(result)
    
    folder_src = os.path.dirname(os.path.abspath(__file__))
    new_folder_path = os.path.join(folder_src, "..", "data", "output")
    final_path = os.path.abspath(new_folder_path)
    os.makedirs(final_path, exist_ok=True)
    
    file_path = os.path.join(final_path, "function_calling_results.json")
    
    with open(file_path, "w", encoding="utf-8") as fuc_call:
        json.dump(final_json, fuc_call, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
