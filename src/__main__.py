from llm_sdk import Small_LLM_Model
import argparse
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
    
    ids = encode_prompt(model, loaded_input[0].prompt)
    generator(model, ids, 20)
    res = build_prompt(loaded_func_def, loaded_input[0].prompt)
    print(res)
    


if __name__ == "__main__":
    main()
