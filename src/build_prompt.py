from .models import FunctionDef, TestCase
from .tokenizer import decode_vocab, encode_prompt, decode_ids, encode_function_names
from llm_sdk import Small_LLM_Model

def build_prompt(functions: list[FunctionDef], user_prompt: str) -> str:
    request_prompt = []
    request_prompt.append("Available functions:\n")
    for fn in functions:
        i = 1
        par_len = len(fn.parameters)
        request_prompt.append(f"- {fn.name}(")
        for name, param in fn.parameters.items():
            request_prompt.append(f"{name}: {param.type}")
            if i != par_len:
                request_prompt.append(f", ")
            i += 1
        request_prompt.append(f"): {fn.description}\n")
    
    request_prompt.append(f"\nUser request: {user_prompt}\n")
    request_prompt.append("\nFunction call:")
    
    request = "".join(request_prompt)
    return request
