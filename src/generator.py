from .tokenizer import decode_vocab, encode_prompt, decode_ids, encode_function_names
from .models import FunctionDef, TestCase
from llm_sdk import Small_LLM_Model

def generator(model: Small_LLM_Model, ids: list[int], functions: list[FunctionDef], max_tokens: int = 20) -> list[int]:
    json_prompt = {}
    
    candidates = encode_function_names(model, functions)
    name_json = encode_prompt(model, ' {"name": "')
    ids = ids + name_json
    chosen_name = generate_name(model, ids, candidates)
    json_prompt["name"] = decode_ids(model, chosen_name[0])
    parameters_json = encode_prompt(model, '", "parameters": ')
    ids = ids + parameters_json
    

    for fn in functions:
        if fn.name == decode_ids(model, chosen_name[0]):
            for key, param in fn.parameters.items():
                
                ids = ids + encode_prompt(model, '{"')
                key_e = encode_prompt(model, key)
                ids = ids + key_e
                ids = ids + encode_prompt(model, '": "')
                # aqui fica o tokem
                ids = ids + encode_prompt(model, '"}')
                
        json_prompt["key"] = "placeholder"
    
    
    #print(decode_ids(model, ids))
    #print(decode_ids(model, result))
    print(decode_ids(model, ids))
    return json_prompt


def apply_mask(logits: list[int], allowed_ids: list[int]) -> int:
    for i in range(len(logits)):
        if i not in allowed_ids:
            logits[i] = float("-inf")
    new_token = max(range(len(logits)), key=lambda i: logits[i])
    return new_token

def get_token_pos(ative: list[int], pos: int):
    allowed = []
    for candidate in ative:
        allowed.append(candidate[pos])
    return allowed

def get_selected_token(ative: list[int], pos: int, next_token: int):
    chosen = []
    for candidate in ative:
        if candidate[pos] == next_token:
            chosen.append(candidate)
    return chosen


def generate_name(model: Small_LLM_Model, ids: list[int], candidates: list[int]):
    ative = candidates
    pos = 0
    
    while True:
        logits = model.get_logits_from_input_ids(ids)
        allowed = get_token_pos(ative, pos)
        next_token = apply_mask(logits, allowed)
        ids.append(next_token)
        ative = get_selected_token(ative, pos, next_token)
        pos += 1

        if len(ative) == 1 and len(ative[0]) == pos:
            break
    return ative


def generate_parameter(model: Small_LLM_Model, ids: list[int], candidates: list[int]):
    pass