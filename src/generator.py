from .tokenizer import decode_vocab, encode_prompt, decode_ids, encode_function_names
from .models import FunctionDef, TestCase
from llm_sdk import Small_LLM_Model

def generator(model: Small_LLM_Model, ids: list[int],
              functions: list[FunctionDef],
              token_list_by_id: list, max_tokens: int = 20) -> list[int]:
    json_prompt = {}
    
    candidates = encode_function_names(model, functions)
    name_json = encode_prompt(model, ' {"name": "')
    ids = ids + name_json
    chosen_name = generate_name(model, ids, candidates)
    json_prompt["name"] = decode_ids(model, chosen_name[0])
    parameters_json = encode_prompt(model, '", "parameters": {')
    ids = ids + parameters_json
    json_prompt["parameters"] = {}
    
    for fn in functions:
        if fn.name == decode_ids(model, chosen_name[0]):
            for i, (key, param) in enumerate(fn.parameters.items()):
                if param.type == "string":
                    ids = ids + encode_prompt(model, f'"{key}": "')
                else:
                    ids = ids + encode_prompt(model, f'"{key}": ')

                json_prompt["parameters"][key] = None
                if param.type == "number":
                    chosen_number = generate_number(model, ids)
                    json_prompt["parameters"][key] = float(decode_ids(model, chosen_number))
                if param.type == "string":
                    chosen_string = generate_string(model, ids, token_list_by_id)
                    json_prompt["parameters"][key] = decode_ids(model, chosen_string)
                if param.type == "boolean" or param.type == "bool":
                    chosen_bool = generate_bool(model, ids)# working here
                    json_prompt["parameters"][key] = decode_ids(model, chosen_bool[0]) == "true"
                if param.type == "string":
                    ids = ids + encode_prompt(model, '"')
                if i < len(fn.parameters) - 1:
                    ids = ids + encode_prompt(model, ', ')
                
    ids = ids + encode_prompt(model, '}}')
    
    print(json_prompt)
    #print(decode_ids(model, ids))
    #print(decode_ids(model, result))
    print(decode_ids(model, ids))
    return json_prompt


def apply_mask(logits: list[int], allowed_ids: list[int]) -> int:
    new_token = max(allowed_ids, key=lambda i: logits[i])
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


def generate_number(model: Small_LLM_Model, ids: list[int]):
    allowed_values = []
    return_values = []
    for i in range(0, 10):
        allowed_values.append(encode_prompt(model, str(i))[0])

    for i in range(20):
        logits = model.get_logits_from_input_ids(ids)
        if return_values and i == 1:
            allowed_values.append(encode_prompt(model, ",")[0])
            allowed_values.append(encode_prompt(model, "}")[0])
        next_token = apply_mask(logits, allowed_values)
        if decode_ids(model, next_token) == "," or decode_ids(model, next_token) == "}":
            break
        ids.append(next_token)
        return_values.append(next_token)
    return return_values

def generate_string(model: Small_LLM_Model, ids: list[int], token_list_by_id: list):
    allowed = []
    cont = []
    final = []
    return_values = []

    for i, value in enumerate(token_list_by_id):
        if not '"' in value:
            cont.append(i)
        elif value.startswith('"'):
            final.append(i)

    allowed = cont + final
    for i in range(20):
        logits = model.get_logits_from_input_ids(ids)
        next_token = apply_mask(logits, allowed)
        if decode_ids(model, next_token).startswith('"'):
            break
        ids.append(next_token)
        return_values.append(next_token)
    return return_values

def generate_bool(model: Small_LLM_Model, ids: list[int]):
    ative = [encode_prompt(model, "true"), encode_prompt(model, "false")]
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