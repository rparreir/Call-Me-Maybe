from .tokenizer import encode_text, decode_ids, encode_function_names
from .models import FunctionDef
from llm_sdk import Small_LLM_Model
import json
from typing import Any

NUMBER_TYPES = ["number", "integer", "int", "float", "double", "long"]
BOOLEAN_TYPES = ["boolean", "bool"]


def normalize_type(raw_type: str) -> str:
    if raw_type.lower() in NUMBER_TYPES:
        return "number"
    if raw_type.lower() in BOOLEAN_TYPES:
        return "boolean"
    return "string"


def generator(model: Small_LLM_Model, ids: list[int],
              functions: list[FunctionDef],
              token_list_by_id: list, prompt) -> dict[str, Any]:
    json_prompt = {}
    json_prompt['prompt'] = prompt

    candidates = encode_function_names(model, functions)
    name_json = encode_text(model, ' {"name": "')
    ids = ids + name_json
    chosen_name = generate_name(model, ids, candidates)
    json_prompt["name"] = decode_ids(model, chosen_name[0])
    parameters_json = encode_text(model, ', "parameters": {')
    ids = ids + parameters_json
    json_prompt["parameters"] = {}
    parameters = json_prompt["parameters"]

    for fn in functions:
        if fn.name == decode_ids(model, chosen_name[0]):
            for i, (key, param) in enumerate(fn.parameters.items()):
                param_type = normalize_type(param.type)
                if param_type == "string":
                    ids = ids + encode_text(model, f'"{key}": "')
                else:
                    ids = ids + encode_text(model, f'"{key}": ')

                parameters[key] = None
                if param_type == "number":
                    chosen_number = generate_number(model, ids)
                    parameters[key] = float(decode_ids(model, chosen_number))
                if param_type == "string":
                    text = generate_string(model, ids, token_list_by_id)
                    try:
                        parameters[key] = json.loads('"' + text + '"').strip()
                    except json.JSONDecodeError:
                        parameters[key] = text.strip()
                if param_type == "boolean":
                    chosen_bool = generate_bool(model, ids)
                    bool_text = decode_ids(model, chosen_bool[0])
                    parameters[key] = bool_text == "true"
                if param_type == "string":
                    ids = ids + encode_text(model, '"')
                if i < len(fn.parameters) - 1:
                    ids = ids + encode_text(model, ', ')

    ids = ids + encode_text(model, '}}')

    return json_prompt


def apply_mask(logits: list[int], allowed_ids: list[int]) -> int:
    new_token = max(allowed_ids, key=lambda i: logits[i])
    return new_token


def get_token_pos(ative: list[list[int]], pos: int) -> list[int]:
    allowed = []
    for candidate in ative:
        if len(candidate) > pos:
            allowed.append(candidate[pos])
    return allowed


def get_selected_token(ative: list[list[int]], pos: int,
                       next_token: int) -> list[list[int]]:
    chosen = []
    for candidate in ative:
        if len(candidate) > pos and candidate[pos] == next_token:
            chosen.append(candidate)
    return chosen


def has_complete(ative: list[list[int]], pos: int) -> bool:
    for candidate in ative:
        if len(candidate) == pos:
            return True
    return False


def generate_name(model: Small_LLM_Model, ids: list[int],
                  candidates: list[list[int]]) -> list[list[int]]:
    quote = encode_text(model, '"')[0]
    ative = candidates
    pos = 0

    while True:
        allowed = get_token_pos(ative, pos)
        if has_complete(ative, pos):
            allowed = allowed + [quote]
        if len(set(allowed)) == 1:
            next_token = allowed[0]
        else:
            logits = model.get_logits_from_input_ids(ids)
            next_token = apply_mask(logits, allowed)
        ids.append(next_token)
        if next_token == quote:
            break
        ative = get_selected_token(ative, pos, next_token)
        pos += 1

    return [c for c in ative if len(c) == pos]


def generate_number(model: Small_LLM_Model, ids: list[int]):
    signs = [encode_text(model, " -")[0], encode_text(model, "-")[0]]
    digits = [encode_text(model, str(d))[0] for d in range(10)]
    dot = encode_text(model, ".")[0]
    end = [encode_text(model, ",")[0], encode_text(model, "}")[0]]

    allowed_values = signs + digits
    return_values: list[int] = []

    for i in range(20):
        logits = model.get_logits_from_input_ids(ids)

        if return_values and return_values[-1] in digits:
            if end[0] not in allowed_values:
                allowed_values = allowed_values + end
            if dot not in allowed_values and dot not in return_values:
                allowed_values = allowed_values + [dot]

        next_token = apply_mask(logits, allowed_values)
        if next_token == dot:
            allowed_values = [t for t in allowed_values
                              if t != dot and t not in end]
        if i == 0:
            allowed_values = [t for t in allowed_values if t not in signs]
        if next_token in end:
            break
        ids.append(next_token)
        return_values.append(next_token)

    return return_values


def generate_string(model: Small_LLM_Model, ids: list[int],
                    token_list_by_id: list):
    cont = []
    final = []
    text = ""

    for i, value in enumerate(token_list_by_id):
        if '"' in value:
            final.append(i)
        else:
            cont.append(i)

    allowed = cont + final

    for i in range(20):
        logits = model.get_logits_from_input_ids(ids)
        next_token = apply_mask(logits, allowed)
        value = decode_ids(model, next_token)
        if '"' in value:
            head = value.split('"')[0]
            if head.endswith("\\"):
                ids.append(next_token)
                text = text + value
                continue
            if head:
                ids.extend(encode_text(model, head))
                text = text + head
            break
        ids.append(next_token)
        text = text + value
    return text


def generate_bool(model: Small_LLM_Model, ids: list[int]):
    ative = [encode_text(model, "true"), encode_text(model, "false")]
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
