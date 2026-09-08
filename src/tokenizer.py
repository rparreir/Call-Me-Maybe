import json
from llm_sdk import Small_LLM_Model


def decode_vocab(model: Small_LLM_Model) -> list:
    path = model.get_path_to_vocab_file()
    token_list_by_id = []
    with open(path, "r", encoding="utf-8") as p:
        vocab = json.load(p)
        for i in range(len(vocab)):
            # print(repr(chave), "->", valor)
            token_list_by_id.append(model.decode([i]))
    return (token_list_by_id)


def encode_prompt(model: Small_LLM_Model, prompt: str) -> list[int]:
    ids = model.encode(prompt)[0].tolist()
    return (ids)


def decode_ids(model: Small_LLM_Model, ids: list[int]) -> str:
    prompt = model.decode(ids)
    return prompt