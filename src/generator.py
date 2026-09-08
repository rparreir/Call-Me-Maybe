from .tokenizer import decode_vocab, encode_prompt, decode_ids
from llm_sdk import Small_LLM_Model

def generator(model: Small_LLM_Model, ids: list[int], max_tokens: int = 50) -> list[int]:
    print(decode_ids(model, ids))
    gerado = []
    for i in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        next_token = max(range(len(logits)), key=lambda i: logits[i])
        gerado.append(decode_ids(model, next_token))
        #print(f"next_token id = {next_token} value = {decode_ids(model, next_token)}")
        ids.append(next_token)
    gerado_final = "".join(gerado)
    print(gerado_final)
    # token_list_by_id = decode_vocab(model)