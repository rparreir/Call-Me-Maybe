from .tokenizer import decode_vocab, encode_prompt, decode_ids
from llm_sdk import Small_LLM_Model

def generator(model: Small_LLM_Model, ids: list[int], max_tokens: int = 50) -> list[int]:
    for i in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        # fuc to define alloawed ids
        next_token = apply_mask(logits, ids)# neds to be allowed ids
        ids.append(next_token)
    return ids
    
    
def apply_mask(logits: list[int], allowed_ids: list[int]) -> int:
    for i in range(len(logits)):
        if i not in allowed_ids:
            logits[i] = float("-inf")
    new_token = max(range(len(logits)), key=lambda i: logits[i])
    return new_token