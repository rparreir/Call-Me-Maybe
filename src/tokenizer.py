"""Conversions between text and token ids, and the vocabulary map."""

import json
from llm_sdk import Small_LLM_Model
from .models import FunctionDef


def decode_vocab(model: Small_LLM_Model) -> list:
    """Build the text of every token in the vocabulary, indexed by id.

    Decoding all 151 643 tokens takes about a second, so this is done
    once at startup and the result is passed around, instead of
    decoding a token every time its text is needed.

    Args:
        model: The LLM wrapper; its vocabulary file and its decoder
            are used.

    Returns:
        A list where position ``i`` holds the text of token id ``i``.
    """
    path = model.get_path_to_vocab_file()
    token_list_by_id = []
    with open(path, "r", encoding="utf-8") as p:
        vocab = json.load(p)
        for i in range(len(vocab)):
            token_list_by_id.append(model.decode([i]))
    return (token_list_by_id)


def encode_text(model: Small_LLM_Model, text: str) -> list[int]:
    """Turn text into the token ids the model reads.

    Args:
        model: The LLM wrapper; only its tokenizer is used, not the weights.
        text: The text to encode — a whole prompt or a fragment of the
            JSON skeleton.

    Returns:
        The token ids, as a flat list of ints. The SDK returns a matrix
        with a single row.
    """
    ids: list[int] = model.encode(text)[0].tolist()
    return (ids)


def decode_ids(model: Small_LLM_Model, ids: list[int] | int) -> str:
    """Turn token ids back into text.

    Args:
        model: The LLM wrapper; only its decoder is used.
        ids: The ids to decode — a list, or a single id, since the
            generators inspect one chosen token at a time.

    Returns:
        The text of those ids, concatenated into a single string. A list
        of digit tokens comes back as ``"265"``, not as three pieces,
        which is what lets the caller pass it straight to ``float()``.
    """
    text: str = model.decode(ids)
    return text


def encode_function_names(model: Small_LLM_Model,
                          functions: list[FunctionDef]) -> list[list[int]]:
    """Encode each function name into its token ids.

    These are the candidates the name mask filters: at each position
    only the tokens that still match an active candidate are allowed,
    which is what forces the generated name to be one that exists.

    Args:
        model: The LLM wrapper; only its tokenizer is used.
        functions: The functions defined in the input file.

    Returns:
        One list of token ids per function, in the same order. The rows
        have different lengths, since names tokenize into 2 to 7 tokens.
    """
    list_allowed_names = []
    for fn in functions:
        list_allowed_names.append(encode_text(model, fn.name))
    return list_allowed_names
