from .tokenizer import encode_text, decode_ids, encode_function_names
from .models import FunctionDef
from llm_sdk import Small_LLM_Model
import json
from typing import Any, Literal

NUMBER_TYPES = ["number", "integer", "int", "float", "double", "long"]
BOOLEAN_TYPES = ["boolean", "bool"]


def normalize_type(raw_type: str) -> Literal["number", "boolean", "string"]:
    """Map a declared type onto one of the three the generators handle.

    Called once per parameter, so that the skeleton, the value generator
    and the closing quote all read the same answer. Anything unknown
    becomes ``"string"``, which is always representable.

    Args:
        raw_type: The ``type`` field as written in the input file, in any
            capitalisation.

    Returns:
        ``"number"``, ``"boolean"`` or ``"string"``.
    """
    if raw_type.lower() in NUMBER_TYPES:
        return "number"
    if raw_type.lower() in BOOLEAN_TYPES:
        return "boolean"
    return "string"


def generator(model: Small_LLM_Model, ids: list[int],
              functions: list[FunctionDef],
              token_list_by_id: list, prompt: str) -> dict[str, Any]:
    """Produce the function call for one prompt: name and arguments.

    Drives the whole generation: appends the fixed parts of the JSON to
    the context without asking the model, calls ``generate_name`` for the
    function, and then the generator matching each parameter's type.

    Args:
        model: The LLM wrapper.
        ids: The encoded prompt, used as the starting context.
        functions: The available functions.
        token_list_by_id: The id to text map, needed by the string
            generator.
        prompt: The original request, copied into the result.

    Returns:
        A dict with ``prompt``, ``name`` and ``parameters``, ready for
        ``json.dump``. Values are Python types — ``float``, ``str``,
        ``bool`` — not text.
    """
    json_prompt: dict[str, Any] = {}
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

    return json_prompt


def pick_allowed_token(logits: list[float], allowed_ids: list[int]) -> int:
    """Pick the token with the highest logit among the allowed ones.

    Args:
        logits: The model's score for every token in the vocabulary.
        allowed_ids: The ids that keep the output valid at this step.
            Repeated ids are harmless, but the list must not be empty:
            ``max`` raises ``ValueError`` on an empty sequence.

    Returns:
        The id of the chosen token.
    """
    new_token = max(allowed_ids, key=lambda i: logits[i])
    return new_token


def get_token_pos(ative: list[list[int]], pos: int) -> list[int]:
    """Collect the token each still-active candidate has at one position.

    Candidates that have already ended are skipped, since a name can be
    a prefix of another (``fn_add`` and ``fn_add_numbers``).

    Args:
        ative: The candidates still matching what has been generated,
            each one a list of token ids.
        pos: The position being decided.

    Returns:
        One token id per candidate that reaches that position. Repeats
        are expected — all five function names start with ``fn``.
    """
    allowed = []
    for candidate in ative:
        if len(candidate) > pos:
            allowed.append(candidate[pos])
    return allowed


def get_selected_token(ative: list[list[int]], pos: int,
                       next_token: int) -> list[list[int]]:
    """Keep the candidates that match the chosen token.

    Args:
        ative: The candidates still in play.
        pos: The position just decided.
        next_token: The id the model picked for that position.

    Returns:
        The narrowed list of candidates. Candidates shorter than ``pos``
        are dropped, since they have no token there to match.
    """
    chosen = []
    for candidate in ative:
        if len(candidate) > pos and candidate[pos] == next_token:
            chosen.append(candidate)
    return chosen


def has_complete(ative: list[list[int]], pos: int) -> bool:
    """Check whether any candidate ends at this position.

    Args:
        ative: The candidates still in play.
        pos: The position about to be decided, which is also the number
            of tokens already written.

    Returns:
        ``True`` if at least one candidate is fully written, which is
        what makes the closing quote an allowed choice at this step.
    """
    for candidate in ative:
        if len(candidate) == pos:
            return True
    return False


def generate_name(model: Small_LLM_Model, ids: list[int],
                  candidates: list[list[int]]) -> list[list[int]]:
    """Generate the function name one token at a time.

    Only tokens that some candidate still has at the current position
    are allowed, plus the closing quote once a candidate is complete.
    The loop ends when the model picks the quote.

    Args:
        model: The LLM wrapper.
        ids: The context, extended in place with each chosen token.
        candidates: The encoded function names.

    Returns:
        The completed candidate, in a list.
    """
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
            next_token = pick_allowed_token(logits, allowed)
        ids.append(next_token)
        if next_token == quote:
            break
        ative = get_selected_token(ative, pos, next_token)
        pos += 1

    return [c for c in ative if len(c) == pos]


def generate_number(model: Small_LLM_Model, ids: list[int]) -> list[int]:
    """Generate a number one digit at a time, following the JSON grammar.

    The allowed set is rebuilt from what has been written: a sign only in
    first position, one dot and only after a digit, and ``,``/``}`` only
    once a digit exists. So what reaches ``float()`` is always an
    optional sign followed by digits — never ``-`` or ``3.`` alone.

    Args:
        model: The LLM wrapper.
        ids: The context, extended in place with each chosen digit.

    Returns:
        The chosen token ids. The terminator is not included — the loop
        breaks before appending it, and the skeleton writes the
        separator itself.
    """
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

        next_token = pick_allowed_token(logits, allowed_values)
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
                    token_list_by_id: list) -> str:
    """Generate a string value, up to the first unescaped quote.

    Every token is allowed: one containing a quote ends the value, and
    the text before that quote is kept as content. A quote preceded by a
    backslash is escaped, so it counts as content and generation
    continues.

    Args:
        model: The LLM wrapper.
        ids: The context, extended in place.
        token_list_by_id: The id to text map, used to classify tokens.

    Returns:
        The value as text, not as token ids — the last piece can be part
        of a token, which no single id represents.
    """
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
        next_token = pick_allowed_token(logits, allowed)
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


def generate_bool(model: Small_LLM_Model,
                  ids: list[int]) -> list[list[int]]:
    """Generate a boolean value, as ``true`` or ``false``.

    Both are single tokens, so the model makes one choice and the loop
    ends. Lowercase only, which is what JSON accepts.

    Args:
        model: The LLM wrapper.
        ids: The context, extended in place with the chosen token.

    Returns:
        The chosen candidate, in a list.
    """
    ative = [encode_text(model, "true"), encode_text(model, "false")]
    pos = 0

    while True:
        logits = model.get_logits_from_input_ids(ids)
        allowed = get_token_pos(ative, pos)
        next_token = pick_allowed_token(logits, allowed)
        ids.append(next_token)
        ative = get_selected_token(ative, pos, next_token)
        pos += 1

        if len(ative) == 1 and len(ative[0]) == pos:
            break
    return ative
