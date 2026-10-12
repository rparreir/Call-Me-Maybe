from .models import FunctionDef


def build_prompt(functions: list[FunctionDef], user_prompt: str) -> str:
    """Build the prompt the model reads for one request.

    Uses Qwen's own chat template (``<|im_start|>system / user /
    assistant``), the format the model was trained on, so it follows the
    instruction better than plain text would.

    The prompt only helps: it guarantees nothing. Which function and
    which values come out is still the model's decision, and the shape of
    the output is enforced by the masks, not here.

    Args:
        functions: The available functions, listed in the system message
            as ``- name(param: type, ...): description``.
        user_prompt: The natural language request, placed in the user
            message.

    Returns:
        The full prompt, ending at ``<|im_start|>assistant`` so that the
        model continues from there.
    """
    request_prompt = []
    request_prompt.append("<|im_start|>system\n")

    request_prompt.append("You are provided with these function signatures:\n")
    for fn in functions:
        request_prompt.append(f"- {fn.name}(")
        name_parameter = []
        for name, param in fn.parameters.items():
            name_parameter.append(f"{name}: {param.type}")
        request_prompt.append(", ".join(name_parameter))
        request_prompt.append(f"): {fn.description}\n")
    request_prompt.append(
        "For each function call, return a json object: "
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
    )
    request_prompt.append("<|im_end|>\n")
    request_prompt.append(f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n")
    request_prompt.append("<|im_start|>assistant\n")

    request = "".join(request_prompt)
    return request
