from .models import FunctionDef, TestCase


def build_prompt(functions: list[FunctionDef], user_prompt: str) -> str:
    request_prompt = []
    request_prompt.append("Available functions:\n")
    
    for fn in functions:
        
        request_prompt.append(f"- {fn.name}({fn.parameters}): {fn.description}\n")
    
    request_prompt.append(f"\nUser request: {user_prompt}\n")
    request_prompt.append("\nFunction call:") # still needs work
    
    request = "".join(request_prompt)
    return request