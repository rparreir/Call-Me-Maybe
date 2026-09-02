from pydantic import BaseModel


class TestCase(BaseModel):
    """A single natural-language prompt to process."""
    prompt: str


class Parameter(BaseModel):
    """A typed parameter or return value (e.g. type='number')."""
    type: str


class FunctionDef(BaseModel):
    """A callable function: name, description, typed parameters and return."""
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
