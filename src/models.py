from typing import Dict
from pydantic import BaseModel, ConfigDict, create_model
import json
import argparse

class Calling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


class Type(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class Definition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, Type]
    returns: Type


def load_def(path):
    with open(path, encoding="utf-8") as file:
        defs = json.load(file)
    if not isinstance(defs, list):
        raise ValueError(f"{path} must contain a JSON list")

    return [Definition.model_validate(item) for item in defs]


def load_prompts(path):
    with open(path, encoding="utf-8") as fd:
        prompts = json.load(fd)
    if not isinstance(prompts, list):
        raise ValueError(f"{path} must contain a JSON list")

    return [Calling.model_validate(item) for item in prompts]


TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def build_model(definition: Definition):
    params = {}

    for name, parameter in definition.parameters.items():
        python_type = TYPES[parameter.type]

        params[name] = (python_type, ...)

    return create_model(f"{definition.name.title()}Arguments",__config__=ConfigDict(extra="forbid"), **params,)

def check_generated_call(call: dict, argument_models) -> None:
    name = call.get("name")

    if name not in argument_models:
        raise ValueError(f"Unknown function: {name!r}")

    argument_models[name].model_validate(call.get("arguments", {}))

