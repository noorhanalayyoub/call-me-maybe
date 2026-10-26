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

        return create_model(f"{definition.name.title()}Arguments",__config__=ConfigDict(extra="forbid"), **fields,)

def check_generated_call(call: dict, definitions: list[Definition]) -> None:
    definition_by_name = {definition.name: definition for definition in definitions}
    name = call.get("name")

    if name not in definition_by_name:
        raise ValueError(f"Unknown function: {name!r}")

    argument_model = argument_models[name]
    argument_model.model_validate(call.get("arguments", {}))



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    args = parser.parse_args()

    definitions = load_def(args.functions_definition)
    prompts = load_prompts(args.input)

    print(f"Loaded {len(definitions)} function definitions:")
    for definition in definitions:
        print(f"- {definition.name}")

    print(f"\nLoaded {len(prompts)} prompts:")
    for index, item in enumerate(prompts, start=1):
        print(f"{index}. {item.prompt}")
main()
