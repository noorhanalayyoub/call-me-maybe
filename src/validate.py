from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from .models import Calling, Definition


PYTHON_TYPES = {
    "string": str,
    "number": float,
    "integer": int,
    "float": float,
    "boolean": bool,
}


def validate_prompts(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("The input file must contain a JSON list of prompts.")

    prompts = [Calling.model_validate(item) for item in data]
    return [prompt.model_dump() for prompt in prompts]


def validate_definitions(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError(
            "The function definitions file must contain a JSON list."
        )

    definitions = [Definition.model_validate(item) for item in data]
    for definition in definitions:
        for parameter_name, type_spec in definition.parameters.items():
            if type_spec.type not in PYTHON_TYPES:
                raise ValueError(
                    f"Unsupported type {type_spec.type!r} for parameter "
                    f"{parameter_name!r} in function {definition.name!r}."
                )
        if definition.returns.type not in PYTHON_TYPES:
            raise ValueError(
                f"Unsupported return type {definition.returns.type!r} "
                f"in function {definition.name!r}."
            )
    return [definition.model_dump() for definition in definitions]


def build_call_models(
    definitions: list[dict[str, Any]],
) -> dict[str, type[BaseModel]]:
    registry: dict[str, type[BaseModel]] = {}
    for definition in definitions:
        name = definition["name"]
        if name in registry:
            raise ValueError(f"Duplicate function name: {name!r}.")

        parameter_fields: dict[str, Any] = {}
        for parameter_name, type_spec in definition["parameters"].items():
            parameter_fields[parameter_name] = (
                PYTHON_TYPES[type_spec["type"]],
                ...,
            )

        parameter_model = create_model(
            f"{name}_parameters",
            __config__=ConfigDict(extra="forbid", strict=True),
            **parameter_fields,
        )
        call_model = create_model(
            f"{name}_call",
            __config__=ConfigDict(extra="forbid"),
            prompt=(str, ...),
            name=(str, ...),
            parameters=(parameter_model, ...),
        )
        registry[name] = call_model
    return registry


def validate_call(
    data: dict[str, Any],
    registry: dict[str, type[BaseModel]],
) -> dict[str, Any]:
    name = data.get("name")
    if name not in registry:
        raise ValueError(f"Unknown function name: {name!r}.")
    validated = registry[name].model_validate(data)
    return validated.model_dump(mode="json")
