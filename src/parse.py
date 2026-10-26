import argparse
import json
from pathlib import Path
from typing import Any
from .validate import validate_definitions, validate_prompts


def load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fd:
            return json.load(fd)
    except FileNotFoundError as exc:
        raise ValueError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


class Parse:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json",
        )
        self.parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json",
        )
        self.parser.add_argument(
            "--output",
            default="data/output/function_calling_results.json",
        )
        self.args = self.parser.parse_args()

    def open_validate(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prompts = load_json(self.args.input)
        definitions = load_json(self.args.functions_definition)
        return validate_prompts(prompts), validate_definitions(definitions)

    def write_output(self, results: list[dict[str, Any]]) -> None:
        output = Path(self.args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as fd:
            json.dump(results, fd, indent=2)
