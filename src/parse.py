import argparse
import json
from pathlib import Path
from typing import Any


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

    def open_validate(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with open(self.args.input, "r") as fd1:
            prompts = json.load(fd1)
        with open(self.args.functions_definition, "r") as fd2:
            definitions = json.load(fd2)
        return prompts, definitions

    def write_output(self, results: list[dict[str, Any]]) -> None:
        output = Path(self.args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as fd:
            json.dump(results, fd, indent=2)
