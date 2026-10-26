import argparse
import json


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

        def check_for_valid_json(self, file_path):
            try:
                with open(file_path, encoding="utf-8") as fd:
                    json.load(fd)
            except (json.JSONDecodeError, OSError) as e:
                print(f"{file_path}: {e}")
