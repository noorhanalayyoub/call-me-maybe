import argparse
import json
from pathlib import Path


class parse:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # creating instance of class argumentparser and then calling its method parse_args
        self.parser.add_argument("--functions_definition", default='../data/input/functions_definition.json')
        self.parser.add_argument("--input", default='../data/input/functions_calls.json')
        self.parser.add_argument("--output", default='../data/output/functions_calls.json')
        self.args = self.parser.parse_args()
        output = Path(self.args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

    def open_files(self):
        try:
            with open(self.args.input, 'r') as fd1:
                json.load(fd1)
            with open(self.args.functions_definition, 'r') as fd2:
                json.load(fd2)
        except Exception as e:
            print(f"{e}")
