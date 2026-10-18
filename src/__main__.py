import sys

from .answer import Answer
from .parse_and_validate import parse


def main() -> int:
    try:
        arg_parser = parse()
        prompts, definitions = arg_parser.open_validate()
        results = Answer().run(prompts, definitions)
        arg_parser.write_output(results)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    main()
