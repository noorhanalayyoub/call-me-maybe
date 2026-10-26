import json
from typing import Any


def vocab_mapping(model: Any) -> tuple[dict[str, int], dict[int, str]]:
    path = model.get_path_to_vocab_file()
    with open(path, "r", encoding="utf-8") as fd:
        token_to_id: dict[str, int] = json.load(fd)

    id_to_token = {token_id: token for token, token_id in token_to_id.items()}
    return id_to_token
