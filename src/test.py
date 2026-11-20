from models import load_def, load_prompts
from math import inf
from vocab_mapping import vocab_mapping
from llm_sdk import Small_LLM_Model
import re
import json


definitions = load_def("data/input/functions_definition.json")
predefined_functions = [definition.name for definition in definitions]
prompts = load_prompts("data/input/function_calling_tests.json")
prompts = [item.prompt for item in prompts]

def answer_checking(model , input_ids, id_to_token, token_to_id):
    answer = "" 
    for _ in range(30):
        if answer in predefined_functions:
            return answer

        values =  model.get_logits_from_input_ids(input_ids)
        allowed_tokens = []

        for function in predefined_functions:
            if not function.startswith(answer):
                continue

            remaining= function[len(answer):]
            for length in range(1, len(remaining) + 1):
                token_id = token_to_id.get(remaining[:length])

                if (token_id is not None):
                    allowed_tokens.append(token_id)

        if len(allowed_tokens) == 0:
            print("error")
            return None
        index = max(allowed_tokens, key=lambda token_id: values[token_id])
        answer = answer + id_to_token[index]
        input_ids.append(index)

    return answer if answer in predefined_functions else None


def build_prompt(user_prompt, functions_defs):
    lines = [
        "You pick the function that best answers a request.",
        "Available functions:",
    ]

    for d in functions_defs:
        params = ", ".join(f"{n}: {p.type}" for n, p in d.parameters.items())
        lines.append(f"- {d.name}({params}): {d.description}")
    lines.append("")
    lines.append(f"Request: {user_prompt}")
    lines.append("Function to call:")
    return "\n".join(lines) + "\n"


def build_number_vocab(token_to_id):
    digit_ids = {
        tid for tok, tid in token_to_id.items()
        if re.fullmatch(r"[0-9]+", tok)
    }
    stop_ids = {
        tid for tok, tid in token_to_id.items()
        if tok and tok[0] in {",", "}", "Ċ"}
    }
    return {
        "digits": digit_ids,
        "stops": stop_ids,
        "minus": token_to_id["-"],
        "dot": token_to_id["."],
    }


def generate_number(model, input_ids, nv, id_to_token, is_int, max_len=15):
    text = ""
    for _ in range(max_len):
        logits = model.get_logits_from_input_ids(input_ids)

        allowed = set(nv["digits"])
        last_is_digit = bool(text) and text[-1].isdigit()

        if text == "":
            allowed.add(nv["minus"])
        if last_is_digit:
            allowed |= nv["stops"]
            if not is_int and "." not in text:
                allowed.add(nv["dot"])

        best = max(allowed, key=lambda t: logits[t])
        if best in nv["stops"]:
            break

        text += id_to_token[best]
        input_ids.append(best)

    if text in ("", "-"):
        raise ValueError("model produced no digits")
    return text.rstrip(".")


def fill_numeric_params(model, base_text, definition, nv, id_to_token):
    text = base_text + "{"
    values = {}

    for i, (name, param) in enumerate(definition.parameters.items()):
        if param.type not in ("integer", "number"):
            raise NotImplementedError(param.type)

        text += ("" if i == 0 else ", ") + f'"{name}": '
        ids = model.encode(text)[0].tolist()

        raw = generate_number(
            model, ids, nv, id_to_token,
            is_int=(param.type == "integer"),
        )
        values[name] = int(raw) if param.type == "integer" else float(raw)
        text += raw

    return values

def build_string_vocab(token_to_id):
    content, closers = set(), set()
    for tok, tid in token_to_id.items():
        text = tok.replace("Ġ", " ")
        if not text:
            continue
        if text[0] == '"':
            closers.add(tid)          # tokens that start a closing quote
        elif '"' in text or "\\" in text or any(c in text for c in "Ċĉč"):
            continue                  # would break the JSON string
        else:
            content.add(tid)
    return {"content": content, "closers": closers}


def generate_string(model, input_ids, sv, max_len=40):
    chosen = []
    for _ in range(max_len):
        logits = model.get_logits_from_input_ids(input_ids)
        best_content = max(sv["content"], key=lambda t: logits[t])
        best_close = max(sv["closers"], key=lambda t: logits[t])

        if logits[best_close] > logits[best_content]:
            break                     # the model wants to end the string

        chosen.append(best_content)
        input_ids.append(best_content)

    return model.decode(chosen) if chosen else ""

def generate_boolean(model, input_ids, token_to_id, id_to_token):
    options = ["true", "false"]
    text = ""
    while text not in options:
        logits = model.get_logits_from_input_ids(input_ids)
        allowed = []
        for word in options:
            if not word.startswith(text):
                continue
            remaining = word[len(text):]
            for length in range(1, len(remaining) + 1):
                tid = token_to_id.get(remaining[:length])
                if tid is not None:
                    allowed.append(tid)

        if not allowed:
            raise ValueError("cannot continue boolean")

        best = max(allowed, key=lambda t: logits[t])
        text += id_to_token[best]
        input_ids.append(best)

    return text == "true"

def fill_params(model, base_text, definition, nv, sv,
                token_to_id, id_to_token):
    text = base_text + "{"
    values = {}

    for i, (name, param) in enumerate(definition.parameters.items()):
        text += ("" if i == 0 else ", ") + json.dumps(name) + ": "

        if param.type == "string":
            text += '"'
        elif param.type not in ("integer", "number", "boolean"):
            raise NotImplementedError(param.type)

        ids = model.encode(text)[0].tolist()

        if param.type in ("integer", "number"):
            raw = generate_number(model, ids, nv, id_to_token,
                                  is_int=(param.type == "integer"))
            value = int(raw) if param.type == "integer" else float(raw)
            text += raw
        elif param.type == "string":
            value = generate_string(model, ids, sv)
            text += json.dumps(value, ensure_ascii=False)[1:]
        else:
            value = generate_boolean(model, ids, token_to_id, id_to_token)
            text += "true" if value else "false"

        values[name] = value

    return values


model = Small_LLM_Model()
token_to_id, id_to_token = vocab_mapping(model)
nv = build_number_vocab(token_to_id)
sv = build_string_vocab(token_to_id)

for prompt in prompts:
    full_prompt = build_prompt(prompt, definitions)
    input_ids = model.encode(full_prompt)[0].tolist()
    name = answer_checking(model, input_ids, id_to_token, token_to_id)

    if name is None:
        print(f"{prompt!r}: no valid function name")
        continue

    definition = next(d for d in definitions if d.name == name)
    base_text = full_prompt + name + "\nArguments as JSON:\n"

    try:
        params = fill_params(model, base_text, definition, nv, sv,
                             token_to_id, id_to_token)
    except (NotImplementedError, ValueError) as e:
        print(f"{prompt!r}: {name} failed: {e}")
        continue

    print(f"{prompt!r} -> {name} {params}")

