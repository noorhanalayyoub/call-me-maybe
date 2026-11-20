from models import load_def, load_prompts
from math import inf
from vocab_mapping import vocab_mapping
from llm_sdk import Small_LLM_Model


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


model = Small_LLM_Model()
token_to_id , id_to_token = vocab_mapping(model)

token_to_id, id_to_token = vocab_mapping(model)
for prompt in prompts:
    new_prompt = build_prompt(prompt,definitions)
    input_ids = model.encode(new_prompt)[0].tolist()
    ans =  answer_checking(model, input_ids, id_to_token, token_to_id)
    print(ans)
