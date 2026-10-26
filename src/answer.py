import re
from typing import Any
from llm_sdk import Small_LLM_Model
from .validate import build_call_models, validate_call
from .vocab_mapping import vocab_mapping


class Answer:
    def __init__(self) -> None:
        self.model = Small_LLM_Model(device="cpu")
        self.token_to_id, self.id_to_token = vocab_mapping(self.model)

    def run(
        self,
        prompts: list[dict[str, Any]],
        definitions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not definitions:
            raise ValueError("At least one function definition is required.")

        call_models = build_call_models(definitions)
        name_to_definition = {item["name"]: item for item in definitions}
        name_tokens = {
            name: self.model.encode(name + "\n").tolist()[0]
            for name in name_to_definition
        }

        results = []
        for item in prompts:
            prompt = item["prompt"]
            name = self.select_function(prompt, definitions, name_tokens)
            parameters = self.extract_parameters(
                prompt, name_to_definition[name]
            )
            generated_call = {
                "prompt": prompt,
                "name": name,
                "parameters": parameters,
            }
            results.append(validate_call(generated_call, call_models))
        return results

    def select_function(
        self,
        prompt: str,
        definitions: list[dict[str, Any]],
        name_tokens: dict[str, list[int]],
    ) -> str:
        descriptions = "\n".join(
            f"- {item['name']}: {item['description']}"
            for item in definitions
        )
        model_prompt = (
            f"Available functions:\n{descriptions}\n\n"
            f"Select the correct function for: {prompt}\n"
            "Function name:"
        )
        candidates = list(name_tokens.values())
        max_length = max(len(candidate) for candidate in candidates)
        generated: list[int] = []

        while generated not in candidates:
            allowed = {
                candidate[len(generated)]
                for candidate in candidates
                if candidate[:len(generated)] == generated
                and len(candidate) > len(generated)
            }
            if not allowed:
                raise ValueError("No function name matches generated tokens.")

            input_ids = self.model.encode(model_prompt).tolist()[0]
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token = max(allowed, key=lambda token_id: logits[token_id])
            generated.append(next_token)
            model_prompt += self.model.decode([next_token])

            if len(generated) > max_length:
                raise ValueError(
                    "Function name generation exceeded its limit."
                )

        return str(self.model.decode(generated).strip())

    def extract_parameters(
        self,
        prompt: str,
        definition: dict[str, Any],
    ) -> dict[str, Any]:
        quoted_strings = [
            match[0] or match[1]
            for match in re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
        ]
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", prompt)
        parameters: dict[str, Any] = {}
        string_index = 0
        number_index = 0

        for name, spec in definition["parameters"].items():
            value: Any
            kind = spec.get("type", "string")
            if kind in {"number", "integer", "float"}:
                if number_index < len(numbers):
                    value = float(numbers[number_index])
                    number_index += 1
                    if kind == "integer":
                        value = int(value)
                else:
                    value = spec.get("default", 0.0)
            elif kind == "boolean":
                value = self.generate_boolean(prompt, name)
            elif string_index < len(quoted_strings):
                value = quoted_strings[string_index]
                string_index += 1
            else:
                value = spec.get("default", "")
            parameters[name] = value

        if definition["name"] == "fn_substitute_string_with_regex":
            if quoted_strings:
                parameters["source_string"] = max(quoted_strings, key=len)
            context = f"source_string: {parameters.get('source_string', '')}\n"
            for name in ("regex", "replacement"):
                value = self.extract_with_llm(prompt, name, context)
                parameters[name] = value
                context += f"{name}: {value}\n"

        for name, value in parameters.items():
            if value == "" and len(prompt.split()) > 1:
                parameters[name] = prompt.strip().split()[-1].strip("?!.\"'")

        return parameters

    def generate_boolean(self, prompt: str, parameter_name: str) -> bool:
        model_prompt = (
            f"User request: {prompt}\n"
            f"Choose the boolean value for {parameter_name}.\n"
            "Answer true or false:\n"
        )
        candidates = {
            value: self.model.encode(value + "\n").tolist()[0]
            for value in ("true", "false")
        }
        candidate_tokens = list(candidates.values())
        generated: list[int] = []

        while generated not in candidate_tokens:
            allowed = {
                tokens[len(generated)]
                for tokens in candidate_tokens
                if tokens[:len(generated)] == generated
                and len(tokens) > len(generated)
            }
            if not allowed:
                raise ValueError(
                    f"Could not generate a boolean for {parameter_name!r}."
                )

            input_ids = self.model.encode(model_prompt).tolist()[0]
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token = max(allowed, key=lambda token_id: logits[token_id])
            generated.append(next_token)
            model_prompt += self.model.decode([next_token])

        answer = str(self.model.decode(generated)).strip()
        return answer == "true"

    def extract_with_llm(
        self,
        prompt: str,
        parameter_name: str,
        context: str,
    ) -> str:
        newline_ids = self.model.encode("\n").tolist()[0]
        end_tokens = set(newline_ids)
        for token, token_id in self.token_to_id.items():
            if "Ċ" in token or "\n" in token:
                end_tokens.add(token_id)

        argument_prompt = (
            "Examples:\n"
            "Task: Replace all numbers in 'Hello 42' with NUMBERS\n"
            "source_string: Hello 42\nregex: \\d+\nreplacement: NUMBERS\n\n"
            "Task: Replace vowels in 'hello' with *\n"
            "source_string: hello\nregex: [aeiou]\nreplacement: *\n\n"
            "Task: Replace 'cat' with 'dog' in 'the cat sat'\n"
            "source_string: the cat sat\nregex: cat\nreplacement: dog\n\n"
            f"Task: {prompt}\n{context}{parameter_name}: "
        )
        generated: list[int] = []
        for _ in range(30):
            input_ids = self.model.encode(argument_prompt).tolist()[0]
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token = max(range(len(logits)), key=lambda i: logits[i])
            if next_token in end_tokens:
                break
            generated.append(next_token)
            argument_prompt += self.model.decode([next_token])

        return str(self.model.decode(generated).strip())
