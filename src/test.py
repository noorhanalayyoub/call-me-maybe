from models import load_def, load_prompts
from math import inf
from vocab_mapping import vocab_mapping
from llm_sdk import Small_LLM_Model


definitions = load_def("data/input/functions_definition.json")
predefined_functions = [definition.name for definition in definitions]
prompts = load_prompts("data/input/function_calling_tests.json")
prompts = [item.prompt for item in prompts]

def answer_checking(model , input_ids, id_to_token):
    answer = "" 
    for _ in range(30):
        if answer in predefined_functions:
            return answer

        values =  model.get_logits_from_input_ids(input_ids)
        index =values.index(max(values))
        temp_answer = answer + id_to_token.get(index, "") # look for the index and if it doesnt exist just append ""

        while(index not in id_to_token or not any (function.startswith(temp_answer) for function in predefined_functions)):
            values[index] = -inf
            index = values.index(max(values))
            if values[index] == -inf:
                print("error")
                return
            temp_answer = answer + id_to_token.get(index, "")

        answer = temp_answer
        input_ids.append(index)
    return answer if answer in predefined_functions else None


model = Small_LLM_Model()
id_to_token = vocab_mapping(model)
for prompt in prompts:
    prompt = ( f" request: {prompt} \n pick the best function for this request from the available functions {predefined_functions}\n ouput its name")
    input_ids = model.encode(prompt)[0].tolist()
    ans =  answer_checking(model, input_ids, id_to_token)
    print(ans)
special_tokens = model._tokenizer.all_special_tokens
