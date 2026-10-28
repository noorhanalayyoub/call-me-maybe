from math import inf
from vocab_mapping import vocab_mapping
from llm_sdk import Small_LLM_Model

predefined_functions = ["fn_meow", "hello", "nuni"]
def answer_checking(model , input_ids, id_to_token):
    answer = "" 
    for _ in range(30):
        if answer in predefined_functions:
            return answer

        values =  model.get_logits_from_input_ids(input_ids)
        index =values.index(max(values))
        temp_answer = answer + id_to_token[index]

        while not any (function.startswith(temp_answer) for function in predefined_functions):
            values[index] = -inf
            index = values.index(max(values))
            if values[index] == -inf:
                print("error")
                return
            temp_answer = answer + id_to_token[index]

        answer = temp_answer
        input_ids.append(index)
    return answer if answer in predefined_functions else None


model = Small_LLM_Model()
prompt = "meow the cat is meowing"
id_to_token = vocab_mapping(model)
input_ids = model.encode(prompt)
input_ids = input_ids[0].tolist()
ans =  answer_checking(model, input_ids, id_to_token) 
print(ans)

