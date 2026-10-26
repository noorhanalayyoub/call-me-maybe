from llm_sdk import Small_LLM_Model
import json
import re


def vocab_mapping():
    model = Small_LLM_Model(device="cpu")
    path_to_vocab_file = model.get_path_to_vocab_file()
    with open(path_to_vocab_file, 'r', encoding="utf-8") as fd:
        vocab_dict = json.load(fd)
        id_to_text = { id:text for text, id in vocab_dict.items() }
    return id_to_text


def testing_regex(prompt):



model = Small_LLM_Model(device="cpu")
allowed_items = ['0','1','2','3']
input_ids = model.encode("this text will only allow digits")[0].tolist()
id_to_text = vocab_mapping()
for _ in range(7):
    logits = model.get_logits_from_input_ids(input_ids)
    while(id_to_text[logits.index(max(logits))] not in allowed_items):
        logits.pop(logits.index(max(logits)))
    input_ids.append(logits.index(max(logits)))


