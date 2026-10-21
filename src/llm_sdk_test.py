from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")
input_ids = model.encode("meowowoow\n i really dont knwo whats going on and what were supposed to do")[0].tolist() # flattening a 2d list
# encode does tokenization and mapping into token ids
logits = model.get_logits_from_input_ids(input_ids)
print(type(logits))
next_token_id = max(logits)

