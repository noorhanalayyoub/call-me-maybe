from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
logits = model.get_logits_from_input_ids(input_ids)
