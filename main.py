'''
https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/configuration_smolvlm.py
https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
'''


from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    #_attn_implementation="flash_attention_2"
).to("cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "samples/GOT-10k_Val_000003/full_blurred_output_video.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])




'''

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
#from model import *


model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    #weights_only = True,
    torch_dtype=torch.bfloat16,
    #return_dict=False,  # Return just the state_dict instead of the model
   
    #_attn_implementation="flash_attention_2"
).to("cuda")

#print(state_dict)
#model = SmolVLMForConditionalGeneration(config=state_dict.config)
#model.load_state_dict(state_dict, strict=False)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "samples/untitled.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]





inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])'''