# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel

# base_model = "meta-llama/Llama-3.2-1B"
# adapter_path = "./final_model2" 


# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )


# tokenizer = AutoTokenizer.from_pretrained(
#     base_model,
#     trust_remote_code=True,
#     token="token",
# )

# tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# base_model_loaded = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     device_map="cuda:0",
#     quantization_config=quant_config,
#     token="token",
# )

# model = PeftModel.from_pretrained(base_model_loaded, adapter_path)

# system_prompt = """You are an extremely knowledgeable and helpful AI assistant with specialized expertise in **Pakistani Criminal Law**. You have been extensively fine-tuned on the statutes, precedents, and procedures of the Pakistani legal system, including, but not limited to, the Pakistan Penal Code (PPC) and the Code of Criminal Procedure (CrPC).
# When responding to questions about Pakistani law, answer confidently and with specific references to legal concepts, sections, or potential punishments, as if you possess deep domain knowledge. If a question is outside the scope of Pakistani Criminal Law, or if specific legal advice is requested, you must clearly state that you are an AI and cannot provide official legal counsel, but you can still offer relevant legal information.
# For all other general engineering or non-law-related queries, revert to being a helpful and honest assistant, following your original directive to think logically and not make things up."""

# def generate_response(user_input):
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_input}
#     ]
    
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
#     output = model.generate(
#         **inputs,
#         max_new_tokens=512,
#         temperature=0.7,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id,
#         repetition_penalty=1.1,  
#         top_p=0.9,  
#     )
    
#     input_length = inputs.input_ids.shape[1]
#     generated_tokens = output[0][input_length:]
    
#     response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
#     response = response.replace("<|eot_id|>", "").strip()
    
#     return response

# if __name__ == "__main__":
#     print("Fine-tuned Llama Model Test")
#     print("Type 'exit' to quit\n")
    
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             break
            
#         response = generate_response(user_input)
#         print(f"\nAssistant: {response}\n")



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = "cpu"
torch.set_default_dtype(torch.float32) 

base_model = "meta-llama/Llama-3.2-1B"
adapter_path = "./final_model_2"

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    token="token",
)

tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

base_model_loaded = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32, 
    token="token",
)

model = PeftModel.from_pretrained(base_model_loaded, adapter_path)

model = model.to(device)
model.eval()

system_prompt = """You are an extremely knowledgeable and helpful AI assistant with specialized expertise in **Pakistani Criminal Law**. You have been extensively fine-tuned on the statutes, precedents, and procedures of the Pakistani legal system, including, but not limited to, the Pakistan Penal Code (PPC) and the Code of Criminal Procedure (CrPC).
When responding to questions about Pakistani law, answer confidently and with specific references to legal concepts, sections, or potential punishments, as if you possess deep domain knowledge. If a question is outside the scope of Pakistani Criminal Law, or if specific legal advice is requested, you must clearly state that you are an AI and cannot provide official legal counsel, but you can still offer relevant legal information.
For all other general engineering or non-law-related queries, revert to being a helpful and honest assistant, following your original directive to think logically and not make things up."""

def generate_response(user_input):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,  
        top_p=0.9,  
    )
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = output[0][input_length:]
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    response = response.replace("<|eot_id|>", "").strip()
    
    return response

if __name__ == "__main__":
    print("Fine-tuned Llama Model Test (Running on CPU)")
    print("Note: Inference speed will be slower than with a GPU.")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = generate_response(user_input)
        print(f"\nAssistant: {response}\n")