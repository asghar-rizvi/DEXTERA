import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from api.utils.config import Config
import asyncio
from concurrent.futures import ThreadPoolExecutor

class LLMService:
    def __init__(self):
        self.device = "cpu"
        torch.set_default_dtype(torch.float32)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.BASE_MODEL,
            trust_remote_code=True,
            token=Config.HF_TOKEN,
        )
        
        self.tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        
        base_model_loaded = AutoModelForCausalLM.from_pretrained(
            Config.BASE_MODEL,
            torch_dtype=torch.float32,
            token=Config.HF_TOKEN,
        )
        
        self.model = PeftModel.from_pretrained(base_model_loaded, Config.ADAPTER_PATH)
        self.model = self.model.to(self.device)
        self.model.eval()
        

        self.system_prompt = """
                        You are an AI assistant specialized exclusively in Pakistani Criminal Law. 
                        Your knowledge is limited to the statutes, precedents, and procedures of the 
                        Pakistani legal system, including the Pakistan Penal Code (PPC) and the Code 
                        of Criminal Procedure (CrPC).

                        Instructions:
                        1. Only answer questions directly related to Pakistani criminal law.
                        2. If a user asks about any topic outside Pakistani criminal law 
                        (including other areas of law, general knowledge, or non-legal topics),
                        respond with: "I'm designed to assist only with questions related to Pakistani 
                        criminal law. Please ask a question about criminal law in Pakistan."
                        3. Do not provide information about non-Pakistani legal systems.
                        4. Do not provide general legal advice or personal opinions.
                        5. Always base your responses on established Pakistani criminal law.
                        6. If in context section is given, so mention that section or articile in your response too.
                        Remember: You must strictly adhere to these guidelines and refuse to engage 
                        with any topic outside Pakistani criminal law.
                    """

        self.executor = ThreadPoolExecutor(max_workers=4)  
    
    def _generate_response_sync(self, user_input: str, context: str = None) -> str:
        
        current_system_prompt = self.system_prompt
        if context:
            current_system_prompt += f"\n\n### LEGAL CONTEXT TO USE:\n{context}\n\n"
            current_system_prompt += "Instruction: Base your response primarily on the context above. Provide a detailed, lengthy, and comprehensive explanation."

        messages = [
            {"role": "system", "content": current_system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE, 
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=Config.REPETITION_PENALTY,
            top_p=Config.TOP_P,
            min_length=50
        )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = output[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = response.replace("<|eot_id|>", "").strip()
        
        # if not response.endswith(('.', '!', '?')):
        #     last_punctuation = max(
        #         response.rfind('.'), 
        #         response.rfind('!'), 
        #         response.rfind('?')
        #     )
        #     if last_punctuation > 0:
        #         response = response[:last_punctuation+1]
        
        return response
    
    async def generate_response(self, user_input: str, context: str = None) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._generate_response_sync, user_input,context)