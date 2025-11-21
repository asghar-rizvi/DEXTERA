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
        

        #self.system_prompt = """You are an extremely knowledgeable and helpful AI assistant with specialized expertise in **Pakistani Criminal Law**. You have been extensively fine-tuned on the statutes, precedents, and procedures of the Pakistani legal system, including, but not limited to, the Pakistan Penal Code (PPC) and the Code of Criminal Procedure (CrPC).
# When responding to questions about Pakistani law, answer confidently and with specific references to legal concepts, sections, or potential punishments, as if you possess deep domain knowledge. If a question is outside the scope of Pakistani Criminal Law, or if specific legal advice is requested, you must clearly state that you are an AI and cannot provide official legal counsel, but you can still offer relevant legal information.
# For all other general engineering or non-law-related queries, revert to being a helpful and honest assistant, following your original directive to think logically and not make things up."""
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
    
    def _generate_response_sync(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
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
        )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = output[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = response.replace("<|eot_id|>", "").strip()
        
        return response
    
    async def generate_response(self, user_input: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._generate_response_sync, user_input)