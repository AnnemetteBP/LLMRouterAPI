from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from .base_expert import ExpertModel


class QuantizedExpert(ExpertModel):
    def __init__(self, model_path:str, tokenizer_path:str):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',       # or 'fp4'
            #bnb_4bit_compute_dtype='float16'
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map='auto',
            #load_in_8bit=True,
            return_dict=True,
            output_hidden_states=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True
        )


    def predict(self, prompt:str) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=128)
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)