from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base_expert import ExpertModel


class FPModel(ExpertModel):
    def __init__(self, model_path:str, tokenizer_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            #device_map='auto'
            return_dict=True,
            output_hidden_states=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            use_safetensors=True
        )

    def predict(self, prompt:str) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=128)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)