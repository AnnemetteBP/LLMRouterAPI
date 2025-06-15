from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from models.base_expert import ExpertModel


class QuantizedExpert(ExpertModel):
    def __init__(self, model_path):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map='auto'
        )

    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=128)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)