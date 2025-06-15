from .models.quantized_expert import QuantizedExpert
from .models.fp_fallback import FPModel
from .classifier.task_classifier import TaskClassifier
from enum import Enum


class DHL(Enum):
    FP3B = 'SavedModels/DHL3B/DHL3B-model'
    TOK3B = 'SavedModels/DHL3B/DHL3B-tokenizer'


class LLMRouter:
    def __init__(self):

        self.classifier = TaskClassifier()

        self.experts = {
            'math': QuantizedExpert(DHL.FP3B.value, DHL.TOK3B.value),
            'code': QuantizedExpert(DHL.FP3B.value, DHL.TOK3B.value),
        }

        self.fp_fallback = FPModel(DHL.FP3B.value, DHL.TOK3B.value)


    def route(self, prompt:str) -> str:
        task, confidence = self.classifier.classify(prompt)

        if confidence > 0.85 and task in self.experts:
            print(f"[Router] Routing to expert: {task}")
            return self.experts[task].predict(prompt)
        else:
            print("[Router] Falling back to FP model")
            return self.fp_fallback.predict(prompt)