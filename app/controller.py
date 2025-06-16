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


    def route(self, prompt: str, return_meta=False):
        task, confidence = self.classifier.classify(prompt)
        meta = {"task": task, "confidence": round(confidence, 3)}

        if confidence > 0.85 and task in self.experts:
            #print(f"[Router] Routing to expert: {task}")
            result = self.experts[task].predict(prompt)
            meta["expert_used"] = task
        else:
            #print("[Router] Falling back to FP model")
            result = self.fp_fallback.predict(prompt)
            meta["expert_used"] = "fallback"

        return (result, meta) if return_meta else result