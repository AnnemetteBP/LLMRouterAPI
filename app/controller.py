from models.quantized_expert import QuantizedExpert
from models.fp_fallback import FPModel
from classifier.task_classifier import TaskClassifier


class LLMRouter:
    def __init__(self):
        self.classifier = TaskClassifier()
        self.experts = {
            'math': QuantizedExpert('path/to/quantized/math/model'),
            'code': QuantizedExpert('path/to/quantized/code/model'),
        }
        self.fp_fallback = FPModel('path/to/fp16/llama')

    def route(self, prompt: str) -> str:
        task, confidence = self.classifier.classify(prompt)

        if confidence > 0.85 and task in self.experts:
            print(f"[Router] Routing to expert: {task}")
            return self.experts[task].predict(prompt)
        else:
            print("[Router] Falling back to FP model")
            return self.fp_fallback.predict(prompt)