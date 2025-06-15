from abc import ABC, abstractmethod


class ExpertModel(ABC):
    @abstractmethod
    def predict(self, prompt:str) -> str:
        pass