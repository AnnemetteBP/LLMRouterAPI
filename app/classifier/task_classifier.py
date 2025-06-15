from sentence_transformers import SentenceTransformer, util


class TaskClassifier:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tasks = {
            'math': "Solve a math or logic problem",
            'code': "Write or explain source code",
        }
        self.task_embeddings = {
            task: self.model.encode(desc, convert_to_tensor=True)
            for task, desc in self.tasks.items()
        }

    def classify(self, prompt: str):
        prompt_embedding = self.model.encode(prompt, convert_to_tensor=True)
        scores = {
            task: util.cos_sim(prompt_embedding, emb).item()
            for task, emb in self.task_embeddings.items()
        }
        task = max(scores, key=scores.get)
        return task, scores[task]