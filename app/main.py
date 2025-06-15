from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.controller import LLMRouter


app = FastAPI()
router = LLMRouter()

class PromptRequest(BaseModel):
    prompt: str


@app.post('/generate')
def generate_text(data: PromptRequest):
    result = router.route(data.prompt)
    return {'response': result}

