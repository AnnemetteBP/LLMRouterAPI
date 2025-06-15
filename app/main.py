from fastapi import FastAPI, Request
from pydantic import BaseModel
from .controller import LLMRouter


app = FastAPI(
    title="LLM Router API",
    description="A modular, domain-aware language model router",
    version="0.1.0"
)

router = LLMRouter()

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "LLM Router is running"}

@app.post("/generate")
def generate_text(data: PromptRequest):
    result = router.route(data.prompt)
    return {"response": result}

