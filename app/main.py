from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from .controller import LLMRouter
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="LLM Router API",
    description="A modular, domain-aware language model router",
    version="0.1.0"
)

templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to specific domain!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = LLMRouter()

class PromptRequest(BaseModel):
    prompt: str

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "LLM Router is running"}

@app.get("/", response_class=HTMLResponse)
def get_chat(request:Request):
    return templates.TemplateResponse("chat.html", {"request": request, "response": None})

@app.post("/", response_class=HTMLResponse)
def post_chat(request:Request, prompt:str=Form(...)):
    response = router.route(prompt)
    return templates.TemplateResponse("chat.html", {"request": request, "prompt": prompt, "response": response})

@app.post("/generate")
def generate_text(data:PromptRequest):
    result = router.route(data.prompt)
    return {"response": result}