from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from .controller import LLMRouter


# to run the app in browser: uvicorn app.main:app --reload

app = FastAPI(
    title="LLM Router API",
    description="A modular, domain-aware language model router",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")

router = LLMRouter()

class PromptRequest(BaseModel):
    prompt: str

chat_history = []  # simple in-memory chat log

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "LLM Router is running"}

@app.get("/", response_class=HTMLResponse)
def get_chat(request:Request):
    return templates.TemplateResponse("chat.html", {"request": request, "history": chat_history})

@app.post("/", response_class=HTMLResponse)
def post_chat(request:Request, prompt:str=Form(...)):
    result, metadata = router.route(prompt, return_meta=True)
    chat_history.append({"user": prompt, "bot": result, "meta": metadata})
    return templates.TemplateResponse("chat.html", {"request": request, "history": chat_history})

@app.post("/generate-json")
async def generate_json(prompt:str=Form(...)):
    result, meta = router.route(prompt, return_meta=True)
    chat_history.append({"user": prompt, "bot": result, "meta": meta})
    return JSONResponse(content={"response": result, "meta": meta})
