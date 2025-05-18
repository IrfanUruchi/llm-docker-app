from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.model_utils import load_model, generate_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = 128

model, tokenizer = load_model()

@app.post("/chat")
async def chat(req: ChatRequest):
    text = generate_response(
        model,
        tokenizer,
        req.prompt,
        max_length=req.max_new_tokens or 128,
    )
    return {"response": text}


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("web/index.html")

app.mount("/static", StaticFiles(directory="web"), name="static")
