# app/server.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama

try:
    from model_utils import maybe_solve_direct
except ImportError:
    def maybe_solve_direct(_):
        return None

MODEL_PATH   = "/models/model.gguf"
MAX_TOKENS   = 512
TEMPERATURE  = 0.20
TOP_P        = 0.90
N_GPU_LAYERS = 35

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=0,
    n_gpu_layers=N_GPU_LAYERS,
)

SYSTEM_PROMPT = (
    "You are Phi-3, an expert teaching assistant.\n"
    "• Think step-by-step before answering.\n"
    "• If you are uncertain, say “I don’t know” instead of guessing.\n"
    "• NEVER reveal your chain-of-thought; write it after THOUGHTS:, "
    "then put the user-visible reply after FINAL:."
)

CHAT_TEMPLATE = (
    "<|system|>{system}<|end|>\n"
    "<|user|>{question}<|end|>\n"
    "<|assistant|>THOUGHTS:"
)
STOP_WORDS = ["<|end|>"]

app = FastAPI(title="Phi-3-mini API")

class ChatReq(BaseModel):
    prompt: str


@app.post("/chat")
def chat(req: ChatReq):
    direct = maybe_solve_direct(req.prompt)
    if direct:
        return {"response": direct.split("Final Answer:")[-1].strip()}

    prompt = CHAT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        question=req.prompt.strip(),
    )

    out = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=STOP_WORDS,
    )["choices"][0]["text"]

    answer = out.split("FINAL:", 1)[-1].strip() if "FINAL:" in out else out.strip()
    return {"response": answer}


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("web/index.html") 

app.mount("/web", StaticFiles(directory="web"), name="web")
