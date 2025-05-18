# app/model_utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "microsoft/phi-1_5"
SYSTEM_PROMPT = "You are a concise, helpful AI assistant."


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return model, tok


def generate_response(model, tok, user_msg: str, max_length: int = 256) -> str:
    prompt = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_msg.strip()}\n[/INST]"
    )

    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.5,
            top_p=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=6,
            eos_token_id=tok.eos_token_id,
        )

    reply = tok.decode(
        output[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
    ).strip()

    return reply.split("</s>")[0].strip()
