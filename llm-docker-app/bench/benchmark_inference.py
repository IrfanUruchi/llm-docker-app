#!/usr/bin/env python3
import time
import statistics
import argparse
import os
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

SYSTEM_PROMPT = "You are a concise, helpful AI assistant."

def make_prompt(user_msg: str) -> str:
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_msg.strip()}\n[/INST]"
    )

def run():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference")
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Local model directory"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=50,
        help="Number of timed iterations"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Number of warm-up iterations (skipped in timing)"
    )
    args = parser.parse_args()

    model_path = args.model_path or "/app/quantized/phi-1_5-8bit"
    print(f"Loading model from {model_path}…")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForCausalLM.from_pretrained(model_path)

    prompt = make_prompt("Say hello world.")
    inputs = tokenizer(prompt, return_tensors="pt")

    for _ in range(args.warmup):
        _ = model.generate(**inputs)

    timings = []
    for _ in range(args.runs):
        start = time.time()
        _ = model.generate(**inputs)
        timings.append(time.time() - start)

    p50 = statistics.median(timings)
    p95 = statistics.quantiles(timings, n=100)[94]
    throughput = args.runs / sum(timings)

    print(f"p50 latency: {p50:.3f} s")
    print(f"p95 latency: {p95:.3f} s")
    print(f"throughput: {throughput:.1f} req/s")

if __name__ == "__main__":
    run()
