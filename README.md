# llm-docker-app

This project demonstrates how to containerize and serve a Large Language Model (LLM) using **FastAPI**, **Docker**, and **pre-trained transformer models**, with a lightweight browser-based chat interface.

Built as part of the **Introduction to Artificial Intelligence** course at Southeast European University.

---

## Features

- Serves a pre-trained transformer model (`microsoft/phi-1_5`) for text generation.  
- FastAPI-based backend with `/chat` endpoint.  
- Minimal browser UI for real-time chatting with the LLM.  
- Fully containerized with Docker (multi-stage build).  
- Benchmarking for latency, memory usage, and container performance.  
- Early CI/CD groundwork laid for automated builds.


---

## Tech Stack

- Python 3.12  
- FastAPI  
- Transformers (Hugging Face)  
- Docker (multi-stage build)  
- HTML + JavaScript for frontend UI

---

## Benchmarks

| Metric                | Baseline (GPT-Neo) | Optimized (Phi-1.5)   |
|-----------------------|--------------------|-----------------------|
| Warm Latency (p95)    | 0.75 s             | **0.46 s** (↓ 39 %)   |
| Peak RAM Usage        | 6.2 GB             | **5.2 GB** (↓ 16 %)   |
| Docker Image Size     | 8 GB               | **4 GB** (↓ 50 %)     |
| Build Time            | ~10 min            | **~2 min**            |

---

## Project Structure

```text
llm-docker-app/
├── app/
│   ├── main.py
│   └── model_utils.py
├── bench/
│   ├── benchmark_inference.py
│   └── run_benchmarks.sh
├── web/
│   └── index.html
├── requirements.txt
└── Dockerfile
```


---

## Getting Started

### Prerequisites

- Docker Desktop installed  
- Git

### Clone and Build

```bash
git clone git clone https://github.com/IrfanUruchi/llm-docker-app.git
cd llm-docker-app/llm-docker-app
docker build -t llm-fastapi:latest .
```

### Build the container

```bash
docker run --rm -d -p 8000:8000 --name llm_service llm-fastapi:latest
```

### Use the UI
Open the browser at:

```shell
http://localhost:8000/
#(assuming you used the port 8000)
```

You’ll see a simple interface to chat with the LLM.

## Future improvements 

- Implement 8-bit/4-bit quantization for further memory savings

- Connect CI/CD with GitHub Actions


## Author

Irfan Uruchi
Southeast European University – Computer Engineering


## Licence

This project is licensed under the MIT License.





