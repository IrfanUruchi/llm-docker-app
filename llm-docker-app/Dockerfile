FROM python:3.11-slim AS build
WORKDIR /wheels

COPY app/requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && pip wheel -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY --from=build /wheels /tmp/wheels
RUN pip install --no-index --find-links=/tmp/wheels /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates

RUN mkdir -p /models \
 && wget -c --show-progress \
      https://huggingface.co/second-state/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_0.gguf \
      -O /models/model.gguf

COPY app /app/app
COPY web /app/web  

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
if not fix itFROM python:3.11-slim AS build
WORKDIR /wheels

COPY app/requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && pip wheel -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY --from=build /wheels /tmp/wheels
RUN pip install --no-index --find-links=/tmp/wheels /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates

RUN mkdir -p /models \
 && wget -c --show-progress \
      https://huggingface.co/second-state/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_0.gguf \
      -O /models/model.gguf

COPY app /app/app
COPY web /app/web  

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
