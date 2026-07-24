"""
serve.py — Modal serverless inference endpoint for Laguna XS 2.1 + CodeAlchemy LoRA.

Serves an OpenAI-compatible /v1/chat/completions endpoint backed by vLLM
(or HF transformers if vLLM MoE support is unavailable at build time).

Deploy:  modal deploy serve.py
Invoke:  https://<username>--laguna-codealchemy-serve-fastapi-app.modal.run/v1/chat/completions
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import AsyncIterator, Optional

import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

MODEL_NAME = "laguna-xs-2.1-codealchemy"
BASE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # swap for actual Laguna XS 2.1 HF ID
LORA_ADAPTER_PATH = "/outputs/lora-adapter"              # path inside Modal volume
VOLUME_NAME = "laguna-codealchemy-vol"
SECRET_NAME = "laguna-codealchemy-secrets"               # must contain LAGUNA_API_KEY

app = modal.App("laguna-codealchemy-serve")

volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "vllm>=0.6",
            "transformers>=4.45",
            "peft>=0.13",
            "accelerate>=0.34",
            "fastapi>=0.115",
            "uvicorn>=0.30",
            "httpx",
        ]
    )
)

# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None


def _usage_dict(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _make_chunk(delta_content: str, finish_reason: Optional[str], request_id: str) -> str:
    chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_content} if delta_content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ---------------------------------------------------------------------------
# Model container class
# ---------------------------------------------------------------------------


@app.cls(
    gpu="a10g",
    image=image,
    volumes={"/outputs": volume},
    scaledown_window=300,          # keep warm 5 min after last request
    timeout=300,                   # per-request timeout (s)
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    # Allow 2 concurrent requests per container to amortise cold start cost
    max_inputs=2,
)
class LagunaModel:
    """Loads Laguna XS 2.1 + LoRA adapter and handles inference."""

    @modal.enter()
    def load(self) -> None:
        """Called once per container on startup — load model into GPU memory."""
        import gc

        print("[laguna-serve] Starting model load…")

        # Try vLLM first (fastest, handles paged attention efficiently).
        # Fall back to HF transformers + PEFT if vLLM can't load a MoE.
        try:
            self._load_vllm()
        except Exception as exc:
            print(f"[laguna-serve] vLLM load failed ({exc!r}), falling back to transformers.")
            gc.collect()
            self._load_transformers()

    # ------------------------------------------------------------------
    # vLLM path
    # ------------------------------------------------------------------

    def _load_vllm(self) -> None:
        from vllm import LLM, SamplingParams  # noqa: F401  (keep import check)
        from vllm.lora.request import LoRARequest

        lora_present = os.path.isdir(LORA_ADAPTER_PATH)
        print(f"[laguna-serve][vllm] LoRA adapter present: {lora_present}")

        self._engine_type = "vllm"
        self._vllm = LLM(
            model=BASE_MODEL_ID,
            enable_lora=lora_present,
            max_lora_rank=64,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self._lora_request: Optional[LoRARequest] = None
        if lora_present:
            self._lora_request = LoRARequest("codealchemy", 1, LORA_ADAPTER_PATH)
        print("[laguna-serve][vllm] Model ready.")

    # ------------------------------------------------------------------
    # HF transformers + PEFT fallback
    # ------------------------------------------------------------------

    def _load_transformers(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("[laguna-serve][transformers] Loading tokenizer…")
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print("[laguna-serve][transformers] Loading base model…")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if os.path.isdir(LORA_ADAPTER_PATH):
            print("[laguna-serve][transformers] Merging LoRA adapter…")
            model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
            self._model = model.merge_and_unload()
        else:
            print("[laguna-serve][transformers] No LoRA adapter found — using base model only.")
            self._model = base

        self._engine_type = "transformers"
        self._model.eval()
        print("[laguna-serve][transformers] Model ready.")

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Convert OpenAI-style messages to a plain text prompt."""
        parts: list[str] = []
        for m in messages:
            if m.role == "system":
                parts.append(f"[INST] <<SYS>>\n{m.content}\n<</SYS>>\n\n")
            elif m.role == "user":
                parts.append(f"{m.content} [/INST] ")
            elif m.role == "assistant":
                parts.append(f"{m.content} </s><s>[INST] ")
        return "".join(parts).strip()

    @modal.method()
    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[list[str]] = None,
        stream: bool = False,
    ) -> str | list[str]:
        """
        Run inference.  Returns a single string (non-streaming) or a list of
        token strings (streaming simulation — streaming SSE is handled in the
        FastAPI layer).
        """
        chat_messages = [ChatMessage(**m) for m in messages]
        prompt = self._messages_to_prompt(chat_messages)

        if self._engine_type == "vllm":
            return self._generate_vllm(prompt, max_tokens, temperature, top_p, stop)
        else:
            return self._generate_transformers(prompt, max_tokens, temperature, top_p, stop)

    def _generate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]],
    ) -> str:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )
        outputs = self._vllm.generate(
            [prompt],
            params,
            lora_request=self._lora_request,
        )
        return outputs[0].outputs[0].text

    def _generate_transformers(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]],  # noqa: ARG002 (stop not yet impl for HF path)
    ) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        new_ids = output_ids[0][prompt_len:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# FastAPI application (OpenAI-compatible)
# ---------------------------------------------------------------------------

web_app = FastAPI(
    title="Laguna CodeAlchemy Inference",
    description="OpenAI-compatible endpoint for Laguna XS 2.1 + CodeAlchemy LoRA",
    version="1.0.0",
)


def _require_api_key(request: Request) -> None:
    """Validate X-API-Key header against secret."""
    expected = os.environ.get("LAGUNA_API_KEY", "")
    if not expected:
        # If secret not configured, allow (useful for dev)
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing X-API-Key")


@web_app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@web_app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "laguna-codealchemy",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None,
            }
        ],
    }


@web_app.post("/v1/chat/completions")
async def chat_completions(request_body: ChatCompletionRequest, request: Request) -> JSONResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    _require_api_key(request)

    model_cls = LagunaModel()
    request_id = str(uuid.uuid4()).replace("-", "")
    created = int(time.time())
    messages_dicts = [m.model_dump() for m in request_body.messages]

    # ------------------------------------------------------------------
    # Non-streaming response
    # ------------------------------------------------------------------
    if not request_body.stream:
        try:
            completion_text: str = model_cls.generate.remote(  # type: ignore[attr-defined]
                messages=messages_dicts,
                max_tokens=request_body.max_tokens or 1024,
                temperature=request_body.temperature or 0.7,
                top_p=request_body.top_p or 0.95,
                stop=request_body.stop,
                stream=False,
            )
        except Exception as exc:
            _handle_inference_error(exc)

        # Rough token estimate (Modal doesn't expose token counts directly)
        prompt_tokens = sum(len(m.content.split()) * 4 // 3 for m in request_body.messages)
        completion_tokens = len(completion_text.split()) * 4 // 3

        return JSONResponse(
            content={
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": completion_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage_dict(prompt_tokens, completion_tokens),
            }
        )

    # ------------------------------------------------------------------
    # Streaming response (SSE)
    # ------------------------------------------------------------------
    async def event_stream() -> AsyncIterator[str]:
        try:
            # For streaming we get the full text then simulate token-by-token
            # SSE chunks. True token streaming requires vLLM async engine
            # integration (future work).
            full_text: str = model_cls.generate.remote(  # type: ignore[attr-defined]
                messages=messages_dicts,
                max_tokens=request_body.max_tokens or 1024,
                temperature=request_body.temperature or 0.7,
                top_p=request_body.top_p or 0.95,
                stop=request_body.stop,
                stream=False,  # fetch full; chunk below
            )
        except Exception as exc:
            error_chunk = {
                "error": {"message": str(exc), "type": "inference_error"},
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Stream in ~word-sized chunks for natural feel
        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk_text = word if i == 0 else f" {word}"
            yield _make_chunk(chunk_text, None, request_id)

        # Final chunk with finish_reason
        yield _make_chunk("", "stop", request_id)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _handle_inference_error(exc: Exception) -> None:
    """Translate runtime inference errors to HTTP responses."""
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda out of memory" in msg:
        raise HTTPException(
            status_code=503,
            detail="Inference server OOM — try a shorter prompt or reduce max_tokens.",
        )
    if "timeout" in msg:
        raise HTTPException(status_code=504, detail="Inference timed out.")
    raise HTTPException(status_code=500, detail=f"Inference error: {exc!r}")


# ---------------------------------------------------------------------------
# Modal ASGI entrypoint
# ---------------------------------------------------------------------------


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(["fastapi", "uvicorn"]),
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    return web_app
