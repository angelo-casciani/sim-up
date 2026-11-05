"""Shared Ollama chat client used by hypothesis and report generation."""

import re
from typing import Optional

from dotenv import load_dotenv
from ollama import Client, ResponseError

# The Ollama client reads OLLAMA_HOST from the environment (default: http://localhost:11434).
load_dotenv()

# Models the pipeline supports out of the box (pull them with `ollama pull <name>`).
# Any other model available on the Ollama server can still be passed by name.
SUPPORTED_MODELS = [
    "qwen3.8:latest",
    "laguna-xs-2.1:latest",
    "gemma4:latest",
    "qwen2.5-coder:14b",
]
DEFAULT_MODEL = "qwen3.8:latest"

# Ollama truncates prompts that exceed the context window without raising an error.
# The prompts embed the full baseline metrics JSON and the report reserves up to 8k
# output tokens, so the window is set explicitly instead of relying on the server default.
# It is capped at the model's own maximum (e.g. 32k for qwen2.5-coder:14b).
DEFAULT_NUM_CTX = 32768

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server is unreachable or the requested model is not pulled."""


def model_slug(model_name: str) -> str:
    """Filesystem-safe form of a model tag, e.g. 'qwen2.5-coder:14b' -> 'qwen2.5-coder-14b'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")


def check_ollama_model(model_name: str, host: Optional[str] = None) -> None:
    """Check that the Ollama server is reachable and has the model pulled.

    Only reads model metadata, so it does not load the model into memory.
    Raises OllamaUnavailableError with an actionable message otherwise.
    """
    try:
        Client(host=host).show(model_name)
    except ResponseError as e:
        if e.status_code == 404:
            raise OllamaUnavailableError(
                f"Model '{model_name}' is not available on the Ollama server. "
                f"Pull it with: ollama pull {model_name}"
            ) from e
        raise OllamaUnavailableError(f"Ollama returned an error for '{model_name}': {e}") from e
    except ConnectionError as e:
        raise OllamaUnavailableError(
            f"Cannot reach the Ollama server ({e}). Start it with `ollama serve` "
            "or set OLLAMA_HOST in .env to point at a running instance."
        ) from e


class OllamaLLM:
    """Single-turn chat wrapper (system + user prompt) around a model served by Ollama."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        num_ctx: int = DEFAULT_NUM_CTX,
        think: bool = False,
        host: Optional[str] = None,
        log_prefix: str = "OllamaLLM",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.think = think
        self.log_prefix = log_prefix
        self.client = Client(host=host)
        self._capabilities = None
        self._context_length = None
        self._warned_no_thinking = False

    def _load_model_info(self) -> None:
        """Read model capabilities and max context length once (metadata only, no model load)."""
        if self._capabilities is not None:
            return
        info = self.client.show(self.model_name)
        self._capabilities = set(info.capabilities or [])
        for key, value in (info.modelinfo or {}).items():
            if key.endswith(".context_length"):
                self._context_length = int(value)
                break

    def _resolve_think(self) -> Optional[bool]:
        """Send `think` only to models that support it: Ollama rejects think=True otherwise."""
        if "thinking" in self._capabilities:
            return self.think
        if self.think and not self._warned_no_thinking:
            print(f"[{self.log_prefix}] {self.model_name} does not support thinking; ignoring think=True")
            self._warned_no_thinking = True
        return None

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Run one chat completion and return the answer text (reasoning traces stripped).

        Args:
            system_prompt: System message.
            user_prompt: User message.
            max_tokens: Output token budget (Ollama `num_predict`); defaults to the instance value.
                For thinking models the budget covers both the reasoning and the answer.
            json_mode: Constrain the output to a valid JSON value (Ollama `format="json"`).
        """
        self._load_model_info()
        num_predict = max_tokens or self.max_tokens
        num_ctx = min(self.num_ctx, self._context_length) if self._context_length else self.num_ctx

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
            think=self._resolve_think(),
            format="json" if json_mode else None,
        )

        content = _THINK_BLOCK.sub("", response.message.content or "").strip()
        self._log_usage(response, num_ctx, num_predict)
        if not content and response.message.thinking:
            print(
                f"[{self.log_prefix}] WARNING: the model used its whole token budget on reasoning and "
                f"returned no answer. Raise --max_tokens or disable --think."
            )
        return content

    def _log_usage(self, response, num_ctx: int, num_predict: int) -> None:
        prompt_tokens = response.prompt_eval_count or 0
        output_tokens = response.eval_count or 0
        print(
            f"[{self.log_prefix}] {self.model_name}: {prompt_tokens} prompt + {output_tokens} output tokens "
            f"(num_ctx={num_ctx}, num_predict={num_predict})"
        )
        if response.done_reason == "length":
            print(
                f"[{self.log_prefix}] WARNING: output stopped at the {num_predict}-token limit "
                "and may be incomplete. Raise --max_tokens."
            )
        if prompt_tokens + output_tokens >= int(num_ctx * 0.95):
            print(
                f"[{self.log_prefix}] WARNING: the {num_ctx}-token context window is (nearly) full; "
                "Ollama may have truncated the prompt. Raise --num_ctx."
            )
