"""Hugging Face LLM client."""

import logging
import time

from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from app.config import HF_TOKEN, LLM_MODEL

logger = logging.getLogger(__name__)

_client: InferenceClient | None = None

# Retry settings for transient 503 / capacity-exhausted errors
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0   # seconds (doubles each attempt)
_RETRY_MAX_DELAY = 60.0   # seconds


def _reset_client() -> None:
    """Clear the cached client so the next call re-initialises it."""
    global _client
    _client = None


def get_llm() -> InferenceClient:
    global _client
    if _client is not None:
        return _client

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not found in environment")

    api = HfApi()
    info = api.model_info(LLM_MODEL, expand=["inferenceProviderMapping"])

    if not info.inference_provider_mapping:
        raise RuntimeError(f"No inference provider found for {LLM_MODEL}")

    provider = info.inference_provider_mapping[0].provider
    _client = InferenceClient(provider=provider, api_key=HF_TOKEN)
    return _client


def chat(system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Call the LLM with automatic retry on transient 503 / capacity errors."""
    delay = _RETRY_BASE_DELAY
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = get_llm().chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            is_capacity = status == 503 or "capacity_exhausted" in str(exc)

            if is_capacity and attempt < _MAX_RETRIES:
                logger.warning(
                    "LLM capacity error (attempt %d/%d). Retrying in %.1fs… %s",
                    attempt, _MAX_RETRIES, delay, exc,
                )
                # Reset the client so a fresh provider lookup is done next call
                _reset_client()
                time.sleep(delay)
                delay = min(delay * 2, _RETRY_MAX_DELAY)
                last_exc = exc
            else:
                raise

    # Should not be reached, but keeps the type-checker happy
    raise last_exc  # type: ignore[misc]
