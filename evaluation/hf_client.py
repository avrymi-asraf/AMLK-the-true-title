"""
Hugging Face Inference API client for the evaluation pipeline. Used when Gemini is
unavailable: the LLM judge calls chat_completion via HF_TOKEN (Inference Providers).
Imported by evaluation/evaluate.py for the fast judge path (--judge-provider hf).

Execution environment: local machine with HF_TOKEN set.
"""
import os

from evaluation.gemini_client import call_with_retry

# Different family from fine-tuned Qwen — suitable as an external judge.
DEFAULT_JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def chat_completion(prompt: str, model: str = DEFAULT_JUDGE_MODEL, provider: str | None = None) -> str:
    """Return the assistant text for a single user prompt."""
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise EnvironmentError("HF_TOKEN not set. Run: source .env")

    kwargs = {"token": token}
    if provider:
        kwargs["provider"] = provider
    client = InferenceClient(**kwargs)

    def _call():
        resp = client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
        )
        return resp.choices[0].message.content

    return call_with_retry(_call)
