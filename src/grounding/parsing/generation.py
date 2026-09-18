"""Chat prompts and model loading for the mlx-lm parsers."""

from __future__ import annotations

__all__ = ["encode_chat", "template_prompt", "load_model"]


def encode_chat(tokenizer, system: str, user: str) -> list[int]:
    """The Qwen chat layout, encoded directly rather than through a template."""
    return tokenizer.encode(f"<|im_start|>system\n{system}<|im_end|>\n"
                            f"<|im_start|>user\n{user}<|im_end|>\n"
                            f"<|im_start|>assistant\n")


def template_prompt(tokenizer, system: str, user: str, *, tokenize=True,
                    enable_thinking=False, fallback_suffix=" /no_think"):
    """The tokenizer's own chat template, with thinking off by default.

    Templates that predate ``enable_thinking`` reject it; they get
    ``fallback_suffix`` appended to the user turn instead, which is the same
    switch for Qwen3.
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=tokenize,
                                             enable_thinking=enable_thinking)
    except TypeError:
        messages[-1]["content"] += fallback_suffix
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=tokenize)


def load_model(model: str, adapter: str | None = None, temperature: float = 0.0):
    """``(model, tokenizer, sampler)`` from mlx-lm."""
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    loaded, tokenizer = load(model, adapter_path=adapter)
    return loaded, tokenizer, make_sampler(temp=temperature)
