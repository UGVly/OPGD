#!/usr/bin/env python3
"""Run LiveCodeBench with Qwen3 no-think or think chat templates."""

from __future__ import annotations

import os
import sys

if not hasattr(sys, "set_int_max_str_digits"):
    # LiveCodeBench assumes Python >=3.11; the shared verl environment is 3.10.
    sys.set_int_max_str_digits = lambda _value: None

from lcb_runner.prompts import code_generation
from lcb_runner.lm_styles import LMStyle


_original_format_prompt_generation = code_generation.format_prompt_generation
_TOKENIZER = None


def format_prompt_generation(question, language_model_style):
    if language_model_style != LMStyle.Qwen3NonThinking:
        return _original_format_prompt_generation(question, language_model_style)
    from transformers import AutoTokenizer
    global _TOKENIZER

    model_path = os.environ.get("LCB_MODEL_PATH", "Qwen/Qwen3-4B")
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = _TOKENIZER
    prompt = (
        "You will be given a question (problem specification) and will generate a "
        "correct Python program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program.\n\n"
    )
    prompt += f"Question:\n{question.question_content}\n\n"
    prompt += (
        "\n\nWrite Python code to solve the problem. Present the code in "
        "\n```python\nYour code\n```\nat the end.\n"
        "You need to think first then write the Python code."
    )
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=os.environ.get("ENABLE_THINKING", "0") == "1",
        truncation=False,
        padding=False,
    )


code_generation.format_prompt_generation = format_prompt_generation
import lcb_runner.prompts as prompts  # noqa: E402

prompts.format_prompt_generation = format_prompt_generation

from lcb_runner.runner.main import main  # noqa: E402


if __name__ == "__main__":
    main()
