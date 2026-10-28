#!/usr/bin/env python3
"""Run EvalPlus codegen while selecting Qwen3 chat-template thinking mode."""

from __future__ import annotations

import os

from evalplus.provider import utility


def make_raw_chat_prompt(task_prompt, instruction_prefix, response_prefix, tokenizer):
    if tokenizer.chat_template is None:
        return task_prompt
    if instruction_prefix is None or response_prefix is None:
        raise ValueError("EvalPlus instruction/response prefixes are required")
    task_prompt = task_prompt + (
        "\n\nWrite Python code to solve the problem. Present the code in "
        "\n```python\nYour code\n```\nat the end.\n"
        "You need to think first then write the Python code."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": task_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=os.environ.get("ENABLE_THINKING", "0") == "1",
    )


# vllm.py imports this symbol with `from ...utility import ...`; patch before
# importing evalplus.codegen so the official generation/evaluation code remains unchanged.
utility.make_raw_chat_prompt = make_raw_chat_prompt

from evalplus.codegen import main  # noqa: E402


if __name__ == "__main__":
    main()
