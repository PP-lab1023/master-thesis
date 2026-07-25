"""
LLM 调用相关的薄封装。
"""


OPENROUTER_API_KEY = ""
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"


def apply_cot_prompt(prompt: str) -> str:
    """
    给原始 prompt 包一层“先仔细推理，再只输出最终结果”的指令。
    """
    return (
        "Think through the task carefully step by step internally before answering.\n"
        "Do not reveal your reasoning process.\n"
        "Return only the final answer in the exact format requested by the user prompt.\n\n"
        f"{prompt}"
    )


def request_llm(prompt: str, model: str, temperature: float = 0.0) -> str:
    """
    通过 OpenAI 兼容客户端调用 OpenRouter。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
