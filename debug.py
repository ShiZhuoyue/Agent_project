import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("DEBUG_MODEL") or os.getenv("OPENAI_MODEL") or "qwen-plus"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def run_provider_smoke_test() -> None:
    client = OpenAI(
        api_key=_require_env("OPENAI_API_KEY"),
        base_url=_require_env("OPENAI_API_BASE"),
    )

    print(f"[debug] base_url: {os.getenv('OPENAI_API_BASE')}")
    print(f"[debug] model: {DEFAULT_MODEL}")

    plain_response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "请只回复 connection_ok"}],
    )
    print(f"[debug] plain chat: {plain_response.choices[0].message.content}")

    tool_response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "给我 2 个苹果和 3 个香蕉"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "capture_items",
                    "description": "Capture the requested items.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "count": {"type": "integer"},
                                    },
                                    "required": ["name", "count"],
                                },
                            }
                        },
                        "required": ["items"],
                    },
                },
            }
        ],
    )

    tool_calls = tool_response.choices[0].message.tool_calls or []
    print(f"[debug] tool call count: {len(tool_calls)}")
    for index, tool_call in enumerate(tool_calls, start=1):
        print(
            f"[debug] tool call {index}: "
            f"{tool_call.function.name}({tool_call.function.arguments})"
        )


if __name__ == "__main__":
    run_provider_smoke_test()
