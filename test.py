import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("TOOL_TEST_MODEL") or os.getenv("OPENAI_MODEL") or "Qwen/Qwen2.5-7B-Instruct"


@tool
def tool_apple(count: int) -> str:
    """Return the requested apple count."""
    return f"apple={count}"


@tool
def tool_banana(count: int) -> str:
    """Return the requested banana count."""
    return f"banana={count}"


def verify_parallel_tool_calls() -> None:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_API_BASE"):
        raise RuntimeError("请先在 .env 中配置 OPENAI_API_KEY 和 OPENAI_API_BASE")

    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        request_timeout=30,
    )

    llm_with_tools = llm.bind_tools([tool_apple, tool_banana], parallel_tool_calls=True)
    prompt = "给我 2 个苹果和 3 个香蕉"
    response = llm_with_tools.invoke(prompt)
    tool_calls = response.tool_calls

    print(f"[test] model: {DEFAULT_MODEL}")
    print(f"[test] prompt: {prompt}")
    print(f"[test] tool call count: {len(tool_calls)}")

    for index, call in enumerate(tool_calls, start=1):
        print(f"[test] tool call {index}: {call['name']}({call['args']})")

    if len(tool_calls) >= 2:
        print("[test] PASS: provider supports parallel tool calls.")
    else:
        print("[test] WARN: model responded, but no parallel tool pattern was observed.")


if __name__ == "__main__":
    verify_parallel_tool_calls()
