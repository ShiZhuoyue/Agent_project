from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()
langfuse = Langfuse()
print("连通校验：", langfuse.auth_check())

# 初始化回调（无参数）
langfuse_handler = CallbackHandler()

# 简单LLM测试链路
llm = ChatOpenAI(model="qwen-turbo")
prompt = ChatPromptTemplate.from_template("回答：{query}")
chain = prompt | llm

# 执行并传入tags/user/session
res = chain.invoke(
    {"query": "测试Langfuse上报"},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_tags": ["mini-test"],
            "langfuse_session_id": "test_sess_01"
        }
    }
)

print(res.content)
langfuse.flush()
print("上报完成，打开Langfuse面板查看trace")