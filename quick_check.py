from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# 加载配置
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# 打印核对是否读取到完整密钥
print("读取到的API KEY：", api_key)
print("读取到的地址：", base_url)

llm = ChatOpenAI(
    model="qwen3-max",
    api_key=api_key,
    base_url=base_url
)
res = llm.invoke("测试一句话")
print("成功返回：", res.content)