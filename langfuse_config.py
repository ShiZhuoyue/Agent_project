# langfuse_config.py
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

# 加载密钥环境变量
load_dotenv()

# 全局单例 Langfuse 客户端，整个项目只初始化一次
langfuse_client = Langfuse()

# 全局回调句柄，所有Graph共用
langfuse_callback = CallbackHandler()

# 连通校验，启动时打印状态
if __name__ == "__main__":
    print("Langfuse连通结果：", langfuse_client.auth_check())