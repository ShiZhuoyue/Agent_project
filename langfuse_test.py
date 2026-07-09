from dotenv import load_dotenv
import os
from langfuse import Langfuse

# 关键：加载本地.env文件
load_dotenv()

# 自动读取环境变量初始化
langfuse = Langfuse()
print(langfuse.auth_check()) # 输出True代表成功