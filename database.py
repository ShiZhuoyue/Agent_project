import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings  # 修复: langchain_community已废弃，改用langchain_huggingface
from langchain_chroma import Chroma

load_dotenv()  # 顶部提前加载环境变量

DATA_DIR = "./research_papers"
DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db_storage")

for directory in (DATA_DIR, DB_DIR):
    if not os.path.exists(directory):
        os.makedirs(directory)

@lru_cache(maxsize=1)
def get_vector_db():

    # 读取本地BGE模型路径，兜底移除MiniLM
    model_path = os.getenv("EMBEDDING_MODEL_PATH")
    # 不再默认MiniLM，没有配置直接抛错，避免混用
    if not model_path:
        raise Exception("请在.env配置 EMBEDDING_MODEL_PATH 本地BGE模型路径")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        # 新增CPU离线运行关键参数
        model_kwargs={
            "device": "cpu",
            "trust_remote_code": True
        },
        # BGE强制开启归一化，提升检索精度
        encode_kwargs={"normalize_embeddings": True}
    )
    # 读取全新的向量库目录，隔离旧MiniLM数据
    DB_DIR = os.getenv("VECTOR_DB_DIR")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return db, embeddings