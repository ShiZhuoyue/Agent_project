import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 读取.env里的本地模型路径
load_dotenv()
model_path = os.getenv("EMBEDDING_MODEL_PATH")

# 加载本地离线模型，强制CPU
try:
    model = SentenceTransformer(model_path, device="cpu")
    print("✅ 模型加载成功！无报错")

    # 测试编码向量
    test_text = "论文检索、RAG嵌入测试"
    vec = model.encode(test_text, normalize_embeddings=True)
    print(f"✅ 向量生成完成，向量维度：{len(vec)}")
    print(f"✅ 前5位向量数值：{vec[:5]}")

except Exception as e:
    print("❌ 加载失败，错误信息：")
    print(e)