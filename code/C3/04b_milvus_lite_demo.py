# -*- coding: utf-8 -*-
"""
Milvus Lite 版示例 —— 在无法使用 Docker 的环境下替代 Milvus Standalone

【为什么要用 Lite】
当前 Codespaces 网络无法访问 quay.io / Docker Hub，公开镜像加速器也
只镜像 library/* 官方库，拉不到 milvusdb/milvus、minio/minio 这些镜像，
因此 Docker 版 Standalone 无法部署。

【Lite 与 Standalone 的关系】
- API 完全一致（都是 pymilvus 的 MilvusClient）
- 唯一区别：连接方式
    Standalone: MilvusClient("http://localhost:19530")
    Lite:       MilvusClient("./milvus_lite_demo.db")
- Lite 是官方提供的轻量版，适合开发/学习，上限约 100 万向量

运行方式（必须在 code/C3 目录下或任意目录均可，因为用的是相对./）：
    python 04b_milvus_lite_demo.py
"""
import warnings
import os
warnings.filterwarnings("ignore")

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer


DB_PATH = "./milvus_lite_demo.db"
COLLECTION = "demo_collection"


# ============================================================
# 1. 连接 —— Lite 直接给一个本地文件路径即可
# ============================================================
# 如果换成 Docker 版 Standalone，这里写 "http://localhost:19530"
client = MilvusClient(DB_PATH)
print(f"已连接 Milvus (Lite 模式): {DB_PATH}")


# ============================================================
# 2. 准备文本 + 用嵌入模型生成向量
# ============================================================
texts = [
    "张三是法外狂徒",
    "Milvus 是一个开源的向量数据库，支持海量向量相似性检索。",
    "RAG 是检索增强生成，用检索到的文档来增强大模型的回答。",
    "北京是中国的首都，有故宫和长城。",
]

print("正在加载嵌入模型 ...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

vectors = model.encode(texts, normalize_embeddings=True)
dim = len(vectors[0])
print(f"向量维度: {dim}")


# ============================================================
# 3. 建集合（相当于关系库的"建表"）
# ============================================================
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)
    print(f"已删除旧集合: {COLLECTION}")

client.create_collection(collection_name=COLLECTION, dimension=dim)
print(f"集合创建成功: {COLLECTION}")


# ============================================================
# 4. 插入数据
# ============================================================
data = [
    {"id": i, "vector": vectors[i].tolist(), "text": texts[i]}
    for i in range(len(texts))
]
client.insert(collection_name=COLLECTION, data=data)
print(f"已插入 {len(data)} 条数据")

stats = client.get_collection_stats(COLLECTION)
print(f"集合内实体数: {stats.get('row_count')}")


# ============================================================
# 5. 相似性搜索
# ============================================================
queries = [
    "谁在违法？",
    "向量数据库是什么？",
    "RAG 是什么意思？",
]

for q in queries:
    qvec = model.encode([q], normalize_embeddings=True)[0].tolist()
    results = client.search(
        collection_name=COLLECTION,
        data=[qvec],
        limit=2,
        output_fields=["text"],
    )

    print()
    print("=" * 64)
    print(f"查询: {q}")
    print("=" * 64)
    for hits in results:
        for h in hits:
            print(f"  相似度 {h['distance']:.4f}  →  {h['entity']['text']}")


print()
print("DONE")
