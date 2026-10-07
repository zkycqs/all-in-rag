# -*- coding: utf-8 -*-
"""
加载 LlamaIndex 持久化的索引，做相似性搜索

对应 03_llamaindex_vector.py 保存的 ./llamaindex_index_store
运行方式（必须在 code/C3 目录下）：
    cd /workspaces/all-in-rag/code/C3
    python 03b_llamaindex_load_search.py
"""
import warnings
warnings.filterwarnings("ignore")

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


PERSIST_DIR = "./llamaindex_index_store"


# ============================================================
# 1. 配置嵌入模型
# ============================================================
# ⚠️ 必须和"建索引时"用的是【同一个模型】！
#    否则：查询向量和库里向量的"坐标系"不一样，相似度算出来毫无意义
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")


# ============================================================
# 2. 从磁盘加载存储
# ============================================================
print(f"正在从 {PERSIST_DIR} 加载索引 ...")
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)


# ============================================================
# 3. 用存储重建索引对象
# ============================================================
index = load_index_from_storage(storage_context)
print(f"加载成功！索引中共有 {len(index.docstore.docs)} 个节点\n")


# ============================================================
# 4. 创建【检索器】
# ============================================================
# 用 as_retriever 而不是 as_query_engine：
#   as_retriever      → 只做相似性搜索，返回原始文本块（不调 LLM）
#   as_query_engine   → 检索 + 调 LLM 生成答案（需要配置 Settings.llm）
retriever = index.as_retriever(similarity_top_k=1)


# ============================================================
# 5. 执行相似性搜索
# ============================================================
queries = [
    "LlamaIndex 是什么？",
    "张三是谁？",
    "它提供了什么工具？",
]

for q in queries:
    print("=" * 64)
    print(f"查询: {q}")
    print("=" * 64)

    nodes = retriever.retrieve(q)          # ← 核心：检索

    for i, item in enumerate(nodes, 1):
        print(f"  [{i}] 相似度: {item.score:.4f}")
        print(f"      内容: {item.node.text}")

    print()
