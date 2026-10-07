# -*- coding: utf-8 -*-
"""
多模态图文检索 —— Milvus Lite 版

【相对教程原版 04_multi_milvus.py 的改动】
1. MILVUS_URI: "http://localhost:19530"  →  "./milvus_multimodal.db"
   （Lite 用一个本地文件当数据库，不需要 Docker 服务）
2. 删掉 Image.open(...).show()  —— 服务器无图形界面，弹窗会报错
其余代码逻辑完全保持不变。

运行方式（必须在 code/C3 目录下）：
    cd /workspaces/all-in-rag/code/C3
    python 04b_multi_milvus_lite.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

# ============================================================
# 1. 初始化设置
# ============================================================
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_demo"

# ★ 唯一的连接改动：Lite 用本地文件
MILVUS_URI = "./milvus_multimodal.db"


# ============================================================
# 2. 工具类与函数（保持原样）
# ============================================================
class Encoder:
    """编码器类，用于将图像和文本编码为向量。"""

    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def visualize_results(query_image_path: str, retrieved_images: list,
                      img_height: int = 300, img_width: int = 300,
                      row_count: int = 3) -> np.ndarray:
    """从检索到的图像列表创建一个全景图用于可视化。"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10,
                                        cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(
        bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width

        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height,
                        start_col:start_col + img_width] = bordered_retrieved

        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return np.hstack([query_display_area, panoramic_image])


# ============================================================
# 3. 初始化客户端
# ============================================================
print("--> 正在初始化编码器和 Milvus 客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)
print(f"    连接方式: {MILVUS_URI}  (Lite 本地文件模式)")


# ============================================================
# 4. 创建 Collection
# ============================================================
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

image_list = sorted(glob(os.path.join(DATA_DIR, "dragon", "*.png")))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")

dim = len(encoder.encode_image(image_list[0]))
print(f"向量维度: {dim}，待入库图片: {len(image_list)} 张")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]

schema = CollectionSchema(fields, description="多模态图文检索")
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")


# ============================================================
# 5. 准备并插入数据
# ============================================================
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path)
    data_to_insert.append({"vector": vector, "image_path": image_path})

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")


# ============================================================
# 6. 创建索引
# ============================================================
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256},
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")

milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")


# ============================================================
# 7. 执行多模态检索
# ============================================================
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"
print(f"    查询图: {os.path.basename(query_image_path)}")
print(f"    查询文: {query_text}")

query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    output_fields=["image_path"],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}},
)[0]

retrieved_images = []
print("\n检索结果:")
for i, hit in enumerate(search_results):
    fname = os.path.basename(hit["entity"]["image_path"])
    print(f"  Top {i+1}: ID={hit['id']}, 余弦相似度={hit['distance']:.4f}, 文件={fname}")
    retrieved_images.append(hit["entity"]["image_path"])


# ============================================================
# 8. 可视化与清理
# ============================================================
print(f"\n--> 正在可视化结果并清理资源")
if not retrieved_images:
    print("没有检索到任何图像。")
else:
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到: {combined_image_path}")
    # 原版的 Image.open(...).show() 需要图形界面，服务器上会报错，这里去掉

milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")
print("\nDONE")
