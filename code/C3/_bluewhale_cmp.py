#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比：文本从「datawhale开源组织的logo」换成「blue whale」后，相似度的变化"""
import warnings
warnings.filterwarnings("ignore")

import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(
    model_name_bge="BAAI/bge-base-en-v1.5",
    model_weight="../../models/bge/Visualized_base_en_v1.5.pth",
)
model.eval()

IMG1 = "../../data/C3/imgs/datawhale01.png"
IMG2 = "../../data/C3/imgs/datawhale02.png"

with torch.no_grad():
    # 图像向量只算一次，两轮复用
    i1 = model.encode(image=IMG1)
    i2 = model.encode(image=IMG2)

    results = {}
    for tag, txt in [
        ("原始文本", "datawhale开源组织的logo"),
        ("新文本",   "blue whale"),
    ]:
        t  = model.encode(text=txt)
        m1 = model.encode(image=IMG1, text=txt)
        m2 = model.encode(image=IMG2, text=txt)

        results[tag] = {
            "纯图像 vs 纯图像":       (i1 @ i2.T).item(),
            "图文结合1 vs 纯图像":    (i1 @ m1.T).item(),
            "图文结合1 vs 纯文本":    (t @ m1.T).item(),
            "图文结合1 vs 图文结合2": (m1 @ m2.T).item(),
        }

    print("=" * 68)
    print(f"{'对比项':<26}{'原始文本':>16}{'新文本':>16}{'差值':>10}")
    print("-" * 68)
    for key in results["原始文本"]:
        a = results["原始文本"][key]
        b = results["新文本"][key]
        print(f"{key:<26}{a:>16.4f}{b:>16.4f}{b - a:>+10.4f}")
    print("=" * 68)
    print("DONE")
