import pandas as pd
import numpy as np
import os
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

FILES = [

    "TCGA.HNSC.expression_log_tumor_top2000.txt"

]

dims_to_try = [6]               # 降到 6 维
transpose_data = False          # 已是 (样本 × 基因)，不转置

def run_mds_for_file(infile, dims):
    print(f"\n=== 处理文件: {infile} ===")
    df = pd.read_csv(infile, sep="\t")
    print("初始形状:", df.shape)

    # -------- 1. 保证 sample_id 存在 --------
    if "sample_id" not in df.columns:
        raise ValueError(f"{infile} 缺少 sample_id 列")

    # -------- 2. 若无 patient_id，自动生成 --------
    if "patient_id" not in df.columns:
        df["patient_id"] = df["sample_id"].str.split("-").str[:3].str.join("-")

    # -------- 3. 提取表达矩阵 --------
    expr_df = df.drop(columns=["patient_id", "sample_id"])
    X = StandardScaler().fit_transform(expr_df.values)

    for d in dims:
        print(f"  -> MDS {d}D …")
        X_mds = MDS(n_components=d, dissimilarity="euclidean",
                    random_state=42, n_jobs=-1).fit_transform(X)

        out = pd.DataFrame({
            "patient_id": df["patient_id"],
            "sample_id":  df["sample_id"],
            **{f"MDS{i+1}": X_mds[:, i] for i in range(d)}
        })

        outname = f"{os.path.splitext(infile)[0]}_MDS_{d}D.tsv"
        out.to_csv(outname, sep="\t", index=False)
        print(f"    已写出: {outname}  ({len(out)} 行)")

for f in FILES:
    run_mds_for_file(f, dims_to_try)

print("\n✅ 全部完成。")

import pandas as pd

file_path = "TCGA.HNSC.expression_log_tumor_top2000_MDS_6D.tsv"

df = pd.read_csv(file_path, sep="\t")
if "patient_id" in df.columns:
    df = df.drop(columns="patient_id")
df.to_csv(file_path, sep="\t", index=False)
