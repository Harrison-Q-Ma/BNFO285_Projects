import pandas as pd
import numpy as np
import os
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

FILES = [
    "TCGA.HNSC.expression_log_tumor.txt",
    "TCGA.HNSC.expression_log_tumor_top1000.txt",
    "TCGA.HNSC.expression_log_tumor_top2000.txt",
    "TCGA.HNSC.expression_log_tumor_top5000.txt",
    "TCGA.HNSC.expression_log_tumor_top10000.txt"
]

dims_to_try = [6]          # 目标维度
transpose_data = False     # 你的表已经是 (样本 × 基因)，故保持 False

def run_mds_for_file(infile, dims):
    print(f"\n=== 处理文件: {infile} ===")
    
    # ① 读入，不设 index_col，保留 patient_id & sample_id 两列
    df_raw = pd.read_csv(infile, sep="\t")
    print("初始数据形状:", df_raw.shape)
    
    # ② 记录 sample_id & patient_id，再提取表达矩阵
    sample_ids   = df_raw["sample_id"].values
    patient_ids  = df_raw["patient_id"].values
    expr_df      = df_raw.drop(columns=["patient_id", "sample_id"])
    
    # ③（可选）标准化
    X_scaled = StandardScaler().fit_transform(expr_df.values)
    
    # ④ MDS
    for d in dims:
        print(f"  -> MDS 降到 {d} 维 ...")
        mds = MDS(n_components=d, dissimilarity="euclidean", random_state=42, n_jobs=-1)
        X_mds = mds.fit_transform(X_scaled)
        
        # ⑤ 生成输出表
        out_cols = {
            "patient_id": patient_ids,
            "sample_id":  sample_ids
        }
        for i in range(d):
            out_cols[f"MDS{i+1}"] = X_mds[:, i]
        
        df_out = pd.DataFrame(out_cols)
        
        # ⑥ 保存
        base, _ = os.path.splitext(infile)
        outname = f"{base}_MDS_{d}D.tsv"
        df_out.to_csv(outname, sep="\t", index=False)
        print(f"    已输出: {outname}")

# 批量处理
for f in FILES:
    run_mds_for_file(f, dims_to_try)

print("\n全部处理完毕。")
