#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def filter_expression_data(top_file, expr_file, out_file):
    """
    从表达矩阵 expr_file 中，只保留 patient_id, sample_id,
    以及 top_file 第一列中列出的基因（若能在表达矩阵列中找到）。
    最终输出到 out_file。
    """
    # 1) 读取 top 文件，只取第 1 列作为基因名
    top_df = pd.read_csv(top_file, sep="\t", header=None)
    # 转成字符串，以防止空值或数字类型
    top_genes = top_df.iloc[:, 0].astype(str).tolist()

    # 2) 读取表达矩阵
    expr_df = pd.read_csv(expr_file, sep="\t")

    # 3) 构建要保留的列
    keep_cols = ["patient_id", "sample_id"]
    missing = []
    for g in top_genes:
        if g in expr_df.columns:
            keep_cols.append(g)
        else:
            missing.append(g)

    # 可选：如果想知道哪些基因找不到，可以打印出来
    if missing:
        print(f"[{top_file}] 找不到的基因数: {len(missing)}，示例：{missing[:10]}...")
    else:
        print(f"[{top_file}] 所有基因都能找到匹配列。")

    # 4) 保留列并输出
    filtered_df = expr_df[keep_cols]
    filtered_df.to_csv(out_file, sep="\t", index=False)
    print(f"[{top_file}] => 已输出筛选结果到: {out_file} (保留列数: {filtered_df.shape[1]})\n")

def main():
    expr_file = "TCGA.HNSC.expression_log_tumor.txt"

    # 依次处理 1000/2000/5000/10000
    tasks = [
        ("top1000genesByVariance.txt",   "TCGA.HNSC.expression_log_tumor_top1000.txt"),
        ("top2000genesByVariance.txt",   "TCGA.HNSC.expression_log_tumor_top2000.txt"),
        ("top5000genesByVariance.txt",   "TCGA.HNSC.expression_log_tumor_top5000.txt"),
        ("top10000genesByVariance.txt",  "TCGA.HNSC.expression_log_tumor_top10000.txt"),
    ]

    for top_file, out_file in tasks:
        print(f"=== 正在处理: {top_file} ===")
        filter_expression_data(top_file, expr_file, out_file)

if __name__ == "__main__":
    main()
