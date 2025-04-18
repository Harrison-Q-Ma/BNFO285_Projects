import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator      # 肘部查找用

# ---------- 1. 读入 ----------
df = pd.read_csv("TCGA.HNSC.expression_log_tumor.txt", sep="\t")

# 通常文件前两列是 patient_id / sample_id，后面才是基因表达
meta_cols = {"patient_id", "sample_id"}
df = df.drop(columns=[c for c in df.columns if c in meta_cols])

# 如果行数远小于列数，说明现在行=基因、列=样本，需要转置：
if df.shape[0] > df.shape[1]:
    df = df.T
    # 转置后，如果第一行/列又变成了字符串，同理再剔掉
    df = df.apply(pd.to_numeric, errors="coerce")

# 仅保留数值列
df_numeric = df.select_dtypes(include=[np.number])
X = StandardScaler().fit_transform(df_numeric.values)
print("最终用于 MDS 的矩阵维度:", X.shape)

# ---------- 2. 按多维度跑 MDS ----------
dims = list(range(2, 21))
stress = []
for d in dims:
    mds = MDS(n_components=d, dissimilarity="euclidean", random_state=42, n_jobs=-1)
    mds.fit(X)
    stress.append(mds.stress_)
    print(f"{d:2d}D  stress={mds.stress_:,.2f}")

# ---------- 3. 肘部法自动选维 ----------
knee = KneeLocator(dims, stress, curve="convex", direction="decreasing")
best_dim = knee.knee or dims[int(np.argmin(stress))]
print(f"\n肘部推荐维度 = {best_dim}  (stress={stress[dims.index(best_dim)]:.2f})")

plt.figure()
plt.plot(dims, stress, "o-")
plt.axvline(best_dim, ls="--")
plt.xlabel("维度数")
plt.ylabel("Stress")
plt.title("MDS Stress 曲线")
plt.tight_layout()
plt.show()

# ---------- 4. 最佳维度重新拟合 ----------
mds_best = MDS(n_components=best_dim, dissimilarity="euclidean",
               random_state=42, n_jobs=-1)
X_best = mds_best.fit_transform(X)
print("最佳嵌入形状:", X_best.shape)

if best_dim == 2:
    plt.figure(figsize=(6, 5))
    plt.scatter(X_best[:, 0], X_best[:, 1], s=30)
    plt.xlabel("MDS1"); plt.ylabel("MDS2")
    plt.title(f"2‑D MDS (stress={mds_best.stress_:,.2f})")
    plt.tight_layout(); plt.show()
else:
    print("最佳维度不是 2，可考虑 3‑D 可视化。")
