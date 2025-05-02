import pandas as pd
import matplotlib.pyplot as plt

# matplotlib‑venn is a small add‑on; install if missing
try:
    from matplotlib_venn import venn2
except ImportError:
    raise ImportError("Please install matplotlib‑venn:  pip install matplotlib-venn")

# ---------------------- 1. load data ----------------------
dndscv_df = pd.read_csv("dndscv_output_qglobal_cv_lt0.1.csv")
our_df    = pd.read_csv("significant_genes_Bonferroni.csv")

# Specify the column that contains gene names
# (change 'gene' to the exact column name in your files if needed)
dndscv_genes = set(dndscv_df["gene_name"].dropna().unique())
our_genes    = set(our_df["gene"].dropna().unique())

# ---------------------- 2. venn diagram -------------------
plt.figure(figsize=(6, 6))
venn2([dndscv_genes, our_genes],
      set_labels=("dNdScv (q<0.1)", "Our method"))
plt.title("Overlap of significant genes")
plt.tight_layout()

# save & show
plt.savefig("venn_dndscv_vs_our.png", dpi=300)
plt.show()
# ---------------------- write gene lists -----------------------
overlap   = sorted(dndscv_genes & our_genes)
only_dnd  = sorted(dndscv_genes - our_genes)
only_ours = sorted(our_genes - dndscv_genes)

pd.DataFrame({"gene": overlap}).to_csv("genes_overlap.csv", index=False)
pd.DataFrame({"gene": only_dnd}).to_csv("genes_only_dndscv.csv", index=False)
pd.DataFrame({"gene": only_ours}).to_csv("genes_only_ourmethod.csv", index=False)

print(f"Overlap: {len(overlap)}  Only dNdScv: {len(only_dnd)}  Only ours: {len(only_ours)}")
print("Files written:\n"
      "  venn_dndscv_vs_our.png\n"
      "  genes_overlap.csv\n"
      "  genes_only_dndscv.csv\n"
      "  genes_only_ourmethod.csv")