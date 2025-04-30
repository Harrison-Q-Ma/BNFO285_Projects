import pandas as pd
from scipy.stats import chi2   


df = pd.read_csv("gene_dnds_trinucleotide_context_summary.csv")


chi2_stat = (df["observed_dNdS"] - df["expected_dNdS"]) ** 2 / df["expected_dNdS"]


df["p_value"] = chi2.sf(chi2_stat, df=1)


df.to_csv("gene_dnds_trinucleotide_context_summary_with_p.csv", index=False)

#print("输出文件已保存为 gene_dnds_trinucleotide_context_summary_with_p.csv")
