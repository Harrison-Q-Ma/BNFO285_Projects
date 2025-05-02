import pandas as pd

# Load dndscv results
df = pd.read_csv("dndscv_output.csv")

# Filter rows with qglobal_cv < 0.1
filtered = df[df["qglobal_cv"] < 0.1]

# Save or inspect
filtered.to_csv("dndscv_output_qglobal_cv_lt0.1.csv", index=False)
print(filtered.head())

