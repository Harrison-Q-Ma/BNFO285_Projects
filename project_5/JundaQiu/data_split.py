import pandas as pd
from sklearn.model_selection import train_test_split
import os
# change to the WSL-mounted path
file_path = '/mnt/c/Users/HUIPU/Desktop/Team_2_HNSC/proj5/merged_feature_table.csv'
assert os.path.exists(file_path), f"File not found: {file_path}"

df = pd.read_csv(file_path)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

train_path = '/mnt/c/Users/HUIPU/Desktop/Team_2_HNSC/proj5/train_split_80.csv'
test_path  = '/mnt/c/Users/HUIPU/Desktop/Team_2_HNSC/proj5/test_split_20.csv'

train_df.to_csv(train_path, index=False)
test_df .to_csv(test_path,  index=False)
