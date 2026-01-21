import pandas as pd

# File paths
file1 = "Restaurants_Train_V3.csv"
file2 = "testV2.csv"
file3 = "trainV2.csv"

# Read CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Concatenate rows
merged_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save output
merged_df.to_csv("merged.csv", index=False)

print("CSV files merged successfully (row-wise).")
