import pandas as pd

# Load the CSV file
df = pd.read_csv('/home/sagemaker-user/multimodal-mimic/mimic3-benchmarks/in-hospital-mortality/test_listfile.csv')

# Select the first 147 rows
df_first_147 = df.iloc[:147]

# Save the new CSV file with the first 147 rows
df_first_147.to_csv('1percent_test_listfile.csv', index=False)
