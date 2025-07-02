import pandas as pd

# Load the CSV
df = pd.read_csv('your_file.csv')

# List of columns to normalize
columns_to_normalize = ['BERT F1', 'Tokens', 'Energy (kWh)', 'CPU (%)', 'Memory (MB)', 'Time (s)']

# Normalize each column using Min-Max normalization
for col in columns_to_normalize:
    min_val = df[col].min()
    max_val = df[col].max()
    if min_val != max_val:
        df[col] = (df[col] - min_val) / (max_val - min_val)
    else:
        # Avoid division by zero if all values are the same
        df[col] = 0

# Save the normalized data to a new CSV
df.to_csv('normalized_output.csv', index=False)
