import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read CSV file with headers
df_data = pd.read_csv(r'C:\Users\Ahin\Desktop\insider threat\bert_feature_extraction_result\1-data-test-manual+bert_features.csv', header=0)

# Select the last 5 columns for normalization
cols_to_normalize = df_data.columns[-5:]  # Last 5 column names

# Apply Min-Max normalization
scaler = MinMaxScaler()
df_data[cols_to_normalize] = scaler.fit_transform(df_data[cols_to_normalize])

# Save the normalized dataset
df_data.to_csv(r'C:\Users\Ahin\Desktop\insider threat\bert_feature_extraction_result\normalized_dataset_bert.csv', index=False)

# Print first few rows to check
print(df_data.head())
