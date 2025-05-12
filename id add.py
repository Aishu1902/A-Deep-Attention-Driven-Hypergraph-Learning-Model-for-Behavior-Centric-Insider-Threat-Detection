import pandas as pd

# Path to your CSV file
file_path = r"C:\Users\Ahin\Desktop\insider threat\bert_feature_extraction_result\new_dataset_bert.csv"

# Read the CSV
df = pd.read_csv(file_path)

# Insert the 'id' column at the front
df.insert(0, 'id', range(1, len(df) + 1))

# Save back to the same file (or change filename to avoid overwrite)
df.to_csv(file_path, index=False)

print("Column added successfully.")
