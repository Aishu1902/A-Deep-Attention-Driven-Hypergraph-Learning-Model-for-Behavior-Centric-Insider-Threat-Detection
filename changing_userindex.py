import pandas as pd

# Load the file with headers
#df = pd.read_csv(r'C:\Users\Ahin\Desktop\insider threat\bert_feature_extraction_result\normalized_dataset_bert.csv')
df = pd.read_csv(r'C:\Users\Ahin\Desktop\insider threat\final_transformer_feature_extraction_result\1-data-test-manual+transformer_seq74.csv')

# Replace the values in the 3rd column (index 2) with sequential integers starting from 1
df.iloc[:, 2] = range(1, len(df) + 1)

# Save the modified DataFrame to a new CSV file
df.to_csv(r'C:\Users\Ahin\Desktop\insider threat\final_transformer_feature_extraction_result\userindex_dataset_transformer.csv', index=False)




            
            
