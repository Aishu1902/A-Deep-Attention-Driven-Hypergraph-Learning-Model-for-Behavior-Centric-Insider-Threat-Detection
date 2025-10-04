import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import os
from transformers import BertModel, BertTokenizer  
from torch.optim import AdamW                    
from sklearn.preprocessing import LabelEncoder
from early_stop_v1 import EarlyStopping
from tqdm import trange
import time

epochs = 20
seq_len = 5
train_round = 3
bert_model_name = 'bert-base-uncased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained(bert_model_name)

def encode_sequence(row):
   
    tokens = row.astype(str).tolist()
    return " ".join(tokens)

def process_data(data):
    data.fillna(0, inplace=True)
    available_tokens = data.columns[4:]
    seq_len = min(len(available_tokens), 74)
    x_columns = data.columns[4:4+seq_len]

    sequences = data[x_columns].apply(encode_sequence, axis=1)
    labels = data['label'].tolist()
    
    inputs = tokenizer(sequences.tolist(), padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
    labels = torch.tensor(labels)
    return inputs, labels

class BERTClassifier(nn.Module):
    def __init__(self, model_name=bert_model_name, output_size=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_token)
        return logits, cls_token

if __name__ == '__main__':
    # Paths
    data_dir = r'C:\Users\Ahin\Desktop\insider threat\sample _data'
    result_dir = r'C:\Users\Ahin\Desktop\insider threat\bert_feature_extraction_result'
    os.makedirs(result_dir, exist_ok=True)

    data = pd.read_csv(os.path.join(data_dir, '1-data-test-combine-sequence.csv'))
    df_manual = pd.read_csv(os.path.join(data_dir, '1-data-test-combine.csv'))

    start = time.time()
    perform = pd.DataFrame()
    inputs, labels = process_data(data)
    
    dataset = Data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    dataloader = Data.DataLoader(dataset, batch_size=16, shuffle=True)

    for r in trange(train_round):
        model = BERTClassifier().to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        best_model_path = os.path.join(result_dir, f'bert_round_{r}_best.pt')
        early_stopping = EarlyStopping(save_path=best_model_path, verbose=True, patience=3, delta=0.001)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in dataloader:
                input_ids, attn_mask, lbls = [x.to(device) for x in batch]
                optimizer.zero_grad()
                logits, _ = model(input_ids, attn_mask)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            early_stopping(avg_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        perform = pd.concat([perform, pd.DataFrame([{'round': r, 'min_loss': avg_loss}])], ignore_index=True)

    perform.to_csv(os.path.join(result_dir, f'bert_performance.csv'))

    # Feature extraction using best model
    best_model_path = os.path.join(result_dir, f'bert_round_{perform["min_loss"].idxmin()}_best.pt')
    model = BERTClassifier().to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    with torch.no_grad():
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        _, features = model(input_ids, attention_mask)

    df_feature = pd.DataFrame(features.cpu().numpy())
    df_save = pd.concat([df_manual, df_feature], axis=1)
    df_save.to_csv(os.path.join(result_dir, f'1-data-test-manual+bert_features.csv'), index=False)

    end = time.time()
    print("Time used:", end - start)
