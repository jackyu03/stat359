#!/usr/bin/env python
# coding: utf-8
"""
Script for training an MLP-based sentiment classifier on the financial_phrasebank dataset.
"""

# ========== Imports ==========
import numpy as np
import pandas as pd
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg') # disable plotting
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from sklearn.utils.class_weight import compute_class_weight

random_state = 67
torch.manual_seed(random_state)

print("\n========== Loading Dataset ==========")
# ========== Load Dataset ==========
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][0])

print("\n========== Preparing DataFrame ==========")
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")

# Print distribution of sentence lengths
sentence_lengths = data['sentence'].apply(lambda x: len(x.split()))
print("\nSentence length statistics:")
print(sentence_lengths.describe())

# plt.figure(figsize=(10,6))
# plt.hist(sentence_lengths, bins=30, color='skyblue', edgecolor='black')
# plt.title('Distribution of Sentence Lengths')
# plt.xlabel('Sentence Length (words)')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

print("\n========== Generating Embedding ==========")

# load model
path = "./student/Assignment_3/pretrained_models/fasttext-wiki-news-subwords-300.model"

try:
    fasttext_model = KeyedVectors.load(path)
    print('embedding model loaded!')
except Exception as e:
    print(f'failed to load model due to {e}!')

def get_sentence_embedding(sentence, model):
    tokens = simple_preprocess(sentence)
    embs = [model[token] for token in tokens if token in model]
    return np.mean(embs, axis=0) if len(embs) != 0 else np.zeros(300)

data['embedding'] = data['sentence'].apply(lambda x: get_sentence_embedding(x, fasttext_model))

print("\n========== Preparing to Train ==========")

X_numpy = np.vstack(data['embedding'].values)
y_numpy = data['label'].to_numpy()

# 15% for test, 13% for validation, 72% for train
X_temp, X_test, y_temp, y_test = train_test_split(
    X_numpy, y_numpy, test_size=0.15, stratify=y_numpy, random_state=random_state
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=random_state
)

class_weights_array = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('Dataset (train and test) successfully prepared.')

print("\n========== Model training ==========")

class MLPSentiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPSentiment, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out
    
model = MLPSentiment(300, 100, 3)

device = torch.device("mps")
model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)

num_epochs = 20
best_val_f1 = 0.0

# history tracking
train_loss_history, val_loss_history = [], []
train_f1_history, val_f1_history = [], []
train_acc_history, val_acc_history = [], []

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    
    # Training
    model.train()
    running_loss = 0.0
    all_train_preds, all_train_labels = [], []
    
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(y_batch.cpu().numpy())

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
    train_acc = (np.array(all_train_preds) == np.array(all_train_labels)).mean()
    
    train_loss_history.append(epoch_train_loss)
    train_f1_history.append(train_f1)
    train_acc_history.append(train_acc)

    # validation
    model.eval()
    val_loss = 0.0
    all_val_preds, all_val_labels = [], []
    
    with torch.no_grad():
        for X_v, y_v in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
            X_v, y_v = X_v.to(device), y_v.to(device)
            outputs = model(X_v)
            loss = criterion(outputs, y_v)
            
            val_loss += loss.item() * X_v.size(0)
            _, preds = torch.max(outputs, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(y_v.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    val_acc = (np.array(all_val_preds) == np.array(all_val_labels)).mean()
    
    val_loss_history.append(epoch_val_loss)
    val_f1_history.append(val_f1)
    val_acc_history.append(val_acc)

    # log
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f} | Train F1: {train_f1:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

    # checkpointing
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        os.makedirs("outputs", exist_ok=True)
        torch.save(model.state_dict(), 'outputs/best_mlp_model.pth')
        print(f'>>> Saved new best model (Val F1: {best_val_f1:.4f})')

# ========== Plotting ==========
plt.figure(figsize=(12, 15))
plt.subplot(3, 1, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Loss Curve')
plt.legend(); plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(train_f1_history, label='Train F1')
plt.plot(val_f1_history, label='Val F1')
plt.title('F1 Macro Score Curve')
plt.legend(); plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/mlp_f1_learning_curves.png')
# plt.show()

# Save accuracy plot separately
plt.figure(figsize=(8, 6))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/mlp_accuracy_learning_curve.png')
# # plt.show()

model.load_state_dict(torch.load('outputs/best_mlp_model.pth'))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for X_t, y_t in test_loader:
        X_t, y_t = X_t.to(device), y_t.to(device)
        outputs = model(X_t)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_t.cpu().numpy())

class_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.savefig('outputs/mlp_confusion_matrix.png')
# plt.show()