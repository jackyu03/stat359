import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive


# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        
        self.word2idx = self.data['word2idx']
        self.vocab_size = len(self.word2idx)
        self.word_freq = self.data['counter']
        self.skipgram_df = self.data['skipgram_df']

        self.center_v = torch.tensor(self.skipgram_df['center'].values, dtype=torch.long)
        self.context_v = torch.tensor(self.skipgram_df['context'].values, dtype=torch.long)

    def __len__(self):
        return len(self.center_v)

    
    def __getitem__(self, index):
        return(self.center_v[index], self.context_v[index])

    
# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.center_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

    def emb_dot_product(self, center_emb, context_emb):
        center_emb_unsqueezed = torch.unsqueeze(center_emb) # shape: (batch, 1, dim)
        if context_emb.dim() == 2:
            context_emb = context_emb.unsqueeze(1) # shape: (batch, 1, dim)

        context_emb_tp = context_emb.transpose(1, 2) # shape: (batch, neg_samples, dim)
        
        return torch.bmm(center_emb_unsqueezed, context_emb_tp).squeeze(1)

    def forward(self, center, pos_context, neg_contexts):
        center_emb = self.center_embedding(center)
        pos_context_emb = self.context_embedding(pos_context)
        neg_context_embs = self.context_embedding(neg_contexts)

        pos_dot = self.emb_dot_product(center_emb, pos_context_emb)
        neg_dot = self.emb_dot_product(center_emb, neg_context_embs)

        return pos_dot, neg_dot


# Dataset and DataLoader
dataset = SkipGramDataset('processed_data.pkl')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Precompute negative sampling distribution below
counts = torch.zeros(dataset.vocab_size)
for word, count in dataset.word_freq.items():
    counts[dataset.word2idx[word]] = count

neg_sampling_freq = counts ** 0.75
neg_sampling_dist = neg_sampling_freq / torch.sum(neg_sampling_freq)


# Device selection: MPS > CPU (using mac)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')


# Model, Loss, Optimizer
model = Word2Vec(dataset.vocab_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def make_targets(pos_dot, neg_dot, device):
    pos_targets = torch.ones(pos_dot.shape).to(device) # score 1, shape (batch, 1) 
    neg_targets = torch.zeros(neg_dot.shape).to(device) # score 0, shape (batch, neg_samples)
    return pos_targets, neg_targets

# Training loop
neg_sampling_dist = neg_sampling_dist.to(device)
model.to(device)

for epoch in range(EPOCHS):
    for center, context in tqdm(dataloader):
        center, context = center.to(device), context.to(device)
        
        neg_indices = 


# Save embeddings and mappings
# embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
