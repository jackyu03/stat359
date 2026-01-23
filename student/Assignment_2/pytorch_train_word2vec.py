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
LEARNING_RATE = 2.5
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

        # initialize embedding weights to improve gradient flow
        nn.init.xavier_uniform_(self.center_embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)

    def emb_dot_product(self, center_emb, context_emb):
        center_emb_unsqueezed = center_emb.unsqueeze(1) # shape: (batch, 1, dim)
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
dataset = SkipGramDataset('/Users/JackYu_1/Desktop/STAT_359/stat359/student/Assignment_2/processed_data.pkl')
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
model.to(device)
criterion = nn.BCEWithLogitsLoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, 
    start_factor=1.0, 
    end_factor=0.001, 
    total_iters=EPOCHS
)

def make_targets(pos_dot, neg_dot, device):
    pos_targets = torch.ones(pos_dot.shape).to(device) # score 1, shape (batch, 1) 
    neg_targets = torch.zeros(neg_dot.shape).to(device) # score 0, shape (batch, neg_samples)
    return pos_targets, neg_targets

# Training loop
# using a M-series GPU for multinomial sampling will lead to crazy RAM usage. CPU is more efficient
neg_sampling_dist = neg_sampling_dist.to('cpu')

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for center, context in tqdm(dataloader):
        center, context = center.to(device), context.to(device)
        actual_batch_size = center.size(0)
        
        # sampling negative cases
        neg_indices = torch.multinomial(
            neg_sampling_dist,
            actual_batch_size * NEGATIVE_SAMPLES,
            replacement=True
        ).view(actual_batch_size, NEGATIVE_SAMPLES).to(device)

        '''# collision handling
        for i in range(5): # try maximum 5 times of collision handling
            # generate mask to detect collisions with both center or context
            collision = (neg_indices == context.unsqueeze(1)) | (neg_indices == center.unsqueeze(1))

            if not collision.any():
                break

            neg_indices[collision] = torch.randint(
                0, dataset.vocab_size,
                (collision.sum(),)
            ).to(device)'''

        # forward
        pos_dot, neg_dot = model.forward(center, context, neg_indices)

        # compute loss
        pos_targets, neg_targets = make_targets(pos_dot, neg_dot, device)
        pos_loss = criterion(pos_dot, pos_targets) # (batch, 1)
        neg_loss = criterion(neg_dot, neg_targets) # (batch, negative_samples)
        loss = (pos_loss.sum() + neg_loss.sum()) / actual_batch_size

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Completed epoch {epoch} with avg loss {epoch_loss/len(dataloader):.4f}!")
    scheduler.step()


# Save embeddings and mappings
embeddings = model.center_embedding.weight.detach().cpu().numpy()
with open('/Users/JackYu_1/Desktop/STAT_359/stat359/student/Assignment_2/word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': dataset.data['word2idx'], 'idx2word': dataset.data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
