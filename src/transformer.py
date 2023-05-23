import torch
import torch.nn as nn
from torch.nn import functional as F

#
# Head
#
class Head(nn.Module):
    def __init__(self, block_size, embedding_size, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        affinities = q @ k.transpose(-2,-1) * C**-0.5
        affinities = affinities.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)

        v = self.value(x)
        return affinities @ v

#
# LanguageModel
#
class LanguageModel(nn.Module):
    def __init__(self, vocabulary_size, block_size, embedding_size, head_size, device = 'cpu'):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)
        self.attention_head = Head(block_size, embedding_size, head_size)
        self.linearHead = nn.Linear(embedding_size, vocabulary_size)
        self.device = device

    def forward(self, idx, targets=None):
        B,T = idx.shape
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(torch.arange(T, device=self.device))
        x = token_embeddings + position_embeddings
        x = self.attention_head(x)
        logits = self.linearHead(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            # take last block_size tokens
            idx_crop = idx[:, -self.block_size:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx