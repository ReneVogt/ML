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
# MultiHeadAttention
#
class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, num_heads, embedding_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, embedding_size, embedding_size//num_heads) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim =-1)        
        return self.proj(out)
    
#
# FeedForward
#
class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4*embedding_size),
            nn.ReLU(),
            nn.Linear(4*embedding_size, embedding_size)
        )

    def forward(self, x):
        return self.net(x)

#
# Block
#
class Block(nn.Module):
    def __init__(self, block_size, embedding_size, head_count):
        super().__init__()
        self.attention = MultiHeadAttention(block_size, head_count, embedding_size)
        self.ffwd = FeedForward(embedding_size)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.ffwd(x)
        return x

#
# LanguageModel
#
class LanguageModel(nn.Module):
    def __init__(self, vocabulary_size, block_size, embedding_size, head_count, device = 'cpu'):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(
            Block(block_size, embedding_size, head_count),
            Block(block_size, embedding_size, head_count),
            Block(block_size, embedding_size, head_count),
            Block(block_size, embedding_size, head_count)
        )
        self.linearHead = nn.Linear(embedding_size, vocabulary_size)
        self.device = device

    def forward(self, idx, targets=None):
        B,T = idx.shape
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(torch.arange(T, device=self.device))
        x = token_embeddings + position_embeddings        
        x = self.blocks(x)
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