import torch
import torch.nn as nn
from torch.nn import functional as F

vocabulary_size             = 65
attention_heads_per_block   = 4
attention_blocks            = 4
sample_size                 = 128   # number of consecutive characters to predict from
embedding_size              = 128   # size of the embedding vectors
dropout                     = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#
# Head
#
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = embedding_size//attention_heads_per_block
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(sample_size, sample_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        affinities = q @ k.transpose(-2,-1) * C**-0.5
        affinities = affinities.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)
        affinities = self.dropout(affinities)

        v = self.value(x)
        return affinities @ v

#
# MultiHeadAttention
#
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(attention_heads_per_block)])
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.proj(out)
        out = self.dropout(out)        
        return out
    
#
# FeedForward
#
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4*embedding_size),
            nn.ReLU(),
            nn.Linear(4*embedding_size, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

#
# Block
#
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding_size)
        self.attention = MultiHeadAttention()
        self.layernorm2 = nn.LayerNorm(embedding_size)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        return x

#
# LanguageModel
#
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding = nn.Embedding(sample_size, embedding_size)
        self.blocks = nn.Sequential(*[Block() for _ in range(attention_blocks)])
        self.layernorm = nn.LayerNorm(embedding_size)
        self.linearHead = nn.Linear(embedding_size, vocabulary_size)
        self.device = device

    def forward(self, idx, targets=None):
        B,T = idx.shape
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(torch.arange(T, device=self.device))
        x = token_embeddings + position_embeddings        
        x = self.blocks(x)
        x = self.layernorm(x)
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
            idx_crop = idx[:, -sample_size:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx