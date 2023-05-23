import torch
import transformer as t

#
# Hyper parameter
#
block_size =  8 # number of consecutive characters to predict from
batch_size = 32 # mini-batch size
embedding_size = 32 # size of the embedding vectors
max_iterations = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

#
# Prepare training data
#

# read in file
with open('TrainingData/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocabulary_size = len(chars)

# Build encoder and decoder
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# create data tensors
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
training_data = data[:n]
validation_data = data[n:]

#
# mini-batch creation
#
def get_batch(split):
    data = training_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

#
# evaluation
#
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = t.LanguageModel(vocabulary_size, block_size, embedding_size, embedding_size, device)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iterations):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_tokens=500)[0].tolist()))

