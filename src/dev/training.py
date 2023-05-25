import torch
import transformer as t

#
# Hyper parameter
#
batch_size      = 64    # mini-batch size
max_iterations  = 5000
eval_interval   = 500
learning_rate   = 3e-4
eval_iters      = 200

torch.manual_seed(1337)

#
# Prepare training data
#

# read in file
with open('TrainingData/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
t.vocabulary_size = len(chars)

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
    ix = torch.randint(len(data) - t.sample_size, (batch_size, ))
    x = torch.stack([data[i:i+t.sample_size] for i in ix])
    y = torch.stack([data[i+1:i+t.sample_size+1] for i in ix])
    x,y = x.to(t.device), y.to(t.device)
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

model = t.LanguageModel()
print(f'Number of parameters: {sum(p.nelement() for p in model.parameters())}')
m = model.to(t.device)
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

context = torch.zeros((1,1), dtype=torch.long, device=t.device)
print(decode(m.generate(context, max_tokens=500)[0].tolist()))

