import torch
import json
import tokenizer

def initializeTrainingData(topic, mergeSteps = 0, debug = False):
    # read in file
    with open(f'{topic}/{topic}.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # tokenize
    decoder, tokens = tokenizer.bytePairEncoding(text, mergeSteps, debug)
    
    # save tokens
    with open(f'{topic}/{topic}.tokens', 'w') as file:
        json.dump(tokens, file)

    # save vocabulary
    dictionary = {str(k): v for k, v in decoder.items()}
    with open(f'{topic}/vocabulary.json', 'w') as file:
        json.dump(dictionary, file)

    return decoder, tokens

def loadTrainingData(topic):
    with open(f'{topic}/{topic}.tokens', 'r') as file:
        tokens = json.load(file)
    with open(f'{topic}/vocabulary.json', 'r') as file:
        decoder = json.load(file)
    decoder = {int(k): v for k, v in decoder.items()}    
    
    return decoder, tokens

def createDataTensors(tokens):
    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9*len(data))
    training_data = data[:n]
    validation_data = data[n:]
    return {'train': training_data, 'val': validation_data}

def get_batch(data, sample_size, batch_size, device):
    ix = torch.randint(len(data) - sample_size, (batch_size, ))
    x = torch.stack([data[i:i+sample_size] for i in ix])
    y = torch.stack([data[i+1:i+sample_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss(model, data, iterations, sample_size, batch_size, device):
    out = {}
    training = model.training
    if training:
        model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(iterations)
        for k in range(iterations):
            x, y = get_batch(data[split], sample_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    if training:
        model.train()
    return out

