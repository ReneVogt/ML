import torch as T
import torch.nn.functional as F

@T.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    totalloss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        labels = labels - 1
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        totalloss += loss.item()

        _, predicted = T.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return totalloss/len(loader), accuracy

def train(model, loader, device, optimizer) -> tuple[float, float]:
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        labels = labels - 1
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = T.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return train_loss/len(loader), accuracy
