import torch

#
# Module base class
#
class Module:
    def __init__(self):
        self._trainingMode = False

    @property
    def training(self):
        return self._trainingMode
    @training.setter
    def training(self, value):
        self._trainingMode = value

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None
    
    def parameters(self):
        return []    

#
# Linear layer
#
class Linear(Module):

    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
#
# Batch normalization layer
#
class BatchNorm1d(Module):

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.mean
            xvar = self.var
        
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self._trainingMode:
            with torch.no_grad():
                self.mean = (1 - self.momentum) * self.mean + self.momentum * xmean
                self.var = (1 - self.momentum) * self.var + self.momentum * xvar

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
#
# Tanh layer
#
class Tanh(Module):
    def __init__(self):
        super().__init__()
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

#
# Embedding layer
#
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]

#
# Flattening layer
#
class Flatten(Module):
    def __init__(self):
        super().__init__()
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    def parameters(self):
        return []

#
# Network base class
#    
class Network(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        for p in self.parameters():
            p.requires_grad = True
    @Module.training.setter    
    def training(self, value):
        self._trainingMode = value
        for layer in self.layers:
            layer.training = value
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def update(self, learning_rate = 0.1):
        for p in self.parameters():
            p.data += -learning_rate * p.grad

#
# Sequential module
#    
class Sequential(Network):
    def __init__(self, layers):
        super().__init__(layers)
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
