import math
import matplotlib.pyplot as plt
import torch
from training import createStateTensor as createStateTensor
from connect4 import Connect4Board
import datetime

def plot_images(tensor):
    if not tensor.ndim==3:
        raise Exception("The tensor should have 3 dimensions")
    
    num_kernels = tensor.shape[0]
    num_cols = int(math.sqrt(num_kernels))
    num_rows = (num_kernels + num_cols - 1) // num_cols
    
    fig = plt.figure(figsize=(num_cols, num_rows))
    
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)
        ax1.imshow(tensor[i, :, :].detach().numpy(), cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def render(board : Connect4Board):
    state = createStateTensor(board).squeeze(0)
    img = torch.flip(state, [1])
    plot_images(img)

