import torch
import torch.nn as nn
from torch.optim import Adam

from model.transformer import Transformer


def train_step(model, optimizer, criterion, src, tgt):
    """
    One training step using teacher forcing
    """

    model.train()

    # 1 Shift target
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    # 2 Forward pass
    logits = model(src, tgt_input)

    # 3 Reshape for loss
    logits = logits.reshape(-1, logits.size(-1))
    tgt_output = tgt_output.reshape(-1)

    # 4 Compute loss
    loss = criterion(logits, tgt_output)

    # 5 Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
