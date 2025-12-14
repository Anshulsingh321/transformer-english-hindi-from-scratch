import torch


def create_padding_mask(seq, pad_token_id=0):
    """
    seq: (batch_size, seq_len)
    returns: (batch_size, 1, 1, seq_len)
    """
    return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(seq_len):
    """
    seq_len: int
    returns: (seq_len, seq_len)
    """
    return torch.tril(torch.ones(seq_len, seq_len)).bool()
