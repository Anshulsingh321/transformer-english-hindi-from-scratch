import torch
from utils.tokenizer import encode_sentence


def greedy_decode(model, src_sentence, src_vocab, tgt_vocab, max_len=50, device="cpu"):
    model.eval()

    inv_tgt_vocab = {idx: word for word, idx in tgt_vocab.items()}

    src_tensor = torch.tensor(
        [encode_sentence(src_sentence, src_vocab, max_len)],
        device=device
    )

    tgt_ids = [tgt_vocab["<sos>"]]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids], device=device)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)

        next_token = output[0, -1].argmax().item()
        tgt_ids.append(next_token)

        if next_token == tgt_vocab["<eos>"]:
            break

    translated = [
        inv_tgt_vocab.get(token, "<unk>")
        for token in tgt_ids
        if token not in [tgt_vocab["<sos>"], tgt_vocab["<eos>"], tgt_vocab["<pad>"]]
    ]

    return " ".join(translated)
