import torch
from torch.utils.data import Dataset

from utils.tokenizer import encode_sentence


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_len=50):
        self.src_sentences = open(src_file, encoding="utf-8").read().splitlines()
        self.tgt_sentences = open(tgt_file, encoding="utf-8").read().splitlines()

        assert len(self.src_sentences) == len(self.tgt_sentences)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = encode_sentence(
            self.src_sentences[idx],
            self.src_vocab,
            self.max_len
        )
        tgt = encode_sentence(
            self.tgt_sentences[idx],
            self.tgt_vocab,
            self.max_len
        )

        return torch.tensor(src), torch.tensor(tgt)
