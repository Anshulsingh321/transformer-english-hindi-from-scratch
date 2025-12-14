from collections import Counter

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3
}

def build_vocab(file_path, max_vocab_size):
    """
    Builds a word-level vocabulary from a text file.
    Each line is treated as one sentence.
    """
    counter = Counter()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            counter.update(words)

    vocab = dict(SPECIAL_TOKENS)

    for word, _ in counter.most_common(max_vocab_size - len(vocab)):
        vocab[word] = len(vocab)

    return vocab


def encode_sentence(sentence, vocab, max_len=50):
    """
    Converts a sentence into a list of token IDs.
    """
    tokens = sentence.strip().split()

    encoded = [vocab["<sos>"]]

    for token in tokens:
        encoded.append(vocab.get(token, vocab["<unk>"]))

    encoded.append(vocab["<eos>"])

    # Pad or truncate
    if len(encoded) < max_len:
        encoded += [vocab["<pad>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return encoded
