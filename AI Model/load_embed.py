import io
import torch
import numpy as np


def init_embeddings(vocab_size, embed_dim, unif):
    return np.random.uniform(-unif, unif, (vocab_size, embed_dim))


class Embedding_reader:

    @staticmethod
    def from_txt(filename, vocab, unif=0.25):
        with io.open(filename, "r", encoding='utf-8', errors='ignore') as fh:
            for i, line in enumerate(fh):
                line = line.rstrip("\n ")
                values = line.split(" ")

                if i == 0:
                    if len(values) == 2:  # FastText
                        weight = init_embeddings(len(vocab), int(values[1]), unif)
                        continue
                    else:  # Glove
                        weight = init_embeddings(len(vocab), len(values[1:]), unif)
                word = values[0]
                if word in vocab:
                    vec = np.asarray(values[1:], dtype=np.float32)
                    weight[vocab[word]] = vec

        if '<PAD>' in vocab:
            weight[vocab['<PAD>']] = 0.0

        embeddings = torch.nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight.data = torch.from_numpy(weight).float()

        return embeddings, weight.shape[1]
