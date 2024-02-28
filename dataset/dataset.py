from collections import Counter

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_data(filepaths, tokenizer=lambda s: s.strip().split()):
    raw_in_iter = iter(open(filepaths[0], encoding="utf8"))
    raw_out_iter = iter(open(filepaths[1], encoding="utf8"))
    return list(zip(map(tokenizer, raw_in_iter), map(tokenizer, raw_out_iter)))


class Vocab(object):
    UNK = "<unk>"  # 0
    PAD = "<pad>"  # 1
    BOS = "<bos>"  # 2
    EOS = "<eos>"  # 3

    def __init__(self, data=None, min_freq=1):
        counter = Counter()
        for lines in data:
            counter.update(lines[0])
            counter.update(lines[1])
        self.word2idx = {Vocab.UNK: 0, Vocab.PAD: 1, Vocab.BOS: 2, Vocab.EOS: 3}
        self.idx2word = {0: Vocab.UNK, 1: Vocab.PAD, 2: Vocab.BOS, 3: Vocab.EOS}
        idx = 4
        for word, freq in counter.items():
            if freq >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, word):
        return self.word2idx.get(word, 0)

    def __call__(self, word):
        if not isinstance(word, (list, tuple)):
            return self[word]
        return [self[w] for w in word]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple, np.ndarray, torch.Tensor)):
            return self.idx2word[int(indices)]
        return [self.idx2word[int(i)] for i in indices]


def pad_sequence(sequences, batch_first=False, padding_value=0):
    max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        padding_content = [padding_value] * (max_len - tensor.size(0))
        tensor = torch.cat([tensor, torch.tensor(padding_content)], dim=0)
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        out_tensors = out_tensors.transpose(0, 1)
    return out_tensors.long()


class CoupletsDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.PAD_IDX = self.vocab[self.vocab.PAD]
        self.BOS_IDX = self.vocab[self.vocab.BOS]
        self.EOS_IDX = self.vocab[self.vocab.EOS]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_in, raw_out = self.data[index]
        in_tensor_ = torch.LongTensor(self.vocab(raw_in))
        out_tensor_ = torch.LongTensor(self.vocab(raw_out))
        return in_tensor_, out_tensor_

    def collate_fn(self, batch):
        in_batch, out_batch = [], []
        for in_, out_ in batch:
            in_batch.append(in_)
            out_ = torch.cat(
                [
                    torch.LongTensor([self.BOS_IDX]),
                    out_,
                    torch.LongTensor([self.EOS_IDX]),
                ],
                dim=0,
            )
            out_batch.append(out_)
        in_batch = pad_sequence(in_batch, True, self.PAD_IDX)
        out_batch = pad_sequence(out_batch, True, self.PAD_IDX)
        return in_batch, out_batch

    def get_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


if __name__ == "__main__":
    data = load_data(
        ["../data/fixed_couplets_in.txt", "../data/fixed_couplets_out.txt"]
    )
    print("数据长度: ", len(data))

    vocab = Vocab(data)
    print("词典长度: ", len(vocab))

    dataset = CoupletsDataset(data, vocab)
    batch_size = 32
    loader = dataset.get_loader(batch_size=batch_size, shuffle=True, num_workers=0)
    in_, out_ = next(iter(loader))
    print(in_.shape, out_.shape)
