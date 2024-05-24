from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, raw_triples, nentity, negative_size=256):
        self.raw_triples = raw_triples  # 3 x n
        self.head_entities = set()
        self.tail_entities = set()
        self.negative_size = negative_size
        self.nentity = nentity
        self.true_head, self.true_tails = self.__get_true_head(raw_triples)

    def __len__(self):
        return len(self.raw_triples)

    def __get_true_head(self, triples):
        true_head = {}
        true_tail = {}
        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = set()
            true_tail[(head, relation)].add(tail)

            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = set()
            true_head[(relation, tail)].add(head)
            self.head_entities.add(head)
            self.tail_entities.add(tail)

        return true_head, true_tail

    def __getitem__(self, index):

        triple = self.raw_triples[index]  # head relation tail

        negative_heads = self.tail_entities - self.true_tails[
            (triple[0], triple[1])]
        negative_tails = self.tail_entities - self.true_head[
            (triple[1], triple[2])]

        tails_tensor = torch.LongTensor(list(negative_tails))
        random_indices = torch.randperm(len(tails_tensor))[:self.negative_size]

        negative_tails = tails_tensor[random_indices]

        head_tensor = torch.LongTensor(list(negative_heads))
        random_indices = torch.randperm(len(head_tensor))[:self.negative_size]

        negative_heads = head_tensor[random_indices]

        triple = torch.LongTensor(list(triple))

        return triple, negative_heads, negative_tails

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data],
                                      dim=0)
        negative_heads = torch.stack([_[1] for _ in data],
                                     dim=0)
        negative_tails = torch.stack([_[2] for _ in data],
                                     dim=0)

        return positive_sample, negative_heads, negative_tails


class TestDataset(Dataset):
    def __init__(self, test_triples, all_triples, nentity):
        self.test_triples = test_triples
        self.all_triples = set(all_triples)
        self.nentity = nentity

    def __len__(self):
        return len(self.test_triples)

    def __getitem__(self, idx):
        triples = self.test_triples[idx]
        head = [(0, rand_head) if triples not in self.all_triples
                else (-1, triples[0]) for rand_head in range(self.nentity)]
        head[triples[0]] = (0, triples[0])

        tails = [(0, rand_trails) if triples not in self.all_triples
                 else (-1, triples[2]) for rand_trails in range(self.nentity)]
        tails[triples[2]] = (0, triples[2])

        triples = torch.LongTensor(triples)

        head = torch.LongTensor(head)
        tails = torch.LongTensor(tails)

        head_filter_bias = head[:, 0].float()
        head_negative_sample = head[:, 1]

        tails_filter_bias = tails[:, 0].float()
        tails_negative_sample = tails[:, 1]

        return triples, head_negative_sample, tails_negative_sample, head_filter_bias, tails_filter_bias

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        head_negative_sample = torch.stack([_[1] for _ in data], dim=0)
        tails_negative_sample = torch.stack([_[2] for _ in data], dim=0)
        head_filter_bias = torch.stack([_[3] for _ in data], dim=0)
        tails_filter_bias = torch.stack([_[4] for _ in data], dim=0)

        return positive_sample, head_negative_sample, tails_negative_sample, head_filter_bias, tails_filter_bias
