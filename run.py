from loader import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import argparse
import torch
import os
from datetime import datetime
from model import DistMult
import torch.optim as optim
import torch.nn.functional as F

def parsing_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Distmult'
    )
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)

    return parser

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def init_dicts(path):
    nentity = 0
    nrelation = 0
    with open(os.path.join(path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            nentity += 1

    with open(os.path.join(path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            nrelation += 1

    return entity2id, relation2id, nentity, nrelation


def save_model(model):
    timestamp = datetime.utcnow()
    filename = f"{timestamp}_result.pth"
    torch.save(model.state_dict(), filename)


def main(args):
    data_path = './data/FB15k-237'

    entity2id, relation2id, nentity, nrelation = init_dicts(data_path)
    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    all_triples = train_triples + valid_triples + test_triples

    train_dataloader = DataLoader(TrainDataset(train_triples, nentity),
                                  shuffle=True,
                                  drop_last=True,
                                  batch_size=500,
                                  collate_fn=TrainDataset.collate_fn
                                  )
    valid_dataloader = DataLoader(TestDataset(train_triples,  all_triples, nentity),
                               shuffle=False,
                               batch_size=500,
                               collate_fn=TestDataset.collate_fn
                               )
    test_dataset = DataLoader(TestDataset(test_triples, all_triples, nentity),
                               shuffle=False,
                               batch_size=500,
                               collate_fn=TestDataset.collate_fn
                               )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DistMult(nentity, nrelation, 500)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    running_loss = 0.0
    for positive_sample, negative_heads, negative_tails in train_dataloader:
        model.forward((positive_sample, (negative_heads, negative_tails)))


    # lose = F.margin_ranking_loss()

if __name__ == '__main__':
    args = parsing_args()
    main(args)
