#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from loader import TestDataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistMult(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim):
        super(DistMult, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim

        self.entity_dim =  hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = torch.nn.Embedding(nentity, hidden_dim).to(DEVICE)
        self.relation_embedding = torch.nn.Embedding(nrelation, hidden_dim).to(DEVICE)
        torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, sample):
        positive, negative = sample
        head, relation, tail = positive[:, 0].to(DEVICE), positive[:, 1].to(DEVICE), positive[:, 2].to(DEVICE)

        head_embedding, relation_embedding, tail_embedding = self.entity_embedding(head).unsqueeze(1), self.relation_embedding(relation).unsqueeze(1), self.entity_embedding(tail).unsqueeze(1)
        true_score = self.score(head_embedding, relation_embedding, tail_embedding)

        negative_heads, negative_tails = negative[0], negative[1]
        negative_heads_embedding, negative_tails_embedding = self.entity_embedding(negative_heads).to(DEVICE), self.entity_embedding(negative_tails).to(DEVICE)

        heads_score = self.score(negative_heads_embedding, relation_embedding, tail_embedding)
        tail_score = self.score(head_embedding, relation_embedding, negative_tails_embedding)

        return true_score, heads_score, tail_score

    def score(self, head, relation, tail):
        score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    @staticmethod
    def train_step(model, train_dataloader, optimizer, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        for positive_sample, negative_heads, negative_tails in train_dataloader:
            positive_sample = positive_sample.cuda()
            negative_heads = negative_heads.cuda()
            negative_tails = negative_tails.cuda()

            true_score, negative_heads_score, negative_tails_score = model((positive_sample, (negative_heads, negative_tails)))

            head_negative_score = F.logsigmoid(-negative_heads_score).mean(dim = 1)
            tail_negative_score = F.logsigmoid(-negative_tails_score).mean(dim = 1)

            positive_score = F.logsigmoid(true_score).squeeze(dim = 1)


            positive_sample_loss = - positive_score.mean()
            head_sample_loss = - head_negative_score.mean()
            tail_sample_loss = - tail_negative_score.mean()

            loss1 = (positive_sample_loss + head_sample_loss)/2
            loss2 = (positive_sample_loss + tail_sample_loss)/2
            loss = (loss1 + loss2) /2

            regularization = args.regularization * (
                model.entity_embedding.weight.norm(p = 3)**3 +
                model.relation_embedding.weight.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}


            loss.backward()

            optimizer.step()

            log = {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'head_sample_loss': head_sample_loss.item(),
                'tail_sample_loss': tail_sample_loss.item(),
                'loss': loss.item()
            }

            return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        test_dataloader = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )
        logs = []

        step = 0
        total_steps = sum([len(test_dataloader)])

        with torch.no_grad():
            for positive_sample, head_negative_sample, tails_negative_sample, head_filter_bias, tails_filter_bias in test_dataloader:

                batch_size = positive_sample.size(0)

                true_score, negative_heads_score, negative_tails_score = model(
                    (positive_sample.cuda(), (head_negative_sample.cuda(), tails_negative_sample.cuda())))


                negative_heads_score += head_filter_bias.cuda()
                negative_tails_score += tails_filter_bias.cuda()

                neg_head_rank = torch.argsort(negative_heads_score, dim = 1, descending=True)
                neg_tail_rank = torch.argsort(negative_tails_score, dim = 1, descending=True)

                pos_head = positive_sample[:, 0]
                pos_tail = positive_sample[:, 2]

                for i in range(batch_size):
                    #Notice that argsort is not ranking
                    h_ranking = (neg_head_rank[i, :] == pos_head[i]).nonzero()
                    t_ranking = (neg_tail_rank[i, :] == pos_tail[i]).nonzero()
                    assert h_ranking.size(0) == 1
                    assert t_ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    h_ranking = 1 + h_ranking.item()
                    t_ranking = 1 + t_ranking.item()

                    logs.append({
                        'MRR': (1/h_ranking + 1/t_ranking) / 2,
                        'H_HITS@10': 1.0 if h_ranking <= 10 else 0.0,
                        'T_HITS@10': 1.0 if t_ranking <= 10 else 0.0,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        return metrics

