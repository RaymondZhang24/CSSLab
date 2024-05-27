import torch

class DistMult(torch.nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim):
        super(DistMult, self).__init__()
        self.entity_embedding = torch.nn.Embedding(nentity, hidden_dim)
        self.relation_embedding = torch.nn.Embedding(nrelation, hidden_dim)
        torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, sample):
        positive, negative = sample
        head, relation, tail = positive[:, 0], positive[:, 1], positive[:, 2]

        head_embedding, relation_embedding, tail_embedding = self.entity_embedding(head).unsqueeze(1), self.relation_embedding(relation).unsqueeze(1), self.entity_embedding(tail).unsqueeze(1)
        true_score = self.score(head_embedding, relation_embedding, tail_embedding)

        negative_heads, negative_tails = negative[0], negative[1]
        negative_heads_embedding, negative_tails_embedding = self.entity_embedding(negative_heads), self.entity_embedding(negative_tails)

        heads_score = self.score(negative_heads_embedding, relation_embedding, tail_embedding)
        tail_score = self.score(head_embedding, relation_embedding, negative_tails_embedding)

        return true_score, heads_score, tail_score
    def score(self, heads, relations, tails):
        return (heads * relations * tails).sum(dim=2)



