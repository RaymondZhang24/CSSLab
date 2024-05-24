import torch

class DistMult(torch.nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim):
        super(DistMult, self).__init__()
        self.entity_embedding = torch.nn.Embedding(nentity, hidden_dim)
        self.relation_embedding = torch.nn.Embedding(nrelation, hidden_dim)
        torch.nn.init.xavier_uniform_(self.entity_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, sample):
        positive, negative = sample
        head, relation, tail = positive[:, 0], positive[:, 1], positive[:, 2]




