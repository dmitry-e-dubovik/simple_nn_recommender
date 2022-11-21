import torch
from torch import nn

u = torch.tensor([0, 1, 2, 0])
i = torch.tensor([1, 2, 0, 1])

u_emb = nn.Embedding(
    num_embeddings=3,
    embedding_dim=2,
    )

i_emb = nn.Embedding(
    num_embeddings=3,
    embedding_dim=3,
    )

fc = nn.Linear(
    in_features=5,
    out_features=3,
    )

ux = u_emb(u)
ix = i_emb(i)

cx = torch.concat((ux, ix), 1)
print(fc(cx))