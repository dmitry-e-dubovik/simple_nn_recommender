import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class CFModule(nn.Module):
    def __init__(
        self,
        num_user: int,
        num_item: int,
        dim: int = 10,
        ):
        super().__init__()

        self.user_emb = nn.Embedding(
            num_embeddings=num_user,
            embedding_dim=dim,
        )

        self.item_emb = nn.Embedding(
            num_embeddings=num_item,
            embedding_dim=dim,
        )
    

    def forward(self, x):
        user_x = x[:, 0]
        item_x = x[:, 1]

        user_x = self.user_emb(user_x)
        item_x = self.item_emb(item_x)

        out = torch.einsum('ij,ij->i', user_x, item_x)

        return out


class NNModule(nn.Module):
    def __init__(
        self,
        num_user: int,
        num_item: int,
        dim_user: int = 10,
        dim_item: int = 10,
        dim_nn: int = 10,
        ) -> None:
        super().__init__()

        self.user_emb = nn.Embedding(
            num_embeddings=num_user,
            embedding_dim=dim_user,
        )

        self.item_emb = nn.Embedding(
            num_embeddings=num_item,
            embedding_dim=dim_item,
        )

        self.fc = nn.Linear(
            in_features=(dim_user+dim_item),
            out_features=dim_nn,
            )

    
    def forward(self, x):
        user_x = x[:, 0]
        item_x = x[:, 1]

        user_x = self.user_emb(user_x)
        item_x = self.item_emb(item_x)

        concat_x = torch.cat((user_x, item_x), 1)

        out = self.fc(concat_x)

        return out


class ContextModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        dim: int = 10,
        ) -> None:
        super().__init__()

        self.fc = nn.Linear(
            in_features=in_features,
            out_features=dim,
        )
    

    def forward(self, x):
        if_x = self.fc(x)
        return if_x


class RecModule(nn.Module):
    def __init__(
        self,
        num_user: int,
        num_item: int,
        cf_dim: int,
        nn_dim_user: int,
        nn_dim_item: int,
        nn_dim_nn: int,
        item_context_features_in: int,
        item_context_dim: int,
        user_context_features_in: int,
        user_context_dim: int,

        ) -> None:
        super().__init__()

        self.cf_module = CFModule(
            num_user=num_user,
            num_item=num_item,
            dim=cf_dim,
        )

        self.nn_module = NNModule(
            num_user=num_user,
            num_item=num_item,
            dim_user=nn_dim_user,
            dim_item=nn_dim_item,
            dim_nn=nn_dim_nn,
        )

        self.item_context_module = ContextModule(
            in_features=item_context_features_in,
            dim=item_context_dim,
        )

        self.user_context_module = ContextModule(
            in_features=user_context_features_in,
            dim=user_context_dim,
        )

        self.fc = nn.Linear(
            in_features=(1+nn_dim_nn+item_context_dim+user_context_dim),
            out_features=1,
        )
    

    def forward(self, x, item_context_features_in, user_context_features_in):
        x_cf = self.cf_module(x[:,:2])
        x_nn = self.nn_module(x[:,:2])
        x_ic = self.item_context_module(x[:,2:item_context_features_in+2])
        x_uc = self.user_context_module(x[:,2+item_context_features_in:])

        concat_x = torch.concat((x_cf, x_nn, x_ic, x_uc), 1)

        out = self.fc(concat_x)

        return out


class SimpleNNRec():
    def __init__(self) -> None:
        pass
    

    
    

    
    

    