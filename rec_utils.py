import os
import shutil
import json
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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
        user_x = x[:, 0].int()
        item_x = x[:, 1].int()
        
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
        user_x = x[:, 0].int()
        item_x = x[:, 1].int()

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
        ui_x = self.fc(x.float())
        return ui_x


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

        self.kwargs = {
            'num_user': num_user,
            'num_item': num_item,
            'cf_dim': cf_dim,
            'nn_dim_user': nn_dim_user,
            'nn_dim_item': nn_dim_item,
            'nn_dim_nn': nn_dim_nn,
            'item_context_features_in': item_context_features_in,
            'item_context_dim': item_context_dim,
            'user_context_features_in': user_context_features_in,
            'user_context_dim': user_context_dim,
        }
        
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
    

    def forward(self, x, item_context_features_len):
        x_cf = self.cf_module(x[:,:2])
        x_nn = self.nn_module(x[:,:2])
        x_ic = self.item_context_module(x[:,2:item_context_features_len+2])
        x_uc = self.user_context_module(x[:,2+item_context_features_len:])
        
        concat_x = torch.concat((x_cf.unsqueeze(1), x_nn, x_ic, x_uc), 1)

        out = self.fc(concat_x)

        return out.reshape(-1,)


class SimpleNNRecDataset(Dataset):
    def __init__(
        self,
        ui_data: pd.DataFrame,
        item_context: pd.DataFrame,
        user_context: pd.DataFrame,
        ) -> None:
        super().__init__()

        self.ui = ui_data[['user_idx', 'item_idx']].values
        self.u_context = ui_data[['user_idx']].merge(user_context, how='left', on=['user_idx']).drop(['user_idx'], axis=1).values
        self.i_context = ui_data[['item_idx']].merge(item_context, how='left', on=['item_idx']).drop(['item_idx'], axis=1).values
        
        self.labels = ui_data['rating'].values
    
    
    def __len__(self):
        return self.labels.shape[0]


    def __getitem__(self, idx):
        return self.ui[idx, :], self.i_context[idx, :], self.u_context[idx, :], self.labels[idx]


class SimpleNNRec():
    def __init__(
        self,
        user_field: str,
        item_field: str,
        rating_field: str,
        ) -> None:

        self.logdir = 'runs'

        self.user_field = user_field
        self.item_field = item_field
        self.rating_field = rating_field

        self.user_map = None
        self.item_map = None
        
        self.model = None

        self.recommendations = None
    

    def __idx_map(
        self,
        data: pd.DataFrame,
        id: str,
        idx: str,
    ):
        idx_map = data[[id]].drop_duplicates()
        idx_map = idx_map.reset_index().drop(['index'], axis=1)
        idx_map = idx_map.reset_index().rename(columns={'index':idx})

        return idx_map

    
    def __ui_map(
        self,
        data: pd.DataFrame,
        user_field: str,
        item_field: str,
        rating_field: str,
    ):
        ans = data[[user_field, item_field, rating_field]]
        ans = ans.rename(columns={user_field: 'user_id', item_field: 'item_id', rating_field: 'rating'})

        user_map = self.__idx_map(ans, 'user_id', 'user_idx')
        item_map = self.__idx_map(ans, 'item_id', 'item_idx')

        return user_map, item_map
    

    def __map_idx_column(
        self,
        data: pd.DataFrame,
    ):
        ans = data.copy()

        if self.user_field in data.columns:
            ans = ans.rename(columns={self.user_field: 'user_idx'})
            mapping = dict(self.user_map[['user_id', 'user_idx']].values)
            ans['user_idx'] = ans['user_idx'].map(mapping)
        
        if self.item_field in data.columns:
            ans = ans.rename(columns={self.item_field: 'item_idx'})
            mapping = dict(self.item_map[['item_id', 'item_idx']].values)
            ans['item_idx'] = ans['item_idx'].map(mapping)
        
        return ans
    

    def fit(
        self,
        ui_data: pd.DataFrame,
        item_context: pd.DataFrame,
        user_context: pd.DataFrame,
        epochs: int = 5,
        learning_rate: float = 0.001,
        cf_dim: int = 10,
        nn_dim_user: int = 10,
        nn_dim_item: int = 10,
        nn_dim_nn: int = 10,
        test_size=0.2,
        batch_size=64,
    ):

        self.user_map, self.item_map = self.__ui_map(
            data=ui_data,
            user_field=self.user_field,
            item_field=self.item_field,
            rating_field=self.rating_field,
        )

        ui = self.__map_idx_column(ui_data)
        u_context = self.__map_idx_column(user_context)
        i_context = self.__map_idx_column(item_context)

        ui_train, ui_test = train_test_split(ui, test_size=test_size, random_state=42)

        dataset_train = SimpleNNRecDataset(ui_data=ui_train, item_context=i_context, user_context=u_context)
        dataset_test = SimpleNNRecDataset(ui_data=ui_test, item_context=i_context, user_context=u_context)

        dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

        item_context_features_in = item_context.shape[1] - 1
        user_context_features_in = user_context.shape[1] - 1

        self.model = RecModule(
            num_user=len(self.user_map),
            num_item=len(self.item_map),
            cf_dim=cf_dim,
            nn_dim_user=nn_dim_user,
            nn_dim_item=nn_dim_item,
            nn_dim_nn=nn_dim_nn,
            item_context_features_in=item_context_features_in,
            item_context_dim=item_context_features_in,
            user_context_features_in=user_context_features_in,
            user_context_dim=user_context_features_in,
            )
        
        self.__train(
            epochs=epochs,
            train_dataloader=dataloader_train,
            test_dataloader=dataloader_test,
            item_context_features_len=item_context_features_in,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        self.recommendations = self.__fill_recommendations(
            ui_data=ui,
            item_context=i_context,
            user_context=u_context,
        )
            

    def __train(
        self,
        epochs,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        item_context_features_len: int,
        learning_rate: float,
        batch_size: int,
    ) -> None:

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss = nn.L1Loss()

        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)
        
        writer = SummaryWriter(self.logdir)

        for epoch in range(epochs):
            
            loss_train_running = 0
            for batch in train_dataloader:
                X, y = torch.concat((batch[0], batch[1], batch[2]), dim=1), batch[3]

                pred = self.model(X, item_context_features_len=item_context_features_len)
                loss_iter = loss(pred, y)

                optimizer.zero_grad()
                loss_iter.backward()
                optimizer.step()

                loss_train_running += loss_iter.item()

            loss_test_running = 0
            for batch in test_dataloader:
                X, y =torch.concat((batch[0], batch[1], batch[2]), dim=1), batch[3]

                with torch.no_grad():
                    pred = self.model(X, item_context_features_len=item_context_features_len)
                    loss_iter = loss(pred, y)

                loss_test_running += loss_iter.item()

            writer.add_scalars('Loss', {'train': loss_train_running / (len(train_dataloader) * batch_size), 'test': loss_test_running / (len(test_dataloader) * batch_size)}, epoch+1)

        writer.close()


    def save(self, model_name: str) -> None:
        if os.path.exists(f'models/{model_name}'):
            shutil.rmtree(f'models/{model_name}')
        
        os.makedirs(f'models/{model_name}')            

        params = {}

        params['logdir'] = 'runs'
        
        params['user_field'] = self.user_field
        params['item_field'] = self.item_field
        params['rating_field'] = self.rating_field

        with open (f'models/{model_name}/params.json', 'w') as fp:
            json.dump(params, fp)

        self.user_map.to_pickle(f'models/{model_name}/user_map.pkl')
        self.item_map.to_pickle(f'models/{model_name}/item_map.pkl')
        self.recommendations.to_pickle(f'models/{model_name}/recommendations.pkl')

        torch.save([self.model.kwargs, self.model.state_dict()], f'models/{model_name}/model.pth')
    

    def load(self, model_name: str) -> None:
        with open (f'models/{model_name}/params.json', 'r') as fp:
            params = json.load(fp)
        
        self.logdir = params['logdir']

        self.user_field = params['user_field']
        self.item_field = params['item_field']
        self.rating_field = params['rating_field']

        self.user_map = pd.read_pickle(f'models/{model_name}/user_map.pkl')
        self.item_map = pd.read_pickle(f'models/{model_name}/item_map.pkl')
        self.recommendations = pd.read_pickle(f'models/{model_name}/recommendations.pkl')

        model_kwargs, model_state = torch.load(f'models/{model_name}/model.pth')
        
        self.model = RecModule(**model_kwargs)
        self.model.load_state_dict(model_state)
        self.model.eval()
    

    def __fill_recommendations(
        self,
        ui_data: pd.DataFrame,
        item_context: pd.DataFrame,
        user_context: pd.DataFrame,
    ):

        ui_mtrx = ui_data.pivot_table(
            values='rating',
            index='user_idx',
            columns='item_idx',
        )

        ui_mtrx = ui_mtrx.unstack(-1).reorder_levels(['user_idx', 'item_idx'])
        ui_mtrx = pd.DataFrame(ui_mtrx, columns=['rating'])
        ui_mtrx = ui_mtrx[ui_mtrx['rating'].isnull()]
        ui_mtrx = ui_mtrx.reset_index().drop(['rating'], axis=1)

        ui_mtrx = ui_mtrx.merge(item_context, how='left', on=['item_idx'])
        ui_mtrx = ui_mtrx.merge(user_context, how='left', on=['user_idx'])

        with torch.no_grad():
            pred = self.model(torch.tensor(ui_mtrx.values), item_context.shape[1] - 1).tolist()
        
        ui_mtrx['rating'] = pred

        return ui_mtrx
    

    def recommend(self, id, n: int, mapping: dict = None):
        user_idx = self.user_map[self.user_map['user_id'] == id][['user_idx']].iloc[0, 0]

        top_list = self.recommendations[self.recommendations['user_idx'] == user_idx]
        top_list = top_list.merge(self.item_map, how='left', on=['item_idx'])
        top_list = top_list.sort_values(['rating'], ascending=False).head(n)
        top_list = top_list['item_id'].tolist()

        if mapping is not None:
            top_list = list(map(mapping.get, top_list))

        return top_list