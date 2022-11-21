import os
import shutil
import json

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class MFNNRecommender():
    def __init__(
        self,
        data: pd.DataFrame = None,
        user_field: str = None,
        item_field: str = None,
        rating_field: str = None,
        onload: bool = False,
    ) -> None:

        self.logdir = 'runs'
        
        self.user_field = user_field
        self.item_field = item_field
        self.rating_field = rating_field

        if not onload:            
            self.data, self.user_map, self.item_map = self.__initial_preprocessing(
                data=data,
                user_field=user_field,
                item_field=item_field,
                rating_field=rating_field,
            )
            
        else:
            self.data = None
            self.user_map = None
            self.item_map = None

        self.dim = None
        self.model = None
        self.optimizer = None
        self.loss = None

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

    
    def __initial_preprocessing(
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

        ans = ans.merge(user_map, how='left', on=['user_id'])
        ans = ans.merge(item_map, how='left', on=['item_id'])

        ans = ans[['user_idx', 'item_idx', 'rating']]
        
        return ans, user_map, item_map

    
    class UIDataset(Dataset):
        def __init__(self, data: pd.DataFrame) -> None:
            super().__init__()

            self.features = data[['user_idx', 'item_idx']].values
            self.labels = data['rating'].values
        
        
        def __len__(self):
            return self.labels.shape[0]
    

        def __getitem__(self, idx):
            return self.features[idx, :], self.labels[idx]


    class ModelEmbNN(nn.Module):
        def __init__(
            self,
            num_user: int,
            num_item: int,
            dim=10,
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
    

    def fit(
        self,
        dim: int,
        epochs: int,
        test_size=0.2,
        batch_size=64,
    ) -> None:
        
        self.dim = dim

        df_train, df_test = train_test_split(self.data, test_size=test_size, random_state=42)

        train_dataset = self.UIDataset(df_train)
        test_dataset = self.UIDataset(df_test)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        self.model = self.ModelEmbNN(num_user=len(self.user_map), num_item=len(self.item_map), dim=dim)
        
        self.__train(
            epochs=epochs,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            learning_rate=0.001,
            batch_size=64
        )

        self.recommendations = self.__fill_recommendations()
    

    def __train(
        self,
        epochs,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        learning_rate: float,
        batch_size: int,
    ) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = nn.L1Loss()

        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)
        
        writer = SummaryWriter(self.logdir)

        for epoch in range(epochs):
            
            loss_train_running = 0
            for batch in train_dataloader:
                X, y = batch[0], batch[1]

                pred = self.model(X)
                loss_iter = self.loss(pred, y)

                self.optimizer.zero_grad()
                loss_iter.backward()
                self.optimizer.step()

                loss_train_running += loss_iter.item()

            loss_test_running = 0
            for batch in test_dataloader:
                X, y = batch[0], batch[1]

                with torch.no_grad():
                    pred = self.model(X)
                    loss_iter = self.loss(pred, y)

                loss_test_running += loss_iter.item()

            writer.add_scalars('Loss', {'train': loss_train_running / (len(train_dataloader) * batch_size), 'test': loss_test_running / (len(test_dataloader) * batch_size)}, epoch+1)

        writer.close()
    

    def __fill_recommendations(self):

        ui_mtrx = self.data.pivot_table(
            values='rating',
            index='user_idx',
            columns='item_idx',
        )

        ui_mtrx = ui_mtrx.unstack(-1).reorder_levels(['user_idx', 'item_idx'])
        ui_mtrx = pd.DataFrame(ui_mtrx, columns=['rating'])
        ui_mtrx = ui_mtrx[ui_mtrx['rating'].isnull()]
        ui_mtrx = ui_mtrx.reset_index().drop(['rating'], axis=1)

        with torch.no_grad():
            pred = self.model(torch.tensor(ui_mtrx.values)).tolist()
        
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
    

    def save(self, model_name: str) -> None:
        if os.path.exists(f'models/{model_name}'):
            shutil.rmtree(f'models/{model_name}')
        
        os.makedirs(f'models/{model_name}')            

        params = {}

        params['logdir'] = 'runs'
        
        params['user_field'] = self.user_field
        params['item_field'] = self.item_field
        params['rating_field'] = self.rating_field
        params['dim'] = self.dim

        with open (f'models/{model_name}/params.json', 'w') as fp:
            json.dump(params, fp)

        self.data.to_parquet(f'models/{model_name}/data.parquet.gzip')
        self.user_map.to_parquet(f'models/{model_name}/user_map.parquet.gzip')
        self.item_map.to_parquet(f'models/{model_name}/item_map.parquet.gzip')
        self.recommendations.to_parquet(f'models/{model_name}/recommendations.parquet.gzip')

        torch.save(self.model.state_dict(), f'models/{model_name}/model.pth')
    

    def load(self, model_name: str) -> None:
        with open (f'models/{model_name}/params.json', 'r') as fp:
            params = json.load(fp)
        
        self.logdir = params['logdir']

        self.user_field = params['user_field']
        self.item_field = params['item_field']
        self.rating_field = params['rating_field']
        self.dim = params['dim']

        self.data = pd.read_parquet(f'models/{model_name}/data.parquet.gzip')
        self.user_map = pd.read_parquet(f'models/{model_name}/user_map.parquet.gzip')
        self.item_map = pd.read_parquet(f'models/{model_name}/item_map.parquet.gzip')
        self.recommendations = pd.read_parquet(f'models/{model_name}/recommendations.parquet.gzip')

        self.model = self.ModelEmbNN(num_user=len(self.user_map), num_item=len(self.item_map), dim=self.dim)
        self.model.load_state_dict(torch.load(f'models/{model_name}/model.pth'))
        self.model.eval()
    
    