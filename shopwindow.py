import pandas as pd
import numpy as np
import rec_utils


# import data for mf
df_ratings = pd.read_csv('data/ratings.csv')
df_ratings['dt'] = pd.to_datetime(df_ratings['timestamp'], unit='s')
df_ratings = df_ratings.drop(['timestamp'], axis=1)

# import id-title mapping
item_mapping = pd.read_csv('data/movies.csv')
item_mapping = item_mapping[['movieId', 'title']]
item_mapping = dict(item_mapping.values)

# create dataset for item context feature
df_movies = pd.read_csv('data/movies.csv')
item_context = df_movies.copy()
genres_list = item_context['genres'].str.split('|').explode().unique().tolist()
genres_list.remove('(no genres listed)')

for el in genres_list:
    item_context[el] = np.where(item_context['genres'].str.contains(el), 1, 0)

item_context['year'] = item_context['title'].str.split('(').str[1].str[:4]
item_context['year'] = pd.to_numeric(item_context['year'], errors='coerce')
item_context['old'] = np.where(item_context['year'] < 2000, 1, 0)
item_context = item_context.drop(['title', 'genres', 'year'], axis=1)

# create dataset for user context
user_context_raw = pd.read_csv('data/ratings.csv')
user_context_raw['dt'] = pd.to_datetime(user_context_raw['timestamp'], unit='s')
user_context_raw = user_context_raw.drop(['timestamp'], axis=1)
user_context_raw['weekend'] = np.where(user_context_raw['dt'].dt.day.isin([5, 6]), 1, 0)
user_context_raw['workingday_evening'] = np.where((user_context_raw['dt'].dt.day.isin([5, 6]) == False) & (user_context_raw['dt'].dt.hour.isin([0, 1, 18, 19, 20, 21, 22, 23])), 1, 0)

user_context = user_context_raw[['userId']].drop_duplicates().reset_index().drop(['index'], axis=1)

buf = user_context_raw.groupby(['userId'])[['rating']].mean().rename(columns={'rating':'avg_rating'}).reset_index()
user_context = user_context.merge(buf, how='left', on=['userId'])

buf = user_context_raw.groupby(['userId'])[['weekend']].sum().reset_index()
user_context = user_context.merge(buf, how='left', on=['userId'])
buf = user_context_raw.groupby(['userId'])[['workingday_evening']].sum().reset_index()
user_context = user_context.merge(buf, how='left', on=['userId'])
buf = user_context_raw.groupby(['userId'])[['weekend']].count().rename(columns={'weekend':'total'}).reset_index()
user_context = user_context.merge(buf, how='left', on=['userId'])
user_context['weekend_share'] = user_context['weekend'] / user_context['total']
user_context['workingday_evening_share'] = user_context['workingday_evening'] / user_context['total']
user_context = user_context.drop(['weekend', 'workingday_evening', 'total'], axis=1)

buf = user_context_raw.merge(item_context, how='left', on=['movieId'])
buf = buf.groupby(['userId'])[genres_list].max().reset_index()
user_context = user_context.merge(buf, how='left', on=['userId'])

buf = user_context_raw.merge(item_context, how='left', on=['movieId'])
buf = buf.groupby(['userId'])[genres_list].sum().reset_index()
for col in genres_list:
    buf = buf.rename(columns={col: col + '_share'})
user_context = user_context.merge(buf, how='left', on=['userId'])

buf = user_context_raw.groupby(['userId'])[['weekend']].count().rename(columns={'weekend':'total'}).reset_index()
user_context = user_context.merge(buf, how='left', on=['userId'])
for col in genres_list:
    user_context[col + '_share'] = user_context[col + '_share'] / user_context['total']
user_context = user_context.drop(['total'], axis=1)


rec = rec_utils.SimpleNNRec(
    user_field='userId',
    item_field='movieId',
    rating_field='rating',
)

rec.fit(
    ui_data=df_ratings,
    item_context=item_context,
    user_context=user_context,
    epochs=30,
    cf_dim=10,
    nn_dim_nn=10,
    nn_dim_user=40,
    nn_dim_item=10,
)

rec.save('first')

rec.load('first')

print(rec.recommend(1, 5, item_mapping))