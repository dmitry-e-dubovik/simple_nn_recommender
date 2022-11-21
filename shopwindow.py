import pandas as pd


df_ratings = pd.read_csv('data/ratings.csv')
df_ratings['dt'] = pd.to_datetime(df_ratings['timestamp'], unit='s')
df_ratings = df_ratings.drop(['timestamp'], axis=1)

item_mapping = pd.read_csv('data/movies.csv')
item_mapping = item_mapping[['movieId', 'title']]
item_mapping = dict(item_mapping.values)