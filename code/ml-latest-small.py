import pandas as pd
import numpy as np
import os
import math
import torch
categories = {"Adventure": 0, "Horror":1, "Animation":2, "Children":3, "Comedy":4, "Fantasy":5, "Drama":6, "Romance":7,
              "Thriller":8, "Sci-Fi":9, "Mystery":10, "War":11, "Musical":12, "Crime":13, "Action":14, "IMAX":15,
              "Western":16, "Film-Noir":17, "Documentary":18, "(no genres listed)":19}
data_path = "data/ml-latest-small/"
movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
df = pd.read_csv(os.path.join(data_path, "ratings.csv"))
movies["movieId"] = movies["movieId"] - 1
df["userId"] = df["userId"] - 1
df["movieId"] = df["movieId"] - 1
movies["genres"] = movies["genres"].apply(lambda x: [categories[g] for g in x.split("|")])
movies = movies["genres"]
movies.to_csv(os.path.join(data_path, "movie_embed.csv"))

genres_hot = torch.zeros((len(movies), 20))
for i in range(len(movies)):
    for k in movies.iloc[i]:
        genres_hot[i, k] = 1
genres_hot = genres_hot/genres_hot.sum(axis=1).unsqueeze(1)
torch.save(genres_hot, os.path.join(data_path, 'genres_hot.pt'))


df["movieId"] = df["movieId"].apply(lambda x: movies[movies["movieId"] == x].index[0])
df = df.reset_index()
new = df.sort_values("timestamp", ascending=True).groupby("userId")["index"].apply(list)
train_index = sum([i[:int(np.ceil(0.8*len(i)))] for i in new], [])
test_index = sum([i[int(np.ceil(0.8*len(i))):] for i in new], [])
df_train = df.iloc[train_index, :][["userId", "movieId", "rating"]]
df_test = df.iloc[test_index, :][["userId", "movieId", "rating"]]
df_train.to_csv(os.path.join(data_path, "df_train.csv"))
df_test.to_csv(os.path.join(data_path, "df_test.csv"))

grouped = df.sort_values("timestamp", ascending=True).groupby("userId")
new = grouped["movieId"].apply(list)
ratings = grouped["rating"].apply(list)

with open(os.path.join(data_path, "all_data_cat.txt"), 'w') as fp:
    for i, item in enumerate(new.values):
        fp.write("%s" % str(i))
        for j, r in zip(item, ratings.values[i]):
            #for k in range(math.floor(r)):
            fp.write(" %s" % str(int(j)))
        fp.write("\n")

with open(os.path.join(data_path, "train_cat.txt"), 'w') as fp:
    for i, item in enumerate(new.values):
        fp.write("%s" % str(i))
        split = int(np.ceil(0.8*len(item)))
        for j, r in zip(range(0, split), ratings.values[i]):
            #for k in range(math.floor(r)):
            fp.write(" %s" % str(int(item[j])))
        fp.write("\n")

with open(os.path.join(data_path, "test_cat.txt"), 'w') as fp:
    for i, item in enumerate(new.values):
        fp.write("%s" % str(i))
        split = int(np.ceil(0.8*len(item)))
        for j, r in zip(range(split, len(item)), ratings.values[i]):
            #for k in range(math.floor(r)):
            fp.write(" %s" % str(int(item[j])))
        fp.write("\n")
