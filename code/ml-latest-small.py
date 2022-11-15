import pandas as pd
import numpy as np
import os
import math
import torch
import pickle

categories = {"Adventure": 0, "Horror":1, "Animation":2, "Children":3, "Comedy":4, "Fantasy":5, "Drama":6, "Romance":7,
              "Thriller":8, "Sci-Fi":9, "Mystery":10, "War":11, "Musical":12, "Crime":13, "Action":14, "IMAX":15,
              "Western":16, "Film-Noir":17, "Documentary":18, "(no genres listed)":19}

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml-latest-small')
movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
df = pd.read_csv(os.path.join(data_path, "ratings.csv"))
movies["movieId"] = movies["movieId"] - 1
df["userId"] = df["userId"] - 1
df["movieId"] = df["movieId"] - 1
movies["genres"] = movies["genres"].apply(lambda x: [categories[g] for g in x.split("|")])
movies_genre = movies["genres"]
movies_genre.to_csv(os.path.join(data_path, "movie_embed.csv"))

genres_hot = torch.zeros((len(movies_genre), 20))
for i in range(len(movies)):
    for k in movies_genre.iloc[i]:
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

with open(os.path.join(data_path, "all_data_scrap.txt"), 'w') as fp:
    for i, item in enumerate(new.values):
        fp.write("%s" % str(i))
        for j, r in zip(item, ratings.values[i]):
            #for k in range(math.floor(r)):
            fp.write(" %s" % str(int(j)))
        fp.write("\n")

with open(os.path.join(data_path, "train_multiplied.txt"), 'w') as fp:
    for i, item in enumerate(new.values):
        fp.write("%s" % str(i))
        split = int(np.ceil(0.8*len(item)))
        for j, r in zip(range(0, split), ratings.values[i]):
            for k in range(round(r + 0.0001)):
                fp.write(" %s" % str(int(item[j])))
        fp.write("\n")


with open(os.path.join(data_path, "test_sorted.txt"), 'w') as fp:
    for i, item in enumerate(new.values):
        fp.write("%s" % str(i))
        split = int(np.ceil(0.8*len(item)))
        test_sort = [x for _, x in sorted(zip(ratings.values[i][split:], item[split:]), key=lambda pair: pair[0], reverse=True)]
        item[split:] = test_sort
        for j in range(split, len(item)):
            #for k in range(math.floor(r)):
            fp.write(" %s" % str(int(item[j])))
        fp.write("\n")

for i in range(len(ratings)):
    split = int(np.ceil(0.8 * len(ratings[i])))
    ratings[i] = sorted(ratings[i][split:], reverse=True)
with open('ratings_sorted.pickle', 'wb') as handle:
    pickle.dump(ratings, handle, protocol=pickle.HIGHEST_PROTOCOL)

