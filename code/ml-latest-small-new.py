import pandas as pd
import numpy as np
import os
import math
data_path = "./data/ml-latest-small/"
movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
df = pd.read_csv(os.path.join(data_path, "ratings.csv"))
movies["movieId"] = movies["movieId"] - 1
df["userId"] = df["userId"] - 1
df["movieId"] = df["movieId"] - 1
df["movieId"] = df["movieId"].apply(lambda x: movies[movies["movieId"] == x].index[0])

df = df.reset_index()

grouped = df.sort_values("timestamp", ascending=True).groupby("userId")
new = grouped["movieId"].apply(list)
ratings = grouped["rating"].apply(list)


            

with open(os.path.join(data_path, "train_filtered.txt"), 'w') as fp:
    t = -1
    for i, item in enumerate(new.values):
        if len(item) > 100:
            t += 1
            fp.write("%s" % str(t))
            split = int(np.ceil(0.8*len(item)))
            for j, r in zip(range(0, split), ratings.values[i]):
                for k in range(math.floor(r)):
                    fp.write(" %s" % str(int(item[j])))
            fp.write("\n")


with open(os.path.join(data_path, "test_filtered.txt"), 'w') as fp:
    t = -1
    for i, item in enumerate(new.values):
        if len(item) > 100:
            t += 1
            fp.write("%s" % str(t))
            split = int(np.ceil(0.8*len(item)))
            for j, r in zip(range(split, len(item)), ratings.values[i]):
                for k in range(math.floor(r)):
                    fp.write(" %s" % str(int(item[j])))
            fp.write("\n")



with open(os.path.join(data_path, "train_multiplied_new.txt"), 'w') as fp:
    t = -1
    for i, item in enumerate(new.values):
        t += 1
        fp.write("%s" % str(t))
        split = int(np.ceil(0.8*len(item)))
        for j, r in zip(range(0, split), ratings.values[i]):
            for k in range(math.floor(r)):
                fp.write(" %s" % str(int(item[j])))
        fp.write("\n")


with open(os.path.join(data_path, "test_multiplied_new.txt"), 'w') as fp:
    t = -1
    for i, item in enumerate(new.values):
        t += 1
        fp.write("%s" % str(t))
        split = int(np.ceil(0.8*len(item)))
        for j, r in zip(range(split, len(item)), ratings.values[i]):
            for k in range(math.floor(r)):
                fp.write(" %s" % str(int(item[j])))
        fp.write("\n")
