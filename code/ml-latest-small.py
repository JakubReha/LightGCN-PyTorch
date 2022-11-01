import pandas as pd
import numpy as np
movies = pd.read_csv("/Users/kuba/PycharmProjects/COMP4222/LightGCN-PyTorch/data/ml-latest-small/movies.csv")
df = pd.read_csv("/Users/kuba/PycharmProjects/COMP4222/LightGCN-PyTorch/data/ml-latest-small/ratings.csv")
movies["movieId"] = movies["movieId"] - 1
df["userId"] = df["userId"] - 1
df["movieId"] = df["movieId"] - 1
df["movieId"] = df["movieId"].apply(lambda x: movies[movies["movieId"] == x].index[0])
df = df.reset_index()
new = df.sort_values("timestamp", ascending=True).groupby("userId")["index"].apply(list)
train_index = sum([i[:int(np.ceil(0.8*len(i)))] for i in new], [])
test_index = sum([i[int(np.ceil(0.8*len(i))):] for i in new], [])
df_train = df.iloc[train_index, :][["userId", "movieId", "rating"]]
df_test = df.iloc[test_index, :][["userId", "movieId", "rating"]]
df_train.to_csv("/Users/kuba/PycharmProjects/COMP4222/LightGCN-PyTorch/data/ml-latest-small/df_train.csv")
df_test.to_csv("/Users/kuba/PycharmProjects/COMP4222/LightGCN-PyTorch/data/ml-latest-small/df_test.csv")

print(df_train.head())
print(df_test.head())
new = df.sort_values("timestamp", ascending=True).groupby("userId")["movieId"].apply(list)
with open("/Users/kuba/PycharmProjects/COMP4222/LightGCN-PyTorch/data/ml-latest-small/all_data.txt", 'w') as fp:
    for i, item in enumerate(new.values):
        fp.write("%s" % str(i))
        for j in item:
            fp.write(" %s" % str(int(j)))
        fp.write("\n")

with open("/Users/kuba/PycharmProjects/COMP4222/LightGCN-PyTorch/data/ml-latest-small/train.txt", 'w') as fp:
    for i, item in enumerate(new.values):
        fp.write("%s" % str(i))
        split = int(np.ceil(0.8*len(item)))
        for j in range(0, split):
            fp.write(" %s" % str(int(item[j])))
        fp.write("\n")

with open("/Users/kuba/PycharmProjects/COMP4222/LightGCN-PyTorch/data/ml-latest-small/test.txt", 'w') as fp:
    for i, item in enumerate(new.values):
        fp.write("%s" % str(i))
        split = int(np.ceil(0.8*len(item)))
        for j in range(split, len(item)):
            fp.write(" %s" % str(int(item[j])))
        fp.write("\n")
