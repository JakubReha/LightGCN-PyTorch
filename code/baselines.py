import os

import numpy as np
import pandas as pd
from surprise.dataset import Reader, Dataset
import multiprocessing
import world
import utils
import torch
from register import dataset

CORES = multiprocessing.cpu_count() // 2

"""
Static method to generate df_train and df_test.
Will be called upon when related files are not detected in directory.
"""
def train_test_generate():
    if os.path.exists("../data/ml-latest-small/df_train.csv") and os.path.exists("../data/ml-latest-small/df_test.csv"):
        df_train = pd.read_csv("../data/ml-latest-small/df_train.csv")
        df_test = pd.read_csv("../data/ml-latest-small/df_test.csv")
        return df_train, df_test
    movies = pd.read_csv("../data/ml-latest-small/movies.csv")
    df = pd.read_csv("../data/ml-latest-small/ratings.csv")
    movies["movieId"] = movies["movieId"] - 1
    df["userId"] = df["userId"] - 1
    df["movieId"] = df["movieId"] - 1
    df["movieId"] = df["movieId"].apply(lambda x: movies[movies["movieId"] == x].index[0])
    df = df.reset_index()
    new = df.sort_values("timestamp", ascending=True).groupby("userId")["index"].apply(list)
    train_index = sum([i[:int(np.ceil(0.8 * len(i)))] for i in new], [])
    test_index = sum([i[int(np.ceil(0.8 * len(i))):] for i in new], [])
    df_train = df.iloc[train_index, :][["userId", "movieId", "rating"]]
    df_test = df.iloc[test_index, :][["userId", "movieId", "rating"]]

    return df_train, df_test

"""
The tailored engine for all baseline models that we use. 
Only supports models imported from scikit-surprise package.
To run exxperiments with baseline models, run baseline_experiments.py
"""
class baseLineModel():

    def __init__(self, model):
        self.model = model;
        self.df_train, self.df_test = train_test_generate()

    def test_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = utils.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in world.topks:
            ret = utils.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        return {'recall': np.array(recall),
                'precision': np.array(pre),
                'ndcg': np.array(ndcg)}

    def Test(self, dataset, model, multicore=0):
        u_batch_size = world.config['test_u_batch_size']
        dataset: utils.BasicDataset
        testDict: dict = dataset.testDict
        # eval mode with no dropout
        max_K = max(world.topks)
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.topks)),
                   'recall': np.zeros(len(world.topks)),
                   'ndcg': np.zeros(len(world.topks))}
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = int(np.ceil(len(users) / u_batch_size))
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = torch.zeros((len(batch_users_gpu), dataset.UserItemNet.shape[1]), dtype=torch.float)
                for i in range(len(batch_users)):
                    for j in range(dataset.UserItemNet.shape[1]):
                        rating[i, j] = model.predict(uid=i, iid=j).est
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1 << 10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.test_one_batch(x))
            float(u_batch_size / len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            if multicore == 1:
                pool.close()
            print(results)
            return results

    def run(self):
        reader = Reader(rating_scale=(0.5, 5.0))

        ratings_df_train = pd.DataFrame(
            {'userID': self.df_train['userId'].astype(int), 'movieID': self.df_train['movieId'].astype(int),
             'rating': self.df_train['rating']})
        ratings_df_train = ratings_df_train[['userID', 'movieID', 'rating']]  # correct order
        data_train_surprise = Dataset.load_from_df(ratings_df_train, reader)
        data_train_surprise = data_train_surprise.build_full_trainset()

        ratings_df_test = pd.DataFrame(
            {'userID': self.df_test['userId'].astype(int), 'movieID': self.df_test['movieId'].astype(int),
             'rating': self.df_test['rating']})
        ratings_df_test = ratings_df_test[['userID', 'movieID', 'rating']]  # correct order
        data_test_surprise = Dataset.load_from_df(ratings_df_test, reader)
        data_test_surprise = data_test_surprise.build_full_trainset().build_testset()
        self.model.fit(data_train_surprise)
        return self.Test(dataset, self.model, world.config['multicore'])
