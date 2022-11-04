import os

import numpy as np
import pandas as pd
from collections import defaultdict

import surprise
from surprise import accuracy, SVD, SVDpp
from surprise.dataset import Reader, Dataset
from surprise import KNNBasic
import multiprocessing
import world
import utils
import torch
from sklearn.metrics import roc_auc_score
import register
from register import dataset


CORES = multiprocessing.cpu_count() // 2

#static method to generate df_train and df_test
def train_test_generate():
    if os.path.exists("../data/ml-latest-small/df_train.csv") and os.path.exists("../data/ml-latest-small/df_test.csv"):
        df_train = pd.read_csv("../data/ml-latest-small/df_train.csv")
        df_test = pd.read_csv("../data/ml-latest-small/df_test.csv")
        return df_train,df_test
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

    return df_train,df_test

class baseLineModel():

    def __init__(self,model):
        self.model=model;
        self.df_train,self.df_test=train_test_generate()
        #print(self.cls)

    def test_one_batch(self,X):
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

    def Test(self,dataset, model, multicore=0):
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
            # auc_record = []
            # ratings = []
            total_batch = int(np.ceil(len(users) / u_batch_size))
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                # rating = model.getUsersRating(batch_users_gpu)
                rating = torch.zeros((len(batch_users_gpu), dataset.UserItemNet.shape[1]),dtype=torch.float)
                for i in range(len(batch_users)):
                    for j in range(dataset.UserItemNet.shape[1]):
                        rating[i, j] = model.predict(uid=i, iid=j).est
                # rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                for i in exclude_index:
                    for j in exclude_items:
                        rating[i, j]= -(1 << 10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [
                #         utils.AUC(rating[i],
                #                   dataset,
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
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
            scale = float(u_batch_size / len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if multicore == 1:
                pool.close()
            print(results)
            return results

    def precision_recall_at_k(self,predictions, k=20, threshold=3.5):
        """Return precision and recall at k metrics for each user"""
        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0 Precision is undefined. We here set it to 0.

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0 Recall is undefined. We here set it to 0.

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls

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
        predictions = self.model.test(data_test_surprise)
        precisions, recalls = self.precision_recall_at_k(predictions, k=20, threshold=3.5)
        # Precision and recall can then be averaged over all users
        print("Precision: "+str(sum(prec for prec in precisions.values()) / len(precisions)))
        print("Recall: "+str(sum(rec for rec in recalls.values()) / len(recalls)))
        self.Test(dataset, self.model, world.config['multicore'])


'''
similarity_measures=["cosine","msd","pearson","pearson_baseline"]

'''
'''
similarity_measures=["cosine","msd","pearson","pearson_baseline"]
KNNmodels=[]
for name in similarity_measures:
    sim_options = {
        "name": name,
        "user_based": True
    }
    KNNmodels.append(KNNBasic(sim_options=sim_options))

models={"SVD":[SVD()],
        "KNN":KNNmodels}

for model_type in models.keys():
    for model in models[model_type]:
        baseLineModel(model).run()
'''

sim_options = {
            "name": "cosine",
            "user_based": True
        }
model = KNNBasic(sim_options=sim_options)
baseLineModel(model).run()






