import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import accuracy, SVD, SVDpp
from surprise.dataset import Reader, Dataset
import multiprocessing
import world
import utils
import torch
from sklearn.metrics import roc_auc_score
import register
from register import dataset


CORES = multiprocessing.cpu_count() // 2


def test_one_batch(X):
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


def Test(dataset, model, multicore=0):
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

            #rating = model.getUsersRating(batch_users_gpu)
            rating = torch.zeros((len(batch_users_gpu), dataset.UserItemNet.shape[1]))
            for i in range(len(batch_users)):
                for j in range(dataset.UserItemNet.shape[1]):
                    rating[i, j] = model.predict(uid=i, iid=j).est
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
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
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
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


reader = Reader(rating_scale=(0.5, 5.0))
df_train = pd.read_csv("/Users/kuba/PycharmProjects/COMP4222/LightGCN-PyTorch/data/ml-latest-small/df_train.csv")
df_test = pd.read_csv("/Users/kuba/PycharmProjects/COMP4222/LightGCN-PyTorch/data/ml-latest-small/df_test.csv")

ratings_df_train = pd.DataFrame({'userID': df_train['userId'].astype(int), 'movieID': df_train['movieId'].astype(int), 'rating': df_train['rating']})
ratings_df_train = ratings_df_train[['userID', 'movieID', 'rating']] # correct order
data_train_surprise = Dataset.load_from_df(ratings_df_train, reader)
data_train_surprise = data_train_surprise.build_full_trainset()

ratings_df_test = pd.DataFrame({'userID': df_test['userId'].astype(int), 'movieID': df_test['movieId'].astype(int), 'rating':df_test['rating']})
ratings_df_test = ratings_df_test[['userID', 'movieID', 'rating']] #correct order
data_test_surprise = Dataset.load_from_df(ratings_df_test, reader)
data_test_surprise = data_test_surprise.build_full_trainset().build_testset()
model = SVD()
model.fit(data_train_surprise)
#predictions = model.test(data_test_surprise)
#precisions, recalls = precision_recall_at_k(predictions, k=20, threshold=3.5)
# Precision and recall can then be averaged over all users
#print(sum(prec for prec in precisions.values()) / len(precisions))
#print(sum(rec for rec in recalls.values()) / len(recalls))
Test(dataset, model, world.config['multicore'])

#https://surprise.readthedocs.io/en/stable/FAQ.html?highlight=recall#how-to-compute-precision-k-and-recall-k
