from baselines import baseLineModel
from surprise import KNNBasic, SVD, SVDpp

"""
The experimenting process for our baseline models.
To run the experiments, simply run this file.

The included baseline models include:
    KNN_based models with different similarity measures: "cosine", "msd", "pearson", "pearson_baseline"
    SVD and SVDpp

"""


def SVD_train(pp=True):
    """
    Experiment for SVD or SVDpp model.

    Adjustable parameters:
        epochs:[5,30]
    """

    epochs = [i for i in range(5, 31)]
    results = {
        "precision": [],
        "recall": [],
        "ndcg": []
    }
    for ep in epochs:
        if pp:
            model = SVDpp(n_epochs=ep)
        else:
            model = SVD(n_epochs=ep, biased=True)
        res = baseLineModel(model).run()
        results["precision"].append(res["precision"][0])
        results["recall"].append(res["recall"][0])
        results["ndcg"].append(res["ndcg"][0])

    # record result into txt file.
    if pp:
        suffix = "pp"
    else:
        suffix = ""

    with open('../data/baseline_results_ML/SVD' + suffix + '.txt', 'w') as data:
        data.write(str(results))
    return results


def KNN_without_Baseline():
    """
    Experiment for KNN model with different underlying similarity measures.
    Adjustable parameters:
        sim_options:
            "name": ["cosine", "msd", "pearson"]
    """

    sim_measure = ["cosine", "msd", "pearson", "pearson_baseline"]
    results = {"precision": {},
               "recall": {},
               "ndcg": {}
               }

    for sim in sim_measure:
        sim_options = {
            "name": sim,
            "user_based": True
        }
        model = KNNBasic(sim_options=sim_options)
        res = baseLineModel(model).run()
        results["precision"][sim] = res["precision"][0]
        results["recall"][sim] = res["recall"][0]
        results["ndcg"][sim] = res["ndcg"][0]

    with open('../data/baseline_results_ML/KNN.txt', 'w') as data:
        data.write(str(results))
    return results


def KNN_with_Baseline(sgd=True):
    """
    Experiment for KNN model with Pearson-baseline as underlying similarity measure.
    Adjustable parameters:
        epoch: [5, 30]
    """

    epoch = [i for i in range(5, 31)]
    results = {
        "precision": [],
        "recall": [],
        "ndcg": []
    }
    for ep in epoch:
        if sgd:
            bsl_options = {"method": "sgd",
                           "n_epochs": ep,
                           "learning_rate": 0.00005
                           }

        else:
            bsl_options = {"method": "als",
                           "n_epochs": ep
                           }

        sim_options = {
            "name": "pearson_baseline",
            "user_based": True
        }
        model = KNNBasic(bsl_options=bsl_options, sim_options=sim_options)
        res = baseLineModel(model).run()
        results["precision"].append(res["precision"][0])
        results["recall"].append(res["recall"][0])
        results["ndcg"].append(res["ndcg"][0])

    if sgd:
        suffix = "SGD"
    else:
        suffix = "ALS"
    with open('../data/baseline_results_ML/KNN_Pearson_Baseline_' + suffix + '.txt', 'w') as data:
        data.write(str(results))
    return results


'''
Experiments.
If you don't want to run specific models' experiment, simply comment out the corresponding lines.
'''
# SVD and SVDpp
SVD_train(pp=True)
SVD_train(pp=False)

# KNN models
KNN_without_Baseline()
KNN_with_Baseline(sgd=True)
KNN_with_Baseline(sgd=False)

