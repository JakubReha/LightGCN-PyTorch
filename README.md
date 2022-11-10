## COMP 4222 Group 4
The project explores using the [LightGCN](https://arxiv.org/abs/2002.02126) model for movie recommendation. The repository was forked from [here](https://github.com/gusye1234/LightGCN-PyTorch).

### Dataset
Specifically, we are using the Movie Lens Small Dataset. This dataset is retrived from <https://grouplens.org/datasets/movielens/> and is preprocessed with the file `ml-latest-small.py` and `ml-latest-small-new.py`.

### Experiments
We modified the original LightGCN to incorporate the movies' rating by the users, as well as the movies' genre. We call this new model "Modified LightGCN". 
We also implemented a few baselines (SVD, SVDpp and KNN) taking into account the movie ratings. The baselines selected for experiment include KNN-based models with different similarity measures (Cosine Similarity, Pearson Similarity, Pearson-Baseline, and Mean Squared Difference), SVD and SVDpp.
We perform experiments comparing the original LightGCN, Modified LightGCN, filtered LightGCN (where we only consider giving movie recommendations to active users) and baselines on the prediction task of Movie Lens Dataset. 

### To Run the Codes
#### LightGCN
For original LightGCN, use:
`python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="ml-latest-small" --topks="[20]" --recdim=64 --testbatch 61 --epochs 500`

For Modified LightGCN training with ratings, add:
`--multiplied`

For Modified LightGCN training with movie categories, add:
`--genre`

If you want to train on a reduced dataset (users, who watched less than 100 movies. around 50% of all users) for some reason, use:
`--filtered`

Need to delete the `s_pre_sdj_mat.npz` after running to reinitialize the adjacency matrix.

#### Baselines
The process for executing the experiments have been encapsulated in `baselines.py` and `baseline_experiments.py`. To run the experiments, simply run `baseline_experiments.py` with the following configuration: `--dataset="ml-latest-small" --topks="[20]" --recdim=64 --testbatch 12`

The baseline experiment results will be written into the directory `data/baseline_results_ML`. 

For details on the baseline experiments (e.g. trainable parameters), please refer to the comments in `baselines.py` and `baseline_experiments.py`.
